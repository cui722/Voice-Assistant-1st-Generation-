import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from core import BASE, WORLD, TOOLS, observation, apply_plan, matches, load_cases


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Engine:
    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()
        self.config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
        self.system = (BASE / "prompts/system.txt").read_text(encoding="utf-8")
        self.system += "\nTool argument domains:\n" + dump(TOOLS)
        self.commit = None

        if backend == "mock":
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Run: python main.py doctor")

        torch.manual_seed(42)
        self.torch = torch
        model_id = self.config["model_id"]
        revision = self.config["revision"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.commit = getattr(self.model.config, "_commit_hash", None)

    def generate(self, obs):
        with self.lock:
            if self.backend == "mock":
                raw = json.dumps({
                    "decision": "clarify",
                    "tool": None,
                    "arguments": {},
                    "slot": "operation",
                    "question": "This is a pipeline check. Which operation would you like to perform?",
                }, ensure_ascii=False)
                return raw, {
                    "latency_s": 0.0,
                    "peak_cuda_allocated_gb": None,
                    "input_tokens": None,
                    "output_tokens": None,
                }

            torch = self.torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()

            messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": dump(obs)},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            ).to("cuda:0")
            input_tokens = inputs["input_ids"].shape[1]
            if input_tokens > self.config["max_input_tokens"]:
                raise ValueError("Input too long. Shorten the context or reset the conversation.")

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = output[0, input_tokens:]
            raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            torch.cuda.synchronize()

            return raw, {
                "latency_s": time.perf_counter() - started,
                "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "input_tokens": input_tokens,
                "output_tokens": int(generated.numel()),
            }


def run_request(engine, item):
    obs = observation(item)
    raw, timing = engine.generate(obs)
    prediction = None
    error = None
    after = copy.deepcopy(obs["context"])
    applied = False

    try:
        prediction = json.loads(raw)
        after, applied = apply_plan(obs["context"], prediction)
    except (ValueError, TypeError) as exc:
        error = str(exc)

    return {
        "input": obs,
        "raw": raw,
        "prediction": prediction,
        "valid_plan": error is None,
        "applied": applied,
        "error": error,
        "after": after,
        **timing,
    }


def doctor():
    import torch

    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        x = torch.ones((32, 32), device="cuda")
        result["cuda_matmul_value"] = float((x @ x)[0, 0].item())
        torch.cuda.synchronize()

    print(dump(result))
    if not result["cuda_available"]:
        raise SystemExit(1)


def evaluate(args):
    case_path = Path(args.cases)
    cases = load_cases(case_path)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[:args.limit]

    engine = Engine(args.backend)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = BASE / "runs" / f"{stamp}_{args.backend}"
    output_dir.mkdir(parents=True, exist_ok=False)

    versions = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "fastapi", "uvicorn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    manifest = {
        "purpose": "DEVELOPMENT_ONLY" if args.backend == "qwen" else "PIPELINE_SMOKE_ONLY",
        "backend": args.backend,
        "config": engine.config,
        "resolved_model_commit": engine.commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "case_count": len(cases),
        "cases_sha256": sha(case_path),
        "world_sha256": sha(BASE / "data/world.json"),
        "prompt_sha256": sha(BASE / "prompts/system.txt"),
        "core_sha256": sha(BASE / "core.py"),
        "main_sha256": sha(BASE / "main.py"),
    }
    if args.backend == "qwen":
        manifest["gpu"] = engine.torch.cuda.get_device_name(0)
    (output_dir / "manifest.json").write_text(dump(manifest), encoding="utf-8")

    # Warm-up only; no weight updates and no gold labels passed to the model.
    engine.generate(observation(cases[0]))

    rows = []
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            result = run_request(engine, case)
            correct = result["valid_plan"] and matches(result["prediction"], case["expected"])
            row = {
                "id": case["id"],
                "category": case["category"],
                "expected": case["expected"],
                "plan_match": correct,
                **result,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(case["id"], "MATCH" if correct else "CHECK", flush=True)

    n = len(rows)
    attempted = sum(
        isinstance(row["prediction"], dict)
        and row["prediction"].get("decision") in ("execute", "cancel")
        for row in rows
    )
    applied = sum(row["applied"] for row in rows)
    wrong = sum(row["applied"] and not row["plan_match"] for row in rows)
    actionable = [
        row for row in rows
        if row["expected"]["decision"] in ("execute", "cancel")
    ]
    unnecessary = sum(
        row["valid_plan"] and row["prediction"]["decision"] == "clarify"
        for row in actionable
    )
    times = sorted(row["latency_s"] for row in rows)
    peaks = [
        row["peak_cuda_allocated_gb"]
        for row in rows
        if row["peak_cuda_allocated_gb"] is not None
    ]
    categories = {}
    for row in rows:
        group = categories.setdefault(row["category"], {"n": 0, "matched": 0})
        group["n"] += 1
        group["matched"] += int(row["plan_match"])

    summary = {
        "purpose": manifest["purpose"],
        "n": n,
        "plan_match_rate": sum(row["plan_match"] for row in rows) / n,
        "valid_plan_rate": sum(row["valid_plan"] for row in rows) / n,
        "attempted_actions": attempted,
        "applied_actions": applied,
        "blocked_actions": attempted - applied,
        "wrong_applied_actions": wrong,
        "wrong_execution_rate": wrong / applied if applied else None,
        "execution_coverage": applied / n,
        "unnecessary_clarification_rate": unnecessary / len(actionable) if actionable else None,
        "latency_mean_s": statistics.mean(times),
        "latency_p50_s": statistics.median(times),
        "latency_p95_s": times[math.ceil(0.95 * n) - 1],
        "peak_cuda_allocated_gb": max(peaks) if peaks else None,
        "by_category": categories,
    }
    (output_dir / "summary.json").write_text(dump(summary), encoding="utf-8")
    print(dump(summary))
    print(f"Results: {output_dir}")


def serve(args):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    import uvicorn

    engine = Engine(args.backend)
    app = FastAPI(title="ReCheck Voice — Week 1")

    @app.get("/", response_class=HTMLResponse)
    def homepage():
        return (BASE / "web.html").read_text(encoding="utf-8")

    @app.get("/context")
    def initial_context():
        return {"backend": args.backend, "context": WORLD}

    @app.post("/infer")
    def infer(body: dict):
        try:
            return run_request(engine, body)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    uvicorn.run(app, host="127.0.0.1", port=args.port)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")

    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--backend", choices=["mock", "qwen"], default="qwen")
    evaluation.add_argument("--cases", default=str(BASE / "data/dev_cases.jsonl"))
    evaluation.add_argument("--limit", type=int)

    server = commands.add_parser("serve")
    server.add_argument("--backend", choices=["mock", "qwen"], default="qwen")
    server.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "eval":
        evaluate(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
