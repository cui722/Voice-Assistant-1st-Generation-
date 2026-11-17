#!/usr/bin/env python3
"""Update the original ReCheck Voice Week 1 starter to English.

Run from the project folder:
    python3 update_recheck_english.py

Or specify the project:
    python3 update_recheck_english.py --project ~/projects/recheck_voice

Uses only the Python standard library. Does not install packages, load a model,
or change the virtual environment, evaluation runs, or private data.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TRANSLATIONS = {
    "\u8fd9\u662f\u94fe\u8def\u81ea\u68c0\u3002\u4f60\u5e0c\u671b\u6267\u884c\u54ea\u4e00\u79cd\u64cd\u4f5c\uff1f": "This is a pipeline check. Which operation would you like to perform?",
    "\u7b2c 1 \u5468 \u00b7 \u6587\u672c\u57fa\u7ebf \u00b7 \u5168\u90e8\u64cd\u4f5c\u5728\u6a21\u62df\u5668\u4e2d\u6267\u884c": "Week 1 | Text baseline | All actions run in the simulator",
    "\u4f8b\u5982\uff1aCall Alex": "Example: Call Alex",
    "\uff08\u4ec5\u68c0\u67e5\u94fe\u8def\uff0c\u56fa\u5b9a\u8fd4\u56de\u8ffd\u95ee\uff09": " (pipeline check only; always asks for clarification)",
    "\uff08\u771f\u5b9e\u672c\u5730\u6a21\u578b\uff09": " (local model)",
    "\u4e0a\u4e0b\u6587\u4e0e\u6a21\u62df\u72b6\u6001\uff08\u53ef\u7f16\u8f91\uff09": "Context and simulator state (editable)",
    "\u6700\u8fd1\u4e00\u6b21\u7ed3\u679c": "Latest result",
    "\u8f93\u51fa\u88ab\u62e6\u622a\uff1a": "Output blocked: ",
    "\u5b8c\u6210\uff1b\u51b3\u7b56\u8017\u65f6 ": "Done; decision latency: ",
    "\u5904\u7406\u4e2d\u2026": "Processing...",
    "\u91cd\u7f6e\u4f1a\u8bdd": "Reset conversation",
    "\u5bfc\u51fa\u4f1a\u8bdd": "Export conversation",
    "\u53d1\u9001": "Send",
    "\u4f1a\u8bdd": "Conversation",
    "\u6a21\u578b\uff1a": "Model: ",
    "\u4f60\uff1a": "You: ",
    " \u79d2": " seconds",
    "Call\u5b66\u6821\u9879\u76ee\u7ec4\u90a3\u4e2aAlex\u3002": "Call Alex from the USC project group.",
    "\u7ed9\u5065\u8eab\u623fAlex\u6253\u7535\u8bdd\uff0c\u4e0d\uff0c\u7ed9\u5b66\u6821\u9879\u76ee\u7ec4\u90a3\u4e2a\u3002": "Call Alex from the gym. No, call Alex from the USC project group.",
    "\u5148\u522b\u62e8\u4e86\uff0c\u53d6\u6d88\u521a\u624d\u7684\u7535\u8bdd\u3002": "Wait, cancel the queued call.",
    "\u7ed9Alex\u53d1\u5fae\u4fe1\u6d88\u606f\u3002": "Send Alex a message on WeChat.",
    "\u5e26\u6211\u53bbUSC\u3002": "Take me to USC.",
    "\u5bfc\u822a\u5230Starbucks\u3002": "Navigate to Starbucks.",
    "\u5bfc\u822a\u5230\u5bb6\u9644\u8fd1\u90a3\u5bb6Starbucks\u3002": "Navigate to the Starbucks near my home.",
    "\u53bbLAX\uff0c\u7b97\u4e86\u6539\u6210USC\u3002": "Navigate to LAX. Actually, take me to USC instead.",
    "\u53d6\u6d88\u5f53\u524d\u5bfc\u822a\u3002": "Cancel the current navigation.",
    "\u628a\u7a7a\u8c03\u8c03\u523020\u5ea6\u3002": "Set the air conditioning to 20 degrees.",
    "\u64ad\u653e\u5468\u6770\u4f26\u7684\u6674\u5929\u3002": "Play Sunny Day by Jay Chou.",
    "Play Coldplay\u7684Yellow\u3002": "Play Yellow by Coldplay.",
    "\u64ad\u653eAdele\u5531\u7684Hello\u3002": "Play Hello by Adele.",
    "\u64ad\u653eYellow\uff0c\u4e0d\uff0c\u6539\u6210\u6674\u5929\u3002": "Play Yellow. No, switch to Sunny Day.",
    "\u6682\u505c\u97f3\u4e50\u3002": "Pause the music.",
    "\u7ee7\u7eed\u64ad\u653e\u3002": "Resume playback.",
    "\u97f3\u91cf\u8c03\u523030\u3002": "Set the volume to 30.",
    "\u628a\u97f3\u91cf\u8c03\u5230\u2026\u2026": "Set the volume to...",
    "\u7ed9Maya\u6253\u7535\u8bdd\uff0c\u7b97\u4e86\u522b\u6253\u4e86\u3002": "Call Maya. Actually, cancel the call.",
    "Call \u674e\u660e\u3002": "Call Ming Li.",
    "\u5b66\u6821\u90a3\u4e2a\u3002": "The one from the USC project group.",
    "\u53bb\u53e6\u4e00\u4e2aStarbucks\u3002": "Navigate to the other Starbucks.",
    "\u53d6\u6d88\u90a3\u4e2a\u3002": "Cancel that.",
    "\u5e2e\u6211\u4e70\u4e00\u5f20\u673a\u7968\u3002": "Buy me a plane ticket.",
    "USC\u5b66\u6821\u9879\u76ee\u7ec4": "USC project group",
    "\u5b66\u6821\u9644\u8fd1": "Near campus",
    "\u5bb6\u9644\u8fd1": "Near home",
    "\u5065\u8eab\u623f": "Gym",
    "\u540c\u5b66": "Classmate",
    "\u738b\u7433": "Lin Wang",
    "\u540c\u4e8b": "Colleague",
    "\u5b66\u6821": "Campus",
    "\u673a\u573a": "Airport",
    "\u5bb6": "Home",
    "\u6674\u5929": "Sunny Day",
    "\u5468\u6770\u4f26": "Jay Chou",
    "lang=\"zh-CN\"": "lang=\"en\"",
    "lang=\"zh\"": "lang=\"en\"",
    "14. Language may be Chinese, English, or mixed.": "14. Use English for every user-facing question or response.\n    This development set contains English-only requests.",
    "add(\"mixed\",": "add(\"paraphrase\",",
    "add('mixed',": "add('paraphrase',"
}

PROJECT_FILES = (
    "main.py", "core.py", "web.html", "prompts/system.txt",
    "data/world.json", "data/dev_cases.jsonl", "tests/test_core.py",
    "README.md", "docs/week1_report.md", "docs/interview_guide.md",
    "config.json", "requirements.txt",
)
REQUIRED_FILES = (
    "main.py", "core.py", "web.html", "prompts/system.txt",
    "data/world.json", "data/dev_cases.jsonl",
)
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
REPLACEMENT_PATTERN = re.compile(
    "|".join(re.escape(key) for key in sorted(TRANSLATIONS, key=len, reverse=True))
)


def translate(text):
    # A single pass prevents one replacement from modifying another.
    return REPLACEMENT_PATTERN.sub(lambda match: TRANSLATIONS[match.group()], text)


def translate_json(value):
    if isinstance(value, str):
        return translate(value)
    if isinstance(value, list):
        return [translate_json(item) for item in value]
    if isinstance(value, dict):
        result = {key: translate_json(item) for key, item in value.items()}
        if result.get("category") == "mixed":
            result["category"] = "paraphrase"
        return result
    return value


def convert(path, text):
    if path.suffix == ".json":
        old = json.loads(text)
        new = translate_json(old)
        if new == old:
            return text
        return json.dumps(new, ensure_ascii=False, indent=2) + "\n"
    if path.suffix == ".jsonl":
        old = [json.loads(line) for line in text.splitlines() if line.strip()]
        new = translate_json(old)
        if new == old:
            return text
        return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in new)
    return translate(text)


def validate_text(path, text):
    if path.suffix == ".py":
        ast.parse(text, filename=str(path))
    elif path.suffix == ".json":
        json.loads(text)
    elif path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                json.loads(line)
    match = CJK.search(text)
    if match:
        line = text.count("\n", 0, match.start()) + 1
        raise ValueError(
            f"Unrecognized non-English text in {path.name}, line {line}. "
            "This updater supports the original starter. No project files "
            "have been changed. Share that source file for an exact translation."
        )


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_atomic(path, data, mode):
    descriptor, name = tempfile.mkstemp(prefix=".recheck-update-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument(
        "--bootstrap", type=Path,
        help="Optional initializer path; defaults to the sibling bootstrap_recheck.py if present.",
    )
    parser.add_argument("--check", action="store_true", help="Preview changes without writing files.")
    args = parser.parse_args()
    root = args.project.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Project folder not found: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        parser.error("This is not the original project folder. Missing: " + ", ".join(missing))

    files = [(root / name, Path("project") / name) for name in PROJECT_FILES if (root / name).is_file()]
    bootstrap = args.bootstrap.expanduser().absolute() if args.bootstrap else root.parent / "bootstrap_recheck.py"
    if args.bootstrap and not bootstrap.is_file():
        parser.error(f"Initializer not found: {bootstrap}")
    if bootstrap.is_file():
        files.append((bootstrap, Path("initializer") / bootstrap.name))

    changes = []
    for path, backup_name in files:
        if path.is_symlink():
            raise ValueError(f"Refusing to replace a symbolic link: {path}")
        before = path.read_bytes()
        text = before.decode("utf-8-sig")
        converted = convert(path, text)
        validate_text(path, converted)
        if converted != text:
            newline = "\r\n" if "\r\n" in text else "\n"
            converted = converted.replace("\r\n", "\n").replace("\n", newline)
            after = converted.encode("utf-8")
            changes.append({
                "path": path, "backup_name": backup_name, "before": before,
                "after": after, "mode": stat.S_IMODE(path.stat().st_mode),
            })

    if not changes:
        print("Already up to date. The supported starter files are in English.")
        return 0
    print("Planned English updates:")
    for change in changes:
        print(f"  {change['path']}")
    if args.check:
        print("Preview complete. No files were changed.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = root.parent / f"{root.name}_before_english_{stamp}"
    backup.mkdir(mode=0o700)
    manifest = {
        "project": str(root), "created_utc": stamp,
        "note": "English-only development inputs replace bilingual inputs. "
                "Earlier runs remain unchanged and use a different dataset.",
        "files": [],
    }
    for change in changes:
        saved = backup / change["backup_name"]
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(change["before"])
        saved.chmod(change["mode"])
        manifest["files"].append({
            "original_path": str(change["path"]),
            "backup_path": str(change["backup_name"]),
            "original_sha256": sha256(change["before"]),
            "updated_sha256": sha256(change["after"]),
        })
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Abort if another editor changed any target after the update was prepared.
    for change in changes:
        if change["path"].read_bytes() != change["before"]:
            raise RuntimeError(f"File changed during preparation: {change['path']}. Update cancelled.")
    written = []
    try:
        for change in changes:
            write_atomic(change["path"], change["after"], change["mode"])
            written.append(change)
    except BaseException:
        for change in reversed(written):
            write_atomic(change["path"], change["before"], change["mode"])
        raise

    print(f"\nSUCCESS: Updated {len(changes)} files to English.")
    print(f"Backup: {backup}")
    print("Virtual environments, evaluation runs, and private data were left unchanged.")
    print("The development cases are now English-only; compare new runs on this same dataset.")
    print("Next, run from the project folder:")
    print("  python3 -m unittest discover -s tests -v")
    print("  python main.py serve --backend mock")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SyntaxError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
