import copy
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
WORLD = json.loads((BASE / "data/world.json").read_text(encoding="utf-8"))

TOOLS = {
    "call.queue": {"contact_id": "contacts"},
    "call.cancel": {},
    "nav.start": {"place_id": "places"},
    "nav.cancel": {},
    "music.play": {"track_id": "tracks"},
    "music.pause": {},
    "music.resume": {},
    "audio.set_volume": {"level": "integer_0_100"},
}
CANCEL_TOOLS = {"call.cancel", "nav.cancel"}
SLOTS = {"contact_id", "place_id", "track_id", "level", "operation"}
FIELDS = {"decision", "tool", "arguments", "slot", "question"}


def plan(decision, tool=None, arguments=None, slot=None, question=None):
    return {
        "decision": decision,
        "tool": tool,
        "arguments": arguments or {},
        "slot": slot,
        "question": question,
    }


def observation(item):
    utterance = item.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("utterance must be a non-empty string")

    history = item.get("history", [])
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    for message in history:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("invalid history message")

    context = copy.deepcopy(item.get("context", WORLD))
    if not isinstance(context, dict):
        raise ValueError("context must be an object")

    for collection in ("contacts", "places", "tracks"):
        entries = context.get(collection)
        if not isinstance(entries, list):
            raise ValueError(f"{collection} must be a list")
        ids = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise ValueError(f"invalid entity in {collection}")
            ids.append(entry["id"])
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs in {collection}")

    state = context.get("state")
    required = {"pending_call", "destination", "track", "playing", "volume"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("invalid state fields")

    patch = item.get("state_patch", {})
    if not isinstance(patch, dict) or not set(patch).issubset(required):
        raise ValueError("invalid state_patch")
    state.update(patch)

    for field, collection in (
        ("pending_call", "contacts"),
        ("destination", "places"),
        ("track", "tracks"),
    ):
        value = state[field]
        valid_ids = {entry["id"] for entry in context[collection]}
        if value is not None and (
            not isinstance(value, str) or value not in valid_ids
        ):
            raise ValueError(f"invalid state reference: {field}")

    if type(state["playing"]) is not bool:
        raise ValueError("playing must be boolean")
    if state["playing"] and state["track"] is None:
        raise ValueError("playing requires a selected track")
    if type(state["volume"]) is not int or not 0 <= state["volume"] <= 100:
        raise ValueError("invalid state volume")

    # Deliberate whitelist: expected answers and case categories never enter prompts.
    return {
        "utterance": utterance,
        "history": copy.deepcopy(history),
        "context": context,
    }


def apply_plan(context, prediction):
    if not isinstance(prediction, dict) or set(prediction) != FIELDS:
        raise ValueError("prediction must have exactly the five required fields")

    decision = prediction["decision"]
    tool = prediction["tool"]
    arguments = prediction["arguments"]
    slot = prediction["slot"]
    question = prediction["question"]

    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")

    after = copy.deepcopy(context)

    if decision == "clarify":
        if tool is not None or arguments:
            raise ValueError("clarification cannot invoke a tool")
        if not isinstance(slot, str) or slot not in SLOTS:
            raise ValueError("invalid clarification slot")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("clarification requires a question")
        return after, False

    if decision == "unsupported":
        if tool is not None or arguments or slot is not None or question is not None:
            raise ValueError("unsupported output has unexpected fields")
        return after, False

    if decision not in ("execute", "cancel"):
        raise ValueError("invalid decision")
    if not isinstance(tool, str) or tool not in TOOLS:
        raise ValueError("unknown tool")
    if slot is not None or question is not None:
        raise ValueError("tool operations cannot contain clarification fields")
    if (decision == "cancel") != (tool in CANCEL_TOOLS):
        raise ValueError("decision and tool disagree")
    if set(arguments) != set(TOOLS[tool]):
        raise ValueError("wrong tool arguments")

    for name, domain in TOOLS[tool].items():
        value = arguments[name]
        if domain == "integer_0_100":
            if type(value) is not int or not 0 <= value <= 100:
                raise ValueError("volume must be an integer from 0 to 100")
        else:
            valid_ids = {entry["id"] for entry in context[domain]}
            if not isinstance(value, str) or value not in valid_ids:
                raise ValueError(f"unknown entity ID for {name}")

    state = after["state"]

    if tool == "call.queue":
        state["pending_call"] = arguments["contact_id"]
    elif tool == "call.cancel":
        state["pending_call"] = None
    elif tool == "nav.start":
        state["destination"] = arguments["place_id"]
    elif tool == "nav.cancel":
        state["destination"] = None
    elif tool == "music.play":
        state["track"] = arguments["track_id"]
        state["playing"] = True
    elif tool in {"music.pause", "music.resume"}:
        if state["track"] is None:
            raise ValueError("no track selected")
        state["playing"] = tool == "music.resume"
    elif tool == "audio.set_volume":
        state["volume"] = arguments["level"]

    return after, True


def matches(prediction, expected):
    if not isinstance(prediction, dict):
        return False
    # Question wording is intentionally not scored automatically.
    return all(
        prediction.get(key) == expected.get(key)
        for key in ("decision", "tool", "arguments", "slot")
    )


def load_cases(path):
    cases = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("empty case file")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    for case in cases:
        obs = observation(case)
        apply_plan(obs["context"], case["expected"])
    return cases
