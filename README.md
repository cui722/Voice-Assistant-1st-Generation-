# ReCheck Voice — Week 1

Text-input development baseline with local Qwen inference and simulated tools.

Run from this directory:

    python -m unittest discover -s tests -v
    python main.py doctor
    python main.py eval --backend mock --limit 3
    python main.py eval --backend qwen
    python main.py serve --backend qwen

Open http://localhost:8000 after the server starts.

All operations are simulated. No actual calls, navigation or external messages
are performed.

data/dev_cases.jsonl is a DEVELOPMENT set, not a held-out benchmark.
Expected labels are never sent to the model.

mock always asks a generic question. Its metrics have no model-performance meaning.

Plan matching checks decision, tool, arguments and clarification slot.
Question usefulness requires manual review.
Latency includes tokenization/generation, excludes model loading and ASR.
CUDA memory reports PyTorch peak allocated memory, not total system GPU usage.

Every evaluation writes manifest.json, summary.json and predictions.jsonl.
Review raw output and blocked operations as well as accepted actions.

The initializer refuses to overwrite an existing project.
Edit generated files directly after initialization.
