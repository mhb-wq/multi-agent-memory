# DG-Mem

DG-Mem is a dual-graph retrieval system for long group conversations. It combines a Content-Centric Memory Graph with a User-Context Graph. This public package contains the two benchmark entry points used in the final evaluation:

- `run_groupmembench.py` for GroupMemBench;
- `run_locomo.py` for LoCoMo.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Place the benchmark repositories beside this repository:

```text
workspace/
  DG-Mem/
  GroupMemBench/
  locomo/
```

## GroupMemBench

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY"
$env:OPENAI_BASE_URL = "BASE_URL"

python run_groupmembench.py `
  --llm-profile openai `
```

## LoCoMo

```powershell
python run_locomo.py `
  --config configs\locomo.toml `
  --no-evaluate `
  --embedding-device cuda `
  --reranker-device cuda `
  --output-jsonl outputs\locomo\retrieval.jsonl
```

```powershell
$env:SILICONFLOW_API_KEY = "YOUR_KEY"
python run_locomo.py `
  --config configs\locomo.toml `
  --llm-profile siliconflow `
  --evaluate `
  --retrieval-input-jsonl outputs\locomo\retrieval.jsonl `
  --output-jsonl outputs\locomo\qwen.jsonl

$env:OPENAI_API_KEY = "YOUR_KEY"
python run_locomo.py `
  --config configs\locomo.toml `
  --llm-profile openai `
  --evaluate `
  --retrieval-input-jsonl outputs\locomo\retrieval.jsonl `
  --output-jsonl outputs\locomo\gpt-4o-mini.jsonl
```

Each evaluated run produces the question-level JSONL plus `.summary.json` and `.summary.md` files containing SingleHop, MultiHop, Temporal, OpenDomain, and question-weighted Average F1/BLEU-1.
