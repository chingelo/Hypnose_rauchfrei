# Finetune Data

Diese Dateien werden aus dem aktuellen Runtime-Code generiert.

Dateien:
- `routing_gold.jsonl`
- `slot_extraction_gold.jsonl`
- `clarification_gold.jsonl`
- `support_abort_gold.jsonl`
- `routing_train.jsonl`
- `routing_eval.jsonl`
- `slot_extraction_train.jsonl`
- `slot_extraction_eval.jsonl`
- `clarification_train.jsonl`
- `clarification_eval.jsonl`
- `support_abort_train.jsonl`
- `support_abort_eval.jsonl`
- `split_manifest.json`

Schemas:
- `schemas/routing_output.schema.json`
- `schemas/slot_extraction_output.schema.json`
- `schemas/clarification_output.schema.json`
- `schemas/support_abort_output.schema.json`

Generator:
- `backend/build_gold_finetune_data.py`
- `backend/build_finetune_splits.py`
- `backend/evaluate_finetune_candidate.py`

Ziel:
- keine Altlasten aus gemischten historischen JSONL-Dateien als Goldstandard verwenden
- stattdessen direkt aus den aktuellen Knoten, Hilfsfunktionen und Tests ableiten
- daraus reproduzierbare `train/eval`-Splits und feste Zielschemas fuer spaetere Modelltests erzeugen

Evaluation:
- offline zur Harness-Validierung:
  - `python backend/evaluate_finetune_candidate.py --provider reference`
- lokal ueber Ollama:
  - `python backend/evaluate_finetune_candidate.py --provider ollama --model mistral:7b-instruct --datasets routing slot_extraction`
