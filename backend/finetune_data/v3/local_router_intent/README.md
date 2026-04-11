# Local Router Package

Dieses Paket bereitet ein lokales Fine-Tuning nur fuer den strukturierten Semantic-Teil vor.

Enthalten:
- `router_sft_train.jsonl`
- `router_sft_eval.jsonl`
- `routing_train_only.jsonl`
- `routing_eval_only.jsonl`
- `slot_train_only.jsonl`
- `slot_eval_only.jsonl`
- `router_package_manifest.json`
- `router_qlora_config.json`

Ziel:
- OpenAI-/FT-Abhaengigkeit fuer `routing` und `slot_extraction` spaeter abloesen
- `clarification` und `support_abort` bewusst weiter deterministisch im Code halten
- Datensatzbasis: `v3`

Format:
- jedes Beispiel enthaelt `messages` im Chat-Format
- Assistant gibt immer nur JSON zurueck
- Routing-Modus: `intent_only`

Naechster Schritt:
- dieses Paket gegen ein lokales 7B-Instruct-Basismodell mit QLoRA trainieren
- danach nur den Router-Teil der Runtime umhaengen, nicht die kompletten Hypnose-Texte

