# Gold V3

Gold V3 haertet `routing` und `slot_extraction` weiter in Richtung kanonischer Label-Ontologie.

Ziele:
- Synonym-Drift wie `first_time`, `see`, `hear`, `smell`, `unknown` weiter zurueckdruecken
- mehr kurze, knappe, aber inhaltlich gueltige Antworten auf den richtigen kanonischen Intent abbilden
- Gruppen-/Person-/Other-Slots weiter gegen umgangssprachliche Formulierungen absichern

Dateien:
- `routing_gold_v3.jsonl`
- `routing_train_v3.jsonl`
- `routing_eval_v3.jsonl`
- `slot_extraction_gold_v3.jsonl`
- `slot_extraction_train_v3.jsonl`
- `slot_extraction_eval_v3.jsonl`
- `gold_v3_review.json`

Generator:
- `backend/build_gold_v3_datasets.py`

