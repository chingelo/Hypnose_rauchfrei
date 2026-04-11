# Gold V2

Gold V2 haertet zunaechst nur `routing` und `slot_extraction`.

Ziele:
- reale Problemfaelle aus der Bug-Historie aufnehmen
- offensichtliche Dubletten entfernen
- V1-Synthese beibehalten, aber durch Grenzfaelle ergaenzen
- sichtbar machen, wo die aktuelle Helper-Logik semantisch noch zu grob ist

Dateien:
- `routing_gold_v2.jsonl`
- `routing_train_v2.jsonl`
- `routing_eval_v2.jsonl`
- `slot_extraction_gold_v2.jsonl`
- `slot_extraction_train_v2.jsonl`
- `slot_extraction_eval_v2.jsonl`
- `gold_v2_review.json`

Generator:
- `backend/build_gold_v2_datasets.py`

Aktueller Stand:
- Gold V2 dient als Soll-Zustand fuer Runtime-Heuristik und spaetere Modell-Evals
- `gold_v2_review.json` zeigt, ob die aktuelle Helper-Logik mit diesem Soll bereits uebereinstimmt

