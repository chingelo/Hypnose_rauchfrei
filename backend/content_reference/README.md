# Content Reference

Dieser Ordner enthaelt lokale Content-Snapshots, die vorher nur in
`C:\Projekte\hypnose_systemV2` lagen.

Aktive lokale Pfade:
- `guided_session_phases.json`
- `phase4_prompts.json`

Zusatzkataloge aus V2:
- `catalog/session/runtime_reference_texts.json`
- `catalog/forms/guided_forms.json`
- `catalog/emdr/emdr_text.json`

Regel:
- Neue Runtime-Abhaengigkeiten sollen nur noch auf lokale Dateien in diesem
  Repo zeigen.
- Direkte externe Pfade nach `hypnose_systemV2` sollen nicht mehr neu
  eingefuehrt werden.

Stand 10.04.2026:
- Die aktiven Backend-Pfade fuer Session-Content und Phase-4-Prompts zeigen
  inzwischen lokal auf dieses Repo.
- Direkte V2-Codefallbacks wurden aus den aktiven Backend-Dateien entfernt.
