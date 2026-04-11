# Live API Guardrails

Diese Regeln gelten fuer dieses Repo ab sofort zwingend:

1. Live-OpenAI ist standardmaessig blockiert.
2. Bulk-Live-Laeufe duerfen nur starten, wenn alle drei Bedingungen gleichzeitig erfuellt sind:
   - `OPENAI_LIVE_API_ALLOWED=1`
   - eine gueltige Freigabedatei `backend/live_api_approval.json`
   - ein explizites CLI-Limit `--max-api-calls <n>`
3. Ohne diese drei Dinge duerfen Validierungs- und FT-Batch-Skripte nur den geplanten Call-Bedarf anzeigen, aber keine Live-Requests senden.
4. Die Freigabedatei ist absichtlich nicht versioniert. Als Vorlage dient `backend/live_api_approval.example.json`.
5. Fuer interaktive manuelle Einzeltests ist weiterhin ein bewusster FT-Lauf moeglich; fuer Massenlaeufe gilt immer die Sperre.

Betroffene Skripte:
- `backend/run_session_validation_matrix.py`
- `backend/run_phase4_semantic_ft_prototype.py --batch`
- `backend/run_session_sandbox.py --batch --semantic-provider ft`

Ziel:
- Keine stillen Massencalls mehr
- Keine versehentliche Aufzehrung des API-Guthabens
- Live-Laeufe nur mit expliziter Freigabe und harter Obergrenze
