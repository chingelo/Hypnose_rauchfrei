# AGENTS.md (test_app)

Diese Regeln sind strikt und ohne Ausnahme einzuhalten.

## Harte Sperrregel

Ab sofort sind folgende Bereiche strikt gesperrt und duerfen NICHT geaendert werden:

1. Diktierfunktion
2. Audio-Chat Funktion (inkl. Voice-Loop Verhalten)

## Verbotene Aenderungen (strict forbidden)

- Keine Aenderung an Logik, Flows, Zustandswechseln oder Events der Diktierfunktion.
- Keine Aenderung an Logik, Flows, Zustandswechseln oder Events des Audio-Chats.
- Keine Aenderung an Start/Stop/Resume/Pause Verhalten von Diktieren oder Audio-Chat.
- Keine Aenderung an WebSocket-Protokoll fuer Audio-Chat (`/audio/listen`, `/audio/speak`).
- Keine Refactors, "kleine Fixes", Umbenennungen oder Format-Only Edits in diesen Bereichen.

## Besonders geschuetzte Stellen

- Frontend: `flutter_app/lib/main.dart`
  - Alles rund um Diktieren (`_startDictation`, `_stopDictation`, `_toggleDictation` etc.)
  - Alles rund um Audio-Chat / Voice-Loop (`_toggleVoiceChat`, `_ensureVoiceLoopConnected`, `_connectVoiceSpeak`, `_connectVoiceListen`, `_handleVoiceSpeakMessage`, `_handleVoiceAudioChunk` etc.)
- Backend: `backend/main.py`
  - Alles rund um `/audio/speak` und `/audio/listen`
  - VoiceLoop-Manager und zugehoerige Audio-Chat-Logik

## Vorgehen bei zukuenftigen Tasks

- Wenn ein Wunsch diese gesperrten Bereiche beruehrt: NICHT implementieren.
- Stattdessen kurz melden: "Gesperrter Bereich laut AGENTS.md (Diktierfunktion/Audio-Chat)."
- Nur ausserhalb der gesperrten Bereiche arbeiten.
