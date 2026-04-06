# Test App (Diktieren + Audio-Chat)

Diese App bildet den Aivora-Nutzwert fuer:

- Diktieren in das Eingabefeld
- Audio-Chat (sprechen, Modellantwort als Sprache)
- Chat mit deinem FT-Modell `ft:gpt-3.5-turbo-1106:personal::AzSLcCUs`

## Struktur

- `backend/` FastAPI-Backend fuer OpenAI-Calls und Session-History
- `flutter_app/` einfache Flutter-Client-App mit Aivora-aehnlicher Input-Leiste

## 1) Backend starten

```powershell
cd C:\Projekte\test_app\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Dann in `.env` deinen echten Key setzen:

```env
OPENAI_API_KEY=sk-...
HYPNOSE_MODEL_ID=ft:gpt-3.5-turbo-1106:personal::AzSLcCUs
```

Start:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 2) Flutter-App starten

```powershell
cd C:\Projekte\test_app\flutter_app
flutter pub get
flutter run --dart-define=BACKEND_URL=http://127.0.0.1:8000
```

Hinweis fuer Android-Emulator:

```powershell
flutter run --dart-define=BACKEND_URL=http://10.0.2.2:8000
```

## Buttons unten

- `-` / `+`: Chat-Schriftgroesse
- `Audio`: Audio-Chat an/aus
- `Mikrofon`: Diktieren an/aus
- `Senden`: Text senden

## Technischer Hinweis

Der Build ist bewusst schlank gehalten. Die komplette Aivora-Voice-Loop-Architektur
(WS `/audio/listen` + `/audio/speak`, Alarm-Sonderlogik, etc.) wird nicht in dieses
Mini-Projekt migriert.
