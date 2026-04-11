# test_app

## Rolle des Repos

Dieses Repo ist aktuell der operative Arbeitskern fuer:
- Session-Sandbox
- Phase-4-Logik
- semantisches Routing
- Finetune-/Eval-Arbeit
- produktnahe Session- und Sicherheitsregeln

Es ist **nicht** mehr sinnvoll, dieses Repo nur als kleines Diktier-/Audio-Chat-
Testprojekt zu lesen. Die aktuelle fachliche Wahrheit liegt in den zentralen
Dokumentationsdateien unter `docs/`.

## Zuerst lesen

Wenn du neu ins Repo kommst, lies in dieser Reihenfolge:

1. [AGENTS.md](/c:/Projekte/test_app/AGENTS.md)
2. [Produktvision und Zielanforderungen](/c:/Projekte/test_app/docs/PRODUKTVISION_UND_ZIELANFORDERUNGEN.md)
3. [Produktkontext und Systemgrenzen](/c:/Projekte/test_app/docs/PRODUKTKONTEXT_UND_SYSTEMGRENZEN.md)
4. [Produktregeln und Sessionmodell](/c:/Projekte/test_app/docs/PRODUKTREGELN_UND_SESSIONMODELL.md)
5. [Konsolidierung und Aufraeumplan](/c:/Projekte/test_app/docs/KONSOLIDIERUNG_UND_AUFRAEUMPLAN.md)
6. [Sandbox Handover](/c:/Projekte/test_app/docs/SANDBOX_HANDOVER_UND_SYSTEMSTATUS_2026-04-10.md)
7. [V2 Code Vergleich und Portplan](/c:/Projekte/test_app/docs/V2_CODE_VERGLEICH_UND_PORTPLAN.md)
8. [Session Access Integration](/c:/Projekte/test_app/docs/SESSION_ACCESS_INTEGRATION.md)
9. [Product Intake und Forms API](/c:/Projekte/test_app/docs/PRODUCT_INTAKE_UND_FORMS_API.md)
10. [V2 Abbau und Loeschcheckliste](/c:/Projekte/test_app/docs/V2_ABBAU_UND_LOESCHCHECKLISTE.md)

## Dokumentationsstruktur

### Master-Dateien

- [PRODUKTVISION_UND_ZIELANFORDERUNGEN.md](/c:/Projekte/test_app/docs/PRODUKTVISION_UND_ZIELANFORDERUNGEN.md)
  Zentrales Zielbild fuer Produkt, KI-Verhalten und Qualitaet
- [PRODUKTKONTEXT_UND_SYSTEMGRENZEN.md](/c:/Projekte/test_app/docs/PRODUKTKONTEXT_UND_SYSTEMGRENZEN.md)
  Verkauf, Website, Kundenkonto, Runtime-Grenzen
- [PRODUKTREGELN_UND_SESSIONMODELL.md](/c:/Projekte/test_app/docs/PRODUKTREGELN_UND_SESSIONMODELL.md)
  Sessionverbrauch, Endstatus, Ersatzfall, Zugriff

### Operative Arbeitsdoku

- [SANDBOX_HANDOVER_UND_SYSTEMSTATUS_2026-04-10.md](/c:/Projekte/test_app/docs/SANDBOX_HANDOVER_UND_SYSTEMSTATUS_2026-04-10.md)
- [SESSION_INAKTIVITAET_UND_VERBRAUCHSREGELN.md](/c:/Projekte/test_app/docs/SESSION_INAKTIVITAET_UND_VERBRAUCHSREGELN.md)
- [SESSION_TIMEOUT_UND_KOSTENMODELL.md](/c:/Projekte/test_app/docs/SESSION_TIMEOUT_UND_KOSTENMODELL.md)

### Therapeutische Referenzen

- [phase4_v2_zweige](/c:/Projekte/test_app/docs/phase4_v2_zweige)
- [imported_reference/hypnose_systemV2](/c:/Projekte/test_app/docs/imported_reference/hypnose_systemV2)
- [therapeutic_reference](/c:/Projekte/test_app/docs/therapeutic_reference)
- [product_safety](/c:/Projekte/test_app/docs/product_safety)
- [V2_CODE_VERGLEICH_UND_PORTPLAN.md](/c:/Projekte/test_app/docs/V2_CODE_VERGLEICH_UND_PORTPLAN.md)
- [SESSION_ACCESS_INTEGRATION.md](/c:/Projekte/test_app/docs/SESSION_ACCESS_INTEGRATION.md)
- [PRODUCT_INTAKE_UND_FORMS_API.md](/c:/Projekte/test_app/docs/PRODUCT_INTAKE_UND_FORMS_API.md)

## Lokale importierte Pflichtquellen

Die wichtigsten frueher externen Referenzquellen aus
`C:\Projekte\hypnose_systemV2` sind jetzt lokal in diesem Repo gespiegelt:

- Produkt- und Website-Referenzen unter
  [docs/imported_reference/hypnose_systemV2](/c:/Projekte/test_app/docs/imported_reference/hypnose_systemV2)
- aktive Content-Snapshots unter
  [backend/content_reference](/c:/Projekte/test_app/backend/content_reference)
- therapeutische Hauptquellen unter
  [docs/therapeutic_reference](/c:/Projekte/test_app/docs/therapeutic_reference)
- Sicherheitsquellen unter
  [docs/product_safety](/c:/Projekte/test_app/docs/product_safety)
- Migrations-Snapshots fuer Test- und Script-Portierung unter
  [migration_reference/hypnose_systemV2](/c:/Projekte/test_app/migration_reference/hypnose_systemV2)
- Code-Snapshots fuer V2-Vergleiche unter
  [migration_reference/hypnose_systemV2/code_source](/c:/Projekte/test_app/migration_reference/hypnose_systemV2/code_source)
- Roh-Snapshot der Phase-4-Textquellen unter
  [migration_reference/hypnose_systemV2/phase4_source_texts](/c:/Projekte/test_app/migration_reference/hypnose_systemV2/phase4_source_texts)
- selektiver Vollarchiv-Snapshot unter
  [hypnose_systemV2_selective_source_archive.zip](/c:/Projekte/test_app/migration_reference/hypnose_systemV2/hypnose_systemV2_selective_source_archive.zip)
- Vollarchiv-Manifest unter
  [selective_source_archive_manifest.txt](/c:/Projekte/test_app/migration_reference/hypnose_systemV2/selective_source_archive_manifest.txt)
- V2-Abbaucheckliste unter
  [docs/V2_ABBAU_UND_LOESCHCHECKLISTE.md](/c:/Projekte/test_app/docs/V2_ABBAU_UND_LOESCHCHECKLISTE.md)

Wichtig:
- Die Master-Dateien in `docs/` sind die verdichtete Wahrheit.
- Die importierten Dateien sind Referenz- und Herkunftsquellen.
- Aktive Backend-Dateien nutzen fuer Content und Produktpfade jetzt lokale
  Quellen in diesem Repo statt direkter V2-Codefallbacks.

## Wichtige Verzeichnisse

- `backend/`
  Operative Runtime, Sandbox, Routing, Tests
- `backend/content_reference/`
  Lokale Snapshots externer Session-/Phase-4-Contentquellen
- `docs/`
  Master-Dokumentation, Regeln, Handover, Referenzen
- `migration_reference/`
  gesicherte Portierungs- und Vergleichsquellen aus V2
- `flutter_app/`
  Client-App

## Startpunkte

### Backend

```powershell
cd C:\Projekte\test_app\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Flutter-App

```powershell
cd C:\Projekte\test_app\flutter_app
flutter pub get
flutter run --dart-define=BACKEND_URL=http://127.0.0.1:8000
```

Fuer Android-Emulator:

```powershell
flutter run --dart-define=BACKEND_URL=http://10.0.2.2:8000
```

## Gesperrte Bereiche

Die Regeln in [AGENTS.md](/c:/Projekte/test_app/AGENTS.md) sind strikt.

Insbesondere gesperrt:
- Diktierfunktion
- Audio-Chat / Voice-Loop

Diese Bereiche duerfen nicht geaendert werden.
