from __future__ import annotations

import argparse
import hashlib
import io
import json
import msvcrt
import os
import re
import shutil
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
import wave
import winsound
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = Any  # type: ignore[misc,assignment]

from live_api_guard import DEFAULT_APPROVAL_FILE, LiveApiCallBudget, build_live_api_budget
from openai_semantic_backend import (
    OpenAISemanticBackend,
    resolve_ft_backend,
    resolve_openai_router_backend,
)
from session_sandbox_orchestrator import (
    ScriptNodeSpec,
    SemanticModelDecision,
    SemanticNodeSpec,
    available_node_ids,
    build_request,
    get_node_spec,
    maybe_render_entry_script,
    render_script_node,
    repair_semantic_payload,
    script_reply_for_decision,
    validate_semantic_decision,
)


ENV_PATHS = (
    Path(r"C:\Projekte\test_app\backend\.env"),
)
TTS_CACHE_DIR = Path(tempfile.gettempdir()) / "test_app_session_sandbox_tts"
DEFAULT_LOCAL_INTENT_ADAPTER_DIR = (
    Path(r"C:\Projekte\test_app\backend\finetune_data\v3\local_router_intent\artifacts\router_qlora_qwen25_3b_v3_intent")
)
_TTS_READY: bool | None = None
_TTS_ERROR: str | None = None
_GOOGLE_TTS_CLIENT: Any = None
_ELEVENLABS_GERMAN_TTS_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgegenueber\b", re.IGNORECASE), "gegenüber"),
    (re.compile(r"\bfuer\b", re.IGNORECASE), "für"),
    (re.compile(r"\bzurueck([a-z]*)\b", re.IGNORECASE), "zurück"),
    (re.compile(r"\brueck([a-z]*)\b", re.IGNORECASE), "rück"),
    (re.compile(r"\bueberg([a-z]*)\b", re.IGNORECASE), "überg"),
    (re.compile(r"\bueber([a-z]*)\b", re.IGNORECASE), "über"),
    (re.compile(r"\bgefuehl([a-z]*)\b", re.IGNORECASE), "gefühl"),
    (re.compile(r"\bfuehl([a-z]*)\b", re.IGNORECASE), "fühl"),
    (re.compile(r"\bspuer([a-z]*)\b", re.IGNORECASE), "spür"),
    (re.compile(r"\bkoerper([a-z]*)\b", re.IGNORECASE), "körper"),
    (re.compile(r"\bhoer([a-z]*)\b", re.IGNORECASE), "hör"),
    (re.compile(r"\bmoech([a-z]*)\b", re.IGNORECASE), "möch"),
    (re.compile(r"\bwaehr([a-z]*)\b", re.IGNORECASE), "währ"),
    (re.compile(r"\bfrueh([a-z]*)\b", re.IGNORECASE), "früh"),
    (re.compile(r"\bmued([a-z]*)\b", re.IGNORECASE), "müd"),
    (re.compile(r"\bberuehr([a-z]*)\b", re.IGNORECASE), "berühr"),
    (re.compile(r"\bdrueck([a-z]*)\b", re.IGNORECASE), "drück"),
    (re.compile(r"\bkoenn([a-z]*)\b", re.IGNORECASE), "könn"),
    (re.compile(r"\bwuerd([a-z]*)\b", re.IGNORECASE), "würd"),
    (re.compile(r"\boeffn([a-z]*)\b", re.IGNORECASE), "öffn"),
    (re.compile(r"\bklaer([a-z]*)\b", re.IGNORECASE), "klär"),
    (re.compile(r"\baehn([a-z]*)\b", re.IGNORECASE), "ähn"),
    (re.compile(r"\baeuss([a-z]*)\b", re.IGNORECASE), "äuß"),
    (re.compile(r"\bvoellig\b", re.IGNORECASE), "völlig"),
    (re.compile(r"\bpraesent\b", re.IGNORECASE), "präsent"),
    (re.compile(r"\bnaeher([a-z]*)\b", re.IGNORECASE), "näher"),
    (re.compile(r"\bnaech([a-z]*)\b", re.IGNORECASE), "näch"),
    (re.compile(r"\bstaerk([a-z]*)\b", re.IGNORECASE), "stärk"),
    (re.compile(r"\bguelt([a-z]*)\b", re.IGNORECASE), "gült"),
    (re.compile(r"\baufloes([a-z]*)\b", re.IGNORECASE), "auflös"),
    (re.compile(r"\bausloes([a-z]*)\b", re.IGNORECASE), "auslös"),
    (re.compile(r"\bgeloes([a-z]*)\b", re.IGNORECASE), "gelös"),
    (re.compile(r"\blaesst\b", re.IGNORECASE), "lässt"),
    (re.compile(r"\bfaellt\b", re.IGNORECASE), "fällt"),
)
AUTO_CONTINUE_ON_SILENCE: dict[str, str] = {
    "session_phase1_preflight_check": "In Ordnung. Dann starten wir jetzt.",
    "session_phase1_anchor_after_setup": "In Ordnung. Dann gehen wir jetzt weiter.",
    "session_phase1_anchor_before_focus": "In Ordnung. Dann richte deine Aufmerksamkeit jetzt langsam weiter nach innen.",
}
PHASE12_SILENCE_TIMEOUTS: tuple[float, ...] = (10.0, 14.0, 18.0, 18.0, 15.0)
PHASE36_SILENCE_TIMEOUTS: tuple[float, ...] = (12.0, 16.0, 20.0, 20.0, 15.0)
MAX_SILENCE_ATTEMPTS_BEFORE_OUTRO = 8
EXPLORATORY_SCENE_SILENCE_NODES = {
    "dark_scene_who",
    "dark_scene_audio_detail",
    "dark_scene_other_sense",
    "dark_scene_first_spuerbar",
    "dark_scene_people_who",
    "dark_scene_happening",
}
SESSION_SOFT_LIMIT_SECONDS = 90 * 60
SESSION_HARD_LIMIT_SECONDS = 120 * 60
SESSION_ABSOLUTE_LIMIT_SECONDS = 150 * 60
INACTIVITY_WARNING_TEXT = (
    "Beachte bitte: Wenn du diese Sitzung nicht in einem zusammenhaengenden Rahmen fortsetzen kannst "
    "und ich ueber laengere Zeit keine Rueckmeldung von dir bekomme, kann ich dich nicht mehr sicher und "
    "stimmig weiter durch diesen Prozess begleiten. Wenn ich in den naechsten Momenten weiter nichts von dir "
    "hoere, beginne ich mit einer ruhigen Ausleitung."
)
INACTIVITY_FINAL_WARNING_TEXT = (
    "Ich nehme noch immer keine Rueckmeldung von dir wahr. "
    "Wenn jetzt weiterhin nichts kommt, leite ich dich ruhig aus der Sitzung heraus "
    "und beende sie anschliessend an einem sicheren Punkt."
)
INACTIVITY_OUTRO_TEXT = (
    "Da weiterhin keine Rueckmeldung kommt, beginne ich jetzt eine ruhige Ausleitung. "
    "Du musst nichts weiter tun. Hoere einfach noch einen Moment zu und lass dich in deinem Tempo wieder ganz ins Hier und Jetzt begleiten."
)
SESSION_WALLCLOCK_END_TEXT = (
    "Die zulaessige Sitzungsdauer ist erreicht. Ich beende die Sitzung jetzt an diesem sicheren Punkt. "
    "Wenn ein technisches Problem vorlag, kann das anschliessend geprueft werden."
)
NODE_EMPTY_INPUT_REPLIES: dict[str, list[str]] = {
    "scene_access_followup": [
        "Falls du gerade nichts erkennen kannst, lass dir einfach kurz einen Moment Zeit. Manchmal braucht es etwas Zeit, bis etwas auftaucht, besonders wenn wir weit zurueckgehen. Wenn etwas erkennbar oder wahrnehmbar wird, reicht dein erster Eindruck.",
        "Es ist in Ordnung, wenn dort noch nicht gleich etwas Greifbares auftaucht. Gerade wenn wir weit zurueckgehen, braucht es manchmal einen Augenblick. Sobald sich etwas zeigt, reicht dein erster Eindruck dazu.",
        "Lass dir dort ruhig noch einen Moment Zeit. Manchmal wird erst nach und nach etwas wahrnehmbar, besonders in fruehen Szenen. Wenn etwas auftaucht, reicht eine kurze Rueckmeldung dazu.",
    ],
    "dark_known_branch": [
        "Ich frage dich noch einmal etwas klarer: Kommt dir dieses Gefuehl bereits bekannt vor, oder begegnest du ihm hier zum ersten Mal?",
        "Es reicht deine erste innere Einordnung. Kommt dir dieses Gefuehl eher bekannt vor, oder begegnest du ihm hier zum ersten Mal?",
        "Spuer nur kurz nach, ohne lange zu ueberlegen: eher bekannt oder eher zum ersten Mal.",
        "Du musst es noch nicht erklaeren. Eine kurze Einordnung reicht: eher bekannt oder eher zum ersten Mal.",
        "Sobald du es einordnen kannst, genuegt eine kurze Rueckmeldung: eher bekannt oder eher zum ersten Mal.",
    ],
    "origin_trigger_known_branch": [
        "Ich frage dich noch einmal etwas klarer: Kennst du dieses Gefuehl bereits aus frueheren Momenten, oder begegnet es dir hier in dieser Szene zum ersten Mal?",
        "Es reicht deine erste innere Einordnung. Kennst du dieses Gefuehl eher schon aus frueheren Momenten, oder ist es hier neu?",
        "Spuer nur kurz nach, ohne lange zu ueberlegen: eher schon bekannt oder eher hier neu.",
        "Du musst es noch nicht erklaeren. Eine kurze Einordnung reicht: eher schon bekannt oder eher hier neu.",
        "Sobald du es einordnen kannst, genuegt eine kurze Rueckmeldung: eher schon bekannt oder eher hier neu.",
    ],
    "origin_cause_owner": [
        "Ich frage dich noch einmal etwas klarer: Geht es hier eher um etwas in dir selbst, das durch {trigger_focus_ref} beruehrt wird, oder ist {trigger_focus_ref} selbst der eigentliche Ausloeser?",
        "Sobald du es einordnen kannst, ordne einfach ein, ob dieses Thema eher in dir selbst liegt oder ob {trigger_focus_ref} eher der eigentliche Ausloeser ist.",
        "Wenn es dir dazu klarer wird, reicht eine kurze Einordnung: eher etwas in dir selbst oder eher {trigger_focus_ref}.",
    ],
    "origin_scene_relevance": [
        "Ich frage dich noch einmal etwas klarer: Ist in genau dieser Szene etwas, das wir hier anschauen oder loesen sollten, oder merkst du eher, dass dich dieses Gefuehl noch weiter zu etwas Frueherem fuehrt?",
        "Sobald du es einordnen kannst, ordne einfach ein, ob in genau dieser Szene etwas angeschaut oder geloest werden sollte oder ob es eher noch weiter zurueck fuehrt.",
        "Wenn es dir dazu klarer wird, reicht eine kurze Einordnung: eher hier bearbeiten oder eher noch weiter zurueck.",
    ],
    "origin_other_target_kind": [
        "Ich frage dich noch einmal etwas klarer: Ist das, was du gerade benannt hast, eher eine Gruppe, eher eine bestimmte Person oder eher etwas anderes in dieser Situation?",
        "Sobald du es einordnen kannst, ordne einfach ein, ob das eher eine Gruppe, eher eine bestimmte Person oder eher etwas anderes in dieser Situation ist.",
        "Wenn es dir dazu klarer wird, reicht eine kurze Einordnung: eher Gruppe, eher bestimmte Person oder eher etwas anderes in der Situation.",
    ],
    "origin_person_name": [
        "Wenn noch nicht klar ist, wer diese Person genau ist, hol sie innerlich etwas naeher heran und schau einfach, ob deutlicher wird, wer das sein koennte. Wenn es klarer wird, reicht der Name oder eine kurze Beschreibung.",
        "Sobald diese Person klarer vor dir steht, reicht der Name oder eine kurze Beschreibung davon, wer sie sein koennte.",
        "Wenn es noch nicht ganz eindeutig ist, ist das in Ordnung. Schau einfach, ob dir zu dieser Person noch etwas klarer wird.",
        "Ich brauche hier nur noch eine kurze Rueckmeldung dazu, wer diese Person ist oder wie du sie beschreiben wuerdest.",
        "Wenn du sie innerlich erkennen kannst, sag mir bitte jetzt einfach den Namen oder beschreibe kurz, wer diese Person ist.",
    ],
    "group_representative_name": [
        "Welche Person aus dieser Gruppe kannst du mir namentlich nennen, die wir stellvertretend fuer diese Gruppendynamik nehmen sollen?",
        "Sobald dir die passende Person klarer wird, reicht der Name oder eine kurze Beschreibung der Person, die fuer dich stellvertretend fuer diese Dynamik steht.",
        "Wenn es dir dazu klarer wird, reicht der Name oder eine kurze Beschreibung dieser Person.",
    ],
    "group_specific_person_name": [
        "Wenn schon klar ist, welche Person aus dieser Gruppe hier entscheidend ist, reicht der Name oder eine kurze Beschreibung genau dieser Person.",
        "Sobald dir die entscheidende Person klarer wird, reicht der Name oder eine kurze Beschreibung der Person, die hier fuer dich entscheidend ist.",
        "Wenn es dir dazu klarer wird, reicht der Name oder eine kurze Beschreibung dieser Person.",
    ],
    "group_multiple_people_name": [
        "Wenn mehrere Personen wichtig sind, reicht zuerst der Name oder eine kurze Beschreibung der Person, mit der wir beginnen sollen.",
        "Sobald dir die erste Person klarer wird, reicht der Name oder eine kurze Beschreibung der ersten Person, mit der wir anfangen sollen.",
        "Wenn es dir dazu klarer wird, reicht der Name oder eine kurze Beschreibung der ersten Person.",
    ],
    "group_multiple_required_name": [
        "Wenn dir die erste Person aus dieser Gruppe klarer wird, reicht der Name oder eine kurze Beschreibung der Person, mit der wir jetzt beginnen sollen.",
        "Sobald dir die erste Person klarer wird, reicht der Name oder eine kurze Beschreibung der ersten Person, mit der wir starten.",
        "Wenn es dir dazu klarer wird, reicht der Name oder eine kurze Beschreibung der ersten Person.",
    ],
    "group_next_person_name": [
        "Welche weitere Person aus dieser Gruppe ist jetzt als naechstes wichtig?",
        "Sobald dir die naechste Person klarer wird, reicht der Name oder eine kurze Beschreibung der weiteren Person aus dieser Gruppe, die jetzt als naechstes wichtig ist.",
        "Wenn es dir dazu klarer wird, reicht der Name oder eine kurze Beschreibung dieser weiteren Person.",
    ],
    "group_person_trigger_reason": [
        "Was genau ist es an {named_person}, das dieses ungute Gefuehl in dir ausloest?",
        "Sobald dir dazu etwas klarer wird, reicht eine kurze Beschreibung davon, was {named_person} in dir ausloest.",
        "Wenn es dir klarer wird, beschreibe einfach kurz, warum {named_person} hier gerade so bedeutsam ist.",
    ],
    "group_person_trigger_role": [
        "Geht dieses ungute Gefuehl eher direkt von {named_person} aus, oder steht {named_person} eher stellvertretend fuer etwas Groesseres, das in dieser Situation wirksam ist?",
        "Sobald dir dazu etwas klarer wird, reicht eine kurze Einordnung, ob das eher direkt von {named_person} ausgeht oder ob {named_person} eher fuer eine groessere Dynamik in dieser Situation steht.",
        "Wenn es dir klarer wird, ordne einfach kurz ein, ob {named_person} selbst der direkte Ausloeser ist oder eher stellvertretend fuer etwas Groesseres in dieser Situation steht.",
    ],
    "group_person_trigger_core": [
        "Worin liegt bei {named_person} der eigentliche Kern: in etwas, das {named_person} getan hat, gesagt hat, ausgestrahlt hat oder in dir ausgeloest hat?",
        "Sobald dir dazu etwas klarer wird, reicht eine kurze Beschreibung davon, worin der eigentliche Kern bei {named_person} liegt.",
        "Wenn es dir klarer wird, beschreibe einfach kurz, was bei {named_person} den Kern dieses Themas ausmacht.",
    ],
    "dark_scene_perception": [
        "Sobald dir dort etwas klarer wird, reicht eine kurze Beschreibung davon, was du dort sonst noch wahrnimmst.",
        "Wenn die Szene klarer wird, beschreibe einfach kurz, was dort sonst noch wahrnehmbar wird.",
        "Es reicht eine kurze Einordnung dazu, was dort sonst noch wahrnehmbar wird.",
    ],
    "dark_scene_mode_clarify": [
        "Sobald du es einordnen kannst, reicht eine kurze Einordnung, ob du dort eher jemanden siehst oder etwas hoerst.",
        "Wenn es dir klarer wird, reicht eine kurze Einordnung: eher sehen, eher hoeren oder beides noch nicht klar.",
        "Es reicht eine kurze Einordnung: eher sehen, eher hoeren oder beides noch nicht klar.",
    ],
    "dark_scene_who": [
        "Sobald dort etwas klarer sichtbar wird, reicht eine kurze Beschreibung davon, was dort fuer dich sichtbar wird.",
        "Wenn die Szene klarer wird, reicht eine kurze Beschreibung von dem, was du dort siehst.",
        "Es reicht eine kurze Beschreibung von dem, was dort sichtbar wird.",
    ],
    "dark_scene_audio_detail": [
        "Sobald dort etwas klarer hoerbar wird, reicht eine kurze Beschreibung davon, was dort fuer dich hoerbar wird.",
        "Wenn die Szene klarer wird, reicht eine kurze Beschreibung von dem, was du dort hoerst.",
        "Es reicht eine kurze Beschreibung von dem, was dort hoerbar wird.",
    ],
    "dark_scene_other_sense": [
        "Sobald dort ueber Koerper, Geruch, Geschmack oder Temperatur etwas deutlicher wird, reicht eine kurze Beschreibung davon, was wahrnehmbar wird.",
        "Wenn es dir dort klarer wird, reicht eine kurze Beschreibung von dem, was ueber diese anderen Sinne auftaucht.",
        "Es reicht eine kurze Beschreibung von dem, was ueber diese anderen Sinne auftaucht.",
    ],
    "dark_scene_first_spuerbar": [
        "Sobald dir dort etwas klarer wird, reicht eine kurze Beschreibung davon, was dort als Erstes am deutlichsten spuerbar wird.",
        "Wenn es dir dort klarer wird, reicht eine kurze Beschreibung von dem, was zuerst am deutlichsten spuerbar ist.",
        "Es reicht eine kurze Beschreibung von dem, was dort zuerst am deutlichsten spuerbar ist.",
    ],
    "dark_scene_people_who": [
        "Sobald du dort etwas klarer erkennen kannst, reicht eine kurze Beschreibung davon, wer oder was dort fuer dich wahrnehmbar wird.",
        "Wenn es dir dort klarer wird, reicht eine kurze Beschreibung davon, wer oder was dort fuer dich erkennbar ist.",
        "Es reicht eine kurze Beschreibung davon, wer oder was dort fuer dich wahrnehmbar wird.",
    ],
    "dark_scene_happening": [
        "Sobald dir die Szene klarer wird, reicht eine kurze Beschreibung davon, was dort gerade passiert.",
        "Wenn es dir dazu klarer wird, beschreibe einfach kurz, was in diesem Moment geschieht.",
        "Es reicht eine kurze Beschreibung davon, was dort in diesem Moment geschieht.",
    ],
    "dark_scene_age": [
        "Sobald dir dein spontanes Alter klarer wird, reicht dein Alter dort als erste Rueckmeldung.",
        "Wenn es dir dazu klarer wird, reicht dein erster Impuls zu deinem Alter in dieser Szene.",
        "Es reicht dein erster Impuls zu deinem Alter in dieser Szene.",
    ],
    "dark_scene_feeling_intensity": [
        "Sobald dir das deutlicher wird, reicht eine kurze Beschreibung davon, wie sich dieses ungute Gefuehl dort zeigt und wie stark es ist.",
        "Wenn es dir dazu klarer wird, beschreibe einfach kurz, wie das ungute Gefuehl dort gerade wirkt.",
        "Es reicht eine kurze Beschreibung davon, wie sich das ungute Gefuehl dort zeigt.",
    ],
    "dark_scene_immediate_feeling": [
        "Sobald dir das klarer wird, reicht eine kurze Beschreibung davon, was du dort ganz unmittelbar fuehlst.",
        "Wenn es dir dazu klarer wird, beschreibe einfach kurz, was dort jetzt unmittelbar gefuehlt wird.",
        "Es reicht eine kurze Beschreibung davon, was dort jetzt unmittelbar gefuehlt wird.",
    ],
}
QUESTION_ANSWER_HINTS: dict[str, str] = {
    "session_phase1_preflight_check": (
        "Hier geht es nur um den Start: ob die Person jetzt beginnen kann oder vorher noch kurz etwas braucht, "
        "fragen will oder technisch anpassen muss."
    ),
    "session_phase1_anchor_after_setup": (
        "Hier geht es nur darum, ob die aeusseren Bedingungen jetzt passen: ruhiger Ort, bequeme Position, "
        "Kopfhoerer falls moeglich, stabile Verbindung und ein sicheres, angenehmes Umfeld."
    ),
    "session_phase1_anchor_before_focus": (
        "Hier geht es nur darum, ob der innere Fokus jetzt stimmig ist. Die Person muss nichts leisten oder erzwingen, "
        "sondern sich nur auf die Stimme und den Prozess einlassen."
    ),
    "session_phase2_ready": (
        "Hier geht es nur um die Bereitschaft fuer den naechsten Schritt. Noch nicht um Tiefe, Wirkung oder Ergebnis."
    ),
    "session_phase2_eyes_closed": (
        "Hier geht es nur darum, ob die Augen jetzt geschlossen sind oder ob noch ein kurzer Moment dafuer gebraucht wird."
    ),
    "session_phase2_scene_found": (
        "Hier geht es nicht um eine Ressource. Gemeint ist ein konkreter Moment oder eine konkrete Situation, "
        "in der das Verlangen nach der Zigarette, der Suchtdruck oder das damit verbundene ungute Gefuehl deutlich spuerbar war, "
        "damit wir mit genau diesem Erleben weiterarbeiten koennen."
    ),
    "session_phase2_feel_clear": (
        "Hier geht es nur darum, ob das Verlangen nach der Zigarette oder das damit verbundene ungute Gefuehl in der gewaehlten Situation jetzt deutlich genug spuerbar ist, "
        "um damit weiterzugehen."
    ),
    "session_phase2_scale_clear": (
        "Hier geht es nur um das Verstaendnis der Belastungsskala: 1 steht fuer ruhig und frei, 10 fuer maximal intensiv."
    ),
    "session_phase2_scale_before": (
        "Hier soll die Person nur einen aktuellen Zahlenwert von 1 bis 10 fuer die Intensitaet des Gefuehls nennen."
    ),
    "session_phase2_scale_after": (
        "Hier soll die Person erneut einen Zahlenwert von 1 bis 10 nennen, nachdem der vorige Schritt durchlaufen wurde."
    ),
    "session_phase2_continue_to_main": (
        "Hier geht es nur darum, ob es fuer die Person jetzt passt, von diesem Vorbereitungsabschnitt in den Hauptteil der Session ueberzugehen."
    ),
    "phase4_common_sees_younger_self": (
        "Hier geht es nur darum, ob die Person ihr damaliges Ich in der Szene jetzt wahrnehmen kann."
    ),
    "phase4_common_understood": (
        "Hier geht es nur darum, ob das damalige Ich die Erklaerung des heutigen Ichs bereits verstanden hat."
    ),
    "phase4_common_feel_after_learning": (
        "Hier soll die Person nur kurz wiedergeben, wie es sich nach diesem Schritt jetzt anfuehlt."
    ),
    "phase4_common_feel_after_aversion": (
        "Hier soll die Person nur kurz wiedergeben, wie es sich nach der Rueckkehr zur ersten Zigarette jetzt anfuehlt."
    ),
    "dark_scene_immediate_feeling": (
        "Hier geht es nur um den ersten unmittelbaren Gefuehlsimpuls in diesem Moment. "
        "Gemeint sind kurze emotionale oder koerpernahe Beschreibungen wie Druck, Enge, Angst, Wut, Scham, Traurigkeit oder Ohnmacht."
    ),
    "dark_known_branch": (
        "Hier reicht der erste innere Eindruck. Gemeint ist nur die kurze Einordnung, ob dieses Gefuehl vertraut wirkt oder ob es sich wie ein erster Ursprung anfuehlt."
    ),
    "origin_trigger_known_branch": (
        "Hier geht es zuerst nur um die Einordnung, ob das in dieser Ursprungsszene auftauchende Gefuehl "
        "bereits aus frueheren Momenten bekannt ist oder ob es sich hier zum ersten Mal so zeigt."
    ),
    "phase4_common_done_signal": (
        "Hier geht es nur darum, ob die Person mit dem Sammelkorb wieder im Raum der Veraenderung beziehungsweise im magischen Sessel angekommen ist."
    ),
    "origin_trigger_source": (
        "Hier geht es nur darum, was oder wer in der Ursprungsszene das ungute Gefuehl am staerksten ausloest. Das kann eine Person, eine Gruppe, eine Aussage, ein Verhalten oder eine Situation sein."
    ),
    "origin_cause_owner": (
        "Hier geht es um die Einordnung, ob das Thema vor allem aus etwas in der Person selbst kommt, das in diesem Moment beruehrt oder aktiviert wird, "
        "oder ob der eigentliche Ausloeser eher bei dem benannten Gegenueber liegt, zum Beispiel bei einer Person oder einer Gruppe."
    ),
    "origin_scene_relevance": (
        "Hier geht es zuerst darum, ob die aktuell erreichte Szene direkt hier bearbeitet werden soll "
        "oder ob sie eher noch weiter zurueck zu einem frueheren Entstehungspunkt fuehrt."
    ),
    "origin_other_target_kind": (
        "Hier geht es nur darum, ob das benannte Gegenueber eher eine Gruppe oder eher eine bestimmte einzelne Person ist."
    ),
    "origin_person_name": (
        "Hier geht es zuerst darum, ob die Person schon konkret erkannt werden kann. Wenn es klarer wird, reicht der Name oder eine kurze Beschreibung. Wenn noch nicht klar ist, wer es genau ist, ist auch genau das eine brauchbare Rueckmeldung."
    ),
    "group_person_trigger_reason": (
        "Hier geht es nur darum, was genau an der ausgewaehlten Person dieses ungute Gefuehl ausloest oder warum diese Person jetzt gerade so bedeutsam ist."
    ),
    "group_person_trigger_role": (
        "Hier geht es nur darum, ob die ausgewaehlte Person selbst der direkte Ausloeser ist oder eher stellvertretend fuer die Gruppendynamik steht."
    ),
    "group_person_trigger_core": (
        "Hier geht es darum, ob schon ein erstes Verstaendnis da ist, warum diese Person so reagiert oder wofuer dieses Verhalten steht. "
        "Wenn das noch nicht klar ist, ist auch genau diese Rueckmeldung brauchbar, weil der anschliessende Perspektivwechsel helfen soll, das besser zu verstehen."
    ),
    "dark_scene_perception": (
        "Hier geht es nur darum, was in dieser dunkleren Ursprungsszene sonst noch wahrnehmbar ist, zum Beispiel Bilder, Stimmen, Geraeusche, Koerperwahrnehmung oder Atmosphaere."
    ),
    "dark_scene_mode_clarify": (
        "Hier geht es nur darum, ob in dieser dunkleren Ursprungsszene eher etwas gesehen oder eher etwas gehoert wird."
    ),
    "scene_access_followup": (
        "Hier geht es nur darum, ob statt eines klaren Bildes ueber einen anderen Sinn schon etwas wahrnehmbar wird, zum Beispiel ueber Hoeren, Riechen, Schmecken oder ein bestimmtes Gefuehl im Koerper. "
        "Wenn kein klares Bild kommt, ist dieser nichtvisuelle Zugang ein voll gueltiger Weg."
    ),
    "dark_scene_who": (
        "Hier geht es nur darum, was in dieser Ursprungsszene konkret sichtbar wird, zum Beispiel Personen, eine Gruppe, ein Raum oder etwas anderes."
    ),
    "dark_scene_audio_detail": (
        "Hier geht es nur darum, was in dieser Ursprungsszene konkret hoerbar wird."
    ),
    "dark_scene_other_sense": (
        "Hier geht es nur darum, was in dieser Ursprungsszene ueber Koerper, Geruch, Geschmack oder Temperatur wahrnehmbar wird, wenn nichts klar gesehen oder gehoert wird."
    ),
    "dark_scene_first_spuerbar": (
        "Hier geht es nur darum, was in dieser Ursprungsszene nach diesem anderen Sinneszugang als Erstes am deutlichsten spuerbar wird."
    ),
    "dark_scene_people_who": (
        "Hier geht es nur darum, wer oder was in dieser Ursprungsszene fuer dich genauer erkennbar wird."
    ),
    "dark_scene_happening": (
        "Hier geht es nur darum, was in dieser Ursprungsszene in diesem Moment gerade geschieht."
    ),
    "dark_scene_age": (
        "Hier geht es nur um das spontane Alter in dieser Ursprungsszene."
    ),
    "dark_scene_feeling_intensity": (
        "Hier geht es nur darum, wie das ungute Gefuehl dort wahrgenommen wird und wie stark es gerade ist."
    ),
    "dark_scene_immediate_feeling": (
        "Hier geht es nur darum, was in dieser Ursprungsszene ganz unmittelbar gefuehlt wird."
    ),
}
STRICT_PHASE4_YES_NO_NODES = {
    "group_image_ready",
    "group_person_ready",
    "group_next_person_check",
    "person_switch_ready",
    "person_switch_hears",
    "person_switch_sees_customer",
    "person_switch_sees_impact",
    "person_switch_heard_customer",
    "person_switch_aware_trigger",
    "person_switch_self_heard",
    "person_switch_self_understands",
}
STRICT_PHASE4_EXPLANATION_NODES = {
    "person_switch_why",
    "group_person_trigger_reason",
    "group_person_trigger_role",
    "group_person_trigger_core",
}
CONTEXTUAL_SAME_NODE_REPLY_NODES = set(STRICT_PHASE4_EXPLANATION_NODES)
CONTEXTUAL_SAME_NODE_REPLY_NODES.add("hell_light_level")
CONTEXTUAL_SAME_NODE_REPLY_NODES.add("origin_cause_owner")
CONTEXTUAL_SAME_NODE_REPLY_NODES.add("origin_trigger_known_branch")
CONTEXTUAL_SAME_NODE_REPLY_NODES.add("origin_scene_relevance")
CONTEXTUAL_SAME_NODE_REPLY_NODES.add("origin_person_name")
CONTEXTUAL_SAME_NODE_REPLY_NODES.add("dark_scene_people_who")
NAMED_PERSON_INPUT_NODES = {
    "origin_person_name",
    "group_representative_name",
    "group_specific_person_name",
    "group_multiple_people_name",
    "group_multiple_required_name",
    "group_next_person_name",
}
CONTEXTUAL_SAME_NODE_REPLY_NODES.update(NAMED_PERSON_INPUT_NODES)
CATEGORY_CHOICE_NODES: dict[str, str] = {
    "group_source_kind": "group_source_kind",
    "origin_other_target_kind": "origin_other_target_kind",
    "origin_cause_owner": "origin_cause_owner",
    "origin_scene_relevance": "origin_scene_relevance",
}


@dataclass(frozen=True)
class BatchCase:
    node_id: str
    customer_message: str
    expected_next_node: str


class LocalIntentRouter:
    def __init__(
        self,
        *,
        adapter_dir: Path,
        base_model: str | None = None,
        max_new_tokens: int = 32,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        self.base_model = base_model or str(adapter_config["base_model_name_or_path"])

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            local_files_only=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True)
        self.model.eval()

    def infer_intent(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized = self.tokenizer(rendered_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = tokenized["input_ids"].shape[1]
        completion_ids = generated[0][prompt_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


SCENARIOS: dict[str, list[BatchCase]] = {
    "phase1_anchor_core": [
        BatchCase("session_phase1_preflight_check", "ja passt alles", "session_phase1_setup_script"),
        BatchCase("session_phase1_anchor_after_setup", "ich glaube es passt jetzt", "session_phase1_mindset_script"),
        BatchCase("session_phase1_anchor_before_focus", "ja wir koennen weiter", "session_phase1_focus_script"),
    ],
    "phase2_core": [
        BatchCase("session_phase2_ready", "ja", "session_phase2_post_ready_script"),
        BatchCase("session_phase2_eyes_closed", "ja", "session_phase2_post_eyes_script"),
        BatchCase("session_phase2_scene_found", "ja", "session_phase2_post_scene_script"),
        BatchCase("session_phase2_feel_clear", "ja das ist spuerbar", "session_phase2_post_feel_script"),
        BatchCase("session_phase2_scale_clear", "ja", "session_phase2_scale_before"),
        BatchCase("session_phase2_scale_before", "8", "session_phase2_post_scale_before_script"),
        BatchCase("session_phase2_scale_after", "5", "session_phase2_post_scale_after_script"),
        BatchCase("session_phase2_continue_to_main", "ja", "session_phase2_end_script"),
    ],
    "common_step_core": [
        BatchCase("phase4_common_sees_younger_self", "ja ich sehe mein damaliges ich", "phase4_common_explain_to_younger"),
        BatchCase("phase4_common_understood", "ja es hat es verstanden", "phase4_common_first_cigarette"),
        BatchCase("phase4_common_feel_after_learning", "das fuehlt sich klar und traurig an", "phase4_common_first_drag"),
        BatchCase("phase4_common_feel_after_aversion", "es fuehlt sich eklig und deutlich an", "phase4_common_collect_moments"),
        BatchCase("phase4_common_done_signal", "ich bin wieder im magischen sessel", "session_phase5_future"),
    ],
    "phase4_to_session_end": [
        BatchCase("hell_feel_branch", "sehr angenehm", "hell_hypnose_loch_intro"),
        BatchCase("hell_hypnose_wait", "jetzt ist es aufgeloest", "hell_post_resolved_terminal"),
        BatchCase("dark_known_branch", "hier ist es zum ersten Mal", "dark_origin_terminal"),
        BatchCase("phase4_common_sees_younger_self", "ja", "phase4_common_explain_to_younger"),
        BatchCase("phase4_common_understood", "ja", "phase4_common_first_cigarette"),
        BatchCase("phase4_common_feel_after_learning", "es fuehlt sich klar an", "phase4_common_first_drag"),
        BatchCase("phase4_common_feel_after_aversion", "sehr unangenehm", "phase4_common_collect_moments"),
        BatchCase("phase4_common_done_signal", "ich bin wieder im sessel", "session_phase5_future"),
    ],
    "phase4_origin_group_path": [
        BatchCase("hell_light_level", "dunkel", "dark_scene_perception"),
        BatchCase("dark_scene_perception", "ich sehe eine gruppe kinder auf dem pausenhof", "dark_scene_people_who"),
        BatchCase("dark_scene_people_who", "hansi und seine clique", "dark_scene_age"),
        BatchCase("dark_scene_age", "12", "dark_scene_feeling_intensity"),
        BatchCase("dark_scene_feeling_intensity", "wie ein druck in der brust, sehr stark", "dark_known_branch"),
        BatchCase("dark_known_branch", "zum ersten mal", "dark_origin_terminal"),
        BatchCase("dark_scene_happening", "sie lachen mich aus weil ich nicht rauche", "origin_trigger_source"),
        BatchCase("origin_trigger_source", "die gruppe", "origin_trigger_known_branch"),
        BatchCase("origin_trigger_known_branch", "das kenne ich von frueher", "dark_backtrace_terminal"),
        BatchCase("origin_trigger_known_branch", "hier ist es neu", "origin_cause_owner"),
        BatchCase("origin_cause_owner", "jemand anderes", "group_branch_intro"),
        BatchCase("group_image_ready", "ja", "group_source_kind"),
        BatchCase("group_source_kind", "die ganze gruppe", "group_whole_scope"),
        BatchCase("group_whole_scope", "eine stellvertretende person reicht", "group_select_representative_intro"),
        BatchCase("group_representative_name", "peter", "group_bring_person_forward"),
        BatchCase("group_person_ready", "ja", "group_person_handoff"),
        BatchCase("group_person_trigger_reason", "er lacht mich aus", "group_person_trigger_role"),
        BatchCase("group_person_trigger_role", "er steht eher fuer die gruppe", "group_person_trigger_core"),
        BatchCase("group_person_trigger_core", "ich glaube er macht das um dazu zu gehoeren", "person_switch_ready_intro"),
        BatchCase("person_switch_ready", "ja", "person_switch_intro"),
    ],
    "phase4_origin_person_path": [
        BatchCase("hell_light_level", "dunkel", "dark_scene_perception"),
        BatchCase("dark_scene_perception", "ich sehe meinen vater", "dark_scene_people_who"),
        BatchCase("dark_scene_people_who", "mein vater", "dark_scene_age"),
        BatchCase("dark_scene_age", "10", "dark_scene_feeling_intensity"),
        BatchCase("dark_scene_feeling_intensity", "enge in der brust", "dark_known_branch"),
        BatchCase("dark_known_branch", "zum ersten mal", "dark_origin_terminal"),
        BatchCase("dark_scene_happening", "er schaut mich streng an", "origin_trigger_source"),
        BatchCase("origin_trigger_source", "mein vater", "origin_trigger_known_branch"),
        BatchCase("origin_trigger_known_branch", "das kenne ich schon", "dark_backtrace_terminal"),
        BatchCase("origin_scene_relevance", "das ist genau hier der kern", "origin_cause_owner"),
        BatchCase("origin_cause_owner", "jemand anderes", "origin_person_branch_intro"),
        BatchCase("group_person_ready", "ja", "group_person_handoff"),
        BatchCase("group_person_trigger_reason", "sein blick setzt mich unter druck", "group_person_trigger_role"),
        BatchCase("group_person_trigger_role", "direkt von ihm", "group_person_trigger_core"),
        BatchCase("group_person_trigger_core", "kontrolle", "person_switch_ready_intro"),
        BatchCase("person_switch_ready", "ja", "person_switch_intro"),
    ],
    "phase4_origin_other_path": [
        BatchCase("hell_light_level", "dunkel", "dark_scene_perception"),
        BatchCase("dark_scene_perception", "ich rieche rauch", "dark_scene_other_sense"),
        BatchCase("dark_scene_other_sense", "druck in der brust", "dark_scene_first_spuerbar"),
        BatchCase("dark_scene_first_spuerbar", "druck", "dark_scene_age"),
        BatchCase("dark_scene_age", "13", "dark_scene_feeling_intensity"),
        BatchCase("dark_scene_feeling_intensity", "ekel und sehr stark", "dark_known_branch"),
        BatchCase("dark_known_branch", "zum ersten mal", "dark_origin_terminal"),
        BatchCase("dark_scene_happening", "ich stehe dort und rieche den rauch", "origin_trigger_source"),
        BatchCase("origin_trigger_source", "der geruch", "origin_trigger_known_branch"),
        BatchCase("origin_trigger_known_branch", "das ist neu fuer mich", "origin_cause_owner"),
        BatchCase("origin_cause_owner", "jemand anderes", "origin_other_target_kind"),
        BatchCase("origin_other_target_kind", "etwas anderes", "origin_self_resolution_intro"),
        BatchCase("origin_self_need", "schutz und halt", "origin_self_release_intro"),
    ],
    "phase4_origin_relevance_backtrace": [
        BatchCase("dark_scene_happening", "mein vater schaut mich an", "origin_trigger_source"),
        BatchCase("origin_trigger_source", "mein vater", "origin_trigger_known_branch"),
        BatchCase("origin_trigger_known_branch", "das kenne ich schon", "dark_backtrace_terminal"),
        BatchCase("origin_scene_relevance", "das fuehrt mich noch weiter zurueck", "dark_backtrace_terminal"),
    ],
    "person_switch_resolution_path": [
        BatchCase("person_switch_hears", "ja", "person_switch_sees_customer"),
        BatchCase("person_switch_sees_customer", "ja", "person_switch_sees_impact"),
        BatchCase("person_switch_sees_impact", "ja", "person_switch_why"),
        BatchCase("person_switch_why", "ueberforderung", "person_switch_aware_trigger"),
        BatchCase("person_switch_aware_trigger", "ja", "person_switch_return_intro"),
        BatchCase("person_switch_self_heard", "ja", "person_switch_self_understands"),
        BatchCase("person_switch_self_understands", "ja", "group_resolution_complete"),
    ],
}


def load_env() -> None:
    for path in ENV_PATHS:
        if path.exists():
            load_dotenv(path, override=False)
            if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_key"):
                continue
            try:
                lines = path.read_text(encoding="utf-8-sig").splitlines()
            except OSError:
                continue
            for raw_line in lines:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip().lstrip("\ufeff")
                value = value.strip()
                if not key or key in os.environ:
                    continue
                os.environ[key] = value


def resolve_openai_client() -> tuple[OpenAI, str]:
    backend = resolve_ft_backend()
    return backend.client, backend.model


def resolve_local_intent_router(
    adapter_dir: str | None = None,
    *,
    base_model: str | None = None,
    max_new_tokens: int = 32,
) -> LocalIntentRouter:
    target = Path(adapter_dir).expanduser() if adapter_dir else DEFAULT_LOCAL_INTENT_ADAPTER_DIR
    if not target.exists():
        raise RuntimeError(f"Lokaler Intent-Adapter nicht gefunden: {target}")
    return LocalIntentRouter(
        adapter_dir=target,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"could not parse JSON from model response: {text}")


DecisionTraceLogger = Callable[[dict[str, Any]], None]


def _emit_decision_trace(
    trace_events: list[dict[str, Any]],
    trace_logger: DecisionTraceLogger | None,
    stage: str,
    **payload: Any,
) -> None:
    event = {"stage": stage, **payload}
    trace_events.append(event)
    if trace_logger is None:
        return
    try:
        trace_logger(event)
    except Exception:
        return


def _attach_decision_trace(parsed: dict[str, Any], trace_events: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(parsed)
    enriched["_trace"] = [dict(event) for event in trace_events]
    return enriched


def _build_stdout_trace_logger(enabled: bool) -> DecisionTraceLogger | None:
    if not enabled:
        return None

    def _logger(event: dict[str, Any]) -> None:
        print("[TRACE]")
        print(json.dumps(event, ensure_ascii=False, indent=2))
        print()

    return _logger


def _resolve_google_tts_client() -> Any:
    global _GOOGLE_TTS_CLIENT
    if _GOOGLE_TTS_CLIENT is not None:
        return _GOOGLE_TTS_CLIENT

    load_env()
    key_path = (os.getenv("GOOGLE_TTS_KEY") or "").strip()
    if not key_path:
        raise RuntimeError("GOOGLE_TTS_KEY fehlt.")

    try:
        from google.cloud import texttospeech
        from google.oauth2 import service_account
    except ModuleNotFoundError as exc:
        raise RuntimeError("google-cloud-texttospeech ist nicht installiert.") from exc

    credentials = service_account.Credentials.from_service_account_file(key_path)
    _GOOGLE_TTS_CLIENT = texttospeech.TextToSpeechClient(credentials=credentials)
    return _GOOGLE_TTS_CLIENT


def _tts_provider() -> str:
    normalized = (os.getenv("SESSION_SANDBOX_TTS_PROVIDER") or "google").strip().lower()
    if normalized in {"eleven", "elevenlabs"}:
        return "elevenlabs"
    return "google"


def _apply_tts_replacement_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _restore_german_umlauts_for_tts(text: str) -> str:
    normalized = str(text or "")
    if not normalized:
        return ""
    restored = normalized
    for pattern, replacement_root in _ELEVENLABS_GERMAN_TTS_RULES:
        restored = pattern.sub(
            lambda match: _apply_tts_replacement_case(
                match.group(0),
                f"{replacement_root}{match.group(1) if match.lastindex else ''}",
            ),
            restored,
        )
    return unicodedata.normalize("NFC", restored)


def _prepare_tts_text(text: str, *, provider: str | None = None) -> str:
    prepared = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not prepared:
        return ""
    if (provider or _tts_provider()).strip().lower() == "elevenlabs":
        return _restore_german_umlauts_for_tts(prepared)
    return prepared


def _elevenlabs_api_key() -> str:
    return (
        os.getenv("SESSION_SANDBOX_ELEVENLABS_API_KEY")
        or os.getenv("ELEVENLABS_API_KEY")
        or os.getenv("ELEVENLABS_API_key")
        or ""
    ).strip()


def _elevenlabs_voice_id() -> str:
    return (
        os.getenv("SESSION_SANDBOX_ELEVENLABS_VOICE_ID")
        or os.getenv("ELEVENLABS_TEST_VOICE_ID")
        or os.getenv("ELEVENLABS_VOICE_ID")
        or ""
    ).strip()


def _elevenlabs_fallback_voice_id() -> str:
    return (os.getenv("ELEVENLABS_FALLBACK_FREE_VOICE_ID") or "").strip()


def _elevenlabs_model_id() -> str:
    return (
        os.getenv("SESSION_SANDBOX_ELEVENLABS_MODEL_ID")
        or os.getenv("ELEVENLABS_MODEL_ID")
        or "eleven_multilingual_v2"
    ).strip() or "eleven_multilingual_v2"


def _elevenlabs_output_format() -> str:
    return (
        os.getenv("SESSION_SANDBOX_ELEVENLABS_OUTPUT_FORMAT")
        or os.getenv("ELEVENLABS_OUTPUT_FORMAT")
        or "pcm_22050"
    ).strip().lower() or "pcm_22050"


def _elevenlabs_voice_settings() -> dict[str, Any]:
    def _float_env(name: str, default: float) -> float:
        raw = (os.getenv(name) or "").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    def _bool_env(name: str, default: bool = False) -> bool:
        raw = (os.getenv(name) or "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    return {
        "stability": _float_env("ELEVENLABS_STABILITY", 0.35),
        "similarity_boost": _float_env("ELEVENLABS_SIMILARITY_BOOST", 0.85),
        "style": _float_env("ELEVENLABS_STYLE", 0.0),
        "use_speaker_boost": _bool_env("ELEVENLABS_SPEAKER_BOOST", False),
    }


def _ensure_elevenlabs_tts_ready() -> None:
    load_env()
    if not _elevenlabs_api_key():
        raise RuntimeError("ELEVENLABS_API_KEY fehlt.")
    if not _elevenlabs_voice_id():
        raise RuntimeError("ELEVENLABS_VOICE_ID fehlt.")
    if not _elevenlabs_output_format().startswith("pcm_"):
        raise RuntimeError("SESSION_SANDBOX unterstuetzt fuer ElevenLabs nur PCM-Ausgabe.")


def _ensure_tts_ready() -> bool:
    global _TTS_READY, _TTS_ERROR
    if _TTS_READY is not None:
        return _TTS_READY
    try:
        if _tts_provider() == "elevenlabs":
            try:
                _ensure_elevenlabs_tts_ready()
            except Exception:
                _resolve_google_tts_client()
        else:
            _resolve_google_tts_client()
    except Exception as exc:
        _TTS_READY = False
        _TTS_ERROR = str(exc)
        return False
    _TTS_READY = True
    _TTS_ERROR = None
    return True


def _tts_voice_name() -> str:
    return (os.getenv("SESSION_SANDBOX_TTS_VOICE_NAME") or "de-DE-Neural2-B").strip() or "de-DE-Neural2-B"


def _tts_speaking_rate() -> float:
    raw = (os.getenv("SESSION_SANDBOX_TTS_SPEAKING_RATE") or "0.92").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.92
    return max(0.75, min(1.15, value))


def _tts_lead_in_ms() -> int:
    raw = (os.getenv("SESSION_SANDBOX_TTS_LEAD_IN_MS") or "180").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 180
    return max(0, min(500, value))


def _tts_post_block_pause_ms() -> int:
    raw = (os.getenv("SESSION_SANDBOX_TTS_POST_BLOCK_PAUSE_MS") or "320").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 320
    return max(0, min(1500, value))


def _chunk_text_for_tts(text: str, max_chars: int = 1800) -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _elevenlabs_sample_rate() -> int:
    output_format = _elevenlabs_output_format()
    if not output_format.startswith("pcm_"):
        raise RuntimeError("SESSION_SANDBOX unterstuetzt fuer ElevenLabs nur PCM-Ausgabe.")
    try:
        value = int(output_format.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Ungueltiges ElevenLabs-Output-Format: {output_format}") from exc
    return max(16000, min(value, 48000))


def _prepend_silence_to_pcm(
    pcm_audio: bytes,
    *,
    sample_rate: int,
    lead_in_ms: int,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    if lead_in_ms <= 0:
        return pcm_audio
    silence_frames = int(sample_rate * (lead_in_ms / 1000.0))
    silence_prefix = b"\x00" * max(0, silence_frames * sample_width * channels)
    return silence_prefix + pcm_audio


def _prepend_silence_to_wav_bytes(wav_bytes: bytes, *, lead_in_ms: int) -> bytes:
    if lead_in_ms <= 0:
        return wav_bytes
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_reader:
        params = wav_reader.getparams()
        frames = wav_reader.readframes(wav_reader.getnframes())
    prefixed_frames = _prepend_silence_to_pcm(
        frames,
        sample_rate=int(params.framerate),
        lead_in_ms=lead_in_ms,
        sample_width=int(params.sampwidth),
        channels=int(params.nchannels),
    )
    output_buffer = io.BytesIO()
    with wave.open(output_buffer, "wb") as wav_writer:
        wav_writer.setnchannels(int(params.nchannels))
        wav_writer.setsampwidth(int(params.sampwidth))
        wav_writer.setframerate(int(params.framerate))
        wav_writer.writeframes(prefixed_frames)
    return output_buffer.getvalue()


def _synthesize_google_wav_to_cache(text: str, *, lead_in: bool = False) -> Path:
    load_env()
    from google.cloud import texttospeech

    prepared = _prepare_tts_text(text, provider="google")
    lead_in_ms = _tts_lead_in_ms() if lead_in else 0
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(
        f"google|{_tts_voice_name()}|{_tts_speaking_rate()}|lead_in={lead_in_ms}|{prepared}".encode("utf-8")
    ).hexdigest()
    wav_path = TTS_CACHE_DIR / f"{cache_key}.wav"
    if wav_path.exists():
        return wav_path

    client = _resolve_google_tts_client()
    synthesis_input = texttospeech.SynthesisInput(text=prepared)
    voice = texttospeech.VoiceSelectionParams(
        language_code="de-DE",
        name=_tts_voice_name(),
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=_tts_speaking_rate(),
    )
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    audio = response.audio_content or b""
    if not audio:
        raise RuntimeError("Leere Google-TTS-Antwort.")
    audio = _prepend_silence_to_wav_bytes(audio, lead_in_ms=lead_in_ms)
    wav_path.write_bytes(audio)
    return wav_path


def _synthesize_elevenlabs_wav_to_cache(text: str, *, lead_in: bool = False) -> Path:
    load_env()
    prepared = _prepare_tts_text(text, provider="elevenlabs")
    if not prepared:
        raise RuntimeError("Leerer ElevenLabs-Text.")

    voice_id = _elevenlabs_voice_id()
    fallback_voice_id = _elevenlabs_fallback_voice_id()
    output_format = _elevenlabs_output_format()
    sample_rate = _elevenlabs_sample_rate()
    model_id = _elevenlabs_model_id()
    voice_settings = _elevenlabs_voice_settings()
    lead_in_ms = _tts_lead_in_ms() if lead_in else 0

    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "provider": "elevenlabs",
                "voice_id": voice_id,
                "model_id": model_id,
                "output_format": output_format,
                "voice_settings": voice_settings,
                "lead_in_ms": lead_in_ms,
                "text": prepared,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    wav_path = TTS_CACHE_DIR / f"{cache_key}.wav"
    if wav_path.exists():
        return wav_path

    payload: dict[str, Any] = {
        "text": prepared,
        "model_id": model_id,
        "language_code": "de",
        "voice_settings": voice_settings,
    }
    timeout_seconds = 40.0
    timeout_raw = (os.getenv("ELEVENLABS_TIMEOUT_SECONDS") or "").strip()
    try:
        if timeout_raw:
            timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 40.0

    def _request_pcm(target_voice_id: str) -> bytes:
        voice_path = urllib.parse.quote(target_voice_id, safe="")
        query = urllib.parse.urlencode({"output_format": output_format})
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_path}?{query}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
                "xi-api-key": _elevenlabs_api_key(),
            },
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            audio_bytes = response.read()
        if not audio_bytes:
            raise RuntimeError("Leere ElevenLabs-Antwort.")
        return audio_bytes

    try:
        pcm_audio = _request_pcm(voice_id)
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body_text = ""
        paid_plan_restricted = (
            exc.code in {401, 402}
            and fallback_voice_id
            and fallback_voice_id != voice_id
            and any(token in body_text for token in ("paid_plan_required", "subscription_required", "library voices"))
        )
        if not paid_plan_restricted:
            detail = body_text[:240].strip()
            if detail:
                raise RuntimeError(f"ElevenLabs TTS fehlgeschlagen ({exc.code}): {detail}") from exc
            raise RuntimeError(f"ElevenLabs TTS fehlgeschlagen ({exc.code}).") from exc
        pcm_audio = _request_pcm(fallback_voice_id)
    except Exception as exc:
        raise RuntimeError(f"ElevenLabs TTS Fehler: {exc}") from exc

    pcm_audio = _prepend_silence_to_pcm(
        pcm_audio,
        sample_rate=sample_rate,
        lead_in_ms=lead_in_ms,
    )
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio)
    return wav_path


def _speak_text(text: str) -> None:
    global _TTS_READY, _TTS_ERROR
    if not _ensure_tts_ready():
        return
    try:
        provider = _tts_provider()
        elevenlabs_failed = False
        for index, chunk in enumerate(_chunk_text_for_tts(text)):
            lead_in = index == 0
            if provider == "elevenlabs" and not elevenlabs_failed:
                try:
                    wav_path = _synthesize_elevenlabs_wav_to_cache(chunk, lead_in=lead_in)
                except Exception as exc:
                    elevenlabs_failed = True
                    print(f"[ElevenLabs Fallback] {exc}")
                    wav_path = _synthesize_google_wav_to_cache(chunk, lead_in=lead_in)
            else:
                wav_path = _synthesize_google_wav_to_cache(chunk, lead_in=lead_in)
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
    except Exception as exc:
        _TTS_READY = False
        _TTS_ERROR = str(exc)
        print(f"[TTS deaktiviert] {_TTS_ERROR}")


def _terminal_width() -> int:
    return max(60, shutil.get_terminal_size(fallback=(100, 30)).columns - 2)


def _terminal_height() -> int:
    return max(20, shutil.get_terminal_size(fallback=(100, 30)).lines)


def _format_display_text(text: str) -> list[str]:
    width = _terminal_width()
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return [""]

    lines: list[str] = []
    for paragraph in normalized.split("\n\n"):
        raw_lines = paragraph.split("\n")
        paragraph_text = " ".join(line.strip() for line in raw_lines if line.strip())
        if not paragraph_text:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            paragraph_text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return lines or [""]


def _print_paged_block(
    title: str,
    text: str,
    *,
    speak: bool = False,
    post_block_pause_ms: int = 0,
) -> None:
    print(f"\n[{title}]")
    lines = _format_display_text(text)
    if speak:
        for line in lines:
            print(line)
        print()
        _speak_text(text)
        if post_block_pause_ms > 0:
            time.sleep(post_block_pause_ms / 1000.0)
        return

    page_size = max(8, _terminal_height() - 6)
    for start in range(0, len(lines), page_size):
        page = lines[start : start + page_size]
        for line in page:
            print(line)
        if start + page_size < len(lines):
            input("\n[Mehr] Enter fuer weiter...")
            print()
    print()


def _decision_for_empty_input(node_id: str) -> SemanticModelDecision:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise ValueError(f"node '{node_id}' is not semantic")

    if "unclear" in spec.routing_rules:
        route = spec.routing_rules["unclear"]
        payload = {
            "intent": "unclear",
            "action": route["action"],
            "next_node": route["next_node"],
            "confidence": 1.0,
            "reason": "Leere Eingabe: Der Kunde hat noch keine inhaltliche Antwort gegeben.",
        }
        return validate_semantic_decision(node_id, payload)

    if "repeat" in spec.routing_rules:
        route = spec.routing_rules["repeat"]
        payload = {
            "intent": "repeat",
            "action": route["action"],
            "next_node": route["next_node"],
            "confidence": 1.0,
            "reason": "Leere Eingabe: Die Frage soll noch einmal gestellt werden.",
        }
        return validate_semantic_decision(node_id, payload)

    raise ValueError(f"node '{node_id}' has no empty-input fallback")


def _empty_input_reply(node_id: str, attempt: int, runtime_slots: dict[str, str] | None = None) -> str:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        return ""

    attempt = max(0, attempt)
    runtime_slots = runtime_slots or {}
    max_attempts_before_outro = _max_silence_attempts_before_outro(node_id)

    if attempt >= max_attempts_before_outro:
        return INACTIVITY_OUTRO_TEXT

    if attempt == max_attempts_before_outro - 1:
        return INACTIVITY_FINAL_WARNING_TEXT

    if attempt == max_attempts_before_outro - 2:
        return INACTIVITY_WARNING_TEXT

    if attempt == max_attempts_before_outro - 3:
        return _diagnostic_empty_input_reply(node_id, spec, runtime_slots)

    node_specific_replies = NODE_EMPTY_INPUT_REPLIES.get(node_id)
    if node_specific_replies:
        return _render_runtime_text(
            node_specific_replies[min(attempt, len(node_specific_replies) - 1)],
            runtime_slots,
        )

    allowed = set(spec.allowed_intents)
    question_text = spec.question_text.strip().rstrip("?")

    if attempt == 0:
        rephrased = _first_empty_input_rephrase(node_id, spec)
        if rephrased:
            return rephrased

    answer_hint = _empty_input_answer_hint(node_id, spec, runtime_slots)

    if "provided_scale" in allowed:
        replies = [
            "Es reicht deine erste Einordnung. Welche Zahl passt gerade am besten zu dem, was du jetzt wahrnimmst?",
            "Du musst es noch nicht weiter erklaeren. Eine Zahl zwischen 1 und 10 reicht voellig.",
            "Nimm nur den ersten Impuls. Welche Zahl zwischen 1 und 10 passt gerade am ehesten?",
            "Sobald du die Intensitaet greifen kannst, genuegt eine einzige Zahl.",
        ]
    elif answer_hint:
        replies = [
            f"Es reicht deine erste Einordnung. {answer_hint}",
            f"Du musst es noch nicht weiter erklaeren. {answer_hint}",
            f"Nimm nur den ersten Impuls. {answer_hint}",
            "Sobald du es greifen kannst, genuegt eine kurze Rueckmeldung dazu.",
        ]
    elif "continue" in allowed:
        replies = [
            "Wenn es fuer dich passt, genuegt ein kurzes Ja. Wenn noch etwas offen ist, kannst du es kurz dazusagen.",
            "Du musst es noch nicht weiter erklaeren. Ein kurzes Ja reicht, wenn wir weitergehen koennen.",
            "Nimm nur den ersten Impuls. Wenn es fuer dich stimmig ist, genuegt ein Ja. Wenn noch etwas offen ist, kannst du es kurz dazusagen.",
            "Sobald es fuer dich passt, genuegt ein kurzes Ja.",
        ]
    elif "ready" in allowed:
        replies = [
            "Dein erster Eindruck dazu genuegt. Eine kurze Rueckmeldung in deinen eigenen Worten reicht voellig.",
            "Du musst es noch nicht perfekt formulieren. Lass es einfach so auftauchen, wie es dir zuerst kommt.",
            "Nimm nur wahr, was dort zuerst fuer dich wesentlich ist, und benenne genau das.",
            "Sobald du es greifen kannst, genuegt eine kurze Rueckmeldung dazu.",
        ]
    else:
        replies = [
            "Dein erster Eindruck dazu genuegt. Eine kurze Rueckmeldung reicht voellig.",
            "Du musst es noch nicht weiter erklaeren. Lass es einfach so auftauchen, wie es dir zuerst kommt.",
            "Nimm nur wahr, was dort zuerst fuer dich wichtig ist, und benenne genau das.",
            "Sobald du es greifen kannst, genuegt eine kurze Rueckmeldung dazu.",
        ]

    return replies[min(attempt - 1, len(replies) - 1)]


def _empty_input_answer_hint(
    node_id: str,
    spec: SemanticNodeSpec,
    runtime_slots: dict[str, str],
) -> str:
    named_person = runtime_slots.get("named_person", "diese Person")
    named_person_ref = _display_named_person_reference_for_runtime(named_person)
    trigger_focus_ref = _reflect_focus_ref_for_therapist(runtime_slots.get("trigger_focus_ref", "dem benannten Ausloeser"))
    hints = {
        "hell_light_level": "Eine kurze Einordnung wie hell, dunkel oder gemischt reicht voellig.",
        "scene_access_followup": "Eine kurze Einordnung reicht: hoerbar, riechbar, schmeckbar, spuerbar oder noch nichts.",
        "dark_scene_mode_clarify": "Eine kurze Einordnung reicht: eher sehen, eher hoeren, beides oder nichts davon klar.",
        "dark_known_branch": "Ein kurzes bekannt oder zum ersten Mal reicht voellig.",
        "origin_trigger_known_branch": "Ein kurzes schon bekannt oder hier neu reicht voellig.",
        "hell_regulation_choice": "Eine kurze Einordnung reicht: mehr Abstand, weniger Helligkeit oder klarerer Fokus.",
        "hell_regulation_check": "Eine kurze Einordnung reicht: eher hell, eher dunkel, beides oder ruhiger.",
        "origin_trigger_source": "Eine kurze Einordnung reicht, was oder wer dort am staerksten auf dieses Gefuehl wirkt.",
        "origin_scene_relevance": "Eine kurze Einordnung reicht: eher hier bearbeiten oder eher noch weiter zurueck.",
        "origin_cause_owner": f"Eine kurze Einordnung reicht: eher etwas in dir selbst oder eher {trigger_focus_ref}.",
        "origin_other_target_kind": "Eine kurze Einordnung reicht: eher Gruppe, eher bestimmte Person oder eher etwas anderes in der Situation.",
        "origin_person_name": "Nenn einfach den Namen oder eine kurze Beschreibung der Person.",
        "group_image_ready": "Ein kurzes Ja oder Nein reicht voellig.",
        "group_source_kind": "Eine kurze Einordnung reicht: ganze Gruppe, eine bestimmte Person oder mehrere einzelne Personen.",
        "group_whole_scope": "Eine kurze Einordnung reicht: eine stellvertretende Person reicht oder mehrere Personen sind noetig.",
        "group_person_ready": "Ein kurzes Ja oder Nein reicht voellig.",
        "group_next_person_check": "Ein kurzes Ja oder Nein reicht voellig.",
        "person_switch_ready": f"Ein kurzes Ja oder Nein reicht voellig. Es geht nur darum, ob der Wechsel in die Perspektive von {named_person_ref} jetzt passt.",
        "person_switch_hears": "Ein kurzes Ja oder Nein reicht voellig.",
        "person_switch_sees_customer": "Ein kurzes Ja oder Nein reicht voellig.",
        "person_switch_sees_impact": "Ein kurzes Ja oder Nein reicht voellig.",
        "person_switch_heard_customer": "Ein kurzes Ja oder Nein reicht voellig.",
        "person_switch_aware_trigger": "Ein kurzes Ja oder Nein reicht voellig.",
        "person_switch_self_heard": "Ein kurzes Ja oder Nein reicht voellig.",
        "person_switch_self_understands": "Ein kurzes Ja oder Nein reicht voellig.",
        "dark_scene_audio_detail": "Beschreibe einfach kurz, was dort genau hoerbar wird.",
        "dark_scene_other_sense": "Beschreibe einfach kurz, was dort ueber Koerper, Geruch, Geschmack oder Temperatur wahrnehmbar wird.",
        "dark_scene_first_spuerbar": "Beschreibe einfach kurz, was dort als Erstes am deutlichsten spuerbar wird.",
        "dark_scene_people_who": "Beschreibe einfach kurz, wer oder was dort fuer dich genauer erkennbar wird.",
    }
    if node_id in hints:
        return hints[node_id]
    if "yes" in spec.allowed_intents and "no" in spec.allowed_intents:
        return "Ein kurzes Ja oder Nein reicht voellig."
    return ""


def _diagnostic_empty_input_reply(
    node_id: str,
    spec: SemanticNodeSpec,
    runtime_slots: dict[str, str],
) -> str:
    named_person = runtime_slots.get("named_person", "diese Person")
    named_person_ref = _display_named_person_reference_for_runtime(named_person)
    trigger_focus_ref = _reflect_focus_ref_for_therapist(runtime_slots.get("trigger_focus_ref", "diesem Ausloeser"))

    scene_access_nodes = {
        "hell_light_level",
        "scene_access_followup",
        "dark_scene_perception",
        "dark_scene_mode_clarify",
        "dark_scene_who",
        "dark_scene_audio_detail",
        "dark_scene_other_sense",
        "dark_scene_first_spuerbar",
        "dark_scene_people_who",
        "dark_scene_happening",
        "dark_scene_age",
        "dark_scene_feeling_intensity",
        "dark_scene_immediate_feeling",
    }
    if node_id in scene_access_nodes:
        return (
            "Wenn dort gerade noch nichts richtig greifbar ist, ist das in Ordnung. "
            "Bleib einfach noch einen Moment dabei und nimm wahr, "
            "ob sich eher ueber den Koerper, einen Geruch, einen Geschmack oder eine Temperatur etwas zeigt."
        )

    if node_id == "origin_cause_owner":
        return (
            f"Ich brauche jetzt nur noch eine kurze klare Einordnung: "
            f"Liegt der eigentliche Grund eher in dir selbst, oder geht er eher von {trigger_focus_ref} aus?"
        )

    if node_id == "group_person_trigger_role":
        return (
            f"Ich brauche jetzt nur noch eine kurze klare Einordnung: "
            f"Geht das eher direkt von {named_person} aus oder steht {named_person} eher fuer eine groessere Dynamik in dieser Situation?"
        )

    if node_id == "person_switch_ready":
        return (
            f"Ich brauche jetzt nur noch eine kurze Rueckmeldung: "
            f"Passt der Perspektivwechsel in die Perspektive von {named_person_ref} jetzt oder noch nicht?"
        )

    if "yes" in spec.allowed_intents and "no" in spec.allowed_intents:
        return "Ich brauche jetzt eine kurze klare Rueckmeldung von dir. Ein einfaches Ja oder Nein reicht voellig."

    answer_hint = _empty_input_answer_hint(node_id, spec, runtime_slots)
    if answer_hint:
        return f"Ich brauche jetzt nur noch eine kurze klare Rueckmeldung. {answer_hint}"

    return (
        "Ich brauche jetzt nur noch eine kurze Rueckmeldung von dir. "
        "Wenn die Frage noch unklar ist oder du noch einen Moment brauchst, gib dazu einfach eine kurze Rueckmeldung."
    )


def _normalize_user_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _first_empty_input_rephrase(node_id: str, spec: SemanticNodeSpec) -> str:
    question_text = spec.question_text.strip().rstrip("?")
    normalized_question = _normalize_user_text(question_text)
    allowed = set(spec.allowed_intents)

    if node_id == "session_phase2_scene_found":
        return (
            "Ich frage dich noch einmal etwas klarer: Hast du jetzt eine konkrete Situation gefunden, "
            "in der das Verlangen nach der Zigarette oder der Suchtdruck deutlich spuerbar war?"
        )

    if node_id == "session_phase2_feel_clear":
        return (
            "Ich frage dich noch einmal etwas klarer: Ist dieses Verlangen oder dieses ungute Gefuehl in der gewaehlten Situation jetzt deutlich genug spuerbar?"
        )

    if "provided_scale" in allowed:
        return "Ich frage dich noch einmal etwas klarer: Welche Zahl zwischen 1 und 10 passt gerade am ehesten zu dem, was du jetzt wahrnimmst?"

    if node_id == "hell_feel_branch":
        return "Ich frage dich noch einmal etwas klarer: Fuehlt sich dieses Helle fuer dich eher angenehm oder eher unangenehm an?"

    if node_id == "hell_hypnose_wait":
        return "Ich frage dich noch einmal etwas klarer: Loest es sich noch auf, brauchst du noch einen Moment oder ist es bereits aufgeloest?"

    if node_id == "origin_scene_relevance":
        return (
            "Ich frage dich noch einmal etwas klarer: Sollen wir das, was dort gerade stark wirkt, "
            "genau hier in dieser Szene bearbeiten, oder fuehrt dich diese Szene eher noch weiter zurueck?"
        )

    if node_id == "dark_known_branch":
        return "Ich frage dich noch einmal etwas klarer: Kommt dir dieses Gefuehl bereits bekannt vor, oder begegnest du ihm hier zum ersten Mal?"

    if node_id == "origin_trigger_known_branch":
        return (
            "Ich frage dich noch einmal etwas klarer: Kennst du dieses Gefuehl bereits aus frueheren Momenten, "
            "oder begegnet es dir hier in dieser Szene zum ersten Mal?"
        )

    if node_id == "scene_access_followup":
        return (
            "Ich frage dich noch einmal etwas klarer: Kannst du etwas hoeren, riechen, schmecken "
            "oder ein bestimmtes Gefuehl wahrnehmen, oder ist noch nichts greifbar?"
        )

    if node_id == "hell_regulation_choice":
        return "Ich frage dich noch einmal etwas klarer: Was hilft dir gerade am meisten, mehr Abstand, weniger Helligkeit oder ein klarerer Fokus?"

    if node_id == "hell_regulation_check":
        return "Ich frage dich noch einmal etwas klarer: Wirkt die Szene jetzt eher hell, eher dunkel, beides oder deutlich ruhiger?"

    if node_id == "group_source_kind":
        return "Ich frage dich noch einmal etwas klarer: Kommt dieses Gefuehl fuer dich eher von der ganzen Gruppe, nur von einer bestimmten Person oder von mehreren einzelnen Personen?"

    if node_id == "group_whole_scope":
        return "Ich frage dich noch einmal etwas klarer: Reicht hier eine stellvertretende Person aus dieser Gruppe oder muessen wir mehrere Personen einzeln einbeziehen?"

    if node_id == "group_next_person_check":
        return "Ich frage dich noch einmal etwas klarer: Gibt es in dieser Gruppe jetzt noch eine weitere Person, die wir ebenfalls noch loesen sollten? Ein einfaches Ja oder Nein reicht."

    if "yes" in allowed:
        if "augen" in normalized_question and "geschlossen" in normalized_question:
            return "Ich frage dich noch einmal etwas klarer: Sind deine Augen jetzt geschlossen? Ein einfaches Ja oder Nein reicht."
        return f"Ich frage dich noch einmal etwas klarer: {question_text}? Ein einfaches Ja oder Nein reicht."

    if "continue" in allowed:
        return f"Ich frage dich noch einmal etwas klarer: {question_text}? Wenn es fuer dich passt, genuegt ein kurzes Ja. Wenn noch etwas offen ist, kannst du es kurz dazusagen."

    if "ready" in allowed:
        return f"Ich frage dich noch einmal etwas klarer: {question_text}?"

    return f"Ich frage dich noch einmal etwas klarer: {question_text}?"


def _contains_any(text: str, patterns: list[str]) -> bool:
    padded_text = f" {text} "
    return any(f" {pattern} " in padded_text for pattern in patterns)


def _matches_any_regex(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


GLOBAL_ABORT_REGEXES: tuple[str, ...] = (
    r"\b(?:abbrechen|stopp|stop|beenden|ende)\b",
    r"\bich (?:mochte|will) aufhoren\b",
)
GLOBAL_CONSENT_UNCLEAR_REGEXES: tuple[str, ...] = (
    r"\bich weiss nicht ob ich (?:das|weitermachen) will\b",
    r"\bich bin nicht sicher ob ich (?:das|weitermachen) will\b",
    r"\bnicht sicher ob ich (?:das|weitermachen) will\b",
    r"\beigentlich habe ich keine lust mehr\b",
    r"\beigentlich will ich nicht mehr\b",
    r"\bich will eigentlich nicht mehr\b",
    r"\bich mochte das eigentlich nicht\b",
)
GLOBAL_FATIGUE_REGEXES: tuple[str, ...] = (
    r"\bich schlafe fast ein\b",
    r"\bich schlafe ein\b",
    r"\bich drifte weg\b",
    r"\bich werde muede\b",
    r"\bich bin muede\b",
    r"\bich bin so muede\b",
    r"\bich bin fast weg\b",
    r"\bich bin kaum noch da\b",
)
GLOBAL_SUPPORT_REGEXES: tuple[str, ...] = (
    r"\bnoch einen moment\b",
    r"\beinen moment\b",
    r"\beinen augenblick\b",
    r"\bwarte\b",
    r"\bich brauche kurz\b",
    r"\bich brauche noch\b",
    r"\bich brauch kurz\b",
    r"\bich brauch noch\b",
    r"\bes ist mir zu viel\b",
    r"\bmir ist es zu viel\b",
    r"\bich bin uberfordert\b",
    r"\bich bin ueberfordert\b",
    r"\bich kann gerade nicht\b",
    r"\bich weiss nicht weiter\b",
    r"\bich weiss gerade nicht weiter\b",
    r"\bnoch nicht soweit\b",
)
GLOBAL_REENGAGEMENT_REGEXES: tuple[str, ...] = (
    r"\bich bin da\b",
    r"\bich bin noch da\b",
    r"\bich bin wieder da\b",
    r"\bbin wieder da\b",
    r"\bbin da\b",
    r"\bich bin hier\b",
    r"\bich bin wieder hier\b",
    r"\bich hoere dich\b",
    r"\bich hoer dich\b",
)
ACUTE_AFFECT_DISTRESS_REGEXES: tuple[str, ...] = (
    r"\bich kann fast nicht atmen\b",
    r"\bich kann kaum atmen\b",
    r"\bich bekomme kaum luft\b",
    r"\bich kriege kaum luft\b",
    r"\bmir bleibt die luft weg\b",
    r"\bes schnuert mir die luft ab\b",
    r"\bes schnuert mir die luft ab\b",
    r"\bpanik\b",
    r"\bpanisch\b",
)
GLOBAL_UNCERTAINTY_REGEXES: tuple[str, ...] = (
    r"\bweiss nicht\b",
    r"\bweiss es nicht\b",
    r"\bich weiss nicht\b",
    r"\bich weiss es nicht\b",
    r"\bweiß nicht\b",
    r"\bich weiß nicht\b",
    r"\bkeine ahnung\b",
    r"\bich habe keine ahnung\b",
    r"\bhab keine ahnung\b",
    r"\bnicht sicher\b",
    r"\bbin mir nicht sicher\b",
    r"\bich bin mir nicht sicher\b",
    r"\bunsicher\b",
    r"\bschwer zu sagen\b",
    r"\bkann ich nicht sagen\b",
)
GLOBAL_QUESTION_REGEXES: tuple[str, ...] = (
    r"\bverstehe ich nicht\b",
    r"\bich verstehe nicht\b",
    r"\bversteh ich nicht\b",
    r"\bdu verstehst mich\b",
    r"\bverstehst du mich\b",
    r"\bdu hast mich nicht verstanden\b",
    r"\bdas war nicht meine antwort\b",
    r"\bunklar\b",
    r"\bwas meinst\b",
    r"\bwie meinst\b",
    r"\bwas soll\b",
    r"\bwie soll\b",
    r"\bwas genau\b",
    r"\bwie genau\b",
    r"\bkannst du (?:das )?erklaren\b",
    r"\bkannst du (?:das )?erklaeren\b",
    r"\berklar mal\b",
    r"\berklaer mal\b",
    r"\bnoch ?mal erklaren\b",
    r"\bnoch ?mal erklaeren\b",
    r"\bwas heisst\b",
    r"\bheisst das\b",
    r"\b(?:warum|wieso|weshalb)\b.*\b(?:stoppt|haengt|hakt)\b",
    r"\b(?:stoppt|haengt|hakt)\b.*\b(?:hier|da|jetzt)\b",
    r"\bgeht es nicht weiter\b",
    r"\bwarum passiert nichts\b",
)
GLOBAL_HOSTILE_REGEXES: tuple[str, ...] = (
    r"\bidiot\b",
    r"\barschloch\b",
    r"\bwixx\w*\b",
    r"\bhurensohn\b",
    r"\bdu bist krank\b",
    r"\bverfick\w*\b",
    r"\bscheiss\w*\b",
)
VISUAL_NEGATIVE_ONLY_REGEXES: tuple[str, ...] = (
    r"\bich sehe nichts\b",
    r"\bsehe nichts\b",
    r"\bseh nichts\b",
    r"\bkein bild\b",
    r"\bnichts sichtbar\b",
    r"\bich sehe niemand(?:en)?\b",
)
FULL_NOTHING_REGEXES: tuple[str, ...] = (
    r"\bich nehme gar nichts wahr\b",
    r"\bich nehme nichts wahr\b",
    r"\bda ist gar nichts\b",
    r"\bda ist nichts\b",
    r"\bgar nichts\b",
    r"\bnoch nichts\b",
)
FEELING_NOT_PERCEIVABLE_REGEXES: tuple[str, ...] = (
    r"\bich nehme es gerade nicht mehr wahr\b",
    r"\bich nehm es gerade nicht mehr wahr\b",
    r"\bich nehme es nicht mehr wahr\b",
    r"\bich nehm es nicht mehr wahr\b",
    r"\bgerade nicht mehr spuerbar\b",
    r"\bgerade nicht mehr spurbar\b",
    r"\bnicht mehr spuerbar\b",
    r"\bnicht mehr spurbar\b",
    r"\bkaum noch spuerbar\b",
    r"\bgerade nicht klar spuerbar\b",
    r"\bich spuere es gerade nicht\b",
    r"\bich spure es gerade nicht\b",
)
EXPLICIT_DARK_REGEXES: tuple[str, ...] = (
    r"\bdunkel\b",
    r"\beher ?dunkel\b",
    r"\bist ?dunkel\b",
    r"\bes ist ?dunkel\b",
    r"\bschattig\b",
    r"\bnicht hell\b",
)
EXPLICIT_HELL_REGEXES: tuple[str, ...] = (
    r"\bhell\b",
    r"\bsehr hell\b",
    r"\beher ?hell\b",
    r"\bist ?hell\b",
    r"\bes ist ?hell\b",
    r"\blichtvoll\b",
    r"\bklar hell\b",
)
EXPLICIT_MIXED_LIGHT_REGEXES: tuple[str, ...] = (
    r"\bbeides\b",
    r"\bhell und dunkel\b",
    r"\bsowohl hell als auch dunkel\b",
)


def _is_question_announcement(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    generic_announcements = {
        "frage",
        "eine frage",
        "ich habe eine frage",
        "ich hab eine frage",
        "ich habe da eine frage",
        "ich hab da eine frage",
        "ich habe noch eine frage",
        "ich hab noch eine frage",
        "ich mochte etwas fragen",
        "ich will etwas fragen",
    }
    return normalized in generic_announcements


def _looks_like_nonanswer_noise(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False

    if re.fullmatch(r"[?!.,;:()\[\]{}<>'\"`~*+\-_/\\|=]+", raw):
        return True

    normalized = _normalize_user_text(raw)
    if not normalized:
        return True

    exact_fillers = {
        "hm",
        "hmm",
        "hmmm",
        "mhm",
        "aha",
        "naja",
        "tja",
        "kp",
        "ka",
        "asdf",
        "qwertz",
        "qwerty",
        "abc123",
        "123abc",
    }
    if normalized in exact_fillers:
        return True

    if re.fullmatch(r"[a-z]", normalized):
        return True

    if re.fullmatch(r"(?:bla|blub|la|ha|hm){2,}", normalized):
        return True

    if re.fullmatch(r"ich bin (?:ein|eine) [a-z]+", normalized) and not _looks_like_age_reply(text):
        return True

    return False


def _looks_like_direct_question(text: str) -> bool:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return False

    if raw.endswith("?"):
        return True

    normalized = _normalize_user_text(raw)
    if not normalized:
        return False

    question_starts = (
        "was ",
        "wie ",
        "warum ",
        "wieso ",
        "weshalb ",
        "wo ",
        "wohin ",
        "woher ",
        "wann ",
        "wer ",
        "mit was ",
        "womit ",
    )
    return normalized.startswith(question_starts)


def _looks_like_same_scene_report(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"\b(?:am|im)\s+(?:gleichen|selben)\s+(?:ort|platz|punkt|moment|stelle)\b",
            r"\b(?:die|diese)\s+gleiche\s+(?:szene|situation)\b",
            r"\bwie\s+(?:vorher|vorhin|zuvor)\b",
            r"\bwieder\s+(?:am|im)\s+(?:gleichen|selben)\b",
            r"\b(?:gleichen|selben)\s+ort\b",
        )
    )


def _is_explicit_person_visibility_ready_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"\b(?:ich\s+)?(?:seh|sehe|erkenne|kenne|weiss)\b.*\b"
            r"(?:ihn|sie|die\s+person|diese\s+person|wer\s+(?:es|sie|die\s+person|diese\s+person)\s+ist)\b",
            r"\b(?:er|sie|die\s+person|diese\s+person)\b.*\b(?:steht|ist)\b.*\b"
            r"(?:direkt\s+)?(?:vor\s+mir|da|hier|klar|deutlich)\b",
            r"\b(?:das\s+bild|die\s+person|diese\s+person)\s+ist\b.*\b(?:klar|deutlich|vor\s+mir|da|hier)\b",
        )
    )


def _is_explicit_audio_contact_ready_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"\b(?:ich\s+)?(?:hoer|hoere)\b.*\b(?:dich|ihn|sie|es)\b",
            r"\b(?:ich\s+)?(?:hab|habe)\s+(?:dich|ihn|sie|es)\s+gehoert\b",
            r"\b(?:ich\s+)?habs\s+gehoert\b",
        )
    )


def _is_explicit_impact_visibility_ready_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    if re.search(
        r"\b(?:seh|sehe|erkenne)\b.*\b(?:nicht|kein|keine|keinen|keinem|keiner|noch\s+nicht)\b",
        normalized,
    ):
        return False

    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"\b(?:ich\s+)?(?:seh|sehe|erkenne)\b.*\b"
            r"(?:was\s+passiert|was\s+ich\b|die\s+wirkung|den\s+effekt|die\s+folge|es)\b",
            r"\b(?:es|das)\s+ist\s+(?:jetzt\s+)?(?:klar|deutlich)\b",
        )
    )


def _is_explicit_heard_customer_ready_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    if re.search(
        r"\b(?:hab|habe|hoer|hoere|versteh|verstehe)\b.*\b(?:nicht|kein|keine|keinen|keinem|keiner|noch\s+nicht)\b",
        normalized,
    ):
        return False

    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"\b(?:ich\s+)?(?:hab|habe)\s+es\s+gehoert\b",
            r"\b(?:ich\s+)?habs\s+gehoert\b",
            r"\b(?:ich\s+)?(?:hoer|hoere)\b.*\b(?:was\b.*\bgesagt|es)\b",
            r"\b(?:ich\s+)?(?:versteh|verstehe)\b.*\b(?:was\s+los\s+ist|was\s+da\s+passiert|warum)\b",
        )
    )


def _is_explicit_impact_visibility_negative_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    return bool(
        re.search(
            r"\b(?:seh|sehe|erkenne)\b.*\b(?:es|was\s+passiert|die\s+wirkung|den\s+effekt|die\s+folge)\b.*\b(?:nicht|kein|keine|keinen|keinem|keiner|noch\s+nicht)\b",
            normalized,
        )
        or re.search(
            r"\b(?:nicht|kein|keine|keinen|keinem|keiner|noch\s+nicht)\b.*\b(?:seh|sehe|erkenne)\b",
            normalized,
        )
    )


def _is_explicit_heard_customer_negative_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    return bool(
        re.search(
            r"\b(?:hab|habe)\s+es\s+(?:nicht|noch\s+nicht)\s+gehoert\b",
            normalized,
        )
        or re.search(
            r"\b(?:hoer|hoere|versteh|verstehe)\b.*\b(?:nicht|kein|keine|keinen|keinem|keiner|noch\s+nicht)\b",
            normalized,
        )
    )


def _is_explicit_single_person_selection(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    if _extract_person_identity_label(text) is not None:
        return True

    return bool(
        re.search(
            r"\b(?:eine|einer|einen|nur eine|nur einer|eine bestimmte|eine einzelne|eine konkrete)\s+"
            r"(?:person|mensch|frau|mann|typ)\b",
            normalized,
        )
    )


def _is_explicit_whole_group_selection(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    if normalized.startswith("alle "):
        return True

    return bool(
        re.search(
            r"\b(?:ganze|gesamte|komplette)\s+(?:gruppe|clique|klasse|familie)\b",
            normalized,
        )
    )


def _is_explicit_multiple_people_selection(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    return bool(
        re.search(
            r"\b(?:mehrere|einige|verschiedene|mehr als eine|ein paar)\b",
            normalized,
        )
    )


def _classify_group_source_kind_reply(text: str) -> str | None:
    if _is_explicit_whole_group_selection(text):
        return "whole_group"

    if _is_explicit_multiple_people_selection(text):
        return "multiple_people"

    if _is_explicit_single_person_selection(text):
        return "one_person"

    return None


def _classify_origin_target_kind_reply(text: str) -> str | None:
    normalized = _normalize_user_text(text)
    if not normalized:
        return None

    if _is_explicit_single_person_selection(text):
        return "person"

    if _is_explicit_whole_group_selection(text) or _is_explicit_multiple_people_selection(text):
        return "group"

    if re.search(r"\b(?:etwas|was)\s+anderes\b", normalized):
        return "other"

    focus_kind = _classify_focus_reference(text)
    if focus_kind in {"group", "other"}:
        return focus_kind

    return None


def _classify_origin_scene_relevance_reply(text: str) -> str | None:
    normalized = _normalize_user_text(text)
    if not normalized:
        return None

    if _matches_any_regex(normalized, GLOBAL_UNCERTAINTY_REGEXES):
        return "unclear"

    older_origin_regexes = (
        r"\bnoch weiter zurueck\b",
        r"\bweiter zurueck\b",
        r"\bnoch tiefer\b",
        r"\bweiter tiefer\b",
        r"\bnoch frueher\b",
        r"\bfrueherer moment\b",
        r"\bdavor\b",
        r"\bvorher\b",
        r"\bnoch nicht der ursprung\b",
        r"\bnicht der ursprung\b",
        r"\bnicht der kern\b",
        r"\bzwischenstation\b",
        r"\bnur ein hinweis\b",
        r"\bfuehrt mich\b.*\b(?:zurueck|frueher|weiter)\b",
        r"\bzeigt mir\b.*\b(?:frueher|zurueck|weiter)\b",
        r"\bweist\b.*\b(?:zurueck|frueher|weiter)\b",
    )
    if _matches_any_regex(normalized, older_origin_regexes):
        return "older_origin"

    resolve_here_regexes = (
        r"\bgenau hier\b",
        r"\bhier loesen\b",
        r"\bhier aufloesen\b",
        r"\bhier bearbeiten\b",
        r"\bhier\b.*\b(?:anschauen|ansehen)\b",
        r"\bin dieser szene\b.*\b(?:loesen|aufloesen|bearbeiten|bleiben)\b",
        r"\bin diesem moment\b.*\b(?:loesen|aufloesen|bearbeiten|bleiben)\b",
        r"\bdas ist der kern\b",
        r"\bdas ist der ursprung\b",
        r"\bdas ist es\b",
        r"\bgenau das\b",
        r"\bsteht fuer\b",
        r"\bweist auf\b",
        r"\bfuehrt zu\b",
        r"\bhat damit zu tun\b",
        r"\bliegt zwischen uns\b",
        r"\bgeht von\b",
    )
    if _matches_any_regex(normalized, resolve_here_regexes):
        return "resolve_here"

    return None


def _is_mixed_owner_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False
    return bool(
        re.search(r"\b(?:beide|beides|von beiden|sowohl .* als auch|irgendwie beides)\b", normalized)
    )


def _classify_origin_cause_owner_reply(text: str) -> str | None:
    normalized = _normalize_user_text(text)
    if not normalized:
        return None

    if _matches_any_regex(normalized, GLOBAL_UNCERTAINTY_REGEXES):
        return "unclear"

    if _is_mixed_owner_reply(text):
        return "unclear"

    if re.search(
        r"\b(?:in mir|von mir|bei mir|aus mir|etwas in mir|etwas von mir|aus mir selbst|in mir selbst|bei mir selbst)\b",
        normalized,
    ):
        return "self"

    if _classify_focus_reference(text) in {"person", "group", "other"}:
        return "someone_else"

    if re.search(
        r"\b(?:die person|diese person|der person|von der person|bei der person|er|sie|von ihm|von ihr|durch ihn|durch sie|jemand anderes|der andere|die andere)\b",
        normalized,
    ):
        return "someone_else"

    return None


KNOWN_STATE_DRIFT_REGEXES: tuple[str, ...] = (
    r"\b(?:ich\s+kenne|kenn\s+ich|kenne\s+ich)\b",
    r"\bbekannt\b",
    r"\bvertraut\b",
    r"\bzum\s+ersten\b",
    r"\berste(?:s)?\s+mal\b",
    r"\bnoch\s+nie\b",
    r"\bnie\s+gehabt\b",
    r"\bursprung\b",
)

SCENE_ACCESS_VISUAL_MARKERS: tuple[str, ...] = (
    "ich sehe",
    "man sieht",
    "sichtbar",
    "vor mir",
    "eine person",
    "ein mensch",
    "eine gruppe",
    "menschen",
    "kinder",
    "mein vater",
    "meine mutter",
)
SCENE_ACCESS_AUDIO_MARKERS: tuple[str, ...] = (
    "ich hoere",
    "man hoert",
    "stimme",
    "stimmen",
    "lachen",
    "schreien",
    "weinen",
    "musik",
    "geraeusch",
    "geraeusche",
    "reden",
)
SCENE_ACCESS_OTHER_SENSE_MARKERS: tuple[str, ...] = (
    "ich rieche",
    "geruch",
    "gestank",
    "rauchgeruch",
    "ich schmecke",
    "geschmack",
    "warm",
    "kalt",
    "heiss",
    "temperatur",
    "druck",
    "enge",
    "koerper",
    "ich spuere",
    "ich spuer",
    "ich fuehle",
)
VISUAL_FRAGMENT_MARKERS: tuple[str, ...] = (
    "farbe",
    "rot",
    "blau",
    "grun",
    "gruen",
    "gelb",
    "orange",
    "lila",
    "violett",
    "rosa",
    "pink",
    "schwarz",
    "weiss",
    "grau",
    "ort",
    "platz",
    "schule",
    "schulhof",
    "zimmer",
    "kuche",
    "kueche",
    "haus",
    "gebaeude",
    "raum",
    "auto",
    "raucherecke",
    "fenster",
    "treppe",
    "wand",
    "decke",
    "boden",
)
IMMEDIATE_FEELING_MARKERS: tuple[str, ...] = (
    "angst",
    "druck",
    "enge",
    "trauer",
    "wut",
    "scham",
    "ohnmacht",
    "einsamkeit",
    "panik",
    "beklemmung",
    "klo",
    "unruhe",
    "ekel",
    "leer",
    "spannung",
    "schwere",
    "schwer",
    "kribbeln",
    "zittern",
    "zitter",
    "herz",
    "herzklopfen",
    "atem",
    "atemnot",
    "atemlos",
    "warm",
    "waerme",
    "heiss",
    "kalt",
    "kaelte",
    "brust",
    "hals",
    "bauch",
    "uebel",
    "uebelkeit",
)


def _looks_like_known_state_statement(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False
    return _matches_any_regex(normalized, KNOWN_STATE_DRIFT_REGEXES)


def _classify_dark_scene_access_reply(text: str) -> str | None:
    normalized = _normalize_user_text(text)
    if not normalized:
        return None

    if normalized in {"beides", "beide", "sowohl als auch"}:
        return "both"
    if normalized in {"nichts", "gar nichts", "noch nichts", "nichts davon", "keins von beiden"}:
        return "nothing"

    if _matches_any_regex(normalized, VISUAL_NEGATIVE_ONLY_REGEXES) and not _matches_any_regex(
        normalized,
        FULL_NOTHING_REGEXES,
    ):
        return "other_sense"

    has_visual = any(marker in normalized for marker in SCENE_ACCESS_VISUAL_MARKERS)
    has_audio = any(marker in normalized for marker in SCENE_ACCESS_AUDIO_MARKERS)
    has_other_sense = any(marker in normalized for marker in SCENE_ACCESS_OTHER_SENSE_MARKERS)

    if has_visual and has_audio:
        return "both"
    if has_visual:
        return "visual"
    if has_audio:
        return "audio"
    if has_other_sense:
        return "other_sense"
    return None


def _looks_like_visual_fragment_reply(text: str) -> bool:
    normalized = _normalize_user_text(_clean_runtime_fragment(text) or text)
    if not normalized:
        return False

    if _matches_any_regex(normalized, GLOBAL_UNCERTAINTY_REGEXES) or _matches_any_regex(
        normalized,
        GLOBAL_QUESTION_REGEXES,
    ):
        return False
    if _is_acknowledgement_only_reply(text):
        return False

    return _contains_any(normalized, VISUAL_FRAGMENT_MARKERS)


def _looks_like_immediate_feeling_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    if any(marker in normalized for marker in IMMEDIATE_FEELING_MARKERS):
        return True

    return bool(re.search(r"\bich\s+(?:fuehle|spuere|spuer)\b", normalized))


def _classify_category_choice_reply(node_id: str, text: str) -> str | None:
    family = CATEGORY_CHOICE_NODES.get(node_id)
    if family == "group_source_kind":
        return _classify_group_source_kind_reply(text)
    if family == "origin_other_target_kind":
        return _classify_origin_target_kind_reply(text)
    if family == "origin_scene_relevance":
        return _classify_origin_scene_relevance_reply(text)
    if family == "origin_cause_owner":
        return _classify_origin_cause_owner_reply(text)
    return None


def _detect_global_meta_intent(
    node_id: str,
    spec: SemanticNodeSpec,
    customer_message: str,
) -> SemanticModelDecision | None:
    if _looks_like_nonanswer_noise(customer_message):
        intents = set(spec.allowed_intents)
        if "unclear" in intents:
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Globale Meta-State-Policy: offensichtliche Nicht-Antwort oder Tastaturmuster erkannt.",
            )
        if "support_needed" in intents:
            return _make_transition_decision_for_intent(
                node_id,
                "support_needed",
                "Globale Meta-State-Policy: offensichtliche Nicht-Antwort erkannt; erst wieder andocken.",
            )

    normalized = _normalize_user_text(customer_message)
    if not normalized:
        return None
    intents = set(spec.allowed_intents)

    if "abort" in intents and _matches_any_regex(normalized, GLOBAL_ABORT_REGEXES):
        return _make_transition_decision_for_intent(node_id, "abort", "Globale Meta-State-Policy: Abbruchwunsch erkannt.")

    if "support_needed" in intents and _matches_any_regex(normalized, GLOBAL_CONSENT_UNCLEAR_REGEXES):
        return _make_transition_decision_for_intent(
            node_id,
            "support_needed",
            "Globale Meta-State-Policy: Ambivalenz oder unklare Zustimmung erkannt.",
        )

    if "support_needed" in intents and _matches_any_regex(normalized, GLOBAL_FATIGUE_REGEXES):
        return _make_transition_decision_for_intent(
            node_id,
            "support_needed",
            "Globale Meta-State-Policy: Wegdriften oder Muedigkeit erkannt.",
        )

    if "support_needed" in intents and _matches_any_regex(normalized, GLOBAL_SUPPORT_REGEXES):
        return _make_transition_decision_for_intent(
            node_id,
            "support_needed",
            "Globale Meta-State-Policy: Unterstuetzungsbedarf erkannt.",
        )

    if "question" in intents and _is_question_announcement(customer_message):
        return _make_transition_decision_for_intent(
            node_id,
            "question",
            "Globale Meta-State-Policy: Frage-Ankuendigung erkannt.",
        )

    if "question" in intents and _matches_any_regex(normalized, GLOBAL_QUESTION_REGEXES):
        return _make_transition_decision_for_intent(
            node_id,
            "question",
            "Globale Meta-State-Policy: Rueckfrage oder Missverstaendnis erkannt.",
        )

    if node_id == "origin_person_name" and re.search(
        r"\b(?:ich\s+)?(?:weiss|kenn|kenne|erkenne)\b.*\bnicht\b.*\b(?:wer|wen)\b",
        normalized,
    ):
        return None

    if _matches_any_regex(normalized, GLOBAL_UNCERTAINTY_REGEXES):
        if "unclear" in intents:
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Globale Meta-State-Policy: echte Unsicherheit statt inhaltlicher Antwort erkannt.",
            )
        if "support_needed" in intents:
            return _make_transition_decision_for_intent(
                node_id,
                "support_needed",
                "Globale Meta-State-Policy: echte Unsicherheit braucht erst Stabilisierung oder Zeit.",
            )

    if "support_needed" in intents and _matches_any_regex(normalized, GLOBAL_HOSTILE_REGEXES):
        return _make_transition_decision_for_intent(
            node_id,
            "support_needed",
            "Globale Meta-State-Policy: Frust oder feindselige Reaktion erkannt.",
        )

    return None


def _is_explicit_dark_scene_description(normalized: str) -> bool:
    return _matches_any_regex(normalized, EXPLICIT_DARK_REGEXES)


def _is_explicit_hell_scene_description(normalized: str) -> bool:
    return _matches_any_regex(normalized, EXPLICIT_HELL_REGEXES)


def _is_explicit_mixed_light_description(normalized: str) -> bool:
    return _matches_any_regex(normalized, EXPLICIT_MIXED_LIGHT_REGEXES)


DARK_SCENE_ACCESS_NODES = {"dark_scene_perception", "dark_scene_mode_clarify"}
DARK_NONVISUAL_ACCESS_NODES = {"dark_scene_other_sense", "dark_scene_first_spuerbar"}
FREEFORM_SCENE_DETAIL_NODES = {
    "dark_scene_who",
    "dark_scene_audio_detail",
    "dark_scene_other_sense",
    "dark_scene_first_spuerbar",
    "dark_scene_people_who",
    "dark_scene_happening",
}


def _predecide_light_level(
    node_id: str,
    customer_message: str,
    normalized: str,
) -> SemanticModelDecision | None:
    if node_id != "hell_light_level":
        return None
    if _is_explicit_dark_scene_description(normalized):
        return _make_transition_decision_for_intent(
            node_id,
            "darker_or_other",
            "Knotensemantik: explizite Dunkel-Einordnung erkannt.",
        )
    if _is_explicit_hell_scene_description(normalized):
        return _make_transition_decision_for_intent(
            node_id,
            "hell_light",
            "Knotensemantik: explizite Hell-Einordnung erkannt.",
        )
    if _is_explicit_mixed_light_description(normalized):
        return _make_transition_decision_for_intent(
            node_id,
            "both",
            "Knotensemantik: gemischte Hell-Dunkel-Einordnung erkannt.",
        )
    if len(normalized) >= 3:
        return _make_transition_decision_for_intent(
            node_id,
            "unclear",
            "Knotensemantik: ohne explizites hell, dunkel oder beides wird die Szene hier nicht frei geraten.",
        )
    return None


def _predecide_dark_scene_access(
    node_id: str,
    customer_message: str,
    normalized: str,
) -> SemanticModelDecision | None:
    if node_id not in DARK_SCENE_ACCESS_NODES:
        return None

    explicit_access_state = _classify_dark_scene_access_reply(customer_message)
    if explicit_access_state is not None:
        return _make_transition_decision_for_intent(
            node_id,
            explicit_access_state,
            "Knotensemantik: explizite Einordnung des Wahrnehmungskanals erkannt.",
        )
    if _looks_like_visual_fragment_reply(customer_message):
        return _make_transition_decision_for_intent(
            node_id,
            "visual",
            "Knotensemantik: visuelles Kurzfragment erkannt; klaere direkt den sichtbaren Inhalt weiter.",
        )
    if _matches_any_regex(normalized, VISUAL_NEGATIVE_ONLY_REGEXES) and not _matches_any_regex(
        normalized,
        FULL_NOTHING_REGEXES,
    ):
        return _make_transition_decision_for_intent(
            node_id,
            "other_sense",
            "Knotensemantik: fehlender visueller Zugang erkannt; oeffne stattdessen den Zugang ueber andere Sinne.",
        )
    return None


def _predecide_scene_access_followup(
    node_id: str,
    customer_message: str,
    normalized: str,
) -> SemanticModelDecision | None:
    if node_id != "scene_access_followup":
        return None

    if _is_explicit_dark_scene_description(normalized):
        return _make_transition_decision_for_intent(
            node_id,
            "visual_dark",
            "Knotensemantik: verspaetete Dunkel-Einordnung auf die erste Wahrnehmungsfrage erkannt.",
        )
    if _is_explicit_hell_scene_description(normalized):
        return _make_transition_decision_for_intent(
            node_id,
            "visual_hell",
            "Knotensemantik: verspaetete Hell-Einordnung auf die erste Wahrnehmungsfrage erkannt.",
        )

    explicit_access_state = _classify_dark_scene_access_reply(customer_message)
    if explicit_access_state in {"visual", "both"}:
        return _make_transition_decision_for_intent(
            node_id,
            "visual_hell",
            "Knotensemantik: es wird jetzt ein visueller Zugang beschrieben; klaere direkt, was gesehen wird.",
        )
    if _looks_like_visual_fragment_reply(customer_message):
        return _make_transition_decision_for_intent(
            node_id,
            "visual_hell",
            "Knotensemantik: visuelles Kurzfragment erkannt; klaere direkt den sichtbaren Inhalt weiter.",
        )
    if explicit_access_state in {"audio", "other_sense"}:
        return _make_transition_decision_for_intent(
            node_id,
            "nonvisual_access",
            "Knotensemantik: es wird ein nichtvisueller Zugang beschrieben; wechsle in die Body-Bridge-Weiterfuehrung.",
        )
    if explicit_access_state == "nothing":
        return _make_transition_decision_for_intent(
            node_id,
            "nothing_yet",
            "Knotensemantik: weiterhin noch kein Zugang erkennbar; wechsle in die Body-Bridge-Weiterfuehrung.",
        )
    return None


def _predecide_nonvisual_access(
    node_id: str,
    customer_message: str,
    normalized: str,
) -> SemanticModelDecision | None:
    if node_id not in DARK_NONVISUAL_ACCESS_NODES:
        return None

    explicit_access_state = _classify_dark_scene_access_reply(customer_message)
    if _looks_like_known_state_statement(customer_message) or _looks_like_same_scene_report(customer_message):
        return _make_transition_decision_for_intent(
            node_id,
            "unclear",
            "Knotensemantik: Antwort passt inhaltlich nicht zur aktuellen Sinnesfrage; bleibe zur Klaerung im aktuellen Knoten.",
        )
    if (
        explicit_access_state in {"visual", "audio", "both"}
        or _looks_like_visual_fragment_reply(customer_message)
        or _is_explicit_dark_scene_description(normalized)
        or _is_explicit_hell_scene_description(normalized)
        or _is_explicit_mixed_light_description(normalized)
    ):
        return _make_transition_decision_for_intent(
            node_id,
            "ready",
            "Knotensemantik: verspaetete Kanal- oder Lichteinordnung erkannt; die Laufzeit korrigiert den Pfad passend.",
        )
    if _is_generic_body_placeholder(customer_message):
        return _make_transition_decision_for_intent(
            node_id,
            "unclear",
            "Knotensemantik: an diesem Koerper-/Sinnesknoten wurde nur ein generischer Platzhalter statt einer konkreten Wahrnehmung genannt.",
        )
    if explicit_access_state == "nothing":
        return _make_transition_decision_for_intent(
            node_id,
            "unclear",
            "Knotensemantik: weiterhin keine konkrete Wahrnehmung benannt; bleibe zur Klaerung im aktuellen Koerper-/Sinnesknoten.",
        )
    if explicit_access_state == "other_sense":
        return _make_transition_decision_for_intent(
            node_id,
            "ready",
            "Knotensemantik: klare Wahrnehmung im Koerper-/Geruchs-/Geschmacks-/Temperaturkanal erkannt.",
        )
    return None


def _predecide_immediate_feeling(
    node_id: str,
    customer_message: str,
    normalized: str,
) -> SemanticModelDecision | None:
    if node_id != "dark_scene_immediate_feeling":
        return None

    if _is_generic_body_placeholder(customer_message):
        return _make_transition_decision_for_intent(
            node_id,
            "unclear",
            "Knotensemantik: an diesem Gefuehlsknoten wurde nur ein generischer Platzhalter statt eines konkreten Gefuehls genannt.",
        )
    if _looks_like_immediate_feeling_reply(customer_message):
        return _make_transition_decision_for_intent(
            node_id,
            "ready",
            "Knotensemantik: konkrete unmittelbare Gefuehls- oder Koerperbeschreibung erkannt.",
        )
    if (
        _classify_dark_scene_access_reply(customer_message) is not None
        or _looks_like_visual_fragment_reply(customer_message)
        or _is_explicit_dark_scene_description(normalized)
        or _is_explicit_hell_scene_description(normalized)
        or _is_explicit_mixed_light_description(normalized)
        or _looks_like_known_state_statement(customer_message)
        or _looks_like_same_scene_report(customer_message)
    ):
        return _make_transition_decision_for_intent(
            node_id,
            "unclear",
            "Knotensemantik: Antwort beschreibt nicht das unmittelbare Gefuehl, sondern einen anderen Kanal oder eine andere Ebene; bleibe zur Klaerung im aktuellen Knoten.",
        )
    return None


def _detect_node_semantic_predecision(
    node_id: str,
    customer_message: str,
) -> SemanticModelDecision | None:
    normalized = _normalize_user_text(customer_message)
    if not normalized:
        return None

    for helper in (
        _predecide_light_level,
        _predecide_dark_scene_access,
        _predecide_scene_access_followup,
        _predecide_nonvisual_access,
        _predecide_immediate_feeling,
    ):
        decision = helper(node_id, customer_message, normalized)
        if decision is not None:
            return decision

    if node_id in {"group_person_ready", "person_switch_sees_customer"}:
        if _is_explicit_person_visibility_ready_reply(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "yes",
                "Knotensemantik: explizite Sichtbarkeits- oder Erkennungsbestaetigung erkannt.",
            )

    if node_id == "person_switch_hears":
        if _is_explicit_audio_contact_ready_reply(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "yes",
                "Knotensemantik: explizite Hoer-Bestaetigung in der uebernommenen Perspektive erkannt.",
            )

    if node_id == "person_switch_sees_impact":
        if _is_explicit_impact_visibility_negative_reply(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "no",
                "Knotensemantik: explizite Verneinung erkannt, dass die Wirkung oder das Geschehen aus dieser Perspektive schon sichtbar ist.",
            )
        if _is_explicit_impact_visibility_ready_reply(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "yes",
                "Knotensemantik: explizite Wirkungs- oder Verlaufserkennung erkannt.",
            )

    if node_id in {"person_switch_heard_customer", "person_switch_self_heard"}:
        if _is_explicit_heard_customer_negative_reply(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "no",
                "Knotensemantik: explizite Verneinung erkannt, dass das Gehoerte oder Verstandene schon angekommen ist.",
            )
        if _is_explicit_heard_customer_ready_reply(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "yes",
                "Knotensemantik: explizite Rueckmeldung erkannt, dass das Gehoerte oder Verstandene angekommen ist.",
            )

    if node_id == "origin_person_name":
        if _looks_like_direct_question(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "question",
                "Knotensemantik: direkte Rueckfrage statt Personenklaerung erkannt.",
            )
        if _extract_person_identity_label(customer_message) is not None:
            return _make_transition_decision_for_intent(
                node_id,
                "ready",
                "Knotensemantik: konkrete Person fuer diesen Namensknoten erkannt.",
            )
        if re.search(
            r"\b(?:ich\s+)?(?:weiss|kenn|kenne|erkenne)\b.*\bnicht\b.*\b(?:wer|wen)\b",
            normalized,
        ):
            return _make_transition_decision_for_intent(
                node_id,
                "unknown_person",
                "Knotensemantik: die Person ist wahrnehmbar, aber noch nicht eindeutig identifizierbar.",
            )
        return _make_transition_decision_for_intent(
            node_id,
            "unclear",
            "Knotensemantik: noch keine konkrete Person benannt.",
        )

    if node_id == "dark_scene_people_who":
        if _looks_like_direct_question(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "question",
                "Knotensemantik: direkte Rueckfrage statt Identifizierung der gesehenen Person oder Gruppe erkannt.",
            )
        if _is_generic_ready_placeholder_for_node(node_id, customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Knotensemantik: die gesehene Person oder Gruppe wurde noch nicht konkret genug beschrieben.",
            )
        if _classify_focus_reference(customer_message) == "group":
            return _make_transition_decision_for_intent(
                node_id,
                "ready",
                "Knotensemantik: die gesehene Gruppe wurde konkret genug beschrieben.",
            )
        if _extract_person_identity_label(customer_message) is not None:
            return _make_transition_decision_for_intent(
                node_id,
                "ready",
                "Knotensemantik: die gesehene Person wurde konkret genug beschrieben.",
            )
        return None

    if node_id in NAMED_PERSON_INPUT_NODES:
        if _looks_like_direct_question(customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "question",
                "Knotensemantik: direkte Rueckfrage statt Personenbenennung erkannt.",
            )
        if _extract_person_identity_label(customer_message) is None:
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Knotensemantik: noch keine konkrete Person benannt.",
            )
        return _make_transition_decision_for_intent(
            node_id,
            "ready",
            "Knotensemantik: konkrete Person fuer diesen Namensknoten erkannt.",
        )

    if node_id in CATEGORY_CHOICE_NODES:
        if len(normalized) <= 1:
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Knotensemantik: die Antwort ist fuer eine belastbare Kategorien-Einordnung zu kurz.",
            )
        explicit_kind = _classify_category_choice_reply(node_id, customer_message)
        if explicit_kind is not None:
            reason_map = {
                "group_source_kind": "Knotensemantik: explizite Einordnung der Gruppendynamik erkannt.",
                "origin_other_target_kind": "Knotensemantik: explizite Einordnung des Ausloesertyps erkannt.",
                "origin_scene_relevance": "Knotensemantik: explizite Einordnung erkannt, ob diese Szene hier bearbeitet oder noch weiter zurueck verfolgt werden soll.",
                "origin_cause_owner": "Knotensemantik: klare Einordnung erkannt, ob der Kern eher im Gegenueber oder eher in dir selbst liegt.",
            }
            return _make_transition_decision_for_intent(
                node_id,
                explicit_kind,
                reason_map[node_id],
            )

    if node_id in {"hell_feel_branch", "hell_regulation_choice"} and _is_explicit_dark_scene_description(normalized):
        return _make_transition_decision_for_intent(
            node_id,
            "reclassified_dark",
            "Knotensemantik: der Kunde korrigiert die Szene als dunkel; wechsle zurueck in den Dunkelpfad.",
        )

    if node_id == "dark_scene_feeling_intensity" and _matches_any_regex(
        normalized,
        FEELING_NOT_PERCEIVABLE_REGEXES,
    ):
        return _make_transition_decision_for_intent(
            node_id,
            "support_needed",
            "Knotensemantik: das ungute Gefuehl ist gerade nicht klar spuerbar; fuehre zuerst wieder in die Szene und stabilisiere den Zugang.",
        )

    if node_id in {"dark_scene_feeling_intensity", "dark_scene_immediate_feeling"} and _matches_any_regex(
        normalized,
        ACUTE_AFFECT_DISTRESS_REGEXES,
    ):
        return _make_transition_decision_for_intent(
            node_id,
            "support_needed",
            "Knotensemantik: akute Belastung im Gefuehlsknoten erkannt; zuerst Abstand, Atmung und Stabilisierung statt weiterer Vertiefung.",
        )

    if node_id in FREEFORM_SCENE_DETAIL_NODES:
        if _is_bare_status_reply(customer_message) or _is_generic_ready_placeholder_for_node(node_id, customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Knotensemantik: noch keine ausreichend konkrete inhaltliche Szenenbeschreibung erkannt.",
            )
        return None

    return None


def _question_announcement_reply(attempt: int) -> str:
    replies = [
        "Dann stell deine Frage jetzt einfach kurz.",
        "Ich brauche die Frage selbst, damit ich dir konkret darauf antworten kann. Stell sie jetzt einfach kurz.",
        "Stell die Frage einfach direkt. Ich antworte dir darauf, und anschliessend gehen wir genau hier weiter.",
    ]
    return replies[min(max(attempt, 0), len(replies) - 1)]


def _extract_scale_value(text: str) -> str | None:
    normalized = _normalize_user_text(text)
    if not normalized:
        return None

    digit_match = re.search(r"\b(10|[1-9])\b", normalized)
    if digit_match:
        return digit_match.group(1)

    word_to_number = {
        "eins": "1",
        "zwei": "2",
        "drei": "3",
        "vier": "4",
        "funf": "5",
        "fuenf": "5",
        "sechs": "6",
        "sieben": "7",
        "acht": "8",
        "neun": "9",
        "zehn": "10",
    }
    for word in normalized.split():
        if word in word_to_number:
            return word_to_number[word]
    return None


def _is_bare_status_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False
    bare_status_replies = {
        "ja",
        "ja klar",
        "ja schon",
        "ich glaube ja",
        "ich denke ja",
        "nein",
        "noch nicht",
        "weiss nicht",
        "ich weiss nicht",
        "nicht klar",
        "ist mir nicht klar",
        "keine ahnung",
    }
    return normalized in bare_status_replies


def _is_acknowledgement_only_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False

    exact_replies = {
        "ja",
        "ja klar",
        "ok",
        "okay",
        "alles klar",
        "klar",
        "verstanden",
        "ich verstehe",
        "ich habe verstanden",
        "hab verstanden",
        "passt",
        "passt so",
        "gut",
    }
    if normalized in exact_replies:
        return True

    acknowledgement_patterns = (
        r"^(?:ja\s+)?ich\s+(?:erkenne|weiss|kenne|sehe)\s+wer\s+(?:es|das|sie|er|die\s+person)\s+ist$",
        r"^(?:ja\s+)?ich\s+(?:erkenne|weiss|kenne|sehe)\s+wer\s+gemeint\s+ist$",
        r"^(?:ja\s+)?ich\s+(?:erkenne|weiss|kenne)\s+es$",
    )
    return any(re.search(pattern, normalized) for pattern in acknowledgement_patterns)


def _is_group_person_trigger_core_short_status_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False
    return normalized in {
        "ja",
        "ja klar",
        "ja schon",
        "noch nicht",
        "weiss nicht",
        "ich weiss nicht",
        "nicht klar",
        "ist mir nicht klar",
        "keine ahnung",
    }


def _is_generic_perception_placeholder(text: str) -> bool:
    normalized = _normalize_user_text(text)
    cleaned = _normalize_user_text(_clean_runtime_fragment(text) or text)
    if cleaned in {"was", "etwas", "irgendwas"}:
        return True

    generic_patterns = (
        r"^ich\s+(?:sehe|seh|hoere|hore|spuere|spuer|fuehle|rieche|schmecke)(?:\s+es)?\s+(?:was|etwas)\b",
        r"^ich\s+nehme(?:\s+es)?\s+(?:was|etwas)\s+wahr\b",
        r"^da\s+ist\s+(?:was|etwas)\b",
        r"^dort\s+ist\s+(?:was|etwas)\b",
    )
    return any(re.search(pattern, normalized) for pattern in generic_patterns)


def _is_generic_visual_placeholder(text: str) -> bool:
    normalized = _normalize_user_text(text)
    cleaned = _normalize_user_text(_clean_runtime_fragment(text) or text)
    if cleaned in {"was", "etwas", "irgendwas"}:
        return True
    return normalized in {
        "ich sehe was",
        "ich sehe etwas",
        "da ist was",
        "da ist etwas",
    }


def _is_generic_audio_placeholder(text: str) -> bool:
    cleaned = _normalize_user_text(_clean_runtime_fragment(text) or text)
    if cleaned in {"was", "etwas", "ein geraeusch", "geraeusch", "ein ton", "ein laut"}:
        return True
    normalized = _normalize_user_text(text)
    return normalized in {
        "ich hore was",
        "ich hore etwas",
        "ich hoere was",
        "ich hoere etwas",
        "da ist ein gerausch",
        "da ist ein geraeusch",
    }


def _is_generic_body_placeholder(text: str) -> bool:
    cleaned = _normalize_user_text(_clean_runtime_fragment(text) or text)
    if cleaned in {"was", "etwas", "irgendwas"}:
        return True
    normalized = _normalize_user_text(text)
    return normalized in {
        "ich spuere was",
        "ich spuere etwas",
        "ich spuer was",
        "ich spuer etwas",
        "ich fuehle was",
        "ich fuehle etwas",
        "ich rieche was",
        "ich rieche etwas",
        "ich schmecke was",
        "ich schmecke etwas",
    }


def _looks_like_age_reply(text: str) -> bool:
    normalized = _normalize_user_text(_clean_runtime_fragment(text) or text)
    if not normalized:
        return False
    if re.search(r"\b\d{1,3}\b", normalized):
        return True
    if "jahr" in normalized or "alt" in normalized:
        return True
    age_words = {
        "eins",
        "zwei",
        "drei",
        "vier",
        "funf",
        "fuenf",
        "sechs",
        "sieben",
        "acht",
        "neun",
        "zehn",
        "elf",
        "zwolf",
        "zwoelf",
        "dreizehn",
        "vierzehn",
        "funfzehn",
        "fuenfzehn",
        "sechzehn",
        "siebzehn",
        "achtzehn",
        "neunzehn",
        "zwanzig",
        "kind",
        "schulkind",
        "jugendlich",
        "jugendlicher",
        "teenager",
        "erwachsen",
        "jung",
        "klein",
    }
    return any(token in age_words for token in normalized.split())


def _is_generic_ready_placeholder_for_node(node_id: str, text: str) -> bool:
    if _is_acknowledgement_only_reply(text):
        return True

    identity_required_nodes = {
        "origin_person_name",
        "group_representative_name",
        "group_specific_person_name",
        "group_multiple_people_name",
        "group_multiple_required_name",
        "group_next_person_name",
        "dark_scene_people_who",
    }
    if node_id in identity_required_nodes:
        cleaned = _clean_runtime_fragment(text) or text
        cleaned_normalized = _normalize_user_text(cleaned)
        generic_identity_replies = {
            "person",
            "eine person",
            "diese person",
            "jemand",
            "jemanden",
            "irgendjemand",
            "was",
            "etwas",
        }
        if cleaned_normalized in generic_identity_replies:
            return True
        if _is_generic_perception_placeholder(text):
            return True

    if node_id == "dark_scene_who":
        return _is_generic_visual_placeholder(text)

    if node_id == "dark_scene_audio_detail":
        return _is_generic_audio_placeholder(text)

    if node_id in {"dark_scene_other_sense", "dark_scene_first_spuerbar", "dark_scene_immediate_feeling"}:
        return _is_generic_body_placeholder(text)

    if node_id in {"dark_scene_happening", "origin_trigger_source", "origin_self_need", "person_switch_why"}:
        return _is_generic_perception_placeholder(text)

    if node_id in STRICT_PHASE4_EXPLANATION_NODES:
        return _is_generic_perception_placeholder(text)

    if node_id == "dark_scene_age":
        return not _looks_like_age_reply(text)

    return False


def _scale_confirmation_prefix(node_id: str, scale_value: str | None) -> str:
    if not scale_value:
        return ""
    if node_id == "session_phase2_post_scale_before_script":
        return f"Okay, {scale_value}."
    if node_id == "session_phase2_post_scale_after_script":
        return f"Gut, jetzt bei {scale_value}."
    return ""


def _strip_person_hedges(normalized_text: str) -> str:
    hedge_prefixes = (
        "ich glaube ",
        "glaube ",
        "ich denke ",
        "denke ",
        "vermutlich ",
        "wahrscheinlich ",
        "vielleicht ",
        "wohl ",
        "eher ",
    )
    stripped = normalized_text
    trimmed = True
    while trimmed:
        trimmed = False
        for prefix in hedge_prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :].strip()
                trimmed = True
    return stripped


def _looks_like_specific_person_label(text: str) -> bool:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return False

    raw = _clean_runtime_fragment(raw) or raw
    raw = re.sub(
        r"^(?:er heisst|sie heisst|heisst|das ist|es ist)\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip() or raw
    normalized = _normalize_user_text(raw)
    if not normalized:
        return False
    normalized_tokens = [part for part in normalized.split() if part]

    if (
        _matches_any_regex(normalized, GLOBAL_UNCERTAINTY_REGEXES)
        or _matches_any_regex(normalized, GLOBAL_QUESTION_REGEXES)
        or _looks_like_direct_question(raw)
        or _is_acknowledgement_only_reply(raw)
        or _is_generic_person_reference(raw)
    ):
        return False

    if _classify_focus_reference(raw) in {"group", "other"}:
        return False

    specific_role_markers = {
        "vater",
        "mutter",
        "papa",
        "mama",
        "bruder",
        "schwester",
        "oma",
        "opa",
        "onkel",
        "tante",
        "lehrer",
        "lehrerin",
        "chef",
        "chefin",
        "freund",
        "freundin",
        "arzt",
        "aerztin",
        "therapeut",
        "therapeutin",
        "kollege",
        "kollegin",
    }
    descriptor_markers = {
        "mit",
        "am",
        "im",
        "beim",
        "von",
        "aus",
        "vor",
        "hinter",
        "neben",
        "gegenueber",
    }
    allowed_role_phrase_tokens = {
        "mein",
        "meine",
        "meinem",
        "meinen",
        "meiner",
        "meines",
        "dein",
        "deine",
        "deinem",
        "deinen",
        "deiner",
        "deines",
        "sein",
        "seine",
        "seinem",
        "seinen",
        "seiner",
        "ihr",
        "ihre",
        "ihrem",
        "ihren",
        "ihrer",
        "ein",
        "eine",
        "einem",
        "einen",
        "einer",
        "der",
        "die",
        "dem",
        "den",
        *specific_role_markers,
    }
    if any(token in specific_role_markers for token in normalized_tokens):
        if all(token in allowed_role_phrase_tokens for token in normalized_tokens):
            return True
        if any(token in descriptor_markers for token in normalized_tokens):
            return True
        return False

    if re.search(r"\b(?:frau|herr)\s+[a-zA-ZaeiouyAEIOUY???????-]{2,}\b", raw, flags=re.IGNORECASE):
        return True

    if re.search(r"\btyp\b", normalized):
        return bool(re.search(r"\btyp\b.*\b(?:mit|am|im|beim|von|aus|vor|hinter)\b", normalized))

    generic_anchor_tokens = {
        "person",
        "jemand",
        "jemanden",
        "irgendjemand",
        "frau",
        "mann",
        "mensch",
        "typ",
    }
    if any(token in generic_anchor_tokens for token in normalized_tokens):
        return any(token in descriptor_markers for token in normalized_tokens)

    stripped = re.sub(
        r"^(?:es ist|das ist|ich nehme|wir nehmen|nehmen wir|ich wuerde|ich wurde|das waere|das ware|die person ist|die person waere|das ist der|das ist die|der|die|das|ein|eine)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    if not stripped:
        return False

    tokens = [part for part in stripped.split() if part]
    if not 1 <= len(tokens) <= 3:
        return False
    if not all(re.fullmatch(r"[a-zA-ZaeiouyAEIOUY???????-]+", token) for token in tokens):
        return False

    non_name_tokens = {
        "spaghetti",
        "pizza",
        "nudeln",
        "wetter",
        "morgen",
        "heute",
        "gestern",
        "montag",
        "dienstag",
        "mittwoch",
        "donnerstag",
        "freitag",
        "samstag",
        "sonntag",
        "druck",
        "enge",
        "angst",
        "wut",
        "scham",
        "traurigkeit",
        "rauch",
        "qualm",
        "zigarette",
        "zigaretten",
        "geruch",
        "gestank",
        "geschmack",
        "stimme",
        "lachen",
        "farbe",
        "rot",
        "blau",
        "grun",
        "gruen",
        "gelb",
        "orange",
        "rosa",
        "pink",
        "lila",
        "violett",
        "schwarz",
        "weiss",
        "grau",
        "moment",
        "situation",
        "ereignis",
        "ort",
        "platz",
        "schule",
        "schulhof",
        "zimmer",
        "kuche",
        "kueche",
        "haus",
        "auto",
        "raum",
        "blick",
        "verhalten",
        "spannung",
        "gefuhl",
        "gefuehl",
        "gebaeude",
        "gebaeude",
    }
    abstract_suffixes = ("ung", "keit", "heit", "schaft", "tion", "erei", "nis", "tum", "ismus", "ment")
    if any(token in non_name_tokens for token in tokens):
        return False
    if any(token.endswith(abstract_suffixes) for token in tokens):
        return False

    raw_tokens = [part.strip(" \t\n\r.,!?;:\"'()[]{}") for part in raw.split() if part.strip(" \t\n\r.,!?;:\"'()[]{}")]
    if any(token[:1].isupper() for token in raw_tokens):
        return True

    if len(tokens) == 1:
        return 2 <= len(tokens[0]) <= 10
    if len(tokens) == 2 and all(token[:1].isupper() for token in raw_tokens):
        return all(2 <= len(token) <= 20 for token in tokens)
    return False


def _is_generic_person_reference(text: str) -> bool:
    normalized = _normalize_user_text(_clean_runtime_fragment(text) or text)
    if not normalized:
        return False

    if re.search(r"\b(?:frau|herr)\s+[a-zA-ZaeiouyAEIOUY???????-]{2,}\b", text or "", flags=re.IGNORECASE):
        return False

    generic_identity_tokens = {
        "person",
        "jemand",
        "jemanden",
        "irgendjemand",
        "frau",
        "mann",
        "mensch",
        "typ",
    }
    specific_role_tokens = {
        "freund",
        "freundin",
        "lehrer",
        "lehrerin",
        "chef",
        "chefin",
        "arzt",
        "aerztin",
        "therapeut",
        "therapeutin",
        "kollege",
        "kollegin",
        "vater",
        "mutter",
        "bruder",
        "schwester",
        "oma",
        "opa",
        "onkel",
        "tante",
    }
    descriptor_markers = {
        "mit",
        "am",
        "im",
        "beim",
        "von",
        "aus",
        "vor",
        "hinter",
        "neben",
        "gegenueber",
    }
    generic_modifier_tokens = {
        "ein",
        "eine",
        "einer",
        "einen",
        "einem",
        "der",
        "die",
        "das",
        "den",
        "dem",
        "diese",
        "dieser",
        "diesem",
        "diesen",
        "jene",
        "jener",
        "jenem",
        "jenen",
        "bestimmte",
        "bestimmten",
        "einzelne",
        "einzelnen",
        "konkrete",
        "konkreten",
        "selber",
        "selbst",
        "da",
        "dort",
        "hier",
        "drueben",
        "druben",
        "hinten",
        "vorne",
        "halt",
        "nur",
        "gerade",
        "grad",
    }
    tokens = [part for part in normalized.split() if part]
    if any(token in specific_role_tokens for token in tokens):
        return False
    if bool(tokens) and any(token in generic_identity_tokens for token in tokens):
        if not any(token in descriptor_markers for token in tokens):
            return True
    if bool(tokens) and any(token in generic_identity_tokens for token in tokens) and all(
        token in generic_identity_tokens or token in generic_modifier_tokens for token in tokens
    ):
        return True

    role_tokens = specific_role_tokens
    vague_role_modifiers = {
        "diese",
        "dieser",
        "diesem",
        "diesen",
        "jene",
        "jener",
        "jenem",
        "jenen",
        "da",
        "dort",
        "hier",
        "hinten",
        "vorne",
        "halt",
        "gerade",
        "grad",
    }
    return (
        bool(tokens)
        and any(token in role_tokens for token in tokens)
        and any(token in vague_role_modifiers for token in tokens)
        and all(token in role_tokens or token in vague_role_modifiers for token in tokens)
    )


def _extract_named_person_label(text: str) -> str | None:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return None

    raw = _clean_runtime_fragment(raw) or raw
    raw = re.sub(
        r"^(?:(?:ja|okay|ok|gut|doch|also|mhm|hm)\b[\s,.:;-]*)+",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip() or raw
    normalized = _normalize_user_text(raw)
    if re.search(
        r"^(?:ich\s+)?(?:erkenne|kenne|seh|sehe|weiss)\s+"
        r"(?:(?:die|diese|eine)\s+person|wer\s+(?:es|sie)\s+ist)\b"
        r"(?:\s+(?:jetzt|schon|klar|besser|genau|vor\s+mir))?$",
        normalized,
    ):
        return None
    if re.search(
        r"^(?:(?:die|diese|eine)\s+)?person\s+"
        r"(?:steht\s+(?:vor\s+mir|jetzt\s+vor\s+mir|klar\s+vor\s+mir)|"
        r"ist\s+(?:da|hier|klar|jetzt\s+da|jetzt\s+klar))$",
        normalized,
    ):
        return None
    generic_placeholders = {
        "eine person",
        "eine bestimmte person",
        "eine einzelne person",
        "eine konkrete person",
        "die person",
        "diese person",
        "der person",
        "die person selber",
        "die person selbst",
        "diese person selber",
        "diese person selbst",
        "person",
        "jemand",
        "jemanden",
        "irgendjemand",
        "was",
        "etwas",
        "eine davon",
        "etwas anderes",
        "was anderes",
        "anderes",
    }
    generic_normalized = _strip_person_hedges(normalized)
    if (
        generic_normalized in generic_placeholders
        or _is_generic_person_reference(raw)
        or _looks_like_direct_question(raw)
    ):
        return None

    if not _looks_like_specific_person_label(raw):
        return None

    raw = raw.strip(" \t\n\r.,!?;:\"'()[]{}")
    patterns = [
        r"^(?:es ist|das ist|ich nehme|wir nehmen|nehmen wir|ich wuerde|ich würde|das waere|das wäre|die person ist|die person waere|das ist der|das ist die)\s+",
        r"^(?:er heisst|sie heisst|heisst)\s+",
        r"^(?:der|die|das)\s+",
    ]
    cleaned = raw
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    if not cleaned:
        cleaned = raw

    cleaned_normalized = _normalize_user_text(cleaned)
    if cleaned_normalized in generic_placeholders or _is_generic_person_reference(cleaned):
        return None

    if cleaned.islower():
        cleaned = " ".join(part.capitalize() for part in cleaned.split())

    return cleaned or None


def _trim_person_identity_candidate(text: str) -> str:
    candidate = re.sub(r"\s+", " ", (text or "").strip())
    if not candidate:
        return ""

    split_pattern = (
        r"\s+(?=(?:vor\s+mir|vor\s+dir|vor\s+ihm|vor\s+ihr|"
        r"steht|stehen|stand|sitzt|sitzen|sass|liegt|liegen|lag|"
        r"raucht|rauchen|lacht|lachen|schaut|schauen|guckt|gucken|"
        r"redet|reden|spricht|sprechen|sagt|sagen|geht|gehen|kommt|kommen|"
        r"schreit|schreien|weint|weinen|er\b|sie\b|ihn\b|ihm\b|ihr\b))"
    )
    trimmed = re.split(split_pattern, candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return trimmed.strip(" \t\n\r.,!?;:\"'()[]{}")


def _extract_person_identity_label(text: str) -> str | None:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return None

    cleaned = _clean_runtime_fragment(raw) or raw
    normalized_cleaned = _normalize_user_text(cleaned)
    if re.search(
        r"^(?:(?:die|diese|eine)\s+)?person\s+"
        r"(?:steht\s+(?:vor\s+mir|jetzt\s+vor\s+mir|klar\s+vor\s+mir)|"
        r"ist\s+(?:da|hier|klar|jetzt\s+da|jetzt\s+klar))$",
        normalized_cleaned,
    ):
        return None

    candidates: list[str] = []
    hedge_only_candidates = {
        "vermutlich",
        "wahrscheinlich",
        "vielleicht",
        "wohl",
        "eher",
        "ich glaube",
        "ich glaub",
        "ich denke",
        "denke",
        "glaube",
        "da",
        "hier",
        "klar",
        "jetzt da",
        "jetzt klar",
    }

    def _add_candidate(candidate: str) -> None:
        collapsed = re.sub(r"\s+", " ", (candidate or "").strip())
        if collapsed and collapsed not in candidates:
            candidates.append(collapsed)

    for base in (cleaned, raw):
        if not base:
            continue
        _add_candidate(_trim_person_identity_candidate(base))

        leadin_match = re.search(r"\b(?:es\s+ist|das\s+ist|ist)\s+(.+)$", base, flags=re.IGNORECASE)
        if leadin_match:
            _add_candidate(_trim_person_identity_candidate(leadin_match.group(1)))

        _add_candidate(base)

    for candidate in candidates:
        normalized_candidate = _normalize_user_text(candidate)
        if not normalized_candidate or normalized_candidate in hedge_only_candidates:
            continue
        if _is_scene_person_reference(candidate):
            continue
        named_person = _extract_named_person_label(candidate)
        if named_person:
            return named_person

    return None


def _is_scene_person_reference(text: str) -> bool:
    normalized = _normalize_user_text(_clean_runtime_fragment(text) or text)
    if not normalized:
        return False

    if re.search(r"\bander(?:e|er|em|en|es)?\b", normalized):
        return False

    if re.search(r"\b(?:er|sie|ihn|ihm|ihr)\b", normalized):
        return True

    if re.search(r"\b(?:bei|beim|von|durch|mit)\s+(?:ihm|ihr|ihn|sie)\b", normalized):
        return True

    return _is_generic_person_reference(normalized)


def _mentions_scene_person_label(text: str, scene_person: str) -> bool:
    normalized_text = _normalize_user_text(text)
    normalized_scene_person = _normalize_user_text(scene_person)
    if not normalized_text or not normalized_scene_person:
        return False
    return bool(re.search(rf"\b{re.escape(normalized_scene_person)}\b", normalized_text))


def _extract_explicit_name_revelation(text: str) -> str | None:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return None
    if not re.search(r"^(?:er heisst|sie heisst|heisst|das ist|es ist)\b", _normalize_user_text(raw)):
        return None
    return _extract_named_person_label(raw)


def _classify_focus_reference(text: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return "unknown"

    raw = _clean_runtime_fragment(raw) or raw
    normalized = _normalize_user_text(raw)
    if not normalized:
        return "unknown"

    if (
        _matches_any_regex(normalized, GLOBAL_UNCERTAINTY_REGEXES)
        or _matches_any_regex(normalized, GLOBAL_QUESTION_REGEXES)
        or _looks_like_direct_question(raw)
    ):
        return "unknown"

    if _is_acknowledgement_only_reply(raw):
        return "unknown"

    if normalized in {"was", "etwas", "irgendwas"}:
        return "unknown"

    group_markers = [
        "gruppe",
        "freunde",
        "freundinnen",
        "clique",
        "klasse",
        "menschen",
        "leute",
        "kinder",
        "jungs",
        "madchen",
        "eltern",
        "familie",
        "alle",
        "anderen",
        "mehrere",
        "paar",
        "haufen",
    ]
    group_patterns = (
        r"^die aus\b",
        r"^die da\b",
        r"^die anderen\b",
        r"^alle am\b",
        r"^alle die\b",
        r"^ein haufen\b",
        r"^ein paar\b",
    )
    if _contains_any(normalized, group_markers) or any(
        re.search(pattern, normalized) for pattern in group_patterns
    ):
        return "group"

    person_role_markers = [
        "vater",
        "mutter",
        "papa",
        "mama",
        "bruder",
        "schwester",
        "oma",
        "opa",
        "onkel",
        "tante",
        "lehrer",
        "lehrerin",
        "chef",
        "chefin",
        "freund",
        "freundin",
        "arzt",
        "aerztin",
        "therapeut",
        "therapeutin",
        "mann",
        "frau",
        "herr",
        "person",
        "typ",
    ]
    if _contains_any(normalized, person_role_markers):
        return "person"

    other_markers = [
        "rauch",
        "qualm",
        "rauchgeruch",
        "zigarette",
        "zigaretten",
        "geruch",
        "gestank",
        "geschmack",
        "farbe",
        "rot",
        "blau",
        "grun",
        "gruen",
        "gelb",
        "orange",
        "lila",
        "violett",
        "rosa",
        "pink",
        "schwarz",
        "weiss",
        "grau",
        "situation",
        "moment",
        "ereignis",
        "ort",
        "platz",
        "schule",
        "schulhof",
        "pause",
        "zimmer",
        "kuche",
        "kueche",
        "haus",
        "auto",
        "raucherecke",
        "stimme",
        "lachen",
        "druck",
        "blick",
        "verhalten",
        "spannung",
        "gefuhl",
        "gefuehl",
        "raum",
    ]
    if _contains_any(normalized, other_markers):
        return "other"

    stripped = re.sub(
        r"^(?:es ist|das ist|ich nehme|wir nehmen|nehmen wir|ich wuerde|ich wurde|das waere|das ware|die person ist|die person waere|das ist der|das ist die|der|die|das|ein|eine)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    parts = [part for part in stripped.split() if part]
    raw_lower = raw.lower().strip()
    raw_candidate = re.sub(
        r"^(?:es ist|das ist|ich nehme|wir nehmen|nehmen wir|ich wuerde|ich wurde|das waere|das ware|die person ist|die person waere|das ist der|das ist die)\s+",
        "",
        raw_lower,
        flags=re.IGNORECASE,
    ).strip()
    starts_with_article = raw_candidate.startswith(
        ("der ", "die ", "das ", "ein ", "eine ", "mein ", "meine ", "dieser ", "diese ", "dieses ")
    )
    if (
        not starts_with_article
        and 1 <= len(parts) <= 3
        and len("".join(parts)) >= 3
        and all(re.fullmatch(r"[a-zA-ZaeiouyAEIOUY???????]+", part) for part in parts)
    ):
        return "person"

    return "unknown"


def _display_trigger_focus_ref(text: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return "dieser Ausloeser"

    normalized = _normalize_user_text(raw)
    if not normalized:
        return raw

    focus_kind = _classify_focus_reference(raw)
    if focus_kind == "group":
        return "diese Gruppe"
    if focus_kind == "person":
        named_person = _extract_person_identity_label(raw)
        if named_person is None:
            return "diese Person"
        return named_person

    meta_prefixes = (
        "ich glaube ",
        "ich glaub ",
        "ich denke ",
        "ich meine ",
        "vielleicht ",
        "es sind ",
        "das sind ",
        "da sind ",
        "ich sehe ",
        "ich glaub es sind ",
        "ich glaube es sind ",
    )
    group_markers = [
        "gruppe",
        "freunde",
        "freundinnen",
        "clique",
        "klasse",
        "alle",
        "kinder",
        "leute",
        "jungs",
        "madchen",
    ]

    if normalized.startswith(meta_prefixes):
        if _contains_any(normalized, group_markers):
            return "diese Gruppe"
        if focus_kind == "person":
            return "diese Person"
        return "das, was du gerade benannt hast"

    if len(normalized.split()) > 8:
        if _contains_any(normalized, group_markers):
            return "diese Gruppe"
        if focus_kind == "person":
            return "diese Person"
        return "das, was du gerade benannt hast"

    return raw


def _reflect_focus_ref_for_therapist(text: str) -> str:
    reflected = _display_trigger_focus_ref(text)
    possessive_map = {
        "Mein ": "dein ",
        "Meine ": "deine ",
        "Meinen ": "deinen ",
        "Meinem ": "deinem ",
        "Meiner ": "deiner ",
        "Meines ": "deines ",
    }
    for source, target in possessive_map.items():
        if reflected.startswith(source):
            return target + reflected[len(source) :]
    return reflected


def _reflect_named_person_for_therapist(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "diese Person"
    possessive_map = {
        "Mein ": "dein ",
        "Meine ": "deine ",
        "Meinen ": "deinen ",
        "Meinem ": "deinem ",
        "Meiner ": "deiner ",
        "Meines ": "deines ",
    }
    for source, target in possessive_map.items():
        if raw.startswith(source):
            return target + raw[len(source) :]
    return raw


def _reflect_customer_statement_for_therapist(text: str) -> str:
    reflected = re.sub(r"\s+", " ", (text or "").strip())
    if not reflected:
        return ""

    replacements = (
        (r"\bmich\b", "dich"),
        (r"\bmir\b", "dir"),
        (r"\bmein\b", "dein"),
        (r"\bmeine\b", "deine"),
        (r"\bmeinem\b", "deinem"),
        (r"\bmeinen\b", "deinen"),
        (r"\bmeiner\b", "deiner"),
        (r"\bmeines\b", "deines"),
        (r"\bich\b", "du"),
    )
    for pattern, replacement in replacements:
        reflected = re.sub(pattern, replacement, reflected, flags=re.IGNORECASE)

    if re.match(r"(?i)^das\s+(?=(?:er|sie|es|du|sein|seine|ihr|ihre|der|die|das|ein|eine)\b)", reflected):
        reflected = re.sub(r"(?i)^das\s+", "dass ", reflected, count=1)

    conjugation_fixes = (
        (r"\bdu\s+nicht\s+rauche\b", "du nicht rauchst"),
        (r"\bdu\s+rauche\b", "du rauchst"),
        (r"\bdu\s+bin\b", "du bist"),
        (r"\bdu\s+habe\b", "du hast"),
        (r"\bdu\s+hab\b", "du hast"),
        (r"\bdu\s+will\b", "du willst"),
        (r"\bdu\s+kann\b", "du kannst"),
        (r"\bdu\s+sehe\b", "du siehst"),
        (r"\bdu\s+seh\b", "du siehst"),
        (r"\bdu\s+weiss\b", "du weisst"),
        (r"\bdu\s+kenne\b", "du kennst"),
        (r"\bdu\s+fuehle\b", "du fuehlst"),
        (r"\bdu\s+spuere\b", "du spuerst"),
        (r"\bdu\s+spuer\b", "du spuerst"),
    )
    for pattern, replacement in conjugation_fixes:
        reflected = re.sub(pattern, replacement, reflected, flags=re.IGNORECASE)

    return reflected


def _format_trigger_reason_intro(trigger_reason: str) -> str:
    reflected_reason = _reflect_customer_statement_for_therapist(trigger_reason).strip()
    if not reflected_reason:
        return ""

    reason_without_punctuation = reflected_reason.rstrip(".!?")
    normalized = _normalize_user_text(reason_without_punctuation)
    if normalized.startswith("das "):
        reason_without_punctuation = "dass " + reason_without_punctuation[4:]
        normalized = _normalize_user_text(reason_without_punctuation)
    if normalized.startswith("dass "):
        return f"Du hast gerade beschrieben, {reason_without_punctuation}."

    return f"Du hast gerade beschrieben: {reason_without_punctuation}."


def _known_customer_feeling(runtime_slots: dict[str, str]) -> str:
    for key in ("dark_scene_immediate_feeling", "dark_scene_feeling_intensity", "dark_scene_first_spuerbar"):
        feeling = _clean_runtime_fragment(str(runtime_slots.get(key, "")))
        if feeling:
            return feeling
    return "ein ungutes Gefuehl"


def _display_named_person_for_runtime(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "diese Person"

    normalized = _normalize_user_text(raw)
    if normalized.startswith(("ein ", "eine ")) and _classify_focus_reference(raw) == "person":
        return "diese Person"
    if _is_generic_person_reference(raw):
        return "diese Person"
    return raw


def _display_named_person_reference_for_runtime(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "dieser Person"

    normalized = _normalize_user_text(raw)
    if normalized.startswith(("ein ", "eine ")) and _classify_focus_reference(raw) == "person":
        return "dieser Person"
    if _is_generic_person_reference(raw):
        return "dieser Person"
    return raw


def _display_customer_reference_for_runtime(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "dein Gegenueber"
    return raw


def _display_customer_reference_dative_for_runtime(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "deinem Gegenueber"
    return raw


def _mentions_people_or_group(text: str) -> bool:
    return _contains_any(
        text,
        [
            "gruppe",
            "gruppen",
            "person",
            "personen",
            "mensch",
            "menschen",
            "leute",
            "kind",
            "kinder",
            "mann",
            "frau",
            "eltern",
            "familie",
            "clique",
            "klasse",
            "vater",
            "mutter",
            "freunde",
            "freundinnen",
            "jungs",
            "madchen",
        ],
    )


def _strip_scene_report_prefix(text: str) -> str:
    stripped = _normalize_user_text(text)
    if not stripped:
        return ""

    prefixes = (
        r"^ich sehe\b",
        r"^ich seh\b",
        r"^man sieht\b",
        r"^ich kann sehen\b",
        r"^ich nehme wahr\b",
        r"^da ist\b",
        r"^dort ist\b",
        r"^da sind\b",
        r"^dort sind\b",
    )
    changed = True
    while changed and stripped:
        changed = False
        for pattern in prefixes:
            candidate = re.sub(pattern, "", stripped, count=1).strip()
            if candidate != stripped:
                stripped = candidate
                changed = True
    return stripped


def _needs_dark_scene_people_followup(text: str) -> bool:
    focus_ref = _strip_scene_report_prefix(text) or text
    focus_kind = _classify_focus_reference(focus_ref)
    if focus_kind == "group":
        return True
    if focus_kind == "person":
        return _is_generic_person_reference(focus_ref)
    return False


def _has_specific_visual_detail(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False
    generic_replies = {
        "ich sehe was",
        "ich sehe etwas",
        "ich sehe wen",
        "ich sehe jemanden",
        "ich sehe jemand",
        "da ist was",
        "da ist etwas",
        "etwas",
        "was",
        "jemand",
        "eine person",
        "eine gruppe",
    }
    if normalized in generic_replies:
        return False
    return len(normalized.split()) >= 2


def _has_specific_audio_detail(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False
    generic_replies = {
        "ich hore was",
        "ich hore etwas",
        "ich hoere was",
        "ich hoere etwas",
        "da ist ein gerausch",
        "da ist ein geraeusch",
        "ein gerausch",
        "ein geraeusch",
        "gerausch",
        "geraeusch",
        "stimmen",
        "lachen",
    }
    if normalized in generic_replies:
        return False
    return len(normalized.split()) >= 2


def _scene_named_person(runtime_slots: dict[str, str]) -> str:
    return str(runtime_slots.get("scene_named_person") or "").strip()


def _canonicalize_person_focus_ref(
    runtime_slots: dict[str, str],
    raw_focus_ref: str,
) -> tuple[str, str | None]:
    focus_ref = str(raw_focus_ref or "").strip()
    if not focus_ref:
        return "", None

    scene_person = _scene_named_person(runtime_slots)
    if scene_person and _is_scene_person_reference(focus_ref):
        return scene_person, scene_person
    if scene_person and _mentions_scene_person_label(focus_ref, scene_person):
        return scene_person, scene_person

    named_person = _extract_person_identity_label(focus_ref)
    if named_person:
        return named_person, named_person

    return focus_ref, None


def _capture_runtime_slots(
    node_id: str,
    user_text: str | None,
    decision: SemanticModelDecision,
    runtime_slots: dict[str, str],
) -> None:
    access_capture_nodes = {
        "scene_access_followup",
        "dark_scene_perception",
        "dark_scene_mode_clarify",
        "dark_scene_other_sense",
        "dark_scene_first_spuerbar",
    }
    if node_id in {"group_multiple_people_name", "group_multiple_required_name"} and decision.intent == "ready":
        runtime_slots["group_loop_active"] = "true"
    elif node_id in {"group_representative_name", "group_specific_person_name"} and decision.intent == "ready":
        runtime_slots["group_loop_active"] = "false"
    elif node_id == "group_next_person_check" and decision.intent == "no":
        runtime_slots["group_loop_active"] = "false"

    if node_id in access_capture_nodes:
        explicit_access_state = _classify_dark_scene_access_reply(user_text or "")
        runtime_slots["dark_audio_pending"] = "true" if decision.intent == "both" or explicit_access_state == "both" else "false"
        if user_text:
            runtime_slots[node_id] = user_text.strip()
        if user_text and (
            decision.intent in {"visual", "both", "visual_hell", "visual_dark"}
            or explicit_access_state in {"visual", "both"}
        ):
            runtime_slots["dark_scene_visual_detail"] = user_text.strip()
        if user_text and (decision.intent == "audio" or explicit_access_state in {"audio", "both"}):
            runtime_slots["dark_scene_audio_detail"] = user_text.strip()

    if node_id == "dark_known_branch" and decision.intent in {"known", "new"}:
        runtime_slots["dark_known_state"] = decision.intent

    if node_id == "origin_trigger_known_branch" and decision.intent in {"known", "new"}:
        runtime_slots["origin_trigger_known_state"] = decision.intent

    if decision.intent != "ready" or not user_text:
        if node_id == "origin_person_name" and decision.intent == "unknown_person":
            runtime_slots.pop("named_person", None)
        if node_id == "origin_other_target_kind" and decision.intent == "person":
            trigger_focus_ref = str(runtime_slots.get("trigger_focus_ref") or "").strip()
            _, named_person = _canonicalize_person_focus_ref(runtime_slots, user_text)
            if not named_person:
                _, named_person = _canonicalize_person_focus_ref(runtime_slots, trigger_focus_ref)
            if named_person:
                runtime_slots["named_person"] = named_person
                runtime_slots["scene_named_person"] = named_person
            else:
                runtime_slots.pop("named_person", None)
        if node_id == "origin_cause_owner" and decision.intent == "someone_else":
            trigger_focus_ref = str(runtime_slots.get("trigger_focus_ref") or "").strip()
            canonical_focus_ref, named_person = _canonicalize_person_focus_ref(runtime_slots, trigger_focus_ref)
            if named_person:
                runtime_slots["trigger_focus_ref"] = canonical_focus_ref
            if named_person:
                runtime_slots["named_person"] = named_person
                runtime_slots["scene_named_person"] = named_person
            else:
                runtime_slots.pop("named_person", None)
        return

    if node_id == "origin_trigger_source":
        trigger_focus_ref, named_person = _canonicalize_person_focus_ref(runtime_slots, user_text.strip())
        runtime_slots["trigger_focus_ref"] = trigger_focus_ref
        if named_person:
            runtime_slots["named_person"] = named_person
            runtime_slots["scene_named_person"] = named_person
        else:
            runtime_slots.pop("named_person", None)
        return

    if node_id == "origin_self_need":
        runtime_slots["origin_self_need"] = user_text.strip()
        return

    if node_id == "group_person_trigger_reason":
        revealed_named_person = _extract_explicit_name_revelation(user_text)
        if revealed_named_person:
            runtime_slots["named_person"] = revealed_named_person

    if node_id == "group_person_trigger_reason":
        runtime_slots["group_person_trigger_reason"] = user_text.strip()
        return
    if node_id == "group_person_trigger_role":
        runtime_slots["group_person_trigger_role"] = user_text.strip()
        return
    if node_id == "group_person_trigger_core":
        runtime_slots["group_person_trigger_core"] = user_text.strip()
        return
    if node_id == "dark_scene_perception":
        runtime_slots["dark_scene_perception"] = user_text.strip()
        return
    if node_id == "dark_scene_who":
        runtime_slots["dark_scene_who"] = user_text.strip()
        runtime_slots["dark_scene_visual_detail"] = user_text.strip()
        return
    if node_id == "dark_scene_audio_detail":
        runtime_slots["dark_scene_audio_detail"] = user_text.strip()
        runtime_slots["dark_audio_pending"] = "false"
        return
    if node_id == "dark_scene_other_sense":
        runtime_slots["dark_scene_other_sense"] = user_text.strip()
        return
    if node_id == "dark_scene_first_spuerbar":
        runtime_slots["dark_scene_first_spuerbar"] = user_text.strip()
        return
    if node_id == "dark_scene_people_who":
        runtime_slots["dark_scene_people_who"] = user_text.strip()
        runtime_slots["dark_scene_visual_detail"] = user_text.strip()
        named_person = _extract_person_identity_label(user_text)
        if named_person:
            runtime_slots["named_person"] = named_person
            runtime_slots["scene_named_person"] = named_person
        return
    if node_id == "dark_scene_happening":
        runtime_slots["dark_scene_happening"] = user_text.strip()
        return
    if node_id == "dark_scene_age":
        runtime_slots["dark_scene_age"] = user_text.strip()
        return
    if node_id == "dark_scene_feeling_intensity":
        runtime_slots["dark_scene_feeling_intensity"] = user_text.strip()
        if decision.intent == "feeling_and_intensity":
            runtime_slots["dark_scene_immediate_feeling"] = user_text.strip()
        return
    if node_id == "dark_scene_immediate_feeling":
        runtime_slots["dark_scene_immediate_feeling"] = user_text.strip()
        return

    if node_id not in {
        "origin_person_name",
        "group_representative_name",
        "group_specific_person_name",
        "group_multiple_people_name",
        "group_multiple_required_name",
        "group_next_person_name",
    }:
        return

    named_person = _extract_person_identity_label(user_text)
    if named_person:
        runtime_slots["named_person"] = named_person
        if node_id == "origin_person_name":
            runtime_slots["scene_named_person"] = named_person
        processed = [part.strip() for part in runtime_slots.get("processed_people", "").split("|") if part.strip()]
        if named_person not in processed:
            processed.append(named_person)
        runtime_slots["processed_people"] = "|".join(processed)


def _clean_runtime_fragment(text: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return ""

    cleaned = re.sub(
        r"^(?:ich\s+(?:sehe|hoere|spuere|fuehle|rieche|schmecke|nehme(?:\s+es)?\s+wahr|merke|bin|stehe|sitze)|"
        r"da\s+ist|es\s+ist|dort\s+ist)\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned.strip(" \t\n\r.,!?;:")


def _describe_origin_scene_age(age_text: str) -> str:
    cleaned = _clean_runtime_fragment(age_text)
    if not cleaned:
        return ""

    normalized = _normalize_user_text(cleaned)
    if re.fullmatch(r"\d{1,3}", normalized):
        return f"{cleaned} Jahre alt"
    if "jahr" in normalized or "alt" in normalized:
        return cleaned
    return f"etwa {cleaned} alt"


def _join_reflection_clauses(clauses: list[str]) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} und {clauses[1]}"
    return ", ".join(clauses[:-1]) + f" und {clauses[-1]}"


def _build_origin_scene_reflection(runtime_slots: dict[str, str]) -> str:
    age_phrase = _describe_origin_scene_age(str(runtime_slots.get("dark_scene_age", "")))

    visual_phrase = _clean_runtime_fragment(
        str(runtime_slots.get("dark_scene_visual_detail", "") or runtime_slots.get("dark_scene_who", ""))
    )
    people_phrase = _clean_runtime_fragment(str(runtime_slots.get("dark_scene_people_who", "")))
    scene_named_person = _scene_named_person(runtime_slots)
    if people_phrase and scene_named_person:
        people_phrase = scene_named_person
    scene_phrase = ""
    if people_phrase:
        scene_phrase = people_phrase
    elif visual_phrase:
        scene_phrase = visual_phrase
    else:
        for key in ("dark_scene_audio_detail", "dark_scene_perception"):
            scene_phrase = _clean_runtime_fragment(str(runtime_slots.get(key, "")))
            if scene_phrase:
                break

    feeling_phrase = ""
    for key in (
        "dark_scene_immediate_feeling",
        "dark_scene_feeling_intensity",
        "dark_scene_first_spuerbar",
        "dark_scene_other_sense",
    ):
        feeling_phrase = _clean_runtime_fragment(str(runtime_slots.get(key, "")))
        if feeling_phrase:
            break

    clauses: list[str] = []
    if age_phrase:
        clauses.append(f"du dort {age_phrase} bist")
    if scene_phrase:
        clauses.append(f"du dort wahrnimmst: {scene_phrase}")
    if feeling_phrase:
        clauses.append(f"sich dieses ungute Gefuehl dort als {feeling_phrase} zeigt")

    if not clauses:
        return "Du bist jetzt genau an diesem inneren Ursprungspunkt angekommen."

    return "Du bist also jetzt an einem Moment gelandet, in dem " + _join_reflection_clauses(clauses) + "."


def _render_runtime_text(text: str, runtime_slots: dict[str, str]) -> str:
    if not text:
        return text

    render_slots = dict(runtime_slots)
    if "anzahl_zigaretten_pro_tag" not in render_slots:
        cigarettes_per_day = str(render_slots.get("zigaretten_pro_tag", "")).strip()
        render_slots["anzahl_zigaretten_pro_tag"] = cigarettes_per_day or "20"
    if "trigger_focus_ref" in render_slots:
        raw_trigger_focus_ref = render_slots.get("trigger_focus_ref", "")
        display_trigger_focus_ref = _display_trigger_focus_ref(raw_trigger_focus_ref)
        scene_named_person = _scene_named_person(render_slots)
        if (
            display_trigger_focus_ref == "diese Person"
            and scene_named_person
            and _classify_focus_reference(raw_trigger_focus_ref) == "person"
        ):
            display_trigger_focus_ref = _display_named_person_for_runtime(scene_named_person)
        render_slots["trigger_focus_ref"] = display_trigger_focus_ref
    raw_customer_name = render_slots.get("customer_name", "")
    render_slots["customer_ref"] = _display_customer_reference_for_runtime(raw_customer_name)
    render_slots["customer_ref_dat"] = _display_customer_reference_dative_for_runtime(raw_customer_name)
    if raw_customer_name:
        render_slots["customer_name"] = _display_customer_reference_for_runtime(raw_customer_name)
    if "named_person" in render_slots:
        raw_named_person = render_slots.get("named_person", "")
        render_slots["named_person"] = _display_named_person_for_runtime(raw_named_person)
        render_slots["named_person_ref"] = _display_named_person_reference_for_runtime(raw_named_person)
    render_slots["origin_scene_reflection"] = _build_origin_scene_reflection(render_slots)

    class _SafeDict(dict[str, str]):
        def __missing__(self, key: str) -> str:
            if key == "named_person":
                return "diese Person"
            if key == "named_person_ref":
                return "dieser Person"
            if key == "customer_ref":
                return "dein Gegenueber"
            if key == "customer_ref_dat":
                return "deinem Gegenueber"
            if key == "trigger_focus_ref":
                return "diese Gruppe"
            if key == "origin_self_need":
                return "genau das, was damals gefehlt hat"
            return "{" + key + "}"

    return text.format_map(_SafeDict(render_slots))


def _ensure_terminal_punctuation(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return stripped + "."


def _identity_runtime_prompt(node_id: str, runtime_slots: dict[str, str] | None = None) -> str:
    runtime_slots = runtime_slots or {}

    if node_id == "dark_scene_people_who":
        dark_scene_visual_detail = str(
            runtime_slots.get("dark_scene_visual_detail", "") or runtime_slots.get("dark_scene_who", "")
        ).strip()
        focus_ref = _strip_scene_report_prefix(dark_scene_visual_detail) or dark_scene_visual_detail
        focus_kind = _classify_focus_reference(focus_ref)
        if focus_kind == "group":
            return (
                "Kannst du schon etwas genauer erkennen, wer diese Personen oder diese Gruppe sein koennten? "
                "Wenn ja, reicht eine kurze Beschreibung. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
            )
        return (
            "Kannst du schon etwas genauer erkennen, wer diese Person sein koennte? "
            "Wenn ja, reicht der Name oder eine kurze Beschreibung. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
        )

    if node_id == "origin_person_name":
        return (
            "Wenn du bei dieser Person bleibst: Welche Person ist es genau? "
            "Wenn es schon klarer ist, reicht der Name oder eine kurze Beschreibung. "
            "Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
        )

    if node_id in NAMED_PERSON_INPUT_NODES:
        return "Nenn mir bitte direkt den Namen oder eine kurze Beschreibung der Person."

    return ""


def _identity_same_node_reply(node_id: str, runtime_slots: dict[str, str] | None = None, *, acknowledgement: bool) -> str:
    runtime_slots = runtime_slots or {}

    if node_id == "dark_scene_people_who":
        dark_scene_visual_detail = str(
            runtime_slots.get("dark_scene_visual_detail", "") or runtime_slots.get("dark_scene_who", "")
        ).strip()
        focus_ref = _strip_scene_report_prefix(dark_scene_visual_detail) or dark_scene_visual_detail
        focus_kind = _classify_focus_reference(focus_ref)
        if focus_kind == "group":
            if acknowledgement:
                return (
                    "Gut. Dann sag mir bitte jetzt direkt, wer diese Personen oder diese Gruppe sein koennten. "
                    "Wenn ja, reicht eine kurze Beschreibung. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
                )
            return (
                "Ich brauche hier noch etwas genauer, wer diese Personen oder diese Gruppe sein koennten. "
                "Wenn du das schon etwas einordnen kannst, reicht eine kurze Beschreibung. "
                "Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
            )
        if acknowledgement:
            return (
                "Gut. Dann sag mir bitte jetzt direkt, wer diese Person sein koennte, also den Namen oder eine kurze Beschreibung. "
                "Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
            )
        return (
            "Ich brauche hier noch etwas genauer, wer diese Person sein koennte. "
            "Wenn du das schon etwas einordnen kannst, reicht der Name oder eine kurze Beschreibung. "
            "Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
        )

    if node_id == "origin_person_name":
        if acknowledgement:
            return (
                "Gut. Dann sag mir bitte jetzt einfach den Namen oder eine kurze Beschreibung der Person. "
                "Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
            )
        return (
            "Ich brauche hier noch etwas genauer, welche Person es ist. "
            "Wenn es schon klarer ist, reicht der Name oder eine kurze Beschreibung. "
            "Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist."
        )

    if node_id in NAMED_PERSON_INPUT_NODES:
        return "Ich brauche hier direkt den Namen oder eine kurze Beschreibung der Person."

    return ""


def _identity_unclear_reply_variants(
    node_id: str,
    runtime_slots: dict[str, str] | None = None,
) -> list[str]:
    runtime_slots = runtime_slots or {}

    if node_id == "dark_scene_people_who":
        dark_scene_visual_detail = str(
            runtime_slots.get("dark_scene_visual_detail", "") or runtime_slots.get("dark_scene_who", "")
        ).strip()
        focus_ref = _strip_scene_report_prefix(dark_scene_visual_detail) or dark_scene_visual_detail
        focus_kind = _classify_focus_reference(focus_ref)
        if focus_kind == "group":
            return [
                "Ich brauche hier noch etwas genauer, wer diese Personen oder diese Gruppe sein koennten. Wenn du das schon etwas einordnen kannst, reicht eine kurze Beschreibung. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
                "Bleib bei deinem ersten Eindruck: Wer koennte diese Gruppe oder diese Personen sein? Eine kurze Beschreibung reicht. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
                "Wenn du noch einen Moment hinschaust: Wird etwas deutlicher, wer diese Personen oder diese Gruppe sein koennten? Eine kurze Beschreibung reicht. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
            ]
        return [
            "Ich brauche hier noch etwas genauer, wer diese Person sein koennte. Wenn du das schon etwas einordnen kannst, reicht der Name oder eine kurze Beschreibung. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
            "Bleib bei deinem ersten Eindruck: Wer koennte diese Person sein? Der Name oder eine kurze Beschreibung reicht. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
            "Wenn du noch einen Moment hinschaust: Wird etwas deutlicher, wer diese Person sein koennte? Der Name oder eine kurze Beschreibung reicht. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
        ]

    if node_id == "origin_person_name":
        return [
            "Ich brauche hier noch etwas genauer, welche Person es ist. Wenn es schon klarer ist, reicht der Name oder eine kurze Beschreibung. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
            "Bleib bei dieser Person und sag mir bitte, wer es genau sein koennte. Der Name oder eine kurze Beschreibung reicht. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
            "Wenn du die Person noch einen Moment wahrnimmst: Wird deutlicher, wer es ist? Der Name oder eine kurze Beschreibung reicht. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
        ]

    if node_id in NAMED_PERSON_INPUT_NODES:
        return [
            "Ich brauche hier direkt den Namen oder eine kurze Beschreibung der Person.",
            "Bleib noch einen Moment bei der Person und sag mir dann direkt den Namen oder eine kurze Beschreibung.",
            "Wenn es etwas klarer wird, reicht direkt der Name oder eine kurze Beschreibung der Person.",
        ]

    return []


def _render_runtime_question(node_id: str, runtime_slots: dict[str, str]) -> str:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        return ""

    base_question = _render_runtime_text(spec.question_text, runtime_slots)
    named_person = runtime_slots.get("named_person", "diese Person")
    named_person_ref = _display_named_person_reference_for_runtime(named_person)
    raw_trigger_focus_ref = runtime_slots.get("trigger_focus_ref", "").strip()
    trigger_focus_ref = _reflect_focus_ref_for_therapist(raw_trigger_focus_ref)
    scene_named_person = _scene_named_person(runtime_slots)
    if (
        trigger_focus_ref == "diese Person"
        and scene_named_person
        and _classify_focus_reference(raw_trigger_focus_ref) == "person"
    ):
        trigger_focus_ref = _reflect_named_person_for_therapist(scene_named_person)
    trigger_reason = _ensure_terminal_punctuation(runtime_slots.get("group_person_trigger_reason", ""))
    trigger_role = _ensure_terminal_punctuation(runtime_slots.get("group_person_trigger_role", ""))
    dark_scene_visual_detail = str(
        runtime_slots.get("dark_scene_visual_detail", "") or runtime_slots.get("dark_scene_who", "")
    ).strip()

    if node_id == "dark_scene_people_who" and dark_scene_visual_detail:
        focus_ref = _strip_scene_report_prefix(dark_scene_visual_detail) or dark_scene_visual_detail
        focus_kind = _classify_focus_reference(focus_ref)
        if focus_kind == "person" and _is_generic_person_reference(focus_ref):
            return _identity_runtime_prompt(node_id, runtime_slots) or base_question
        if focus_kind == "group":
            return _identity_runtime_prompt(node_id, runtime_slots) or base_question

    if node_id == "origin_trigger_known_branch" and trigger_focus_ref:
        return (
            f"Du hast gerade beschrieben, dass {trigger_focus_ref} in diesem Moment stark wirkt. "
            "Wenn du jetzt bei dem Gefuehl bleibst, das dort auftaucht: "
            "Kennst du dieses Gefuehl schon aus frueheren Momenten, oder ist es hier in dieser Szene zum ersten Mal da?"
        )

    if node_id == "origin_scene_relevance" and trigger_focus_ref:
        return (
            f"Du hast gerade beschrieben, dass {trigger_focus_ref} in diesem Moment stark wirkt. "
            "Wenn du dabei bleibst: Ist in genau dieser Szene etwas, das wir hier anschauen oder loesen sollten, "
            "oder merkst du eher, dass dich dieses Gefuehl noch weiter zu etwas Frueherem fuehrt?"
        )

    if node_id == "origin_cause_owner" and trigger_focus_ref:
        return (
            f"Du hast gerade beschrieben, dass {trigger_focus_ref} dieses ungute Gefuehl in diesem Moment ausloest. "
            f"Wenn du genau hier in dieser Szene weiterarbeitest: Liegt der eigentliche Grund eher in etwas in dir, "
            f"das dadurch beruehrt wird, oder ist {trigger_focus_ref} selbst eher der Kern dieses Themas?"
        )

    if node_id == "origin_other_target_kind" and trigger_focus_ref:
        return (
            f"Du hast gerade beschrieben, dass {trigger_focus_ref} in diesem Moment stark wirkt. "
            "Wenn du dabei bleibst: Weist das eher auf eine Gruppe, eher auf eine bestimmte Person "
            "oder eher auf etwas anderes in dieser Situation?"
        )

    if node_id == "origin_self_need" and trigger_focus_ref:
        return (
            f"Du hast gerade beschrieben, dass {trigger_focus_ref} in diesem Moment stark wirkt. "
            "Wenn du jetzt noch tiefer in genau diese Szene spuerst: "
            "Was hat dir dort am meisten gefehlt, oder was haettest du in diesem Moment am meisten gebraucht?"
        )

    if node_id == "group_person_trigger_role" and trigger_reason:
        return (
            f"{_format_trigger_reason_intro(trigger_reason)} "
            f"Geht dieses ungute Gefuehl direkt von {named_person} aus, "
            f"oder steht {named_person} eher stellvertretend fuer etwas Groesseres, das in dieser Situation wirksam ist?"
        )

    if node_id == "group_person_trigger_core" and trigger_reason:
        normalized_role = _normalize_user_text(trigger_role)
        if trigger_role and _contains_any(
            normalized_role,
            ["gruppe", "alle", "von allen", "mehrere", "alle anderen"],
        ):
            return (
                f"{_format_trigger_reason_intro(trigger_reason)} "
                f"Und du hast eingeordnet, dass {named_person} dabei eher fuer diese Gruppe steht. "
                f"Weisst du schon, warum {named_person} in dieser Situation so reagiert oder wofuer das in dieser Gruppe steht? "
                "Wenn das noch nicht klar ist, ist auch das eine stimmige Rueckmeldung."
            )
        return (
            f"{_format_trigger_reason_intro(trigger_reason)} "
            f"Weisst du schon, warum {named_person} in dieser Situation so reagiert oder was dahinterliegt? "
            "Wenn das noch nicht klar ist, ist auch das eine stimmige Rueckmeldung."
        )

    if node_id == "person_switch_sees_impact":
        customer_ref = _display_customer_reference_for_runtime(str(runtime_slots.get("customer_name", "")))
        customer_ref_dat = _display_customer_reference_dative_for_runtime(str(runtime_slots.get("customer_name", "")))
        feeling = _known_customer_feeling(runtime_slots)
        return (
            f"Siehst du, dass {customer_ref} vor dir steht und dass dein Verhalten in diesem Moment "
            f"bei {customer_ref_dat} gerade {feeling} ausloest?"
        )

    if node_id == "person_switch_heard_customer":
        customer_ref = _display_customer_reference_for_runtime(str(runtime_slots.get("customer_name", "")))
        customer_ref_dat = _display_customer_reference_dative_for_runtime(str(runtime_slots.get("customer_name", "")))
        feeling = _known_customer_feeling(runtime_slots)
        return (
            f"Hast du verstanden, dass dein Verhalten bei {customer_ref_dat} gerade {feeling} ausloest "
            f"und dass genau dieser Moment spaeter fuer {customer_ref} mit dem Rauchen verknuepft werden kann?"
        )

    if node_id == "person_switch_why":
        customer_ref_dat = _display_customer_reference_dative_for_runtime(str(runtime_slots.get("customer_name", "")))
        named_person_ref = _display_named_person_reference_for_runtime(str(runtime_slots.get("named_person", "")))
        return (
            f"Bleib in der Perspektive von {named_person_ref}. Was ist in dir in diesem Moment los, "
            f"wie geht es dir dort, und warum reagierst du {customer_ref_dat} gegenueber genau so?"
        )

    if node_id == "person_switch_ready":
        return f"Bist du bereit, jetzt in die Perspektive von {named_person_ref} zu wechseln?"

    return base_question


_CUSTOMER_FACING_META_TAIL_PATTERNS = (
    re.compile(
        "(?i)"
        "(?:\\s+|^)"
        "(?:zur(?:ue|\u00fc|u)ck)"
        "\\s+(?:bei|zu|zum|zur)"
        "\\s+(?:(?:diesem|dem|aktuellen|dieser|der)\\s+)?"
        "(?:schritt|knoten|phase|stelle|frage)"
        "\\.?\\s*$"
    ),
    re.compile(
        "(?i)"
        "(?:[\\s,;:.\\-–—]+|^)"
        "aber\\s+beschreibe\\s+nur\\s+das\\s+unmittelbare\\s+gef(?:ue|ü|u)hl,?"
        "\\s*nicht\\s+gedanken\\s+oder\\s+ursachen"
        "\\.?\\s*$"
    ),
)


def _sanitize_customer_facing_answer(text: str) -> str:
    sanitized = (text or "").strip()
    if not sanitized:
        return ""

    for pattern in _CUSTOMER_FACING_META_TAIL_PATTERNS:
        sanitized = pattern.sub("", sanitized).strip()

    return re.sub(r"\s{2,}", " ", sanitized).strip()


def _acknowledgement_only_same_node_reply(
    node_id: str,
    customer_message: str,
    runtime_slots: dict[str, str] | None = None,
) -> str:
    if not _is_acknowledgement_only_reply(customer_message):
        return ""

    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        return ""

    runtime_slots = runtime_slots or {}

    identity_reply = _identity_same_node_reply(node_id, runtime_slots, acknowledgement=True)
    if identity_reply:
        return identity_reply

    answer_hint = _empty_input_answer_hint(node_id, spec, runtime_slots).strip()
    if answer_hint:
        return f"Gut. Dann gib mir bitte jetzt direkt diese Rueckmeldung. {answer_hint}"

    current_question = _render_runtime_question(node_id, runtime_slots).strip()
    if current_question:
        return f"Gut. Dann antworte mir bitte jetzt direkt auf diese Frage: {current_question}"

    return ""


def _answer_question_in_context(
    openai_backend_or_client: Any,
    model: str,
    node_id: str,
    customer_question: str,
    *,
    session_context: str = "",
    runtime_slots: dict[str, str] | None = None,
    live_api_budget: LiveApiCallBudget | None = None,
) -> str:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        return ""

    if openai_backend_or_client is None:
        return _sanitize_customer_facing_answer(spec.same_node_replies.get("question", ""))

    system_prompt = (
        "Du beantwortest eine konkrete Rueckfrage innerhalb einer gefuehrten Hypnosesitzung. "
        "Bleib ruhig, knapp, klar und therapeutisch passend. "
        "Beantworte nur die konkrete Frage des Kunden im Kontext des aktuellen Knotens. "
        "Aendere den Flow nicht, fuehre keine neue Phase ein und bleib beim aktuellen Schritt. "
        "Halte dich inhaltlich strikt an den Hinweis zum Knoten. Dieser Hinweis ist Ground Truth dafuer, worum es an dieser Stelle geht. "
        "Wenn deine allgemeine Hypnoseannahme und der Knotenhinweis voneinander abweichen, gilt immer der Knotenhinweis. "
        "Vermeide generische Hypnose-Floskeln, esoterische Behauptungen und Heilsversprechen. "
        "Verwende keine Formulierungen wie 'Heilung', wenn die Frage das nicht selbst anspricht. "
        "Wenn nach Hypnose allgemein gefragt wird, erklaere schlicht, dass die Person der Stimme folgt, die Aufmerksamkeit nach innen lenkt, ansprechbar bleibt und wir Schritt fuer Schritt durch den Prozess gehen. "
        "Wenn sinnvoll, leite sanft wieder zur eigentlichen Frage zurueck, aber ohne Meta-Formulierungen ueber Schritte, Knoten, Phasen oder den Prozess selbst. "
        "Vermeide insbesondere Saetze wie 'zurueck bei diesem Schritt', 'am aktuellen Knoten' oder aehnliche technische Prozesssprache. "
        "Gib nur reinen Antworttext zurueck."
    )
    user_payload = {
        "node_id": node_id,
        "node_goal": spec.node_goal,
        "current_question": _render_runtime_question(node_id, runtime_slots or {}),
        "node_hint": QUESTION_ANSWER_HINTS.get(node_id, ""),
        "session_context": session_context,
        "customer_question": customer_question,
    }
    try:
        if isinstance(openai_backend_or_client, OpenAISemanticBackend):
            answer = openai_backend_or_client.generate_text_reply(
                system_prompt=system_prompt,
                user_payload=user_payload,
                live_api_budget=live_api_budget,
            )
        else:
            if live_api_budget is not None:
                live_api_budget.consume(f"{node_id}:question_reply")
            completion = openai_backend_or_client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
                ],
            )
            answer = (completion.choices[0].message.content or "").strip()
        if answer:
            sanitized_answer = _sanitize_customer_facing_answer(answer)
            if sanitized_answer:
                return sanitized_answer
    except Exception:
        pass

    return _sanitize_customer_facing_answer(spec.same_node_replies.get("question", ""))


def _should_use_contextual_same_node_reply(node_id: str, decision: SemanticModelDecision) -> bool:
    return node_id in CONTEXTUAL_SAME_NODE_REPLY_NODES and decision.intent in {"unclear", "support_needed"}


def _contextual_same_node_reply(
    openai_backend_or_client: Any,
    model: str,
    node_id: str,
    decision: SemanticModelDecision,
    *,
    customer_message: str,
    clarify_attempt: int = 0,
    session_context: str = "",
    runtime_slots: dict[str, str] | None = None,
    silence: bool = False,
    live_api_budget: LiveApiCallBudget | None = None,
) -> str:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        return ""

    if node_id == "hell_light_level" and _looks_like_same_scene_report(customer_message):
        return (
            "Gut. Dann kann es sein, dass du wieder am gleichen oder an einem sehr aehnlichen Moment gelandet bist. "
            "Ich brauche dort von dir nur kurz die Einordnung, ob diese Szene eher hell, eher dunkel oder gemischt wirkt."
        )

    fallback_reply = _dynamic_same_node_reply(
        node_id,
        decision,
        clarify_attempt,
        spec.same_node_replies.get(decision.intent, ""),
        runtime_slots=runtime_slots,
    )
    rendered_fallback = _sanitize_customer_facing_answer(_render_runtime_text(fallback_reply, runtime_slots or {}))

    acknowledgement_reply = _acknowledgement_only_same_node_reply(
        node_id,
        customer_message,
        runtime_slots,
    )
    if acknowledgement_reply:
        return acknowledgement_reply

    if silence:
        return rendered_fallback

    if openai_backend_or_client is None or not _should_use_contextual_same_node_reply(node_id, decision):
        return rendered_fallback

    system_prompt = (
        "Du formulierst eine kurze, konkrete Anschlussantwort innerhalb derselben therapeutischen Frage. "
        "Der Flow bleibt im aktuellen Knoten. "
        "Deine Aufgabe ist nicht, den naechsten Zweig zu waehlen, sondern die aktuelle Frage so zu praezisieren, "
        "dass der Kunde versteht, worauf er achten soll. "
        "Nutze den Knotenhinweis als Ground Truth dafuer, worum es an dieser Stelle geht. "
        "Wenn bereits inhaltliche Hinweise aus dem vorherigen Schritt vorliegen, knuepfe konkret daran an. "
        "Wenn die Person benannt wurde, verwende den Namen statt generischer Platzhalter. "
                "Vermeide befehlige Standardfloskeln und leere Warteformeln. "
        "Vermeide leere Meta-Saetze und antworte stattdessen inhaltlich konkret. "
        "Wenn keine neue Kundenaussage vorliegt, formuliere dieselbe Frage einfach klarer und greife bereits bekannte Hinweise auf. "
        "Bleib ruhig, knapp, therapeutisch passend und bei maximal drei Saetzen. "
        "Gib nur reinen Antworttext zurueck."
    )
    user_payload = {
        "node_id": node_id,
        "node_goal": spec.node_goal,
        "current_question": _render_runtime_question(node_id, runtime_slots or {}),
        "node_hint": QUESTION_ANSWER_HINTS.get(node_id, ""),
        "session_context": session_context,
        "runtime_slots": runtime_slots or {},
        "customer_message": customer_message,
        "current_intent": decision.intent,
        "clarify_attempt": max(0, int(clarify_attempt)),
        "silence": silence,
    }
    try:
        if isinstance(openai_backend_or_client, OpenAISemanticBackend):
            answer = openai_backend_or_client.generate_text_reply(
                system_prompt=system_prompt,
                user_payload=user_payload,
                live_api_budget=live_api_budget,
            )
        else:
            if live_api_budget is not None:
                live_api_budget.consume(f"{node_id}:contextual_reply")
            completion = openai_backend_or_client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
                ],
            )
            answer = (completion.choices[0].message.content or "").strip()
        if answer:
            sanitized_answer = _sanitize_customer_facing_answer(answer)
            if sanitized_answer:
                return sanitized_answer
    except Exception:
        pass

    return rendered_fallback


def _local_session_decision(
    node_id: str,
    customer_message: str,
    *,
    restrict_scope: bool = True,
) -> SemanticModelDecision | None:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        return None

    meta_decision = _detect_global_meta_intent(node_id, spec, customer_message)
    if meta_decision is not None:
        return meta_decision

    normalized = _normalize_user_text(customer_message)
    if not normalized:
        return None

    intents = set(spec.allowed_intents)
    positive_patterns = [
        "ja",
        "ok",
        "okay",
        "alles klar",
        "passt",
        "passt so",
        "bin bereit",
        "ich bin bereit",
        "ich bin soweit",
        "ich habe eine situation",
        "ich hab eine situation",
        "hab eine situation",
        "soweit",
        "bereit",
        "gefunden",
        "geschlossen",
        "klar",
        "deutlich spurbar",
        "deutlich spuerbar",
        "spurb ar",
        "spuerbar",
        "ich sehe es",
        "ich bin wieder im sessel",
        "ich bin wieder im magischen sessel",
    ]
    negative_patterns = [
        "nein",
        "no",
        "noe",
        "nee",
        "nope",
        "noch nicht",
        "gar nicht",
        "nicht soweit",
        "nicht gefunden",
        "nicht spurbar",
        "nicht spuerbar",
        "augen noch offen",
        "nicht klar",
    ]

    node_semantic_decision = _detect_node_semantic_predecision(node_id, customer_message)
    if node_semantic_decision is not None:
        return node_semantic_decision

    if restrict_scope and not (
        node_id.startswith("session_")
        or node_id.startswith("phase4_common_")
        or node_id in STRICT_PHASE4_YES_NO_NODES
        or node_id in STRICT_PHASE4_EXPLANATION_NODES
    ):
        return None

    if "provided_scale" in intents:
        digit_match = re.search(r"\b(10|[1-9])\b", normalized)
        if digit_match:
            return _make_transition_decision_for_intent(node_id, "provided_scale", "Lokaler Parser: Skalenwert erkannt.")
        word_to_number = {
            "eins": 1,
            "zwei": 2,
            "drei": 3,
            "vier": 4,
            "funf": 5,
            "fuenf": 5,
            "sechs": 6,
            "sieben": 7,
            "acht": 8,
            "neun": 9,
            "zehn": 10,
        }
        if any(word in normalized.split() for word in word_to_number):
            return _make_transition_decision_for_intent(node_id, "provided_scale", "Lokaler Parser: Skalenwort erkannt.")
        return None

    if "yes" in intents:
        if _contains_any(normalized, positive_patterns):
            return _make_transition_decision_for_intent(node_id, "yes", "Lokaler Parser: positive Bestaetigung erkannt.")
        if _contains_any(normalized, negative_patterns):
            return _make_transition_decision_for_intent(node_id, "no", "Lokaler Parser: negative Bestaetigung erkannt.")
        return None

    if "continue" in intents:
        if _contains_any(normalized, positive_patterns):
            return _make_transition_decision_for_intent(node_id, "continue", "Lokaler Parser: Weitergehen erkannt.")
        return None

    if "ready" in intents:
        if node_id in STRICT_PHASE4_EXPLANATION_NODES and (
            _contains_any(normalized, positive_patterns) or _contains_any(normalized, negative_patterns)
        ):
            if node_id == "group_person_trigger_core" and _is_group_person_trigger_core_short_status_reply(customer_message):
                return _make_transition_decision_for_intent(
                    node_id,
                    "ready",
                    "Lokaler Parser: kurzer Statushinweis an diesem Knoten als brauchbare Rueckmeldung erkannt.",
                )
            if _is_acknowledgement_only_reply(customer_message):
                return _make_transition_decision_for_intent(
                    node_id,
                    "unclear",
                    "Lokaler Parser: bestaetigende Meta-Antwort ohne inhaltliche Erklaerung erkannt.",
                )
            if not _is_bare_status_reply(customer_message):
                if not restrict_scope:
                    return _make_transition_decision_for_intent(
                        node_id,
                        "ready",
                        "Lokaler Parser: inhaltliche freie Rueckmeldung an Erklaerungsknoten erkannt.",
                    )
                return None
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Lokaler Parser: fuer diesen Knoten wird eine inhaltliche Erklaerung statt eines blossen Ja/Nein benoetigt.",
            )
        if node_id in STRICT_PHASE4_EXPLANATION_NODES:
            if node_id == "group_person_trigger_core" and _is_group_person_trigger_core_short_status_reply(customer_message):
                return _make_transition_decision_for_intent(
                    node_id,
                    "ready",
                    "Lokaler Parser: kurzer Statushinweis an diesem Knoten als brauchbare Rueckmeldung erkannt.",
                )
            if _is_generic_ready_placeholder_for_node(node_id, customer_message):
                return _make_transition_decision_for_intent(
                    node_id,
                    "unclear",
                    "Lokaler Parser: bestaetigende oder zu vage Rueckmeldung ohne tragfaehige Erklaerung erkannt.",
                )
            if not restrict_scope:
                return _make_transition_decision_for_intent(
                    node_id,
                    "ready",
                    "Lokaler Parser: inhaltliche freie Rueckmeldung an Erklaerungsknoten erkannt.",
                )
            return None
        if _is_generic_ready_placeholder_for_node(node_id, customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Lokaler Parser: noch keine ausreichend konkrete inhaltliche Rueckmeldung erkannt.",
            )
        return _make_transition_decision_for_intent(node_id, "ready", "Lokaler Parser: sinnvolle freie Rueckmeldung erkannt.")

    return None


def _local_router_predecision(node_id: str, customer_message: str) -> SemanticModelDecision | None:
    normalized = _normalize_user_text(customer_message)
    if _is_acknowledgement_only_reply(customer_message) or _looks_like_nonanswer_noise(customer_message):
        local_decision = _local_session_decision(node_id, customer_message, restrict_scope=False)
        if local_decision is not None and local_decision.intent != "ready":
            return local_decision

    spec = get_node_spec(node_id)
    if isinstance(spec, SemanticNodeSpec):
        if "ready" in spec.allowed_intents and _is_generic_ready_placeholder_for_node(node_id, customer_message):
            return _make_transition_decision_for_intent(
                node_id,
                "unclear",
                "Lokaler Guardrail: generischer Platzhalter ohne ausreichenden Inhalt erkannt; erst weiterklaeren statt inhaltlich zu raten.",
            )

    if node_id == "hell_light_level":
        if _is_explicit_dark_scene_description(normalized):
            return _make_transition_decision_for_intent(
                node_id,
                "darker_or_other",
                "Lokaler Guardrail: eindeutige Dunkel-Einordnung erkannt.",
            )
        if _is_explicit_hell_scene_description(normalized):
            return _make_transition_decision_for_intent(
                node_id,
                "hell_light",
                "Lokaler Guardrail: eindeutige Hell-Einordnung erkannt.",
            )
        if _is_explicit_mixed_light_description(normalized):
            return _make_transition_decision_for_intent(
                node_id,
                "both",
                "Lokaler Guardrail: gemischte Hell-Dunkel-Einordnung erkannt.",
            )
    if node_id == "group_source_kind":
        explicit_kind = _classify_group_source_kind_reply(customer_message)
        if explicit_kind is not None:
            return _make_transition_decision_for_intent(
                node_id,
                explicit_kind,
                "Lokaler Guardrail: explizite Gruppeneinordnung erkannt.",
            )
    return None


def _dynamic_same_node_reply(
    node_id: str,
    decision: SemanticModelDecision,
    attempt: int,
    fallback_reply: str,
    runtime_slots: dict[str, str] | None = None,
) -> str:
    attempt = max(0, attempt)
    runtime_slots = runtime_slots or {}

    customer_ref = _display_customer_reference_for_runtime(str(runtime_slots.get("customer_name", "")))
    customer_ref_dat = _display_customer_reference_dative_for_runtime(str(runtime_slots.get("customer_name", "")))
    known_feeling = _known_customer_feeling(runtime_slots)

    if decision.intent == "no":
        if node_id == "person_switch_sees_impact":
            replies = [
                f"Schau noch einen Moment genauer hin: {customer_ref} steht vor dir, und dein Verhalten loest bei {customer_ref_dat} gerade {known_feeling} aus. Wird dir das jetzt klarer? Ein Ja oder Nein reicht.",
                f"Bleib noch einen Moment in dieser Perspektive. {customer_ref} ist direkt vor dir, und dein Verhalten trifft {customer_ref_dat} gerade spuerbar. Wird dir das jetzt klarer? Ein Ja oder Nein reicht.",
                f"Nimm nur diesen einen Punkt wahr: Was du gerade tust, loest bei {customer_ref_dat} etwas aus. Wenn du das jetzt klarer siehst, reicht ein Ja. Wenn noch nicht, ein Nein.",
            ]
            return replies[min(attempt, len(replies) - 1)]

        if node_id == "person_switch_heard_customer":
            replies = [
                f"Bleib noch einen Moment in dieser Perspektive: Dein Verhalten loest bei {customer_ref_dat} gerade {known_feeling} aus, und genau dieser Moment kann spaeter fuer {customer_ref} mit dem Rauchen verknuepft werden. Wird dir das jetzt klarer? Ein Ja oder Nein reicht.",
                f"Schau noch einen Moment weiter hin und nimm wahr, was bei {customer_ref_dat} gerade geschieht. Wenn dir klarer wird, dass dein Verhalten dort gerade {known_feeling} ausloest, reicht ein Ja. Wenn noch nicht, ein Nein.",
                f"Es geht nur um diese eine Einordnung: Wird dir jetzt klarer, was dein Verhalten gerade bei {customer_ref_dat} ausloest und warum dieser Moment spaeter wichtig werden kann? Ein Ja oder Nein reicht.",
            ]
            return replies[min(attempt, len(replies) - 1)]

        if node_id == "session_phase2_ready":
            replies = [
                "Sobald du bereit bist, genuegt ein kurzes Ja.",
                "Kein Problem. Wir gehen erst weiter, wenn es fuer dich stimmt. Wenn du magst, kannst du kurz dazusagen, was du vorher noch brauchst.",
                "Dann bleiben wir noch einen Augenblick hier. Sobald du soweit bist, genuegt ein Ja. Wenn dich noch etwas zurueckhaelt, kannst du es kurz benennen.",
            ]
            return replies[min(attempt, len(replies) - 1)]

        replies = [
            fallback_reply,
            "Sobald es fuer dich passt, gehen wir hier weiter.",
            "Kein Problem. Wir gehen erst weiter, wenn es fuer dich stimmig ist. Wenn noch etwas offen ist, kannst du dazu kurz Rueckmeldung geben.",
        ]
        return replies[min(attempt, len(replies) - 1)]

    if decision.intent == "support_needed":
        if node_id in {"dark_scene_feeling_intensity", "dark_scene_immediate_feeling"}:
            replies = [
                fallback_reply,
                "Bleib nicht tiefer in der Szene. Spuer erst den Stuhl unter dir und nimm den Atem wieder etwas ruhiger wahr. Wenn das Gefuehl wieder greifbar, aber gut aushaltbar ist, gehen wir von dort aus weiter.",
                "Wir muessen das jetzt nicht verstaerken. Gib dem Gefuehl erst so viel Abstand, dass du wieder gut bei dir bleiben kannst. Sobald es sich etwas ruhiger anfuehlt, reicht eine kurze Rueckmeldung dazu.",
            ]
            return replies[min(attempt, len(replies) - 1)]
        replies = [
            fallback_reply,
            "Bleib noch einen Moment bei dir. Wenn du etwas brauchst, reicht eine kurze Rueckmeldung dazu.",
            "Nimm dir die Zeit, die du gerade brauchst. Sobald es wieder stimmig ist, gehen wir von hier aus weiter.",
        ]
        return replies[min(attempt, len(replies) - 1)]

    if decision.intent == "unclear":
        if node_id == "person_switch_sees_impact":
            replies = [
                f"Es reicht deine erste Einordnung. Wird dir jetzt klarer, dass dein Verhalten bei {customer_ref_dat} gerade {known_feeling} ausloest? Ein Ja oder Nein reicht.",
                f"Bleib bei diesem einen Punkt: {customer_ref} steht vor dir, und dein Verhalten loest dort gerade etwas aus. Wenn dir das jetzt klarer wird, reicht ein Ja oder Nein.",
                f"Ich brauche hier nur eine kurze Einordnung dazu, ob du die Wirkung deines Verhaltens auf {customer_ref} jetzt klarer siehst. Ein Ja oder Nein reicht.",
            ]
            return replies[min(attempt, len(replies) - 1)]

        if node_id == "person_switch_heard_customer":
            replies = [
                f"Es reicht deine erste Einordnung. Wird dir jetzt klarer, dass dein Verhalten bei {customer_ref_dat} gerade {known_feeling} ausloest? Ein Ja oder Nein reicht.",
                f"Bleib noch einen Moment in dieser Perspektive und nimm nur wahr, was bei {customer_ref_dat} gerade ankommt. Wenn dir das jetzt klarer wird, reicht ein Ja oder Nein.",
                f"Ich brauche hier nur eine kurze Einordnung dazu, ob dir jetzt klarer wird, was dein Verhalten bei {customer_ref_dat} ausloest. Ein Ja oder Nein reicht.",
            ]
            return replies[min(attempt, len(replies) - 1)]

        if node_id == "person_switch_why":
            named_person_ref = _display_named_person_reference_for_runtime(str(runtime_slots.get("named_person", "")))
            replies = [
                f"Bleib in der Perspektive von {named_person_ref}. Nimm nur kurz wahr, was in dir in diesem Moment los ist und warum du {customer_ref_dat} gegenueber so reagierst. Dein erster Eindruck reicht.",
                f"Es geht hier nicht um Ja oder Nein, sondern nur um den ersten Eindruck aus der Perspektive von {named_person_ref}: Was passiert in dir dort, und warum handelst du {customer_ref_dat} gegenueber so?",
                f"Bleib noch einen Moment in der Perspektive von {named_person_ref} und beschreibe dann kurz, wie es dir dort geht und was dahintersteht, dass du so reagierst.",
            ]
            return replies[min(attempt, len(replies) - 1)]

        spec = get_node_spec(node_id)
        if isinstance(spec, SemanticNodeSpec):
            identity_replies = _identity_unclear_reply_variants(node_id, runtime_slots)
            if identity_replies:
                replies = identity_replies
                return replies[min(attempt, len(replies) - 1)]
            answer_hint = _empty_input_answer_hint(node_id, spec, runtime_slots or {})
            replies = [
                fallback_reply,
                (
                    f"Bleib ruhig bei deinem ersten Eindruck. {answer_hint}"
                    if answer_hint
                    else "Bleib ruhig bei deinem ersten Eindruck. Eine kurze Einordnung reicht voellig."
                ),
                (
                    "Wenn es gerade noch nicht ganz klar ist, ist das in Ordnung. "
                    f"Nimm nur wahr, was sich als Erstes zeigt. {answer_hint}"
                    if answer_hint
                    else "Wenn es gerade noch nicht ganz klar ist, ist das in Ordnung. Nimm nur wahr, was sich als Erstes zeigt. Eine kurze Rueckmeldung zu deinem ersten Eindruck genuegt."
                ),
            ]
            return replies[min(attempt, len(replies) - 1)]

    return fallback_reply


def _is_phase12_node(node_id: str) -> bool:
    return node_id.startswith("session_phase1_") or node_id.startswith("session_phase2_")


def _max_silence_attempts_before_outro(node_id: str) -> int:
    if node_id in EXPLORATORY_SCENE_SILENCE_NODES:
        return MAX_SILENCE_ATTEMPTS_BEFORE_OUTRO + 2
    return MAX_SILENCE_ATTEMPTS_BEFORE_OUTRO


def _is_reengagement_signal(text: str) -> bool:
    normalized = _normalize_user_text(text)
    if not normalized:
        return False
    return _matches_any_regex(normalized, GLOBAL_REENGAGEMENT_REGEXES)


def _silence_timeout_seconds(node_id: str, attempt: int) -> float:
    raw = (os.getenv("SESSION_SANDBOX_SILENCE_SECONDS") or "").strip()
    try:
        if raw:
            value = float(raw)
            return max(2.0, min(30.0, value))
    except ValueError:
        pass

    schedule = PHASE12_SILENCE_TIMEOUTS if _is_phase12_node(node_id) else PHASE36_SILENCE_TIMEOUTS
    try:
        value = schedule[min(max(0, int(attempt)), len(schedule) - 1)]
    except ValueError:
        value = schedule[0]
    return max(2.0, min(30.0, value))


def _timed_input(prompt: str, timeout_seconds: float) -> str | None:
    print(prompt, end="", flush=True)
    buffer: list[str] = []
    deadline = time.time() + timeout_seconds

    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwche()

            if char in {"\r", "\n"}:
                print()
                return "".join(buffer).strip()

            if char == "\003":
                raise KeyboardInterrupt

            if char in {"\b", "\x7f"}:
                if buffer:
                    buffer.pop()
                    print(" \b", end="", flush=True)
                continue

            if char in {"\x00", "\xe0"}:
                _ = msvcrt.getwch()
                continue

            buffer.append(char)
            deadline = float("inf")
            continue

        if time.time() >= deadline:
            print()
            return None

        time.sleep(0.05)


def _make_transition_decision_for_intent(node_id: str, intent: str, reason: str) -> SemanticModelDecision:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise ValueError(f"node '{node_id}' is not semantic")
    route = spec.routing_rules[intent]
    payload = {
        "intent": intent,
        "action": route["action"],
        "next_node": route["next_node"],
        "confidence": 1.0,
        "reason": reason,
    }
    return validate_semantic_decision(node_id, payload)


def _build_local_intent_prompt(
    node_id: str,
    customer_message: str,
    *,
    session_context: str = "",
    runtime_slots: dict[str, str] | None = None,
) -> str:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise ValueError(f"node '{node_id}' is not semantic")

    payload = {
        "task": "routing_intent",
        "node_id": node_id,
        "question_text": _render_runtime_question(node_id, runtime_slots or {}),
        "node_goal": spec.node_goal,
        "allowed_intents": list(spec.allowed_intents),
        "customer_message": customer_message,
        "session_context": session_context,
        "runtime_slots": runtime_slots or {},
        "rules": [
            "Gib nur ein JSON-Objekt aus.",
            "Erlaube nur das Feld `intent`.",
            "intent muss exakt einer der allowed_intents sein.",
            "Keine Synonyme oder freien Labels erfinden.",
        ],
    }
    intro = (
        "Du bist ein strukturiertes Intent-Klassifikationsmodell fuer eine deterministische Hypnose-Runtime. "
        "Klassifiziere die Nutzerantwort nur innerhalb des aktuellen Knotens und gib nur das JSON-Objekt zurueck."
    )
    return intro + "\n\n" + json.dumps(payload, ensure_ascii=True, indent=2)


def _local_intent_payload_to_decision(
    node_id: str,
    payload: dict[str, Any],
    *,
    source: str = "local_intent_router",
) -> tuple[dict[str, Any], SemanticModelDecision]:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise ValueError(f"node '{node_id}' is not semantic")
    intent = str(payload.get("intent") or "").strip()
    if intent not in spec.allowed_intents:
        raise ValueError(
            f"Lokaler Intent-Router lieferte ungueltigen Intent fuer {node_id}: {intent!r}"
        )
    decision = _make_transition_decision_for_intent(
        node_id,
        intent,
        f"Lokaler Intent-Router: Intent `{intent}` erkannt.",
    )
    parsed = {
        "intent": decision.intent,
        "action": decision.action,
        "next_node": decision.next_node,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "source": source,
    }
    return parsed, decision


def _local_intent_fallback_decision(
    node_id: str,
    *,
    raw_response: str = "",
    error: str = "",
) -> tuple[dict[str, Any], SemanticModelDecision]:
    detail = error.strip() or "ungueltige Router-Antwort"
    decision = _make_transition_decision_for_intent(
        node_id,
        "unclear",
        f"Lokaler Intent-Router lieferte keine brauchbare kanonische Antwort ({detail}); sicherer Fallback auf 'unclear'.",
    )
    parsed = {
        "intent": decision.intent,
        "action": decision.action,
        "next_node": decision.next_node,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "source": "local_intent_router_fallback",
    }
    if raw_response:
        parsed["raw_response"] = raw_response
    if error:
        parsed["error"] = error
    return parsed, decision


def _handle_silence(
    node_id: str,
    attempt: int,
    runtime_slots: dict[str, str] | None = None,
) -> tuple[SemanticModelDecision | None, str]:
    if attempt >= _max_silence_attempts_before_outro(node_id):
        return None, INACTIVITY_OUTRO_TEXT

    if node_id in AUTO_CONTINUE_ON_SILENCE:
        decision = _make_transition_decision_for_intent(
            node_id,
            "continue",
            "Keine Antwort innerhalb des Zeitfensters; dieser Knoten darf still weiterlaufen.",
        )
        return decision, AUTO_CONTINUE_ON_SILENCE[node_id]

    if node_id == "hell_light_level" and attempt >= 0:
        decision = _make_transition_decision_for_intent(
            "scene_access_followup",
            "unclear",
            "Keine Antwort auf die erste Hell-Dunkel-Frage; wechsle direkt in die V2-Folgefrage fuer sichtbare, hoerbare, spuerbare oder fehlende Wahrnehmung.",
        )
        return decision, ""

    if node_id == "scene_access_followup" and attempt >= 1:
        decision = _make_transition_decision_for_intent(
            node_id,
            "nothing_yet",
            "Keine Antwort auch nach der Folgefrage; wechsle in die Body-Bridge-Weiterfuehrung.",
        )
        return decision, ""

    if node_id == "dark_scene_perception" and attempt == 0:
        decision = validate_semantic_decision(
            node_id,
            {
                "intent": "repeat",
                "action": "repeat",
                "next_node": "dark_scene_perception",
                "confidence": 1.0,
                "reason": "Keine Antwort auf die offene Wahrnehmungsfrage; bleibe zuerst im aktuellen Knoten und frage weicher nach.",
            },
        )
        reply = _empty_input_reply(node_id, attempt, runtime_slots)
        return decision, reply

    decision = _decision_for_empty_input(node_id)
    reply = _empty_input_reply(node_id, attempt, runtime_slots)
    return decision, reply


def _route_runtime_next_node(
    node_id: str,
    decision: SemanticModelDecision,
    runtime_slots: dict[str, str],
    user_text: str | None = None,
) -> str:
    normalized_user_text = _normalize_user_text(user_text or "")
    trigger_focus_ref = str(runtime_slots.get("trigger_focus_ref") or "").strip()
    named_person = str(runtime_slots.get("named_person") or "").strip()
    scene_named_person = _scene_named_person(runtime_slots)
    access_reclassify_nodes = {
        "scene_access_followup",
        "dark_scene_other_sense",
        "dark_scene_first_spuerbar",
    }
    access_detail_nodes = {
        "scene_access_followup",
        "dark_scene_perception",
        "dark_scene_other_sense",
        "dark_scene_first_spuerbar",
    }

    def _origin_person_target_node(*, include_user_text: bool) -> str:
        derived_name = named_person or scene_named_person or _extract_person_identity_label(trigger_focus_ref)
        if not derived_name and include_user_text:
            derived_name = _extract_person_identity_label(user_text or "")
        return "origin_person_branch_intro" if derived_name else "origin_person_name"

    if node_id == "origin_cause_owner" and decision.intent == "someone_else":
        focus_kind = _classify_focus_reference(trigger_focus_ref)
        if focus_kind == "person":
            return _origin_person_target_node(include_user_text=False)
        if focus_kind == "group":
            return "group_branch_intro"

    if (
        node_id == "origin_trigger_source"
        and decision.intent == "ready"
        and runtime_slots.get("dark_known_state") == "new"
    ):
        return "origin_cause_owner"

    if node_id == "origin_other_target_kind":
        explicit_kind = _classify_origin_target_kind_reply(user_text or "")
        if explicit_kind == "person":
            return _origin_person_target_node(include_user_text=True)
        if explicit_kind == "group":
            return "group_branch_intro"
        if explicit_kind == "other":
            return "origin_self_resolution_intro"
        focus_kind = _classify_focus_reference(trigger_focus_ref)
        if focus_kind == "group" and decision.next_node in {"origin_person_branch_intro", "origin_person_name"}:
            return "group_branch_intro"
        if focus_kind == "other" and decision.next_node in {"origin_person_branch_intro", "origin_person_name", "group_branch_intro"}:
            return "origin_self_resolution_intro"
        if focus_kind == "person" and decision.next_node in {"origin_person_branch_intro", "origin_person_name"}:
            return _origin_person_target_node(include_user_text=True)

    if node_id in access_reclassify_nodes:
        if _is_explicit_dark_scene_description(normalized_user_text):
            return "dark_scene_perception"
        if _is_explicit_hell_scene_description(normalized_user_text):
            return "hell_feel_branch"
        if _is_explicit_mixed_light_description(normalized_user_text):
            return "dark_follow_darker_intro"

    candidate_next_node = decision.next_node
    if node_id in {"dark_scene_other_sense", "dark_scene_first_spuerbar"}:
        explicit_access_state = _classify_dark_scene_access_reply(user_text or "")
        if explicit_access_state in {"visual", "both"} or _looks_like_visual_fragment_reply(user_text or ""):
            candidate_next_node = "dark_scene_who"
        elif explicit_access_state == "audio":
            candidate_next_node = "dark_scene_audio_detail"

    if node_id in access_detail_nodes:
        if candidate_next_node == "dark_scene_who":
            visual_detail = _normalize_user_text(
                runtime_slots.get("dark_scene_visual_detail", "") or runtime_slots.get(node_id, "")
            )
            if _needs_dark_scene_people_followup(visual_detail):
                return "dark_scene_people_who"
            if _has_specific_visual_detail(visual_detail):
                if runtime_slots.get("dark_audio_pending") == "true":
                    return "dark_scene_audio_detail"
                return "dark_scene_age"
        if candidate_next_node == "dark_scene_audio_detail":
            audio_detail = _normalize_user_text(
                runtime_slots.get("dark_scene_audio_detail", "") or runtime_slots.get(node_id, "")
            )
            if _has_specific_audio_detail(audio_detail):
                return "dark_scene_age"
        if candidate_next_node != decision.next_node:
            return candidate_next_node

    if node_id == "dark_scene_who":
        if runtime_slots.get("dark_audio_pending") == "true":
            return "dark_scene_audio_detail"
        visual_detail = _normalize_user_text(
            runtime_slots.get("dark_scene_visual_detail", "") or runtime_slots.get("dark_scene_who", "")
        )
        if _needs_dark_scene_people_followup(visual_detail):
            return "dark_scene_people_who"
    if node_id == "dark_scene_audio_detail":
        visual_detail = _normalize_user_text(
            runtime_slots.get("dark_scene_visual_detail", "") or runtime_slots.get("dark_scene_who", "")
        )
        if _needs_dark_scene_people_followup(visual_detail):
            return "dark_scene_people_who"
    if (
        node_id == "person_switch_self_understands"
        and decision.next_node == "group_resolution_complete"
        and runtime_slots.get("group_loop_active") == "true"
    ):
        return "group_next_person_check"
    return decision.next_node


def call_semantic_node(
    openai_backend_or_client: Any,
    model: str,
    node_id: str,
    customer_message: str,
    *,
    clarify_attempt: int = 0,
    session_context: str = "",
    runtime_slots: dict[str, str] | None = None,
    local_intent_router: LocalIntentRouter | None = None,
    live_api_budget: LiveApiCallBudget | None = None,
    trace_logger: DecisionTraceLogger | None = None,
) -> tuple[dict[str, Any], Any]:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise ValueError(f"node '{node_id}' is not semantic")

    trace_events: list[dict[str, Any]] = []
    _emit_decision_trace(
        trace_events,
        trace_logger,
        "call_start",
        node_id=node_id,
        customer_message=customer_message,
        clarify_attempt=clarify_attempt,
        has_runtime_slots=bool(runtime_slots),
        has_local_intent_router=local_intent_router is not None,
        semantic_backend_type=type(openai_backend_or_client).__name__ if openai_backend_or_client is not None else None,
        model=model,
    )

    local_decision = _local_session_decision(node_id, customer_message)
    if local_decision is not None:
        _emit_decision_trace(
            trace_events,
            trace_logger,
            "local_session_decision_hit",
            intent=local_decision.intent,
            action=local_decision.action,
            next_node=local_decision.next_node,
            confidence=local_decision.confidence,
            reason=local_decision.reason,
        )
        return (
            _attach_decision_trace(
                {
                    "intent": local_decision.intent,
                    "action": local_decision.action,
                    "next_node": local_decision.next_node,
                    "confidence": local_decision.confidence,
                    "reason": local_decision.reason,
                    "source": "local_parser",
                },
                trace_events,
            ),
            local_decision,
        )

    if local_intent_router is not None:
        predecision = _local_router_predecision(node_id, customer_message)
        if predecision is not None:
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "local_router_predecision_hit",
                intent=predecision.intent,
                action=predecision.action,
                next_node=predecision.next_node,
                confidence=predecision.confidence,
                reason=predecision.reason,
            )
            return (
                _attach_decision_trace(
                    {
                        "intent": predecision.intent,
                        "action": predecision.action,
                        "next_node": predecision.next_node,
                        "confidence": predecision.confidence,
                        "reason": predecision.reason,
                        "source": "local_parser",
                    },
                    trace_events,
                ),
                predecision,
            )
        prompt = _build_local_intent_prompt(
            node_id,
            customer_message,
            session_context=session_context,
            runtime_slots=runtime_slots,
        )
        _emit_decision_trace(
            trace_events,
            trace_logger,
            "local_intent_prompt_built",
            prompt=prompt,
        )
        raw_text = ""
        try:
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "local_intent_router_call_start",
                node_id=node_id,
            )
            raw_text = local_intent_router.infer_intent(prompt)
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "local_intent_router_raw_response",
                raw_response=raw_text,
            )
            parsed = parse_json_object(raw_text)
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "local_intent_router_parsed_response",
                parsed=parsed,
            )
            decision_payload, decision = _local_intent_payload_to_decision(node_id, parsed)
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "local_intent_decision_validated",
                parsed=decision_payload,
                intent=decision.intent,
                action=decision.action,
                next_node=decision.next_node,
                confidence=decision.confidence,
            )
            return _attach_decision_trace(decision_payload, trace_events), decision
        except Exception as exc:
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "local_intent_router_exception",
                raw_response=raw_text,
                error=str(exc),
            )
            fallback_payload, fallback_decision = _local_intent_fallback_decision(
                node_id,
                raw_response=raw_text,
                error=str(exc),
            )
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "local_intent_fallback_decision",
                parsed=fallback_payload,
                intent=fallback_decision.intent,
                action=fallback_decision.action,
                next_node=fallback_decision.next_node,
                confidence=fallback_decision.confidence,
            )
            return _attach_decision_trace(fallback_payload, trace_events), fallback_decision

    payload = build_request(
        node_id,
        customer_message,
        clarify_attempt=clarify_attempt,
        session_context=session_context,
    )
    rendered_question = _render_runtime_question(node_id, runtime_slots or {})
    if rendered_question:
        payload["runtime_question"] = rendered_question
    if runtime_slots:
        payload["runtime_slots"] = dict(runtime_slots)
    _emit_decision_trace(
        trace_events,
        trace_logger,
        "semantic_request_prepared",
        runtime_question=rendered_question,
        payload=payload,
    )
    messages = [
        {
            "role": "system",
            "content": (
                spec.system_prompt
                + (
                    f"\n\nKonkrete Laufzeitfrage fuer diesen Durchlauf:\n\"{rendered_question}\""
                    if rendered_question
                    else ""
                )
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    effective_system_prompt = messages[0]["content"]
    if isinstance(openai_backend_or_client, OpenAISemanticBackend):
        effective_system_prompt = openai_backend_or_client.compose_semantic_system_prompt(
            spec.system_prompt,
            runtime_question=rendered_question,
        )
    _emit_decision_trace(
        trace_events,
        trace_logger,
        "semantic_system_prompt_prepared",
        system_prompt=effective_system_prompt,
    )
    try:
        if isinstance(openai_backend_or_client, OpenAISemanticBackend):
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "semantic_backend_infer_start",
                backend_profile=getattr(openai_backend_or_client, "profile", None),
                backend_api_mode=getattr(openai_backend_or_client, "api_mode", None),
                model=getattr(openai_backend_or_client, "model", model),
            )
            parsed = openai_backend_or_client.infer_semantic_json(
                system_prompt=effective_system_prompt,
                user_payload=payload,
                live_api_budget=live_api_budget,
            )
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "semantic_backend_infer_result",
                parsed=parsed,
            )
        elif openai_backend_or_client is None:
            raise RuntimeError("Kein OpenAI-Backend fuer semantisches Routing konfiguriert.")
        else:
            if live_api_budget is not None:
                live_api_budget.consume(f"{node_id}:response_format")
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "semantic_chat_completion_start",
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
            )
            completion = openai_backend_or_client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = completion.choices[0].message.content or ""
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "semantic_chat_completion_raw_response",
                raw_response=content,
            )
            parsed = parse_json_object(content)
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "semantic_chat_completion_parsed_response",
                parsed=parsed,
            )
    except Exception as exc:
        _emit_decision_trace(
            trace_events,
            trace_logger,
            "semantic_primary_exception",
            error=str(exc),
        )
        if openai_backend_or_client is None:
            raise
        if isinstance(openai_backend_or_client, OpenAISemanticBackend):
            client = openai_backend_or_client.client
            model = openai_backend_or_client.model
            if live_api_budget is not None:
                live_api_budget.consume(f"{node_id}:chat_fallback")
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "semantic_chat_fallback_start",
                model=model,
                messages=messages,
            )
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                **openai_backend_or_client.chat_completion_options(),
            )
            content = completion.choices[0].message.content or ""
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "semantic_chat_fallback_raw_response",
                raw_response=content,
            )
            parsed = parse_json_object(content)
            _emit_decision_trace(
                trace_events,
                trace_logger,
                "semantic_chat_fallback_parsed_response",
                parsed=parsed,
            )
        else:
            raise
    repaired = repair_semantic_payload(node_id, parsed)
    _emit_decision_trace(
        trace_events,
        trace_logger,
        "semantic_payload_repaired",
        parsed_before_repair=parsed,
        parsed_after_repair=repaired,
    )
    decision = validate_semantic_decision(node_id, repaired)
    _emit_decision_trace(
        trace_events,
        trace_logger,
        "semantic_decision_validated",
        intent=decision.intent,
        action=decision.action,
        next_node=decision.next_node,
        confidence=decision.confidence,
        reason=decision.reason,
    )
    return _attach_decision_trace(repaired, trace_events), decision


def _resolve_semantic_backends(
    semantic_provider: str,
    *,
    local_intent_adapter_dir: str | None = None,
    local_intent_base_model: str | None = None,
) -> tuple[OpenAISemanticBackend | None, str, LocalIntentRouter | None]:
    if semantic_provider == "ft":
        backend = resolve_ft_backend()
        return backend, backend.model, None
    if semantic_provider == "openai-router":
        backend = resolve_openai_router_backend()
        return backend, backend.model, None
    if semantic_provider == "local-intent":
        router = resolve_local_intent_router(
            local_intent_adapter_dir,
            base_model=local_intent_base_model,
        )
        return None, "", router
    raise ValueError(f"Unsupported semantic provider: {semantic_provider}")


def run_batch(
    scenario_name: str,
    *,
    debug_model: bool = False,
    semantic_provider: str = "ft",
    local_intent_adapter_dir: str | None = None,
    local_intent_base_model: str | None = None,
    live_api_budget: LiveApiCallBudget | None = None,
    initial_runtime_slots: dict[str, str] | None = None,
) -> int:
    openai_backend, model, local_intent_router = _resolve_semantic_backends(
        semantic_provider,
        local_intent_adapter_dir=local_intent_adapter_dir,
        local_intent_base_model=local_intent_base_model,
    )
    trace_logger = _build_stdout_trace_logger(debug_model)
    scenario = SCENARIOS.get(scenario_name) or []
    if not scenario:
        available = ", ".join(sorted(SCENARIOS))
        print(f"Unknown scenario '{scenario_name}'. Available: {available}")
        return 1

    session_context = ""
    runtime_slots: dict[str, str] = dict(initial_runtime_slots or {})
    failures = 0
    for case in scenario:
        entry_script = _render_runtime_text(maybe_render_entry_script(case.node_id), runtime_slots)
        if entry_script:
            print(f"ENTRY {case.node_id}: {entry_script}")
        parsed, decision = call_semantic_node(
            openai_backend,
            model,
            case.node_id,
            case.customer_message,
            session_context=session_context,
            runtime_slots=runtime_slots,
            local_intent_router=local_intent_router,
            live_api_budget=live_api_budget,
            trace_logger=trace_logger,
        )
        _capture_runtime_slots(case.node_id, case.customer_message, decision, runtime_slots)
        routed_next_node = _route_runtime_next_node(case.node_id, decision, runtime_slots, case.customer_message)
        script_reply = _render_runtime_text(script_reply_for_decision(case.node_id, decision), runtime_slots)
        success = routed_next_node == case.expected_next_node
        if not success:
            failures += 1
        print(f"NODE: {case.node_id}")
        print(f"INPUT: {case.customer_message}")
        if debug_model:
            print(f"MODEL: {json.dumps(parsed, ensure_ascii=False)}")
        print(f"DECISION: intent={decision.intent} action={decision.action} next={routed_next_node} confidence={decision.confidence}")
        if script_reply:
            print(f"SCRIPT_REPLY: {script_reply}")
        if routed_next_node in available_node_ids():
            next_spec = get_node_spec(routed_next_node)
            if isinstance(next_spec, ScriptNodeSpec):
                next_script, _ = render_script_node(routed_next_node)
                print(f"NEXT_SCRIPT: {_render_runtime_text(next_script, runtime_slots)}")
            elif next_spec.question_text:
                print(f"NEXT_QUESTION: {_render_runtime_question(routed_next_node, runtime_slots)}")
        print(f"EXPECTED_NEXT: {case.expected_next_node}")
        print(f"RESULT: {'PASS' if success else 'FAIL'}")
        print("-" * 100)
        session_context = script_reply or session_context
    return 1 if failures else 0


def run_interactive(
    start_node: str,
    *,
    debug_model: bool = False,
    speak: bool = False,
    semantic_provider: str = "openai-router",
    local_intent_adapter_dir: str | None = None,
    local_intent_base_model: str | None = None,
    live_api_budget: LiveApiCallBudget | None = None,
    initial_runtime_slots: dict[str, str] | None = None,
) -> int:
    openai_backend, model, local_intent_router = _resolve_semantic_backends(
        semantic_provider,
        local_intent_adapter_dir=local_intent_adapter_dir,
        local_intent_base_model=local_intent_base_model,
    )
    trace_logger = _build_stdout_trace_logger(debug_model)
    node_id = start_node
    session_context = ""
    runtime_slots: dict[str, str] = dict(initial_runtime_slots or {})
    clarify_attempts: dict[str, int] = {}
    pending_script_prefixes: dict[str, str] = {}
    last_node_id: str | None = None
    reprint_current_question = True
    session_started_at = time.monotonic()
    soft_limit_logged = False
    hard_limit_logged = False
    pending_session_end_status: tuple[str, str] | None = None

    print("Gefuehrte Hypnosesitzung")
    print("Zum Beenden kannst du jederzeit 'exit' eingeben.")
    if speak and not _ensure_tts_ready():
        print(f"[TTS deaktiviert] {_TTS_ERROR}")

    while True:
        current_spec = get_node_spec(node_id)
        if isinstance(current_spec, ScriptNodeSpec):
            script_text, next_node = render_script_node(node_id)
            prefix = pending_script_prefixes.pop(node_id, "")
            if prefix:
                script_text = f"{prefix}\n\n{script_text}"
            post_block_pause_ms = 0
            if speak and next_node in available_node_ids():
                next_spec = get_node_spec(next_node)
                if not isinstance(next_spec, ScriptNodeSpec):
                    post_block_pause_ms = _tts_post_block_pause_ms()
            _print_paged_block(
                "SCRIPT",
                _render_runtime_text(script_text, runtime_slots),
                speak=speak,
                post_block_pause_ms=post_block_pause_ms,
            )
            if next_node is None:
                if pending_session_end_status is not None:
                    status, reason = pending_session_end_status
                    print(f"[SESSION_END]\nstatus={status} reason={reason}\n")
                else:
                    print("Session sandbox finished.")
                return 0
            last_node_id = node_id
            node_id = next_node
            reprint_current_question = True
            continue

        if node_id != last_node_id and current_spec.entry_script:
            _print_paged_block(
                "SCRIPT",
                _render_runtime_text(current_spec.entry_script, runtime_slots),
                speak=speak,
                post_block_pause_ms=_tts_post_block_pause_ms() if speak else 0,
            )

        if reprint_current_question:
            _print_paged_block("FRAGE", _render_runtime_question(node_id, runtime_slots), speak=speak)

        elapsed_seconds = time.monotonic() - session_started_at
        if elapsed_seconds >= SESSION_ABSOLUTE_LIMIT_SECONDS:
            pending_script_prefixes["session_phase6_outro"] = SESSION_WALLCLOCK_END_TEXT
            pending_session_end_status = ("ended_by_inactivity", "absolute_wallclock_limit_safe_outro")
            last_node_id = node_id
            node_id = "session_phase6_outro"
            reprint_current_question = True
            continue
        if elapsed_seconds >= SESSION_HARD_LIMIT_SECONDS and not hard_limit_logged:
            hard_limit_logged = True
        if elapsed_seconds >= SESSION_SOFT_LIMIT_SECONDS and not soft_limit_logged:
            soft_limit_logged = True

        clarify_attempt = clarify_attempts.get(node_id, 0)
        silence_timeout = _silence_timeout_seconds(node_id, clarify_attempt)
        user_text = _timed_input("> ", silence_timeout)
        if user_text is not None and user_text.lower() in {"exit", "quit", "abbrechen"}:
            print("Session sandbox stopped.")
            return 0
        if user_text is not None and clarify_attempt > 0 and _is_reengagement_signal(user_text):
            reengagement_reply = "Gut. Dann bleiben wir genau hier. Ich stelle dir die Frage noch einmal."
            clarify_attempts[node_id] = 0
            _print_paged_block("SCRIPT", reengagement_reply, speak=speak)
            session_context = reengagement_reply
            reprint_current_question = True
            last_node_id = node_id
            continue

        parsed: dict[str, Any] | None = None
        scale_value: str | None = None
        if user_text is None or not user_text:
            decision, script_reply = _handle_silence(node_id, clarify_attempt, runtime_slots)
            if decision is None:
                rendered_outro_prefix = _render_runtime_text(script_reply, runtime_slots) if script_reply else ""
                if rendered_outro_prefix:
                    pending_script_prefixes["session_phase6_outro"] = rendered_outro_prefix
                pending_session_end_status = ("ended_by_inactivity", "no_customer_response_safe_outro")
                clarify_attempts[node_id] = 0
                reprint_current_question = True
                last_node_id = node_id
                node_id = "session_phase6_outro"
                continue
            routed_next_node = _route_runtime_next_node(node_id, decision, runtime_slots)
            print(
                f"[DECISION]\nintent={decision.intent} action={decision.action} next={routed_next_node} confidence={decision.confidence}\n"
            )
            if script_reply:
                rendered_silence_reply = _render_runtime_text(script_reply, runtime_slots)
                _print_paged_block("SCRIPT", rendered_silence_reply, speak=speak)
                session_context = rendered_silence_reply
            if routed_next_node == node_id:
                clarify_attempts[node_id] = clarify_attempt + 1
                reprint_current_question = False
            else:
                clarify_attempts[node_id] = 0
                reprint_current_question = True
                last_node_id = node_id
                node_id = routed_next_node
                continue
            last_node_id = node_id
            continue
        else:
            scale_value = _extract_scale_value(user_text)
            parsed, decision = call_semantic_node(
                openai_backend,
                model,
                node_id,
                user_text,
                clarify_attempt=clarify_attempt,
                session_context=session_context,
                runtime_slots=runtime_slots,
                local_intent_router=local_intent_router,
                live_api_budget=live_api_budget,
                trace_logger=trace_logger,
            )
        if debug_model and parsed is not None:
            _print_paged_block("MODEL", json.dumps(parsed, ensure_ascii=False, indent=2))
        _capture_runtime_slots(node_id, user_text, decision, runtime_slots)
        routed_next_node = _route_runtime_next_node(node_id, decision, runtime_slots, user_text)
        print(
            f"[DECISION]\nintent={decision.intent} action={decision.action} next={routed_next_node} confidence={decision.confidence}\n"
        )

        question_announcement = bool(
            user_text
            and decision.intent == "question"
            and _is_question_announcement(user_text)
        )

        if decision.intent == "question":
            if question_announcement:
                script_reply = _question_announcement_reply(clarify_attempt)
            else:
                script_reply = _answer_question_in_context(
                    openai_backend,
                    model,
                    node_id,
                    user_text,
                    session_context=session_context,
                    runtime_slots=runtime_slots,
                    live_api_budget=live_api_budget,
                )
        else:
            script_reply = script_reply_for_decision(node_id, decision)
            if routed_next_node == node_id and _should_use_contextual_same_node_reply(node_id, decision):
                script_reply = _contextual_same_node_reply(
                    openai_backend,
                    model,
                    node_id,
                    decision,
                    customer_message=user_text,
                    clarify_attempt=clarify_attempt,
                    session_context=session_context,
                    runtime_slots=runtime_slots,
                    live_api_budget=live_api_budget,
                )
            elif script_reply and routed_next_node == node_id:
                script_reply = _dynamic_same_node_reply(
                    node_id,
                    decision,
                    clarify_attempt,
                    script_reply,
                    runtime_slots=runtime_slots,
                )
        script_reply = _render_runtime_text(script_reply, runtime_slots)
        if script_reply:
            _print_paged_block("SCRIPT", script_reply, speak=speak)
            session_context = script_reply

        if routed_next_node == node_id:
            clarify_attempts[node_id] = clarify_attempt + 1
            reprint_current_question = False if script_reply else not question_announcement
        else:
            scale_prefix = _scale_confirmation_prefix(routed_next_node, scale_value)
            if scale_prefix:
                pending_script_prefixes[routed_next_node] = scale_prefix
            clarify_attempts[node_id] = 0
            reprint_current_question = True

        last_node_id = node_id
        node_id = routed_next_node


def estimate_batch_live_api_calls(scenario_name: str) -> int:
    return len(SCENARIOS.get(scenario_name) or [])


def _build_initial_runtime_slots(customer_name: str | None) -> dict[str, str]:
    runtime_slots: dict[str, str] = {}
    resolved_customer_name = str(customer_name or os.getenv("SESSION_SANDBOX_CUSTOMER_NAME", "") or "Nico").strip()
    if resolved_customer_name:
        runtime_slots["customer_name"] = resolved_customer_name
    return runtime_slots


def main() -> int:
    parser = argparse.ArgumentParser(description="Session sandbox runner outside the protected audio-chat")
    parser.add_argument("--node", default="session_phase1_intro", help="Start node id")
    parser.add_argument("--batch", action="store_true", help="Run predefined scenario")
    parser.add_argument("--scenario", default="phase2_core", help="Scenario name for --batch")
    parser.add_argument(
        "--debug-model",
        action="store_true",
        help="Show full routing trace including prompts, raw model responses, repaired payloads and final JSON.",
    )
    parser.add_argument("--speak", action="store_true", help="Read scripts and questions aloud via cloud TTS")
    parser.add_argument(
        "--semantic-provider",
        choices=["ft", "openai-router", "local-intent"],
        default="openai-router",
        help="Semantischer Routing-Pfad: altes FT, aktueller OpenAI-Router oder lokaler Intent-Router.",
    )
    parser.add_argument(
        "--local-intent-adapter-dir",
        default=str(DEFAULT_LOCAL_INTENT_ADAPTER_DIR),
        help="Pfad zum lokal trainierten Intent-Adapter.",
    )
    parser.add_argument(
        "--local-intent-base-model",
        help="Optionaler Override fuer das Base-Modell des lokalen Intent-Adapters.",
    )
    parser.add_argument("--live-api", action="store_true", help="Erlaubt echte OpenAI-Laeufe fuer Batch oder interaktiv.")
    parser.add_argument("--max-api-calls", type=int, help="Verpflichtendes hartes Limit fuer Live-OpenAI-Calls.")
    parser.add_argument(
        "--approval-file",
        default=str(DEFAULT_APPROVAL_FILE),
        help="Pfad zur Repo-Freigabedatei fuer Live-OpenAI.",
    )
    parser.add_argument(
        "--customer-name",
        help="Optionaler Kundenname fuer personalisierte Perspektivwechsel-Fragen. Alternativ per SESSION_SANDBOX_CUSTOMER_NAME.",
    )
    args = parser.parse_args()
    initial_runtime_slots = _build_initial_runtime_slots(args.customer_name)

    if args.node not in available_node_ids():
        available = ", ".join(sorted(available_node_ids()))
        raise SystemExit(f"Unknown start node '{args.node}'. Available: {available}")

    live_api_budget: LiveApiCallBudget | None = None
    if args.semantic_provider in {"ft", "openai-router"}:
        if args.batch:
            estimated_calls = estimate_batch_live_api_calls(args.scenario)
            if not args.live_api:
                print("Live-OpenAI ist fuer API-Batchs standardmaessig blockiert.")
                print(f"Geschaetzte Live-Calls fuer dieses Szenario: {estimated_calls}")
                print(
                    "Zum bewussten Live-Lauf braucht es: OPENAI_LIVE_API_ALLOWED=1, "
                    "eine gueltige backend/live_api_approval.json und --live-api --max-api-calls <n>."
                )
                return 0
            live_api_budget = build_live_api_budget(
                "run_session_sandbox.py",
                estimated_calls=estimated_calls,
                requested_max_calls=args.max_api_calls,
                approval_file=args.approval_file,
            )
            print(live_api_budget.summary())
        else:
            if not args.live_api:
                print("Live-OpenAI ist fuer interaktive API-Laeufe standardmaessig blockiert.")
                print(
                    "Zum bewussten Live-Lauf braucht es: OPENAI_LIVE_API_ALLOWED=1, "
                    "eine gueltige backend/live_api_approval.json und --live-api --max-api-calls <n>."
                )
                return 0
            live_api_budget = build_live_api_budget(
                "run_session_sandbox.py",
                estimated_calls=args.max_api_calls or 1,
                requested_max_calls=args.max_api_calls,
                approval_file=args.approval_file,
            )
            print(live_api_budget.summary())

    if args.batch:
        return run_batch(
            args.scenario,
            debug_model=args.debug_model,
            semantic_provider=args.semantic_provider,
            local_intent_adapter_dir=args.local_intent_adapter_dir,
            local_intent_base_model=args.local_intent_base_model,
            live_api_budget=live_api_budget,
            initial_runtime_slots=initial_runtime_slots,
        )
    return run_interactive(
        args.node,
        debug_model=args.debug_model,
        speak=args.speak,
        semantic_provider=args.semantic_provider,
        local_intent_adapter_dir=args.local_intent_adapter_dir,
        local_intent_base_model=args.local_intent_base_model,
        live_api_budget=live_api_budget,
        initial_runtime_slots=initial_runtime_slots,
    )


if __name__ == "__main__":
    raise SystemExit(main())
