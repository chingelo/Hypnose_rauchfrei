from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from phase4_semantic_prompt_prototype import (
    ScriptNodeSpec,
    SemanticModelDecision,
    SemanticNodeSpec,
    available_node_ids as phase4_available_node_ids,
    get_node_spec as get_phase4_node_spec,
    repair_semantic_payload as repair_phase4_payload,
    script_reply_for_decision as phase4_script_reply,
    validate_semantic_decision as validate_phase4_decision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDED_SESSION_PHASES_CANDIDATE_PATHS = (
    PROJECT_ROOT / "backend" / "content_reference" / "guided_session_phases.json",
)


def _resolve_guided_session_phases_path() -> Path:
    for path in GUIDED_SESSION_PHASES_CANDIDATE_PATHS:
        if path.exists():
            return path
    return GUIDED_SESSION_PHASES_CANDIDATE_PATHS[0]


GUIDED_SESSION_PHASES_PATH = _resolve_guided_session_phases_path()


def _fix_guided_text(text: str) -> str:
    fixed = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
    fixed = re.sub(r"\[\[(?:kurz|mittel|lang)\]\]", "", fixed)
    fixed = re.sub(r"\[\[pause:[^\]]+\]\]", "", fixed)
    fixed = re.sub(r"\[\[(?:wait_yesno|wait_scale_1_10)\]\]\s*", "", fixed)
    fixed = fixed.replace("\r\n", "\n").replace("\r", "\n")
    fixed = re.sub(r"[ \t]+\n", "\n", fixed)
    fixed = re.sub(r"\n{3,}", "\n\n", fixed)
    return fixed.strip()


def _load_guided_session_phases() -> dict[str, str]:
    data = json.loads(GUIDED_SESSION_PHASES_PATH.read_text(encoding="utf-8"))
    return {key: _fix_guided_text(value) for key, value in data.items()}


GUIDED_SESSION_PHASES = _load_guided_session_phases()


def _extract_phase1_segments(text: str) -> dict[str, str]:
    text = re.sub(
        r"Schalte dein Telefon in den Flugmodus\.\.\. und erlaube dir, fuer diese Zeit ganz bei dir selbst zu sein\.\s*",
        "",
        text,
        count=1,
    )
    split_marker_1 = "Und jetzt ist noch etwas ganz Wichtiges:"
    split_marker_2 = "Ich werde dich Schritt fuer Schritt durch diesen Prozess fuehren... und du kannst mir ganz einfach folgen."

    before_mindset, after_mindset = text.split(split_marker_1, 1)
    before_focus, after_focus = after_mindset.split(split_marker_2, 1)

    return {
        "preflight_script": (
            "Wenn du bereit bist, starten wir jetzt."
        ),
        "preflight_question": "Wenn du bereit bist, gib mir einfach ein kurzes Ja. Falls du vorher noch etwas brauchst oder eine Frage hast, kannst du es mir gleich sagen.",
        "setup_script": before_mindset.strip(),
        "setup_anchor_question": "Wenn bis hier alles gut passt, genuegt ein kurzes Ja. Falls noch etwas offen ist, nimm dir einen Moment und sag mir dann einfach Bescheid.",
        "mindset_script": (split_marker_1 + after_mindset.split(split_marker_2, 1)[0]).strip(),
        "focus_anchor_question": "Wenn das fuer dich so stimmig ist, genuegt ein kurzes Ja. Falls vorher noch etwas offen ist, kannst du mir jetzt kurz Bescheid geben.",
        "focus_script": (split_marker_2 + after_focus).strip(),
    }


PHASE1_SEGMENTS = _extract_phase1_segments(GUIDED_SESSION_PHASES["1"])


def _extract_phase2_segments(text: str) -> dict[str, str]:
    ready_prompt = "Bist du bereit? Bitte bestätige kurz mit Ja oder Nein."
    eyes_prompt = "Hast du deine Augen jetzt geschlossen? Bitte bestätige kurz mit Ja oder Nein."
    scene_prompt = "Hast du eine solche Situation gefunden? Bitte bestätige kurz mit Ja oder Nein."
    feel_prompt = "Hast du es jetzt deutlich spürbar? Bitte bestätige kurz mit Ja oder Nein."
    scale_clear_prompt = "Ist die Skala für dich klar? Bitte bestätige kurz mit Ja oder Nein."
    scale_before_prompt = "Wo liegst du gerade auf dieser Skala? Nenne mir bitte eine Zahl von 1 bis 10."
    scale_after_prompt = "Wo liegst du jetzt auf der Skala? Nenne mir bitte eine Zahl von 1 bis 10."
    continue_prompt = "Passt das für dich? Bitte bestätige kurz mit Ja oder Nein."

    before_ready, after_ready = text.split(ready_prompt, 1)
    between_ready_and_eyes, after_eyes = after_ready.strip().split(eyes_prompt, 1)
    between_eyes_and_scene, after_scene = after_eyes.strip().split(scene_prompt, 1)
    between_scene_and_feel, after_feel = after_scene.strip().split(feel_prompt, 1)
    between_feel_and_scale_clear, after_scale_clear = after_feel.strip().split(scale_clear_prompt, 1)
    before_scale_before, after_scale_before = after_scale_clear.strip().split(scale_before_prompt, 1)
    between_scale_before_and_after, after_scale_after = after_scale_before.strip().split(scale_after_prompt, 1)
    between_scale_after_and_continue, after_continue = after_scale_after.strip().split(continue_prompt, 1)

    return {
        "intro_script": before_ready.strip(),
        "ready_question": "Bist du bereit?",
        "post_ready_script": between_ready_and_eyes.strip(),
        "eyes_question": "Hast du deine Augen jetzt geschlossen?",
        "post_eyes_script": between_eyes_and_scene.strip(),
        "scene_question": "Hast du eine solche Situation gefunden?",
        "post_scene_script": between_scene_and_feel.strip(),
        "feel_question": "Hast du es jetzt deutlich spuerbar?",
        "post_feel_script": between_feel_and_scale_clear.strip(),
        "scale_clear_question": "Ist die Skala fuer dich klar?",
        "post_scale_clear_script": before_scale_before.strip(),
        "scale_before_question": scale_before_prompt,
        "post_scale_before_script": between_scale_before_and_after.strip(),
        "scale_after_question": scale_after_prompt,
        "post_scale_after_script": between_scale_after_and_continue.strip(),
        "continue_question": "Passt das fuer dich?",
        "phase2_end_script": after_continue.strip(),
    }


PHASE2_SEGMENTS = _extract_phase2_segments(GUIDED_SESSION_PHASES["2"])
PHASE2_SCENE_GUIDANCE_SCRIPT = (
    "Sehr gut.\n\n"
    "Gehe jetzt gedanklich in einen konkreten Moment, in dem das Verlangen nach einer Zigarette besonders stark war. "
    "Du kannst eine aktuelle Situation nehmen oder eine Erinnerung, in der du die Sucht oder das Verlangen deutlich in dir wahrnehmen konntest.\n\n"
    "Gemeint ist einfach ein Moment oder eine Situation, in der es dir leicht faellt, dieses Verlangen jetzt noch einmal klar zu spueren und wahrzunehmen."
)


PHASE4_INTRO_SCRIPT = (
    "Und waehrend du jetzt in deinem Raum der Veraenderung angekommen bist, befindest du dich in einem Raum mit vielen Tueren. "
    "Und hinter jeder Tuer verbirgt sich ein Loesungsweg. Du weisst: Jede Tuer fuehrt dich naeher zu einem Leben ohne Rauchen. "
    "Und intuitiv entscheidest du dich jetzt fuer eine dieser Tueren. Genau die richtige. Du gehst auf sie zu, oeffnest sie und trittst ein. "
    "Du befindest dich nun in einem besonderen Raum. In der Mitte steht ein magischer Sessel. Ein Sessel, der genau dafuer gemacht ist, "
    "dich an den Ursprung zurueckzufuehren, an den Moment, in dem alles begonnen hat. Das ist der Moment, in dem dieses Verlangen ueberhaupt "
    "entstanden ist, das dich spaeter zum Rauchen gebracht hat. Setz dich jetzt auf diesen Sessel. Und sobald du sitzt, spuerst du eine Kraft, "
    "die dich traegt und fuehrt.\n\n"
    "Und jetzt rufst du noch einmal den Gedanken oder die Situation auf, die du zuvor fuer dich aktiviert hast. Ich bitte dich jetzt, dieses Gefuehl "
    "noch einmal in dir entstehen zu lassen, indem du genau wieder in diese Situation hineingehst, oder in eine andere, die sich fuer dich jetzt stimmig "
    "anfuehlt. Wichtig ist nur, dass du dieses Gefuehl jetzt noch einmal deutlich in dir wahrnehmen kannst. Spuere dieses Gefuehl jetzt ganz bewusst. "
    "Und wenn du es wahrnimmst, dann verstaerke es noch einmal, so wie zuvor, als wuerdest du innerlich an deinem Verstaerkungsregler drehen. Mach es intensiver, "
    "noch intensiver, so stark, wie du es kennst.\n\n"
    "Und jetzt, wo du dieses Gefuehl so deutlich wahrnehmen kannst, werde ich gleich von fuenf bis null zaehlen. Und bei null angekommen wird dich dein Unterbewusstsein "
    "ganz automatisch an den Ursprung zurueckfuehren, an den Moment, in dem dieses Gefuehl ueberhaupt entstanden ist. Dein Unterbewusstsein weiss ganz genau, wo es dich "
    "auf deiner Lebenslinie zurueckfuehren muss, damit wir dort auch alles loesen koennen, was geloest werden muss. Uebergib dieses Gefuehl jetzt gedanklich deinem Unterbewusstsein. "
    "Fuenf, es beginnt zu suchen und hat die Situation gefunden. Vier, es bringt dir die Situation naeher. Drei, du bist bereits in der Situation. Zwei, das Bild wird klarer. "
    "Eins, du bist mitten drin. Null.\n\n"
    "Und jetzt bist du genau dort. Am Ursprung. Dort, wo alles begonnen hat."
)

PHASE4_POST_COUNTDOWN_ENTRY_SCRIPT = (
    "Und jetzt bist du genau dort. Am Ursprung. Dort, wo alles begonnen hat."
)


SESSION_ABORT_TEXT = "Gut. Dann unterbrechen wir die Session-Sandbox hier."


def _make_yes_no_node(
    node_id: str,
    question_text: str,
    node_goal: str,
    yes_next: str,
    no_reply: str,
    *,
    support_reply: str,
    question_reply: str,
    clarify_reply: str,
) -> SemanticNodeSpec:
    return SemanticNodeSpec(
        node_id=node_id,
        question_text=question_text,
        node_goal=node_goal,
        intent_meanings={
            "yes": "Der Kunde bestaetigt klar, dass die Frage mit Ja beantwortet ist oder dass der naechste Schritt bereits zutrifft.",
            "no": "Der Kunde verneint klar oder signalisiert, dass dieser Schritt noch nicht erreicht oder noch nicht passend ist.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Ja/Nein-Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie er antworten oder worauf er achten soll.",
            "support_needed": "Der Kunde braucht Stabilisierung, Erklaerung oder einen Moment, bevor er die Frage beantworten kann.",
        },
        allowed_actions=("transition", "support", "clarify", "repeat", "answer_question", "abort"),
        allowed_next_nodes=(yes_next, node_id, "session_abort_confirmation"),
        routing_rules={
            "yes": {"action": "transition", "next_node": yes_next},
            "no": {"action": "support", "next_node": node_id},
            "unclear": {"action": "clarify", "next_node": node_id},
            "repeat": {"action": "repeat", "next_node": node_id},
            "abort": {"action": "abort", "next_node": "session_abort_confirmation"},
            "question": {"action": "answer_question", "next_node": node_id},
            "support_needed": {"action": "support", "next_node": node_id},
        },
        same_node_replies={
            "no": no_reply,
            "unclear": clarify_reply,
            "repeat": question_text,
            "question": question_reply,
            "support_needed": support_reply,
            "abort": SESSION_ABORT_TEXT,
        },
    )


def _make_scale_node(node_id: str, question_text: str, node_goal: str, next_node: str) -> SemanticNodeSpec:
    return SemanticNodeSpec(
        node_id=node_id,
        question_text=question_text,
        node_goal=node_goal,
        intent_meanings={
            "provided_scale": "Der Kunde nennt eine Zahl, einen klaren Skalenwert oder eine eindeutige Einordnung im Bereich von 1 bis 10.",
            "unclear": "Die Antwort enthaelt keinen klaren Skalenwert von 1 bis 10.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie die Skala gemeint ist oder wie er antworten soll.",
            "support_needed": "Der Kunde braucht Unterstuetzung oder Stabilisierung, bevor er einen Skalenwert nennen kann.",
        },
        allowed_actions=("transition", "clarify", "repeat", "answer_question", "support", "abort"),
        allowed_next_nodes=(next_node, node_id, "session_abort_confirmation"),
        routing_rules={
            "provided_scale": {"action": "transition", "next_node": next_node},
            "unclear": {"action": "clarify", "next_node": node_id},
            "repeat": {"action": "repeat", "next_node": node_id},
            "abort": {"action": "abort", "next_node": "session_abort_confirmation"},
            "question": {"action": "answer_question", "next_node": node_id},
            "support_needed": {"action": "support", "next_node": node_id},
        },
        same_node_replies={
            "unclear": "Nenne mir bitte einfach eine Zahl von 1 bis 10, die deine aktuelle Intensitaet am besten beschreibt.",
            "repeat": question_text,
            "question": "Die Skala meint nur deine momentane Intensitaet: 1 ist ruhig und frei, 10 ist maximal intensiv. Nenne mir einfach die Zahl, die gerade am besten passt.",
            "support_needed": "Gut. Sobald sich eine Zahl klarer anfuehlt, nenne mir einfach den Wert, der gerade am naechsten liegt.",
            "abort": SESSION_ABORT_TEXT,
        },
    )


def _make_ready_node(
    node_id: str,
    question_text: str,
    node_goal: str,
    next_node: str,
    *,
    clarify_reply: str,
    question_reply: str,
    support_reply: str,
) -> SemanticNodeSpec:
    return SemanticNodeSpec(
        node_id=node_id,
        question_text=question_text,
        node_goal=node_goal,
        intent_meanings={
            "ready": "Der Kunde gibt ein Zeichen, antwortet inhaltlich oder signalisiert, dass der Prozess an diesem Punkt weitergehen kann.",
            "unclear": "Die Antwort ist zu unklar, um weiterzugehen.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, was genau jetzt gemeint ist.",
            "support_needed": "Der Kunde braucht Unterstuetzung, Stabilisierung oder etwas mehr Zeit, bevor es weitergehen kann.",
        },
        allowed_actions=("transition", "clarify", "repeat", "answer_question", "support", "abort"),
        allowed_next_nodes=(next_node, node_id, "session_abort_confirmation"),
        routing_rules={
            "ready": {"action": "transition", "next_node": next_node},
            "unclear": {"action": "clarify", "next_node": node_id},
            "repeat": {"action": "repeat", "next_node": node_id},
            "abort": {"action": "abort", "next_node": "session_abort_confirmation"},
            "question": {"action": "answer_question", "next_node": node_id},
            "support_needed": {"action": "support", "next_node": node_id},
        },
        same_node_replies={
            "unclear": clarify_reply,
            "repeat": question_text,
            "question": question_reply,
            "support_needed": support_reply,
            "abort": SESSION_ABORT_TEXT,
        },
    )


def _make_phase1_anchor_node(
    node_id: str,
    question_text: str,
    node_goal: str,
    next_node: str,
    *,
    continue_reply: str = "",
    technical_reply: str,
    question_reply: str,
    support_reply: str,
    clarify_reply: str,
) -> SemanticNodeSpec:
    same_node_replies = {
        "unclear": clarify_reply,
        "repeat": question_text,
        "question": question_reply,
        "technical_issue": technical_reply,
        "support_needed": support_reply,
        "abort": SESSION_ABORT_TEXT,
    }
    if continue_reply:
        same_node_replies["continue"] = continue_reply

    return SemanticNodeSpec(
        node_id=node_id,
        question_text=question_text,
        node_goal=node_goal,
        intent_meanings={
            "continue": "Der Kunde signalisiert, dass alles passt und der naechste Abschnitt einfach weiterlaufen kann.",
            "technical_issue": "Der Kunde beschreibt ein technisches oder aeusseres Thema wie Ton, Kopfhoerer, Verbindung, Raum oder Sitzposition.",
            "question": "Der Kunde stellt eine Rueckfrage oder will kurz wissen, wie dieser Teil gemeint ist.",
            "support_needed": "Der Kunde braucht einen Moment, mehr Orientierung, Beruhigung oder Stabilisierung, bevor es weitergehen soll.",
            "unclear": "Die Antwort ist zu unklar, um sicher weiterzugehen.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
        },
        allowed_actions=("transition", "support", "answer_question", "clarify", "repeat", "abort"),
        allowed_next_nodes=(next_node, node_id, "session_abort_confirmation"),
        routing_rules={
            "continue": {"action": "transition", "next_node": next_node},
            "technical_issue": {"action": "support", "next_node": node_id},
            "question": {"action": "answer_question", "next_node": node_id},
            "support_needed": {"action": "support", "next_node": node_id},
            "unclear": {"action": "clarify", "next_node": node_id},
            "repeat": {"action": "repeat", "next_node": node_id},
            "abort": {"action": "abort", "next_node": "session_abort_confirmation"},
        },
        same_node_replies=same_node_replies,
    )


PHASE4_COMMON_PRESENT_SELF_INTRO = (
    "Dann holst du jetzt dein jetziges Ich direkt in diese Szene hinein. Dein jetziges erwachsenes Ich kommt jetzt zu deinem damaligen Ich. "
    "Es bringt genau das mit, was damals gefehlt hat. "
    "Ganz ruhig. Ganz klar. Ganz praesent. Und dein jetziges Ich darf jetzt mit deinem damaligen Ich sprechen."
)

PHASE4_COMMON_EXPLAIN_TO_YOUNGER = (
    "Okay, dann darfst du jetzt mit deinem jetzigen Ich und all den Erfahrungen, die du bis jetzt mit dem Rauchen gemacht hast, und mit all deinen Erkenntnissen darueber, "
    "wie sehr es dir geschadet hat, deinem damaligen Ich erklaeren, was aus diesem Weg spaeter entstehen wird. Was das Rauchen spaeter mit dir gemacht hat. Welche Folgen daraus "
    "entstanden sind. Dass du spaeter sogar in Therapie gehen musst, um diese Sucht wieder loszuwerden. Und dass dir das Rauchen keinen wirklichen Nutzen bringt, sondern dir nur schadet. "
    "Ich lasse euch jetzt einen Moment Zeit, bis dein damaliges Ich es wirklich verstehen kann. Sobald du das Gefuehl hast, dass dein damaliges Ich es verstanden hat, gib mir ein kurzes Zeichen."
)

PHASE4_COMMON_FIRST_CIGARETTE = (
    "Dann geh jetzt mit deinem jetzigen Ich und deinem damaligen Ich zu dem Moment, in dem du deine erste Zigarette rauchen wirst, zu dem Moment des ersten Zuges. Gib mir Bescheid, sobald du dort bist.\n\n"
    "Und jetzt erinnere dich ganz genau an diesen Moment. An den ersten Zug. So ekelhaft, so giftig, so kratzig, so geschmacklos. Erinnere dich daran, wie es in deinen Atemwegen gekratzt hat und wie du vielleicht husten musstest. "
    "Oder wie du vielleicht nur so getan hast, als wuerdest du richtig daran ziehen, weil du damals schon wusstest, dass das Folgen haben koennte.\n\n"
    "Erinnere dich daran, wie du am Anfang vermutlich nur ein oder zwei Zuege genommen hast. Wie du zu Beginn kaum mehr als ein oder zwei Zigaretten selbst geraucht hast. "
    "Und wie lange es vielleicht sogar gedauert hat, bis du dir ueberhaupt dein erstes eigenes Paeckchen Zigaretten gekauft hast.\n\n"
    "Erinnere dich auch daran, wie sehr du dich am Anfang vielleicht sogar darum bemuehen musstest, die Zigarette ueberhaupt zu moegen. Und wie dein Koerper dir schon damals gezeigt hat, dass er das weder braucht noch will.\n\n"
    "Und erinnere dich daran, dass es in deinem Leben kaum etwas anderes gibt, bei dem du dich so sehr bemuehen musstest, etwas Ekelhaftes, Schaedliches und Teures ueberhaupt gern zu bekommen.\n\n"
    "Oder gibt es sonst irgendetwas, das du wirklich gern hast und trotzdem {anzahl_zigaretten_pro_tag} Mal pro Tag, sieben Tage die Woche, zu dir nimmst?\n\n"
    "Stell dir vor: Selbst wenn wir Crevetten, Schokolade oder Suesses gern haben, nehmen wir das nicht {anzahl_zigaretten_pro_tag} Mal am Tag, sieben Tage die Woche, zu uns. Weil wir genau wissen, dass es unserem Koerper schaden, uns die Lust daran verderben oder einfach zu viel werden koennte.\n\n"
)

PHASE4_COMMON_FIRST_CIGARETTE_CONSEQUENCES = (
    "Und spuere jetzt auch, wie sicher du damals vielleicht davon ueberzeugt warst, jederzeit wieder damit aufhoeren zu koennen. Wie sicher du vielleicht geglaubt hast, niemals einer von diesen Rauchern zu werden, die sich damit ueber Jahre selbst schaden. "
    "Und wie du vielleicht dachtest, du koenntest ja jederzeit wieder aufhoeren, es dann aber trotzdem nicht getan hast.\n\n"
    "Und erinnere dich auch an all die Momente, in denen die Zigarette dein Leben mitbestimmt hat. An Momente, in denen du keine mehr hattest. An Momente, in denen du dir Gedanken machen musstest, ob du noch genug Zigaretten hast. "
    "An Momente, in denen alles geschlossen war und du keine mehr bekommen konntest.\n\n"
    "Spuere, was das wirklich bedeutet: ein Leben, das von etwas bestimmt wird, statt ein selbstbestimmtes Leben zu fuehren.\n\n"
    "Und frage dich, wer das eigentlich wirklich moechte. Von etwas oder jemandem bestimmt zu werden, statt selbst zu entscheiden, wie man sein Leben fuehren will.\n\n"
    "Du moechtest ein selbstbestimmtes Leben fuehren. Du moechtest selbst bestimmen, was du tust, wie du lebst und warum du etwas tust. Und doch hat dich all die Jahre immer wieder diese Sucht bestimmt, gesteuert und gefuehrt.\n\n"
    "Spuere also noch einmal ganz klar, wie unnatuerlich es eigentlich war, dass du dich damals ueberhaupt darum bemuehen musstest, die Zigarette moegen zu lernen."
)

PHASE4_COMMON_FIRST_DRAG = (
    "Und jetzt geh noch einmal zu deiner ersten Zigarette zurueck und zieh dort ganz bewusst noch einmal daran. Lass diesen ekelhaften, giftigen Rauch noch einmal durch deine Atemwege bis in deine Lunge ziehen und nimm ganz klar wahr, "
    "wie ekelhaft und giftig das war. Wie es gekratzt hat. Wie es geschmeckt hat.\n\n"
    "Nimm auch den Geruch noch einmal ganz deutlich wahr. Diesen beissenden, unangenehmen, abgestandenen Geruch. Und mach dir bewusst, wohin dich diese Sucht fuehren wuerde, wenn du diesen Weg immer weitergehen wuerdest. "
    "Naemlich in Krankheit, Zerfall und am Ende in den Tod."
)

PHASE4_COMMON_COLLECT_MOMENTS = (
    "Dann darfst du jetzt mit deinem damaligen Ich durch all die spaeteren Momente deines Lebens gehen, in denen Rauchen ein Bestandteil geworden ist. Durch alle Situationen. Durch all die Momente, in denen du begonnen hast, dir selbst Zigaretten zu kaufen. "
    "Durch alle Szenen. Durch alle Erinnerungen. Durch all die Momente, in denen du geraucht hast oder das Rauchen mit dir etwas gemacht hat.\n\n"
    "Und waehrend du diese Momente durchlaeufst, darfst du wahrnehmen, was du dir damit angetan hast. Was dein Koerper mitgemacht hat. Was deine Atemwege mitgetragen haben. Was deine Lunge ausgehalten hat. Was dich das gekostet hat. An Kraft. An Freiheit. An Geld.\n\n"
    "Und all diese Momente sammelst du jetzt ein. Alle zusammen. Wie in einem grossen Sammelkorb. Du nimmst jede Szene, jede Erinnerung, jeden Augenblick, der mit dem Rauchen verbunden war, und legst alles in diesen Sammelkorb.\n\n"
    "Und ich lasse dir jetzt einen Moment Zeit, das alles einmal gedanklich durchlaufen zu lassen. Spuere diese Momente gemeinsam. Spuere, was du dir damit angetan hast. Und bring dann diesen ganzen Sammelkorb mit zurueck bis an den heutigen Tag."
)

PHASE4_COMMON_DONE_PROMPT = (
    "Sobald du alle diese Momente durchlaufen hast und wieder im Raum der Veraenderung in deinem magischen Sessel angekommen bist, gib mir kurz Bescheid."
)


SESSION_NODE_SPECS: dict[str, SemanticNodeSpec | ScriptNodeSpec] = {
    "session_phase1_intro": ScriptNodeSpec("session_phase1_intro", PHASE1_SEGMENTS["preflight_script"], "session_phase1_preflight_check"),
    "session_phase1_preflight_check": _make_phase1_anchor_node(
        "session_phase1_preflight_check",
        PHASE1_SEGMENTS["preflight_question"],
        "Pruefe vor dem eigentlichen Einstieg, ob der Kunde startklar ist oder ob vorab noch eine Frage, ein technisches Thema oder ein kurzer Moment noetig ist.",
        "session_phase1_setup_script",
        technical_reply="Pruef jetzt kurz Lautstaerke, Kopfhoerer, Verbindung oder deine Position und richte alles so ein, dass es fuer dich stimmig ist. Wenn alles passt, sag einfach Ja.",
        question_reply="Vor dem Start geht es nur darum, ob Ort, Ton und deine Position fuer dich passen oder ob du vorher noch kurz etwas klaeren willst.",
        support_reply="Gut. Wenn alles fuer dich bereit ist, sag einfach Ja.",
        clarify_reply="Wenn alles startklar ist, sag einfach Ja. Wenn du vorher noch etwas brauchst oder klaeren willst, sag es jetzt.",
    ),
    "session_phase1_setup_script": ScriptNodeSpec("session_phase1_setup_script", PHASE1_SEGMENTS["setup_script"], "session_phase1_anchor_after_setup"),
    "session_phase1_anchor_after_setup": _make_phase1_anchor_node(
        "session_phase1_anchor_after_setup",
        PHASE1_SEGMENTS["setup_anchor_question"],
        "Halte nach dem aeusseren Setup einen kurzen Ankerpunkt offen, damit der Kunde technische oder kurze inhaltliche Fragen unterbringen kann, ohne den Fluss lang zu unterbrechen.",
        "session_phase1_mindset_script",
        technical_reply="Wenn bei Ort, Ton, Verbindung oder deiner Position noch etwas nicht passt, richte es jetzt kurz ein. Wenn alles stimmt, sag einfach Ja.",
        question_reply="Bis hier geht es nur darum, dass du ungestoert, bequem und technisch gut eingerichtet bist. Wenn das passt, gehen wir direkt weiter.",
        support_reply="Gut. Richte es in deinem Tempo passend ein. Wenn alles stimmt, sag einfach Ja.",
        clarify_reply="Wenn bis hier alles passt, sag einfach Ja. Wenn du noch etwas brauchst oder kurz klaeren willst, sag es jetzt.",
    ),
    "session_phase1_mindset_script": ScriptNodeSpec("session_phase1_mindset_script", PHASE1_SEGMENTS["mindset_script"], "session_phase1_anchor_before_focus"),
    "session_phase1_anchor_before_focus": _make_phase1_anchor_node(
        "session_phase1_anchor_before_focus",
        PHASE1_SEGMENTS["focus_anchor_question"],
        "Halte vor dem Uebergang in den inneren Fokus einen letzten kurzen Ankerpunkt offen, ohne daraus einen langen Dialog zu machen.",
        "session_phase1_focus_script",
        technical_reply="Wenn vorher noch etwas an Ton, Verbindung oder deiner Position angepasst werden muss, mach das jetzt kurz. Wenn es passt, sag einfach Ja.",
        question_reply="Hier geht es nur darum, dass du nichts leisten musst und dich jetzt einfach auf den inneren Fokus einlassen kannst. Wenn das fuer dich stimmt, gehen wir weiter.",
        support_reply="Gut. Sobald es sich fuer dich stimmig anfuehlt, sag einfach Ja.",
        clarify_reply="Wenn das fuer dich so stimmig ist, sag einfach Ja. Wenn du vorher noch etwas fragen oder brauchst, sag es jetzt.",
    ),
    "session_phase1_focus_script": ScriptNodeSpec("session_phase1_focus_script", PHASE1_SEGMENTS["focus_script"], "session_phase2_intro_script"),
    "session_phase2_intro_script": ScriptNodeSpec("session_phase2_intro_script", PHASE2_SEGMENTS["intro_script"], "session_phase2_ready"),
    "session_phase2_ready": _make_yes_no_node(
        "session_phase2_ready",
        PHASE2_SEGMENTS["ready_question"],
        "Erkenne, ob der Kunde bereit ist, die EMDR-Vorbereitung zu starten.",
        "session_phase2_post_ready_script",
        "Gut. Sobald du bereit bist, sag einfach Ja.",
        support_reply="Gut. Komm kurz bei dir an. Sobald du soweit bist, sag einfach Ja.",
        question_reply="Hier geht es nur darum, ob wir jetzt beginnen koennen oder ob du noch etwas brauchst.",
        clarify_reply="Wenn du bereit bist, sag einfach Ja. Wenn vorher noch etwas offen ist, sag mir kurz Bescheid.",
    ),
    "session_phase2_post_ready_script": ScriptNodeSpec("session_phase2_post_ready_script", PHASE2_SEGMENTS["post_ready_script"], "session_phase2_eyes_closed"),
    "session_phase2_eyes_closed": _make_yes_no_node(
        "session_phase2_eyes_closed",
        PHASE2_SEGMENTS["eyes_question"],
        "Erkenne, ob der Kunde die Augen geschlossen hat und bereit fuer die innere Fokussierung ist.",
        "session_phase2_post_eyes_script",
        "Gut. Dann schliesse jetzt bitte in deinem Tempo die Augen und gib mir kurz Bescheid, sobald das fuer dich passt.",
        support_reply="Gut. Schliesse die Augen erst dann, wenn es sich fuer dich stimmig anfuehlt.",
        question_reply="Hier geht es nur darum, ob deine Augen jetzt geschlossen sind oder ob du noch einen Moment brauchst.",
        clarify_reply="Wenn deine Augen jetzt geschlossen sind, sag einfach Ja. Wenn noch etwas Zeit noetig ist, bleib einfach kurz dabei.",
    ),
    "session_phase2_post_eyes_script": ScriptNodeSpec("session_phase2_post_eyes_script", PHASE2_SCENE_GUIDANCE_SCRIPT, "session_phase2_scene_found"),
    "session_phase2_scene_found": _make_yes_no_node(
        "session_phase2_scene_found",
        PHASE2_SEGMENTS["scene_question"],
        "Erkenne, ob der Kunde eine passende Situation fuer die Aktivierung gefunden hat.",
        "session_phase2_post_scene_script",
        "Gut. Geh gedanklich in einen konkreten Moment, in dem das Verlangen nach der Zigarette deutlich spuerbar war. Sobald du eine passende Situation gefunden hast, sag einfach Ja.",
        support_reply="Gut. Dann such nicht mit Druck. Lass einfach einen Moment auftauchen, in dem das Verlangen oder die Sucht nach der Zigarette deutlich zu spueren war. Wenn eine passende Situation da ist, gib mir kurz ein Ja.",
        question_reply="Gemeint ist ein konkreter Moment oder eine Situation, in der du das Verlangen nach der Zigarette oder den Suchtdruck deutlich in dir wahrnehmen konntest. Du kannst eine aktuelle Situation oder eine starke Erinnerung nehmen.",
        clarify_reply="Wenn du eine passende Situation gefunden hast, sag einfach Ja. Wenn noch nichts Passendes da ist, bleib einfach noch kurz bei der Suche.",
    ),
    "session_phase2_post_scene_script": ScriptNodeSpec("session_phase2_post_scene_script", PHASE2_SEGMENTS["post_scene_script"], "session_phase2_feel_clear"),
    "session_phase2_feel_clear": _make_yes_no_node(
        "session_phase2_feel_clear",
        PHASE2_SEGMENTS["feel_question"],
        "Erkenne, ob das Gefuehl jetzt deutlich genug spuerbar ist, um weiterzugehen.",
        "session_phase2_post_feel_script",
        "Gut. Dann geh noch einen Moment tiefer in die Situation hinein und lass das Gefuehl etwas deutlicher werden. Wenn es klar spuerbar ist, sag Ja.",
        support_reply="Gut. Dann lass dir kurz Zeit. Nimm nur so viel wahr, wie es stabil moeglich ist, und sag mir Ja, sobald es deutlich genug spuerbar ist.",
        question_reply="Es reicht, wenn du kurz einschaetzt, ob das Gefuehl jetzt deutlich genug spuerbar ist.",
        clarify_reply="Wenn das Gefuehl jetzt deutlich genug spuerbar ist, sag einfach Ja. Wenn noch etwas Zeit noetig ist, bleib noch kurz dort.",
    ),
    "session_phase2_post_feel_script": ScriptNodeSpec("session_phase2_post_feel_script", PHASE2_SEGMENTS["post_feel_script"], "session_phase2_scale_clear"),
    "session_phase2_scale_clear": _make_yes_no_node(
        "session_phase2_scale_clear",
        PHASE2_SEGMENTS["scale_clear_question"],
        "Erkenne, ob die Belastungsskala fuer den Kunden klar ist.",
        "session_phase2_scale_before",
        "Gut. Dann schau dir die Skala noch einmal kurz an: 1 ist ruhig und frei, 10 ist maximal intensiv. Wenn das klar ist, sag Ja.",
        support_reply="Gut. Schau dir die Skala einfach noch kurz an. Es reicht, wenn du spaeter die Zahl nennst, die sich am naechsten anfuehlt.",
        question_reply="Die Skala meint nur deine momentane Intensitaet: 1 ist ruhig und frei, 10 ist maximal intensiv. Sobald das klar ist, sag Ja.",
        clarify_reply="Wenn die Skala fuer dich klar ist, sag einfach Ja. Wenn daran noch etwas unklar ist, sag mir kurz Bescheid.",
    ),
    "session_phase2_scale_before": _make_scale_node(
        "session_phase2_scale_before",
        PHASE2_SEGMENTS["scale_before_question"],
        "Erkenne, dass der Kunde einen aktuellen Belastungswert auf der Skala nennt.",
        "session_phase2_post_scale_before_script",
    ),
    "session_phase2_post_scale_before_script": ScriptNodeSpec("session_phase2_post_scale_before_script", PHASE2_SEGMENTS["post_scale_before_script"], "session_phase2_scale_after"),
    "session_phase2_scale_after": _make_scale_node(
        "session_phase2_scale_after",
        PHASE2_SEGMENTS["scale_after_question"],
        "Erkenne, dass der Kunde nach der Runde erneut einen Skalenwert nennt.",
        "session_phase2_post_scale_after_script",
    ),
    "session_phase2_post_scale_after_script": ScriptNodeSpec("session_phase2_post_scale_after_script", PHASE2_SEGMENTS["post_scale_after_script"], "session_phase2_continue_to_main"),
    "session_phase2_continue_to_main": _make_yes_no_node(
        "session_phase2_continue_to_main",
        PHASE2_SEGMENTS["continue_question"],
        "Erkenne, ob der Kunde bereit ist, von der EMDR-Vorbereitung in den Hauptteil der Session zu wechseln.",
        "session_phase2_end_script",
        "Gut. Wenn es fuer dich passt, sag einfach Ja.",
        support_reply="Gut. Dann bleib noch einen Moment bei dir. Sobald es fuer dich stimmig ist, koennen wir weitergehen.",
        question_reply="Es geht nur darum, ob es fuer dich jetzt passt, in die naechste Phase ueberzugehen.",
        clarify_reply="Wenn das fuer dich jetzt passt, sag einfach Ja. Wenn noch etwas Zeit noetig ist, bleib einfach kurz dabei.",
    ),
    "session_phase2_end_script": ScriptNodeSpec("session_phase2_end_script", PHASE2_SEGMENTS["phase2_end_script"], "session_phase3_induction"),
    "session_phase3_induction": ScriptNodeSpec("session_phase3_induction", GUIDED_SESSION_PHASES["3"], "session_phase4_intro"),
    "session_phase4_intro": ScriptNodeSpec("session_phase4_intro", PHASE4_INTRO_SCRIPT, "hell_light_level"),
    "session_phase4_post_countdown_entry": ScriptNodeSpec(
        "session_phase4_post_countdown_entry",
        PHASE4_POST_COUNTDOWN_ENTRY_SCRIPT,
        "hell_light_level",
    ),
    "phase4_common_present_self_intro": ScriptNodeSpec("phase4_common_present_self_intro", PHASE4_COMMON_PRESENT_SELF_INTRO, "phase4_common_sees_younger_self"),
    "phase4_common_sees_younger_self": _make_yes_no_node(
        "phase4_common_sees_younger_self",
        "Bist du jetzt dort und siehst du dein damaliges Ich?",
        "Erkenne, ob der Kunde sein damaliges Ich in der Szene jetzt wahrnehmen kann.",
        "phase4_common_explain_to_younger",
        "Gut. Lass die Szene etwas klarer werden, bis du dein damaliges Ich dort wahrnehmen kannst.",
        support_reply="Gut. Dann bleib ruhig bei dir und lass die Szene sich in deinem eigenen Tempo etwas klarer zeigen. Sobald du dein damaliges Ich wahrnimmst, sag Ja.",
        question_reply="Es reicht, wenn du kurz einschaetzt, ob du dein damaliges Ich dort jetzt wahrnehmen kannst.",
        clarify_reply="Wenn du dein damaliges Ich jetzt sehen kannst, sag einfach Ja. Wenn die Szene noch unscharf ist, bleib noch einen Moment dabei.",
    ),
    "phase4_common_explain_to_younger": ScriptNodeSpec("phase4_common_explain_to_younger", PHASE4_COMMON_EXPLAIN_TO_YOUNGER, "phase4_common_understood"),
    "phase4_common_understood": _make_yes_no_node(
        "phase4_common_understood",
        "Hat dein damaliges Ich es verstanden?",
        "Erkenne, ob das damalige Ich die Erklaerung des heutigen Ichs verstanden hat.",
        "phase4_common_first_cigarette",
        "Gut. Dann bleib noch einen Moment bei dieser Erklaerung, bis dein damaliges Ich es wirklich verstehen kann. Gib mir dann wieder ein Ja.",
        support_reply="Gut. Dann lass euch noch etwas Zeit. Wenn du merkst, dass dein damaliges Ich es wirklich verstanden hat, sag einfach Ja.",
        question_reply="Hier geht es nur darum, ob dein damaliges Ich die Erklaerung bereits aufgenommen und verstanden hat.",
        clarify_reply="Wenn dein damaliges Ich es jetzt verstanden hat, sag einfach Ja. Wenn es noch etwas Zeit braucht, bleib noch kurz dabei.",
    ),
    "phase4_common_first_cigarette": ScriptNodeSpec(
        "phase4_common_first_cigarette",
        PHASE4_COMMON_FIRST_CIGARETTE,
        "phase4_common_first_cigarette_consequences",
    ),
    "phase4_common_first_cigarette_consequences": ScriptNodeSpec(
        "phase4_common_first_cigarette_consequences",
        PHASE4_COMMON_FIRST_CIGARETTE_CONSEQUENCES,
        "phase4_common_feel_after_learning",
    ),
    "phase4_common_feel_after_learning": _make_ready_node(
        "phase4_common_feel_after_learning",
        "Wie fuehlt sich das jetzt fuer dich an?",
        "Nimm jede sinnvolle Rueckmeldung nach der ersten Reflexionsrunde auf und fuehre dann in die naechste Scriptstufe weiter.",
        "phase4_common_first_drag",
        clarify_reply="Spuer noch einmal kurz nach und sag mir einfach, wie sich das jetzt fuer dich anfuehlt.",
        question_reply="Hier musst du nichts Besonderes leisten. Gib einfach kurz wieder, wie es sich fuer dich gerade anfuehlt.",
        support_reply="Gut. Dann bleib noch einen Moment ruhig bei dir und gib mir erst dann kurz Rueckmeldung, wenn du etwas dazu sagen kannst.",
    ),
    "phase4_common_first_drag": ScriptNodeSpec("phase4_common_first_drag", PHASE4_COMMON_FIRST_DRAG, "phase4_common_feel_after_aversion"),
    "phase4_common_feel_after_aversion": _make_ready_node(
        "phase4_common_feel_after_aversion",
        "Wie fuehlt sich das jetzt fuer dich an?",
        "Nimm jede sinnvolle Rueckmeldung nach der Aversionseinbettung auf und fuehre dann in die Sammelphase weiter.",
        "phase4_common_collect_moments",
        clarify_reply="Spuer noch einmal kurz nach und sag mir einfach, wie sich das jetzt fuer dich anfuehlt.",
        question_reply="Hier reicht es, wenn du kurz beschreibst, wie es sich jetzt fuer dich anfuehlt.",
        support_reply="Gut. Antworte erst dann, wenn du es gut spuerbar einordnen kannst.",
    ),
    "phase4_common_collect_moments": ScriptNodeSpec("phase4_common_collect_moments", PHASE4_COMMON_COLLECT_MOMENTS, "phase4_common_done_signal"),
    "phase4_common_done_signal": _make_ready_node(
        "phase4_common_done_signal",
        PHASE4_COMMON_DONE_PROMPT,
        "Erkenne, ob der Kunde mit dem Sammelkorb wieder im Raum der Veraenderung angekommen ist und wir in Phase 5 weitergehen koennen.",
        "session_phase5_future",
        clarify_reply="Gib mir einfach kurz Bescheid, sobald du wieder im Raum der Veraenderung in deinem magischen Sessel angekommen bist.",
        question_reply="Hier reicht ein kurzes Zeichen oder eine kurze Rueckmeldung, sobald du wieder im Raum der Veraenderung angekommen bist.",
        support_reply="Gut. Melde dich erst dann, wenn du wieder im Raum der Veraenderung angekommen bist.",
    ),
    "session_phase5_future": ScriptNodeSpec("session_phase5_future", GUIDED_SESSION_PHASES["5"], "session_phase6_outro"),
    "session_phase6_outro": ScriptNodeSpec("session_phase6_outro", GUIDED_SESSION_PHASES["6"], None),
    "session_abort_confirmation": ScriptNodeSpec("session_abort_confirmation", SESSION_ABORT_TEXT, None),
}


EXTERNAL_SCRIPT_NEXT_OVERRIDES = {
    "dark_backtrace_terminal": "hell_light_level",
    "dark_origin_terminal": "dark_scene_happening",
    "origin_self_release_intro": "phase4_common_present_self_intro",
    "group_resolution_complete": "phase4_common_present_self_intro",
}


def get_node_spec(node_id: str) -> SemanticNodeSpec | ScriptNodeSpec:
    if node_id in SESSION_NODE_SPECS:
        return SESSION_NODE_SPECS[node_id]
    return get_phase4_node_spec(node_id)


def get_semantic_node_spec(node_id: str) -> SemanticNodeSpec:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise ValueError(f"node '{node_id}' is not a semantic node")
    return spec


def build_request(
    node_id: str,
    customer_message: str,
    *,
    clarify_attempt: int = 0,
    session_context: str = "",
) -> dict[str, Any]:
    return get_semantic_node_spec(node_id).as_request(customer_message, clarify_attempt, session_context)


def repair_semantic_payload(node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if node_id in SESSION_NODE_SPECS:
        spec = get_semantic_node_spec(node_id)
        repaired = dict(payload)
        intent = str(repaired.get("intent") or "").strip()
        action = str(repaired.get("action") or "").strip()
        next_node = str(repaired.get("next_node") or "").strip()
        routed = spec.routing_rules.get(intent) if intent in spec.allowed_intents else None
        if routed is not None:
            if not action or action not in spec.allowed_actions or action != routed["action"]:
                repaired["action"] = routed["action"]
                action = routed["action"]
            if not next_node or next_node not in spec.allowed_next_nodes or next_node != routed["next_node"]:
                repaired["next_node"] = routed["next_node"]
                next_node = routed["next_node"]
        if (not intent or intent not in spec.allowed_intents) and action and next_node:
            matching_intents = [
                name
                for name, route in spec.routing_rules.items()
                if route["action"] == action and route["next_node"] == next_node
            ]
            if len(matching_intents) == 1:
                repaired["intent"] = matching_intents[0]
        if "confidence" not in repaired and repaired.get("intent"):
            repaired["confidence"] = 0.75
        if (not str(repaired.get("reason") or "").strip()) and repaired.get("intent"):
            repaired["reason"] = "Auto-repaired missing reason from partial model output."
        return repaired
    return repair_phase4_payload(node_id, payload)


def validate_semantic_decision(node_id: str, payload: dict[str, Any]) -> SemanticModelDecision:
    if node_id in SESSION_NODE_SPECS:
        spec = get_semantic_node_spec(node_id)
        required = {"intent", "action", "next_node", "confidence", "reason"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"semantic decision missing keys: {', '.join(missing)}")
        intent = str(payload["intent"]).strip()
        action = str(payload["action"]).strip()
        next_node = str(payload["next_node"]).strip()
        confidence = float(payload["confidence"])
        reason = str(payload["reason"]).strip()
        if intent not in spec.allowed_intents:
            raise ValueError(f"invalid intent for {node_id}: {intent}")
        if action not in spec.allowed_actions:
            raise ValueError(f"invalid action for {node_id}: {action}")
        if next_node not in spec.allowed_next_nodes:
            raise ValueError(f"invalid next_node for {node_id}: {next_node}")
        if not reason:
            raise ValueError("reason must not be empty")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"confidence out of range: {confidence}")
        routed = spec.routing_rules.get(intent)
        if routed is not None:
            if action != routed["action"]:
                raise ValueError(
                    f"invalid action routing for {node_id}: intent '{intent}' expects '{routed['action']}', got '{action}'"
                )
            if next_node != routed["next_node"]:
                raise ValueError(
                    f"invalid next_node routing for {node_id}: intent '{intent}' expects '{routed['next_node']}', got '{next_node}'"
                )
        return SemanticModelDecision(
            intent=intent,
            action=action,
            next_node=next_node,
            confidence=confidence,
            reason=reason,
        )
    return validate_phase4_decision(node_id, payload)


def script_reply_for_decision(node_id: str, decision: SemanticModelDecision) -> str:
    if node_id in SESSION_NODE_SPECS:
        spec = get_semantic_node_spec(node_id)
        return spec.same_node_replies.get(decision.intent, "")
    return phase4_script_reply(node_id, decision)


def render_script_node(node_id: str) -> tuple[str, str | None]:
    spec = get_node_spec(node_id)
    if not isinstance(spec, ScriptNodeSpec):
        raise ValueError(f"node '{node_id}' is not a script node")
    next_node = spec.next_node
    if next_node is None and node_id in EXTERNAL_SCRIPT_NEXT_OVERRIDES:
        next_node = EXTERNAL_SCRIPT_NEXT_OVERRIDES[node_id]
    return spec.script_text, next_node


def maybe_render_entry_script(node_id: str) -> str:
    spec = get_node_spec(node_id)
    if isinstance(spec, SemanticNodeSpec):
        return spec.entry_script.strip()
    return ""


def available_node_ids() -> set[str]:
    return set(SESSION_NODE_SPECS) | set(phase4_available_node_ids())
