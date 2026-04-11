from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticModelDecision:
    intent: str
    action: str
    next_node: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class SemanticNodeSpec:
    node_id: str
    question_text: str
    node_goal: str
    intent_meanings: dict[str, str]
    allowed_actions: tuple[str, ...]
    allowed_next_nodes: tuple[str, ...]
    routing_rules: dict[str, dict[str, str]]
    same_node_replies: dict[str, str]
    entry_script: str = ""

    @property
    def allowed_intents(self) -> tuple[str, ...]:
        return tuple(self.intent_meanings.keys())

    @property
    def system_prompt(self) -> str:
        intent_lines = "\n".join(
            f"- {intent}: {description}" for intent, description in self.intent_meanings.items()
        )
        action_lines = "\n".join(f"- {action}" for action in self.allowed_actions)
        next_node_lines = "\n".join(f"- {node}" for node in self.allowed_next_nodes)
        priority_lines: list[str] = []
        if "question" in self.intent_meanings:
            priority_lines.append(
                "- Wenn der Kunde hauptsaechlich eine Rueckfrage stellt oder um Erklaerung bittet, waehle `question` statt eines Gefuehls- oder Status-Intents."
            )
            priority_lines.append(
                "- Formulierungen wie 'wie meinst du das', 'was meinst du', 'wie soll ich antworten', 'kannst du das erklaeren' oder 'ich verstehe die Frage nicht' sind immer `question`, auch wenn im Knotenkontext schon Begriffe wie Person, Gruppe oder Gefuehl vorkommen."
            )
            priority_lines.append(
                "- Auch Rueckmeldungen darueber, dass du den Kunden nicht verstanden hast oder dass seine vorige Antwort nicht angekommen ist, sind `question`, zum Beispiel 'du verstehst mich nicht', 'du hast mich nicht verstanden' oder 'das war nicht meine Antwort'."
            )
        if "support_needed" in self.intent_meanings:
            priority_lines.append(
                "- Wenn der Kunde hauptsaechlich ausdrueckt, dass es gerade zu viel ist, er Ueberforderung, Stabilisierung, Beruhigung oder einen sicheren Rahmen braucht, waehle `support_needed` statt eines normalen Branch-Intents."
            )
            priority_lines.append(
                "- Formulierungen wie 'es ist mir zu viel', 'ich brauche einen Moment', 'ich kann gerade nicht', 'ich bin ueberfordert' oder 'ich muss mich erst beruhigen' sind immer `support_needed` und niemals ein normaler Inhalts-Branch."
            )
            priority_lines.append(
                "- Wenn der Kunde wegdriftet, muede wird, zum Beispiel 'ich schlafe fast ein' sagt, innerlich blockiert ist, unsicher ist, ob er ueberhaupt weitermachen will, zum Beispiel 'eigentlich habe ich keine Lust mehr', oder gereizt und frustriert reagiert, ist das ebenfalls `support_needed` und niemals ein normaler Inhalts-Branch."
            )
        if "ready" in self.intent_meanings and not ("yes" in self.intent_meanings and "no" in self.intent_meanings):
            priority_lines.append(
                "- Bei freien Inhaltsknoten zaehlen auch sehr kurze, aber inhaltlich passende Antworten als `ready`, wenn sie die Frage semantisch beantworten. Ein einzelnes Gefuehl, Koerperwort, Sinneseindruck, Ort, Name oder Ereignis kann bereits genuegen, zum Beispiel 'druck', 'angst', 'enge', 'trauer', 'lachen', 'mein vater' oder 'pausenhof'. Waehle nur dann `unclear`, wenn die Antwort die aktuelle Frage wirklich nicht inhaltlich beantwortet."
            )
            priority_lines.append(
                "- Beliebige Nonsense-, Platzhalter- oder Gegenstandsworte ohne Bezug zur Frage wie 'banane', 'kartoffel', 'asdf' oder 'irgendwas' sind niemals `ready`, sondern `unclear`, solange sie die aktuelle Frage nicht wirklich beantworten."
            )
            priority_lines.append(
                "- Wenn eine freie Antwort mit einem kurzen Einleitungswort wie 'ja', 'nein', 'ich glaube' oder 'doch' beginnt, danach aber eine inhaltliche Erklaerung folgt, bewerte den inhaltlichen Teil. Ein fuehrendes Einleitungswort macht die Antwort nicht automatisch `unclear`."
            )
        if {"visual", "audio", "other_sense"} <= set(self.intent_meanings):
            priority_lines.append(
                "- Bei Wahrnehmungsknoten gilt: einzelne Hoerhinweise wie 'lachen', 'schreien', 'stimmen', 'weinen', 'geraeusche' oder 'rufe' zaehlen als `audio`. Einzelne Koerper- oder Nichtbildhinweise wie 'druck', 'enge', 'klo??', 'uebelkeit', 'hitze', 'kaelte', 'geruch' oder 'geschmack' zaehlen als `other_sense`. Einzelne Bildhinweise wie 'pausenhof', 'klasse', 'zimmer', 'mann', 'frau', 'vater', 'mutter', 'gruppe' oder konkrete Personen zaehlen als `visual`."
            )
        if "nothing_yet" in self.intent_meanings:
            priority_lines.append(
                "- Formulierungen wie 'noch nichts', 'gar nichts', 'ich nehme noch nichts wahr' oder 'da ist noch nichts' sind `nothing_yet`, nicht `nonvisual_access`."
            )
        if "nothing" in self.intent_meanings:
            priority_lines.append(
                "- Formulierungen wie 'noch nichts', 'gar nichts', 'ich nehme nichts wahr' oder 'da ist noch nichts' sind `nothing`, nicht ein anderer Wahrnehmungs-Intent."
            )
            if {"audio", "other_sense"} & set(self.intent_meanings):
                priority_lines.append(
                    "- Wenn der Kunde nur sagt, dass noch kein Bild da ist, zum Beispiel 'ich sehe nichts', 'kein Bild' oder 'nichts sichtbar', ist das noch nicht automatisch `nothing`. Das bedeutet nur fehlenden visuellen Zugang. Solange nicht klar gesagt wird, dass gar nichts wahrnehmbar ist, oeffne zuerst den Zugang ueber andere Sinne."
                )
        if {"whole_group", "one_person", "multiple_people"} <= set(self.intent_meanings):
            priority_lines.append(
                "- Bei Gruppen-Einordnung gilt: eine einzelne benannte Person oder eine einzelne Rolle wie 'Peter', 'mein Vater', 'der Lehrer' zaehlt als `one_person`. Woerter wie 'Gruppe', 'Klasse', 'Clique', 'alle' oder 'alle zusammen' zaehlen als `whole_group`. Mehrere klar genannte Personen zaehlen als `multiple_people`."
            )
        if "yes" in self.intent_meanings and "no" in self.intent_meanings:
            priority_lines.append(
                "- Bei Ja/Nein-Knoten zaehlen auch freie semantische Bestaetigungen oder Verneinungen. Wenn der Kunde in freier Sprache erkennbar bestaetigt, dass der gefragte Zustand erreicht ist, waehle `yes` auch ohne das Wort 'ja'. Wenn der Kunde in freier Sprache erkennbar sagt, dass der Zustand noch nicht erreicht ist oder noch Zeit braucht, waehle `no`."
            )
            priority_lines.append(
                "- Formulierungen wie 'noch nicht', 'noch nicht klar', 'noch nicht ganz', 'nicht ganz', 'ich brauche noch einen Moment' oder 'sehe ich noch nicht' sind an Ja/Nein-Knoten `no`, solange keine Ueberforderung oder Stabilisierungsbitte ausgedrueckt wird."
            )
        branch_intents = [
            intent
            for intent in self.allowed_intents
            if intent not in {"unclear", "repeat", "abort", "question", "support_needed", "yes", "no", "ready"}
        ]
        if branch_intents:
            priority_lines.append(
                "- Wenn dieser Knoten eine geschlossene Auswahl zwischen bestimmten Branches verlangt, darfst du nur dann einen inhaltlichen Branch-Intent waehlen, wenn die Antwort semantisch wirklich eine dieser Kategorien trifft. Meta-Rueckfragen, Ueberforderung oder offensichtliche Nonsense-Worte duerfen niemals nur aus dem Kontext heraus auf einen Branch gemappt werden."
            )
        if "Welche Person" in self.question_text or self.question_text.startswith("Wen "):
            priority_lines.append(
                "- Wenn hier nach einer Person gefragt wird, zaehlen nur Namen, Rollen oder echte Personenbeschreibungen. Beliebige Gegenstaende oder Nonsense-Worte sind `unclear`."
            )
        priority_block = ""
        if priority_lines:
            priority_block = "Prioritaetsregeln:\n" + "\n".join(priority_lines) + "\n\n"
        return (
            "Du bist ein semantischer Knoten-Orchestrator fuer eine therapeutisch gefuehrte Hypnosesitzung.\n\n"
            f"Du befindest dich im Knoten `{self.node_id}`.\n"
            f"Die aktuelle Kundenfrage lautet:\n\"{self.question_text}\"\n\n"
            f"Ziel dieses Knotens:\n{self.node_goal}\n\n"
            "Wichtige Regeln:\n"
            "- Harte Prioritaet: Offensichtliche Meta-Rueckfragen zur Frage oder Antwortweise sind immer `question`, sofern dieser Intent erlaubt ist.\n"
            "- Harte Prioritaet: Offensichtliche Ueberforderungs-, Stabilisierungs- oder Pausen-Signale sind immer `support_needed`, sofern dieser Intent erlaubt ist.\n"
            "- Harte Prioritaet: Offensichtliche Nonsense-, Platzhalter- oder gegenstandsbezogene Woerter ohne Bezug zur Frage sind `unclear` und duerfen niemals nur aus Knotenkontext oder Runtime-Slots auf einen Inhalts-Branch gemappt werden.\n"
            "- Runtime-Slots, Session-Kontext und vorangehende Antworten sind nur Hilfskontext. Sie duerfen niemals die aktuelle Kundenantwort ersetzen. Wenn die aktuelle Antwort selbst keine klare Branch-Information enthaelt, darfst du keinen Inhalts-Branch nur aus dem Kontext erraten.\n"
            "- Interpretiere die freie Antwort des Kunden semantisch.\n"
            "- Erwarte keine festen Woerter oder Beispielsaetze.\n"
            "- Waehle nur aus den erlaubten Intents, Aktionen und naechsten Knoten.\n"
            "- Erfinde keine neuen Zweige, keine neuen Knoten und keine neue Logik.\n"
            "- Wenn die Antwort unklar ist oder der Kunde eine Rueckfrage stellt oder Unterstuetzung braucht, bleibe im aktuellen Knoten.\n"
            "- Du entscheidest nur den Branch. Du schreibst keinen therapeutischen Antworttext.\n"
            "- Die Rueckgabe muss ein einzelnes JSON-Objekt ohne Markdown oder Zusatztext sein.\n"
            "- Gib immer exakt diese fuenf Felder zurueck: intent, action, next_node, confidence, reason.\n\n"
            f"{priority_block}"
            "Erlaubte Intents:\n"
            f"{intent_lines}\n\n"
            "Erlaubte Aktionen:\n"
            f"{action_lines}\n\n"
            "Erlaubte next_node Werte:\n"
            f"{next_node_lines}\n"
        )

    def as_request(self, customer_message: str, clarify_attempt: int, session_context: str) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "question_text": self.question_text,
            "node_goal": self.node_goal,
            "intent_meanings": self.intent_meanings,
            "allowed_intents": list(self.allowed_intents),
            "allowed_actions": list(self.allowed_actions),
            "allowed_next_nodes": list(self.allowed_next_nodes),
            "routing_rules": self.routing_rules,
            "customer_message": customer_message,
            "clarify_attempt": max(0, int(clarify_attempt)),
            "session_context": session_context.strip(),
        }


@dataclass(frozen=True)
class ScriptNodeSpec:
    node_id: str
    script_text: str
    next_node: str | None = None


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
            "yes": "Der Kunde bestaetigt klar oder in freier Sprache semantisch, dass die Frage mit Ja beantwortet ist oder dass dieser Schritt jetzt erreicht ist.",
            "no": "Der Kunde verneint klar oder signalisiert auch in freier Sprache, dass dieser Schritt noch nicht erreicht oder noch nicht passend ist.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Ja/Nein-Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage oder braucht eine Erklaerung zur Frage.",
            "support_needed": "Der Kunde braucht Stabilisierung, Orientierung oder noch einen Moment, bevor er antworten kann.",
        },
        allowed_actions=("transition", "support", "clarify", "repeat", "answer_question", "abort"),
        allowed_next_nodes=(yes_next, node_id, "abort_confirmation"),
        routing_rules={
            "yes": {"action": "transition", "next_node": yes_next},
            "no": {"action": "support", "next_node": node_id},
            "unclear": {"action": "clarify", "next_node": node_id},
            "repeat": {"action": "repeat", "next_node": node_id},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": node_id},
            "support_needed": {"action": "support", "next_node": node_id},
        },
        same_node_replies={
            "no": no_reply,
            "unclear": clarify_reply,
            "repeat": question_text,
            "question": question_reply,
            "support_needed": support_reply,
            "abort": NODE_SCRIPTS["abort"],
        },
    )


def _make_ready_transition_node(
    node_id: str,
    question_text: str,
    node_goal: str,
    next_node: str,
    *,
    clarify_reply: str,
    question_reply: str,
    support_reply: str,
    ready_meaning: str = "Der Kunde gibt eine freie, inhaltlich sinnvolle Rueckmeldung, die zeigt, dass dieser Schritt innerlich bearbeitet wurde und wir weitergehen koennen.",
) -> SemanticNodeSpec:
    return SemanticNodeSpec(
        node_id=node_id,
        question_text=question_text,
        node_goal=node_goal,
        intent_meanings={
            "ready": ready_meaning,
            "unclear": "Die Antwort ist zu unklar, um sicher weiterzugehen.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage oder braucht eine Erklaerung zur Frage.",
            "support_needed": "Der Kunde braucht Stabilisierung, Orientierung oder noch einen Moment, bevor er antworten kann.",
        },
        allowed_actions=("transition", "clarify", "repeat", "answer_question", "support", "abort"),
        allowed_next_nodes=(next_node, node_id, "abort_confirmation"),
        routing_rules={
            "ready": {"action": "transition", "next_node": next_node},
            "unclear": {"action": "clarify", "next_node": node_id},
            "repeat": {"action": "repeat", "next_node": node_id},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": node_id},
            "support_needed": {"action": "support", "next_node": node_id},
        },
        same_node_replies={
            "unclear": clarify_reply,
            "repeat": question_text,
            "question": question_reply,
            "support_needed": support_reply,
            "abort": NODE_SCRIPTS["abort"],
        },
    )


NODE_SCRIPTS = {
    "hell_feel_question": "Wie fuehlt sich dieses Helle fuer dich an: sehr angenehm oder unangenehm?",
    "hell_hypnose_loch_block": (
        "Sehr gut. Das ist haeufig ein Hypnose-Loch. In solchen Momenten ist dein Unterbewusstsein schon fuer dich am arbeiten und loest das, was zu loesen gilt, direkt fuer dich auf. Bleib jetzt einfach in diesem sehr angenehmen Zustand und lass alles geschehen, was sich dort loesen oder ordnen will.\n\n"
        "Und waehrend du dort in diesem hellen, sehr angenehmen Zustand bleibst, loest sich dieses urspruengliche ungute Gefuehl vollstaendig auf."
    ),
    "hell_hypnose_wait_question": "Wie fuehlt es sich jetzt fuer dich an: loest es sich noch auf, brauchst du noch einen Moment, oder hat es sich bereits aufgeloest?",
    "hell_wait_more": "Bleib noch einen kurzen Moment in diesem hellen, sehr angenehmen Zustand. Lass alles weiter geschehen, was sich dort gerade loesen oder ordnen will.",
    "hell_wait_resolved": "Sehr gut. Dort hat sich bereits etwas Wichtiges geloest. Von dort aus gehen wir jetzt ruhig in den naechsten Schritt weiter.",
    "hell_regulation_intro": "Dann lass das Licht jetzt etwas weicher werden und bring es auf eine Distanz, die sich fuer dich sicher und gut steuerbar anfuehlt.",
    "hell_regulation_choice": "Was hilft dir gerade am meisten: mehr Abstand, weniger Helligkeit oder ein klarerer Fokus?",
    "hell_regulation_check": "Wie wirkt die Szene jetzt: eher hell, eher dunkel, beides oder deutlich ruhiger?",
    "dark_known_question": "Wenn du dieses Gefuehl jetzt so wahrnimmst: Kommt es dir aus frueheren Momenten schon bekannt vor, oder zeigt es sich hier zum ersten Mal?",
    "dark_backtrace_intro": (
        "Dann zeigt sich, dass das noch nicht der erste Entstehungspunkt ist. Nimm das Gefuehl noch einmal klar auf und uebergib es deinem Unterbewusstsein. "
        "Ich werde jetzt nochmals von fuenf bis null zaehlen, und dein Unterbewusstsein wird dich bei null an den frueheren Ursprung zurueckfuehren. "
        "Fuenf, es beginnt zu suchen und hat die Situation gefunden. Vier, es bringt dir die Situation naeher. Drei, du bist bereits in der Situation. "
        "Zwei, das Bild wird klarer. Eins, du bist mitten drin. Null."
    ),
    "dark_origin_intro": (
        "Gut. Dann bleiben wir jetzt in genau dieser Szene, denn hier zeigt sich ein wichtiger Ursprung.\n\n"
        "{origin_scene_reflection}\n\n"
        "Bleib jetzt in genau diesem Moment und lass die Situation noch deutlicher werden, "
        "damit wir besser einordnen koennen, was sich hier zeigt und ob genau hier etwas angeschaut "
        "oder geloest werden sollte."
    ),
    "origin_trigger_source_question": "Wenn du in diesem Moment bleibst: Was oder wer loest dieses ungute Gefuehl dort am staerksten in dir aus?",
    "origin_trigger_known_question": (
        "Wenn du jetzt bei diesem Gefuehl bleibst: "
        "Kommt es dir aus frueheren Momenten schon bekannt vor, "
        "oder zeigt es sich hier in dieser Szene zum ersten Mal?"
    ),
    "origin_scene_relevance_question": (
        "Wenn du bei dem bleibst, was dort gerade am staerksten wirkt: "
        "Ist in genau dieser Szene etwas, das wir hier anschauen oder loesen sollten, "
        "oder merkst du eher, dass dich dieses Gefuehl noch weiter zu etwas Frueherem fuehrt?"
    ),
    "origin_cause_owner_question": "Liegt der eigentliche Grund eher in dir selbst oder eher bei jemand anderem?",
    "origin_other_target_kind_question": (
        "Wenn du bei dem bleibst, was dort gerade stark auf dich wirkt: "
        "weist es eher auf eine Gruppe, eher auf eine bestimmte Person oder eher auf etwas anderes in dieser Situation?"
    ),
    "origin_person_name_question": "Wenn du bei dieser Person bleibst: Welche Person ist es genau?",
    "origin_person_unknown_intro": (
        "Wenn noch nicht klar ist, wer diese Person genau ist, ist das in Ordnung. "
        "Hol diese Person jetzt etwas naeher zu dir heran und schau einfach, ob deutlicher wird, wer das sein koennte. "
        "Auch wenn es noch nicht ganz eindeutig ist, koennen wir von hier aus mit dieser Person weitergehen."
    ),
    "origin_person_branch_intro": (
        "Dann nehmen wir diese Person jetzt etwas klarer in den Fokus, "
        "damit du besser wahrnehmen kannst, was zwischen euch in diesem Moment wirkt."
    ),
    "origin_self_resolution_intro": (
        "Dann wird jetzt deutlicher, dass dieser Impuls damals aus dir selbst heraus entstanden ist und wir genau dort ansetzen."
    ),
    "origin_self_need_question": "Wenn du genau in diesem Moment bleibst: Was hat dir dort am meisten gefehlt, oder was haettest du dort am meisten gebraucht?",
    "origin_self_release_intro": (
        "Gut. Dann gib deinem damaligen Ich jetzt genau das, was dort gefehlt hat: {origin_self_need}.\n\n"
        "Lass dein damaliges Ich genau das in diesem Moment aufnehmen, bis innerlich etwas ruhiger, stimmiger und vollständiger wird."
    ),
    "dark_follow_darker_intro": "Dann folgen wir jetzt dem dunkleren Anteil, weil dort meist der emotionale Kern liegt.",
    "clarify_feel": "Spuer kurz nach: fuehlt sich dieses Helle fuer dich eher angenehm oder eher unangenehm an?",
    "support_feel": "Bleib ruhig bei dir und nimm es nur so weit wahr, wie es sich stabil anfuehlt. Spuer nur nach, ob dieses Helle eher angenehm oder eher unangenehm wirkt.",
    "question_feel": "Achte hier nur darauf, wie sich dieses Helle fuer dich im Erleben anfuehlt. Ist es fuer dich eher angenehm oder eher unangenehm?",
    "clarify_wait": "Pruef noch einmal kurz in dich hinein: loest es sich noch auf, brauchst du noch einen Moment, oder ist es bereits aufgeloest?",
    "support_wait": "Bleib ruhig dort und lass den Prozess in deinem eigenen Tempo weiterlaufen. Wenn du magst, ordne gleich kurz ein, ob es sich noch aufloest, ob du noch einen Moment brauchst oder ob es bereits aufgeloest ist.",
    "question_wait": "Achte jetzt nur darauf, ob es sich noch weiter aufloest, ob du noch einen Moment brauchst oder ob es bereits ganz aufgeloest ist.",
    "clarify_light": "Nimm dort bitte nur kurz wahr, ob diese Szene eher hell, eher dunkel oder gemischt wirkt.",
    "support_light": "Schau einfach weiter hin. Sobald es fuer dich klarer wirkt, reicht eine kurze Einordnung, ob es dort eher hell oder eher dunkel ist.",
    "question_light": "Hier geht es nur um die Helligkeit der Szene. Eine kurze Einordnung, ob sie eher hell, eher dunkel oder gemischt wirkt, reicht voellig.",
    "scene_access_followup_question": "Wenn dort noch kein klares Bild auftaucht, ist das in Ordnung. Nimm einfach wahr, ob du etwas hoeren, riechen, schmecken oder ein bestimmtes Gefuehl wahrnehmen kannst.",
    "scene_access_body_bridge_intro": "Dann gehen wir jetzt ohne klares Bild ueber das weiter, was du im Koerper oder auf andere Weise deutlich wahrnimmst. Auch das ist ein gueltiger Weg.",
    "clarify_regulation": "Nimm kurz wahr, was dir jetzt am meisten helfen wuerde: mehr Abstand, weniger Helligkeit oder ein klarerer Fokus.",
    "support_regulation": "Bleib ruhig bei dir. Spuer einfach nach, ob dir gerade eher mehr Abstand, weniger Helligkeit oder ein klarerer Fokus helfen wuerde.",
    "question_regulation": "Waehl einfach das, was dir gerade am meisten helfen wuerde: mehr Abstand, weniger Helligkeit oder ein klarerer Fokus.",
    "clarify_regulation_check": "Nimm kurz wahr, ob die Szene jetzt eher hell, eher dunkel, beides oder deutlich ruhiger wirkt.",
    "support_regulation_check": "Sobald sich eine Richtung klarer abzeichnet, reicht eine kurze Einordnung: eher hell, eher dunkel, beides oder deutlich ruhiger.",
    "question_regulation_check": "Achte jetzt nur darauf, wie die Szene wirkt: eher hell, eher dunkel, beides oder deutlich ruhiger.",
    "clarify_dark_known": "Ich meine nicht, ob du das Gefuehl benennen kannst. Achte nur darauf, ob es dir aus frueheren Momenten schon bekannt vorkommt oder ob es sich hier zum ersten Mal zeigt.",
    "support_dark_known": "Bleib ruhig bei dir und spuer nur nach, ob dir dieses Gefuehl aus frueheren Momenten schon bekannt vorkommt oder ob es sich hier zum ersten Mal zeigt.",
    "question_dark_known": "Hier geht es nicht darum, das Gefuehl genauer zu erklaeren. Es reicht, wenn du kurz einordnest, ob es dir aus frueheren Momenten schon bekannt vorkommt oder ob es sich hier zum ersten Mal zeigt.",
    "dark_scene_perception_question": "Und was nimmst du dort sonst noch wahr, siehst du oder hoerst du was?",
    "dark_scene_mode_clarify_question": "Okay, was kannst du wahrnehmen? Siehst du jemand oder hoerst du was?",
    "dark_scene_who_question": "Was siehst du dort genau?",
    "dark_scene_audio_detail_question": "Was hoerst du dort genau?",
    "dark_scene_other_sense_question": "Wenn du dort nichts klar siehst oder hoerst: Was nimmst du ueber Koerper, Geruch, Geschmack oder Temperatur wahr?",
    "dark_scene_first_spuerbar_question": "Und was ist dort als Erstes am deutlichsten spuerbar?",
    "dark_scene_people_who_question": "Kannst du mir sagen, wer oder was dort fuer dich erkennbar wird?",
    "dark_scene_happening_question": "Was passiert dort in diesem Moment gerade?",
    "dark_scene_age_question": "Wie alt bist du dort? Dein erster Impuls zu deinem Alter reicht.",
    "dark_scene_feeling_intensity_question": "Wie zeigt sich dieses ungute Gefuehl dort gerade? Ist es eher sehr stark, oder wie wuerdest du es in diesem Moment beschreiben?",
    "dark_scene_immediate_feeling_question": (
        "Wenn du da jetzt direkt hineinspuerst: Wie zeigt sich dieses Gefuehl in diesem Moment ganz unmittelbar?"
    ),
    "group_branch_intro": (
        "Dann nimm jetzt genau diese Gruppe naeher zu dir heran: {trigger_focus_ref}. Genau dort zeigt sich dieses Gefuehl in diesem Moment besonders deutlich.\n\n"
        "Lass diese Gruppe nun so weit naeher kommen, bis das Bild klar vor dir steht."
    ),
    "group_image_ready_question": "Ist das Bild mit dieser Gruppe jetzt direkt vor dir: {trigger_focus_ref}?",
    "group_source_kind_question": (
        "Wenn du jetzt vor dieser Gruppe stehst: Hast du das Gefuehl, dass dieses ungute Gefuehl von der ganzen Gruppe kommt, "
        "nur von einer bestimmten Person oder von mehreren einzelnen Personen innerhalb der Gruppe?"
    ),
    "group_whole_scope_question": (
        "Hast du das Gefuehl, dass wir mit jedem Einzelnen aus dieser Gruppe sprechen muessen, damit sich das aufloesen kann, "
        "oder reicht es, wenn wir eine stellvertretende Person aus dieser Gruppe auswaehlen, die fuer diesen Gruppendruck oder diese Dynamik steht?"
    ),
    "group_select_representative_intro": "Dann waehlen wir jetzt genau die Person aus dieser Gruppe aus, die fuer dich diese Dynamik am staerksten verkoerpert.",
    "group_representative_name_question": "Welche Person aus dieser Gruppe moechtest du als stellvertretende Person benennen?",
    "group_specific_person_intro": "Dann wird deutlicher, dass nicht die ganze Gruppe selbst entscheidend ist, sondern eine bestimmte Person daraus.",
    "group_specific_person_question": "Welche Person aus dieser Gruppe ist es genau?",
    "group_multiple_people_intro": "Dann wird deutlich, dass sich diese Dynamik nicht nur auf eine Person reduzieren laesst, sondern dass mehrere einzelne Personen darin wichtig sind.",
    "group_multiple_people_question": "Welche Person davon ist fuer dich im Moment die wichtigste, mit der wir beginnen sollen?",
    "group_multiple_required_intro": "Dann halten wir fest: Hier reicht nicht nur eine stellvertretende Person, sondern wir muessen mehrere Personen aus dieser Gruppe einzeln einbeziehen.",
    "group_multiple_required_question": "Mit welcher Person aus dieser Gruppe sollen wir beginnen?",
    "group_bring_person_forward": "Dann lass {named_person} jetzt naeher zu dir kommen, bis {named_person} klar vor dir steht und fuer dich deutlich wahrnehmbar wird.",
    "group_person_ready_question": "Steht {named_person} jetzt klar vor dir?",
    "group_person_handoff": "Dann arbeiten wir jetzt mit {named_person} weiter.",
    "group_person_trigger_reason_question": "Schau jetzt {named_person} noch einmal genau an. Was genau ist es an {named_person}, das dieses ungute Gefuehl in dir ausloest?",
    "group_person_trigger_role_question": "Geht dieses ungute Gefuehl direkt von {named_person} aus, oder steht {named_person} eher stellvertretend fuer etwas Groesseres, das in dieser Situation wirksam ist?",
    "group_person_trigger_core_question": "Weisst du schon, warum {named_person} in dieser Situation so reagiert oder was dahinterliegt? Wenn das noch nicht klar ist, ist auch das eine stimmige Rueckmeldung.",
    "group_next_person_check_question": "Gibt es in dieser Gruppe jetzt noch eine weitere Person, die wir ebenfalls noch loesen sollten?",
    "group_next_person_name_question": "Welche weitere Person aus dieser Gruppe ist als naechstes wichtig?",
    "person_switch_ready_intro": (
        "Gut. Dann erklaere ich dir kurz den naechsten Schritt.\n\n"
        "Wir wechseln gleich fuer einen Moment in die Perspektive von {named_person_ref}. "
        "Nicht damit du dich darin verlierst, sondern damit du wahrnehmen kannst, was bei {named_person_ref} "
        "in diesem Moment innerlich los war, warum {named_person} so reagiert hat und was dahinterstand. "
        "Danach holen wir dich wieder klar in deine eigene Perspektive zurueck."
    ),
    "person_switch_ready_question": "Bist du bereit, jetzt in die Perspektive von {named_person_ref} zu wechseln?",
    "person_switch_intro": (
        "Okay. Dann zaehle ich jetzt von drei bis null. Bei null schnippe ich mit den Fingern, und du springst mit deinem Geist aus deinem eigenen Koerper direkt in den Koerper von {named_person_ref}. "
        "In dem Moment, in dem du in den Energiekreis von {named_person_ref} eintrittst, kannst du wahrnehmen, wie {named_person} in diesem Moment gedacht, gefuehlt und erlebt hat. "
        "Du wechselst direkt in die Perspektive von {named_person_ref}. Du nimmst dort wahr, fuehlst und denkst genau so, wie es in diesem Moment war, und kannst zugleich {customer_ref} vor dir wahrnehmen.\n\n"
        "Drei. Zwei. Eins. Null. (Fingerschnipp)"
    ),
    "person_switch_hears_question": "Hallo {named_person}, hoerst du mich?",
    "person_switch_sees_customer_question": "Siehst du, dass {customer_ref} vor dir steht?",
    "person_switch_sees_impact_question": "Siehst du, dass {customer_ref} vor dir steht und dass dein Verhalten in diesem Moment bei {customer_ref_dat} etwas ausloest?",
    "person_switch_heard_customer_question": "Hast du verstanden, was dein Verhalten in diesem Moment bei {customer_ref_dat} ausloest und warum dieser Moment spaeter wichtig werden kann?",
    "person_switch_why_question": "Bleib in der Perspektive von {named_person_ref}: Was ist in dir in diesem Moment los, wie geht es dir dort, und warum reagierst du {customer_ref_dat} gegenueber genau so?",
    "person_switch_aware_trigger_question": "Ist dir bewusst, dass genau dieser Moment spaeter fuer {customer_ref} zu einem Ausloeser fuer das Rauchen wird?",
    "person_switch_return_intro": (
        "Dann zaehle ich jetzt nochmals von drei bis null. Bei null schnippe ich mit den Fingern, und du springst wieder in deinen eigenen Koerper zurueck.\n\n"
        "Drei. Zwei. Eins. Null. (Fingerschnipp)"
    ),
    "person_switch_self_heard_question": "Hast du gehoert, was {named_person} gesagt hat, und konntest du wahrnehmen, was der Grund dafuer war?",
    "person_switch_self_understands_question": "Verstehst du jetzt schon ein wenig besser, warum {named_person} so reagiert hat, und hilft dir das gerade?",
    "person_switch_reenter_intro": "Dann wechseln wir noch einmal in die Perspektive von {named_person_ref}. Ich zaehle wieder von drei bis null, dann folgt der Fingerschnipp.",
    "person_switch_fault_question": "Erkennst du den Fehler in deinem Verhalten in diesem Moment?",
    "person_switch_apology_question": "Dann entschuldige dich jetzt klar und ehrlich bei {customer_ref_dat}. Sprich die Entschuldigung jetzt aus.",
    "person_switch_insight_question": "Hoer ganz tief in dich hinein: Was erkennst du ueber deinen Anteil in diesem Moment?",
    "person_switch_back_to_self": "Dann gehst du wieder in deinen eigenen Koerper zurueck. Ich zaehle von drei bis null, dann folgt der Fingerschnipp.",
    "person_switch_apology_heard": "Du hast {named_person} jetzt gehoert, und {named_person} hat den Mut gefunden, sich bei dir zu entschuldigen.",
    "person_switch_forgiveness_question": "Kannst du {named_person} dafuer verzeihen? Verzeihen heisst hier nicht gutheissen, sondern dass du es fuer dich abschliessen und hinter dir lassen kannst. Waere das fuer dich jetzt in Ordnung?",
    "person_switch_forgiveness_clarify_question": "Was muss zuerst noch geklaert werden, damit du es fuer dich abschliessen und hinter dir lassen kannst?",
    "person_switch_forgiveness_check_question": "Hilft dir diese Klaerung jetzt schon ein Stueck weiter?",
    "person_switch_without_apology_question": "Hilft dir dieses Verstaendnis trotzdem weiter, auch wenn {named_person} es gerade noch nicht voll annehmen kann?",
    "person_switch_without_apology_fallback": "Du hast vermutlich wahrgenommen, dass {named_person} innerlich beruehrt ist, es aber aus eigener Sturheit, Scham oder Schutz noch nicht voll aussprechen kann. Das ist ein Thema, das die andere Person fuer sich loesen muss und nichts ueber deinen Wert sagt.",
    "person_switch_without_apology_check_question": "Hilft dir diese Einordnung fuer dich weiter?",
    "person_switch_open_issues_question": "Gibt es in diesem Moment jetzt noch andere Menschen oder andere Dinge, die fuer dich stoerend sind und die wir noch loesen sollten?",
    "person_switch_open_issues_detail_question": "Was genau ist dort noch offen, damit wir es als naechstes gezielt loesen koennen?",
    "person_switch_direct_expression_question": "Wenn {named_person} jetzt vor dir steht: Was moechtest du {named_person} jetzt sagen?",
    "person_switch_direct_response_question": "Und was kommt jetzt von {named_person} zurueck, oder was nimmst du jetzt wahr?",
    "group_resolution_complete": "Lass genau das jetzt einen Moment wirken. Von dort aus holen wir nun dein jetziges Ich mit dazu.",
    "abort": "In Ordnung. Dann unterbrechen wir hier.",
}


NODE_SPECS: dict[str, SemanticNodeSpec | ScriptNodeSpec] = {
    "hell_light_level": SemanticNodeSpec(
        node_id="hell_light_level",
        question_text="Was nimmst du dort wahr? Ist es eher hell oder dunkel?",
        node_goal="Ordne die Ursprungsszene im V2-Einstieg zuerst als eher hell oder eher dunkel ein. Wenn sie hell ist, fuehre in den Hell-Zweig. Wenn sie dunkel ist, fuehre zuerst in den Dunkel-Szenenblock weiter.",
        intent_meanings={
            "hell_light": "Der Kunde beschreibt die Szene als hell, sehr hell, lichtvoll, weiss, leer-hell oder klar hell. Hell und sehr hell fuehren hier beide in denselben Hell-Gefuehlszweig.",
            "darker_or_other": "Der Kunde beschreibt die Szene als eher dunkel, dunkel, schattig oder klar nicht hell.",
            "both": "Der Kunde beschreibt zugleich helle und dunklere Anteile oder sagt sinngemaess beides.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung von hell, dunkel oder beides.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie er antworten oder worauf er achten soll.",
            "support_needed": "Der Kunde braucht Stabilisierung oder wirkt ueberfordert, bevor er die Szene einordnen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "hell_feel_branch",
            "dark_scene_perception",
            "dark_follow_darker_intro",
            "hell_light_level",
            "abort_confirmation",
        ),
        routing_rules={
            "hell_light": {"action": "transition", "next_node": "hell_feel_branch"},
            "darker_or_other": {"action": "transition", "next_node": "dark_scene_perception"},
            "both": {"action": "transition", "next_node": "dark_follow_darker_intro"},
            "unclear": {"action": "clarify", "next_node": "hell_light_level"},
            "repeat": {"action": "repeat", "next_node": "hell_light_level"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "hell_light_level"},
            "support_needed": {"action": "support", "next_node": "hell_light_level"},
        },
        same_node_replies={
            "unclear": NODE_SCRIPTS["clarify_light"],
            "repeat": "Was nimmst du dort wahr? Ist es eher hell oder dunkel?",
            "question": NODE_SCRIPTS["question_light"],
            "support_needed": NODE_SCRIPTS["support_light"],
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "scene_access_followup": SemanticNodeSpec(
        node_id="scene_access_followup",
        question_text=NODE_SCRIPTS["scene_access_followup_question"],
        node_goal="Wenn nach der ersten Wahrnehmungsfrage noch kein klares Bild da ist, oeffne andere Wahrnehmungskanaele wie Hoeren, Riechen, Schmecken oder ein bestimmtes Gefuehl im Koerper. Wenn spaeter doch ein Bild auftaucht, bleibe im Flow und frage direkt nach dem konkreten visuellen Inhalt weiter. Wenn nur nichtvisuelle Wahrnehmung oder weiterhin noch nichts da ist, wechsle in die Body-Bridge-Weiterfuehrung.",
        intent_meanings={
            "visual_hell": "Der Kunde sagt, dass jetzt etwas Sichtbares da ist. Der naechste Schritt klaert direkt, was genau gesehen wird.",
            "visual_dark": "Der Kunde sagt, dass jetzt etwas Sichtbares da ist. Der naechste Schritt klaert direkt, was genau gesehen wird.",
            "nonvisual_access": "Der Kunde nimmt etwas wahr, aber eher ueber Hoeren, Riechen, Schmecken, Spueren, Koerpergefuehl, Druck, Stimme, Atmosphaere oder eine andere nicht primaer visuelle Wahrnehmung.",
            "nothing_yet": "Der Kunde sagt, dass weiterhin noch nichts wahrnehmbar ist, noch kein Bild da ist oder noch nichts davon zugänglich ist.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage oder braucht eine Erklaerung zur Frage.",
            "support_needed": "Der Kunde braucht Stabilisierung oder noch einen sicheren Rahmen, bevor er weiter wahrnehmen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "dark_scene_who",
            "scene_access_body_bridge_intro",
            "scene_access_followup",
            "abort_confirmation",
        ),
        routing_rules={
            "visual_hell": {"action": "transition", "next_node": "dark_scene_who"},
            "visual_dark": {"action": "transition", "next_node": "dark_scene_who"},
            "nonvisual_access": {"action": "transition", "next_node": "scene_access_body_bridge_intro"},
            "nothing_yet": {"action": "transition", "next_node": "scene_access_body_bridge_intro"},
            "unclear": {"action": "clarify", "next_node": "scene_access_followup"},
            "repeat": {"action": "repeat", "next_node": "scene_access_followup"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "scene_access_followup"},
            "support_needed": {"action": "support", "next_node": "scene_access_followup"},
        },
        same_node_replies={
            "unclear": "Es reicht eine kurze Einordnung, ob du etwas hoerst, riechst, schmeckst, ein bestimmtes Gefuehl wahrnimmst oder ob noch nichts greifbar ist.",
            "repeat": NODE_SCRIPTS["scene_access_followup_question"],
            "question": "Es geht nur darum, ob statt eines Bildes ueber einen anderen Sinn schon etwas wahrnehmbar wird, zum Beispiel ueber Hoeren, Riechen, Schmecken oder ein bestimmtes Gefuehl.",
            "support_needed": "Bleib ruhig bei dir. Eine kurze Einordnung genuegt, ob du etwas hoerst, riechst, schmeckst, ein bestimmtes Gefuehl wahrnimmst oder ob noch nichts greifbar ist.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "scene_access_body_bridge_intro": ScriptNodeSpec(
        node_id="scene_access_body_bridge_intro",
        script_text=NODE_SCRIPTS["scene_access_body_bridge_intro"],
        next_node="dark_scene_other_sense",
    ),
    "hell_feel_branch": SemanticNodeSpec(
        node_id="hell_feel_branch",
        question_text=NODE_SCRIPTS["hell_feel_question"],
        node_goal="Erkenne im Hell-Zweig, ob sich dieses Helle fuer den Kunden angenehm oder unangenehm anfuehlt. Wenn es angenehm ist, fuehre direkt in den Hypnose-Loch-Zweig. Wenn es unangenehm ist, fuehre in die Regulation.",
        intent_meanings={
            "pleasant": "Der Kunde beschreibt das Helle als angenehm, sehr angenehm, warm, ruhig, gut, sicher, weit, leer auf gute Weise oder klar positiv. In diesem Knoten fuehren sowohl hell plus angenehm als auch sehr hell plus angenehm direkt in den Hypnose-Loch-Zweig.",
            "unpleasant": "Der Kunde beschreibt das Helle als unangenehm, drueckend, zu viel, stoerend, stressig, bedrohlich oder klar negativ.",
            "reclassified_dark": "Der Kunde korrigiert die vorherige Einordnung und macht klar, dass die Szene eigentlich dunkel oder nicht hell ist. Dann muss zurueck in den Dunkelpfad gewechselt werden, statt den Hell-Zweig weiterzufuehren.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung in angenehm oder unangenehm.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie er antworten oder worauf er achten soll.",
            "support_needed": "Der Kunde braucht Stabilisierung oder wirkt ueberfordert, bevor er das Gefuehl einordnen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "hell_hypnose_loch_intro",
            "hell_regulation_choice",
            "dark_scene_perception",
            "hell_feel_branch",
            "abort_confirmation",
        ),
        routing_rules={
            "pleasant": {"action": "transition", "next_node": "hell_hypnose_loch_intro"},
            "unpleasant": {"action": "transition", "next_node": "hell_regulation_choice"},
            "reclassified_dark": {"action": "transition", "next_node": "dark_scene_perception"},
            "unclear": {"action": "clarify", "next_node": "hell_feel_branch"},
            "repeat": {"action": "repeat", "next_node": "hell_feel_branch"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "hell_feel_branch"},
            "support_needed": {"action": "support", "next_node": "hell_feel_branch"},
        },
        same_node_replies={
            "unclear": NODE_SCRIPTS["clarify_feel"],
            "repeat": NODE_SCRIPTS["hell_feel_question"],
            "question": NODE_SCRIPTS["question_feel"],
            "support_needed": NODE_SCRIPTS["support_feel"],
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "hell_hypnose_loch_intro": ScriptNodeSpec(
        node_id="hell_hypnose_loch_intro",
        script_text=NODE_SCRIPTS["hell_hypnose_loch_block"],
        next_node="hell_hypnose_wait",
    ),
    "hell_hypnose_wait": SemanticNodeSpec(
        node_id="hell_hypnose_wait",
        question_text=NODE_SCRIPTS["hell_hypnose_wait_question"],
        node_goal="Begleite den Hypnose-Loch-Zweig weiter und erkenne, ob sich der Prozess bereits aufgeloest hat, noch im Loesen ist oder noch etwas Zeit braucht.",
        intent_meanings={
            "resolved": "Der Kunde sagt oder meint, dass es bereits aufgeloest, weg, ruhig oder nicht mehr spuerbar ist.",
            "resolving": "Der Kunde beschreibt, dass sich noch etwas loest, bewegt, arbeitet oder leichter wird, aber noch nicht ganz fertig ist.",
            "need_more_time": "Der Kunde braucht noch einen Moment, noch etwas Zeit oder sagt sinngemaess noch nicht ganz.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung in aufgeloest, noch im Loesen oder braucht noch Zeit.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie er den Zustand einordnen soll.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Begleitung, bevor er den Zustand einordnen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "hell_post_resolved_terminal",
            "hell_hypnose_wait",
            "abort_confirmation",
        ),
        routing_rules={
            "resolved": {"action": "transition", "next_node": "hell_post_resolved_terminal"},
            "resolving": {"action": "transition", "next_node": "hell_hypnose_wait"},
            "need_more_time": {"action": "transition", "next_node": "hell_hypnose_wait"},
            "unclear": {"action": "clarify", "next_node": "hell_hypnose_wait"},
            "repeat": {"action": "repeat", "next_node": "hell_hypnose_wait"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "hell_hypnose_wait"},
            "support_needed": {"action": "support", "next_node": "hell_hypnose_wait"},
        },
        same_node_replies={
            "resolving": NODE_SCRIPTS["hell_wait_more"],
            "need_more_time": NODE_SCRIPTS["hell_wait_more"],
            "unclear": NODE_SCRIPTS["clarify_wait"],
            "repeat": NODE_SCRIPTS["hell_hypnose_wait_question"],
            "question": NODE_SCRIPTS["question_wait"],
            "support_needed": NODE_SCRIPTS["support_wait"],
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "hell_post_resolved_terminal": ScriptNodeSpec(
        node_id="hell_post_resolved_terminal",
        script_text=NODE_SCRIPTS["hell_wait_resolved"],
        next_node="dark_known_branch",
    ),
    "hell_regulation_choice": SemanticNodeSpec(
        node_id="hell_regulation_choice",
        question_text=NODE_SCRIPTS["hell_regulation_choice"],
        node_goal="Fuehre den unangenehmen hellen Zustand ueber Regulation weiter. Erkenne, welche Regulierung der Kunde gerade braucht.",
        intent_meanings={
            "distance": "Der Kunde moechte mehr Abstand oder Distanz zur Szene.",
            "less_brightness": "Der Kunde moechte weniger Helligkeit, weicheres Licht oder weniger Intensitaet.",
            "focus": "Der Kunde moechte einen klareren Fokus oder eine gezieltere Ausrichtung.",
            "reclassified_dark": "Der Kunde korrigiert die vorherige Einordnung und macht klar, dass die Szene eigentlich dunkel oder nicht hell ist. Dann muss zurueck in den Dunkelpfad gewechselt werden, statt die Hell-Regulation weiterzufuehren.",
            "unclear": "Es ist nicht klar, welche Regulierung gerade am meisten hilft.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie er auswaehlen soll.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Begleitung, bevor er die Regulierung benennen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "hell_regulation_check",
            "dark_scene_perception",
            "hell_regulation_choice",
            "abort_confirmation",
        ),
        routing_rules={
            "distance": {"action": "transition", "next_node": "hell_regulation_check"},
            "less_brightness": {"action": "transition", "next_node": "hell_regulation_check"},
            "focus": {"action": "transition", "next_node": "hell_regulation_check"},
            "reclassified_dark": {"action": "transition", "next_node": "dark_scene_perception"},
            "unclear": {"action": "clarify", "next_node": "hell_regulation_choice"},
            "repeat": {"action": "repeat", "next_node": "hell_regulation_choice"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "hell_regulation_choice"},
            "support_needed": {"action": "support", "next_node": "hell_regulation_choice"},
        },
        same_node_replies={
            "unclear": NODE_SCRIPTS["clarify_regulation"],
            "repeat": NODE_SCRIPTS["hell_regulation_choice"],
            "question": NODE_SCRIPTS["question_regulation"],
            "support_needed": NODE_SCRIPTS["support_regulation"],
            "abort": NODE_SCRIPTS["abort"],
        },
        entry_script=NODE_SCRIPTS["hell_regulation_intro"],
    ),
    "hell_regulation_check": SemanticNodeSpec(
        node_id="hell_regulation_check",
        question_text=NODE_SCRIPTS["hell_regulation_check"],
        node_goal="Pruefe nach der Regulierung, ob die Szene jetzt weiter hell ist oder ob sie eher dunkel, beides oder deutlich ruhiger wirkt.",
        intent_meanings={
            "still_hell": "Die Szene wirkt jetzt weiterhin eher hell oder hell genug, um erneut im Hell-Zweig weiterzugehen.",
            "dark_or_both_or_quieter": "Die Szene wirkt jetzt eher dunkel, beides oder deutlich ruhiger und fuehrt damit aus dem Hell-Zweig heraus.",
            "unclear": "Die Wirkung nach der Regulierung ist noch nicht klar benennbar.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie er die Wirkung einordnen soll.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Begleitung, bevor er die Wirkung benennen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "hell_feel_branch",
            "dark_known_branch",
            "hell_regulation_check",
            "abort_confirmation",
        ),
        routing_rules={
            "still_hell": {"action": "transition", "next_node": "hell_feel_branch"},
            "dark_or_both_or_quieter": {"action": "transition", "next_node": "dark_known_branch"},
            "unclear": {"action": "clarify", "next_node": "hell_regulation_check"},
            "repeat": {"action": "repeat", "next_node": "hell_regulation_check"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "hell_regulation_check"},
            "support_needed": {"action": "support", "next_node": "hell_regulation_check"},
        },
        same_node_replies={
            "unclear": NODE_SCRIPTS["clarify_regulation_check"],
            "repeat": NODE_SCRIPTS["hell_regulation_check"],
            "question": NODE_SCRIPTS["question_regulation_check"],
            "support_needed": NODE_SCRIPTS["support_regulation_check"],
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "dark_follow_darker_intro": ScriptNodeSpec(
        node_id="dark_follow_darker_intro",
        script_text=NODE_SCRIPTS["dark_follow_darker_intro"],
        next_node="dark_scene_perception",
    ),
    "dark_scene_perception": SemanticNodeSpec(
        node_id="dark_scene_perception",
        question_text=NODE_SCRIPTS["dark_scene_perception_question"],
        node_goal="Klaere im V2-Dunkelzweig zuerst offen, ob in der Szene vor allem etwas gesehen oder gehoert wird. Wenn visuell etwas da ist, fuehre in die visuelle Detailfrage. Wenn auditiv etwas da ist, fuehre in die auditive Detailfrage. Wenn beides da ist, beginne visuell und hole die auditive Spur danach nach. Wenn nichts davon klar ist, fuehre zuerst in die Klaerfrage und danach gegebenenfalls in den Koerper-/Geruch-/Geschmack-Zugang.",
        intent_meanings={
            "visual": "Der Kunde sagt oder meint, dass er etwas sieht oder dass etwas visuell wahrnehmbar ist.",
            "audio": "Der Kunde sagt oder meint, dass er etwas hoert oder dass etwas auditiv wahrnehmbar ist.",
            "both": "Der Kunde sagt oder meint, dass sowohl etwas gesehen als auch etwas gehoert wird.",
            "other_sense": "Der Kunde nimmt nichts klar visuell oder auditiv wahr, aber etwas ueber Koerper, Geruch, Geschmack, Temperatur oder eine andere nichtvisuelle Wahrnehmung.",
            "nothing": "Der Kunde sagt, dass er weder klar etwas sieht noch hoert oder dass gerade nichts davon da ist.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage oder braucht eine Erklaerung zur Frage.",
            "support_needed": "Der Kunde braucht Stabilisierung oder einen sichereren Rahmen, bevor er weiter wahrnehmen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "dark_scene_who",
            "dark_scene_audio_detail",
            "dark_scene_mode_clarify",
            "dark_scene_other_sense",
            "dark_scene_perception",
            "abort_confirmation",
        ),
        routing_rules={
            "visual": {"action": "transition", "next_node": "dark_scene_who"},
            "audio": {"action": "transition", "next_node": "dark_scene_audio_detail"},
            "both": {"action": "transition", "next_node": "dark_scene_who"},
            "other_sense": {"action": "transition", "next_node": "dark_scene_other_sense"},
            "nothing": {"action": "transition", "next_node": "dark_scene_mode_clarify"},
            "unclear": {"action": "clarify", "next_node": "dark_scene_mode_clarify"},
            "repeat": {"action": "repeat", "next_node": "dark_scene_perception"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "dark_scene_perception"},
            "support_needed": {"action": "support", "next_node": "dark_scene_perception"},
        },
        same_node_replies={
            "unclear": "Es reicht eine kurze Einordnung, ob dort eher etwas zu sehen ist, etwas zu hoeren ist oder ob noch nichts davon klar da ist.",
            "repeat": NODE_SCRIPTS["dark_scene_perception_question"],
            "question": "Hier geht es nur darum, ob in dieser Szene vor allem etwas zu sehen oder zu hoeren ist.",
            "support_needed": "Bleib ruhig bei dir. Wenn es klarer wird, reicht eine kurze Einordnung, ob dort eher etwas zu sehen oder zu hoeren ist.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "dark_scene_mode_clarify": SemanticNodeSpec(
        node_id="dark_scene_mode_clarify",
        question_text=NODE_SCRIPTS["dark_scene_mode_clarify_question"],
        node_goal="Klaere nach unklarer erster Rueckmeldung, ob in der Szene eher etwas gesehen, eher etwas gehoert, beides oder weiterhin nichts von beidem wahrnehmbar ist.",
        intent_meanings={
            "visual": "Der Kunde sagt oder meint, dass jetzt etwas visuell wahrnehmbar ist.",
            "audio": "Der Kunde sagt oder meint, dass jetzt etwas auditiv wahrnehmbar ist.",
            "both": "Der Kunde sagt oder meint, dass sowohl etwas gesehen als auch etwas gehoert wird.",
            "other_sense": "Der Kunde nimmt stattdessen etwas ueber Koerper, Geruch, Geschmack, Temperatur oder eine andere nichtvisuelle Wahrnehmung wahr.",
            "nothing": "Der Kunde sagt, dass weiterhin weder etwas klar gesehen noch gehoert wird.",
            "unclear": "Die Antwort bleibt unklar.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage oder braucht eine Erklaerung zur Frage.",
            "support_needed": "Der Kunde braucht Stabilisierung oder einen sichereren Rahmen, bevor er weiter wahrnehmen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "dark_scene_who",
            "dark_scene_audio_detail",
            "dark_scene_other_sense",
            "dark_scene_mode_clarify",
            "abort_confirmation",
        ),
        routing_rules={
            "visual": {"action": "transition", "next_node": "dark_scene_who"},
            "audio": {"action": "transition", "next_node": "dark_scene_audio_detail"},
            "both": {"action": "transition", "next_node": "dark_scene_who"},
            "other_sense": {"action": "transition", "next_node": "dark_scene_other_sense"},
            "nothing": {"action": "transition", "next_node": "dark_scene_other_sense"},
            "unclear": {"action": "clarify", "next_node": "dark_scene_mode_clarify"},
            "repeat": {"action": "repeat", "next_node": "dark_scene_mode_clarify"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "dark_scene_mode_clarify"},
            "support_needed": {"action": "support", "next_node": "dark_scene_mode_clarify"},
        },
        same_node_replies={
            "unclear": "Es reicht eine kurze Einordnung, ob du dort eher jemanden siehst, etwas hoerst oder ob beides noch nicht klar da ist.",
            "repeat": NODE_SCRIPTS["dark_scene_mode_clarify_question"],
            "question": "Hier geht es nur darum, ob in dieser Szene eher etwas gesehen oder gehoert wird.",
            "support_needed": "Bleib ruhig bei dir. Wenn es klarer wird, reicht eine kurze Einordnung, ob du dort eher jemanden siehst oder etwas hoerst.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "dark_scene_who": _make_ready_transition_node(
        "dark_scene_who",
        NODE_SCRIPTS["dark_scene_who_question"],
        "Klaere im V2-Dunkelzweig, was dort konkret gesehen wird, zum Beispiel Personen, eine Gruppe, ein Raum, ein Gegenstand oder eine bestimmte Szene.",
        "dark_scene_age",
        clarify_reply="Wenn etwas klarer wird, sag einfach kurz, was dort zu sehen ist. Dein erster Eindruck dazu reicht.",
        question_reply="Hier geht es nur darum, was dort konkret sichtbar wird, zum Beispiel Personen, eine Gruppe, ein Raum oder etwas anderes.",
        support_reply="Wenn dort etwas klarer sichtbar wird, reicht eine kurze Beschreibung davon, was du dort siehst.",
        ready_meaning="Der Kunde benennt oder beschreibt frei, was dort konkret sichtbar wird, zum Beispiel Personen, eine Gruppe, ein Raum oder etwas anderes.",
    ),
    "dark_scene_audio_detail": _make_ready_transition_node(
        "dark_scene_audio_detail",
        NODE_SCRIPTS["dark_scene_audio_detail_question"],
        "Klaere im V2-Dunkelzweig, was dort konkret gehoert wird.",
        "dark_scene_age",
        clarify_reply="Wenn etwas deutlicher hoerbar wird, sag einfach kurz, was du dort hoerst. Dein erster Eindruck dazu reicht.",
        question_reply="Hier geht es nur darum, was dort konkret hoerbar wird.",
        support_reply="Wenn dort etwas klarer hoerbar wird, reicht eine kurze Beschreibung davon, was du dort hoerst.",
        ready_meaning="Der Kunde benennt oder beschreibt frei, was dort konkret gehoert wird.",
    ),
    "dark_scene_other_sense": _make_ready_transition_node(
        "dark_scene_other_sense",
        NODE_SCRIPTS["dark_scene_other_sense_question"],
        "Wenn nichts klar gesehen oder gehoert wird, oeffne im V2-Dunkelzweig den Zugang ueber Koerper, Geruch, Geschmack oder Temperatur.",
        "dark_scene_first_spuerbar",
        clarify_reply="Wenn sich ueber Koerper, Geruch, Geschmack oder Temperatur etwas zeigt, sag einfach kurz, was wahrnehmbar wird. Dein erster Eindruck dazu reicht.",
        question_reply="Hier geht es nur darum, ob dort ueber Koerper, Geruch, Geschmack oder Temperatur etwas wahrnehmbar wird.",
        support_reply="Bleib ruhig bei dir und spuer einfach nach, ob dort eher Druck, Enge, Temperatur, ein Geruch oder etwas Aehnliches auftaucht.",
        ready_meaning="Der Kunde beschreibt frei, was dort ueber Koerper, Geruch, Geschmack oder Temperatur wahrnehmbar wird.",
    ),
    "dark_scene_first_spuerbar": _make_ready_transition_node(
        "dark_scene_first_spuerbar",
        NODE_SCRIPTS["dark_scene_first_spuerbar_question"],
        "Klaere im V2-Dunkelzweig, was dort nach dem anderen Sinneszugang als Erstes am deutlichsten spuerbar wird.",
        "dark_scene_age",
        clarify_reply="Wenn etwas deutlicher spuerbar wird, sag einfach kurz, was dort als Erstes am staerksten auffaellt. Dein erster Eindruck dazu reicht.",
        question_reply="Hier geht es nur darum, was dort als Erstes am deutlichsten spuerbar wird.",
        support_reply="Bleib ruhig dabei. Wenn es deutlicher wird, reicht eine kurze Beschreibung davon, was dort zuerst am staerksten spuerbar ist.",
        ready_meaning="Der Kunde beschreibt frei, was dort als Erstes am deutlichsten spuerbar wird.",
    ),
    "dark_scene_people_who": _make_ready_transition_node(
        "dark_scene_people_who",
        NODE_SCRIPTS["dark_scene_people_who_question"],
        "Wenn im visuellen Detail Personen oder Gruppen erkannt wurden, klaere reduziert, wer dort genau zu sehen ist, ohne schon in die spaetere Aufloesungsvertiefung zu springen.",
        "dark_scene_age",
        clarify_reply="Sobald du dort etwas klarer erkennen kannst, sag einfach kurz, wer oder was dort fuer dich wahrnehmbar wird oder was du sonst noch dort bemerkst. Dein erster Eindruck dazu reicht.",
        question_reply="Hier geht es nur darum, was oder wer dort fuer dich genauer erkennbar wird.",
        support_reply="Bleib noch einen Moment dabei. Wenn etwas klarer wird, reicht eine kurze Beschreibung davon, wer oder was dort fuer dich wahrnehmbar ist.",
        ready_meaning="Der Kunde benennt oder beschreibt frei, wer oder was dort fuer ihn erkennbar wird.",
    ),
    "dark_scene_happening": _make_ready_transition_node(
        "dark_scene_happening",
        NODE_SCRIPTS["dark_scene_happening_question"],
        "Starte die Aufloesungsvertiefung nach gefundenem Ursprung damit, was in diesem Moment der Szene gerade geschieht.",
        "origin_trigger_source",
        clarify_reply="Wenn sich der Moment klarer zeigt, sag einfach kurz, was dort gerade geschieht. Dein erster Eindruck dazu reicht.",
        question_reply="Hier geht es nur darum, was in dieser Szene gerade geschieht.",
        support_reply="Schau noch einen Moment hin. Wenn es klarer wird, reicht eine kurze Beschreibung davon, was dort gerade passiert.",
        ready_meaning="Der Kunde beschreibt frei, was in diesem Moment der Szene gerade passiert.",
    ),
    "dark_scene_age": _make_ready_transition_node(
        "dark_scene_age",
        NODE_SCRIPTS["dark_scene_age_question"],
        "Klaere im V2-Dunkelzweig das spontane Alter in dieser Szene.",
        "dark_scene_feeling_intensity",
        clarify_reply="Nimm einfach den ersten Impuls zu deinem Alter dort. Das reicht voellig.",
        question_reply="Hier reicht dein spontanes Alter in dieser Szene.",
        support_reply="Nimm kurz den ersten Impuls dazu auf. Dein Alter dort als erste Rueckmeldung genuegt.",
        ready_meaning="Der Kunde nennt oder beschreibt frei ein spontanes Alter in dieser Szene.",
    ),
    "dark_scene_feeling_intensity": SemanticNodeSpec(
        node_id="dark_scene_feeling_intensity",
        question_text=NODE_SCRIPTS["dark_scene_feeling_intensity_question"],
        node_goal="Klaere im V2-Dunkelzweig, wie das ungute Gefuehl dort wahrgenommen wird und wie stark es ist. Wenn die Antwort das unmittelbare Gefuehl bereits klar mitliefert, frage nicht noch einmal danach.",
        intent_meanings={
            "feeling_and_intensity": "Der Kunde beschreibt bereits ein konkretes unmittelbares Gefuehl oder eine koerperlich-emotionale Qualitaet und eventuell auch die Staerke, zum Beispiel 'druck in der brust', 'angst', 'enge', 'kloÃŸ im hals', 'ich fuehle druck' oder aehnliche direkte Gefuehls- und Koerperbeschreibungen. Dann ist keine weitere Rueckfrage nach dem unmittelbaren Gefuehl noetig.",
            "intensity_only": "Der Kunde beschreibt vor allem die Intensitaet oder allgemeine Staerke, ohne das unmittelbare Gefuehl schon klar zu benennen, zum Beispiel 'sehr stark', 'intensiv', 'acht von zehn' oder aehnliche reine Staerke-Angaben.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage oder braucht eine Erklaerung zur Frage.",
            "support_needed": "Der Kunde braucht Stabilisierung, Orientierung oder noch einen Moment, bevor er antworten kann.",
        },
        allowed_actions=("transition", "clarify", "repeat", "answer_question", "support", "abort"),
        allowed_next_nodes=("dark_known_branch", "dark_scene_immediate_feeling", "dark_scene_feeling_intensity", "abort_confirmation"),
        routing_rules={
            "feeling_and_intensity": {"action": "transition", "next_node": "dark_known_branch"},
            "intensity_only": {"action": "transition", "next_node": "dark_scene_immediate_feeling"},
            "unclear": {"action": "clarify", "next_node": "dark_scene_feeling_intensity"},
            "repeat": {"action": "repeat", "next_node": "dark_scene_feeling_intensity"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "dark_scene_feeling_intensity"},
            "support_needed": {"action": "support", "next_node": "dark_scene_feeling_intensity"},
        },
        same_node_replies={
            "unclear": "Nimm kurz wahr, wie sich dieses ungute Gefuehl dort gerade zeigt oder wie stark es sich fuer dich in diesem Moment anfuehlt. Dein erster Eindruck reicht.",
            "repeat": NODE_SCRIPTS["dark_scene_feeling_intensity_question"],
            "question": "Hier geht es nur darum, wie sich dieses ungute Gefuehl in dieser Szene gerade zeigt oder anfuehlt. Wenn es gerade nicht klar spuerbar ist, ist auch genau das eine brauchbare Rueckmeldung.",
            "support_needed": "Dann geh jetzt nicht tiefer in die Szene. Nimm zuerst den Abstand wahr, der sich fuer dich gerade sicher anfuehlt, spuer den Stuhl unter dir und komm mit dem Atem wieder etwas mehr bei dir an. Wenn das Gefuehl wieder greifbar, aber gut aushaltbar ist, reicht eine kurze Rueckmeldung dazu.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "dark_scene_immediate_feeling": _make_ready_transition_node(
        "dark_scene_immediate_feeling",
        NODE_SCRIPTS["dark_scene_immediate_feeling_question"],
        "Klaere im V2-Dunkelzweig, was in diesem Moment ganz unmittelbar gefuehlt wird, bevor die bekannte-oder-neu-Frage gestellt wird.",
        "dark_known_branch",
        clarify_reply="Ich meine nicht, wie stark es ist, sondern wie sich dieses Gefuehl in diesem Moment direkt zeigt, wenn du kurz hineinspuerst.",
        question_reply="Hier geht es nicht mehr um die Staerke, sondern nur darum, wie sich dieses Gefuehl in diesem Moment ganz unmittelbar zeigt.",
        support_reply="Bleib einen Moment bei dir und lass erst etwas Abstand in die Szene kommen. Spuer den Atem und den Kontakt zum Stuhl. Wenn sich das Gefuehl wieder greifbar und gut aushaltbar zeigt, reicht eine kurze Rueckmeldung dazu.",
        ready_meaning="Der Kunde nennt oder beschreibt frei, was in diesem Moment ganz unmittelbar gefuehlt wird. Auch eine sehr kurze, aber inhaltlich passende Antwort wie 'druck', 'angst', 'enge', 'trauer', 'scham', 'wut', 'ohnmacht' oder 'einsamkeit' zaehlt hier bereits als brauchbare Antwort.",
    ),
    "dark_known_branch": SemanticNodeSpec(
        node_id="dark_known_branch",
        question_text=NODE_SCRIPTS["dark_known_question"],
        node_goal="Erkenne nach dem dunkleren emotionalen Kern, ob dieses Gefuehl dem Kunden aus frueheren Momenten bereits bekannt ist oder ob es sich hier um den ersten Ursprung handelt.",
        intent_meanings={
            "known": "Der Kunde sagt oder meint, dass dieses Gefuehl schon aus frueheren Momenten bekannt ist, schon einmal da war oder sich vertraut anfuehlt.",
            "new": "Der Kunde sagt oder meint, dass dieses Gefuehl hier zum ersten Mal da ist oder dass dies der erste Ursprung ist.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung in bekannt oder zum ersten Mal.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie er die Frage einordnen soll.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Begleitung, bevor er das Gefuehl einordnen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "dark_backtrace_terminal",
            "dark_origin_terminal",
            "dark_known_branch",
            "abort_confirmation",
        ),
        routing_rules={
            "known": {"action": "transition", "next_node": "dark_backtrace_terminal"},
            "new": {"action": "transition", "next_node": "dark_origin_terminal"},
            "unclear": {"action": "clarify", "next_node": "dark_known_branch"},
            "repeat": {"action": "repeat", "next_node": "dark_known_branch"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "dark_known_branch"},
            "support_needed": {"action": "support", "next_node": "dark_known_branch"},
        },
        same_node_replies={
            "unclear": NODE_SCRIPTS["clarify_dark_known"],
            "repeat": NODE_SCRIPTS["dark_known_question"],
            "question": NODE_SCRIPTS["question_dark_known"],
            "support_needed": NODE_SCRIPTS["support_dark_known"],
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "dark_backtrace_terminal": ScriptNodeSpec(
        node_id="dark_backtrace_terminal",
        script_text=NODE_SCRIPTS["dark_backtrace_intro"],
        next_node=None,
    ),
    "dark_origin_terminal": ScriptNodeSpec(
        node_id="dark_origin_terminal",
        script_text=NODE_SCRIPTS["dark_origin_intro"],
        next_node=None,
    ),
    "origin_trigger_source": _make_ready_transition_node(
        "origin_trigger_source",
        NODE_SCRIPTS["origin_trigger_source_question"],
        "Klaere in der Ursprungsszene, was oder wer das ungute Gefuehl dort am staerksten ausloest, damit der weitere Aufloesungspfad sauber bestimmt werden kann.",
        "origin_trigger_known_branch",
        clarify_reply="Bleib nur bei deinem ersten Eindruck: Ist es dort eher eine Person, eine Gruppe, ein Blick, ein Satz, eine Handlung oder etwas anderes, das dieses ungute Gefuehl am staerksten in dir ausloest?",
        question_reply="Hier geht es nur darum, den staerksten Ausloeser in diesem Moment zu benennen. Das kann zum Beispiel eine Person, eine Gruppe, ein Blick, ein Satz, eine Handlung oder etwas anderes in dieser Situation sein.",
        support_reply="Bleib noch einen Moment in dieser Szene und nimm nur das wahr, was dort am staerksten auf dieses ungute Gefuehl wirkt. Der erste Eindruck dazu genuegt voellig.",
        ready_meaning="Der Kunde nennt oder beschreibt frei, was oder wer dieses ungute Gefuehl in der Ursprungsszene am staerksten ausloest.",
    ),
    "origin_trigger_known_branch": SemanticNodeSpec(
        node_id="origin_trigger_known_branch",
        question_text=NODE_SCRIPTS["origin_trigger_known_question"],
        node_goal=(
            "Klaere nach dem benannten Ausloeser zuerst, ob das in dieser Szene auftauchende Gefuehl "
            "dem Kunden bereits aus frueheren Momenten bekannt ist oder ob es sich hier neu zeigt. "
            "Wenn es bereits bekannt ist, fuehrt der Pfad weiter zurueck. Wenn es sich hier neu zeigt, "
            "wird genau in dieser Szene weitergearbeitet."
        ),
        intent_meanings={
            "known": (
                "Der Kunde sagt oder meint, dass dieses Gefuehl schon aus frueheren Momenten bekannt ist, "
                "schon einmal da war oder sich vertraut anfuehlt."
            ),
            "new": (
                "Der Kunde sagt oder meint, dass dieses Gefuehl sich hier zum ersten Mal zeigt "
                "oder ihm in dieser Szene neu begegnet."
            ),
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie diese Einordnung gemeint ist.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Orientierung, bevor er diese Einordnung treffen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "dark_backtrace_terminal",
            "origin_cause_owner",
            "origin_trigger_known_branch",
            "abort_confirmation",
        ),
        routing_rules={
            "known": {"action": "transition", "next_node": "dark_backtrace_terminal"},
            "new": {"action": "transition", "next_node": "origin_cause_owner"},
            "unclear": {"action": "clarify", "next_node": "origin_trigger_known_branch"},
            "repeat": {"action": "repeat", "next_node": "origin_trigger_known_branch"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "origin_trigger_known_branch"},
            "support_needed": {"action": "support", "next_node": "origin_trigger_known_branch"},
        },
        same_node_replies={
            "unclear": NODE_SCRIPTS["clarify_dark_known"],
            "repeat": NODE_SCRIPTS["origin_trigger_known_question"],
            "question": NODE_SCRIPTS["question_dark_known"],
            "support_needed": NODE_SCRIPTS["support_dark_known"],
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "origin_scene_relevance": SemanticNodeSpec(
        node_id="origin_scene_relevance",
        question_text=NODE_SCRIPTS["origin_scene_relevance_question"],
        node_goal=(
            "Klaere nach dem benannten Ausloeser zuerst, ob diese Szene jetzt direkt bearbeitet werden soll "
            "oder ob sie eher auf einen noch frueheren Entstehungspunkt verweist."
        ),
        intent_meanings={
            "resolve_here": (
                "Der Kunde macht klar, dass diese Szene oder das hier benannte Material genau hier weiter bearbeitet "
                "oder aufgeloest werden sollte."
            ),
            "older_origin": (
                "Der Kunde macht klar, dass diese Szene eher noch weiter zurueck fuehrt, noch nicht der eigentliche "
                "Ursprung ist oder auf einen frueheren Entstehungspunkt verweist."
            ),
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie diese Einordnung gemeint ist.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Orientierung, bevor er diese Einordnung treffen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=("origin_cause_owner", "dark_backtrace_terminal", "origin_scene_relevance", "abort_confirmation"),
        routing_rules={
            "resolve_here": {"action": "transition", "next_node": "origin_cause_owner"},
            "older_origin": {"action": "transition", "next_node": "dark_backtrace_terminal"},
            "unclear": {"action": "clarify", "next_node": "origin_scene_relevance"},
            "repeat": {"action": "repeat", "next_node": "origin_scene_relevance"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "origin_scene_relevance"},
            "support_needed": {"action": "support", "next_node": "origin_scene_relevance"},
        },
        same_node_replies={
            "unclear": (
                "Spuer noch einmal kurz nach: Ist in genau dieser Szene etwas, das wir hier anschauen oder loesen sollten, "
                "oder merkst du eher, dass dich dieses Gefuehl noch weiter zu etwas Frueherem fuehrt?"
            ),
            "repeat": NODE_SCRIPTS["origin_scene_relevance_question"],
            "question": (
                "Ich meine: Zeigt das, was du gerade beschrieben hast, eher etwas, das wir genau in dieser Szene "
                "anschauen oder loesen sollten, oder merkst du eher, dass dieses Gefuehl noch weiter zu einem frueheren Ursprung fuehrt?"
            ),
            "support_needed": (
                "Bleib ruhig bei dem, was dort auftaucht. Wenn dir klarer wird, ob in genau dieser Szene etwas "
                "angeschaut oder geloest werden sollte oder ob dich dieses Gefuehl eher noch weiter zurueck fuehrt, "
                "reicht eine kurze Rueckmeldung."
            ),
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "origin_cause_owner": SemanticNodeSpec(
        node_id="origin_cause_owner",
        question_text=NODE_SCRIPTS["origin_cause_owner_question"],
        node_goal="Klaere, ob der eigentliche Grund eher in der Person selbst liegt oder eher bei jemand anderem liegt.",
        intent_meanings={
            "self": "Der Kunde beschreibt, dass der eigentliche Grund eher in ihm selbst, in einem inneren Anteil oder in seiner eigenen Reaktion liegt.",
            "someone_else": "Der Kunde beschreibt, dass der eigentliche Grund eher bei jemand anderem, einer Person oder einer Gruppe liegt.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie diese Einordnung gemeint ist.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Orientierung, bevor er diese Einordnung treffen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=("origin_self_resolution_intro", "origin_other_target_kind", "origin_cause_owner", "abort_confirmation"),
        routing_rules={
            "self": {"action": "transition", "next_node": "origin_self_resolution_intro"},
            "someone_else": {"action": "transition", "next_node": "origin_other_target_kind"},
            "unclear": {"action": "clarify", "next_node": "origin_cause_owner"},
            "repeat": {"action": "repeat", "next_node": "origin_cause_owner"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "origin_cause_owner"},
            "support_needed": {"action": "support", "next_node": "origin_cause_owner"},
        },
        same_node_replies={
            "unclear": "Spuer noch einmal kurz nach: Geht es hier eher um etwas in dir selbst, das dadurch beruehrt wird, oder liegt der eigentliche Ausloeser eher bei dem, was du gerade benannt hast?",
            "repeat": NODE_SCRIPTS["origin_cause_owner_question"],
            "question": "Ich meine: Kommt das ungute Gefuehl vor allem daher, dass in dir selbst etwas beruehrt wird, oder ist das, was du gerade benannt hast, der eigentliche Ausloeser?",
            "support_needed": "Spuer einfach weiter nach. Wenn dir klarer wird, ob dieses Thema eher etwas in dir selbst beruehrt oder ob der eigentliche Ausloeser eher bei dem liegt, was du gerade benannt hast, genuegt eine kurze Rueckmeldung.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "origin_other_target_kind": SemanticNodeSpec(
        node_id="origin_other_target_kind",
        question_text=NODE_SCRIPTS["origin_other_target_kind_question"],
        node_goal="Klaere, ob das benannte Gegenueber eher eine Gruppe, eher eine bestimmte Person oder eher etwas anderes in der Situation ist, damit der passende Zwischenblock vor dem Switch gewaehlt wird.",
        intent_meanings={
            "group": "Der Kunde beschreibt, dass das Benannte eher eine Gruppe, mehrere Menschen oder eine Gruppendynamik ist.",
            "person": "Der Kunde beschreibt, dass das Benannte eher eine bestimmte einzelne Person ist.",
            "other": "Der Kunde beschreibt, dass das Benannte weder eine einzelne Person noch eine Gruppe ist, sondern ein inhaltlich sinnvoller nicht-personaler Ausloeser innerhalb der Situation, zum Beispiel ein Geruch, eine Farbe, ein Blick, ein Verhalten, ein Ereignis oder ein Ort. Beliebige Nonsense- oder Platzhalterworte wie 'Banane' sind hier niemals `other`, sondern `unclear`.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie diese Einordnung gemeint ist.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Orientierung, bevor er diese Einordnung treffen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=("group_branch_intro", "origin_person_name", "origin_person_branch_intro", "origin_self_resolution_intro", "origin_other_target_kind", "abort_confirmation"),
        routing_rules={
            "group": {"action": "transition", "next_node": "group_branch_intro"},
            "person": {"action": "transition", "next_node": "origin_person_name"},
            "other": {"action": "transition", "next_node": "origin_self_resolution_intro"},
            "unclear": {"action": "clarify", "next_node": "origin_other_target_kind"},
            "repeat": {"action": "repeat", "next_node": "origin_other_target_kind"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "origin_other_target_kind"},
            "support_needed": {"action": "support", "next_node": "origin_other_target_kind"},
        },
        same_node_replies={
            "unclear": "Spuer noch einmal kurz nach: Ist das, was du dort meinst, eher eine Gruppe, eher eine bestimmte Person oder eher etwas anderes in dieser Situation?",
            "repeat": NODE_SCRIPTS["origin_other_target_kind_question"],
            "question": "Hier geht es nur darum, ob das, was du dort meinst, eher eine Gruppe, eher eine bestimmte einzelne Person oder eher etwas anderes in dieser Situation ist.",
            "support_needed": "Spuer einfach weiter nach. Wenn dir klarer wird, ob du dort eher eine Gruppe, eher eine bestimmte Person oder eher etwas anderes in dieser Situation meinst, genuegt eine kurze Rueckmeldung.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "origin_person_name": SemanticNodeSpec(
        node_id="origin_person_name",
        question_text=NODE_SCRIPTS["origin_person_name_question"],
        node_goal="Klaere zuerst, ob die gemeinte Person bereits konkret erkannt werden kann. Wenn sie konkret benannt oder beschrieben werden kann, arbeite mit dieser Person weiter. Wenn die Person zwar als Person wahrnehmbar ist, aber noch nicht identifizierbar, gehe neutral mit dieser Person weiter, ohne einen falschen Namen oder Platzhalter zu uebernehmen.",
        intent_meanings={
            "ready": "Der Kunde nennt oder beschreibt die konkrete Person, mit der jetzt weitergearbeitet werden soll.",
            "unknown_person": "Der Kunde macht klar, dass es zwar um eine Person geht, aber noch nicht klar ist, wer diese Person genau ist oder wie sie heisst.",
            "unclear": "Die Antwort reicht noch nicht fuer eine klare Einordnung oder benennt die Person noch nicht konkret.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage dazu, was hier mit der Person gemeint ist.",
            "support_needed": "Der Kunde braucht einen sicheren Moment, damit diese Person klarer vor dem inneren Auge auftauchen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "origin_person_branch_intro",
            "origin_person_unknown_intro",
            "origin_person_name",
            "abort_confirmation",
        ),
        routing_rules={
            "ready": {"action": "transition", "next_node": "origin_person_branch_intro"},
            "unknown_person": {"action": "transition", "next_node": "origin_person_unknown_intro"},
            "unclear": {"action": "clarify", "next_node": "origin_person_name"},
            "repeat": {"action": "repeat", "next_node": "origin_person_name"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "origin_person_name"},
            "support_needed": {"action": "support", "next_node": "origin_person_name"},
        },
        same_node_replies={
            "unclear": "Wenn noch nicht klar ist, wer diese Person genau ist, hol sie innerlich etwas naeher heran und schau einfach, ob deutlicher wird, wer das sein koennte. Wenn es klarer wird, reicht der Name oder eine kurze Beschreibung.",
            "repeat": NODE_SCRIPTS["origin_person_name_question"],
            "question": "Hier geht es nur darum, ob du schon genauer erkennst, wer diese Person ist. Wenn es klarer wird, reicht der Name oder eine kurze Beschreibung. Wenn noch nicht klar ist, wer es genau ist, ist auch genau das eine brauchbare Rueckmeldung.",
            "support_needed": "Bleib noch einen Moment bei dieser Person. Lass sie innerlich etwas naeher kommen und schau einfach, ob deutlicher wird, wer das sein koennte.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "origin_person_unknown_intro": ScriptNodeSpec(
        node_id="origin_person_unknown_intro",
        script_text=NODE_SCRIPTS["origin_person_unknown_intro"],
        next_node="group_person_ready",
    ),
    "origin_self_resolution_intro": ScriptNodeSpec(
        node_id="origin_self_resolution_intro",
        script_text=NODE_SCRIPTS["origin_self_resolution_intro"],
        next_node="origin_self_need",
    ),
    "origin_self_need": _make_ready_transition_node(
        "origin_self_need",
        NODE_SCRIPTS["origin_self_need_question"],
        "Klaere im Selbst-Zweig, was dem damaligen Ich in diesem Moment gefehlt hat oder was es am meisten gebraucht haette, bevor das heutige Ich in die Szene geholt wird.",
        "origin_self_release_intro",
        clarify_reply="Bleib noch einen Moment in genau dieser Szene und spuer nach: Was hat dir dort am meisten gefehlt, oder was haettest du in diesem Moment am meisten gebraucht? Dein erster Eindruck reicht.",
        question_reply="Ich meine: Was haette dein damaliges Ich in genau diesem Moment am meisten gebraucht. Zum Beispiel Schutz, Halt, Orientierung, Sicherheit, Trost, Ruhe oder etwas ganz anderes, das sich fuer dich dort stimmig anfuehlt.",
        support_reply="Bleib ruhig bei dir und spuer nur so weit nach, wie es sich stabil anfuehlt. Wenn dir klarer wird, was dort gefehlt hat, reicht eine kurze Rueckmeldung dazu.",
        ready_meaning="Der Kunde beschreibt frei, was dem damaligen Ich in diesem Moment gefehlt hat oder was es am meisten gebraucht haette.",
    ),
    "origin_self_release_intro": ScriptNodeSpec(
        node_id="origin_self_release_intro",
        script_text=NODE_SCRIPTS["origin_self_release_intro"],
        next_node=None,
    ),
    "origin_person_branch_intro": ScriptNodeSpec(
        node_id="origin_person_branch_intro",
        script_text=NODE_SCRIPTS["origin_person_branch_intro"],
        next_node="group_bring_person_forward",
    ),
    "group_branch_intro": ScriptNodeSpec(
        node_id="group_branch_intro",
        script_text=NODE_SCRIPTS["group_branch_intro"],
        next_node="group_image_ready",
    ),
    "group_image_ready": _make_yes_no_node(
        "group_image_ready",
        NODE_SCRIPTS["group_image_ready_question"],
        "Pruefe im Gruppen-Zweig, ob die relevante Gruppe jetzt klar vor dem Kunden steht.",
        "group_source_kind",
        "Lass diese Gruppe noch etwas naeher kommen, bis das Bild klar vor dir steht. Sobald das Bild klar da ist, genuegt ein Ja.",
        support_reply="Bleib ruhig bei dir und lass die Gruppe in deinem eigenen Tempo etwas klarer werden. Sobald sie deutlich vor dir steht, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob die Gruppe jetzt klar vor dir steht oder ob sie noch etwas naeher kommen darf.",
        clarify_reply="Wenn die Gruppe jetzt klar vor dir steht, genuegt ein Ja. Wenn noch etwas Zeit noetig ist, bleib einfach kurz bei dem Bild.",
    ),
    "group_source_kind": SemanticNodeSpec(
        node_id="group_source_kind",
        question_text=NODE_SCRIPTS["group_source_kind_question"],
        node_goal="Klaere, ob die Dynamik von der ganzen Gruppe, von einer bestimmten Person oder von mehreren einzelnen Personen innerhalb der Gruppe ausgeht.",
        intent_meanings={
            "whole_group": "Der Kunde beschreibt, dass das ungute Gefuehl von der ganzen Gruppe oder der gesamten Gruppendynamik ausgeht.",
            "one_person": "Der Kunde beschreibt, dass eine bestimmte einzelne Person aus der Gruppe der eigentliche Ausloeser ist. Auch ein einzelner Name oder eine einzelne Rolle wie 'Peter', 'mein Vater' oder 'der Lehrer' zaehlt hier bereits als `one_person`.",
            "multiple_people": "Der Kunde beschreibt, dass mehrere einzelne Personen innerhalb der Gruppe relevant sind.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie er diese Gruppendynamik einordnen soll.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Orientierung, bevor er die Gruppendynamik einordnen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "group_whole_scope",
            "group_specific_person_intro",
            "group_multiple_people_intro",
            "group_source_kind",
            "abort_confirmation",
        ),
        routing_rules={
            "whole_group": {"action": "transition", "next_node": "group_whole_scope"},
            "one_person": {"action": "transition", "next_node": "group_specific_person_intro"},
            "multiple_people": {"action": "transition", "next_node": "group_multiple_people_intro"},
            "unclear": {"action": "clarify", "next_node": "group_source_kind"},
            "repeat": {"action": "repeat", "next_node": "group_source_kind"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "group_source_kind"},
            "support_needed": {"action": "support", "next_node": "group_source_kind"},
        },
        same_node_replies={
            "unclear": "Spuer noch einmal kurz nach: Geht dieses ungute Gefuehl eher von der ganzen Gruppe, nur von einer bestimmten Person oder von mehreren einzelnen Personen aus?",
            "repeat": NODE_SCRIPTS["group_source_kind_question"],
            "question": "Hier reicht es, wenn du nur kurz einordnest, ob die eigentliche Dynamik von der ganzen Gruppe, von einer Person oder von mehreren einzelnen Personen ausgeht.",
            "support_needed": "Spuer einfach weiter nach. Wenn dir klarer wird, ob du eher die ganze Gruppe, eine Person oder mehrere einzelne Personen wahrnimmst, genuegt eine kurze Rueckmeldung.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "group_whole_scope": SemanticNodeSpec(
        node_id="group_whole_scope",
        question_text=NODE_SCRIPTS["group_whole_scope_question"],
        node_goal="Entscheide, ob fuer die Aufloesung eine stellvertretende Person aus der Gruppe ausreicht oder ob mehrere Personen einzeln bearbeitet werden muessen.",
        intent_meanings={
            "representative_enough": "Der Kunde beschreibt, dass eine stellvertretende Person aus der Gruppe ausreicht.",
            "multiple_required": "Der Kunde beschreibt, dass mehrere Personen aus der Gruppe einzeln bearbeitet werden muessen.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage, wie diese Auswahl gemeint ist.",
            "support_needed": "Der Kunde braucht Stabilisierung oder Orientierung, bevor er die Auswahl treffen kann.",
        },
        allowed_actions=("transition", "clarify", "support", "answer_question", "repeat", "abort"),
        allowed_next_nodes=(
            "group_select_representative_intro",
            "group_multiple_required_intro",
            "group_whole_scope",
            "abort_confirmation",
        ),
        routing_rules={
            "representative_enough": {"action": "transition", "next_node": "group_select_representative_intro"},
            "multiple_required": {"action": "transition", "next_node": "group_multiple_required_intro"},
            "unclear": {"action": "clarify", "next_node": "group_whole_scope"},
            "repeat": {"action": "repeat", "next_node": "group_whole_scope"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "group_whole_scope"},
            "support_needed": {"action": "support", "next_node": "group_whole_scope"},
        },
        same_node_replies={
            "unclear": "Spuer noch einmal kurz nach: Reicht hier eine stellvertretende Person aus dieser Gruppe, oder muessen mehrere Personen einzeln einbezogen werden?",
            "repeat": NODE_SCRIPTS["group_whole_scope_question"],
            "question": "Hier geht es nur darum, ob eine Person stellvertretend fuer die Gruppendynamik reicht oder ob du mehrere einzelne Personen brauchst.",
            "support_needed": "Spuer einfach weiter nach. Wenn dir klarer wird, ob eine Person stellvertretend reicht oder ob mehrere Personen wichtig sind, genuegt eine kurze Rueckmeldung.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "group_select_representative_intro": ScriptNodeSpec(
        node_id="group_select_representative_intro",
        script_text=NODE_SCRIPTS["group_select_representative_intro"],
        next_node="group_representative_name",
    ),
    "group_representative_name": _make_ready_transition_node(
        "group_representative_name",
        NODE_SCRIPTS["group_representative_name_question"],
        "Nimm die frei genannte stellvertretende Person auf und fuehre danach dazu, diese Person direkt vor sich zu holen.",
        "group_bring_person_forward",
        clarify_reply="Wenn dir die Person aus dieser Gruppe klarer wird, reicht der Name oder eine kurze Beschreibung der Person, die diese Dynamik am staerksten verkoerpert.",
        question_reply="Hier reicht der Name oder eine kurze Beschreibung der Person, die fuer dich stellvertretend fuer diese Gruppendynamik steht.",
        support_reply="Wenn dir die passende Person klarer wird, reicht der Name oder eine kurze Beschreibung der Person, die fuer dich am staerksten fuer diese Gruppendynamik steht.",
        ready_meaning="Der Kunde nennt eine Person oder beschreibt eindeutig, mit wem wir aus der Gruppe weiterarbeiten sollen.",
    ),
    "group_specific_person_intro": ScriptNodeSpec(
        node_id="group_specific_person_intro",
        script_text=NODE_SCRIPTS["group_specific_person_intro"],
        next_node="group_specific_person_name",
    ),
    "group_specific_person_name": _make_ready_transition_node(
        "group_specific_person_name",
        NODE_SCRIPTS["group_specific_person_question"],
        "Nimm die genannte konkrete Person aus der Gruppe auf und fuehre danach dazu, diese Person direkt vor sich zu holen.",
        "group_bring_person_forward",
        clarify_reply="Wenn dir die entscheidende Person aus dieser Gruppe klarer wird, reicht der Name oder eine kurze Beschreibung.",
        question_reply="Hier reicht der Name oder eine kurze Beschreibung der Person, die innerhalb der Gruppe fuer dich der eigentliche Ausloeser ist.",
        support_reply="Wenn dir die Person klarer wird, reicht der Name oder eine kurze Beschreibung der Person, die hier innerhalb der Gruppe am wichtigsten ist.",
        ready_meaning="Der Kunde nennt die bestimmte Person aus der Gruppe, die hier der eigentliche Ausloeser ist.",
    ),
    "group_multiple_people_intro": ScriptNodeSpec(
        node_id="group_multiple_people_intro",
        script_text=NODE_SCRIPTS["group_multiple_people_intro"],
        next_node="group_multiple_people_name",
    ),
    "group_multiple_people_name": _make_ready_transition_node(
        "group_multiple_people_name",
        NODE_SCRIPTS["group_multiple_people_question"],
        "Nimm auf, mit welcher Person aus mehreren relevanten Gruppenmitgliedern zuerst begonnen werden soll.",
        "group_bring_person_forward",
        clarify_reply="Wenn dir die erste Person klarer wird, reicht der Name oder eine kurze Beschreibung der Person, mit der wir innerhalb dieser Gruppe zuerst beginnen sollen.",
        question_reply="Hier geht es nur darum, mit welcher der mehreren wichtigen Personen wir als erstes anfangen sollen.",
        support_reply="Wenn dir die Reihenfolge klarer wird, reicht der Name oder eine kurze Beschreibung der Person, mit der wir zuerst beginnen sollen.",
        ready_meaning="Der Kunde nennt die erste Person, mit der wir innerhalb einer Mehrpersonen-Dynamik beginnen sollen.",
    ),
    "group_multiple_required_intro": ScriptNodeSpec(
        node_id="group_multiple_required_intro",
        script_text=NODE_SCRIPTS["group_multiple_required_intro"],
        next_node="group_multiple_required_name",
    ),
    "group_multiple_required_name": _make_ready_transition_node(
        "group_multiple_required_name",
        NODE_SCRIPTS["group_multiple_required_question"],
        "Nimm auf, mit welcher Person aus einer Gruppe mit mehreren noetigen Einzelloesungen zuerst begonnen werden soll.",
        "group_bring_person_forward",
        clarify_reply="Wenn dir die erste Person aus dieser Gruppe klarer wird, reicht der Name oder eine kurze Beschreibung der Person, mit der wir beginnen sollen.",
        question_reply="Hier geht es nur darum, mit welcher Person aus der Gruppe wir als erstes starten sollen.",
        support_reply="Wenn dir die erste Person klarer wird, reicht der Name oder eine kurze Beschreibung der ersten Person, mit der wir beginnen sollen.",
        ready_meaning="Der Kunde nennt die erste Person aus der Gruppe, mit der wir bei mehreren noetigen Einzelloesungen beginnen sollen.",
    ),
    "group_bring_person_forward": ScriptNodeSpec(
        node_id="group_bring_person_forward",
        script_text=NODE_SCRIPTS["group_bring_person_forward"],
        next_node="group_person_ready",
    ),
    "group_person_ready": _make_yes_no_node(
        "group_person_ready",
        NODE_SCRIPTS["group_person_ready_question"],
        "Pruefe, ob die relevante Person aus der Gruppe jetzt klar vor dem Kunden steht.",
        "group_person_handoff",
        "Lass {named_person} noch etwas naeher kommen, bis {named_person} klar vor dir steht. Wenn das Bild klar ist, genuegt ein Ja.",
        support_reply="Bleib ruhig bei dir und lass {named_person} in deinem eigenen Tempo klarer werden. Sobald {named_person} deutlich vor dir steht, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob {named_person} jetzt klar vor dir steht oder ob du noch einen Moment brauchst.",
        clarify_reply="Wenn {named_person} jetzt klar vor dir steht, genuegt ein Ja. Wenn noch etwas Zeit noetig ist, bleib einfach kurz bei dem Bild.",
    ),
    "group_person_handoff": ScriptNodeSpec(
        node_id="group_person_handoff",
        script_text=NODE_SCRIPTS["group_person_handoff"],
        next_node="group_person_trigger_reason",
    ),
    "group_person_trigger_reason": _make_ready_transition_node(
        "group_person_trigger_reason",
        NODE_SCRIPTS["group_person_trigger_reason_question"],
        "Klaere vor dem Perspektivwechsel, was genau an der ausgewaehlten Person dieses ungute Gefuehl ausloest.",
        "group_person_trigger_role",
        clarify_reply="Nimm kurz wahr, was es an {named_person} genau ist, das dieses ungute Gefuehl in dir ausloest. Dein erster Eindruck reicht.",
        question_reply="Ich meine: Was genau ist es an {named_person}, das dieses ungute Gefuehl in dir ausloest. Zum Beispiel etwas an der Art, an den Worten, am Verhalten, am Blick oder an dem Druck, den du bei {named_person} spuerst.",
        support_reply="Schau {named_person} noch einmal genau an. Wenn dir dort etwas klarer wird, reicht eine kurze Beschreibung davon, was du als Ausloeser wahrnimmst.",
        ready_meaning="Der Kunde beschreibt inhaltlich, was genau an {named_person} dieses ungute Gefuehl ausloest oder warum {named_person} jetzt relevant ist.",
    ),
    "group_person_trigger_role": _make_ready_transition_node(
        "group_person_trigger_role",
        NODE_SCRIPTS["group_person_trigger_role_question"],
        "Klaere, ob die ausgewaehlte Person selbst der direkte Ausloeser ist oder eher stellvertretend fuer etwas Groesseres in dieser Situation steht.",
        "group_person_trigger_core",
        clarify_reply="Ordne kurz ein, ob das eher direkt von {named_person} ausgeht oder ob {named_person} eher stellvertretend fuer etwas Groesseres in dieser Situation steht.",
        question_reply="Ich meine: Geht dieses ungute Gefuehl eher direkt von {named_person} selbst aus, oder steht {named_person} eher stellvertretend fuer etwas Groesseres, das in dieser Situation wirksam ist.",
        support_reply="Spuer bei {named_person} weiter nach. Wenn dir klarer wird, ob das eher direkt von {named_person} ausgeht oder ob {named_person} eher fuer etwas Groesseres steht, das in dieser Situation wirksam ist, genuegt eine kurze Rueckmeldung.",
        ready_meaning="Der Kunde ordnet inhaltlich ein, ob {named_person} selbst der direkte Ausloeser ist oder eher stellvertretend fuer etwas Groesseres in dieser Situation steht. Nur echte Rollen-Einordnungen wie 'direkt von ihm', 'er steht fuer die Gruppe' oder aehnliche Aussagen zaehlen hier als `ready`. Rueckfragen, Ueberforderung oder Nonsense sind hier niemals `ready`.",
    ),
    "group_person_trigger_core": _make_ready_transition_node(
        "group_person_trigger_core",
        NODE_SCRIPTS["group_person_trigger_core_question"],
        "Klaere vor dem Perspektivwechsel, ob schon ein erstes Verstaendnis da ist, warum diese Person so reagiert oder wofuer dieses Verhalten steht.",
        "person_switch_ready_intro",
        clarify_reply="Nimm kurz wahr, ob du schon ein Gefuehl dafuer hast, warum {named_person} so reagiert oder was dahinterliegt. Wenn es noch nicht klar ist, ist auch das eine stimmige Rueckmeldung.",
        question_reply="Ich meine: Weisst du schon, warum {named_person} in dieser Situation so reagiert oder wofuer das steht. Wenn es dir noch nicht klar ist, ist genau das auch eine brauchbare Rueckmeldung.",
        support_reply="Spuer bei {named_person} weiter nach. Wenn dir klarer wird, warum {named_person} so reagiert oder was dahinterliegt, genuegt eine kurze Rueckmeldung.",
        ready_meaning="Der Kunde beschreibt inhaltlich, warum {named_person} so reagiert oder was dahinterliegt. Auch sehr kurze, aber bedeutungstragende Antworten wie 'dazugehoeren', 'anerkennung', 'macht', 'abwertung', 'gruppendruck', 'schutz', 'angst' oder 'scham' zaehlen hier bereits als brauchbare Rueckmeldung. Antworten wie 'ja, sie lachen weil...', 'ich glaube, es geht um Zugehoerigkeit' oder aehnliche Erklaerungen zaehlen ebenfalls als brauchbare Rueckmeldung. Es zaehlt auch als brauchbare Rueckmeldung, wenn der Kunde nur kurz signalisiert, dass es schon klar ist oder noch nicht klar ist, zum Beispiel mit 'ja', 'noch nicht', 'weiss nicht' oder 'ist mir noch nicht klar'.",
    ),
    "person_switch_ready_intro": ScriptNodeSpec(
        node_id="person_switch_ready_intro",
        script_text=NODE_SCRIPTS["person_switch_ready_intro"],
        next_node="person_switch_ready",
    ),
    "person_switch_ready": _make_yes_no_node(
        "person_switch_ready",
        NODE_SCRIPTS["person_switch_ready_question"],
        "Pruefe kurz vor dem Countdown, ob der Kunde jetzt bereit fuer den Perspektivwechsel in die andere Person ist.",
        "person_switch_intro",
        "Wenn es fuer dich passt, gehen wir gleich in den Perspektivwechsel.",
        support_reply="Bleib noch einen Augenblick bei dir. Sobald es fuer dich stimmig ist, gehen wir in den Perspektivwechsel.",
        question_reply="Hier geht es nur darum, ob es fuer dich jetzt passt, in die Perspektive von {named_person_ref} zu wechseln.",
        clarify_reply="Wenn es fuer dich jetzt passt, in die Perspektive von {named_person_ref} zu wechseln, genuegt ein Ja. Wenn noch etwas Zeit noetig ist, bleib einfach kurz bei dir.",
    ),
    "group_next_person_check": SemanticNodeSpec(
        node_id="group_next_person_check",
        question_text=NODE_SCRIPTS["group_next_person_check_question"],
        node_goal="Pruefe nach einer bereits bearbeiteten Person aus einer Mehrpersonen-Gruppendynamik, ob noch weitere Personen aus derselben Gruppe geloest werden sollten.",
        intent_meanings={
            "yes": "Der Kunde bestaetigt klar, dass in dieser Gruppe noch eine weitere Person offen ist, die ebenfalls bearbeitet werden sollte.",
            "no": "Der Kunde verneint klar und signalisiert, dass fuer diesen Gruppenabschnitt keine weitere Person mehr offen ist.",
            "unclear": "Die Antwort reicht nicht fuer eine klare Ja/Nein-Einordnung.",
            "repeat": "Der Kunde moechte, dass die Frage wiederholt wird.",
            "abort": "Der Kunde moechte abbrechen.",
            "question": "Der Kunde stellt eine Rueckfrage oder braucht eine Erklaerung zur Frage.",
            "support_needed": "Der Kunde braucht Stabilisierung, Orientierung oder noch einen Moment, bevor er antworten kann.",
        },
        allowed_actions=("transition", "support", "clarify", "repeat", "answer_question", "abort"),
        allowed_next_nodes=("group_next_person_name", "group_resolution_complete", "group_next_person_check", "abort_confirmation"),
        routing_rules={
            "yes": {"action": "transition", "next_node": "group_next_person_name"},
            "no": {"action": "transition", "next_node": "group_resolution_complete"},
            "unclear": {"action": "clarify", "next_node": "group_next_person_check"},
            "repeat": {"action": "repeat", "next_node": "group_next_person_check"},
            "abort": {"action": "abort", "next_node": "abort_confirmation"},
            "question": {"action": "answer_question", "next_node": "group_next_person_check"},
            "support_needed": {"action": "support", "next_node": "group_next_person_check"},
        },
        same_node_replies={
            "unclear": "Spuer noch einmal kurz nach: Gibt es in dieser Gruppe jetzt noch eine weitere Person, die wir ebenfalls noch loesen sollten?",
            "repeat": NODE_SCRIPTS["group_next_person_check_question"],
            "question": "Hier geht es nur darum, ob in dieser Gruppe jetzt noch eine weitere Person offen ist, die wir ebenfalls noch loesen sollten.",
            "support_needed": "Spuer noch einmal ruhig in diese Gruppe hinein. Wenn dort noch jemand offen ist, genuegt ein Ja. Wenn nicht, genuegt ein Nein.",
            "abort": NODE_SCRIPTS["abort"],
        },
    ),
    "group_next_person_name": _make_ready_transition_node(
        "group_next_person_name",
        NODE_SCRIPTS["group_next_person_name_question"],
        "Nimm auf, welche weitere Person aus derselben Gruppe als naechstes bearbeitet werden soll, und fuehre dann wieder in den normalen Personenpfad.",
        "group_bring_person_forward",
        clarify_reply="Wenn dir die weitere Person klarer wird, reicht der Name oder eine kurze Beschreibung der Person, mit der wir jetzt als naechstes weitermachen sollen.",
        question_reply="Hier geht es nur darum, welche weitere Person aus dieser Gruppe jetzt als naechstes wichtig ist.",
        support_reply="Wenn dir die naechste Person klarer wird, reicht der Name oder eine kurze Beschreibung der weiteren Person aus dieser Gruppe, die jetzt als naechstes wichtig ist.",
        ready_meaning="Der Kunde nennt die naechste Person aus derselben Gruppe, die jetzt als naechstes bearbeitet werden soll.",
    ),
    "person_switch_intro": ScriptNodeSpec(
        node_id="person_switch_intro",
        script_text=NODE_SCRIPTS["person_switch_intro"],
        next_node="person_switch_hears",
    ),
    "person_switch_hears": _make_yes_no_node(
        "person_switch_hears",
        NODE_SCRIPTS["person_switch_hears_question"],
        "Pruefe, ob der Perspektivwechsel in die andere Person jetzt ansprechbar ist.",
        "person_switch_sees_customer",
        "Bleib noch einen Moment in dieser Perspektive, bis du mich dort klar wahrnehmen kannst. Wenn du mich hoerst, genuegt ein Ja.",
        support_reply="Komm in dieser Perspektive in deinem Tempo an. Sobald du mich dort klar hoerst, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob du mich aus der Perspektive von {named_person} jetzt hoeren kannst.",
        clarify_reply="Wenn du mich jetzt hoerst, genuegt ein Ja. Wenn noch etwas Zeit noetig ist, bleib einfach kurz in dieser Perspektive.",
    ),
    "person_switch_sees_customer": _make_yes_no_node(
        "person_switch_sees_customer",
        NODE_SCRIPTS["person_switch_sees_customer_question"],
        "Pruefe, ob die uebernommene Perspektive das Gegenueber jetzt wahrnimmt.",
        "person_switch_sees_impact",
        "Lass das Bild noch einen Moment klarer werden, bis {customer_ref} deutlich vor dir steht. Wenn du es siehst, genuegt ein Ja.",
        support_reply="Lass das Bild in deinem Tempo deutlicher werden. Sobald {customer_ref} klar vor dir steht, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob {customer_ref} aus dieser Perspektive jetzt deutlich vor dir steht.",
        clarify_reply="Wenn {customer_ref} jetzt klar vor dir steht, genuegt ein Ja. Wenn noch etwas Zeit noetig ist, bleib einfach kurz bei dem Bild.",
    ),
    "person_switch_sees_impact": _make_yes_no_node(
        "person_switch_sees_impact",
        NODE_SCRIPTS["person_switch_sees_impact_question"],
        "Pruefe, ob aus der uebernommenen Perspektive jetzt sichtbar wird, was in diesem Moment passiert und was im Gegenueber ausgeloest wird.",
        "person_switch_why",
        "Schau noch einen Moment genauer hin, bis fuer dich klarer wird, was in diesem Moment passiert und was du bei {customer_ref_dat} ausloest. Wenn es klar wird, genuegt ein Ja.",
        support_reply="Schau nur so weit hin, wie es sich stabil anfuehlt. Sobald klarer wird, was dort passiert und was du bei {customer_ref_dat} ausloest, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob aus dieser Perspektive sichtbar wird, dass dein Verhalten bei {customer_ref_dat} gerade etwas ausloest.",
        clarify_reply="Wenn dort jetzt klar wird, was in diesem Moment passiert und was du bei {customer_ref_dat} ausloest, genuegt ein Ja. Wenn noch nicht, genuegt ein Nein oder du bleibst noch kurz dabei.",
    ),
    "person_switch_heard_customer": _make_yes_no_node(
        "person_switch_heard_customer",
        NODE_SCRIPTS["person_switch_heard_customer_question"],
        "Pruefe, ob aus der uebernommenen Perspektive jetzt klar wird, wie sich das Gegenueber fuehlt und was dieser Moment spaeter ausloest.",
        "person_switch_why",
        "Bleib noch einen Moment in dieser Perspektive und schau weiter hin, bis fuer dich klarer wird, wie sich {customer_ref} fuehlt und was dieser Moment spaeter bei {customer_ref_dat} ausloest. Wenn es klar wird, genuegt ein Ja.",
        support_reply="Bleib ruhig in dieser Perspektive und schau nur so weit hin, wie es sich stabil anfuehlt. Sobald dir klarer wird, wie sich {customer_ref} fuehlt und was dieser Moment spaeter bei {customer_ref_dat} ausloest, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob aus dieser Perspektive spuerbar wird, was bei {customer_ref_dat} passiert und welche spaetere Wirkung dieser Moment haben wird.",
        clarify_reply="Wenn fuer dich jetzt klar wird, wie sich {customer_ref} fuehlt und was dieser Moment spaeter bei {customer_ref_dat} ausloest, genuegt ein Ja. Wenn noch nicht, genuegt ein Nein oder du bleibst noch kurz dabei.",
    ),
    "person_switch_why": _make_ready_transition_node(
        "person_switch_why",
        NODE_SCRIPTS["person_switch_why_question"],
        "Nimm die freie Erklaerung auf, was in der anderen Person los ist und warum sie so reagiert.",
        "person_switch_aware_trigger",
        clarify_reply="Bleib in der Perspektive von {named_person_ref}. Nimm kurz wahr, was in dir in dem Moment los ist, wie es dir dort geht und warum du genau so reagierst. Dein erster Eindruck reicht.",
        question_reply="Hier reicht es, wenn du aus der Perspektive von {named_person_ref} frei wiedergibst, was dort los ist, wie es dir dort geht und warum du so reagierst.",
        support_reply="Bleib in der Perspektive von {named_person_ref}. Wenn dir dort klarer wird, was in dir los ist und warum du so reagierst, reicht eine kurze Beschreibung.",
        ready_meaning="Der Kunde beschreibt frei, was in {named_person} in dem Moment los ist und warum {named_person} so reagiert. Auch kurze abstrakte Erklaerungen wie 'ueberforderung', 'angst', 'druck', 'scham', 'wut', 'ohnmacht' oder 'dazugehoeren' zaehlen hier bereits als brauchbare Antwort.",
    ),
    "person_switch_aware_trigger": _make_yes_no_node(
        "person_switch_aware_trigger",
        NODE_SCRIPTS["person_switch_aware_trigger_question"],
        "Pruefe, ob in der uebernommenen Perspektive jetzt bewusst wird, dass genau dieser Moment spaeter mit dem Rauchen verknuepft ist.",
        "person_switch_return_intro",
        "Bleib noch einen Moment in dieser Perspektive und schau weiter hin, bis klarer wird, ob genau dieser Moment spaeter fuer {customer_ref} mit dem Rauchen verknuepft ist. Wenn es klar wird, genuegt ein Ja.",
        support_reply="Bleib noch einen Moment ruhig in dieser Perspektive und schau nur so weit hin, wie es sich gut aushalten laesst. Sobald dort etwas klarer wird, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob aus dieser Perspektive sichtbar wird, dass genau dieser Moment spaeter fuer {customer_ref} ein Ausloeser fuer das Rauchen wird.",
        clarify_reply="Wenn dir jetzt bewusst wird, dass genau dieser Moment spaeter fuer {customer_ref} mit dem Rauchen verknuepft ist, genuegt ein Ja. Wenn noch nicht, genuegt ein Nein oder du bleibst noch kurz dabei.",
    ),
    "person_switch_return_intro": ScriptNodeSpec(
        node_id="person_switch_return_intro",
        script_text=NODE_SCRIPTS["person_switch_return_intro"],
        next_node="person_switch_self_heard",
    ),
    "person_switch_self_heard": _make_yes_no_node(
        "person_switch_self_heard",
        NODE_SCRIPTS["person_switch_self_heard_question"],
        "Pruefe, ob der Kunde nach dem Perspektivwechsel gehoert hat, was {named_person} gesagt hat, und ob der Grund dafuer wahrnehmbar wurde.",
        "person_switch_self_understands",
        "Spuer noch einmal in Ruhe nach, was {named_person} gesagt hat und was dir dadurch ueber den eigentlichen Grund klar wird. Wenn es fuer dich klarer wird, genuegt ein Ja.",
        support_reply="Spuer noch einmal nach, was du aus diesem Perspektivwechsel fuer dich mitgenommen hast. Sobald dir klarer wird, was {named_person} gesagt hat und warum, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob du gehoert hast, was {named_person} gesagt hat, und ob der Grund fuer das Verhalten von {named_person} dadurch klarer geworden ist.",
        clarify_reply="Wenn du gehoert hast, was {named_person} gesagt hat, und dir der Grund dadurch klarer geworden ist, genuegt ein Ja. Wenn noch nicht, genuegt ein Nein oder du bleibst noch kurz dabei.",
    ),
    "person_switch_self_understands": _make_yes_no_node(
        "person_switch_self_understands",
        NODE_SCRIPTS["person_switch_self_understands_question"],
        "Pruefe, ob dieses neue Verstaendnis jetzt bereits hilfreich genug ist, um diesen Gruppen-Zweig abzuschliessen.",
        "group_resolution_complete",
        "Bleib noch einen Moment bei dem, was sich jetzt zeigen will. Sobald es fuer dich stimmiger wird, genuegt ein Ja.",
        support_reply="Bleib ruhig bei dir und spuer nur nach, was sich durch dieses Verstaendnis bereits veraendert hat. Wenn es fuer dich passt, genuegt ein Ja.",
        question_reply="Hier geht es nur darum, ob dir dieses neue Verstaendnis jetzt schon ein Stueck weiterhilft.",
        clarify_reply="Spuer noch einmal kurz nach: Hilft dir dieses neue Verstaendnis jetzt bereits, oder braucht es noch einen Moment?",
    ),
    "group_resolution_complete": ScriptNodeSpec(
        node_id="group_resolution_complete",
        script_text=NODE_SCRIPTS["group_resolution_complete"],
        next_node=None,
    ),
    "abort_confirmation": ScriptNodeSpec(
        node_id="abort_confirmation",
        script_text=NODE_SCRIPTS["abort"],
        next_node=None,
    ),
}


def get_node_spec(node_id: str) -> SemanticNodeSpec | ScriptNodeSpec:
    try:
        return NODE_SPECS[node_id]
    except KeyError as exc:
        available = ", ".join(sorted(NODE_SPECS))
        raise ValueError(f"unknown semantic node '{node_id}'. Available: {available}") from exc


def get_semantic_node_spec(node_id: str) -> SemanticNodeSpec:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise ValueError(f"node '{node_id}' is not a semantic node")
    return spec


def get_script_node_spec(node_id: str) -> ScriptNodeSpec:
    spec = get_node_spec(node_id)
    if not isinstance(spec, ScriptNodeSpec):
        raise ValueError(f"node '{node_id}' is not a script node")
    return spec


def build_request(
    node_id: str,
    customer_message: str,
    *,
    clarify_attempt: int = 0,
    session_context: str = "",
) -> dict[str, Any]:
    return get_semantic_node_spec(node_id).as_request(customer_message, clarify_attempt, session_context)


def expected_output_schema(node_id: str) -> dict[str, Any]:
    spec = get_semantic_node_spec(node_id)
    return {
        "type": "object",
        "required": ["intent", "action", "next_node", "confidence", "reason"],
        "properties": {
            "intent": {"type": "string", "enum": list(spec.allowed_intents)},
            "action": {"type": "string", "enum": list(spec.allowed_actions)},
            "next_node": {"type": "string", "enum": list(spec.allowed_next_nodes)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "additionalProperties": True,
    }


def validate_semantic_decision(node_id: str, payload: dict[str, Any]) -> SemanticModelDecision:
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


def repair_semantic_payload(node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
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
            intent = matching_intents[0]

    if "confidence" not in repaired and intent:
        repaired["confidence"] = 0.75

    if (not str(repaired.get("reason") or "").strip()) and intent:
        repaired["reason"] = "Auto-repaired missing reason from partial model output."

    return repaired


def script_reply_for_decision(node_id: str, decision: SemanticModelDecision) -> str:
    spec = get_semantic_node_spec(node_id)
    return spec.same_node_replies.get(decision.intent, "")


def available_node_ids() -> set[str]:
    return set(NODE_SPECS)
