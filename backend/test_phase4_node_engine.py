import unittest

from phase4_node_engine import advance_node, parse_hypnose_progress, parse_known_vs_new, parse_pleasantness


class Phase4NodeEngineTests(unittest.TestCase):
    def test_hypnose_progress_resolving_routes_to_wait(self) -> None:
        result = parse_hypnose_progress("es loest sich auf")
        self.assertEqual(result.intent, "resolving")
        self.assertFalse(result.needs_clarification)

        decision = advance_node("hell_hypnose_pause", "es loest sich auf")
        self.assertEqual(decision.next_node, "hell_hypnose_wait")
        self.assertIn("noch im Loesen", decision.reply_text)

    def test_hypnose_progress_need_more_time_routes_to_wait(self) -> None:
        result = parse_hypnose_progress("ich brauche noch einen Moment")
        self.assertEqual(result.intent, "need_more_time")

        decision = advance_node("hell_hypnose_pause", "ich brauche noch einen Moment")
        self.assertEqual(decision.next_node, "hell_hypnose_wait")
        self.assertIn("noch einen Moment Zeit", decision.reply_text)

    def test_hypnose_progress_resolved_routes_forward(self) -> None:
        result = parse_hypnose_progress("es ist bereits aufgeloest")
        self.assertEqual(result.intent, "resolved")

        decision = advance_node("hell_hypnose_pause", "es ist bereits aufgeloest")
        self.assertEqual(decision.next_node, "dark_known")
        self.assertIn("bereits etwas Wichtiges geloest", decision.reply_text)

    def test_hypnose_progress_unclear_clarifies(self) -> None:
        result = parse_hypnose_progress("ich weiss es gerade nicht")
        self.assertEqual(result.intent, "unclear")
        self.assertGreaterEqual(result.confidence, 0.9)

        decision = advance_node("hell_hypnose_pause", "ich weiss es gerade nicht")
        self.assertEqual(decision.next_node, "clarify_same_node")
        self.assertIn("bereits aufgeloest", decision.reply_text)

    def test_pleasantness_routes_pleasant(self) -> None:
        result = parse_pleasantness("sehr angenehm")
        self.assertEqual(result.intent, "pleasant")

        decision = advance_node("pleasantness", "sehr angenehm")
        self.assertEqual(decision.next_node, "hell_hypnose_pause")
        self.assertIn("angenehmen Eindruck", decision.reply_text)

    def test_pleasantness_tolerates_small_typos(self) -> None:
        result = parse_pleasantness("fuehlt sich angehnem an")
        self.assertEqual(result.intent, "pleasant")

        decision = advance_node("pleasantness", "fuehlt sich angehnem an")
        self.assertEqual(decision.next_node, "hell_hypnose_pause")

    def test_pleasantness_routes_unpleasant(self) -> None:
        result = parse_pleasantness("unangenehm und drueckend")
        self.assertEqual(result.intent, "unpleasant")

        decision = advance_node("pleasantness", "unangenehm und drueckend")
        self.assertEqual(decision.next_node, "hell_regulation_choice")
        self.assertIn("regulieren", decision.reply_text)

    def test_known_vs_new_routes_known(self) -> None:
        result = parse_known_vs_new("ja, das kenne ich schon von frueher")
        self.assertEqual(result.intent, "known")

        decision = advance_node("known_vs_new", "ja, das kenne ich schon von frueher")
        self.assertEqual(decision.next_node, "dark_backtrace_countdown")
        self.assertIn("noch aelter", decision.reply_text)

    def test_known_vs_new_routes_new(self) -> None:
        result = parse_known_vs_new("das ist zum ersten mal da")
        self.assertEqual(result.intent, "new")

        decision = advance_node("known_vs_new", "das ist zum ersten mal da")
        self.assertEqual(decision.next_node, "dark_origin_scene")
        self.assertIn("wichtiger Ursprung", decision.reply_text)

    def test_repeat_and_abort_are_generic(self) -> None:
        repeat = advance_node("pleasantness", "bitte nochmal")
        self.assertEqual(repeat.next_node, "repeat_same_question")
        self.assertEqual(repeat.reply_text, "Wie fuehlt sich dieses Helle fuer dich an?")

        abort = advance_node("known_vs_new", "ich moechte abbrechen")
        self.assertEqual(abort.next_node, "abort_confirmation")

    def test_negated_resolved_does_not_false_positive(self) -> None:
        result = parse_hypnose_progress("es ist noch nicht aufgeloest")
        self.assertEqual(result.intent, "need_more_time")


if __name__ == "__main__":
    unittest.main()
