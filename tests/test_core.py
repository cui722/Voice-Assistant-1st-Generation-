import copy
import json
import unittest
from core import BASE, WORLD, observation, apply_plan, load_cases, plan


class CoreTests(unittest.TestCase):
    def test_all_development_labels_are_valid(self):
        cases = load_cases(BASE / "data/dev_cases.jsonl")
        self.assertEqual(len(cases), 30)
        for case in cases:
            obs = observation(case)
            self.assertEqual(set(obs), {"utterance", "history", "context"})
            apply_plan(obs["context"], case["expected"])

    def test_gold_is_not_in_model_input(self):
        item = {
            "utterance": "hello",
            "expected": {"secret": "DO_NOT_LEAK_123"},
            "category": "SECRET_CATEGORY",
        }
        text = json.dumps(observation(item))
        self.assertNotIn("DO_NOT_LEAK_123", text)
        self.assertNotIn("SECRET_CATEGORY", text)

    def test_call_updates_copy_only(self):
        before = copy.deepcopy(WORLD)
        after, applied = apply_plan(
            before, plan("execute", "call.queue", {"contact_id": "c_maya"})
        )
        self.assertTrue(applied)
        self.assertEqual(after["state"]["pending_call"], "c_maya")
        self.assertIsNone(before["state"]["pending_call"])

    def test_cancel_preserves_other_operation(self):
        context = observation({
            "utterance": "cancel the call",
            "state_patch": {"pending_call": "c_maya", "destination": "p_lax"},
        })["context"]
        after, applied = apply_plan(context, plan("cancel", "call.cancel"))
        self.assertTrue(applied)
        self.assertIsNone(after["state"]["pending_call"])
        self.assertEqual(after["state"]["destination"], "p_lax")

    def test_clarify_does_not_mutate_state(self):
        after, applied = apply_plan(
            WORLD, plan("clarify", slot="contact_id", question="Which Alex?")
        )
        self.assertFalse(applied)
        self.assertEqual(after, WORLD)

    def test_unknown_entity_and_invalid_volume_are_blocked(self):
        invalid = [
            plan("execute", "call.queue", {"contact_id": "invented"}),
            plan("execute", "audio.set_volume", {"level": 150}),
            plan("execute", "audio.set_volume", {"level": True}),
        ]
        for prediction in invalid:
            with self.assertRaises(ValueError):
                apply_plan(WORLD, prediction)

    def test_resume_without_track_is_blocked(self):
        with self.assertRaises(ValueError):
            apply_plan(WORLD, plan("execute", "music.resume"))


if __name__ == "__main__":
    unittest.main()
