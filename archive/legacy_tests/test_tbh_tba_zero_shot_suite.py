from __future__ import annotations

import unittest
from pathlib import Path

from upeftguard.utilities.core.manifest import (
    infer_attack_sample_identities,
    parse_joint_manifest_json_by_model_name,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestTbhTbaZeroShotSuite(unittest.TestCase):
    def test_committed_tbh_tba_zero_shot_manifests_exist_and_parse(self):
        manifest_dir = REPO_ROOT / "manifests" / "zero_shots" / "attack_wise"
        manifest_paths = sorted(manifest_dir.glob("llama2_7b_tbh_tba_zero_shot_*.json"))

        self.assertEqual(
            [path.name for path in manifest_paths],
            [
                "llama2_7b_tbh_tba_zero_shot_tba_to_tbh.json",
                "llama2_7b_tbh_tba_zero_shot_tbh_to_tba.json",
            ],
        )

        expected_positive_attacks = {
            "llama2_7b_tbh_tba_zero_shot_tba_to_tbh.json": (
                {"toxic_backdoors_alpaca"},
                {"toxic_backdoors_hard"},
            ),
            "llama2_7b_tbh_tba_zero_shot_tbh_to_tba.json": (
                {"toxic_backdoors_hard"},
                {"toxic_backdoors_alpaca"},
            ),
        }

        for manifest_path in manifest_paths:
            train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=manifest_path)
            train_identities = infer_attack_sample_identities(train_items)
            infer_identities = infer_attack_sample_identities(infer_items)

            self.assertEqual(sum(1 for item in train_items if item.label == 0), 250)
            self.assertEqual(sum(1 for item in train_items if item.label == 1), 250)
            self.assertEqual(sum(1 for item in infer_items if item.label == 0), 250)
            self.assertEqual(sum(1 for item in infer_items if item.label == 1), 250)

            train_positive_attacks = {
                identity.attack_name
                for item, identity in zip(train_items, train_identities)
                if item.label == 1
            }
            infer_positive_attacks = {
                identity.attack_name
                for item, identity in zip(infer_items, infer_identities)
                if item.label == 1
            }
            self.assertEqual(
                (train_positive_attacks, infer_positive_attacks),
                expected_positive_attacks[manifest_path.name],
            )


if __name__ == "__main__":
    unittest.main()
