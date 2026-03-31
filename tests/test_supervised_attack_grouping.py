import unittest
from pathlib import Path

import numpy as np

from upeftguard.supervised.pipeline import _build_threshold_specs, _summarize_attack_groups
from upeftguard.utilities.core.manifest import ManifestItem, infer_attack_sample_identities


def make_manifest_item(subset_name: str, label: int, index: int) -> ManifestItem:
    model_name = f"{subset_name}_label{label}_{index}"
    model_dir = Path("/tmp/data") / subset_name / model_name
    return ManifestItem(
        raw_entry=str(model_dir),
        model_dir=model_dir,
        adapter_path=model_dir / "adapter_model.safetensors",
        model_name=model_name,
        label=label,
    )


class TestSupervisedAttackGrouping(unittest.TestCase):
    def test_mixed_subset_names_share_global_clean_pool(self):
        items = [
            make_manifest_item("llama2_7b_imdb_syntactic_rank256_qv", 0, 0),
            make_manifest_item("llama2_7b_imdb_syntactic_rank256_qv", 1, 1),
            make_manifest_item("llama2_7b_toxic_backdoors_alpaca_rank256_qv", 0, 0),
            make_manifest_item("llama2_7b_toxic_backdoors_alpaca_rank256_qv", 1, 1),
        ]
        identities = infer_attack_sample_identities(items)
        scores = np.asarray([0.05, 0.95, 0.15, 0.9], dtype=np.float64)
        summary = _summarize_attack_groups(
            sample_identities=identities,
            labels=[0, 1, 0, 1],
            scores=scores,
            threshold_specs=_build_threshold_specs(scores, [50.0]),
        )

        self.assertEqual(summary["n_attacks"], 2)
        self.assertCountEqual(list(summary["attacks"].keys()), ["syntactic", "toxic_backdoors_alpaca"])
        self.assertEqual(summary["clean_pool"]["n_samples"], 2)
        self.assertCountEqual(
            summary["clean_pool"]["source_subsets"],
            [
                "llama2_7b_imdb_syntactic_rank256_qv",
                "llama2_7b_toxic_backdoors_alpaca_rank256_qv",
            ],
        )
        self.assertEqual(summary["attacks"]["syntactic"]["label_counts"]["clean"], 2)
        self.assertEqual(summary["attacks"]["syntactic"]["label_counts"]["backdoored"], 1)
        self.assertEqual(summary["attacks"]["toxic_backdoors_alpaca"]["label_counts"]["clean"], 2)
        self.assertEqual(summary["attacks"]["toxic_backdoors_alpaca"]["label_counts"]["backdoored"], 1)
        self.assertEqual(
            summary["attacks"]["toxic_backdoors_alpaca"]["source_subsets"],
            ["llama2_7b_toxic_backdoors_alpaca_rank256_qv"],
        )

    def test_single_attack_subset_uses_folder_name_instead_of_unknown(self):
        items = [
            make_manifest_item("llama2_7b_toxic_backdoors_alpaca_rank256_qv", 0, 0),
            make_manifest_item("llama2_7b_toxic_backdoors_alpaca_rank256_qv", 1, 1),
        ]
        identities = infer_attack_sample_identities(items)
        scores = np.asarray([0.1, 0.9], dtype=np.float64)
        summary = _summarize_attack_groups(
            sample_identities=identities,
            labels=[0, 1],
            scores=scores,
            threshold_specs=_build_threshold_specs(scores, [50.0]),
        )

        self.assertEqual(summary["clean_pool"]["n_samples"], 1)
        self.assertCountEqual(list(summary["attacks"].keys()), ["toxic_backdoors_alpaca"])
        self.assertNotIn("unknown", summary["attacks"])

    def test_known_attack_aliases_are_canonicalized(self):
        items = [
            make_manifest_item("llama2_7b_imdb_RIPPLE_rank256_qv", 1, 0),
            make_manifest_item("llama2_7b_imdb_insertsent_rank256_qv", 1, 1),
            make_manifest_item("llama2_7b_imdb_syntactic_rank256_qv", 1, 2),
            make_manifest_item("llama2_7b_imdb_stykbd_rank256_qv", 1, 3),
        ]
        identities = infer_attack_sample_identities(items)

        self.assertEqual([identity.attack_name for identity in identities], ["RIPPLE", "insertsent", "syntactic", "stybkd"])


if __name__ == "__main__":
    unittest.main()
