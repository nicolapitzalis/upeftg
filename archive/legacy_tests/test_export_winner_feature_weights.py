import csv
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from upeftguard.utilities.artifacts.export_winner_feature_weights import (
    export_winner_feature_weights,
    merge_winner_feature_weights_export,
    prepare_winner_feature_weights_export,
    run_winner_feature_weights_export_worker,
)


def _read_coefficients(path: Path) -> dict[str, float]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {str(row["feature_name"]): float(row["coefficient"]) for row in rows}


class TestExportWinnerFeatureWeights(unittest.TestCase):
    def test_distributed_permutation_export_matches_local_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "runs" / "supervised" / "kernel_export_test"
            for subdir in ["features", "models", "reports", "plots", "logs"]:
                (run_dir / subdir).mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(123)
            x_train = rng.normal(size=(18, 4)).astype(np.float32)
            y_train = np.asarray([0] * 9 + [1] * 9, dtype=np.int32)
            x_train[y_train == 1, 0] += 2.0
            x_train[y_train == 1, 1] += 1.0

            model = Pipeline(
                steps=[
                    ("normalizer", StandardScaler()),
                    ("model", SVC(kernel="rbf", C=1.0, gamma=0.5, random_state=42)),
                ]
            )
            model.fit(x_train, y_train)
            joblib.dump(model, run_dir / "models" / "best_model.joblib")

            feature_names = [
                "block0.energy",
                "block0.kurtosis",
                "block1.energy",
                "block1.kurtosis",
            ]
            report_path = run_dir / "reports" / "supervised_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "representation": {
                            "extractor_metadata": {
                                "feature_names": feature_names,
                                "feature_dim": len(feature_names),
                            }
                        },
                        "tuning": {
                            "winner": {
                                "model_name": "kernel_svm",
                                "params": {"C": 1.0, "gamma": 0.5, "class_weight": None},
                                "task_index": 0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
            tuning_manifest_path.write_text(
                json.dumps(
                    {
                        "extractor": {
                            "metadata": {
                                "feature_names": feature_names,
                                "feature_dim": len(feature_names),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            local_outputs = export_winner_feature_weights(
                run_dir=run_dir,
                output_prefix="local_weights",
                report_path=report_path,
                manifest_path=tuning_manifest_path,
                train_features=x_train,
                train_labels=y_train,
                random_state=42,
                n_jobs=1,
            )

            train_feature_path = run_dir / "features" / "winner_feature_weights_train_features.npy"
            train_label_path = run_dir / "features" / "winner_feature_weights_train_labels.npy"
            np.save(train_feature_path, x_train)
            np.save(train_label_path, y_train)

            prepared = prepare_winner_feature_weights_export(
                run_dir=run_dir,
                output_prefix="distributed_weights",
                report_path=report_path,
                manifest_path=tuning_manifest_path,
                train_feature_matrix_path=train_feature_path,
                train_labels_path=train_label_path,
                random_state=42,
                n_tasks=3,
            )
            self.assertEqual(prepared["mode"], "permutation")
            self.assertEqual(prepared["n_tasks"], 3)

            for task_index in range(prepared["n_tasks"]):
                result = run_winner_feature_weights_export_worker(
                    run_dir=run_dir,
                    output_prefix="distributed_weights",
                    task_index=task_index,
                    n_jobs=1,
                )
                self.assertEqual(result["status"], "ok")

            merged = merge_winner_feature_weights_export(
                run_dir=run_dir,
                output_prefix="distributed_weights",
                report_path=report_path,
                manifest_path=tuning_manifest_path,
            )

            self.assertEqual(
                _read_coefficients(Path(local_outputs["coefficient_csv"])),
                _read_coefficients(Path(merged["coefficient_csv"])),
            )

            with open(local_outputs["metadata_json"], "r", encoding="utf-8") as f:
                local_metadata = json.load(f)
            with open(merged["metadata_json"], "r", encoding="utf-8") as f:
                merged_metadata = json.load(f)

            self.assertEqual(local_metadata["importance_type"], "permutation_importance")
            self.assertEqual(merged_metadata["importance_type"], "permutation_importance")
            self.assertEqual(merged_metadata["execution_mode"], "distributed_permutation")
            self.assertFalse((run_dir / "reports" / "distributed_weights_manifest.json").exists())
            self.assertFalse((run_dir / "reports" / "distributed_weights_parts").exists())
