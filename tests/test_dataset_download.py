from contextlib import redirect_stdout
import io
import unittest
from unittest import mock

from upeftguard.utilities.data import dataset_download


class DatasetDownloadParseArgsTests(unittest.TestCase):
    def test_all_with_dataset_selects_entire_folder(self) -> None:
        args = dataset_download.parse_args(
            ["--dataset", "custom_folder", "--all"]
        )

        self.assertTrue(args.clean_request.uses_all)
        self.assertTrue(args.backdoored_request.uses_all)
        self.assertEqual(
            args.clean_sources,
            [dataset_download.Source("custom_folder", 0)],
        )
        self.assertEqual(
            args.backdoored_sources,
            [dataset_download.Source("custom_folder", 1)],
        )

    def test_all_cannot_be_combined_with_partial_selection_flags(self) -> None:
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                dataset_download.parse_args(
                    ["--dataset", "custom_folder", "--all", "--clean", "10"]
                )

    def test_backdoor_alias_with_dataset(self) -> None:
        args = dataset_download.parse_args(
            ["--dataset", "custom_folder", "--backdoor", "150", "152"]
        )

        self.assertEqual(args.backdoored_request.start, 150)
        self.assertEqual(args.backdoored_request.end, 152)
        self.assertEqual(
            args.backdoored_sources,
            [dataset_download.Source("custom_folder", 1)],
        )

    def test_dataset_is_required_without_show_list(self) -> None:
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                dataset_download.parse_args(["--backdoored", "60"])

    def test_dataset_flag_builds_label0_and_label1_sources(self) -> None:
        args = dataset_download.parse_args(
            [
                "--dataset",
                "custom_clean_like_folder",
                "--dataset",
                "custom_backdoor_like_folder",
                "--clean",
                "10",
                "--backdoored",
                "4",
            ]
        )

        self.assertEqual(
            args.clean_sources,
            [
                dataset_download.Source("custom_clean_like_folder", 0),
                dataset_download.Source("custom_backdoor_like_folder", 0),
            ],
        )
        self.assertEqual(
            args.backdoored_sources,
            [
                dataset_download.Source("custom_clean_like_folder", 1),
                dataset_download.Source("custom_backdoor_like_folder", 1),
            ],
        )

    def test_show_list_does_not_require_selection_flags(self) -> None:
        args = dataset_download.parse_args(["--show-list"])

        self.assertTrue(args.show_list)
        self.assertTrue(args.clean_request.is_empty)
        self.assertTrue(args.backdoored_request.is_empty)


class DatasetDownloadMainTests(unittest.TestCase):
    def test_all_download_selects_all_patterns_for_dataset(self) -> None:
        stdout = io.StringIO()
        dataset = "llama2_7b_imdb_insertsent_rank256_qv"
        with mock.patch.object(
            dataset_download,
            "list_padbench_folders",
            return_value=[dataset],
        ) as mock_folders:
            with mock.patch.object(
                dataset_download,
                "list_available_indices",
                return_value={
                    (dataset, 0): [0, 2],
                    (dataset, 1): [1, 3],
                },
            ) as mock_indices:
                with mock.patch.object(
                    dataset_download,
                    "list_selected_model_sizes",
                    return_value={
                        f"{dataset}/{dataset}_label0_0": 100,
                        f"{dataset}/{dataset}_label0_2": 100,
                        f"{dataset}/{dataset}_label1_1": 100,
                        f"{dataset}/{dataset}_label1_3": 100,
                    },
                ) as mock_sizes:
                    with mock.patch.object(dataset_download, "snapshot_download") as mock_download:
                        with mock.patch(
                            "sys.argv",
                            [
                                "download_dataset",
                                "--dataset",
                                dataset,
                                "--all",
                            ],
                        ):
                            with redirect_stdout(stdout):
                                dataset_download.main()

        mock_folders.assert_called_once_with(dataset_download.DEFAULT_REPO_ID)
        mock_indices.assert_called_once_with(dataset_download.DEFAULT_REPO_ID, [dataset])
        mock_sizes.assert_called_once()
        mock_download.assert_called_once_with(
            repo_id=dataset_download.DEFAULT_REPO_ID,
            repo_type="dataset",
            allow_patterns=[
                f"{dataset}/{dataset}_label0_0/*",
                f"{dataset}/{dataset}_label0_2/*",
                f"{dataset}/{dataset}_label1_1/*",
                f"{dataset}/{dataset}_label1_3/*",
            ],
            local_dir=mock.ANY,
        )

    def test_show_list_prints_available_folders_and_exits(self) -> None:
        stdout = io.StringIO()
        with mock.patch.object(
            dataset_download,
            "list_padbench_folders",
            return_value=["folder_alpha", "folder_beta"],
        ) as mock_list:
            with mock.patch(
                "sys.argv",
                ["download_dataset", "--show-list"],
            ):
                with redirect_stdout(stdout):
                    dataset_download.main()

        output = stdout.getvalue()
        self.assertIn("Listing PADBench folders", output)
        self.assertIn("Available PADBench folders", output)
        self.assertIn("folder_alpha", output)
        self.assertIn("folder_beta", output)
        mock_list.assert_called_once_with(dataset_download.DEFAULT_REPO_ID)

    def test_show_list_with_dataset_prints_clean_and_backdoor_indices(self) -> None:
        stdout = io.StringIO()
        with mock.patch.object(
            dataset_download,
            "list_padbench_folders",
            return_value=["llama2_7b_toxic_backdoors_alpaca_rank256_qv"],
        ) as mock_folders:
            with mock.patch.object(
                dataset_download,
                "list_available_indices",
                return_value={
                    ("llama2_7b_toxic_backdoors_alpaca_rank256_qv", 0): [0, 2, 4],
                    ("llama2_7b_toxic_backdoors_alpaca_rank256_qv", 1): [1, 3],
                },
            ) as mock_indices:
                with mock.patch(
                    "sys.argv",
                    [
                        "download_dataset",
                        "--dataset",
                        "llama2_7b_toxic_backdoors_alpaca_rank256_qv",
                        "--show-list",
                    ],
                ):
                    with redirect_stdout(stdout):
                        dataset_download.main()

        output = stdout.getvalue()
        self.assertIn("Listing available indices", output)
        self.assertIn("llama2_7b_toxic_backdoors_alpaca_rank256_qv:", output)
        self.assertIn("clean (label0): 0, 2, 4", output)
        self.assertIn("backdoor (label1): 1, 3", output)
        mock_folders.assert_called_once_with(dataset_download.DEFAULT_REPO_ID)
        mock_indices.assert_called_once_with(
            dataset_download.DEFAULT_REPO_ID,
            ["llama2_7b_toxic_backdoors_alpaca_rank256_qv"],
        )


if __name__ == "__main__":
    unittest.main()
