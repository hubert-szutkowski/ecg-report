import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/AAMI_classification_multistages')))

import multistage_train


def test_build_parser_supports_binary_and_multiclass_tasks():
    parser = multistage_train.build_parser()
    args = parser.parse_args(["--data-dir", "data", "--task", "both"])

    assert args.data_dir == "data"
    assert args.task == "both"
    assert args.selected_samples == 20


def test_default_task_is_both():
    parser = multistage_train.build_parser()
    args = parser.parse_args(["--data-dir", "data"])

    assert args.task == "both"
