import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/evaluation')))
from fold_pairing import group_patients_by_fold_pair


def test_same_index_patients_are_paired_correctly():
    binary_folds = [
        {"fold": 0, "val_record_ids": ["100", "101"]},
        {"fold": 1, "val_record_ids": ["102"]},
    ]
    multiclass_folds = [
        {"fold": 0, "val_record_ids": ["100", "101"]},
        {"fold": 1, "val_record_ids": ["102"]},
    ]

    pairs = group_patients_by_fold_pair(binary_folds, multiclass_folds)

    assert pairs == {(0, 0): ["100", "101"], (1, 1): ["102"]}


def test_mismatched_indices_are_still_recovered():
    # Patient "100" is in binary fold 0's val set but multiclass fold 2's val set - the old
    # same-index intersection would drop it entirely; this must recover it under (0, 2).
    binary_folds = [
        {"fold": 0, "val_record_ids": ["100", "101"]},
        {"fold": 1, "val_record_ids": ["102"]},
    ]
    multiclass_folds = [
        {"fold": 2, "val_record_ids": ["100"]},
        {"fold": 1, "val_record_ids": ["101", "102"]},
    ]

    pairs = group_patients_by_fold_pair(binary_folds, multiclass_folds)

    assert pairs == {(0, 2): ["100"], (0, 1): ["101"], (1, 1): ["102"]}


def test_patient_missing_from_one_stage_is_excluded():
    # "999" only ever appears in binary val sets (e.g. an all-N patient, never eligible for
    # multiclass) - can't be cascade-evaluated since no held-out multiclass model exists for it.
    binary_folds = [{"fold": 0, "val_record_ids": ["100", "999"]}]
    multiclass_folds = [{"fold": 0, "val_record_ids": ["100"]}]

    pairs = group_patients_by_fold_pair(binary_folds, multiclass_folds)

    assert pairs == {(0, 0): ["100"]}
    all_patients = [p for group in pairs.values() for p in group]
    assert "999" not in all_patients


def test_no_patient_appears_in_more_than_one_pair():
    binary_folds = [
        {"fold": 0, "val_record_ids": ["100", "101", "102"]},
        {"fold": 1, "val_record_ids": ["103"]},
    ]
    multiclass_folds = [
        {"fold": 0, "val_record_ids": ["100"]},
        {"fold": 1, "val_record_ids": ["101", "102", "103"]},
    ]

    pairs = group_patients_by_fold_pair(binary_folds, multiclass_folds)

    all_patients = [p for group in pairs.values() for p in group]
    assert len(all_patients) == len(set(all_patients))
    assert set(all_patients) == {"100", "101", "102", "103"}


def test_empty_inputs_return_empty_dict():
    assert group_patients_by_fold_pair([], []) == {}
