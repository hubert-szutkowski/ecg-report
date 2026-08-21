def group_patients_by_fold_pair(binary_folds: list, multiclass_folds: list) -> dict:
    """
    Maps each patient held out from BOTH stages to the (binary_fold, multiclass_fold) index
    pair of the specific models that never saw them during training, instead of assuming the
    two independently-computed CV splits happen to agree at the same fold index.

    Binary and multiclass folds come from two separate StratifiedGroupKFold calls over
    different populations (binary keeps all windows including N; multiclass drops N entirely)
    and different stratification targets, so "fold 3" in one has no reason to contain the same
    patients as "fold 3" in the other - measured on a real run, assuming they matched silently
    evaluated only 5 of ~37 eligible patients because the same-index intersection was mostly
    empty by chance.

    Parameters:
        - binary_folds: list of {"fold": int, "val_record_ids": [...], ...} (a
          fold_splits_binary.json's "folds" list)
        - multiclass_folds: same shape, from fold_splits_multiclass.json
    Returns:
        - dict {(binary_fold, multiclass_fold): [patient_id, ...]} covering every patient held
          out of BOTH stages somewhere. A patient held out of only one stage (or neither) is
          excluded - the cascade needs a model from each stage that never trained on them.
    """
    patient_to_binary_fold = {}
    for split in binary_folds:
        for patient in split["val_record_ids"]:
            patient_to_binary_fold[patient] = split["fold"]

    patient_to_multiclass_fold = {}
    for split in multiclass_folds:
        for patient in split["val_record_ids"]:
            patient_to_multiclass_fold[patient] = split["fold"]

    common_patients = set(patient_to_binary_fold) & set(patient_to_multiclass_fold)

    pairs: dict = {}
    for patient in sorted(common_patients):
        key = (patient_to_binary_fold[patient], patient_to_multiclass_fold[patient])
        pairs.setdefault(key, []).append(patient)

    return pairs
