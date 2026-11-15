"""Vectorize Coalesced inference votes without changing clause evaluation."""
import numpy as np
from tmu.weight_bank import WeightBank
from tmu.clause_bank.clause_bank import ClauseBank
from tmu.clause_bank.clause_bank_tensor_core import ClauseBankTensorCore

_GET_WEIGHTS = WeightBank.get_weights
_PREDICT = {ClauseBank: ClauseBank.calculate_clause_outputs_predict,
            ClauseBankTensorCore: ClauseBankTensorCore.calculate_clause_outputs_predict}


def coalesced_weight_matrix(model):
    """Custom score, bank, or weight hooks retain the upstream scalar path."""
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    if getattr(model.compute_class_sums, '__func__', None) is not TMCoalescedClassifier._compute_class_sums_default:
        return None
    bank = model.clause_bank
    if (type(bank) not in _PREDICT or
            getattr(bank.calculate_clause_outputs_predict, '__func__', None) is not _PREDICT[type(bank)]):
        return None
    rows = [model.weight_banks[cls] for cls in range(model.number_of_classes)]
    if not rows or any(type(row) is not WeightBank or
                       getattr(row.get_weights, '__func__', None) is not _GET_WEIGHTS for row in rows):
        return None
    weights = [row.get_weights() for row in rows]
    if any(row.dtype != np.int32 or row.shape != (bank.number_of_clauses,) for row in weights):
        return None
    return np.stack(weights)


def coalesced_class_sums(model, encoded, row, clip, weights):
    outputs = model.clause_bank.calculate_clause_outputs_predict(encoded, row)
    sums = (weights @ outputs).astype(np.int32)
    # Keep NumPy's scalar clipping and promotion contract, including large T.
    return [np.clip(value, -model.T, model.T) for value in sums] if clip else sums
