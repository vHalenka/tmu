"""Coalesced scoring retains upstream values, dtypes, hooks and RNG state."""
import pickle
import numpy as np
import pytest
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.clause_bank.tensor_shared import coalesced_weight_matrix


def make_model(classes=3):
    rng = np.random.RandomState(17)
    x = rng.randint(0, 2, (9, 33)).astype(np.uint32)
    y = (np.arange(len(x)) % classes).astype(np.uint32)
    model = TMCoalescedClassifier(8, 17, 3.9, seed=7)
    model.init(x, y)
    for clause in range(8):
        model.clause_bank.set_ta_state(clause, clause, 255)
    for cls in range(classes):
        model.weight_banks[cls].get_weights()[:] = rng.randint(-100, 100, 8)
    return model, x


@pytest.mark.parametrize('clip', [False, True])
@pytest.mark.parametrize('threshold', [7, 2**31 + 3])
def test_coalesced_prediction_matches_scalar_upstream_including_wrap(clip, threshold):
    model, x = make_model()
    model.T = threshold
    model.weight_banks[0].get_weights()[:] = np.iinfo(np.int32).max
    model.weight_banks[1].get_weights()[:] = np.iinfo(np.int32).min
    encoded = model.clause_bank.prepare_X(x)
    expected = np.array([model.compute_class_sums(encoded, row, clip) for row in range(len(x))])
    before = pickle.dumps(model.rng.get_state())
    labels, scores = model.predict(x, clip_class_sum=clip, return_class_sums=True)
    np.testing.assert_array_equal(scores, expected)
    assert scores.dtype == expected.dtype
    np.testing.assert_array_equal(labels, np.argmax(expected, axis=1))
    assert pickle.dumps(model.rng.get_state()) == before
    model.weight_banks[0].get_weights()[:] = -1
    changed = np.array([model.compute_class_sums(encoded, row, clip) for row in range(len(x))])
    np.testing.assert_array_equal(model.predict(x, clip, True)[1], changed)


def test_coalesced_normal_weights_use_matrix_and_custom_scoring_keeps_hook():
    model, x = make_model()
    assert coalesced_weight_matrix(model).shape == (3, 8)
    calls = []
    def scores(encoded_X_test, ith_sample, clip_class_sum):
        calls.append(ith_sample)
        return [ith_sample, -ith_sample, 1]
    model.compute_class_sums = scores
    assert coalesced_weight_matrix(model) is None
    labels, actual = model.predict(x, return_class_sums=True)
    assert calls == list(range(len(x)))
    np.testing.assert_array_equal(actual, [[row, -row, 1] for row in range(len(x))])


def test_coalesced_custom_weight_getter_and_bank_keep_scalar_dispatch():
    model, _ = make_model()
    row = model.weight_banks[0]
    original = row.get_weights
    row.get_weights = lambda: original()
    assert coalesced_weight_matrix(model) is None
    del row.get_weights
    model.clause_bank.calculate_clause_outputs_predict = lambda *_: np.ones(8, np.uint32)
    assert coalesced_weight_matrix(model) is None


def test_coalesced_pickle_does_not_store_prediction_weight_caches():
    model, x = make_model()
    expected = model.predict(x, return_class_sums=True)
    restored = pickle.loads(pickle.dumps(model))
    actual = restored.predict(x, return_class_sums=True)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


def test_coalesced_keyword_only_score_hook_keeps_upstream_call_contract():
    model, x = make_model()
    calls = []
    def hook(*, encoded_X_test, ith_sample, clip_class_sum):
        calls.append(ith_sample)
        return [1, ith_sample, -1]
    model.compute_class_sums = hook
    model.predict(x)
    assert calls == list(range(len(x)))


def test_unknown_bank_with_plain_function_uses_scalar_path():
    from types import SimpleNamespace
    model, _ = make_model()
    model.clause_bank = SimpleNamespace(number_of_clauses=8,
        calculate_clause_outputs_predict=lambda *_: np.ones(8, np.uint32))
    assert coalesced_weight_matrix(model) is None


def test_class_level_weight_hook_keeps_per_sample_calls(monkeypatch):
    from tmu.weight_bank import WeightBank
    model, x = make_model()
    original, calls = WeightBank.get_weights, []
    def getter(row):
        calls.append(row)
        return original(row)
    monkeypatch.setattr(WeightBank, 'get_weights', getter)
    model.predict(x)
    assert len(calls) == len(x) * model.number_of_classes
