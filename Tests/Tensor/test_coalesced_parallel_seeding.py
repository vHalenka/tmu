import numpy as np
import pytest
from tmu.models.tensor_coalesced import parallel_seed_states


def test_parallel_seeds_match_splitmix_vectors_and_mcg_state_class():
    expected=np.array([0xe220a8397b1dcdaf,0x6e789e6aa1b965f7,0x06c45d188009454f],np.uint64)
    np.testing.assert_array_equal(parallel_seed_states(0,3),expected)
    for seed in [1,7,31,91,2**64-1]:
        states=parallel_seed_states(seed,4096)
        assert states.dtype==np.uint64
        assert np.all(states&np.uint64(3)==3)
        assert len(np.unique(states))==len(states)
        np.testing.assert_array_equal(states[:64],parallel_seed_states(seed,64))


def test_parallel_low_seed_bits_change_clause_outputs():
    def first_outputs(seed):
        x=parallel_seed_states(seed,256)
        return ((x^(x>>np.uint64(22))) >> (np.uint64(22)+(x>>np.uint64(61)))).astype(np.uint32)
    a,b=first_outputs(7),first_outputs(31)
    assert np.count_nonzero(a!=b)==256


@pytest.fixture
def gpu():
    try:
        from tmu.clause_bank import clause_bank_tensor_core as core
        core._ensure_context()
    except (ImportError,RuntimeError) as error:
        pytest.skip(str(error))
    return core


def test_parallel_checkpoint_preserves_existing_even_streams(monkeypatch,gpu):
    import pickle
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    import tmu.models.tensor_coalesced as backend
    x=np.random.RandomState(123).randint(0,2,(32,4)).astype(np.uint32);y=x[:,0].copy()
    model=TMCoalescedClassifier(number_of_clauses=8,T=10,s=3.9,seed=7,platform='tensor',weighted_clauses=True)
    model.fit(x,y)
    assert np.all(model._coalesced_parallel_state & np.uint64(3)==3)
    # Old checkpoints can contain even states. Never silently reset/reseed them.
    model._coalesced_parallel_state=np.arange(2,18,2,dtype=np.uint64)
    restored=pickle.loads(pickle.dumps(model))
    def forbidden(*args):raise AssertionError('Existing checkpoint was reseeded')
    monkeypatch.setattr(backend,'parallel_seed_states',forbidden)
    model.fit(x,y);restored.fit(x,y)
    model.clause_bank._pull();restored.clause_bank._pull()
    np.testing.assert_array_equal(model.clause_bank.clause_bank,restored.clause_bank.clause_bank)
    np.testing.assert_array_equal(model._coalesced_parallel_state,restored._coalesced_parallel_state)
    for cls in model.weight_banks:
        np.testing.assert_array_equal(model.weight_banks[cls].get_weights(),restored.weight_banks[cls].get_weights())
