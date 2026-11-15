import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    try:
        from tmu.clause_bank import clause_bank_tensor_core as core
        core._ensure_context()
    except (ImportError, RuntimeError) as error:
        pytest.skip(str(error))

@pytest.mark.parametrize('shape,patch', [((6,6,2),[(3,3),(5,5)]),((32,32,24),[(3,3),(5,5)])])
def test_shared_scores_match_existing_matcher_and_numpy(shape,patch):
    import pickle
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    from tmu.models.tensor_coalesced import shared_class_sums
    rng=np.random.RandomState(15)
    x=rng.randint(0,2,(4,*shape)).astype(np.uint32);y=np.arange(4,dtype=np.uint32)%3
    m=TMCoalescedClassifier(17,19,3.9,platform='tensor',patch_dim=patch,seed=7,
        image_invariance='horizontal',patch_invariance='horizontal')
    m.fit(x,y);b=m.clause_bank
    weights=[m.weight_banks[c].get_weights() for c in range(3)]
    expected=b.predict_class_sums(list(range(3)),[b]*3,weights,None,device_x=b.stage_images(x),n_rows=len(x),preserve_rng=True)
    np.testing.assert_array_equal(shared_class_sums(m,x),expected)
    packed=b.prepare_X(x).reshape(len(x),b.number_of_patches,b.number_of_ta_chunks)
    b._pull();actions=b.clause_bank.reshape(17,b.number_of_ta_chunks,b.number_of_state_bits_ta)[:,:,-1].copy();actions[:,-1]&=b._filter
    votes=np.array([[bool(a.any()) and np.any(np.all((row&a)==a,axis=1)) for a in actions] for row in packed],np.int32)
    np.testing.assert_array_equal(shared_class_sums(m,x),votes@np.stack(weights).T)
    for c in range(3):m.weight_banks[c].get_weights()[:]=rng.randint(-2147483648,2147483647,17,dtype=np.int32)
    np.testing.assert_array_equal(shared_class_sums(m,x),votes@np.stack(weights).T)

    # Checkpoint and fallback retain scores without serializing CUDA handles.
    expected=shared_class_sums(m,x)
    restored=pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(shared_class_sums(restored,x),expected)
    fallback=b._uses_fused_conv
    b._uses_fused_conv=lambda:False
    np.testing.assert_array_equal(shared_class_sums(m,x),expected)
    b._uses_fused_conv=fallback
    # Further learning must invalidate cached extracted masks.
    m.fit(x,y)
    reference=b.predict_class_sums(list(range(3)),[b]*3,[m.weight_banks[c].get_weights() for c in range(3)],None,device_x=b.stage_images(x),n_rows=len(x),preserve_rng=True)
    np.testing.assert_array_equal(shared_class_sums(m,x),reference)
