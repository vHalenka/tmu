import pickle
import numpy as np
import pytest


@pytest.fixture(scope='module')
def gpu():
    try:
        from tmu.clause_bank import clause_bank_tensor_core as core
        core._ensure_context()
    except (ImportError,RuntimeError) as error:
        pytest.skip(str(error))
    return core


def test_specialized_geometry_keeps_mutable_learning_parameters_runtime(gpu):
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    from tmu.models.tensor_coalesced import CoalescedGPU
    gen=np.random.RandomState(17)
    x=gen.randint(0,2,(23,20,20,2)).astype(np.uint32);y=(np.arange(len(x))%3).astype(np.uint32)
    models=[]
    for serial in [True,False]:
        m=TMCoalescedClassifier(33,19,3.9,platform='tensor',seed=91,patch_dim=(3,3),
            batch_size=7,focused_negative_sampling=True,weighted_clauses=True,
            clause_drop_p=.1,literal_drop_p=.1)
        m.init(x,y);trainer=m._coalesced_gpu=CoalescedGPU(m)
        if trainer.cooperative is None or trainer.maximum_blocks<2:
            pytest.skip("Cooperative multi-block CUDA launch unavailable")
        assert trainer.compiled_geometry==(m.clause_bank.number_of_ta_chunks,
            m.clause_bank.number_of_literals,m.clause_bank.number_of_patches,m.clause_bank.number_of_state_bits_ta)
        if serial:trainer.blocks=1
        else:assert trainer.blocks>1
        models.append(m)
    for epoch in range(3):
        if epoch:
            for m in models:
                m.T=23;m.s=m.clause_bank.s=5.3
                m.type_i_p=.37;m.type_ii_p=1.63
                m.max_positive_clauses=17;m.clause_bank.max_included_literals=9
        for m in models:m.fit(x,y);m.clause_bank._pull()
        a,b=models
        np.testing.assert_array_equal(a.clause_bank.clause_bank,b.clause_bank.clause_bank)
        np.testing.assert_array_equal(a._coalesced_parallel_state,b._coalesced_parallel_state)
        assert a._coalesced_pcg_state==b._coalesced_pcg_state
        assert pickle.dumps(a.rng.get_state())==pickle.dumps(b.rng.get_state())
        for c in range(3):np.testing.assert_array_equal(a.weight_banks[c].get_weights(),b.weight_banks[c].get_weights())
        for u,v in zip(a.predict(x,return_class_sums=True),b.predict(x,return_class_sums=True)):np.testing.assert_array_equal(u,v)
