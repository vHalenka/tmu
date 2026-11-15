import pickle
import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    try:
        from tmu.clause_bank import clause_bank_tensor_core as core
        core._ensure_context()
    except (ImportError, RuntimeError) as error:
        pytest.skip(str(error))


@pytest.mark.parametrize('batch', [4, 100])
@pytest.mark.parametrize('balanced', [False, True])
def test_streamed_packing_preserves_epoch_and_rng(balanced, batch):
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    rng = np.random.RandomState(29)
    x = rng.randint(0, 2, (123, 6, 6, 2)).astype(np.uint32)
    y = (np.arange(len(x)) % 3).astype(np.uint32)
    models = [TMCoalescedClassifier(17, 19, 3.9, platform='tensor', seed=31,
        batch_size=batch, pack_batches=streamed,
        patch_dim=[(3, 3), (5, 5)], image_invariance='horizontal',
        patch_invariance='horizontal', output_balancing=balanced,
        focused_negative_sampling=True, weighted_clauses=True,
        clause_drop_p=.1, literal_drop_p=.1, max_included_literals=8)
        for streamed in [False, True]]
    for _ in range(2):
        for m in models:
            sizes = []
            stage = m.clause_bank.stage_images if m.initialized else None
            if stage is None:
                m.init(x, y)
                stage = m.clause_bank.stage_images
            if m.pack_batches:
                def bounded(images):
                    sizes.append(len(images))
                    return stage(images)
                m.clause_bank.stage_images = bounded
            m.fit(x, y)
            if m.pack_batches:
                assert sizes and max(sizes) <= batch
                m.clause_bank.stage_images = stage
            m.clause_bank._pull()
        a, b = models
        assert b.clause_bank._enc_cache is None
        for name in ['_coalesced_parallel_state', 'clause_active', 'literal_active', 'update_ps']:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        np.testing.assert_array_equal(a.clause_bank.clause_bank, b.clause_bank.clause_bank)
        for c in range(3):
            np.testing.assert_array_equal(a.weight_banks[c].get_weights(), b.weight_banks[c].get_weights())
        assert a._coalesced_pcg_state == b._coalesced_pcg_state
        assert pickle.dumps(a.rng.get_state()) == pickle.dumps(b.rng.get_state())
        for u, v in zip(a.predict(x, return_class_sums=True), b.predict(x, return_class_sums=True)):
            np.testing.assert_array_equal(u, v)
