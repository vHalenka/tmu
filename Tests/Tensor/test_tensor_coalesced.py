"""Coalesced GPU contracts; parallel feedback can learn a different model."""
import pickle
import numpy as np
import pytest
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier


def data(rows=27, features=17):
    x = np.random.RandomState(123).randint(0, 2, (rows, features)).astype(np.uint32)
    y = (np.arange(rows) % 3).astype(np.uint32)
    return x, y


def test_feedback_mode_is_no_longer_a_constructor_option():
    for mode in ['ordered', 'parallel']:
        with pytest.raises(TypeError, match='feedback_mode'):
            TMCoalescedClassifier(12, 13, 3.9, feedback_mode=mode)


def test_ordered_checkpoint_cannot_silently_resume_as_parallel():
    model = TMCoalescedClassifier(12, 13, 3.9, platform='tensor', seed=7)
    model.feedback_mode = 'ordered'  # Attribute from an older checkpoint.
    x, y = data()
    with pytest.raises(ValueError, match='only supports parallel'):
        model.fit(x, y)


@pytest.mark.parametrize('batch', [0, -1, 1.5])
def test_invalid_batch_size_is_rejected(batch):
    with pytest.raises(ValueError, match='batch_size'):
        TMCoalescedClassifier(12, 13, 3.9, batch_size=batch)


def test_schedule_keeps_upstream_groups_and_drops_incomplete_tail():
    from tmu.models.tensor_coalesced import sample_schedule
    labels = np.array([0, 0, 2, 1, 1, 2, 0], np.uint32)
    order = np.array([6, 0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(sample_schedule(order, labels, 3, False), order[:6])
    np.testing.assert_array_equal(sample_schedule(order, labels, 3, True), [6, 3, 2])


def test_one_class_schedule_keeps_every_row():
    from tmu.models.tensor_coalesced import sample_schedule
    np.testing.assert_array_equal(sample_schedule([2, 0, 1], np.zeros(3, np.uint32), 1, True), [2, 0, 1])


@pytest.fixture(scope='module')
def gpu():
    try:
        from tmu.clause_bank import clause_bank_tensor_core as core
        core._ensure_context()
    except (ImportError, RuntimeError) as error:
        pytest.skip(str(error))
    return core


def signature(model, x):
    bank = model.clause_bank
    if hasattr(bank, '_pull'):
        bank._pull()
    return (bank.clause_bank.copy(),
            np.stack([model.weight_banks[c].get_weights() for c in range(model.number_of_classes)]),
            model.predict(x, return_class_sums=True), pickle.dumps(model.rng.get_state()))


def assert_same(left, right):
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])
    for a, b in zip(left[2], right[2]):
        np.testing.assert_array_equal(a, b)
    assert left[3] == right[3]




def test_gpu_checkpoint_continues_same_learning_and_handles_public_edits(gpu):
    x, y = data(33, 33)
    model = TMCoalescedClassifier(18, 19, 5., platform='tensor', seed=17,
                                  batch_size=7)
    model.fit(x, y)
    weights = model.weight_banks[0].get_weights()
    weights[:] = -2
    model.set_ta_state(0, 0, 255)
    encoded = model.clause_bank.prepare_X(x)
    expected_scores = np.array([model.compute_class_sums(encoded, row, False) for row in range(len(x))])
    np.testing.assert_array_equal(model.predict(x, return_class_sums=True)[1], expected_scores)
    snapshot = pickle.dumps(model)
    model.fit(x, y)
    expected = signature(model, x)
    restored = pickle.loads(snapshot)
    restored.fit(x, y)
    assert_same(signature(restored, x), expected)
    assert model.weight_banks[0].get_weights() is weights
    np.testing.assert_array_equal(weights, restored.weight_banks[0].get_weights())
    empty = restored.predict(x[:0], return_class_sums=True)
    assert empty[0].shape == (0,) and empty[1].shape == (0, 3)


def test_parallel_is_repeatable_but_does_not_promise_cpu_parity(gpu):
    x, y = data(60, 17)
    signatures = []
    for platform in ['CPU', 'tensor', 'tensor']:
        model = TMCoalescedClassifier(16, 17, 3.9, platform=platform, seed=7)
        model.fit(x, y)
        signatures.append(signature(model, x))
    assert_same(signatures[1], signatures[2])
    assert not np.array_equal(signatures[0][0], signatures[1][0])


@pytest.mark.parametrize('options,match', [({'type_iii_feedback': True}, 'Type III'),
                                         ({'seed': 0}, 'seed'),
                                         ({'patch_dim': (2, 2)}, 'spatial')])
def test_unsupported_gpu_training_is_rejected_before_running(gpu, options, match):
    x, y = data()
    options = dict(dict(seed=7), **options)
    model = TMCoalescedClassifier(12, 13, 3.9, platform='tensor', **options)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        model.fit(x, y)



@pytest.mark.parametrize('threshold', [.5, 13.5])
def test_changed_fractional_threshold_is_rejected_before_gpu_initialization(threshold):
    x, y = data()
    model = TMCoalescedClassifier(12, 13, 3.9, platform='tensor', seed=7)
    model.T = threshold
    with pytest.raises(ValueError, match='T must'):
        model.fit(x, y)


@pytest.mark.parametrize('blocks',[1,4])
def test_parallel_launch_arguments_match_cuda_signature(monkeypatch,blocks):
    from types import SimpleNamespace
    from pathlib import Path
    from tmu.models.tensor_coalesced import CoalescedGPU
    from .test_tensor_speedup_execution import Driver
    import re
    x,y=data(6,9)
    model=TMCoalescedClassifier(12,13,3.9,seed=7,batch_size=4)
    model.init(x,y);bank=model.clause_bank;driver=Driver();calls=[]
    bank._ta_gpu=driver.mem_alloc(bank.clause_bank.nbytes)
    bank._prediction_generation=0
    monkeypatch.setattr(bank,'_push',lambda:None,raising=False)
    encoded=driver.mem_alloc(24)
    monkeypatch.setattr(bank,'stage_images',lambda _:encoded,raising=False)
    bank._filter=(1<<(bank.number_of_literals%32))-1 if bank.number_of_literals%32 else 0xffffffff
    model.clause_active=np.ones(12,np.int32)
    model.literal_active=np.full(bank.number_of_ta_chunks,0xffffffff,np.uint32)
    trainer=CoalescedGPU.__new__(CoalescedGPU)
    trainer.compiled_geometry=(bank.number_of_ta_chunks,bank.number_of_literals,bank.number_of_patches,bank.number_of_state_bits_ta)
    trainer.bank=bank;trainer.cuda=driver;trainer.blocks=blocks
    trainer.buffers={};trainer.weights=None;trainer.weight_snapshot=None
    trainer.function=SimpleNamespace(prepared_call=lambda grid,block,*args:calls.append(args))
    trainer.cooperative=SimpleNamespace(launch=lambda grid,block,args:calls.append(args))
    trainer.train(model,x,np.arange(len(x),dtype=np.int32),y)
    root=Path(__file__).resolve().parents[2]/'tmu/clause_bank/cuda'
    filename='coalesced.cu' if blocks==1 else 'coalesced_parallel.cu'
    function='tc_coalesced_train' if blocks==1 else 'tc_coalesced_parallel_train'
    declaration=re.search(r'void '+function+r'\((.*?)\)\s*\{',(root/filename).read_text(),re.S)[1]
    assert len(calls)==2
    assert all(len(args)==len(declaration.split(',')) for args in calls)
    pointer_count=13 if blocks==1 else 15
    assert [args[pointer_count] for args in calls]==[4,2]
