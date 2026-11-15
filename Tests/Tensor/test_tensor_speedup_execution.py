"""Exercise the real upstream execution loops with a recording CUDA driver."""
from types import SimpleNamespace
import pickle
import numpy as np
import pytest
from tmu.clause_bank import clause_bank_tensor_core as core


class Pointer:
    def __init__(self, address): self.address = address
    def __int__(self): return self.address


class Driver:
    def __init__(self):
        self.buffers, self.uploads, self.feedback, self.extracts = {}, [], [], 0
    def mem_alloc(self, size):
        pointer = Pointer((len(self.buffers) + 1) * 1000000)
        self.buffers[int(pointer)] = bytearray(max(4, size))
        return pointer
    def memory(self, pointer):
        address = int(pointer)
        for base, buffer in self.buffers.items():
            if base <= address < base + len(buffer):
                return memoryview(buffer)[address-base:]
        raise AssertionError('invalid device pointer')
    def read(self, pointer, dtype, count):
        return np.frombuffer(self.memory(pointer), dtype=dtype, count=count).copy()
    def memcpy_htod(self, pointer, values):
        values = np.ascontiguousarray(values)
        data = values.tobytes()
        self.memory(pointer)[:len(data)] = data
        self.uploads.append((int(pointer), values.copy()))
    def memcpy_dtoh(self, values, pointer):
        values.view(np.uint8).reshape(-1)[:] = np.frombuffer(self.memory(pointer), np.uint8, values.nbytes)


class Kernel:
    def __init__(self, call): self.call = call
    def prepared_call(self, grid, block, *args, **kwargs): return self.call(*args)


def make_bank(monkeypatch, patches=1, clauses=4, bank_class=None):
    driver = Driver()
    monkeypatch.setattr(core, 'cuda', driver)
    monkeypatch.setattr(core, '_ensure_context', lambda: None)
    monkeypatch.setattr(core, '_CTX', SimpleNamespace(synchronize=lambda: None))
    bank = (bank_class or core.ClauseBankTensorCore).__new__(bank_class or core.ClauseBankTensorCore)
    bank._tc_scratch, bank._x_gpu, bank._x_bytes = {}, None, 0
    bank._runtime = None
    bank.number_of_clauses, bank.number_of_ta_chunks = clauses, 2
    bank.number_of_patches, bank.number_of_state_bits_ta = patches, 8
    bank.number_of_literals, bank._filter, bank.batch_size = 34, 3, 32
    bank.s, bank.boost_true_positive_feedback, bank.max_included_literals = 3.9, 1, 34
    bank._host_newer, bank._host_valid = False, True
    bank._prediction_generation = 0
    bank.rng = np.random.RandomState(81)
    bank.clause_bank = np.zeros(clauses * 2 * 8, np.uint32)
    bank._ta_gpu = driver.mem_alloc(bank.clause_bank.nbytes)
    driver.memcpy_htod(bank._ta_gpu, bank.clause_bank)
    def extract(*args): driver.extracts += 1
    bank._ex_fn = Kernel(extract)
    def gather(source, indices, output, rows, stride):
        ids = driver.read(indices, np.int32, rows)
        data = driver.read(source, np.uint32, (int(ids.max()) + 1) * stride).reshape(-1, stride)
        driver.memcpy_htod(output, data[ids])
    bank._gather_fn = Kernel(gather)
    def match(x, masks, counts, weights, fired, sums, rows, slots, reject, write):
        driver.memcpy_htod(sums, np.zeros((rows, slots), np.int32))
    bank._imma = match
    bank._uses_fused_conv = lambda: True
    bank.conv_seeds = []
    def conv(x, masks, counts, weights, fired, selected, sums, rows, slots, patches, reject, seed, write):
        bank.conv_seeds.append(seed)
        driver.memcpy_htod(sums, np.zeros((rows, slots), np.int32))
    bank._conv_match = conv
    def feedback(*args):
        n, stride = args[10], args[13] * args[28]
        driver.feedback.append(dict(x=driver.read(args[2], np.uint32, n*stride).reshape(n, stride),
            target=driver.read(args[5], np.int32, n), other=driver.read(args[6], np.int32, n), seed=args[26],
            parameters=tuple(args[10:27]) + (args[28],),
            active=driver.read(args[7], np.int32, args[11]*args[12]),
            literals=driver.read(args[8], np.uint32, args[13])))
    bank._fb_fn = Kernel(feedback)
    return bank, driver


def test_encoded_upload_reuses_content_and_invalidates_mutable_alias(monkeypatch):
    bank, driver = make_bank(monkeypatch)
    data = np.arange(20, dtype=np.uint32).reshape(10, 2)
    bank._upload_x(data)
    count = len(driver.uploads)
    bank._upload_x(data.copy())
    assert len(driver.uploads) == count
    data[0, 0] += 1
    bank._upload_x(data)
    assert len(driver.uploads) == count + 1


@pytest.mark.parametrize('patches', [1, 3])
@pytest.mark.parametrize('samples', [1, 33, 69])
def test_epoch_staging_keeps_upstream_rows_batches_and_rng(monkeypatch, patches, samples):
    bank, driver = make_bank(monkeypatch, patches)
    x = np.arange(samples * patches * 2, dtype=np.uint32).reshape(samples, patches * 2)
    order = np.random.RandomState(8).permutation(samples).astype(np.int32)
    target, other = order % 2, 1 - order % 2
    other[::3] = -1
    device = None
    if patches > 1:
        device = driver.mem_alloc(x.nbytes)
        driver.memcpy_htod(device, x)
    rng = np.random.RandomState(81)
    step = 1 if patches > 1 else 32
    expected = []
    for start in range(0, samples, step):
        if patches > 1: expected.append(int(rng.randint(1, 2**31-1)))
        expected.append(int(rng.randint(1, 2**31-1)))
    weights = [np.arange(4, dtype=np.int32), -np.arange(4, dtype=np.int32)]
    bank.run_epoch([0, 1], [bank, bank], weights, x, order, target, other,
                   np.ones((2, 4), np.int32), np.array([0xffffffff, 3], np.uint32),
                   13, 1., 1., True, True, True, device_x=device)
    assert len(driver.feedback) == (samples + step-1)//step
    for index, entry in enumerate(driver.feedback):
        start, stop = index * step, min(samples, (index+1)*step)
        np.testing.assert_array_equal(entry['x'], x[order[start:stop]])
        np.testing.assert_array_equal(entry['target'], target[start:stop])
        np.testing.assert_array_equal(entry['other'], other[start:stop])
        assert entry['seed'] == expected[index * (2 if patches > 1 else 1) + (1 if patches > 1 else 0)]
    assert pickle.dumps(bank.rng.get_state()) == pickle.dumps(rng.get_state())
    assert ('upload', 'epoch_schedule') in bank._tc_scratch


def test_prediction_metadata_reuse_and_training_ta_weight_invalidation(monkeypatch):
    bank, driver = make_bank(monkeypatch)
    x = np.arange(8, dtype=np.uint32).reshape(4, 2)
    weights = [np.arange(4, dtype=np.int32)]
    bank.predict_class_sums([0], [bank], weights, x)
    count, transfers = driver.extracts, len(driver.uploads)
    bank.predict_class_sums([0], [bank], weights, x)
    assert driver.extracts == count
    assert len(driver.uploads) == transfers + 1  # simulated match output only
    weights[0][0] = 19
    bank.predict_class_sums([0], [bank], weights, x)
    assert driver.extracts == count
    assert any(np.array_equal(data, [[19, 1, 2, 3]]) for _, data in driver.uploads)
    bank.set_ta_state(0, 0, 200)
    bank.predict_class_sums([0], [bank], weights, x)
    assert driver.extracts == count + 1
    core._commit_device_epoch([0], [bank], weights, np.array(weights))
    bank.predict_class_sums([0], [bank], weights, x)
    assert driver.extracts == count + 2


def test_flat_device_prediction_does_not_materialize_host_encoded_rows(monkeypatch):
    bank, driver = make_bank(monkeypatch)
    device = driver.mem_alloc(8 * 4)
    driver.memcpy_htod(device, np.arange(8, dtype=np.uint32))
    actual = bank.predict_class_sums([0], [bank], [np.ones(4, np.int32)], None,
                                     device_x=device, n_rows=4)
    assert actual.shape == (4, 1)


@pytest.fixture(scope='module')
def gpu_core():
    try:
        import pycuda.driver as cuda
    except ImportError as error:
        pytest.skip('CUDA driver unavailable: ' + str(error))
    try:
        cuda.init()
        if not cuda.Device.count() or cuda.Device(0).compute_capability() < (8, 0):
            pytest.skip('SM80+ GPU required')
        core._ensure_context()
    except (cuda.Error, RuntimeError) as error:
        pytest.skip('CUDA context unavailable: ' + str(error))
    return core


@pytest.mark.parametrize('shape,patch', [((3, 33), None), ((3, 5, 7, 2), (3, 2)),
                                       ((2, 4, 4, 48), (2, 2))])
def test_gpu_staged_input_matches_upstream_c_encoder_and_mutations(gpu_core, shape, patch):
    from tmu.models.classification.vanilla_classifier import TMClassifier
    x = np.random.RandomState(17).randint(0, 4, shape).astype(np.uint32)
    model = TMClassifier(8, 13, 3.9, platform='tensor', patch_dim=patch, seed=7)
    model.init(x, np.arange(len(x), dtype=np.uint32) % 2)
    bank = model.clause_banks[0]
    for mutation in (False, True):
        if mutation: x[:] = 1
        import tmu.tools
        expected = tmu.tools.encode(x, len(x), bank.number_of_patches,
                                    bank.number_of_ta_chunks, bank.dim, bank.patch_dim, 0)
        actual = np.empty_like(expected)
        pointer = bank.stage_images(x)
        gpu_core.cuda.memcpy_dtoh(actual, pointer)
        np.testing.assert_array_equal(actual, expected)


def test_empty_staging_has_no_zero_grid_launch(monkeypatch):
    bank, driver = make_bank(monkeypatch)
    bank.dim, bank.patch_dim = (17, 1, 1), (17, 1)
    bank._encode_fn = Kernel(lambda *_: (_ for _ in ()).throw(AssertionError('zero-grid launch')))
    pointer = bank.stage_images(np.empty((0, 17), np.uint32))
    assert len(driver.memory(pointer)) >= 4


def test_prediction_cache_does_not_retain_replaced_banks(monkeypatch):
    import weakref
    bank, driver = make_bank(monkeypatch)
    other = core.ClauseBankTensorCore.__new__(core.ClauseBankTensorCore)
    other.__dict__.update(bank.__dict__)
    other._ta_gpu = driver.mem_alloc(bank.clause_bank.nbytes)
    other._tc_scratch = {}
    bank.predict_class_sums([0, 1], [bank, other], [np.ones(4, np.int32)] * 2,
                            np.ones((2, 2), np.uint32))
    reference = weakref.ref(other)
    del other
    assert reference() is None, 'cached state must not retain another bank and its device memory'
    assert '_prediction_identity' not in bank.__getstate__()


def test_bank_replacement_invalidates_masks_even_if_device_pointer_is_reused(monkeypatch):
    bank, driver = make_bank(monkeypatch)
    x, weights = np.ones((2, 2), np.uint32), [np.ones(4, np.int32)]
    bank.predict_class_sums([0], [bank], weights, x)
    before = driver.extracts
    replacement = core.ClauseBankTensorCore.__new__(core.ClauseBankTensorCore)
    replacement.__dict__.update(bank.__dict__)
    replacement.__dict__.pop('_prediction_identity', None)
    replacement._tc_scratch = {}
    bank.predict_class_sums([0], [replacement], weights, x)
    assert driver.extracts == before + 1


@pytest.mark.parametrize('dtype', [np.uint8, np.uint32])
def test_staging_reuses_equal_strided_input_and_reencodes_mutation(monkeypatch, dtype):
    import tmu.tools
    bank, driver = make_bank(monkeypatch)
    bank.dim, bank.patch_dim = (17, 1, 1), (17, 1)
    calls = []
    def encode(source, output, rows, dx, dy, dz, pw, ph, append, classes):
        raw = driver.read(source, np.uint32, rows * dx * dy * dz).reshape(rows, -1)
        driver.memcpy_htod(output, tmu.tools.encode(raw, rows, 1, 2, (dx, dy, dz), (pw, ph), classes))
        calls.append(rows)
    bank._encode_fn = Kernel(encode)
    x = np.arange(3 * 17, dtype=dtype).reshape(3, 17)[:, ::-1]
    original = bank.stage_images(x)
    assert bank.stage_images(x.copy()) is original
    assert calls == [3]
    x[0, 0] = 1
    updated = bank.stage_images(x)
    assert updated is not original and calls == [3, 3]
    expected = bank.prepare_X(x)
    actual = np.empty_like(expected)
    driver.memcpy_dtoh(actual, updated)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('operation', ['fit', 'predict'])
def test_flat_classifier_retains_custom_prepare_x_hook(monkeypatch, operation):
    from tmu.models.classification.vanilla_classifier import TMClassifier
    from tmu.weight_bank import WeightBank
    bank, driver = make_bank(monkeypatch)
    x, encoded = np.ones((3, 17), np.uint32), np.full((3, 2), 19, np.uint32)
    bank.prepare_X = lambda _: encoded
    bank.stage_images = lambda _: (_ for _ in ()).throw(AssertionError('custom encoding bypassed'))
    calls = []
    bank.run_epoch = lambda *args, **kwargs: calls.append((args[3], kwargs['device_x']))
    bank.predict_class_sums = lambda ids, banks, weights, data, **kw: (calls.append((data, kw.get('device_x')))
        or np.zeros((len(x), 2), np.int32))
    class Rows(dict):
        def classes(self): return [0, 1]
        def sample(self, exclude): return 1 - exclude[0]
    model = TMClassifier.__new__(TMClassifier)
    model.clause_banks = {0: bank, 1: bank}
    model.weight_banks = Rows({i: WeightBank(np.ones(4, np.int32)) for i in (0, 1)})
    model.rng, model.T = np.random.RandomState(7), 13
    model.type_i_p, model.type_ii_p = 1., 1.
    model.weighted_clauses, model.type_i_feedback, model.type_ii_feedback = True, True, True
    model.mechanism_clause_active = lambda: np.ones((2, 4), np.int32)
    model.mechanism_literal_active = lambda: np.full(2, 0xffffffff, np.uint32)
    if operation == 'fit': model._fit_tensor(x, np.array([0, 1, 0], np.uint32), False)
    else: model._predict_tensor(x, False, True)
    assert len(calls) == 1 and calls[0][0] is encoded and calls[0][1] is None


def test_failed_mask_replacement_invalidates_previous_prediction_metadata(monkeypatch):
    bank, driver = make_bank(monkeypatch)
    x, weights = np.ones((2, 2), np.uint32), [np.ones(4, np.int32)]
    bank.predict_class_sums([0], [bank], weights, x)
    original = bank._extract
    bank._extract = lambda *_: (_ for _ in ()).throw(RuntimeError('extract failed'))
    with pytest.raises(RuntimeError, match='extract failed'):
        bank.predict_class_sums([1], [bank], weights, x)
    bank._extract = original
    before = driver.extracts
    bank.predict_class_sums([0], [bank], weights, x)
    assert driver.extracts == before + 1
