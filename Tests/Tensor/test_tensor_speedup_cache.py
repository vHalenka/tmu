"""Exact input and upload cache behavior without a CUDA context."""
import numpy as np
import pytest
from tmu.clause_bank.tensor_cache import array_key, cached_upload


class Driver:
    def __init__(self):
        self.uploads = []
    def memcpy_htod(self, pointer, values):
        self.uploads.append((pointer, values.copy()))


class Owner:
    def __init__(self):
        self._tc_scratch = {}
        self.buffers = {}
    def _tc_buf(self, name, size):
        pointer, capacity = self.buffers.get(name, (None, 0))
        if pointer is None or size > capacity:
            pointer, capacity = object(), size
            self.buffers[name] = pointer, capacity
        return pointer


@pytest.mark.parametrize('dtype', [np.uint32, np.int32, np.uint8])
def test_upload_cache_reuses_unchanged_values_and_detects_alias_mutation(dtype):
    owner, driver = Owner(), Driver()
    values = np.arange(15, dtype=dtype).reshape(3, 5)
    first = cached_upload(owner, 'input', values, driver)
    assert cached_upload(owner, 'input', values.copy(), driver) is first
    assert len(driver.uploads) == 1
    values[1, 2] += 1
    assert cached_upload(owner, 'input', values, driver) is first
    assert len(driver.uploads) == 2
    np.testing.assert_array_equal(driver.uploads[-1][1], values)


def test_upload_cache_tracks_shape_dtype_capacity_and_strided_input():
    owner, driver = Owner(), Driver()
    source = np.arange(16, dtype=np.uint32).reshape(4, 4)
    cached_upload(owner, 'input', source[:, ::2], driver)
    assert driver.uploads[-1][1].flags.c_contiguous
    cached_upload(owner, 'input', source[:, ::2].reshape(8), driver)
    cached_upload(owner, 'input', source[:, ::2].astype(np.int32), driver)
    old = cached_upload(owner, 'input', source, driver)
    new = cached_upload(owner, 'input', np.arange(100, dtype=np.uint32), driver)
    assert old is not new
    assert len(driver.uploads) == 5


def test_content_key_detects_mutation_shape_and_dtype_without_pointer_identity():
    values = np.arange(15, dtype=np.uint32).reshape(3, 5)
    key = array_key(values)
    assert array_key(values.copy()) == key
    assert array_key(values[:, ::-1][:, ::-1]) == key
    assert array_key(values.reshape(5, 3)) != key
    assert array_key(values.astype(np.int32)) != key
    values[0, 0] += 1
    assert array_key(values) != key


def test_failed_upload_is_not_cached():
    owner, driver = Owner(), Driver()
    original = driver.memcpy_htod
    driver.memcpy_htod = lambda *_: (_ for _ in ()).throw(RuntimeError('copy failed'))
    with pytest.raises(RuntimeError, match='copy failed'):
        cached_upload(owner, 'input', np.ones(3, np.uint32), driver)
    driver.memcpy_htod = original
    cached_upload(owner, 'input', np.ones(3, np.uint32), driver)
    assert len(driver.uploads) == 1


def test_failed_replacement_invalidates_the_previous_upload_snapshot():
    owner, driver = Owner(), Driver()
    old = np.ones(3, np.uint32)
    cached_upload(owner, 'input', old, driver)
    original = driver.memcpy_htod
    driver.memcpy_htod = lambda *_: (_ for _ in ()).throw(RuntimeError('copy failed'))
    with pytest.raises(RuntimeError, match='copy failed'):
        cached_upload(owner, 'input', old + 1, driver)
    driver.memcpy_htod = original
    cached_upload(owner, 'input', old, driver)
    assert len(driver.uploads) == 2
