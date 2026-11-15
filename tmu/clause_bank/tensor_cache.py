"""Content-validated caches for existing Tensor execution buffers."""
import numpy as np
import xxhash


def array_key(values):
    values = np.ascontiguousarray(values)
    return values.shape, values.dtype.str, xxhash.xxh3_64_digest(values)


def cached_upload(owner, name, values, driver):
    """Reuse a device buffer only while its exact host contents are unchanged."""
    values = np.ascontiguousarray(values)
    pointer = owner._tc_buf(name, values.nbytes)
    key = ('upload', name)
    cached = owner._tc_scratch.get(key)
    if (cached is None or cached[0] is not pointer
            or cached[1].shape != values.shape or cached[1].dtype != values.dtype
            or not np.array_equal(cached[1], values)):
        owner._tc_scratch.pop(key, None)
        driver.memcpy_htod(pointer, values)
        owner._tc_scratch[key] = pointer, values.copy()
    return pointer


def can_stage_flat(bank):
    """Keep custom flat encoders on the upstream prepare_X call path."""
    from tmu.clause_bank.clause_bank_tensor_core import ClauseBankTensorCore
    return (type(bank) is ClauseBankTensorCore
            and getattr(bank.prepare_X, '__func__', None) is ClauseBankTensorCore._prepare_X_default
            and getattr(bank.stage_images, '__func__', None) is ClauseBankTensorCore._stage_images_default)
