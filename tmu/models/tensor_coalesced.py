"""Coalesced GPU training. Parallel feedback can learn a different model."""
from pathlib import Path
import numpy as np


def parallel_seed_states(seed, clauses):
    """Avalanche clause identities, then select the maximal-period MCG class.

    Used only for new models. Saved per-clause states continue unchanged.
    SplitMix64 is a seed mixer; feedback still uses the existing PCG sampler.
    """
    mask = (1 << 64) - 1
    states = []
    for clause in range(clauses):
        value = (int(seed) + (clause + 1) * 0x9e3779b97f4a7c15) & mask
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
        value ^= value >> 31
        states.append(value | 3)
    return np.asarray(states, dtype=np.uint64)


def sample_schedule(order, labels, classes, balanced):
    """Keep upstream's complete class-sized groups, including its tail rule."""
    order = np.asarray(order, dtype=np.int32)
    if not balanced:
        return np.ascontiguousarray(order[:len(order) // classes * classes])
    pending = np.empty(classes, dtype=np.int32)
    seen = np.zeros(classes, dtype=bool)
    result = []
    for sample in order:
        label = labels[sample]
        if not seen[label]:
            pending[label] = sample
            seen[label] = True
        if seen.all():
            result.extend(pending)
            seen.fill(False)
    return np.asarray(result, dtype=np.int32)


def input_dimensions(X):
    if np.ndim(X)==2:return (X.shape[1],1,1)
    if np.ndim(X)==3:return (X.shape[1],X.shape[2],1)
    if np.ndim(X)==4:return tuple(X.shape[1:])
    raise ValueError('X must be a feature matrix or spatial image array')


def validate_training(model, X, Y):
    if getattr(model, 'feedback_mode', 'parallel') != 'parallel':
        raise ValueError('Tensor Coalesced training only supports parallel feedback')
    if model.type_iii_feedback:
        raise NotImplementedError('Coalesced Tensor training does not support Type III feedback')
    if model.patch_dim is None and np.ndim(X) != 2:
        raise ValueError('Spatial input requires patch_dim')
    if model.patch_dim is not None and np.ndim(X) not in (3, 4):
        raise ValueError('patch_dim requires spatial image input')
    if model.seed == 0:
        raise ValueError('Coalesced Tensor training requires a nonzero seed or seed=None')
    if not np.isfinite(model.s) or float(model.s) <= 0:
        raise ValueError('s must be finite and greater than zero')
    if model.s <= 1 and model.boost_true_positive_feedback != 1:
        raise ValueError('s <= 1 requires boost_true_positive_feedback=1')
    if not isinstance(model.T, (int, np.integer)) or not 0 < model.T <= np.iinfo(np.int32).max:
        raise ValueError('T must be a positive int32 value')
    if np.ndim(Y) != 1 or len(X) != len(Y):
        raise ValueError('X and Y must have the same number of rows and Y must be 1D')
    labels = np.asarray(Y)
    if labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError('Y must contain nonnegative integer class labels')
    if not model.initialized and not len(X):
        raise ValueError('At least one training row is required to initialize the model')
    if model.initialized:
        if input_dimensions(X) != tuple(model.clause_bank.dim):
            raise ValueError('The input dimensions cannot change after initialization')
        if len(Y) and int(labels.max()) >= model.number_of_classes:
            raise ValueError('Y contains a class outside the initialized model')


class CoalescedGPU:
    """Own training buffers; public weights are refreshed at each fit boundary."""
    def __init__(self, model):
        from tmu.clause_bank import clause_bank_tensor_core as core
        from tmu.clause_bank.clause_bank_cuda import load_cuda_kernel
        self.bank = model.clause_bank
        self.cuda = core.cuda
        directory = Path(core.__file__).parent / 'cuda'
        module = load_cuda_kernel([(directory / 'cpu_random.cuh').read_text()], 'cuda/coalesced.cu')
        self.function = module.get_function('tc_coalesced_train')
        self.function.prepare('P' * 13 + 'i' * 12 + 'I' + 'f' + 'd' * 2)
        self.module = module
        self.buffers = {}
        self.weights = None
        self.weight_snapshot = None
        self.blocks = self.maximum_blocks = 1
        self.cooperative = None
        from tmu.clause_bank.cuda_cooperative import driver, CooperativeUnavailable, DriverError
        try:
            supported = driver().supported()
        except (CooperativeUnavailable, DriverError):
            supported = False  # Capability discovery only; nothing has launched.
        if supported:
            from tmu.clause_bank.coalesced_compile import load_cooperative_kernel
            source = '\n'.join((directory / name).read_text() for name in
                               ['cpu_random.cuh', 'coalesced.cu', 'coalesced_parallel.cu'])
            # Geometry is fixed for a clause bank; mutable learning parameters
            # remain runtime arguments. Cache identity includes these defines.
            self.compiled_geometry = (self.bank.number_of_ta_chunks, self.bank.number_of_literals,
                self.bank.number_of_patches, self.bank.number_of_state_bits_ta)
            defines = dict(TC_STATIC_CHUNKS=self.compiled_geometry[0],
                TC_STATIC_LITERALS=self.compiled_geometry[1], TC_STATIC_PATCHES=self.compiled_geometry[2],
                TC_STATIC_BITS=self.compiled_geometry[3],
                TC_MATCH_WORDS=max(1,min(256,(self.bank.number_of_patches+31)//32)),
                TC_MASK_WORDS=max(1,min(256,self.bank.number_of_ta_chunks)))
            source = ''.join('#define %s %d\n' % item for item in defines.items()) + source
            self.cooperative = load_cooperative_kernel(source, 'tc_coalesced_parallel_train',
                                                 'P' * 15 + 'i' * 12 + 'I' + 'f' + 'd' * 2)
            sms = core._CTX.get_device().get_attribute(core.cuda.device_attribute.MULTIPROCESSOR_COUNT)
            self.maximum_blocks = max(1, self.cooperative.active_blocks(128) * sms)
            self.blocks = min((self.bank.number_of_clauses + 3) // 4, self.maximum_blocks)

    def buffer(self, name, values):
        values = np.ascontiguousarray(values)
        pointer, capacity = self.buffers.get(name, (None, 0))
        if pointer is None or capacity < values.nbytes:
            pointer = self.cuda.mem_alloc(max(4, values.nbytes))
            self.buffers[name] = pointer, values.nbytes
        if values.nbytes:
            self.cuda.memcpy_htod(pointer, values)
        return pointer

    def train(self, model, X, schedule, labels):
        bank, cuda = self.bank, self.cuda
        geometry = (bank.number_of_ta_chunks,bank.number_of_literals,bank.number_of_patches,bank.number_of_state_bits_ta)
        if self.cooperative is not None and geometry != self.compiled_geometry:
            raise ValueError('Clause-bank geometry changed after cooperative kernel initialization')
        classes, clauses = model.number_of_classes, bank.number_of_clauses
        weights = np.stack([model.weight_banks[c].get_weights() for c in range(classes)])
        if self.weight_snapshot is None or not np.array_equal(weights, self.weight_snapshot):
            self.weights = self.buffer('weights', weights)
        active = self.buffer('active', model.clause_active)
        literal = self.buffer('literal', model.literal_active)
        votes = self.buffer('votes', np.zeros(clauses, np.uint8))
        draws_host = np.zeros(classes, np.float64)
        draws = self.buffer('draws', draws_host)
        decisions = self.buffer('decisions', np.zeros(clauses, np.uint8))
        masks = self.buffer('feedback_masks', np.zeros((clauses, bank.number_of_ta_chunks), np.uint32))
        seed = 0xcafef00dd15ea5e5 if model.seed is None else int(model.seed)
        # Keep this inert checkpoint field for existing parallel checkpoints.
        model._coalesced_pcg_state = getattr(model, '_coalesced_pcg_state', seed)
        parallel_host = getattr(model, '_coalesced_parallel_state', None)
        if parallel_host is None:
            parallel_host = parallel_seed_states(seed, clauses)
        parallel_host = parallel_host.copy()
        parallel = self.buffer('parallel', parallel_host)
        mt = model.rng.get_state()
        keys_host = mt[1].copy()
        position_host = np.array([mt[2]], np.int32)
        keys = self.buffer('mt_keys', keys_host)
        position = self.buffer('mt_position', position_host)
        indices = self.buffer('indices', schedule)
        targets = self.buffer('targets', labels[schedule].astype(np.int32))
        use_cooperative = self.blocks > 1
        control = self.buffer('cooperative_control', np.zeros(16, np.uint32)) if use_cooperative else None
        bank._push()
        streamed = getattr(model, "pack_batches", False)
        encoded = None if streamed else bank.stage_images(X)
        self.weight_snapshot = None  # Device training changes these values.
        batch = int(model.batch_size)
        for start in range(0, len(schedule), batch):
            count = min(batch, len(schedule) - start)
            if streamed:
                # Preserve the epoch schedule and GPU RNG state across launches.
                # Reindex only the packed rows; targets remain in schedule order.
                encoded = bank.stage_images(X[schedule[start:start + count]])
                batch_indices = self.buffer("batch_indices", np.arange(count, dtype=np.int32))
            else:
                batch_indices = int(indices) + start * 4
            bank._host_valid, bank._host_newer = False, False
            bank._prediction_generation += 1
            arguments = (
                bank._ta_gpu, encoded, self.weights,
                active, literal, batch_indices, int(targets) + start * 4,
                votes, draws, masks, parallel, keys, position,
                count, clauses, bank.number_of_ta_chunks, bank.number_of_literals,
                classes, bank.number_of_patches, int(model.T), int(model.max_positive_clauses),
                bank.number_of_state_bits_ta, bank.boost_true_positive_feedback,
                bank.max_included_literals, int(model.focused_negative_sampling),
                int(bank._filter), float(bank.s), float(model.type_i_p), float(model.type_ii_p))
            if use_cooperative:
                # The entire grid is resident; a launch error must propagate.
                self.cooperative.launch((self.blocks, 1, 1), (128, 1, 1),
                                        arguments[:9] + (decisions,) + arguments[9:13] + (control,) + arguments[13:])
            else:
                self.function.prepared_call((1, 1, 1), (128, 1, 1), *arguments)
            if streamed:
                # Complete consumers before releasing their input allocation.
                cuda.Context.synchronize()
                bank._enc_cache = None
                encoded = None
        # These copies finish outstanding work before local staging can be freed.
        cuda.memcpy_dtoh(weights, self.weights)
        cuda.memcpy_dtoh(parallel_host, parallel)
        cuda.memcpy_dtoh(keys_host, keys)
        cuda.memcpy_dtoh(position_host, position)
        cuda.memcpy_dtoh(draws_host, draws)
        for c in range(classes):
            model.weight_banks[c].get_weights()[:] = weights[c]
        self.weight_snapshot = weights.copy()
        model._coalesced_parallel_state = parallel_host
        model.rng.set_state((mt[0], keys_host, int(position_host[0]), mt[3], mt[4]))
        model.update_ps = draws_host


def fit(model, X, Y, shuffle):
    bank = model.clause_bank
    model.clause_active = (model.rng.rand(model.number_of_clauses) >= model.clause_drop_p).astype(np.int32)
    keep = model.rng.rand(bank.number_of_literals) >= model.literal_drop_p
    if not model.feature_negation:
        keep[bank.number_of_literals // 2:] = False
    model.literal_active = np.zeros(bank.number_of_ta_chunks, dtype='<u4')
    packed = np.packbits(keep, bitorder='little')
    model.literal_active.view(np.uint8)[:len(packed)] = packed
    order = np.arange(len(X), dtype=np.int32)
    if shuffle:
        model.rng.shuffle(order)
    labels = np.ascontiguousarray(Y, dtype=np.uint32)
    schedule = sample_schedule(order, labels, model.number_of_classes, model.output_balancing)
    model.update_ps = np.zeros(model.number_of_classes, np.float64)
    if not len(schedule):
        return
    trainer = getattr(model, '_coalesced_gpu', None)
    if trainer is None or trainer.bank is not bank:
        trainer = model._coalesced_gpu = CoalescedGPU(model)
    trainer.train(model, X, schedule, labels)


def shared_class_sums(model, X):
    """Match shared clauses once, then reduce all class weights on the GPU."""
    from tmu.clause_bank.clause_bank_cuda import load_cuda_kernel
    from tmu.clause_bank import clause_bank_tensor_core as core
    bank=model.clause_bank
    n=len(X);classes=model.number_of_classes
    result=np.empty((n,classes),np.int32)
    if not n:return result
    bank._push()
    kernels=getattr(bank,'_coalesced_predict_kernels',None)
    if kernels is None:
        module=load_cuda_kernel([], 'cuda/coalesced_predict.cu')
        votes=module.get_function('tc_shared_votes');votes.prepare('PPPiiiiI')
        sums=module.get_function('tc_shared_sums');sums.prepare('PPPiiii')
        kernels=bank._coalesced_predict_kernels=(module,votes,sums)
    encoded=bank.stage_images(X)
    weights=np.stack([model.weight_banks[c].get_weights() for c in range(classes)])
    wp=bank._tc_buf('shared_predict_weights',weights.nbytes)
    core.cuda.memcpy_htod(wp,weights)
    fused=bank._uses_fused_conv()
    if fused:
        ptr=bank._tc_buf('shared_predict_ptr',8)
        lit=bank._tc_buf('shared_predict_literal',bank.number_of_ta_chunks*4)
        core.cuda.memcpy_htod(ptr,np.array([int(bank._ta_gpu)],np.uint64))
        core.cuda.memcpy_htod(lit,np.full(bank.number_of_ta_chunks,0xffffffff,np.uint32))
        masks=bank._tc_buf('shared_predict_masks',bank._mask_bytes(1))
        counts=bank._tc_buf('shared_predict_counts',bank.number_of_clauses*4)
        included=bank._tc_buf('shared_predict_included',bank.number_of_clauses*4)
        generation=getattr(bank,'_prediction_generation',0)
        if bank._tc_scratch.get('shared_predict_generation') != generation:
            bank._extract(ptr,lit,masks,counts,included,1,0)
            bank._tc_scratch['shared_predict_generation']=generation
    # Bound intermediate votes and the CUDA grid's y dimension.
    limit=min(65535,max(1,(8*1024*1024)//bank.number_of_clauses))
    for start in range(0,n,limit):
        count=min(limit,n-start)
        vp=bank._tc_buf('shared_predict_votes',count*bank.number_of_clauses)
        sp=bank._tc_buf('shared_predict_sums',count*classes*4)
        xp=int(encoded)+start*bank.number_of_patches*bank.number_of_ta_chunks*4
        if fused:
            patches=bank._tc_buf('shared_predict_patches',count*bank.number_of_clauses*4)
            unused=bank._tc_buf('shared_predict_unused_sums',count*4)
            bank._conv_match(xp,masks,counts,wp,vp,patches,unused,count,1,bank.number_of_patches,1,1,1)
        else:
            kernels[1].prepared_call(((bank.number_of_clauses+3)//4,count,1),(128,1,1),
                bank._ta_gpu,xp,vp,bank.number_of_clauses,bank.number_of_ta_chunks,
                bank.number_of_state_bits_ta,bank.number_of_patches,int(bank._filter))
        kernels[2].prepared_call((count,classes,1),(128,1,1),vp,wp,sp,bank.number_of_clauses,classes,count,int(fused))
        core.cuda.memcpy_dtoh(result[start:start+count],sp)
    return result


def predict(model, X, clip, return_sums):
    if not model.initialized:
        raise RuntimeError('Call fit before predicting')
    if input_dimensions(X) != tuple(model.clause_bank.dim):
        raise ValueError('X must have the initialized input dimensions')
    classes = list(range(model.number_of_classes))
    if not len(X):
        sums = np.empty((0, len(classes)), np.int32)
    elif model.clause_bank.number_of_patches > 1 and model.number_of_classes <= 65535:
        sums = shared_class_sums(model, X)
    else:
        bank = model.clause_bank
        sums = bank.predict_class_sums(classes, [bank] * len(classes),
                                      [model.weight_banks[c].get_weights() for c in classes],
                                      None, device_x=bank.stage_images(X), n_rows=len(X), preserve_rng=True)
    if clip:
        sums = np.clip(sums, -model.T, model.T)
    labels = np.argmax(sums, axis=1)
    return (labels, sums) if return_sums else labels
