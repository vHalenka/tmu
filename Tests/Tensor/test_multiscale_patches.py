"""Multiple patch sizes: CPU encodes on CPU, GPU backends encode on GPU."""
import pickle
import numpy as np
import pytest
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier


def images(rows=12):
    x = np.random.RandomState(12).randint(0, 3, (rows, 5, 4, 2)).astype(np.uint32)
    y = (np.arange(rows) % 3).astype(np.uint32)
    return x, y


def expected_encoding(x, sizes):
    from tmu.tools import encode
    w, h = x.shape[1:3]
    depth = x.shape[3] if x.ndim == 4 else 1
    mw, mh = max(p[0] for p in sizes), max(p[1] for p in sizes)
    nx, ny = max(w-p[0] for p in sizes), max(h-p[1] for p in sizes)
    features = nx + ny + mw*mh*(depth+1) + len(sizes)
    candidates = []
    for scale, (pw, ph) in enumerate(sizes):
        count = (w-pw+1)*(h-ph+1)
        local_features = w-pw+h-ph+pw*ph*depth
        chunks = (local_features*2+31)//32
        native = encode(x, len(x), count, chunks, (w,h,depth), (pw,ph), 0).reshape(len(x),count,chunks)
        local = np.unpackbits(native.view(np.uint8),axis=2,bitorder='little')[:,:,:local_features]
        positive = np.zeros((len(x), count, features), np.uint8)
        ly, lx = h-ph, w-pw
        positive[:,:,:ly] = local[:,:,:ly]
        positive[:,:,ny:ny+lx] = local[:,:,ly:ly+lx]
        content = positive[:,:,ny+nx:ny+nx+mw*mh*depth].reshape(len(x),count,mh,mw,depth)
        content[:,:,:ph,:pw] = local[:,:,ly+lx:].reshape(len(x),count,ph,pw,depth)
        valid = positive[:,:,ny+nx+mw*mh*depth:-len(sizes)].reshape(len(x),count,mh,mw)
        valid[:,:,:ph,:pw] = 1
        positive[:,:,-len(sizes)+scale] = 1
        candidates.append(positive)
    positive = np.concatenate(candidates,axis=1)
    bits = np.concatenate([positive,1-positive],axis=2)
    bits = np.pad(bits,((0,0),(0,0),(0,(-bits.shape[2])%32)))
    packed = np.packbits(bits,axis=2,bitorder='little').view('<u4')
    return packed.reshape(len(x),-1),features


def bank_of(model):
    return model.clause_bank if isinstance(model, TMCoalescedClassifier) else model.clause_banks[0]


@pytest.mark.parametrize('kind', [TMClassifier, TMCoalescedClassifier])
def test_cpu_geometry_and_encoding_follow_shared_scale_layout(kind, monkeypatch):
    from tmu.util import patch_encoding
    monkeypatch.setattr(patch_encoding, 'stage_patches', lambda *_: (_ for _ in ()).throw(AssertionError('CPU must not use GPU encoding')))
    x,y = images(3)
    sizes = [(3,2),(2,3)]
    expected,features = expected_encoding(x,sizes)
    model=kind(12,13,3.9,platform='CPU',patch_dim=sizes,seed=7)
    model.init(x,y);bank=bank_of(model)
    assert bank.number_of_features==features
    assert bank.number_of_patches==17
    np.testing.assert_array_equal(bank.prepare_X(x),expected)
    sizes[0]=(1,1)
    assert model.patch_dim==((3,2),(2,3))
    with pytest.raises(ValueError,match='immutable'):
        model.patch_dim=[(1,1),(2,2)]


@pytest.mark.parametrize('kind', [TMClassifier, TMCoalescedClassifier])
def test_cpu_multiscale_training_prediction_and_checkpoint(kind):
    x,y=images()
    model=kind(12,13,3.9,platform='CPU',patch_dim=[(3,2),(2,3)],seed=7)
    model.fit(x,y)
    before=model.predict(x,return_class_sums=True)
    restored=pickle.loads(pickle.dumps(model))
    after=restored.predict(x,return_class_sums=True)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    assert model.transform(x).shape[0]==len(x)


@pytest.mark.parametrize('kind',[TMClassifier,TMCoalescedClassifier])
def test_singleton_patch_list_keeps_legacy_encoding(kind):
    x,y=images(3)
    a=kind(12,13,3.9,patch_dim=(3,2),seed=7);a.init(x,y)
    b=kind(12,13,3.9,patch_dim=[(3,2)],seed=7);b.init(x,y)
    assert b.patch_dim==(3,2)
    np.testing.assert_array_equal(bank_of(a).prepare_X(x),bank_of(b).prepare_X(x))


@pytest.mark.parametrize('sizes',[[],[(0,2),(2,2)],[(2,2),(2,2)],[(1.5,2),(2,2)],[(6,2),(2,2)]])
def test_invalid_sizes_are_rejected(sizes):
    x,y=images(3)
    with pytest.raises(ValueError):TMClassifier(12,13,3.9,patch_dim=sizes,seed=7).init(x,y)


@pytest.fixture(scope='module')
def gpu():
    from tmu.clause_bank import clause_bank_tensor_core as core
    try:core._ensure_context()
    except (ImportError,RuntimeError) as error:pytest.skip(str(error))
    return core


@pytest.mark.parametrize('sizes', [[(3,2),(2,3)],(3,2)])
def test_gpu_patch_encoding_matches_cpu_without_using_cpu_encoder(gpu,sizes,monkeypatch):
    from tmu.util import patch_encoding
    x,y=images(3)
    cpu=TMClassifier(12,13,3.9,patch_dim=sizes,seed=7);cpu.init(x,y)
    expected=bank_of(cpu).prepare_X(x)
    model=TMClassifier(12,13,3.9,platform='tensor',patch_dim=sizes,seed=7);model.init(x,y)
    monkeypatch.setattr(patch_encoding,'encode_cpu',lambda *_:(_ for _ in ()).throw(AssertionError('GPU must not use CPU patch encoder')))
    ptr=bank_of(model).stage_images(x)
    actual=np.empty_like(expected);gpu.cuda.memcpy_dtoh(actual,ptr)
    np.testing.assert_array_equal(actual,expected)




def test_multiscale_coalesced_checkpoint_and_gpu_encoding(gpu,monkeypatch):
    from tmu.util import patch_encoding
    from .test_tensor_coalesced import signature,assert_same
    x,y=images()
    model=TMCoalescedClassifier(12,13,3.9,platform='tensor',patch_dim=[(3,2),(2,3)],seed=7)
    monkeypatch.setattr(patch_encoding,'encode_cpu',lambda *_:(_ for _ in ()).throw(AssertionError('GPU must not use CPU encoding')))
    model.fit(x,y);snapshot=pickle.dumps(model)
    model.fit(x,y);expected=signature(model,x)
    restored=pickle.loads(snapshot);restored.fit(x,y)
    assert_same(signature(restored,x),expected)


def test_vanilla_multiscale_gpu_training_and_restore(gpu,monkeypatch):
    from tmu.util import patch_encoding
    x,y=images()
    model=TMClassifier(12,13,3.9,platform='tensor',patch_dim=[(3,2),(2,3)],seed=7)
    monkeypatch.setattr(patch_encoding,'encode_cpu',lambda *_:(_ for _ in ()).throw(AssertionError('GPU must not use CPU encoding')))
    model.fit(x,y);expected=model.predict(x,return_class_sums=True)
    restored=pickle.loads(pickle.dumps(model))
    for a,b in zip(expected,restored.predict(x,return_class_sums=True)):np.testing.assert_array_equal(a,b)


def test_gpu_patchwise_inspection_matches_cpu(gpu):
    x,y=images(3);sizes=[(3,2),(2,3)]
    cpu=TMCoalescedClassifier(12,13,3.9,patch_dim=sizes,seed=7);cpu.init(x,y)
    model=TMCoalescedClassifier(12,13,3.9,platform='tensor',patch_dim=sizes,seed=7);model.init(x,y)
    for clause,literal in [(0,0),(1,15)]:
        cpu.set_ta_state(clause,literal,255);model.set_ta_state(clause,literal,255)
    np.testing.assert_array_equal(model.transform_patchwise(x),cpu.transform_patchwise(x))


def test_multiscale_staging_routes_before_single_size_arithmetic(monkeypatch):
    from tmu.clause_bank import clause_bank_tensor_core as core
    from tmu.util import patch_encoding
    bank=core.ClauseBankTensorCore.__new__(core.ClauseBankTensorCore)
    bank.__dict__.update(patch_dim=((3,2),(2,3)),dim=(5,4,2),number_of_patches=17,
                         number_of_ta_chunks=3,_enc_cache=None)
    calls=[];output=object()
    monkeypatch.setattr(core,'_ensure_context',lambda:calls.append('context'))
    monkeypatch.setattr(patch_encoding,'stage_patches',lambda *args:(calls.append('encode') or output))
    x,_=images(3)
    assert bank.stage_images(x) is output
    assert bank.stage_images(x.copy()) is output
    assert calls==['context','encode']




@pytest.mark.parametrize("sizes", [(3,2), [(3,2),(2,3)]])
@pytest.mark.parametrize("fused", [False, True])
def test_gpu_convolution_prediction_preserves_bank_rng(gpu,monkeypatch,sizes,fused):
    x,y=images()
    model=TMCoalescedClassifier(12,13,3.9,platform="tensor",patch_dim=sizes,seed=7)
    model.fit(x,y)
    bank=model.clause_bank
    monkeypatch.setattr(bank,"_uses_fused_conv",lambda:fused)
    before=pickle.dumps(bank.rng.get_state())
    first=model.predict(x,return_class_sums=True)
    second=model.predict(x,return_class_sums=True)
    assert pickle.dumps(bank.rng.get_state())==before
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
