"""Whole-image and local-window horizontal/vertical reflection on CPU/GPU."""
import pickle
import numpy as np
import pytest
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.util import patch_encoding as encoding

MODES=('none','horizontal','vertical','both')
OPTIONS=[(a,b) for a in MODES for b in MODES if (a,b)!=('none','none')]


def orientations(value,mode):
    result=[value]
    if mode in ('horizontal','both'):result.append(value[:,::-1])
    if mode in ('vertical','both'):result.append(value[::-1,:])
    if mode=='both':result.append(value[::-1,::-1])
    return result


def data(rows=12):
    x=np.random.RandomState(111).randint(0,3,(rows,5,4,2)).astype(np.uint32)
    return x,(np.arange(rows)%3).astype(np.uint32)


def oracle(x,sizes,image,patch):
    raw=x.copy()
    if image!='none':
        for i in range(len(raw)):
            raw[i]=min(orientations(raw[i],image),key=lambda a:(a==1).ravel().tolist())
    # Existing independent multiscale oracle supplies geometry and non-content bits.
    from .test_multiscale_patches import expected_encoding
    sizes=[sizes] if isinstance(sizes[0],int) else sizes
    if len(sizes)==1:
        from tmu.tools import encode
        w,h,d=raw.shape[1:];pw,ph=sizes[0];features=(w-pw)+(h-ph)+pw*ph*d
        packed=encode(raw,len(raw),(w-pw+1)*(h-ph+1),(features*2+31)//32,(w,h,d),(pw,ph),0)
    else:packed,features=expected_encoding(raw,sizes)
    chunks=(features*2+31)//32;bits=np.unpackbits(packed.reshape(len(raw),-1,chunks).view(np.uint8),axis=2,bitorder='little')
    if patch!='none':
        w,h,d=raw.shape[1:];mw=max(p[0] for p in sizes);mh=max(p[1] for p in sizes)
        start=max(h-ph for pw,ph in sizes)+max(w-pw for pw,ph in sizes)
        offset=0
        for pw,ph in sizes:
            count=(w-pw+1)*(h-ph+1)
            for i in range(len(raw)):
                for j in range(offset,offset+count):
                    canvas=bits[i,j,start:start+mw*mh*d].reshape(mh,mw,d)
                    original=canvas[:ph,:pw].copy()
                    canvas[:ph,:pw]=min(orientations(original,patch),key=lambda a:a.ravel().tolist())
            offset+=count
        bits[:,:,features:features*2]=1-bits[:,:,:features]
    return np.packbits(bits,axis=2,bitorder='little').view('<u4').reshape(len(raw),-1)


@pytest.mark.parametrize('sizes',[(3,2),[(3,2),(2,3)]])
@pytest.mark.parametrize('image,patch',OPTIONS)
def test_cpu_encoding_matches_independent_oracle(sizes,image,patch,monkeypatch):
    x,y=data(4)
    monkeypatch.setattr(encoding,'stage_patches',lambda *_a,**_k:pytest.fail('CPU called GPU encoder'))
    model=TMClassifier(12,13,3.9,patch_dim=sizes,image_invariance=image,patch_invariance=patch,seed=7)
    model.init(x,y);bank=model.clause_banks[0]
    np.testing.assert_array_equal(bank.prepare_X(x),oracle(x,sizes,image,patch))
    assert bank.prepare_X(x[:0]).shape==(0,bank.number_of_patches*bank.number_of_ta_chunks)
    for view in orientations(x.transpose(1,2,0,3),image):
        np.testing.assert_array_equal(bank.prepare_X(x),bank.prepare_X(view.transpose(2,0,1,3)))


@pytest.mark.parametrize('image',['horizontal','vertical','both'])
@pytest.mark.parametrize('kind',[TMClassifier,TMCoalescedClassifier])
@pytest.mark.parametrize('sizes',[(3,2),[(3,2),(2,3)]])
def test_cpu_image_invariant_scores_checkpoint_and_options(kind,sizes,image):
    x,y=data();model=kind(12,13,3.9,seed=7,patch_dim=sizes,image_invariance=image,patch_invariance='both')
    model.fit(x,y);scores=model.predict(x,return_class_sums=True)
    views=[v.transpose(2,0,1,3) for v in orientations(x.transpose(1,2,0,3),image)]
    results=[model.predict(v,return_class_sums=True) for v in views]
    results.append(pickle.loads(pickle.dumps(model)).predict(x,return_class_sums=True))
    for got in results:
        for a,b in zip(scores,got):np.testing.assert_array_equal(a,b)
    bank=model.clause_bank if kind is TMCoalescedClassifier else model.clause_banks[0]
    for obj in [model,bank]:
        with pytest.raises(ValueError,match='immutable'):obj.image_invariance='none'
        with pytest.raises(ValueError,match='immutable'):obj.patch_invariance='none'


@pytest.mark.parametrize('options',[{'image_invariance':'diagonal'},{'patch_invariance':True},{'image_invariance':'horizontal'},{'patch_invariance':'horizontal'}])
def test_invalid_options_and_nonspatial_inputs_are_rejected(options):
    with pytest.raises(ValueError):TMClassifier(12,13,3.9,**options).init(np.zeros((3,5),np.uint32),np.arange(3,dtype=np.uint32))


@pytest.fixture(scope='module')
def gpu():
    try:
        from tmu.clause_bank import clause_bank_tensor_core as core
        core._ensure_context()
    except (ImportError,RuntimeError) as error:pytest.skip(str(error))
    return core


@pytest.mark.parametrize('sizes',[(3,2),[(3,2),(2,3)]])
@pytest.mark.parametrize('image,patch',OPTIONS)
def test_gpu_encoding_matches_oracle_without_cpu_encoding(gpu,sizes,image,patch,monkeypatch):
    x,y=data(4);expected=oracle(x,sizes,image,patch)
    model=TMClassifier(12,13,3.9,platform='tensor',seed=7,patch_dim=sizes,image_invariance=image,patch_invariance=patch)
    model.init(x,y);bank=model.clause_banks[0]
    monkeypatch.setattr(encoding,'encode_cpu',lambda *_a,**_k:pytest.fail('GPU called CPU encoder'))
    np.testing.assert_array_equal(bank.prepare_X(x),expected)
    if image=='horizontal':np.testing.assert_array_equal(bank.prepare_X(x[:,:,::-1]),expected)


@pytest.mark.parametrize('image',['horizontal','vertical','both'])
@pytest.mark.parametrize('kind',[TMClassifier,TMCoalescedClassifier])
def test_gpu_image_invariance_and_checkpoint(gpu,kind,image):
    x,y=data()
    model=kind(12,13,3.9,platform='tensor',seed=7,patch_dim=[(3,2),(2,3)],image_invariance=image,patch_invariance='both')
    model.fit(x,y);before=model.predict(x,return_class_sums=True)
    for view in orientations(x.transpose(1,2,0,3),image):
        for a,b in zip(before,model.predict(view.transpose(2,0,1,3),return_class_sums=True)):np.testing.assert_array_equal(a,b)
    # Vanilla prediction advances bank RNG; checkpoint after the prediction calls.
    saved=pickle.dumps(model)
    model.fit(x,y);expected=model.predict(x,return_class_sums=True)
    restored=pickle.loads(saved);restored.fit(x,y)
    for a,b in zip(expected,restored.predict(x,return_class_sums=True)):np.testing.assert_array_equal(a,b)


def test_coalesced_batches_above_64_preserve_learning(gpu):
    x=np.random.RandomState(22).randint(0,2,(132,9)).astype(np.uint32);y=(np.arange(132)%3).astype(np.uint32)
    models=[TMCoalescedClassifier(12,13,3.9,platform='tensor',seed=7,batch_size=b) for b in [1,100,256]]
    signatures=[]
    for model in models:
        model.fit(x,y);model.clause_bank._pull()
        signatures.append((model.clause_bank.clause_bank.copy(),np.stack([model.weight_banks[c].get_weights() for c in range(3)]),model._coalesced_parallel_state.copy(),pickle.dumps(model.rng.get_state())))
    for got in signatures[1:]:
        for a,b in zip(signatures[0][:3],got[:3]):np.testing.assert_array_equal(a,b)
        assert signatures[0][3]==got[3]
