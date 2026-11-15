"""Independent integer score oracle and cooperative GPU equivalence."""
from pathlib import Path
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

def test_parallel_scores_match_independent_signed_wrap_oracle(gpu):
    from tmu.clause_bank import clause_bank_tensor_core as core
    from pycuda.compiler import SourceModule
    core._ensure_context();folder=Path(core.__file__).parent/'cuda'
    source=(folder/'cpu_random.cuh').read_text()+(folder/'coalesced.cu').read_text()+r'''
extern "C" __global__ void probe_scores(const int *weights,const unsigned char *votes,const int *active,int clauses,int classes,int target,int T,double *draws,int *stats){
    tc_coalesced_scores(weights,votes,active,clauses,classes,target,T,draws,stats,stats+1);
}
'''
    module=SourceModule(source,no_extern_c=True);fn=module.get_function('probe_scores');rng=np.random.RandomState(18)
    for clauses in [1,31,64,129]:
        for classes in [1,3,16,35]:
            weights=rng.randint(-2147483648,2147483647,(classes,clauses),dtype=np.int32)
            votes=rng.randint(0,2,clauses).astype(np.uint8);active=rng.randint(0,2,clauses).astype(np.int32)
            for target in [0,classes-1]:
                scores=[]
                for row in weights:
                    u=sum(int(w) for w,v,a in zip(row,votes,active) if v and a)&0xffffffff
                    scores.append(u if u<0x80000000 else u-0x100000000)
                want=np.array([0 if c==target else (50+max(-50,min(50,s)))/100 for c,s in enumerate(scores)],np.float64)
                draws=np.empty(classes,np.float64);stats=np.empty(2,np.int32)
                fn(core.cuda.In(weights),core.cuda.In(votes),core.cuda.In(active),np.int32(clauses),np.int32(classes),np.int32(target),np.int32(50),core.cuda.Out(draws),core.cuda.Out(stats),block=(128,1,1),grid=(1,1))
                np.testing.assert_array_equal(draws,want)
                assert stats[0]==max(-50,min(50,scores[target]))
                assert stats[1]==np.count_nonzero(weights[target]>=0)

@pytest.mark.parametrize('patches',[None,[(3,2),(2,3)]])
@pytest.mark.parametrize('clauses',[33,129])
def test_cooperative_parallel_matches_single_block_and_checkpoint(gpu,patches,clauses):
    import pickle
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    from tmu.models.tensor_coalesced import CoalescedGPU
    rng=np.random.RandomState(6)
    shape=(17,) if patches is None else (5,4,2)
    x=rng.randint(0,2,(18,*shape)).astype(np.uint32);y=(np.arange(len(x))%3).astype(np.uint32)
    models=[]
    for blocks in [1,4]:
        model=TMCoalescedClassifier(clauses,17,3.9,seed=31,platform='tensor',patch_dim=patches,clause_drop_p=0.1,literal_drop_p=0.1,focused_negative_sampling=clauses==129,output_balancing=patches is not None,type_i_ii_ratio=0.37,max_positive_clauses=clauses//2,max_included_literals=8)
        model.init(x,y);trainer=model._coalesced_gpu=CoalescedGPU(model)
        if trainer.maximum_blocks<4:pytest.skip("Four-block cooperative CUDA launch unavailable")
        trainer.blocks=blocks
        models.append(model)
    def fingerprint(m):
        m.clause_bank._pull()
        return (m.clause_bank.clause_bank.copy(),np.stack([m.weight_banks[c].get_weights() for c in range(3)]),m.predict(x,return_class_sums=True),m._coalesced_parallel_state.copy(),m._coalesced_pcg_state,pickle.dumps(m.rng.get_state()),pickle.dumps(m.clause_bank.rng.get_state()),m.update_ps.copy())
    def equal(a,b):
        for i in [0,1,3,7]:np.testing.assert_array_equal(a[i],b[i])
        for x,y in zip(a[2],b[2]):np.testing.assert_array_equal(x,y)
        assert a[4:7]==b[4:7]
    for _ in range(3):
        for model in models:model.fit(x,y)
        equal(fingerprint(models[0]),fingerprint(models[1]))
    saved=pickle.dumps(models[1]);models[1].fit(x,y);expected=fingerprint(models[1]);restored=pickle.loads(saved);restored.fit(x,y);equal(fingerprint(restored),expected)




@pytest.mark.parametrize('shape,patch', [((4097,),None),((92,92),(1,1))])
def test_cooperative_wide_or_many_patch_fallback_matches_single_block(gpu,shape,patch):
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    from tmu.models.tensor_coalesced import CoalescedGPU
    rng=np.random.RandomState(7);x=rng.randint(0,2,(6,*shape)).astype(np.uint32);y=(np.arange(6)%3).astype(np.uint32)
    models=[]
    for blocks in [1,4]:
        m=TMCoalescedClassifier(5,17,100.0,platform='tensor',patch_dim=patch,seed=7,number_of_state_bits_ta=16)
        m.init(x,y);trainer=m._coalesced_gpu=CoalescedGPU(m)
        if trainer.maximum_blocks<4:pytest.skip('Four-block cooperative CUDA launch unavailable')
        trainer.blocks=blocks;m.fit(x,y);m.clause_bank._pull();models.append(m)
    a,b=models
    np.testing.assert_array_equal(a.clause_bank.clause_bank,b.clause_bank.clause_bank)
    np.testing.assert_array_equal(a._coalesced_parallel_state,b._coalesced_parallel_state)
    for c in range(3):np.testing.assert_array_equal(a.weight_banks[c].get_weights(),b.weight_banks[c].get_weights())
    np.testing.assert_array_equal(a.predict(x),b.predict(x))

def test_cooperative_single_class_keeps_native_draws_and_zero_negative_probabilities(gpu):
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    from tmu.models.tensor_coalesced import CoalescedGPU
    x=np.random.RandomState(19).randint(0,2,(6,33)).astype(np.uint32);y=np.zeros(6,np.uint32);models=[]
    for blocks in [1,2]:
        m=TMCoalescedClassifier(5,17,1.0,platform='tensor',seed=7,focused_negative_sampling=True,number_of_state_bits_ta=1)
        m.init(x,y);trainer=m._coalesced_gpu=CoalescedGPU(m)
        if trainer.maximum_blocks<2:pytest.skip('Two-block cooperative CUDA launch unavailable')
        trainer.blocks=blocks
        for _ in range(2):m.fit(x,y)
        m.clause_bank._pull();models.append(m)
    a,b=models
    np.testing.assert_array_equal(a.clause_bank.clause_bank,b.clause_bank.clause_bank)
    np.testing.assert_array_equal(a._coalesced_parallel_state,b._coalesced_parallel_state)
    np.testing.assert_array_equal(a.weight_banks[0].get_weights(),b.weight_banks[0].get_weights())
    np.testing.assert_array_equal(a.update_ps,np.zeros(1))
    np.testing.assert_array_equal(b.update_ps,np.zeros(1))

def test_cooperative_cache_reuses_actual_binary_and_invalidates_source_and_flags(gpu,tmp_path,monkeypatch):
    from tmu.clause_bank.cuda_cooperative import driver
    if not driver().supported():pytest.skip('Cooperative launch unavailable')
    from tmu.clause_bank.coalesced_compile import load_cooperative_kernel
    import pycuda.compiler as compiler
    source='#ifndef PROBE\n#define PROBE 41\n#endif\nextern "C" __global__ void cache_probe(int *out){if(threadIdx.x==0)out[0]=PROBE;}\n'
    def result(kernel):
        pointer=gpu.cuda.mem_alloc(4);out=np.empty(1,np.int32);kernel.launch((1,1,1),(32,1,1),(pointer,));gpu.cuda.memcpy_dtoh(out,pointer);return int(out[0])
    assert result(load_cooperative_kernel(source,'cache_probe','P',tmp_path))==41
    with monkeypatch.context() as patch:
        def unavailable(*a,**kw):raise AssertionError('Warm persistent cache must not invoke NVCC')
        patch.setattr(compiler,'compile',unavailable);patch.setattr(compiler,'get_nvcc_version',unavailable)
        assert result(load_cooperative_kernel(source,'cache_probe','P',tmp_path))==41
    assert result(load_cooperative_kernel(source.replace('41','42'),'cache_probe','P',tmp_path))==42
    monkeypatch.setattr(compiler,'DEFAULT_NVCC_FLAGS',list(compiler.DEFAULT_NVCC_FLAGS)+['-DPROBE=43'])
    assert result(load_cooperative_kernel(source,'cache_probe','P',tmp_path))==43


def test_cooperative_cache_repairs_corrupt_binary_before_launch(gpu,tmp_path):
    from tmu.clause_bank.cuda_cooperative import driver
    if not driver().supported():pytest.skip('Cooperative launch unavailable')
    from tmu.clause_bank.coalesced_compile import load_cooperative_kernel
    import hashlib
    source='extern "C" __global__ void cache_repair(int *out){if(threadIdx.x==0)out[0]=73;}\n'
    def result(kernel):
        pointer=gpu.cuda.mem_alloc(4);out=np.empty(1,np.int32);kernel.launch((1,1,1),(32,1,1),(pointer,));gpu.cuda.memcpy_dtoh(out,pointer);return int(out[0])
    assert result(load_cooperative_kernel(source,'cache_repair','P',tmp_path))==73
    cache=next(tmp_path.glob('*.cubin-cache'));cache.write_bytes(b'corrupt cache')
    assert result(load_cooperative_kernel(source,'cache_repair','P',tmp_path))==73
    checksum,separator,binary=cache.read_bytes().partition(b'\n');assert separator and checksum==hashlib.sha256(binary).hexdigest().encode()


def test_cooperative_cache_invalidates_changed_header(gpu,tmp_path,monkeypatch):
    from tmu.clause_bank.cuda_cooperative import driver
    if not driver().supported():pytest.skip('Cooperative launch unsupported')
    from tmu.clause_bank.coalesced_compile import load_cooperative_kernel
    import pycuda.compiler as compiler
    import pycuda.driver as cuda
    header=tmp_path/'probe_value.h';header.write_text('#define VALUE 17\n')
    monkeypatch.setattr(compiler,'DEFAULT_NVCC_FLAGS',list(compiler.DEFAULT_NVCC_FLAGS)+['-I'+str(tmp_path)])
    source='#include "probe_value.h"\nextern "C" __global__ void header_probe(int *out){out[0]=VALUE;}\n'
    def result():
        kernel=load_cooperative_kernel(source,'header_probe','P',tmp_path/'cache')
        out=cuda.mem_alloc(4);kernel.launch((1,1,1),(32,1,1),(out,))
        values=np.empty(1,np.int32);cuda.memcpy_dtoh(values,out);return int(values[0])
    assert result()==17
    header.write_text('#define VALUE 29\n')
    assert result()==29
