from pathlib import Path
import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    try:
        from tmu.clause_bank import clause_bank_tensor_core as core
        core._ensure_context()
    except (ImportError, RuntimeError) as error:
        pytest.skip(str(error))

def helpers():
    from tmu.clause_bank import clause_bank_tensor_core as core
    from pycuda.compiler import SourceModule
    core._ensure_context();folder=Path(core.__file__).parent/'cuda'
    text=(folder/'coalesced_parallel.cu').read_text()
    added=text[text.index('__device__ __forceinline__ unsigned tc_exact_mod'):text.index('__device__ int tc_warp_matching_patch')]
    source=(folder/'cpu_random.cuh').read_text()+(folder/'coalesced.cu').read_text()+added
    return core.cuda,SourceModule,source

def test_exact_mod_full_uint32_range_samples_and_divisor_edges():
    cuda,compiler,source=helpers()
    source+=r'''
extern "C" __global__ void probe(const unsigned *x,const unsigned *d,unsigned *out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){unsigned m=(unsigned)((1ULL<<32)/d[i]);out[i]=tc_exact_mod(x[i],d[i],m);}
}
'''
    fn=compiler(source,no_extern_c=True).get_function('probe')
    divisors=np.array([1,2,3,7,16,31,32,33,548,1362,65537,2147483647,4294967295],np.uint32)
    rng=np.random.RandomState(63)
    x=np.concatenate([rng.randint(0,0xffffffff,131072,dtype=np.uint32),np.arange(0xfffff000,0x100000000,dtype=np.uint64).astype(np.uint32),divisors,divisors-1])
    d=np.resize(divisors,len(x))
    edges=np.concatenate([np.zeros_like(divisors),divisors-1,divisors,
        np.minimum(divisors.astype(np.uint64)+1,np.uint64(0xffffffff)).astype(np.uint32),
        np.full_like(divisors,0xffffffff)])
    x=np.concatenate([x,edges]);d=np.concatenate([d,np.tile(divisors,5)])
    out=np.empty_like(x)
    fn(cuda.In(x),cuda.In(d),cuda.Out(out),np.int32(len(x)),block=(256,1,1),grid=((len(x)+255)//256,1))
    np.testing.assert_array_equal(out,x%d)

@pytest.mark.parametrize('literals,s',[(1,10.),(2,1.01),(17,3.9),(548,10.),(1362,10.),(1370,10.),(1370,1.001),(1370,1.e8),(65537,10.)])
def test_exact_mask_and_rng_match_original(literals,s):
    cuda,compiler,source=helpers()
    source+=r'''
extern "C" __global__ void masks(unsigned *old_mask,unsigned *new_mask,uint64_t *old_state,uint64_t *new_state,int chunks,int literals,float s){
    int i=threadIdx.x;TCPCG32 a{old_state[i]},b{new_state[i]};
    tc_cpu_feedback_mask(old_mask+(unsigned long long)i*chunks,chunks,literals,s,&a);
    unsigned m=(unsigned)((1ULL<<32)/(unsigned)literals);
    tc_cooperative_feedback_mask(new_mask+(unsigned long long)i*chunks,chunks,literals,s,&b,m);
    old_state[i]=a.state;new_state[i]=b.state;
}
'''
    fn=compiler(source,no_extern_c=True).get_function('masks')
    chunks=(literals+31)//32
    a=np.empty((32,chunks),np.uint32);b=np.empty_like(a)
    state=np.random.RandomState(13).randint(0,0xffffffffffffffff,32,dtype=np.uint64)|np.uint64(3);sa=state.copy();sb=state.copy()
    fn(cuda.Out(a),cuda.Out(b),cuda.InOut(sa),cuda.InOut(sb),np.int32(chunks),np.int32(literals),np.float32(s),block=(32,1,1),grid=(1,1))
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(sa,sb)
