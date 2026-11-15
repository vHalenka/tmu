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

@pytest.mark.parametrize('chunks,patches',[(2,1),(43,1684),(64,33),(32,65),(33,65),(65,300),(43,8193)])
@pytest.mark.parametrize('sample',[False,True])
def test_patch_rank_and_rng_equal_serial_scan(chunks,patches,sample):
    from tmu.clause_bank import clause_bank_tensor_core as core
    from pycuda.compiler import SourceModule
    core._ensure_context();cuda=core.cuda
    folder=Path(core.__file__).parent/'cuda'
    parallel=(folder/'coalesced_parallel.cu').read_text()
    helper=parallel[parallel.index('__device__ int tc_warp_matching_patch'):parallel.index('__device__ void tc_grid_parallel_feedback')]
    source=(folder/'cpu_random.cuh').read_text()+(folder/'coalesced.cu').read_text()+helper+r'''
extern "C" __global__ void probe(const unsigned *ta,const unsigned *x,const unsigned *literal,int *out,uint64_t *state,int chunks,int bits,int patches,unsigned filter,int sample){
    __shared__ unsigned masks[256];
    TCPCG32 rng{1234567ULL};
    if(threadIdx.x==0){out[0]=tc_matching_patch(ta,x,literal,0,chunks,bits,filter,patches,sample?&rng:nullptr);state[0]=rng.state;}
    __syncwarp();rng.state=1234567ULL;
    int chosen=tc_warp_matching_patch(ta,x,literal,0,chunks,bits,filter,patches,sample?&rng:nullptr,sample?masks:nullptr);
    if(threadIdx.x==0){out[1]=chosen;state[1]=rng.state;}
}
'''
    fn=SourceModule(source,no_extern_c=True).get_function('probe')
    gen=np.random.RandomState(45);bits=8
    for kind in ['empty','reject','mixed','dropped','varied','boundary_tie']:
        ta=np.zeros((chunks,bits),np.uint32)
        if kind!='empty':ta[:,-1]=gen.randint(1,0xffffffff,chunks,dtype=np.uint32)
        x=np.zeros((patches,chunks),np.uint32)
        if kind=='mixed':x[::7]=0xffffffff
        if kind in ['varied','boundary_tie']:
            if kind=='boundary_tie':
                ta[:,-1]=1
                for k in [31,32,63]:
                    if k<chunks:ta[k,-1]=0x1fff
            actions=ta[:,-1].copy();actions[-1]&=np.uint32(0x1fff)
            x=gen.randint(0,0xffffffff,(patches,chunks),dtype=np.uint32)
            x[::5]|=actions
            x[1::7]|=actions
        literal=np.full(chunks,0xffffffff if kind!='dropped' else 0,np.uint32)
        result=np.zeros(2,np.int32);states=np.zeros(2,np.uint64)
        fn(cuda.In(ta),cuda.In(x),cuda.In(literal),cuda.Out(result),cuda.Out(states),
           np.int32(chunks),np.int32(bits),np.int32(patches),np.uint32(0x1fff),np.int32(sample),block=(32,1,1),grid=(1,1))
        assert result[0]==result[1],(kind,result)
        assert states[0]==states[1],(kind,states)
