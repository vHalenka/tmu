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

@pytest.mark.parametrize('bits,initial_state,expected_state,draws',[(1,1,1,3),(8,129,128,2)])
@pytest.mark.parametrize('certificate',[0,1])
def test_no_match_certificate_invalidates_when_decrement_enables_match(certificate,bits,initial_state,expected_state,draws):
    from tmu.clause_bank import clause_bank_tensor_core as core
    from pycuda.compiler import SourceModule
    core._ensure_context();cuda=core.cuda
    folder=Path(core.__file__).parent/'cuda'
    parallel=(folder/'coalesced_parallel.cu').read_text()
    helpers=parallel[parallel.index('__device__'):parallel.index('extern "C" __global__ void tc_coalesced_parallel_train')]
    source=(folder/'cpu_random.cuh').read_text()+(folder/'coalesced.cu').read_text()+helpers+r'''
extern "C" __global__ void probe(unsigned *ta,uint64_t *states,int *observed,int initial,int bits){
    __shared__ unsigned x[1],literal[1],masks[4];
    __shared__ int active[4],weights[4];
    __shared__ unsigned char votes[4],cache[4];
    if(threadIdx.x==0){
        x[0]=2;literal[0]=3;
        for(int c=0;c<4;++c){active[c]=1;weights[c]=1;votes[c]=0;cache[c]=initial;}
    }
    __syncthreads();
    tc_grid_parallel_feedback(ta,x,active,literal,weights,votes,cache,states,masks,
        4,1,2,bits,3u,1,2,1.f,1.f,1,1,1,0,1,2147483648u);
    __syncthreads();
    if(threadIdx.x==0)for(int c=0;c<4;++c)observed[c]=cache[c];
    __syncthreads();
    // The decrement made an empty clause. Type II must now see a match,
    // consume its selection draw and add the previously false literal.
    tc_grid_parallel_feedback(ta,x,active,literal,weights,votes,cache,states,masks,
        4,1,2,bits,3u,1,2,1.f,1.f,0,0,1,0,1,2147483648u);
}
'''
    fn=SourceModule(source,no_extern_c=True).get_function('probe')
    ta=np.zeros((4,1,bits),np.uint32)
    for bit in range(bits):ta[:,0,bit]=(initial_state>>bit)&1
    states=np.arange(7,23,4,dtype=np.uint64);observed=np.full(4,-1,np.int32)
    expected=states.copy()
    for _ in range(draws):expected*=np.uint64(6364136223846793005)
    fn(cuda.InOut(ta),cuda.InOut(states),cuda.Out(observed),np.int32(certificate),np.int32(bits),block=(128,1,1),grid=(1,1))
    expected_ta=np.zeros_like(ta)
    for bit in range(bits):expected_ta[:,0,bit]=(expected_state>>bit)&1
    np.testing.assert_array_equal(ta,expected_ta)
    np.testing.assert_array_equal(observed,np.full(4,certificate if bits==8 else 0,np.int32))
    np.testing.assert_array_equal(states,expected)
