// One shared clause match per image; class weights do not affect matching.
extern "C" __global__ void tc_shared_votes(const unsigned *ta,const unsigned *x,
    unsigned char *votes,int clauses,int chunks,int bits,int patches,unsigned filter)
{
    int lane=threadIdx.x&31;
    int clause=blockIdx.x*4+(threadIdx.x>>5), image=blockIdx.y;
    if(clause>=clauses)return;
    const unsigned *plane=ta+(unsigned long long)clause*chunks*bits;
    bool nonempty=false;
    for(int k=lane;k<chunks;k+=32)
        nonempty|=(plane[k*bits+bits-1]&(k==chunks-1?filter:0xffffffffu))!=0;
    if(!__any_sync(0xffffffffu,nonempty)){
        if(lane==0)votes[(unsigned long long)image*clauses+clause]=0;
        return;
    }
    bool fired=false;
    for(int base=0;base<patches;base+=32){
        int patch=base+lane;bool match=patch<patches;
        if(match){
            const unsigned *row=x+((unsigned long long)image*patches+patch)*chunks;
            for(int k=0;k<chunks;++k){
                unsigned action=plane[k*bits+bits-1]&(k==chunks-1?filter:0xffffffffu);
                if(action&~row[k]){match=false;break;}
            }
        }
        if(__any_sync(0xffffffffu,match)){fired=true;break;}
    }
    if(lane==0)votes[(unsigned long long)image*clauses+clause]=fired;
}

extern "C" __global__ void tc_shared_sums(const unsigned char *votes,const int *weights,
    int *sums,int clauses,int classes,int rows,int column_major)
{
    __shared__ unsigned partial[128];
    int image=blockIdx.x,cls=blockIdx.y;unsigned total=0;
    for(int c=threadIdx.x;c<clauses;c+=blockDim.x)
        if(votes[column_major?(unsigned long long)c*rows+image:(unsigned long long)image*clauses+c])total+=(unsigned)weights[(unsigned long long)cls*clauses+c];
    partial[threadIdx.x]=total;__syncthreads();
    for(int stride=64;stride;stride>>=1){
        if(threadIdx.x<stride)partial[threadIdx.x]+=partial[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0)sums[(unsigned long long)image*classes+cls]=(int)partial[0];
}
