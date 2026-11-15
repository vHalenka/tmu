// Independent clause streams distributed over a resident cooperative grid.
// Example/class order and all PCG draws match the single-CTA parallel mode.
#include <cooperative_groups.h>

struct TCCoalescedControl {
    int step_class, weight_enabled, proceed, target_sum, positive_count;
    double probability;
};

__device__ __forceinline__ unsigned tc_exact_mod(unsigned value,unsigned divisor,unsigned reciprocal)
{
    if(divisor==1)return 0;
    // reciprocal=floor(2^32/divisor): q underestimates by at most one.
    unsigned q=__umulhi(value,reciprocal);
    unsigned r=value-q*divisor;
    return r>=divisor?r-divisor:r;
}

__device__ void tc_cooperative_feedback_mask(
    unsigned *mask,int chunks,int literals,float s,TCPCG32 *rng,unsigned reciprocal)
{
    for(int k=0;k<chunks;++k)mask[k]=0u;
    float p=(float)(1.0/(double)s);
    float mean=(float)literals*p;
    float variance=(float)literals*p*(1.0f-p);
    int active=tc_pcg32_normal(rng,(double)mean,(double)variance);
    active=tc_cpu_clip(active,0,literals);
    // Two consecutive candidate outputs expose independent address arithmetic.
    // Commit in stream order; a scalar tail stops at the exact final draw.
    while(active>=2){
        uint64_t state=rng->state;
        TCPCG32 first{state},second{state*UINT64_C(6364136223846793005)};
        unsigned a=tc_exact_mod(tc_pcg32_next(&first),(unsigned)literals,reciprocal);
        unsigned b=tc_exact_mod(tc_pcg32_next(&second),(unsigned)literals,reciprocal);
        if(!(mask[a>>5]&(1u<<(a&31)))){mask[a>>5]|=1u<<(a&31);--active;}
        if(!(mask[b>>5]&(1u<<(b&31)))){mask[b>>5]|=1u<<(b&31);--active;}
        rng->state=state*UINT64_C(7520897724310334953);
    }
    while(active>0){
        unsigned literal=tc_exact_mod(tc_pcg32_next(rng),(unsigned)literals,reciprocal);
        if(!(mask[literal>>5]&(1u<<(literal&31)))){mask[literal>>5]|=1u<<(literal&31);--active;}
    }
}

__device__ int tc_warp_matching_patch(
    const unsigned *ta,const unsigned *x,const unsigned *literal,
    int clause,int chunks,int bits,unsigned filter,int patches,
    TCPCG32 *rng,unsigned *match_masks)
{
    int lane=threadIdx.x&31;
    if(patches>8192){
        int chosen=lane==0?tc_matching_patch(ta,x,literal,clause,chunks,bits,filter,patches,rng):-1;
        return __shfl_sync(0xffffffffu,chosen,0);
    }
    // Cache the included words in warp registers and skip empty words.
    // Every lane uses the same chunk order; patch rank and RNG draws are unchanged.
    unsigned a0=0,a1=0,b0=0,b1=0;
    if(chunks<=64){
        const unsigned *plane=ta+(unsigned long long)clause*chunks*bits;
        if(lane<chunks)a0=plane[lane*bits+bits-1]&literal[lane]&(lane==chunks-1?filter:0xffffffffu);
        int k=lane+32;
        if(k<chunks)a1=plane[k*bits+bits-1]&literal[k]&(k==chunks-1?filter:0xffffffffu);
        b0=__ballot_sync(0xffffffffu,a0!=0);b1=__ballot_sync(0xffffffffu,a1!=0);
        if(!(b0|b1)){
            int chosen=lane==0?(rng?(int)(tc_pcg32_next(rng)%(unsigned)patches):0):0;
            return __shfl_sync(0xffffffffu,chosen,0);
        }
    }
    int anchor=-1;unsigned anchor_action=0;
    if(chunks<=64 && patches>32 && __popc(b0)+__popc(b1)>1){
        unsigned first=(unsigned)(__popc(a0)*128+127-lane);
        unsigned second=(unsigned)(__popc(a1)*128+95-lane);
        unsigned best=first>second?first:second;
        for(int offset=16;offset;offset>>=1){unsigned other=__shfl_down_sync(0xffffffffu,best,offset);if(other>best)best=other;}
        best=__shfl_sync(0xffffffffu,best,0);anchor=127-(best&127);
        anchor_action=__shfl_sync(0xffffffffu,anchor>=32?a1:a0,anchor&31);
        if(anchor<32)b0&=~(1u<<anchor);else b1&=~(1u<<(anchor-32));
    }
    int count=0;
    for(int base=0;base<patches;base+=32){
        int patch=base+lane;
        bool match=patch<patches;
        if(match && anchor>=0)match=(anchor_action&~x[(unsigned long long)patch*chunks+anchor])==0;
        if(chunks<=64){
            bool alive=true;
            for(int group=0;group<2 && alive;++group){
                unsigned todo=group?b1:b0;
                while(todo){
                    int k=__ffs(todo)-1;todo&=todo-1;
                    unsigned action=__shfl_sync(0xffffffffu,group?a1:a0,k);
                    if(match && (action&~x[(unsigned long long)patch*chunks+group*32+k]))match=false;
                    alive=__any_sync(0xffffffffu,match);
                    if(!alive)break;
                }
            }
        }else if(match)match=tc_cpu_clause_match(ta,x+(unsigned long long)patch*chunks,literal,clause,chunks,bits,filter);
        unsigned found=__ballot_sync(0xffffffffu,match);
        if(!rng){if(found)return base+__ffs(found)-1;continue;}
        if(lane==0){match_masks[base>>5]=found;count+=__popc(found);}
    }
    int chosen=-1;
    if(lane==0 && count){
        unsigned rank=rng?tc_pcg32_next(rng)%(unsigned)count:0;
        for(int group=0;group<(patches+31)/32;++group){
            unsigned found=match_masks[group];int n=__popc(found);
            if(rank<(unsigned)n){while(rank--)found&=found-1;chosen=group*32+__ffs(found)-1;break;}
            rank-=n;
        }
    }
    return __shfl_sync(0xffffffffu,chosen,0);
}

__device__ void tc_grid_parallel_feedback(
    unsigned *ta,const unsigned *x,const int *active,const unsigned *literal,
    int *weights,const unsigned char *votes,unsigned char *empty_cache,uint64_t *states,unsigned *masks,
    int clauses,int chunks,int literals,int bits,unsigned filter,int boost,
    int max_included,float s,float probability,int type_i,int reinforce,
    int patches,int weight_pass,int weight_enabled,unsigned reciprocal)
{
#ifdef TC_STATIC_CHUNKS
    __shared__ unsigned match_scratch[4][TC_MATCH_WORDS];
    __shared__ unsigned mask_scratch[4][TC_MASK_WORDS];
#else
    __shared__ unsigned match_scratch[4][256];
    __shared__ unsigned mask_scratch[4][256];
#endif
    int lane=threadIdx.x&31,local_warp=threadIdx.x>>5;
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    int warps=(gridDim.x*blockDim.x)>>5;
    for(int clause=warp;clause<clauses;clause+=warps){
        TCPCG32 rng{0};if(lane==0)rng.state=states[clause];
        if(weight_pass){
            if(lane==0 && weight_enabled && active[clause] && votes[clause] && tc_pcg32_uniform(&rng)<=probability)
                weights[clause]=(int)((unsigned)weights[clause]+(reinforce?1u:0xffffffffu));
        }else{
            int apply=0;
            if(lane==0){
                float draw=tc_pcg32_uniform(&rng);
                bool sign=type_i?(reinforce?weights[clause]>=0:weights[clause]<0):(reinforce?weights[clause]<0:weights[clause]>=0);
                apply=draw<=probability && active[clause] && sign;
            }
            apply=__shfl_sync(0xffffffffu,apply,0);
            if(apply){
                // Voting proved non-matching clauses need no repeat scan until
                // their TA state changes. A no-match query consumes no RNG draw.
                int known_empty=lane==0?empty_cache[clause]:0;
                known_empty=__shfl_sync(0xffffffffu,known_empty,0);
                int patch=known_empty?-1:tc_warp_matching_patch(ta,x,literal,clause,chunks,bits,filter,patches,&rng,match_scratch[local_warp]);
                unsigned *mask=chunks<=256?mask_scratch[local_warp]:masks+(unsigned long long)clause*chunks;
                int decision=0;
                if(lane==0){
                    if(type_i && s>1.0f)tc_cooperative_feedback_mask(mask,chunks,literals,s,&rng,reciprocal);
                    decision=type_i?(patch>=0 && tc_cpu_included_count(ta,clause,chunks,bits,filter)<=max_included?1:2):(patch>=0?3:0);
                }
                decision=__shfl_sync(0xffffffffu,decision,0);
                __syncwarp();
                const unsigned *patch_x=x+(unsigned long long)(patch>=0?patch:0)*chunks;
                unsigned *plane=ta+(unsigned long long)clause*chunks*bits;
                bool actions_changed=false;
                for(int k=lane;k<chunks;k+=32){
                    unsigned valid=k==chunks-1?filter:0xffffffffu;
                    unsigned lit=literal[k]&valid;unsigned *word=plane+k*bits;
                    unsigned before_actions=word[bits-1];
                    if(decision==1){
                        unsigned inc=lit&patch_x[k];if(boost!=1 && s>1.0f)inc&=~mask[k];
                        tc_inc(word,inc,bits);tc_dec(word,(lit&~patch_x[k])&(s>1.0f?mask[k]:0xffffffffu),bits);
                    }else if(decision==2){tc_dec(word,lit&(s>1.0f?mask[k]:0xffffffffu),bits);}
                    else if(decision==3){tc_inc(word,lit&~patch_x[k],bits);}
                    actions_changed|=before_actions!=word[bits-1];
                }
                __syncwarp();
                bool changed=__any_sync(0xffffffffu,actions_changed);
                if(lane==0 && changed)empty_cache[clause]=0;
            }
        }
        if(lane==0)states[clause]=rng.state;
    }
}

// Let NVCC allocate registers for this geometry and CUDA choose resident blocks.
extern "C" __global__ void tc_coalesced_parallel_train(
    unsigned *ta,const unsigned *X,int *weights,const int *active,
    const unsigned *literal,const int *indices,const int *targets,
    unsigned char *votes,double *draws,unsigned char *decisions,unsigned *masks,
    uint64_t *parallel_states,
    uint32_t *mt_keys,int *mt_position,TCCoalescedControl *control,
    int samples,int clauses,int chunks,int literals,int classes,int patches,
    int T,int max_positive,int bits,int boost,int max_included,int focused,
    unsigned filter,float s,double type_i_p,double type_ii_p)
{
    cooperative_groups::grid_group grid=cooperative_groups::this_grid();
#ifdef TC_STATIC_CHUNKS
    chunks=TC_STATIC_CHUNKS;literals=TC_STATIC_LITERALS;
    patches=TC_STATIC_PATCHES;bits=TC_STATIC_BITS;
#endif
    int thread=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned divisor=(unsigned)literals;
    unsigned reciprocal=divisor?0xffffffffu/divisor+((divisor&(divisor-1u))==0?1u:0u):0;
    TCMT19937 mt{mt_keys,mt_position};
    for(int sample=0;sample<samples;++sample){
        const unsigned *x=X+(unsigned long long)indices[sample]*patches*chunks;
        if(patches>256){
            int lane=threadIdx.x&31;
            for(int clause=thread>>5;clause<clauses;clause+=(gridDim.x*blockDim.x)>>5){
                int chosen=tc_warp_matching_patch(ta,x,literal,clause,chunks,bits,filter,patches,nullptr,nullptr);
                if(lane==0){votes[clause]=chosen>=0;decisions[clause]=chosen<0;}
            }
        }else{
            for(int clause=thread;clause<clauses;clause+=gridDim.x*blockDim.x){
                int chosen=tc_matching_patch(ta,x,literal,clause,chunks,bits,filter,patches,nullptr);
                votes[clause]=chosen>=0;decisions[clause]=chosen<0;
            }
        }
        grid.sync();
        tc_coalesced_scores(weights,votes,active,clauses,classes,targets[sample],T,draws,&control->target_sum,&control->positive_count);
        grid.sync();
        if(thread==0){
            control->step_class=targets[sample];control->probability=((double)T-control->target_sum)/(2.0*T);
            control->weight_enabled=control->positive_count<max_positive;
            (void)tc_mt19937_interval(&mt,1u);
        }
        grid.sync();
        for(int pass=0;pass<3;++pass){
            double p=control->probability;float chance=(float)(pass==0?p*type_i_p:pass==1?p*type_ii_p:p);
            tc_grid_parallel_feedback(ta,x,active,literal,weights+(unsigned long long)control->step_class*clauses,votes,decisions,parallel_states,masks,clauses,chunks,literals,bits,filter,boost,max_included,s,chance,pass==0,1,patches,pass==2,control->weight_enabled,reciprocal);
        }
        // Each warp owns the same clauses in all three feedback passes.
        // Only the class boundary needs a grid-wide synchronization.
        grid.sync();
        if(thread==0){
            int target=targets[sample];double total=0.0;
            for(int cls=0;cls<classes;++cls)total+=draws[cls];
            control->proceed=total!=0.0;
            if(control->proceed){
                int chosen=-1;
                if(focused){
                    double cdf_total=0.0;for(int cls=0;cls<classes;++cls)cdf_total+=draws[cls]/total;
                    double mark=tc_mt19937_uniform(&mt),running=0.0;
                    for(int cls=0;cls<classes;++cls){running+=draws[cls]/total;if(chosen<0 && running/cdf_total>mark)chosen=cls;}
                    if(chosen<0)chosen=classes-1;
                }else{do{chosen=(int)tc_mt19937_interval(&mt,(unsigned)(classes-1));}while(chosen==target);}
                control->step_class=chosen;control->probability=draws[chosen];
            }
        }
        grid.sync();
        if(control->proceed){
            for(int pass=0;pass<3;++pass){
                double p=control->probability;float chance=(float)(pass==0?p*type_i_p:pass==1?p*type_ii_p:p);
                tc_grid_parallel_feedback(ta,x,active,literal,weights+(unsigned long long)control->step_class*clauses,votes,decisions,parallel_states,masks,clauses,chunks,literals,bits,filter,boost,max_included,s,chance,pass==0,0,patches,pass==2,1,reciprocal);
            }
        }
        grid.sync();
    }
}
