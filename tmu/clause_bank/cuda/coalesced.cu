// Coalesced GPU training. Ordered draws follow the CPU loops.
// Parallel draws belong to each clause and can learn a different model.
// cpu_random.cuh is prepended by the Python loader.
__device__ void tc_inc(unsigned int *ta_state, unsigned int active, int bits)
{
    unsigned int carry = active;
    for (int b = 0; b < bits; ++b) {
        unsigned int carry_next = ta_state[b] & carry;
        ta_state[b] ^= carry;
        carry = carry_next;
    }
    if (carry) {
        for (int b = 0; b < bits; ++b) {
            ta_state[b] |= carry;
        }
    }
}

__device__ void tc_dec(unsigned int *ta_state, unsigned int active, int bits)
{
    unsigned int carry = active;
    for (int b = 0; b < bits; ++b) {
        unsigned int carry_next = (~ta_state[b]) & carry;
        ta_state[b] ^= carry;
        carry = carry_next;
    }
    if (carry) {
        for (int b = 0; b < bits; ++b) {
            ta_state[b] &= ~carry;
        }
    }
}

// Single-block parallel shared-output training with per-clause PCG streams.
// cpu_random.cuh must precede this file in the concatenated CUDA source.

__device__ __forceinline__ int tc_cpu_clip(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

__device__ bool tc_cpu_clause_match(
    const unsigned *ta, const unsigned *x, const unsigned *literal_active,
    int clause, int chunks, int bits, unsigned filter)
{
    const unsigned *plane = ta + (unsigned long long)clause * chunks * bits;
    for (int k = 0; k < chunks; ++k) {
        unsigned valid = k == chunks - 1 ? filter : 0xffffffffu;
        unsigned active = literal_active ? literal_active[k] : valid;
        if (plane[k * bits + bits - 1] & active & valid & ~x[k]) return false;
    }
    return true;
}

// Return one uniformly chosen matching patch, or -1. A null stream
// only checks whether any patch matches and consumes no random numbers.
__device__ int tc_matching_patch(
    const unsigned *ta, const unsigned *x, const unsigned *literal,
    int clause, int chunks, int bits, unsigned filter, int patches, TCPCG32 *rng)
{
    int count=0;
    for(int patch=0;patch<patches;++patch)
        if(tc_cpu_clause_match(ta,x+(unsigned long long)patch*chunks,literal,clause,chunks,bits,filter)){
            if(!rng)return patch;
            ++count;
        }
    if(!count)return -1;
    unsigned selected=tc_pcg32_next(rng)%(unsigned)count;
    for(int patch=0;patch<patches;++patch)
        if(tc_cpu_clause_match(ta,x+(unsigned long long)patch*chunks,literal,clause,chunks,bits,filter)){
            if(selected--==0)return patch;
        }
    return -1;
}

__device__ int tc_cpu_included_count(
    const unsigned *ta, int clause, int chunks, int bits, unsigned filter)
{
    const unsigned *plane = ta + (unsigned long long)clause * chunks * bits;
    int included = 0;
    for (int k = 0; k < chunks; ++k) {
        unsigned valid = k == chunks - 1 ? filter : 0xffffffffu;
        included += __popc(plane[k * bits + bits - 1] & valid);
    }
    return included;
}

__device__ void tc_cpu_feedback_mask(
    unsigned *mask, int chunks, int literals, float s, TCPCG32 *rng)
{
    for (int k = 0; k < chunks; ++k) mask[k] = 0u;
    float p = (float)(1.0 / (double)s);
    float mean = (float)literals * p;
    float variance = (float)literals * p * (1.0f - p);
    int active = tc_pcg32_normal(rng, (double)mean, (double)variance);
    active = tc_cpu_clip(active, 0, literals);
    while (active-- > 0) {
        int literal;
        do {
            literal = (int)(tc_pcg32_next(rng) % (unsigned)literals);
        } while (mask[literal >> 5] & (1u << (literal & 31)));
        mask[literal >> 5] |= 1u << (literal & 31);
    }
}

// Each clause owns its random stream. This is intentionally a different
// seeded learning trajectory from CPU training.
__device__ void tc_parallel_feedback_pass(
    unsigned *ta, const unsigned *x, const int *active, const unsigned *literal,
    const int *weights, uint64_t *states, unsigned *masks,
    int clauses, int chunks, int literals, int bits, unsigned filter,
    int boost, int max_included, float s, float probability,
    int type_i, int reinforce, int patches)
{
    for (int clause = threadIdx.x; clause < clauses; clause += blockDim.x) {
        TCPCG32 rng{states[clause]};
        float draw = tc_pcg32_uniform(&rng);
        bool sign = type_i
            ? (reinforce ? weights[clause] >= 0 : weights[clause] < 0)
            : (reinforce ? weights[clause] < 0 : weights[clause] >= 0);
        if (draw <= probability && active[clause] && sign) {
            int patch=tc_matching_patch(ta,x,literal,clause,chunks,bits,filter,patches,&rng);
            bool fired=patch>=0;
            const unsigned *patch_x=x+(unsigned long long)(fired?patch:0)*chunks;
            unsigned *mask = masks + (unsigned long long)clause * chunks;
            if (type_i && s > 1.0f)
                tc_cpu_feedback_mask(mask, chunks, literals, s, &rng);
            int decision = type_i
                ? (fired && tc_cpu_included_count(ta, clause, chunks, bits, filter) <= max_included ? 1 : 2)
                : (fired ? 3 : 0);
            unsigned *plane = ta + (unsigned long long)clause * chunks * bits;
            for (int k = 0; k < chunks; ++k) {
                unsigned valid = k == chunks - 1 ? filter : 0xffffffffu;
                unsigned lit = literal[k] & valid;
                unsigned *word = plane + k * bits;
                if (decision == 1) {
                    unsigned inc = lit & patch_x[k];
                    if (boost != 1 && s > 1.0f) inc &= ~mask[k];
                    tc_inc(word, inc, bits);
                    tc_dec(word, (lit & ~patch_x[k]) & (s > 1.0f ? mask[k] : 0xffffffffu), bits);
                } else if (decision == 2) {
                    tc_dec(word, lit & (s > 1.0f ? mask[k] : 0xffffffffu), bits);
                } else if (decision == 3) {
                    tc_inc(word, lit & ~patch_x[k], bits);
                }
            }
        }
        states[clause] = rng.state;
    }
    __syncthreads();
}

__device__ void tc_coalesced_class_step(
    unsigned *ta, const unsigned *x, int *weights, const unsigned char *votes,
    const int *active, const unsigned *literal, unsigned *masks, uint64_t *parallel_states,
    int clauses, int chunks, int literals, int bits, unsigned filter,
    int boost, int max_included, float s, double type_i_p, double type_ii_p,
    double probability, int reinforce, int weight_enabled, int patches)
{
    tc_parallel_feedback_pass(ta, x, active, literal, weights, parallel_states,
        masks, clauses, chunks, literals, bits, filter, boost, max_included,
        s, (float)(probability * type_i_p), 1, reinforce, patches);
    tc_parallel_feedback_pass(ta, x, active, literal, weights, parallel_states,
        masks, clauses, chunks, literals, bits, filter, boost, max_included,
        s, (float)(probability * type_ii_p), 0, reinforce, patches);
    if (weight_enabled) {
        for (int clause = threadIdx.x; clause < clauses; clause += blockDim.x) {
            TCPCG32 rng{parallel_states[clause]};
            if (active[clause] && votes[clause] && tc_pcg32_uniform(&rng) <= (float)probability)
                weights[clause] = (int)((unsigned)weights[clause] + (reinforce ? 1u : 0xffffffffu));
            parallel_states[clause] = rng.state;
        }
    }
    __syncthreads();
}

// Class scores depend on the pre-feedback votes and their own weight row.
// Target feedback changes only the target row, excluded from negative draws.
// Unsigned warp reductions preserve native int32 wrap regardless of grouping.
__device__ void tc_coalesced_scores(
    const int *weights, const unsigned char *votes, const int *active,
    int clauses, int classes, int target, int T, double *draws,
    int *target_sum, int *positive_count)
{
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int warps = (gridDim.x * blockDim.x) >> 5;
    for (int cls = warp; cls < classes; cls += warps) {
        unsigned sum = 0;
        int positive = 0;
        const int *row = weights + (unsigned long long)cls * clauses;
        for (int clause = lane; clause < clauses; clause += 32) {
            int weight = row[clause];
            if (active[clause] && votes[clause]) sum += (unsigned)weight;
            if (cls == target) positive += weight >= 0;
        }
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
            positive += __shfl_down_sync(0xffffffffu, positive, offset);
        }
        if (lane == 0) {
            int clipped = tc_cpu_clip((int)sum, -T, T);
            draws[cls] = cls == target ? 0.0 : ((double)T + clipped) / (2.0 * T);
            if (cls == target) {
                *target_sum = clipped;
                *positive_count = positive;
            }
        }
    }
}

extern "C" __global__ void tc_coalesced_train(
    unsigned *ta, const unsigned *X, int *weights,
    const int *active, const unsigned *literal, const int *indices,
    const int *targets, unsigned char *votes, double *draws,
    unsigned *masks, uint64_t *parallel_states, uint32_t *mt_keys, int *mt_position,
    int samples, int clauses, int chunks, int literals, int classes, int patches,
    int T, int max_positive, int bits, int boost, int max_included,
    int focused, unsigned filter, float s,
    double type_i_p, double type_ii_p)
{
    if (blockIdx.x != 0) return;
    TCMT19937 mt{mt_keys, mt_position};
    __shared__ int step_class, weight_enabled, proceed, target_sum, positive_count;
    __shared__ double probability;
    for (int sample = 0; sample < samples; ++sample) {
        const unsigned *x = X + (unsigned long long)indices[sample] * patches * chunks;
        for (int clause = threadIdx.x; clause < clauses; clause += blockDim.x)
            votes[clause] = tc_matching_patch(ta,x,literal,clause,chunks,bits,filter,patches,nullptr)>=0;
        __syncthreads();
        tc_coalesced_scores(weights, votes, active, clauses, classes,
                            targets[sample], T, draws, &target_sum, &positive_count);
        __syncthreads();
        if (threadIdx.x == 0) {
            step_class = targets[sample];
            probability = ((double)T - target_sum) / (2.0 * T);
            weight_enabled = positive_count < max_positive;
            // Upstream draws this even when Type III feedback is disabled.
            (void)tc_mt19937_interval(&mt, 1u);
        }
        __syncthreads();
        tc_coalesced_class_step(ta, x, weights + (unsigned long long)step_class * clauses,
            votes, active, literal, masks, parallel_states,
            clauses, chunks, literals, bits, filter, boost, max_included, s,
            type_i_p, type_ii_p, probability, 1, weight_enabled, patches);
        if (threadIdx.x == 0) {
            int target = targets[sample];
            double total = 0.0;
            for (int cls = 0; cls < classes; ++cls) total += draws[cls];
            proceed = total != 0.0;
            if (proceed) {
                int chosen = -1;
                if (focused) {
                    double cdf_total = 0.0;
                    for (int cls = 0; cls < classes; ++cls) cdf_total += draws[cls] / total;
                    double mark = tc_mt19937_uniform(&mt), running = 0.0;
                    for (int cls = 0; cls < classes; ++cls) {
                        running += draws[cls] / total;
                        if (chosen < 0 && running / cdf_total > mark) chosen = cls;
                    }
                    if (chosen < 0) chosen = classes - 1;
                } else {
                    do { chosen = (int)tc_mt19937_interval(&mt, (unsigned)(classes - 1)); }
                    while (chosen == target);
                }
                step_class = chosen;
                probability = draws[chosen];
            }
        }
        __syncthreads();
        if (proceed)
            tc_coalesced_class_step(ta, x, weights + (unsigned long long)step_class * clauses,
                votes, active, literal, masks, parallel_states,
                clauses, chunks, literals, bits, filter, boost, max_included, s,
                type_i_p, type_ii_p, probability, 0, 1, patches);
        __syncthreads();
    }
}
