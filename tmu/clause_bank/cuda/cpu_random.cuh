// CPU-compatible random streams for resident CUDA code. State belongs to the
// caller, so kernels can preserve draw order across launches and checkpoints.
#ifndef TMU_CUDA_CPU_RANDOM_CUH
#define TMU_CUDA_CPU_RANDOM_CUH

#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
#define TC_CPU_RNG_INLINE __host__ __device__ __forceinline__
#else
#define TC_CPU_RNG_INLINE inline
#endif

struct TCPCG32 {
    uint64_t state;
};

TC_CPU_RNG_INLINE uint32_t tc_pcg32_next(TCPCG32 *rng)
{
    // tmu/lib/src/random/pcg32_fast.c uses the old state for output.
    uint64_t x = rng->state;
    uint32_t count = (uint32_t)(x >> 61);
    rng->state = x * UINT64_C(6364136223846793005);
    return (uint32_t)((x ^ (x >> 22)) >> (22 + count));
}

TC_CPU_RNG_INLINE float tc_pcg32_uniform(TCPCG32 *rng)
{
    // Match the float casts in ClauseBank.c and WeightBank.c. The denominator
    // rounds UINT32_MAX up to 2^32 in float precision.
    return ((float)tc_pcg32_next(rng)) / ((float)UINT32_MAX);
}

TC_CPU_RNG_INLINE int tc_pcg32_normal(TCPCG32 *rng, double mean, double variance)
{
    // Match normal() in tmu/lib/include/fast_rand.h, including its uint32
    // addition before conversion to double.
    uint32_t raw1 = tc_pcg32_next(rng);
    uint32_t raw2 = tc_pcg32_next(rng);
    double u1 = (double)(uint32_t)(raw1 + 1u) / ((double)UINT32_MAX + 1.0);
    double u2 = (double)raw2 / (double)UINT32_MAX;
    double n1 = sqrt(-2.0 * log(u1)) * sin(8.0 * atan(1.0) * u2);
    return (int)round(mean + sqrt(variance) * n1);
}

struct TCMT19937 {
    uint32_t *keys;
    int *position;
};

TC_CPU_RNG_INLINE uint32_t tc_mt19937_next(TCMT19937 *rng)
{
    uint32_t *key = rng->keys;
    int pos = *rng->position;
    if (pos >= 624) {
        for (int i = 0; i < 227; ++i) {
            uint32_t y = (key[i] & 0x80000000u) | (key[i + 1] & 0x7fffffffu);
            key[i] = key[i + 397] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        for (int i = 227; i < 623; ++i) {
            uint32_t y = (key[i] & 0x80000000u) | (key[i + 1] & 0x7fffffffu);
            key[i] = key[i - 227] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        uint32_t y = (key[623] & 0x80000000u) | (key[0] & 0x7fffffffu);
        key[623] = key[396] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        pos = 0;
    }
    uint32_t y = key[pos++];
    *rng->position = pos;
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

TC_CPU_RNG_INLINE double tc_mt19937_uniform(TCMT19937 *rng)
{
    // NumPy legacy RandomState's rk_double: two tempered words make 53 bits.
    uint32_t a = tc_mt19937_next(rng) >> 5;
    uint32_t b = tc_mt19937_next(rng) >> 6;
    return ((double)a * 67108864.0 + (double)b) / 9007199254740992.0;
}

TC_CPU_RNG_INLINE uint32_t tc_mt19937_interval(TCMT19937 *rng, uint32_t max)
{
    // NumPy legacy RandomState's rk_interval used by shuffle and randint.
    uint32_t mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    uint32_t value;
    do {
        value = tc_mt19937_next(rng) & mask;
    } while (value > max);
    return value;
}


#undef TC_CPU_RNG_INLINE
#endif
