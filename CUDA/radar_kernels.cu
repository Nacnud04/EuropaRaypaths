#include <cuComplex.h>


// Tile size for range bins processed per-block in shared memory
#define REFL_TILE_NR 128

__global__ void reflRadarSignal(float* d_SltRng, cuFloatComplex* d_fRe,
                                cuFloatComplex* refl_sig, float r0, float dr, int nr,
                                float range_res, float lam, int nfacets) {

    // each block will process a contiguous tile of range bins at a time and
    // accumulate partial sums into shared memory, then flush once to global memory.
    extern __shared__ float s_mem[]; // size = 2 * tile_size (real, imag)
    float* s_real = s_mem; 
    float* s_imag = s_mem + REFL_TILE_NR;

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + tid;

    // number of facets handled by this thread stride
    int stride = blockDim.x * gridDim.x;

    for (int tileStart = 0; tileStart < nr; tileStart += REFL_TILE_NR) {

        int tileSize = min(REFL_TILE_NR, nr - tileStart);

        // initialize shared accumulators
        for (int i = tid; i < tileSize; i += blockDim.x) {
            s_real[i] = 0.0f;
            s_imag[i] = 0.0f;
        }
        __syncthreads();

        // each thread processes multiple facets (strided)
        for (int fid = id; fid < nfacets; fid += stride) {

            // load per-facet data into registers once
            float sRng = d_SltRng[fid];
            cuFloatComplex fRe = d_fRe[fid];

            // precompute phase multiplier (depends only on facet)
            float phase = (2.0f * pi / lam) * sRng;
            float twoPhase = 2.0f * phase;

            // iterate over bins in this tile
            for (int ti = 0; ti < tileSize; ++ti) {
                int ir = tileStart + ti;
                float r = r0 + ir * dr;
                float delta_r = (r - sRng) / range_res;

                // compute phase_exp using sincosf for efficiency
                float c, s;
                sincosf(twoPhase, &s, &c); // sincosf returns sin then cos? using (s, c)
                // careful: sincosf stores sin to first param, cos to second; we want (cos, sin)
                cuFloatComplex phase_exp = make_cuFloatComplex(c, s);

                float scale_val = sinc(delta_r);

                // contrib = phase_exp * fRe * scale
                cuFloatComplex contrib = cuCmulf(phase_exp, cuCmulf(fRe, make_cuFloatComplex(scale_val, 0.0f)));

                // accumulate into shared memory (use atomicAdd on shared floats)
                float cre = cuCrealf(contrib);
                float cim = cuCimagf(contrib);
                atomicAdd(&s_real[ti], cre);
                atomicAdd(&s_imag[ti], cim);
            }

        }

        __syncthreads();

        // flush shared accumulators to global memory once per block
        for (int i = tid; i < tileSize; i += blockDim.x) {
            atomicAdd(&(refl_sig[tileStart + i].x), s_real[i]);
            atomicAdd(&(refl_sig[tileStart + i].y), s_imag[i]);
        }
        __syncthreads();
    }

}


__device__ int reradiate_index(int id0)
 {
    // placeholder for future redistribution of energy
    return id0;
 }


// Tile size reuse for refracted signal
#define REFR_TILE_NR 128

__global__ void refrRadarSignal(float* d_SltRng, 
                                cuFloatComplex* d_fReflEI, cuFloatComplex* d_fReflEO,
                                cuFloatComplex* refr_sig, 
                                float r0, float dr, int nr,
                                float range_res, float P, float G, float fs, float lam, int nfacets) {

    extern __shared__ float s_mem2[]; // 2 * tile for real/imag
    float* s_real = s_mem2;
    float* s_imag = s_mem2 + REFR_TILE_NR;

    int tid = threadIdx.x;
    int id0 = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int tileStart = 0; tileStart < nr; tileStart += REFR_TILE_NR) {

        int tileSize = min(REFR_TILE_NR, nr - tileStart);

        // initialize shared accumulators
        for (int i = tid; i < tileSize; i += blockDim.x) {
            s_real[i] = 0.0f;
            s_imag[i] = 0.0f;
        }
        __syncthreads();

        // process facets in a strided fashion
        for (int fid = id0; fid < nfacets; fid += stride) {
            int fid1 = reradiate_index(fid);

            // compute sltrng and reradConst once per facet
            float sltrng = (d_SltRng[fid] + d_SltRng[fid1]) * 0.5f;
            cuFloatComplex reradConst = cuCmulf(d_fReflEI[fid], d_fReflEO[fid1]);
            reradConst = cuCmulf(reradConst, make_cuFloatComplex(radarEq(P, G, 1, lam, sltrng, fs), 0.0f));

            float phase = (2.0f * pi / lam) * sltrng;
            float twoPhase = 2.0f * phase;

            for (int ti = 0; ti < tileSize; ++ti) {
                int ir = tileStart + ti;
                float r = r0 + ir * dr;
                float delta_r = (r - sltrng) / range_res;

                float c, s;
                sincosf(twoPhase, &s, &c);
                cuFloatComplex phase_exp = make_cuFloatComplex(c, s);

                float scale_val = sinc(delta_r);

                cuFloatComplex contrib = cuCmulf(phase_exp, cuCmulf(reradConst, make_cuFloatComplex(scale_val, 0.0f)));

                float cre = cuCrealf(contrib);
                float cim = cuCimagf(contrib);
                atomicAdd(&s_real[ti], cre);
                atomicAdd(&s_imag[ti], cim);
            }
        }

        __syncthreads();

        // flush to global
        for (int i = tid; i < tileSize; i += blockDim.x) {
            atomicAdd(&(refr_sig[tileStart + i].x), s_real[i]);
            atomicAdd(&(refr_sig[tileStart + i].y), s_imag[i]);
        }
        __syncthreads();
    }

}


__global__ void combineRadarSignals(cuFloatComplex* refl_sig, cuFloatComplex* refr_sig,
                                    cuFloatComplex* total_sig, int nr) {

    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < nr) {

        total_sig[ir] = cuCaddf(refl_sig[ir], refr_sig[ir]);

    }
}