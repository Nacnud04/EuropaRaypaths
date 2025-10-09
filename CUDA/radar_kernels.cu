#include <cuComplex.h>


// Tile size for range bins processed per-block in shared memory
#define REFL_TILE_NR 128

__global__ void reflRadarSignal(float* d_SltRng, float* d_fRe,
                                cuFloatComplex* refl_sig, float r0, float dr, int nr,
                                float range_res, float lam, int nfacets) {


    // Each block processes a contiguous tile of range bins. For each bin we
    // perform a block-level reduction: each thread computes a partial sum over
    // its assigned facets, then we reduce within warps using shuffles and
    // across warps using a small shared-memory array. Finally thread 0 in the
    // block does one atomicAdd into global memory per bin. This removes the
    // many atomic operations per-facet and drastically reduces contention.

    extern __shared__ float s_mem[]; // sized by launcher; we use it for warp partials
    // layout: s_mem[0..maxWarps-1] = real partials per-warp
    //         s_mem[REFL_TILE_NR .. REFL_TILE_NR+maxWarps-1] = imag partials
    float* s_warp_real = s_mem;
    float* s_warp_imag = s_mem + REFL_TILE_NR; // safe because launcher provides >= REFL_TILE_NR

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    const unsigned warpMask = 0xffffffffu;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = (blockDim.x + 31) / 32;

    for (int tileStart = 0; tileStart < nr; tileStart += REFL_TILE_NR) {

        int tileSize = min(REFL_TILE_NR, nr - tileStart);

        for (int ti = 0; ti < tileSize; ++ti) {

            int ir = tileStart + ti;

            // each thread computes partial sum for this bin across its facets
            float acc_re = 0.0f;
            float acc_im = 0.0f;

            // iterate facets assigned to this thread
            for (int fid = id; fid < nfacets; fid += stride) {
                float sRng = d_SltRng[fid];
                float fRe = d_fRe[fid];

                float r = r0 + ir * dr;
                float delta_r = (r - sRng) / range_res;

                float phase = (4.0f * 3.14159265f / lam) * sRng;
                float c, s;
                sincosf(phase, &s, &c);
                cuFloatComplex phase_exp = make_cuFloatComplex(c, s);

                float scale_val = sinc(delta_r);
                cuFloatComplex contrib = cuCmulf(phase_exp, make_cuFloatComplex(scale_val * fRe, 0.0f));
                acc_re += cuCrealf(contrib);
                acc_im += cuCimagf(contrib);
            }

            // warp-level reduction using shuffle
            for (int offset = 16; offset > 0; offset >>= 1) {
                float cre = __shfl_down_sync(warpMask, acc_re, offset);
                float cim = __shfl_down_sync(warpMask, acc_im, offset);
                acc_re += cre;
                acc_im += cim;
            }

            // lane 0 of each warp writes its partial to shared memory
            if (lane == 0) {
                s_warp_real[warpId] = acc_re;
                s_warp_imag[warpId] = acc_im;
            }
            __syncthreads();

            // reduce the warp partials using first warp
            float block_re = 0.0f;
            float block_im = 0.0f;
            if (warpId == 0) {
                // each thread in warp 0 loads one warp partial (if within numWarps)
                int idx = lane;
                if (idx < numWarps) {
                    block_re = s_warp_real[idx];
                    block_im = s_warp_imag[idx];
                }

                // reduce across lanes of warp 0
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float cre = __shfl_down_sync(warpMask, block_re, offset);
                    float cim = __shfl_down_sync(warpMask, block_im, offset);
                    block_re += cre;
                    block_im += cim;
                }

                // lane 0 now has the block's total for this bin
                if (lane == 0) {
                    atomicAdd(&(refl_sig[ir].x), block_re);
                    atomicAdd(&(refl_sig[ir].y), block_im);
                }
            }
            __syncthreads();
        }
    }
}


__device__ int reradiate_index(int id0)
 {
    // placeholder for future redistribution of energy
    return id0;
 }


// Tile size reuse for refracted signal
#define REFR_TILE_NR 128

__global__ void refrRadarSignal(float* d_SltRng, float* d_Rtd, 
                                float* d_fReflEI, float* d_fReflEO,
                                cuFloatComplex* refr_sig, 
                                float r0, float dr, int nr, float c, float c2, 
                                float range_res, float P, float G, float fs, float lam, int nfacets) {

    // Similar block-level reduction as in reflRadarSignal. Each block will
    // compute a partial sum per range-bin and perform a single atomicAdd per
    // bin when flushing to global memory.

    extern __shared__ float s_mem2[]; // provided by launcher; reused for warp partials
    float* s_warp_real = s_mem2;
    float* s_warp_imag = s_mem2 + REFR_TILE_NR;

    int tid = threadIdx.x;
    int id0 = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    const unsigned warpMask = 0xffffffffu;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = (blockDim.x + 31) / 32;

    for (int tileStart = 0; tileStart < nr; tileStart += REFR_TILE_NR) {

        int tileSize = min(REFR_TILE_NR, nr - tileStart);

        for (int ti = 0; ti < tileSize; ++ti) {

            int ir = tileStart + ti;

            float acc_re = 0.0f;
            float acc_im = 0.0f;

            // accumulate over facets assigned to this thread
            for (int fid = id0; fid < nfacets; fid += stride) {
                int fid1 = reradiate_index(fid);

                float sltrng = (d_SltRng[fid] + d_SltRng[fid1]) * 0.5f;
                
                float reradConst = d_fReflEI[fid] * d_fReflEO[fid1];
                reradConst = reradConst * radarEq(P, G, 1, lam, sltrng, fs);

                // slantrange equivalent time
                float rngt = (sltrng - d_Rtd[fid]) + d_Rtd[fid] * (c / c2); 

                float r = r0 + ir * dr;
                float delta_r = (r - (rngt)) / range_res;

                float phase = (4.0f * 3.14159265f / lam) * rngt;
                float c, s;
                sincosf(phase, &s, &c);
                cuFloatComplex phase_exp = make_cuFloatComplex(c, s);

                float scale_val = sinc(delta_r);
                cuFloatComplex contrib = cuCmulf(phase_exp, make_cuFloatComplex(scale_val * reradConst, 0.0f));
                acc_re += cuCrealf(contrib);
                acc_im += cuCimagf(contrib);
            }

            // warp-level reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                float cre = __shfl_down_sync(warpMask, acc_re, offset);
                float cim = __shfl_down_sync(warpMask, acc_im, offset);
                acc_re += cre;
                acc_im += cim;
            }

            if (lane == 0) {
                s_warp_real[warpId] = acc_re;
                s_warp_imag[warpId] = acc_im;
            }
            __syncthreads();

            // reduce across warps in warp 0
            float block_re = 0.0f;
            float block_im = 0.0f;
            if (warpId == 0) {
                int idx = lane;
                if (idx < numWarps) {
                    block_re = s_warp_real[idx];
                    block_im = s_warp_imag[idx];
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float cre = __shfl_down_sync(warpMask, block_re, offset);
                    float cim = __shfl_down_sync(warpMask, block_im, offset);
                    block_re += cre;
                    block_im += cim;
                }
                if (lane == 0) {
                    atomicAdd(&(refr_sig[ir].x), block_re);
                    atomicAdd(&(refr_sig[ir].y), block_im);
                }
            }
            __syncthreads();
        }
    }

}


__global__ void combineRadarSignals(cuFloatComplex* refl_sig, cuFloatComplex* refr_sig,
                                    cuFloatComplex* total_sig, int nr) {

    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < nr) {

        total_sig[ir] = cuCaddf(refl_sig[ir], refr_sig[ir]);

    }
}