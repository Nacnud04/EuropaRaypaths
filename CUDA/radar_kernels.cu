/******************************************************************************
 * File:        radar_kernels.cu
 * Author:      Duncan Byrne
 * Institution: Univeristy of Colorado Boulder
 * Department:  Aerospace Engineering Sciences
 * Email:       duncan.byrne@colorado.edu
 * Date:        2025-11-07
 *
 * Description:
 *    File providing CUDA functions relevant to radar signal generation
 *
 * Contents:
 *    - reflRdrSignal: Kernel to compute reflected radar signal
 *    - refrRdrSignal: Kernel to compute refracted radar signal
 *    - rerad_funct:   Device function for target reradiation behavior
 *    - reradiate_index: Device function for target reradiation index mapping
 *    - combineRadarSignals: Kernel to sum reflected and refracted signals
 *    - genReflPhasor: Kernel to generate reflection phasors
 *    - genRefrPhasor: Kernel to generate refraction phasors
 *    - genPhasorTrace: Function to bin phasors into range bins
 *    - genChirp: Kernel to generate chirp signal
 *    - genCenteredChirp: Kernel to generate a centered chirp signal
 *    - genCenteredChirpPadded: Kernel to generate padded centered chirp
 *    - realToComplex: Kernel to convert real signal to complex
 *    - complexPointwiseMul: Kernel for pointwise complex multiplication
 *    - scaleComplex: Kernel to scale complex array
 *    - convolvePhasorChirp: Function to convolve phasor with chirp using FFT (circular)
 *    - convolvePhasorChirpLinear: Same as above but linear convolution.
 *
 * Usage:
 *    #include "radar_kernels.cu"
 * 
 * Notes:
 *   - Requires CUDA toolkit with cuComplex and cuFFT libraries.
 *
 ******************************************************************************/

#include <cuComplex.h>
#include <math.h>

// thrust includes for phasor computation
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/complex.h>


// FFT for chirp convolution
#include <cufft.h>


// Tile size for range bins processed per-block in shared memory
#define REFL_TILE_NR 128
#define sqrt2pi 2.506628275f

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

// -- TARGET RERADIATION FUNCTIONS ---
// 0 = return all energy (corner reflector)
// 1 = specular reflector (1 degree boxcar)

// this function returns a scalar proportional to the amount of energy reradiated
// by the target from an inbound ray
__device__ float rerad_funct(int funcnum, float th1, float th2){
    if (funcnum == 0) {
        return 1;
    } else if (funcnum == 1) {
        if (th1 * (180/pi) < 1) {
            return 1;
        } else {return 0;}
    } else if (funcnum == 2) {
        if (th1 * (180/pi) < 2) {
            return 1;
        } else {return 0;}
    } else if (funcnum == 3) {
        if (th1 * (pi/180) < 3) {
            return 1;
        } else {return 0;}
    } else if (funcnum == 4) {
        float sigma = (pi/180);
        float center = 0;
        return expf(-0.5*pow((th1-center), 2)/(sigma*sigma));
    } else {
        return 1;
    }
}

__device__ int reradiate_index(int id0)
 {
    // placeholder for future redistribution of energy
    return id0;
 }


// Tile size reuse for refracted signal
#define REFR_TILE_NR 128

__global__ void refrRadarSignal(float* d_SltRng, float* d_Rtd, float* d_Rth, 
                                float* d_fReflEI, float* d_fReflEO,
                                cuFloatComplex* refr_sig, 
                                float r0, float dr, int nr, float c, float c2, int target_fun,
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
                
                float reradConst = rerad_funct(target_fun, d_Rth[fid], d_Rth[fid1]) * d_fReflEI[fid] * d_fReflEO[fid1];
                reradConst = reradConst * radarEq(P, G, fs, lam, sltrng);

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


__global__ void genReflPhasor(cuFloatComplex* refl_phasor, short* refl_rbs, 
                              float* d_fReflE, float* d_SltRng, 
                              float lam, float range_res, float nfacets,
                              float rst, float dr, int nr) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < nfacets) {

        // evaluate the phasor exponent
        float phase = (4.0f * 3.14159265f / lam) * d_SltRng[id];
        
        // exponentiate
        float c, s;
        sincosf(phase, &s, &c);
        cuFloatComplex phasor = make_cuFloatComplex(c, s);

        // scale by the coefficient
        refl_phasor[id] = cuCmulf(phasor, make_cuFloatComplex(d_fReflE[id], 0.0f));

        // compute the best range bin based on slant range
        short bin = (short)((d_SltRng[id] - rst) / dr);

        // if bin is out of range, set to bin 0, and zero the phasor
        if ((bin < 0) || (bin >= nr)) {
            bin = 0;
            refl_phasor[id] = make_cuFloatComplex(0.0f, 0.0f);
        }

        // move bin into array
        refl_rbs[id] = bin;

    }

}


__global__ void genRefrPhasor(cuFloatComplex* refr_phasor, short* refr_rbs,
                              float* d_fRfrSR, float* d_fReflEI, float* d_fReflEO, 
                              float* d_Tth, float* d_Ttd, int target_fun,
                              float P, float G, float lam, float fs, int nfacets,
                              float rst, float dr, int nr, float c, float c2) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < nfacets) {

        // get total slant ranged
        int id1 = reradiate_index(id);
        float sltrng = (d_fRfrSR[id] + d_fRfrSR[id1]) * 0.5f;

        // compute reradiation constant based on target reradiation function,
        // the radar equation, and other losses in fReflEI and fReflEO
        float reradConst = rerad_funct(target_fun, d_Tth[id], d_Tth[id1]) * d_fReflEI[id] * d_fReflEO[id1];
              reradConst = reradConst * radarEq(P, G, fs, lam, sltrng);

        // compute slant range equivalent time
        float rngt = (sltrng - d_Ttd[id]) + d_Ttd[id] * (c / c2);

        // evaluate the phasor exponent
        float phase = (4.0f * 3.14159265f / lam) * rngt;
        
        // exponentiate
        float c, s;
        sincosf(phase, &s, &c);
        cuFloatComplex phasor = make_cuFloatComplex(c, s);

        // scale by the coefficient
        refr_phasor[id] = cuCmulf(phasor, make_cuFloatComplex(reradConst, 0.0f));

        // compute the best range bin based on slant range
        short bin = (short)((rngt - rst) / dr);

        // if bin is out of range, set to bin 0, and zero the phasor
        if ((bin < 0) || (bin >= nr)) {
            bin = 0;
            refr_phasor[id] = make_cuFloatComplex(0.0f, 0.0f);
        }

        // move bin into array
        refr_rbs[id] = bin;
        
    }

}

struct cuComplexAdd {
    __host__ __device__
    cuFloatComplex operator()(const cuFloatComplex& a, const cuFloatComplex& b) const {
        return make_cuFloatComplex(a.x + b.x, a.y + b.y);
    }
};

void genPhasorTrace(cuFloatComplex* d_phasorTrace,
                    short* d_rbs, cuFloatComplex* d_phasors,
                    int nfacets, int nr){

    // turn pointers into thrust pointers
    thrust::device_ptr<short> keys_begin(d_rbs);
    thrust::device_ptr<short> keys_end(d_rbs + nfacets);
    thrust::device_ptr<cuFloatComplex> vals_begin(d_phasors);

    // create temporary buffers for sorted values and keys
    thrust::device_vector<short> sorted_keys(keys_begin, keys_end);
    thrust::device_vector<cuFloatComplex> sorted_vals(vals_begin, vals_begin + nfacets);

    // SORT BY KEY (EQUAL RANGE BINS CONSECUTIVE)
    thrust::sort_by_key(sorted_keys.begin(), sorted_keys.end(), sorted_vals.begin());

    // REDUCE BY KEY (SUM VALUES WITHIN EQUAL RANGE BINS)
    thrust::device_vector<short> unique_keys(nfacets);
    thrust::device_vector<cuFloatComplex> reduced_vals(nfacets);

    auto new_end = thrust::reduce_by_key(
        sorted_keys.begin(), sorted_keys.end(),
        sorted_vals.begin(),
        unique_keys.begin(),
        reduced_vals.begin(),
        thrust::equal_to<short>(),
        cuComplexAdd()
    );

    size_t num_unique = new_end.first - unique_keys.begin();
    //std::cout << "Number of unique range bins in phasor trace: " << num_unique << std::endl;

    // SCATTER REDUCED VALUES INTO OUTPUT PHASOR TRACE
    thrust::device_ptr<cuFloatComplex> output_begin(d_phasorTrace);
    thrust::fill(output_begin, output_begin + nr, make_cuFloatComplex(0.0f, 0.0f));

    // scatter reduced results into output array (phasor trace)
    thrust::scatter(
        reduced_vals.begin(), reduced_vals.begin() + num_unique,
        unique_keys.begin(), output_begin
    );

}


__global__ void genChirp(float* d_chirp, float rst, float dr, int nr, float range_res){

    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < nr) {

        float r = rst + ir * dr;
        float chirp_cen = rst;
        d_chirp[ir] = sinc((r-chirp_cen)/range_res);
        d_chirp[ir] += sinc((r-(rst + nr * dr))/range_res);

    }

}


__global__ void genCenteredChirp(float* d_chirp, float rst, float dr, int nr, float range_res){

    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < nr) {

        float r = rst + ir * dr;
        float chirp_cen = rst + (nr / 2) * dr;

        d_chirp[ir] = sinc((r - chirp_cen) / range_res);

    }
}


// Generate a centered chirp inside a padded buffer of length paddedNr.
// The chirp's nominal center corresponds to the center of an unpadded
// kernel of length `nr` (chirp center at rst + (nr/2)*dr). We place that
// center at index `kernel_center = nr/2` inside the padded buffer so that
// downstream extraction (starting at kernel_center) matches the original
// "same" extraction convention used by convolvePhasorChirpLinear.
__global__ void genCenteredChirpPadded(float* d_chirp, float rst, float dr, int nr, int paddedNr, float range_res){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < paddedNr) {

        int kernel_center = nr / 2; // where the original kernel center sits in the padded buffer
        // chirp center in meters
        float chirp_cen = rst + (nr / 2) * dr;

        // offset (in bins) from the chirp center for this padded index
        float offset_bins = (float)(i - kernel_center);
        float offset_range = offset_bins * dr;

        d_chirp[i] = sinc(offset_range / range_res);
    }
}


__global__ void realToComplex(const float* realSignal, cuFloatComplex* complexSignal, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        complexSignal[i].x = realSignal[i];
        complexSignal[i].y = 0.0f;
    }
}

__global__ void complexPointwiseMul(cuFloatComplex* a, const cuFloatComplex* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        cuFloatComplex A = a[i];
        cuFloatComplex B = b[i];
        a[i] = make_cuFloatComplex(A.x * B.x - A.y * B.y,
                                   A.x * B.y + A.y * B.x);
    }
}

__global__ void scaleComplex(cuFloatComplex* data, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

// Elementwise add: dest += src  (both length N)
__global__ void addComplexArrays(cuFloatComplex* dest, const cuFloatComplex* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dest[i].x += src[i].x;
        dest[i].y += src[i].y;
    }
}

void convolvePhasorChirp(cuFloatComplex* d_phasorTrace, float* d_chirp, 
                         cuFloatComplex* d_sig, int nr) {

    // Allocate device memory for complex version of chirp
    cuFloatComplex* d_chirpComplex;
    cudaMalloc(&d_chirpComplex, sizeof(cuFloatComplex) * nr);

    // Convert real chirp to complex
    int threads = 256;
    int blocks = (nr + threads - 1) / threads;
    realToComplex<<<blocks, threads>>>(d_chirp, d_chirpComplex, nr);

    // Copy phasor trace to output buffer (d_sig)
    cudaMemcpy(d_sig, d_phasorTrace, sizeof(cuFloatComplex) * nr, cudaMemcpyDeviceToDevice);

    // Create FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, nr, CUFFT_C2C, 1);

    // Forward FFT
    cufftExecC2C(plan, d_sig, d_sig, CUFFT_FORWARD);
    cufftExecC2C(plan, d_chirpComplex, d_chirpComplex, CUFFT_FORWARD);

    // Multiply in frequency domain
    complexPointwiseMul<<<blocks, threads>>>(d_sig, d_chirpComplex, nr);

    // Inverse FFT
    cufftExecC2C(plan, d_sig, d_sig, CUFFT_INVERSE);

    // Normalize by array length
    float scale = 1.0f / nr;
    scaleComplex<<<blocks, threads>>>(d_sig, nr, scale);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_chirpComplex);
}

void convolvePhasorChirpLinear(cuFloatComplex* d_phasorTrace, float* d_chirp,
                               cuFloatComplex* d_sig, int nr) {

    // zero padded size for linear convolution
    int paddedNr = 2 * nr;              

    // allocate device memory for padded complex chirp and phasor
    cuFloatComplex* d_PADchirp;
    cuFloatComplex* d_PADphasor;
    cudaMalloc(&d_PADchirp, sizeof(cuFloatComplex) * paddedNr);
    cudaMalloc(&d_PADphasor, sizeof(cuFloatComplex) * paddedNr);

    // zero fill both padded arrays
    cudaMemset(d_PADchirp, 0, sizeof(cuFloatComplex) * paddedNr);
    cudaMemset(d_PADphasor, 0, sizeof(cuFloatComplex) * paddedNr);

    // move phasor into padded array at beginning (indices 0..nr-1)
    cudaMemcpy(d_PADphasor, d_phasorTrace, sizeof(cuFloatComplex) * nr, cudaMemcpyDeviceToDevice);

    // convert chirp (real) -> complex directly into the padded chirp buffer.
    // Here we expect `d_chirp` to already contain the padded chirp of length
    // `paddedNr` (for the linear convolution path we generate a 2*nr chirp).
    int threads = 256;
    int blocks = (paddedNr + threads - 1) / threads;
    // convert the padded real chirp directly into the padded complex buffer
    realToComplex<<<blocks, threads>>>(d_chirp, d_PADchirp, paddedNr);

    // create FFT plan for padded length
    cufftHandle plan;
    cufftPlan1d(&plan, paddedNr, CUFFT_C2C, 1);

    // forward FFTs
    cufftExecC2C(plan, d_PADphasor, d_PADphasor, CUFFT_FORWARD);
    cufftExecC2C(plan, d_PADchirp,  d_PADchirp,  CUFFT_FORWARD);

    // multiply in frequency domain
    blocks = (paddedNr + threads - 1) / threads;
    complexPointwiseMul<<<blocks, threads>>>(d_PADphasor, d_PADchirp, paddedNr);

    // inverse FFT
    cufftExecC2C(plan, d_PADphasor, d_PADphasor, CUFFT_INVERSE);

    // normalize by padded length
    float scale = 1.0f / paddedNr;
    scaleComplex<<<blocks, threads>>>(d_PADphasor, paddedNr, scale);

    // linear convolution result lives in d_PADphasor[0 .. 2*nr-2].
    // to return "same"-mode (length nr) aligned with the original phasor,
    // extract nr samples starting at kernel_center which matches where we
    // generated/centered the chirp above.
    int kernel_center = nr / 2; // floor

    // copy the centered 'same' portion back to d_sig
    cudaMemcpy(d_sig, d_PADphasor + kernel_center, sizeof(cuFloatComplex) * nr, cudaMemcpyDeviceToDevice);

    // cleanup
    cufftDestroy(plan);
    cudaFree(d_PADchirp);
    cudaFree(d_PADphasor);
}