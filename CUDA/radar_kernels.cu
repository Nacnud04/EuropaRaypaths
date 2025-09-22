#include <cuComplex.h>

const float pi = 3.141592653589793f;

__global__ void reflRadarSignal(float* d_SltRng, cuFloatComplex* d_fRe,
                                cuFloatComplex* refl_sig, float r0, float dr, int nr,
                                float range_res, float lam, int nfacets) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nfacets) {

        // iterate over time steps when computing chirp
        for (int ir=0; ir < nr; ir++) {

            // range at current step
            float r = r0 + ir * dr;
            
            // compute delta_r
            float delta_r = (r - d_SltRng[id]) / range_res;

            // get phase change component
            float phase = (2 * pi / lam) * d_SltRng[id];

            // add to reflected signal array
            // compute contribution for this facet/range bin
            cuFloatComplex phase_exp = make_cuFloatComplex(slowCos(2*phase), slowSin(2*phase));
            cuFloatComplex scale = make_cuFloatComplex(sinc(delta_r), 0.0f);
            cuFloatComplex contrib = cuCmulf(phase_exp, cuCmulf(d_fRe[id], scale));

            // accumulate atomically into refl_sig (separate real and imag parts)
            float cre = cuCrealf(contrib);
            float cim = cuCimagf(contrib);
            atomicAdd(&(refl_sig[ir].x), cre);
            atomicAdd(&(refl_sig[ir].y), cim);

        }

    }
}


__device__ int reradiate_index(int id0)
 {
    // placeholder for future redistribution of energy
    return id0;
 }

__global__ void refrRadarSignal(float* d_SltRng, 
                                cuFloatComplex* d_fReflEI, cuFloatComplex* d_fReflEO,
                                cuFloatComplex* refr_sig, 
                                float r0, float dr, int nr,
                                float range_res, float lam, int nfacets) {

    int id0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (id0 < nfacets) {

        int id1 = reradiate_index(id0);

        // get "oneway travel time slantrange"
        float sltrng = (d_SltRng[id0] + d_SltRng[id1]) / 2.0f;

        // get effective reradiation constant
        cuFloatComplex reradConst = cuCmulf(d_fReflEI[id0], d_fReflEO[id1]);

        // iterate over time steps when computing chirp
        for (int ir=0; ir < nr; ir++) {

            // range at current step
            float r = r0 + ir * dr;
            
            // compute delta_r
            float delta_r = (r - sltrng) / range_res;

            // get phase change component
            float phase = (2 * pi / lam) * sltrng;

            // add to reflected signal array
            // compute contribution for this facet/range bin
            cuFloatComplex phase_exp = make_cuFloatComplex(slowCos(2*phase), slowSin(2*phase));
            cuFloatComplex scale = make_cuFloatComplex(sinc(delta_r), 0.0f);
            cuFloatComplex contrib = cuCmulf(phase_exp, cuCmulf(reradConst, scale));

            // accumulate atomically into refl_sig (separate real and imag parts)
            float cre = cuCrealf(contrib);
            float cim = cuCimagf(contrib);
            atomicAdd(&(refr_sig[ir].x), cre);
            atomicAdd(&(refr_sig[ir].y), cim);

        }

    }
}
