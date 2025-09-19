#include <cuComplex.h>

__global__ void constructReflectedSignal(float* d_Itd, cuFloatComplex* d_fRe,
                                         cuFloatComplex* refl_sig, float r0, float dr, int nr,
                                         float range_res, float lam, float c_1, float G, int nfacets) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nfacets) {

        // iterate over time steps when computing chirp
        for (int ir=0; ir < nr; ir++) {

            // range at current step
            float r = r0 + ir * dr;
            
            // compute delta_r
            float delta_r = (r - d_Itd[id]) / range_res;

            // get phase change component
            float phase = (2 * 3.14159265f / lam) * d_Itd[id];

            // add to reflected signal array
            refl_sig[ir] = cuCaddf(
                refl_sig[ir],
                // multiplication of sinc component with phase exponent
                cuCmulf(make_cuFloatComplex(slowCos(2*phase), slowSin(2*phase)),
                         // phase exponent multipled by scale component
                         cuCmulf(d_fRe[id], make_cuFloatComplex(sinc(delta_r), 0.0f)))
            );

        }

    }
}