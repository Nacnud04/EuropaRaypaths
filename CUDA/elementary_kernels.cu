// --- TRIG FUNCTIONS ---

// slow but accurate sin function
__device__ void slowSinBulk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sinf(input[idx]);
    }
}
__device__ float slowSin(float input) {
    return sinf(input);
}
__device__ float sinGPU(float x) {
    return slowSin(x);
    // Approximate sin(x) using a Taylor series expansion
    // sin(x) ≈ x - x^3/6 + x^5/120 - x^7/5040 for small x
    //float x2 = x * x;
    //return x * (1 - x2 / 6 + x2 * x2 / 120 - x2 * x2 * x2 / 5040);
}

// sinc function
__device__ float sinc(float x) {
    if (fabsf(x) < 1e-6f) return 1.0f;
    return slowSin(M_PI * x) / (M_PI * x);
}

// slow but accurate arcsin function
__device__ void slowArcSinBulk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = asinf(input[idx]);
    }
}
__device__ float slowArcSin(float input) {
    return asinf(input);
}

// slow but accurate cos function
__device__ void slowCosBulk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = cosf(input[idx]);
    }
}
__device__ float slowCos(float input) {
    return cosf(input);
}
__device__ float cosGPU(float x) {
    return slowCos(x);
    // Approximate cos(x) using a Taylor series expansion
    // cos(x) ≈ 1 - x^2/2 + x^4/24 - x^6/720 for small x
    //float x2 = x * x;
    //return 1 - x2 / 2 + x2 * x2 / 24 - x2 * x2 * x2 / 720;
}

// slow but accurate arccos function
__device__ void slowArcCosBulk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = acosf(input[idx]);
    }
}
__device__ float slowArcCos(float input) {
    return acosf(input);
}

// slow but accurate atan2 function
__device__ float slowAtan2(float y, float x) {
    return atan2f(y, x);
}

// --- VECTOR OPERATIONS ---

// distance between a 3d point and a 3d point array
__device__ void pointDistanceBulk(float x1,  float y1,  float z1, 
                              float* x2, float* y2, float* z2,
                              float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dx = x2[idx] - x1;
        float dy = y2[idx] - y1;
        float dz = z2[idx] - z1;
        output[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}

// distance between two 3D point arrays
__device__ void pointDistanceVecBulk(float* x1, float* y1, float* z1, 
                              float* x2, float* y2, float* z2,
                              float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dx = x2[idx] - x1[idx];
        float dy = y2[idx] - y1[idx];
        float dz = z2[idx] - z1[idx];
        output[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}

// dot product of two 3D vector arrays
__device__ void dotProductBulk(float* x1, float* y1, float* z1, 
                           float* x2, float* y2, float* z2,
                           float* output, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = x1[idx] * x2[idx] + y1[idx] * y2[idx] + z1[idx] * z2[idx];
    }
}
__device__ float dotProduct(float x1, float y1, float z1, 
                            float x2, float y2, float z2) {

    return x1 * x2 + y1 * y2 + z1 * z2;
    
}

// dot product of two 3D vector arrays with negate option
__device__ void dotProductNegateBulk(float* x1, float* y1, float* z1, 
                                 float* x2, float* y2, float* z2,
                                 float* output, int n, bool negate) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = x1[idx] * x2[idx] + y1[idx] * y2[idx] + z1[idx] * z2[idx];
        if (negate) {
            output[idx] = -output[idx];
        }
    }
}

