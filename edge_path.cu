// edge_path.cu — counting kernel + launch wrapper.

#include "edge_path.cuh"

#include <cuda_runtime.h>

namespace edge_path_impl {

// One pass over the vertex array accumulating:
//   result[0] = Σ_{v : isStaticActive[v]}   degree[v]
//   result[1] = Σ_{v : isOverloadActive[v]} degree[v]
__global__ void countEdgesPerPathKernel(unsigned long long n,
                                        const bool *isStaticActive,
                                        const bool *isOverloadActive,
                                        const SIZE_TYPE *degreeD,
                                        unsigned long long *result) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long local_static   = 0;
    unsigned long long local_overload = 0;
    for (unsigned long long i = tid; i < n; i += stride) {
        SIZE_TYPE d = degreeD[i];
        if (isStaticActive[i])   local_static   += d;
        if (isOverloadActive[i]) local_overload += d;
    }
    if (local_static)   atomicAdd(&result[0], local_static);
    if (local_overload) atomicAdd(&result[1], local_overload);
}

void launchCountEdgesPerPath(unsigned long long n,
                             const bool *isStaticActive,
                             const bool *isOverloadActive,
                             const SIZE_TYPE *degreeD,
                             unsigned long long *d_result2) {
    constexpr int BLOCK = 256;
    constexpr int GRID = 512;
    countEdgesPerPathKernel<<<GRID, BLOCK>>>(n, isStaticActive, isOverloadActive,
                                             degreeD, d_result2);
}

} // namespace edge_path_impl
