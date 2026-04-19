// cache_density.cu — kernel definition and launch wrapper.
// Separated from cache_density.cuh to avoid multiple-definition linker
// errors when the header is included from multiple translation units.

#include "cache_density.cuh"

#include <cuda_runtime.h>

namespace cache_density_impl {

// One pass over the vertex array counts:
//   result[0] = |{v : isActive[v] && isInStatic[v]}|   (active & cached)
//   result[1] = |{v : isActive[v]}|                    (active total)
// active_out_of_cache can then be derived as result[1] - result[0].
__global__ void countActiveKernel(unsigned long long n,
                                  const bool *isActive,
                                  const bool *isInStatic,
                                  unsigned long long *result) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long local_in_cache = 0;
    unsigned long long local_total = 0;
    for (unsigned long long i = tid; i < n; i += stride) {
        if (isActive[i]) {
            local_total++;
            if (isInStatic[i]) local_in_cache++;
        }
    }
    if (local_in_cache) atomicAdd(&result[0], local_in_cache);
    if (local_total)    atomicAdd(&result[1], local_total);
}

void launchCountActive(unsigned long long n, const bool *isActive,
                       const bool *isInStatic,
                       unsigned long long *d_result2) {
    constexpr int BLOCK = 256;
    constexpr int GRID = 512;
    countActiveKernel<<<GRID, BLOCK>>>(n, isActive, isInStatic, d_result2);
}

} // namespace cache_density_impl
