// cache_density.cuh
//
// Per-iteration "cache density" + "miss rate" measurement for Liberator
// model 7.
//
// Definition:
//   cache                = Liberator's static edge region on GPU (vertices
//                          with isInStaticD[v] = true, determined at graph
//                          load time, constant across iterations).
//
//   active_in_cache[it]  = |{v : isActive[v] && isInStatic[v]}|, sampled at
//                          the START of iteration it, before setLabelDefault
//                          clears processed vertices.
//   active_total[it]     = |{v : isActive[v]}|                 (same instant)
//   active_out_cache[it] = active_total[it] - active_in_cache[it]
//                          = active nodes NOT in the static cache this iter
//
//   total_node_in_cache  = |{v : isInStatic[v]}|   (constant)
//
// Summary stats (micro-averaged across iters):
//   average_cache_density = Σ active_in_cache / (iters × static_vertex_count)
//   average_miss_rate     = Σ active_out_cache / Σ active_total
//                         = fraction of the frontier that had to fall back
//                           to the overload (host zerocopy) path.
//
#pragma once

#include "globals.cuh"

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace cache_density_impl {

// Defined in cache_density.cu.  Writes two atoms into `d_result2`:
//   d_result2[0] = |active ∩ static|
//   d_result2[1] = |active|
void launchCountActive(unsigned long long n, const bool *isActive,
                       const bool *isInStatic,
                       unsigned long long *d_result2);

inline std::string basenameNoExt(const std::string &path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}

} // namespace cache_density_impl

class CacheDensityRecorder {
public:
    SIZE_TYPE vertexArrSize;
    SIZE_TYPE staticVertexCount;         // |{v : isInStatic[v]=true}|, constant
    // Per-iter history
    std::vector<SIZE_TYPE> historyInCache;   // active ∩ static
    std::vector<SIZE_TYPE> historyTotal;     // |active|
    unsigned long long sumInCache = 0;
    unsigned long long sumTotal   = 0;
    std::string algo;
    std::string datasetName;
    long long sourceNode;
    bool enabled;    // false = record() is no-op (no sync / kernel / D2H)

private:
    unsigned long long *d_counter2 = nullptr;  // 2-element device scratch

public:
    // If `csvPath` is empty we skip all per-iter work (no d_counter alloc,
    // no thrust::count, no kernel, no D2H sync).  This keeps the instrumentation
    // free of performance impact when the caller is not asking for CSV output.
    CacheDensityRecorder(SIZE_TYPE n, bool *d_isInStatic,
                         const std::string &algoName,
                         const std::string &inputPath,
                         long long srcNode,
                         const std::string &csvPath)
        : vertexArrSize(n), algo(algoName),
          datasetName(cache_density_impl::basenameNoExt(inputPath)),
          sourceNode(srcNode),
          enabled(!csvPath.empty()) {
        if (!enabled) return;
        thrust::device_ptr<bool> p(d_isInStatic);
        staticVertexCount = (SIZE_TYPE)thrust::count(p, p + n, true);
        cudaError_t err = cudaMalloc(&d_counter2, 2 * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            std::cerr << "[CACHE] cudaMalloc d_counter2 failed: "
                      << cudaGetErrorString(err) << std::endl;
            d_counter2 = nullptr;
        }
    }

    ~CacheDensityRecorder() {
        if (d_counter2) cudaFree(d_counter2);
    }

    // Sample active_in_cache AND active_total at the current instant.
    // Call at the START of each iter BEFORE setLabelDefault (which clears
    // isActive for processed vertices).
    void record(bool *d_isActive, bool *d_isInStatic) {
        if (!enabled || !d_counter2) return;        // ← fast path when disabled
        cudaMemset(d_counter2, 0, 2 * sizeof(unsigned long long));
        cache_density_impl::launchCountActive(
            (unsigned long long)vertexArrSize, d_isActive, d_isInStatic, d_counter2);
        unsigned long long h[2] = {0, 0};
        cudaMemcpy(h, d_counter2, 2 * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        historyInCache.push_back((SIZE_TYPE)h[0]);
        historyTotal.push_back((SIZE_TYPE)h[1]);
        sumInCache += h[0];
        sumTotal   += h[1];
    }

    // Σ active_in_cache / (iters * static_vertex_count)
    double averageDensity() const {
        if (staticVertexCount == 0 || historyInCache.empty()) return 0.0;
        return (double)sumInCache /
               ((double)staticVertexCount * (double)historyInCache.size());
    }

    // Σ (active - active_in_cache) / Σ active   (micro-averaged)
    double averageMissRate() const {
        if (sumTotal == 0) return 0.0;
        return (double)(sumTotal - sumInCache) / (double)sumTotal;
    }

    void printSummary() const {
        if (!enabled) return;
        std::cout << "[CACHE] algo=" << algo
                  << " dataset=" << datasetName
                  << " source=" << sourceNode
                  << " total_iters=" << historyInCache.size()
                  << " sum_active_in_cache=" << sumInCache
                  << " sum_active_total=" << sumTotal
                  << " static_vertex_count=" << staticVertexCount
                  << " average_cache_density=" << averageDensity()
                  << " average_miss_rate=" << averageMissRate()
                  << std::endl;
    }

    // Append per-iter history into a CSV file.  Creates a header if file
    // doesn't exist yet.  Columns:
    //   algo,dataset,source,run,iter,active_in_cache,active_total,
    //   active_out_of_cache,static_vertex_count,per_iter_density,per_iter_miss_rate
    void writeCsv(const std::string &path, int runIndex) const {
        if (path.empty()) return;
        struct stat st;
        bool exists = (stat(path.c_str(), &st) == 0 && st.st_size > 0);
        std::ofstream f(path, std::ios::app);
        if (!f.is_open()) {
            std::cerr << "[CACHE] failed to open CSV: " << path << std::endl;
            return;
        }
        if (!exists) {
            f << "algo,dataset,source,run,iter,"
                 "active_in_cache,active_total,active_out_of_cache,"
                 "static_vertex_count,per_iter_density,per_iter_miss_rate\n";
        }
        int iters = (int)historyInCache.size();
        for (int i = 0; i < iters; ++i) {
            SIZE_TYPE aic   = historyInCache[i];
            SIZE_TYPE atot  = historyTotal[i];
            SIZE_TYPE aoc   = (atot > aic) ? (atot - aic) : 0;
            double density = (staticVertexCount == 0)
                               ? 0.0
                               : ((double)aic / (double)staticVertexCount);
            double miss    = (atot == 0) ? 0.0 : ((double)aoc / (double)atot);
            f << algo << "," << datasetName << "," << sourceNode << ","
              << runIndex << "," << (i + 1) << ","
              << aic << "," << atot << "," << aoc << ","
              << staticVertexCount << "," << density << "," << miss << "\n";
        }
        f.close();
    }
};
