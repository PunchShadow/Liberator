// edge_path.cuh
//
// Per-iteration edge-path breakdown for Liberator model 7.
//
// Rationale: align with CAGA's approach (per-path edge counters), so we
// measure which edges are served from which path.
//
// Liberator model 7 has two edge paths:
//   Static path    — vertices with isInStatic[v]=true; their edges live in
//                    staticEdgeListD (GPU-resident, never moves across iters).
//                    Analogous to CAGA's "cache" / cached chunks.
//   Overload path  — vertices with isInStatic[v]=false; their edges live in
//                    graph.edgeArray (host-pinned) and are read via UVA /
//                    zero-copy (New_*_kernelDynamic*).  No staging to a
//                    persistent GPU buffer. Analogous to CAGA's "sparse"
//                    path (HP slab H2D) conceptually, but Liberator uses a
//                    simpler zerocopy model — so the more accurate label is
//                    "overload / zerocopy".
//
// Definition of "edge path count":
//   For each iteration, we count the number of edges that will be PROCESSED
//   by each path.  For vertex v, the static kernel iterates `degreeD[v]`
//   edges when v is active (same for overload).  So:
//     edges_static[iter]   = Σ_{v : isStaticActive[v]  } degreeD[v]
//     edges_overload[iter] = Σ_{v : isOverloadActive[v]} degreeD[v]
//
// Summary stats (micro-averaged across all iters of a run):
//   total_edges_static / total_edges_overload
//   static_ratio   = total_edges_static   / (static+overload)
//   overload_ratio = total_edges_overload / (static+overload)
//
#pragma once

#include "globals.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace edge_path_impl {

// Defined in edge_path.cu.
void launchCountEdgesPerPath(unsigned long long n,
                             const bool *isStaticActive,
                             const bool *isOverloadActive,
                             const SIZE_TYPE *degreeD,
                             unsigned long long *d_result2);

inline std::string basenameNoExt(const std::string &path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}

} // namespace edge_path_impl

class EdgePathRecorder {
public:
    SIZE_TYPE vertexArrSize;
    std::vector<unsigned long long> historyStatic;    // edges in static path
    std::vector<unsigned long long> historyOverload;  // edges in overload path
    unsigned long long sumStatic   = 0;
    unsigned long long sumOverload = 0;
    std::string algo;
    std::string datasetName;
    long long sourceNode;
    bool enabled;    // false = record() is no-op (no sync / kernel / D2H)

private:
    unsigned long long *d_counter2 = nullptr;   // 2-element device scratch

public:
    // If `csvPath` is empty we skip all per-iter work (no d_counter alloc,
    // no kernel, no D2H).  Keeps instrumentation cost 0 when not requested.
    EdgePathRecorder(SIZE_TYPE n, const std::string &algoName,
                     const std::string &inputPath, long long srcNode,
                     const std::string &csvPath)
        : vertexArrSize(n), algo(algoName),
          datasetName(edge_path_impl::basenameNoExt(inputPath)),
          sourceNode(srcNode),
          enabled(!csvPath.empty()) {
        if (!enabled) return;
        cudaError_t err = cudaMalloc(&d_counter2, 2 * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            std::cerr << "[EDGE_PATH] cudaMalloc d_counter2 failed: "
                      << cudaGetErrorString(err) << std::endl;
            d_counter2 = nullptr;
        }
    }

    ~EdgePathRecorder() {
        if (d_counter2) cudaFree(d_counter2);
    }

    // Sample both counters at the current instant.
    //
    // Must be called AFTER `setStaticAndOverloadLabelBool` has populated
    // `isStaticActive[]` and `isOverloadActive[]` for this iteration, but
    // BEFORE the compute kernels run.  That window exists in every Liberator
    // model-7 main loop between the split kernel and the first processing
    // kernel.
    void record(bool *d_isStaticActive, bool *d_isOverloadActive,
                SIZE_TYPE *d_degree) {
        if (!enabled || !d_counter2) return;     // ← fast path when disabled
        cudaMemset(d_counter2, 0, 2 * sizeof(unsigned long long));
        edge_path_impl::launchCountEdgesPerPath(
            (unsigned long long)vertexArrSize,
            d_isStaticActive, d_isOverloadActive, d_degree, d_counter2);
        unsigned long long h[2] = {0, 0};
        cudaMemcpy(h, d_counter2, 2 * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        historyStatic.push_back(h[0]);
        historyOverload.push_back(h[1]);
        sumStatic   += h[0];
        sumOverload += h[1];
    }

    double staticRatio() const {
        unsigned long long t = sumStatic + sumOverload;
        return (t == 0) ? 0.0 : (double)sumStatic / (double)t;
    }

    double overloadRatio() const {
        unsigned long long t = sumStatic + sumOverload;
        return (t == 0) ? 0.0 : (double)sumOverload / (double)t;
    }

    void printSummary() const {
        if (!enabled) return;
        unsigned long long total = sumStatic + sumOverload;
        std::cout << "[EDGE_PATH] algo=" << algo
                  << " dataset=" << datasetName
                  << " source=" << sourceNode
                  << " total_iters=" << historyStatic.size()
                  << " total_edges=" << total
                  << " edges_static=" << sumStatic
                  << " edges_overload=" << sumOverload
                  << " static_ratio=" << staticRatio()
                  << " overload_ratio=" << overloadRatio()
                  << std::endl;
    }

    // Append per-iter history to CSV file.  Creates header if the file is
    // new.  Columns:
    //   algo,dataset,source,run,iter,edges_static,edges_overload,
    //   per_iter_static_ratio
    void writeCsv(const std::string &path, int runIndex) const {
        if (path.empty()) return;
        struct stat st;
        bool exists = (stat(path.c_str(), &st) == 0 && st.st_size > 0);
        std::ofstream f(path, std::ios::app);
        if (!f.is_open()) {
            std::cerr << "[EDGE_PATH] failed to open CSV: " << path << std::endl;
            return;
        }
        if (!exists) {
            f << "algo,dataset,source,run,iter,"
                 "edges_static,edges_overload,per_iter_static_ratio\n";
        }
        int iters = (int)historyStatic.size();
        for (int i = 0; i < iters; ++i) {
            unsigned long long es = historyStatic[i];
            unsigned long long eo = historyOverload[i];
            unsigned long long tot = es + eo;
            double r = (tot == 0) ? 0.0 : ((double)es / (double)tot);
            f << algo << "," << datasetName << "," << sourceNode << ","
              << runIndex << "," << (i + 1) << ","
              << es << "," << eo << "," << r << "\n";
        }
        f.close();
    }
};
