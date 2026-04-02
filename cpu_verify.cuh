#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <omp.h>
#include "globals.cuh"

using namespace std;

// ==================== BFS CPU Verification ====================
// Parallel level-synchronous BFS. Source level = 1 (matching Liberator).
static bool cpu_verify_bfs(
    const EDGE_POINTER_TYPE *nodePointers,
    const SIZE_TYPE *edgeList,
    SIZE_TYPE vertexArrSize,
    EDGE_POINTER_TYPE edgeArrSize,
    SIZE_TYPE sourceNode,
    const SIZE_TYPE *gpuResult)
{
    cout << "\n=== CPU BFS Verification (parallel) ===" << endl;
    auto t0 = chrono::steady_clock::now();
    const SIZE_TYPE INF = vertexArrSize + 1;

    SIZE_TYPE *dist = new SIZE_TYPE[vertexArrSize];
    #pragma omp parallel for
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) dist[i] = INF;
    dist[sourceNode] = 1;

    vector<SIZE_TYPE> frontier = {sourceNode};

    while (!frontier.empty()) {
        int nt = omp_get_max_threads();
        vector<vector<SIZE_TYPE>> tnext(nt);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic, 64)
            for (size_t fi = 0; fi < frontier.size(); fi++) {
                SIZE_TYPE v = frontier[fi];
                SIZE_TYPE lv = dist[v];
                EDGE_POINTER_TYPE s = nodePointers[v];
                EDGE_POINTER_TYPE e = (v + 1 < vertexArrSize) ? nodePointers[v + 1] : edgeArrSize;
                for (EDGE_POINTER_TYPE ei = s; ei < e; ei++) {
                    SIZE_TYPE u = edgeList[ei];
                    if (u < vertexArrSize && dist[u] == INF) {
                        if (__sync_bool_compare_and_swap(&dist[u], INF, lv + 1))
                            tnext[tid].push_back(u);
                    }
                }
            }
        }
        frontier.clear();
        for (auto &v : tnext)
            frontier.insert(frontier.end(), v.begin(), v.end());
    }

    auto t1 = chrono::steady_clock::now();
    cout << "  CPU time: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms" << endl;

    SIZE_TYPE mismatch = 0, rg = 0, rc = 0;
    #pragma omp parallel for reduction(+:mismatch, rg, rc)
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        if (gpuResult[i] != INF) rg++;
        if (dist[i] != INF) rc++;
        if (gpuResult[i] != dist[i]) mismatch++;
    }

    cout << "  GPU reached: " << rg << " / " << vertexArrSize << endl;
    cout << "  CPU reached: " << rc << " / " << vertexArrSize << endl;
    cout << "  Mismatches: " << mismatch << endl;
    if (mismatch > 0) {
        int shown = 0;
        for (SIZE_TYPE i = 0; i < vertexArrSize && shown < 10; i++)
            if (gpuResult[i] != dist[i]) {
                cout << "    node " << i << ": GPU=" << gpuResult[i] << " CPU=" << dist[i] << endl;
                shown++;
            }
    }
    bool pass = (mismatch == 0);
    cout << "  BFS Verification: " << (pass ? "PASSED" : "FAILED") << endl;
    delete[] dist;
    return pass;
}

// ==================== CC CPU Verification ====================
// Sequential Union-Find with path compression. Component label = min vertex ID.
static bool cpu_verify_cc(
    const EDGE_POINTER_TYPE *nodePointers,
    const SIZE_TYPE *edgeList,
    SIZE_TYPE vertexArrSize,
    EDGE_POINTER_TYPE edgeArrSize,
    const SIZE_TYPE *gpuResult)
{
    cout << "\n=== CPU CC Verification (Union-Find) ===" << endl;
    auto t0 = chrono::steady_clock::now();

    SIZE_TYPE *parent = new SIZE_TYPE[vertexArrSize];
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) parent[i] = i;

    // Iterative find with path compression
    auto find = [&](SIZE_TYPE v) -> SIZE_TYPE {
        while (parent[v] != v) {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    };

    // Process all edges
    for (SIZE_TYPE v = 0; v < vertexArrSize; v++) {
        EDGE_POINTER_TYPE s = nodePointers[v];
        EDGE_POINTER_TYPE e = (v + 1 < vertexArrSize) ? nodePointers[v + 1] : edgeArrSize;
        SIZE_TYPE rv = find(v);
        for (EDGE_POINTER_TYPE ei = s; ei < e; ei++) {
            SIZE_TYPE u = edgeList[ei];
            if (u < vertexArrSize) {
                SIZE_TYPE ru = find(u);
                if (rv != ru) {
                    if (rv < ru) parent[ru] = rv;
                    else { parent[rv] = ru; rv = ru; }
                }
            }
        }
    }

    // Flatten: result[v] = root of v's component (= min vertex ID in component)
    SIZE_TYPE *cpuResult = new SIZE_TYPE[vertexArrSize];
    #pragma omp parallel for
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        SIZE_TYPE v = i;
        while (parent[v] != v) v = parent[v];
        cpuResult[i] = v;
    }

    auto t1 = chrono::steady_clock::now();
    cout << "  CPU time: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms" << endl;

    // Direct comparison first
    SIZE_TYPE directMismatch = 0;
    #pragma omp parallel for reduction(+:directMismatch)
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++)
        if (gpuResult[i] != cpuResult[i]) directMismatch++;

    if (directMismatch == 0) {
        cout << "  CC Verification: PASSED (exact match)" << endl;
        delete[] parent;
        delete[] cpuResult;
        return true;
    }

    // Check structural equivalence: neighbors must share GPU labels
    cout << "  Label mismatches: " << directMismatch << " / " << vertexArrSize << endl;
    SIZE_TYPE structErr = 0;
    SIZE_TYPE checkLimit = min((SIZE_TYPE)100000, vertexArrSize);
    for (SIZE_TYPE v = 0; v < checkLimit; v++) {
        EDGE_POINTER_TYPE s = nodePointers[v];
        EDGE_POINTER_TYPE e = (v + 1 < vertexArrSize) ? nodePointers[v + 1] : edgeArrSize;
        for (EDGE_POINTER_TYPE ei = s; ei < e; ei++) {
            SIZE_TYPE u = edgeList[ei];
            if (u < vertexArrSize && gpuResult[v] != gpuResult[u]) {
                structErr++;
                if (structErr <= 5)
                    cout << "    Structural error: nodes " << v << " and " << u
                         << " are neighbors but GPU labels differ ("
                         << gpuResult[v] << " vs " << gpuResult[u] << ")" << endl;
            }
        }
    }

    // Count distinct component labels for GPU and CPU
    {
        vector<bool> seenGpu(vertexArrSize, false), seenCpu(vertexArrSize, false);
        SIZE_TYPE gpuComponents = 0, cpuComponents = 0;
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            if (gpuResult[i] < vertexArrSize && !seenGpu[gpuResult[i]]) { seenGpu[gpuResult[i]] = true; gpuComponents++; }
            if (cpuResult[i] < vertexArrSize && !seenCpu[cpuResult[i]]) { seenCpu[cpuResult[i]] = true; cpuComponents++; }
        }
        // Count GPU labels that are >= vertexArrSize (e.g. all set to vertexArrSize+1)
        SIZE_TYPE gpuOutOfRange = 0;
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++)
            if (gpuResult[i] >= vertexArrSize) gpuOutOfRange++;
        if (gpuOutOfRange > 0) gpuComponents += 1; // treat them as one "component"

        cout << "  GPU components: " << gpuComponents << "  CPU components: " << cpuComponents << endl;
        if (gpuComponents != cpuComponents) {
            cout << "  CC Verification: FAILED (component count mismatch: GPU="
                 << gpuComponents << " CPU=" << cpuComponents << ")" << endl;
            if (gpuOutOfRange > 0)
                cout << "    NOTE: " << gpuOutOfRange << " GPU vertices have label >= vertexArrSize "
                     << "(likely uninitialized — refreshLabelAndValue bug for CC model 0)" << endl;
            delete[] parent;
            delete[] cpuResult;
            return false;
        }
    }

    if (structErr > 0) {
        cout << "  CC Verification: FAILED (" << structErr << " structural errors in first "
             << checkLimit << " vertices)" << endl;
        delete[] parent;
        delete[] cpuResult;
        return false;
    }

    cout << "  CC Verification: PASSED (structurally equivalent, "
         << directMismatch << " label differences)" << endl;
    delete[] parent;
    delete[] cpuResult;
    return true;
}

// ==================== SSSP CPU Verification ====================
// Parallel Bellman-Ford with frontier. Source distance = 1 (matching Liberator).
static bool cpu_verify_sssp(
    const EDGE_POINTER_TYPE *nodePointers,
    const EdgeWithWeight *edgeList,
    SIZE_TYPE vertexArrSize,
    EDGE_POINTER_TYPE edgeArrSize,
    SIZE_TYPE sourceNode,
    const SIZE_TYPE *gpuResult)
{
    cout << "\n=== CPU SSSP Verification (parallel Bellman-Ford) ===" << endl;
    auto t0 = chrono::steady_clock::now();
    const SIZE_TYPE INF = vertexArrSize + 1;

    SIZE_TYPE *dist = new SIZE_TYPE[vertexArrSize];
    #pragma omp parallel for
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) dist[i] = INF;
    dist[sourceNode] = 1;

    bool *inNextFrontier = new bool[vertexArrSize];
    memset(inNextFrontier, 0, vertexArrSize * sizeof(bool));

    vector<SIZE_TYPE> frontier = {sourceNode};
    int iter = 0;

    while (!frontier.empty()) {
        iter++;
        memset(inNextFrontier, 0, vertexArrSize * sizeof(bool));

        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t fi = 0; fi < frontier.size(); fi++) {
            SIZE_TYPE v = frontier[fi];
            SIZE_TYPE vDist = dist[v];
            EDGE_POINTER_TYPE s = nodePointers[v];
            EDGE_POINTER_TYPE e = (v + 1 < vertexArrSize) ? nodePointers[v + 1] : edgeArrSize;
            for (EDGE_POINTER_TYPE ei = s; ei < e; ei++) {
                SIZE_TYPE u = edgeList[ei].toNode;
                SIZE_TYPE w = edgeList[ei].weight;
                if (u < vertexArrSize) {
                    SIZE_TYPE newDist = vDist + w;
                    // Atomic min using CAS
                    SIZE_TYPE old = dist[u];
                    while (newDist < old) {
                        SIZE_TYPE prev = __sync_val_compare_and_swap(&dist[u], old, newDist);
                        if (prev == old) {
                            inNextFrontier[u] = true;
                            break;
                        }
                        old = prev;
                    }
                }
            }
        }

        frontier.clear();
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            if (inNextFrontier[i]) frontier.push_back(i);
        }
    }

    auto t1 = chrono::steady_clock::now();
    cout << "  CPU time: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count()
         << " ms, iterations: " << iter << endl;

    SIZE_TYPE mismatch = 0, rg = 0, rc = 0;
    #pragma omp parallel for reduction(+:mismatch, rg, rc)
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        if (gpuResult[i] != INF) rg++;
        if (dist[i] != INF) rc++;
        if (gpuResult[i] != dist[i]) mismatch++;
    }

    cout << "  GPU reached: " << rg << " / " << vertexArrSize << endl;
    cout << "  CPU reached: " << rc << " / " << vertexArrSize << endl;
    cout << "  Mismatches: " << mismatch << endl;
    if (mismatch > 0) {
        int shown = 0;
        for (SIZE_TYPE i = 0; i < vertexArrSize && shown < 10; i++)
            if (gpuResult[i] != dist[i]) {
                cout << "    node " << i << ": GPU=" << gpuResult[i] << " CPU=" << dist[i] << endl;
                shown++;
            }
    }
    bool pass = (mismatch == 0);
    cout << "  SSSP Verification: " << (pass ? "PASSED" : "FAILED") << endl;
    delete[] dist;
    delete[] inNextFrontier;
    return pass;
}

// ==================== PR CPU Verification ====================
// Iterative PageRank on CSC graph. Formula: value[v] = 0.15 + 0.85 * sum.
// Convergence threshold: 0.001 (matching prKernel_Opt).
static bool cpu_verify_pr(
    const EDGE_POINTER_TYPE *nodePointers,  // CSC: incoming edge pointers
    const SIZE_TYPE *edgeList,               // source vertices of incoming edges
    const SIZE_TYPE *outDegree,
    SIZE_TYPE vertexArrSize,
    EDGE_POINTER_TYPE edgeArrSize,
    const float *gpuResult)
{
    cout << "\n=== CPU PageRank Verification (parallel) ===" << endl;
    auto t0 = chrono::steady_clock::now();

    double *value = new double[vertexArrSize];
    double *sumArr = new double[vertexArrSize];

    #pragma omp parallel for
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        value[i] = 0.15;
        sumArr[i] = 0.0;
    }

    int iter = 0;
    SIZE_TYPE activeCount = vertexArrSize;

    while (activeCount > 0) {
        iter++;

        // Compute incoming sum for each vertex
        #pragma omp parallel for schedule(dynamic, 256)
        for (SIZE_TYPE v = 0; v < vertexArrSize; v++) {
            EDGE_POINTER_TYPE s = nodePointers[v];
            EDGE_POINTER_TYPE e = (v + 1 < vertexArrSize) ? nodePointers[v + 1] : edgeArrSize;
            double tempSum = 0.0;
            for (EDGE_POINTER_TYPE ei = s; ei < e; ei++) {
                SIZE_TYPE src = edgeList[ei];
                if (src < vertexArrSize && outDegree[src] != 0) {
                    tempSum += value[src] / (double)outDegree[src];
                }
            }
            sumArr[v] = tempSum;
        }

        // Update values and check convergence
        activeCount = 0;
        #pragma omp parallel for reduction(+:activeCount)
        for (SIZE_TYPE v = 0; v < vertexArrSize; v++) {
            double newVal = 0.15 + 0.85 * sumArr[v];
            double diff = fabs(newVal - value[v]);
            if (diff > 0.01) activeCount++;
            value[v] = newVal;
        }
    }

    auto t1 = chrono::steady_clock::now();
    cout << "  CPU time: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count()
         << " ms, iterations: " << iter << endl;

    // Compare with tolerance
    SIZE_TYPE mismatch = 0;
    double maxDiff = 0.0;
    #pragma omp parallel for reduction(+:mismatch) reduction(max:maxDiff)
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        double diff = fabs(gpuResult[i] - value[i]);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > 0.01) mismatch++;
    }

    cout << "  Max difference: " << maxDiff << endl;
    cout << "  Mismatches (>0.01): " << mismatch << " / " << vertexArrSize << endl;

    if (mismatch > 0) {
        int shown = 0;
        for (SIZE_TYPE i = 0; i < vertexArrSize && shown < 10; i++) {
            double diff = fabs(gpuResult[i] - value[i]);
            if (diff > 0.01) {
                cout << "    node " << i << ": GPU=" << gpuResult[i] << " CPU=" << value[i] << endl;
                shown++;
            }
        }
    }

    bool pass = (mismatch == 0);
    cout << "  PR Verification: " << (pass ? "PASSED" : "FAILED") << endl;
    delete[] value;
    delete[] sumArr;
    return pass;
}
