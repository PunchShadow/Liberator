//
// Created by gxl on 2021/2/1.
//

#ifndef PTGRAPH_GRAPHMETA_CUH
#define PTGRAPH_GRAPHMETA_CUH

#include <string>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thread>
#include <type_traits>
#include <cstring>
#include "TimeRecord.cuh"
#include "globals.cuh"

// Helper: assign a uint64 value to EdgeType, handling EdgeWithWeight specially
template <typename EdgeType>
inline void assignEdgeFromU64(EdgeType &dst, uint64_t val, uint64_t weight = 1) {
    dst = static_cast<EdgeType>(val);
}
template <>
inline void assignEdgeFromU64<EdgeWithWeight>(EdgeWithWeight &dst, uint64_t val, uint64_t weight) {
    dst.toNode = val;
    dst.weight = weight;
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// GPU memory allocation tracker
struct GpuMemTracker {
    size_t totalAllocated = 0;
    size_t budgetBytes = 0; // 0 = no limit

    void setBudget(size_t bytes) { budgetBytes = bytes; }

    template<typename T>
    cudaError_t trackMalloc(T **ptr, size_t size, const char *name) {
        cudaError_t err = cudaMalloc(ptr, size);
        if (err == cudaSuccess) {
            totalAllocated += size;
            double sizeMB = size / (1024.0 * 1024.0);
            printf("  [cudaMalloc] %-35s %10.2f MB  (cumulative: %10.2f MB)\n",
                   name, sizeMB, totalAllocated / (1024.0 * 1024.0));
        } else {
            fprintf(stderr, "  [cudaMalloc FAILED] %s: %s (requested %.2f MB)\n",
                    name, cudaGetErrorString(err), size / (1024.0 * 1024.0));
        }
        return err;
    }

    void printSummary() {
        printf("\n========== GPU Memory Allocation Summary ==========\n");
        printf("  Total allocated:   %10.2f MB  (%6.2f GB)\n",
               totalAllocated / (1024.0 * 1024.0), totalAllocated / (1024.0 * 1024.0 * 1024.0));
        if (budgetBytes > 0) {
            printf("  Memory budget:     %10.2f MB  (%6.2f GB)\n",
                   budgetBytes / (1024.0 * 1024.0), budgetBytes / (1024.0 * 1024.0 * 1024.0));
            if (totalAllocated > budgetBytes) {
                printf("  *** OVER BUDGET by %.2f MB! ***\n",
                       (totalAllocated - budgetBytes) / (1024.0 * 1024.0));
            } else {
                printf("  Within budget (%.2f MB remaining)\n",
                       (budgetBytes - totalAllocated) / (1024.0 * 1024.0));
            }
        }
        size_t freeBytes, totalBytes;
        cudaMemGetInfo(&freeBytes, &totalBytes);
        printf("  Actual GPU free:   %10.2f MB  (%6.2f GB)\n",
               freeBytes / (1024.0 * 1024.0), freeBytes / (1024.0 * 1024.0 * 1024.0));
        printf("  Actual GPU total:  %10.2f MB  (%6.2f GB)\n",
               totalBytes / (1024.0 * 1024.0), totalBytes / (1024.0 * 1024.0 * 1024.0));
        printf("====================================================\n\n");
    }
};

// Global GPU memory tracker instance
static GpuMemTracker g_gpuMemTracker;

// Convenience macro for tracked allocation
#define GPU_MALLOC(ptr, size, name) gpuErrorcheck(g_gpuMemTracker.trackMalloc((ptr), (size), (name)))

struct PartEdgeListInfo {
    SIZE_TYPE partActiveNodeNums;
    SIZE_TYPE partEdgeNums;
    SIZE_TYPE partStartIndex;
};

using namespace std;
#define OLD_MODEL 0
#define NEW_MODEL1 1
#define NEW_MODEL2 2
#define NEW_MODEL3 3
template<class EdgeType>
class TestMeta {
public:
    ~TestMeta();
};

template<class EdgeType>
TestMeta<EdgeType>::~TestMeta() {

}

template<class EdgeType>
class GraphMeta {
public:
    int model;
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);
    SIZE_TYPE partOverloadSize;
    EDGE_POINTER_TYPE overloadSize;
    SIZE_TYPE sourceNode = 0;
    SIZE_TYPE vertexArrSize;
    EDGE_POINTER_TYPE edgeArrSize;
    EDGE_POINTER_TYPE *nodePointers;
    EDGE_POINTER_TYPE *nodePointersD;
    EdgeType *edgeArray;
    //special for pr
    SIZE_TYPE *outDegree;
    SIZE_TYPE *degree;
    bool *label;
    float *valuePr;
    SIZE_TYPE *value;
    bool *isInStatic;
    SIZE_TYPE *overloadNodeList;
    EDGE_POINTER_TYPE *staticNodePointer;
    EDGE_POINTER_TYPE *activeOverloadNodePointers;//no need for model 7
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    EdgeType *overloadEdgeList;//no need for model 7
    //GPU
    SIZE_TYPE *resultD;
    cudaStream_t steamStatic, streamDynamic;
    SIZE_TYPE *prefixSumTemp;
    EdgeType *staticEdgeListD;
    EdgeType *overloadEdgeListD;//no need for model 7
    bool *isInStaticD;
    SIZE_TYPE *overloadNodeListD;
    SIZE_TYPE *staticNodeListD;
    EDGE_POINTER_TYPE *staticNodePointerD;
    SIZE_TYPE *degreeD;
    SIZE_TYPE *outDegreeD;
    // async need two labels
    bool *isActiveD;
    thrust::device_ptr<bool> activeLablingThrust;
    thrust::device_ptr<bool> actStaticLablingThrust;
    thrust::device_ptr<bool> actOverLablingThrust;
    thrust::device_ptr<EDGE_POINTER_TYPE> actOverDegreeThrust;
    bool *isStaticActive;
    bool *isOverloadActive;
    SIZE_TYPE *valueD;
    float *valuePrD;
    float *sumD;
    //float *Diff;
    thrust::device_ptr<float> sumDThrust;
    thrust::device_ptr<float> DiffDThrust;
    //SIZE_TYPE *activeNodeListD;
    //SIZE_TYPE *activeNodeLabelingPrefixD;
    //SIZE_TYPE *overloadLabelingPrefixD;
    EDGE_POINTER_TYPE *activeOverloadNodePointersD;//no need for model 7
    EDGE_POINTER_TYPE *activeOverloadDegreeD;//no need for model 7
    double adviseRate;
    int paramSize;
    ALG_TYPE algType;
    long preMoveDataTime = 0;
    size_t gpuMemoryLimitBytes = 0; // 0 = use actual GPU memory

    void readDataFromFile(const string &fileName, bool isPagerank);

    void readDataFromBCSR(const string &fileName, bool isPagerank);

    void readDataFromBCSR64(const string &fileName, bool isPagerank);

    void readGraph(const string &fileName, bool isPagerank);

    void transFileUintToUlong(const string &fileName);

    ~GraphMeta();

    void setPrestoreRatio(double adviseK, int paramSize) {
        this->adviseRate = adviseK;
        this->paramSize = paramSize;
    }

    void initGraphHost();

    void initGraphDevice();

    void refreshLabelAndValue();

    void initAndSetStaticNodePointers();

    void setAlgType(ALG_TYPE type) {
        algType = type;
    }

    void setSourceNode(SIZE_TYPE sourceNode) {
        this->sourceNode = sourceNode;
    }

    void fillEdgeArrByMultiThread(SIZE_TYPE overloadNodeSize);

    void caculatePartInfoForEdgeList(SIZE_TYPE overloadNodeNum, EDGE_POINTER_TYPE overloadEdgeNum);
    void checkNode(SIZE_TYPE node);
    void checkNodeforPR(SIZE_TYPE node);
    void checkNodeforSSSP(SIZE_TYPE node);
    void writevalue(string filename);
    void setmodel(int _model){
        this->model = _model;
    }
    void setGpuMemoryLimit(double limitGB){
        if(limitGB > 0){
            this->gpuMemoryLimitBytes = (size_t)(limitGB * 1024ULL * 1024ULL * 1024ULL);
        }
    }
    bool checkgraph();
    SIZE_TYPE ret_max_partition_size(){
        return max_partition_size;
    }
private:
    SIZE_TYPE max_partition_size;
    SIZE_TYPE max_static_node;
    SIZE_TYPE total_gpu_size;
    uint fragmentSize = 4096;

    void getMaxPartitionSize();

    void initLableAndValue();
    
};

template<class EdgeType>
void GraphMeta<EdgeType>::readDataFromFile(const string &fileName, bool isPagerank) {
    cout << "readDataFromFile" << endl;
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &this->vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char *) &this->edgeArrSize, sizeof(EDGE_POINTER_TYPE));
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << endl;
    if (isPagerank) {
        outDegree = new SIZE_TYPE [vertexArrSize];
        uint *outDegreeU32 = new uint[vertexArrSize];
        infile.read((char *) outDegreeU32, sizeof(uint) * vertexArrSize);
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            outDegree[i] = (SIZE_TYPE) outDegreeU32[i];
        }
        delete[] outDegreeU32;
    }
    if(model==7) {
        nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
        infile.read((char *) nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
        gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(EdgeType)*edgeArrSize));
        // Read edges: original format may store smaller types, read directly
        // (original Liberator binary format stores edges at EDGE_POINTER_TYPE width)
        infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    }
    else{
        nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
        infile.read((char *) nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
        edgeArray = new EdgeType[edgeArrSize];
        infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    }
    
    infile.close();
    
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << endl;
}

static inline bool endsWith(const string &str, const string &suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

template<class EdgeType>
void GraphMeta<EdgeType>::readDataFromBCSR(const string &fileName, bool isPagerank) {
    cout << "readDataFromBCSR (Subway format)" << endl;
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    if (!infile.is_open()) {
        cerr << "Error: cannot open file " << fileName << endl;
        exit(1);
    }

    // Subway bcsr/bwcsr: header is two uint32 values
    uint num_nodes, num_edges;
    infile.read((char *) &num_nodes, sizeof(uint));
    infile.read((char *) &num_edges, sizeof(uint));
    this->vertexArrSize = num_nodes;
    this->edgeArrSize = num_edges;
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << endl;

    // Read uint32 nodePointers and widen to EDGE_POINTER_TYPE (64-bit)
    uint *nodePointersU32 = new uint[num_nodes];
    infile.read((char *) nodePointersU32, sizeof(uint) * num_nodes);

    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    for (uint i = 0; i < num_nodes; i++) {
        nodePointers[i] = (EDGE_POINTER_TYPE) nodePointersU32[i];
    }

    // Verify 32-bit → 64-bit widening
    {
        bool ok = true;
        // Check monotonicity
        for (uint i = 1; i < num_nodes; i++) {
            if (nodePointers[i] < nodePointers[i - 1]) {
                cerr << "ERROR: nodePointers not monotonic at " << i
                     << ": " << nodePointers[i-1] << " > " << nodePointers[i] << endl;
                ok = false;
                break;
            }
        }
        // Check last vertex's edge range does not exceed edgeArrSize
        EDGE_POINTER_TYPE lastEnd = nodePointers[num_nodes - 1]
            + (edgeArrSize - nodePointers[num_nodes - 1]);
        if (lastEnd != edgeArrSize) {
            cerr << "ERROR: last vertex edge range mismatch: " << lastEnd
                 << " != edgeArrSize " << edgeArrSize << endl;
            ok = false;
        }
        // Check widening preserved values (spot-check first, mid, last)
        uint checkIdx[] = {0, num_nodes / 2, num_nodes - 1};
        for (uint idx : checkIdx) {
            if (nodePointers[idx] != (EDGE_POINTER_TYPE) nodePointersU32[idx]) {
                cerr << "ERROR: widening mismatch at " << idx
                     << ": u32=" << nodePointersU32[idx]
                     << " u64=" << nodePointers[idx] << endl;
                ok = false;
            }
        }
        if (ok) {
            cout << "  ✓ 32→64 bit widening verified: "
                 << "nodePointers[0]=" << nodePointers[0]
                 << " nodePointers[" << num_nodes-1 << "]=" << nodePointers[num_nodes-1]
                 << " edgeArrSize=" << edgeArrSize << endl;
        } else {
            cerr << "FATAL: 32→64 bit widening verification FAILED" << endl;
            exit(1);
        }
    }
    delete[] nodePointersU32;

    // Allocate edge array
    if (model == 7) {
        gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(EdgeType) * edgeArrSize));
    } else {
        edgeArray = new EdgeType[edgeArrSize];
    }

    // Detect format mismatch between file and EdgeType, convert if needed
    // NOTE: Binary files store 32-bit data. We always read into temp buffers
    // and widen to 64-bit types since SIZE_TYPE is now uint64_t.
    bool fileIsWeighted = endsWith(fileName, ".bwcsr");
    const size_t CHUNK = 1 << 20; // 1M edges per chunk

    if (fileIsWeighted && !std::is_same<EdgeType, EdgeWithWeight>::value) {
        // .bwcsr file but EdgeType has no weight (e.g., uint64_t for BFS/CC/PR)
        // Read 32-bit EdgeWithWeight pairs, extract toNode and widen
        struct EdgeWithWeight32 { uint32_t toNode; uint32_t weight; };
        cout << "Auto-converting: .bwcsr -> stripping weights (32->64 bit)" << endl;
        EdgeWithWeight32 *buf = new EdgeWithWeight32[CHUNK];
        EDGE_POINTER_TYPE offset = 0;
        EDGE_POINTER_TYPE remaining = edgeArrSize;
        while (remaining > 0) {
            size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
            infile.read((char *)buf, sizeof(EdgeWithWeight32) * n);
            for (size_t i = 0; i < n; i++) {
                assignEdgeFromU64(edgeArray[offset + i], (uint64_t)buf[i].toNode);
            }
            offset += n;
            remaining -= n;
        }
        delete[] buf;
    } else if (!fileIsWeighted && std::is_same<EdgeType, EdgeWithWeight>::value) {
        // .bcsr file but EdgeType is EdgeWithWeight (SSSP)
        // Read uint32 toNode, widen to uint64_t, set default weight = 1
        cout << "Auto-converting: .bcsr -> adding default weight=1 (32->64 bit)" << endl;
        uint32_t *buf = new uint32_t[CHUNK];
        EDGE_POINTER_TYPE offset = 0;
        EDGE_POINTER_TYPE remaining = edgeArrSize;
        while (remaining > 0) {
            size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
            infile.read((char *)buf, sizeof(uint32_t) * n);
            for (size_t i = 0; i < n; i++) {
                EdgeWithWeight ew;
                ew.toNode = (uint64_t)buf[i];
                ew.weight = 1;
                memcpy(&edgeArray[offset + i], &ew, sizeof(EdgeType));
            }
            offset += n;
            remaining -= n;
        }
        delete[] buf;
    } else if (fileIsWeighted && std::is_same<EdgeType, EdgeWithWeight>::value) {
        // .bwcsr file and EdgeType is EdgeWithWeight (SSSP)
        // Read 32-bit pairs and widen to 64-bit EdgeWithWeight
        struct EdgeWithWeight32 { uint32_t toNode; uint32_t weight; };
        cout << "Reading .bwcsr with 32->64 bit widening" << endl;
        EdgeWithWeight32 *buf = new EdgeWithWeight32[CHUNK];
        EDGE_POINTER_TYPE offset = 0;
        EDGE_POINTER_TYPE remaining = edgeArrSize;
        while (remaining > 0) {
            size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
            infile.read((char *)buf, sizeof(EdgeWithWeight32) * n);
            for (size_t i = 0; i < n; i++) {
                EdgeWithWeight ew;
                ew.toNode = (uint64_t)buf[i].toNode;
                ew.weight = (uint64_t)buf[i].weight;
                memcpy(&edgeArray[offset + i], &ew, sizeof(ew));
            }
            offset += n;
            remaining -= n;
        }
        delete[] buf;
    } else {
        // .bcsr file with non-weighted EdgeType (uint64_t for BFS/CC/PR)
        // Read 32-bit edges and widen to 64-bit
        cout << "Reading .bcsr with 32->64 bit edge widening" << endl;
        uint32_t *buf = new uint32_t[CHUNK];
        EDGE_POINTER_TYPE offset = 0;
        EDGE_POINTER_TYPE remaining = edgeArrSize;
        while (remaining > 0) {
            size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
            infile.read((char *)buf, sizeof(uint32_t) * n);
            for (size_t i = 0; i < n; i++) {
                assignEdgeFromU64(edgeArray[offset + i], (uint64_t)buf[i]);
            }
            offset += n;
            remaining -= n;
        }
        delete[] buf;
    }

    infile.close();

    // For PageRank, compute outDegree from nodePointers
    if (isPagerank) {
        outDegree = new SIZE_TYPE[vertexArrSize];
        for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
            outDegree[i] = (SIZE_TYPE)(nodePointers[i + 1] - nodePointers[i]);
        }
        outDegree[vertexArrSize - 1] = (SIZE_TYPE)(edgeArrSize - nodePointers[vertexArrSize - 1]);
    }

    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromBCSR " << duration << " ms" << endl;
}

template<class EdgeType>
void GraphMeta<EdgeType>::readDataFromBCSR64(const string &fileName, bool isPagerank) {
    cout << "readDataFromBCSR64 (64-bit binary CSR format)" << endl;
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    if (!infile.is_open()) {
        cerr << "Error: cannot open file " << fileName << endl;
        exit(1);
    }

    // bcsr64/bwcsr64: header is two uint64_t values
    uint64_t num_nodes, num_edges;
    infile.read((char *) &num_nodes, sizeof(uint64_t));
    infile.read((char *) &num_edges, sizeof(uint64_t));
    this->vertexArrSize = num_nodes;
    this->edgeArrSize = num_edges;
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << endl;

    // Read uint64_t offsets directly into nodePointers (already EDGE_POINTER_TYPE = uint64_t)
    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    infile.read((char *) nodePointers, sizeof(uint64_t) * vertexArrSize);
    if (!infile) {
        cerr << "Failed to read bcsr64 node pointers from " << fileName << endl;
        exit(1);
    }

    // Allocate edge array
    if (model == 7) {
        gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(EdgeType) * edgeArrSize));
    } else {
        edgeArray = new EdgeType[edgeArrSize];
    }

    bool fileIsWeighted = endsWith(fileName, ".bwcsr64");

    if (fileIsWeighted && std::is_same<EdgeType, EdgeWithWeight>::value) {
        // bwcsr64: interleaved {uint64_t end, uint64_t w8} per edge
        // Read directly as EdgeWithWeight (toNode=uint64_t, weight=uint64_t)
        cout << "Reading bwcsr64 format with EdgeWithWeight" << endl;
        struct EdgeWeighted64 { uint64_t toNode; uint64_t weight; };
        const size_t CHUNK = 1 << 20;
        EDGE_POINTER_TYPE offset = 0;
        EDGE_POINTER_TYPE remaining = edgeArrSize;
        EdgeWeighted64 *buf = new EdgeWeighted64[CHUNK];
        while (remaining > 0) {
            size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
            infile.read((char *)buf, sizeof(EdgeWeighted64) * n);
            if (!infile) {
                cerr << "Failed to read bwcsr64 weighted edges from " << fileName << endl;
                exit(1);
            }
            for (size_t i = 0; i < n; i++) {
                EdgeWithWeight ew;
                ew.toNode = buf[i].toNode;
                ew.weight = buf[i].weight;
                memcpy(&edgeArray[offset + i], &ew, sizeof(EdgeType));
            }
            offset += n;
            remaining -= n;
        }
        delete[] buf;
    } else if (fileIsWeighted && !std::is_same<EdgeType, EdgeWithWeight>::value) {
        // bwcsr64 file but EdgeType is plain (BFS/CC/PR) — strip weights
        cout << "Reading bwcsr64 format, stripping weights" << endl;
        struct EdgeWeighted64 { uint64_t toNode; uint64_t weight; };
        const size_t CHUNK = 1 << 20;
        EDGE_POINTER_TYPE offset = 0;
        EDGE_POINTER_TYPE remaining = edgeArrSize;
        EdgeWeighted64 *buf = new EdgeWeighted64[CHUNK];
        while (remaining > 0) {
            size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
            infile.read((char *)buf, sizeof(EdgeWeighted64) * n);
            if (!infile) {
                cerr << "Failed to read bwcsr64 edges from " << fileName << endl;
                exit(1);
            }
            for (size_t i = 0; i < n; i++) {
                assignEdgeFromU64(edgeArray[offset + i], buf[i].toNode);
            }
            offset += n;
            remaining -= n;
        }
        delete[] buf;
    } else if (!fileIsWeighted && std::is_same<EdgeType, EdgeWithWeight>::value) {
        // bcsr64 file but EdgeType is EdgeWithWeight — add default weight
        cout << "Reading bcsr64 format, adding default weight=1" << endl;
        const size_t CHUNK = 1 << 20;
        uint64_t *buf = new uint64_t[CHUNK];
        EDGE_POINTER_TYPE offset = 0;
        EDGE_POINTER_TYPE remaining = edgeArrSize;
        while (remaining > 0) {
            size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
            infile.read((char *)buf, sizeof(uint64_t) * n);
            if (!infile) {
                cerr << "Failed to read bcsr64 edges from " << fileName << endl;
                exit(1);
            }
            for (size_t i = 0; i < n; i++) {
                EdgeWithWeight ew;
                ew.toNode = buf[i];
                ew.weight = 1;
                memcpy(&edgeArray[offset + i], &ew, sizeof(EdgeType));
            }
            offset += n;
            remaining -= n;
        }
        delete[] buf;
    } else {
        // bcsr64 file with plain EdgeType — read uint64_t edges directly
        cout << "Reading bcsr64 format (direct uint64_t edges)" << endl;
        infile.read((char *) edgeArray, sizeof(uint64_t) * edgeArrSize);
        if (!infile) {
            cerr << "Failed to read bcsr64 edges from " << fileName << endl;
            exit(1);
        }
    }

    infile.close();

    // For PageRank, compute outDegree from nodePointers
    if (isPagerank) {
        outDegree = new SIZE_TYPE[vertexArrSize];
        for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
            outDegree[i] = (SIZE_TYPE)(nodePointers[i + 1] - nodePointers[i]);
        }
        outDegree[vertexArrSize - 1] = (SIZE_TYPE)(edgeArrSize - nodePointers[vertexArrSize - 1]);
    }

    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromBCSR64 " << duration << " ms" << endl;
}

template<class EdgeType>
void GraphMeta<EdgeType>::readGraph(const string &fileName, bool isPagerank) {
    if (endsWith(fileName, ".bcsr64") || endsWith(fileName, ".bwcsr64")) {
        readDataFromBCSR64(fileName, isPagerank);
    } else if (endsWith(fileName, ".bcsr") || endsWith(fileName, ".bwcsr")) {
        readDataFromBCSR(fileName, isPagerank);
    } else {
        readDataFromFile(fileName, isPagerank);
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::transFileUintToUlong(const string &fileName) {
    ifstream infile(fileName, ios::in | ios::binary);
    // Source file stores 32-bit header — read into temp vars and widen
    uint num_nodes_u32, num_edges_u32;
    infile.read((char *) &num_nodes_u32, sizeof(uint));
    infile.read((char *) &num_edges_u32, sizeof(uint));
    this->vertexArrSize = num_nodes_u32;
    this->edgeArrSize = num_edges_u32;
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << endl;
    outDegree = new SIZE_TYPE[vertexArrSize];
    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    // Source file stores 32-bit nodePointers — read into temp buffer and widen
    uint *nodePointersU32 = new uint[vertexArrSize];
    infile.read((char *) nodePointersU32, sizeof(uint) * vertexArrSize);
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        nodePointers[i] = (EDGE_POINTER_TYPE) nodePointersU32[i];
    }
    delete[] nodePointersU32;
    // Source file stores 32-bit edges — read into temp buffer and widen
    if(model==7) {
        gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(EdgeType)*edgeArrSize));
    } else {
        edgeArray = new EdgeType[edgeArrSize];
    }
    {
        const size_t CHUNK = 1 << 20;
        uint32_t *buf = new uint32_t[CHUNK];
        EDGE_POINTER_TYPE offset = 0;
        EDGE_POINTER_TYPE remaining = edgeArrSize;
        while (remaining > 0) {
            size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
            infile.read((char *)buf, sizeof(uint32_t) * n);
            for (size_t i = 0; i < n; i++) {
                assignEdgeFromU64(edgeArray[offset + i], (uint64_t)buf[i]);
            }
            offset += n;
            remaining -= n;
        }
        delete[] buf;
    }
    infile.close();
    vector<ulong> transData(edgeArrSize);
    for (SIZE_TYPE i = 0; i < edgeArrSize; i++) {
        transData[i] = edgeArray[i];
    }

    std::ofstream outfile(fileName.substr(0, fileName.length() - 4) + "lcsr", std::ofstream::binary);
    // Output format intentionally writes 32-bit header for compatibility
    outfile.write((char *) &num_nodes_u32, sizeof(unsigned int));
    outfile.write((char *) &num_edges_u32, sizeof(unsigned int));
    // Write widened nodePointers as 32-bit (truncating back for legacy format)
    uint *nodePointersOut = new uint[vertexArrSize];
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        nodePointersOut[i] = (uint) nodePointers[i];
    }
    outfile.write((char *) nodePointersOut, sizeof(unsigned int) * vertexArrSize);
    delete[] nodePointersOut;
    outfile.write((char *) transData.data(), sizeof(ulong) * edgeArrSize);

    outfile.close();
}
template<class EdgeType>
void GraphMeta<EdgeType>::checkNode(SIZE_TYPE node)
{
    EDGE_POINTER_TYPE pointer=nodePointers[node];
    cout<<"check node "<<node<<endl;
    cout<<"pointer: "<<pointer<<endl;
    
    SIZE_TYPE _degree = nodePointers[node+1]-nodePointers[node];
    cout<<"degree: "<<_degree<<endl;
 
}

template<class EdgeType>
void GraphMeta<EdgeType>::checkNodeforPR(SIZE_TYPE node)
{
    EDGE_POINTER_TYPE pointer=nodePointers[node];
    cout<<"check node "<<node<<endl;
    cout<<"pointer: "<<pointer<<endl;
    SIZE_TYPE _degree = outDegree[node];
    cout<<"degree: "<<_degree<<endl;
}
template<class EdgeType>
void GraphMeta<EdgeType>::checkNodeforSSSP(SIZE_TYPE node)
{
    EDGE_POINTER_TYPE pointer=nodePointers[node];
    cout<<"check node "<<node<<endl;
    cout<<"pointer: "<<pointer<<endl;
   
    EdgeWithWeight edge = edgeArray[pointer];
    cout<<"edge to "<<edge.toNode<<" weighted "<<edge.weight<<endl;
    
}
template<class EdgeType>
bool GraphMeta<EdgeType>::checkgraph(){
    bool flag=true;
    cout<<"checkgraph()"<<endl;
    for(SIZE_TYPE i=0;i<vertexArrSize;i++){
        if(nodePointers[i]>=edgeArrSize){
            cout<<"pointer error at "<<i<<" with "<<nodePointers[i]<<endl;
            flag=false;
        }
    }
    for(EDGE_POINTER_TYPE i=0;i<edgeArrSize;i++){
        if(edgeArray[i]>=vertexArrSize){
            cout<<"edge error at "<<i<<" with "<<edgeArray[i]<<endl;
            flag=false;
        }
    }
    if(flag)
    cout<<"check graph correct!!!!!!"<<endl;
    return flag;
}
template<class EdgeType>
GraphMeta<EdgeType>::~GraphMeta() {
    if(model==OLD_MODEL){
        delete[] edgeArray;
        delete[] nodePointers;
        cout << "~GraphMeta" << endl;
        return;
    }
    if(model==7){
        cudaFree(edgeArray);
        cudaFree(nodePointers);
        cout << "~GraphMeta" << endl;
        return;
    }
    
    //delete[] outDegree;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initGraphHost() {
    cout << "initGraphHost()" << endl;
    degree = new SIZE_TYPE[vertexArrSize];
    isInStatic = new bool[vertexArrSize];
    overloadNodeList = new SIZE_TYPE[vertexArrSize];
    if(model!=7)
    activeOverloadNodePointers = new EDGE_POINTER_TYPE[vertexArrSize];

    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        if (nodePointers[i] > edgeArrSize) {
            cout << i << "   " << nodePointers[i] << endl;
            break;
        }
        degree[i] = nodePointers[i + 1] - nodePointers[i];
        
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    getMaxPartitionSize();
    initLableAndValue();
    if(model==OLD_MODEL){
        overloadEdgeList = (EdgeType *) malloc(overloadSize * sizeof(EdgeType));
    }
    else if(model!=7){
        gpuErrorcheck(cudaMallocManaged(&overloadEdgeList,sizeof(EdgeType)*overloadSize));
        gpuErrorcheck(cudaMemAdvise(overloadEdgeList,overloadSize*sizeof(EdgeType),cudaMemAdviseSetAccessedBy,0));
        //gpuErrorcheck(cudaMemAdvise(overloadEdgeList,overloadSize*sizeof(EdgeType),cudaMemAdviseSetReadMostly,0));
    }
    
    staticNodePointer = new EDGE_POINTER_TYPE[max_static_node+1];
    for (SIZE_TYPE i = 0; i < max_static_node+1; i++) {
        staticNodePointer[i] = nodePointers[i];
    }
}


template<class EdgeType>
void GraphMeta<EdgeType>::initGraphDevice() {
    cout << "initGraphDevice()" << endl;
    
    GPU_MALLOC(&resultD, grid.x * sizeof(SIZE_TYPE), "resultD");
    GPU_MALLOC(&prefixSumTemp, vertexArrSize * sizeof(SIZE_TYPE), "prefixSumTemp");
    //uint* tempResult = new uint[grid.x];
    //memset(tempResult, 0, sizeof(int) * grid.x);
    //cudaMemcpy(resultD, tempResult, grid.x * sizeof(int), cudaMemcpyHostToDevice);

    gpuErrorcheck(cudaPeekAtLastError());
    //cudaMemset(resultD, 0, grid.x * sizeof(uint));

    gpuErrorcheck(cudaStreamCreate(&steamStatic));
    gpuErrorcheck(cudaStreamCreate(&streamDynamic));
    //pre store
    TimeRecord<chrono::milliseconds> preMoveTimer("pre move data");
    preMoveTimer.startRecord();
    GPU_MALLOC(&staticEdgeListD, max_partition_size * sizeof(EdgeType), "staticEdgeListD");
    gpuErrorcheck(cudaMemcpy(staticEdgeListD, edgeArray, max_partition_size * sizeof(EdgeType), cudaMemcpyHostToDevice));
    preMoveTimer.endRecord();
    preMoveDataTime = preMoveTimer.getDuration();
    preMoveTimer.print();
    preMoveTimer.clearRecord();

    GPU_MALLOC(&isInStaticD, vertexArrSize * sizeof(bool), "isInStaticD");
    GPU_MALLOC(&overloadNodeListD, vertexArrSize * sizeof(SIZE_TYPE), "overloadNodeListD");
    GPU_MALLOC(&staticNodeListD, vertexArrSize * sizeof(SIZE_TYPE), "staticNodeListD");

    GPU_MALLOC(&staticNodePointerD, (max_static_node+1) * sizeof(EDGE_POINTER_TYPE), "staticNodePointerD");
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, staticNodePointer, (max_static_node+1) * sizeof(EDGE_POINTER_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaPeekAtLastError());
    cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    //test new model
    if(model==OLD_MODEL) {
        GPU_MALLOC(&overloadEdgeListD, partOverloadSize * sizeof(EdgeType), "overloadEdgeListD");
    } else{
        GPU_MALLOC(&nodePointersD, vertexArrSize*sizeof(EDGE_POINTER_TYPE), "nodePointersD");
        cudaMemcpy(nodePointersD,nodePointers,vertexArrSize*sizeof(EDGE_POINTER_TYPE),cudaMemcpyHostToDevice);
    }
    GPU_MALLOC(&degreeD, vertexArrSize * sizeof(SIZE_TYPE), "degreeD");
    GPU_MALLOC(&isActiveD, vertexArrSize * sizeof(bool), "isActiveD");
    GPU_MALLOC(&isStaticActive, vertexArrSize * sizeof(bool), "isStaticActive");
    GPU_MALLOC(&isOverloadActive, vertexArrSize * sizeof(bool), "isOverloadActive");
    //cudaMalloc(&activeNodeLabelingPrefixD, vertexArrSize * sizeof(SIZE_TYPE));
    //cudaMalloc(&overloadLabelingPrefixD, vertexArrSize * sizeof(SIZE_TYPE));
 
    //cudaMalloc(&activeNodeListD, vertexArrSize * sizeof(SIZE_TYPE));
    if(model!=7){
        GPU_MALLOC(&activeOverloadNodePointersD, vertexArrSize * sizeof(EDGE_POINTER_TYPE), "activeOverloadNodePointersD");
        GPU_MALLOC(&activeOverloadDegreeD, vertexArrSize * sizeof(EDGE_POINTER_TYPE), "activeOverloadDegreeD");
    }
    cudaMemcpy(degreeD, degree, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool));
    cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool));
    if(algType == PR) {
        
            GPU_MALLOC(&outDegreeD, vertexArrSize * sizeof(SIZE_TYPE), "outDegreeD");
            cudaMemcpy(outDegreeD, outDegree, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
            GPU_MALLOC(&valuePrD, vertexArrSize * sizeof(float), "valuePrD");
            cudaMemcpy(valuePrD, valuePr, vertexArrSize * sizeof(float), cudaMemcpyHostToDevice);
            GPU_MALLOC(&sumD, vertexArrSize * sizeof(float), "sumD");
            cudaMemset(sumD, 0, vertexArrSize * sizeof(float));
            //cudaMalloc(&Diff,vertexArrSize*sizeof(float));
            //cudaMemset(Diff,0.0,vertexArrSize*sizeof(float));
            sumDThrust = thrust::device_ptr<float>(sumD);
            //DiffDThrust = thrust::device_ptr<float>(Diff);
        
    } else {
        GPU_MALLOC(&valueD, vertexArrSize * sizeof(SIZE_TYPE), "valueD");
        cudaMemcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    }
    activeLablingThrust = thrust::device_ptr<bool>(isActiveD);
    actStaticLablingThrust = thrust::device_ptr<bool>(isStaticActive);
    actOverLablingThrust = thrust::device_ptr<bool>(isOverloadActive);
    if(model!=7)
    actOverDegreeThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(activeOverloadDegreeD);
    gpuErrorcheck(cudaPeekAtLastError());
    g_gpuMemTracker.printSummary();
    cout << "initGraphDevice() end" << endl;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initAndSetStaticNodePointers() {
    staticNodePointer = new EDGE_POINTER_TYPE[vertexArrSize];
    /*memcpy(staticNodePointer, nodePointers, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&staticNodePointerD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemcpy(staticNodePointerD, nodePointers, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);*/


}


template<class EdgeType>
void GraphMeta<EdgeType>::getMaxPartitionSize() {
    int deviceID;
    cudaDeviceProp dev{};
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);
    size_t totalMemory;
    size_t availMemory;
    cudaMemGetInfo(&availMemory, &totalMemory);
    if(gpuMemoryLimitBytes > 0 && gpuMemoryLimitBytes < availMemory){
        cout << "GPU memory limit set to " << gpuMemoryLimitBytes / (1024.0*1024.0*1024.0) << " GB" << endl;
        availMemory = gpuMemoryLimitBytes;
        g_gpuMemTracker.setBudget(gpuMemoryLimitBytes);
    } else {
        g_gpuMemTracker.setBudget(availMemory);
    }
    g_gpuMemTracker.totalAllocated = 0; // reset for this run
    size_t reduceMem;
    if(algType==PR){
        reduceMem = (size_t)(paramSize-2) * sizeof(SIZE_TYPE) * (size_t) vertexArrSize;
        reduceMem += sizeof(float) * 2 * (size_t)vertexArrSize;
    }
    else
    reduceMem = (size_t)paramSize * sizeof(SIZE_TYPE) * (size_t) vertexArrSize + (size_t)vertexArrSize*sizeof(EDGE_POINTER_TYPE);

    cout << "reduceMem " << reduceMem  << " ParamsSize " << paramSize << endl;
    cout << "availMemory " << availMemory << " totalMemory " << totalMemory << endl;
    if (reduceMem >= availMemory) {
        cout << "WARNING: per-vertex arrays (" << reduceMem / (1024.0*1024.0) << " MB) exceed available GPU memory ("
             << availMemory / (1024.0*1024.0) << " MB). Setting edge partition to minimum." << endl;
        total_gpu_size = 0;
    } else {
        cout << "available memory for edges "<< (availMemory - reduceMem) << " sizeof EdgeType is "<<sizeof(EdgeType)<<endl;
        total_gpu_size = (availMemory - reduceMem) / sizeof(EdgeType);
    }
    cout<<"total_gpu_size: "<<total_gpu_size<<endl;
    //getchar();
    //float adviseK = (10 - (float) edgeListSize / (float) totalSize) / 9;
    //uint dynamicDataMax = edgeListSize * edgeSize -i
    
        float adviseK = (10 - (double) edgeArrSize / (double) total_gpu_size) / 9;
            cout << "adviseK " << adviseK << endl;
            if (adviseK < 0) {
                adviseK = 0.5;
                cout << "adviseK " << adviseK << endl;
            }
            if (adviseK > 1) {
                adviseK = 1.0;
                cout << "adviseK " << adviseK << endl;
            }
            cout << "adviseRate " << adviseRate << endl;
            if (adviseRate > 0) {
                adviseK = adviseRate;
            }
            if(model!=7){
            max_partition_size = adviseK * total_gpu_size;
            }
            else
            max_partition_size = total_gpu_size;

            if (max_partition_size > edgeArrSize) {
                max_partition_size = edgeArrSize;
                cout<<"GPU fill all the edges!!!"<<endl;
            }
            
            printf("static memory is %zu  max static edge size is %lu\n gpu total edge size %lu \n",
                (reduceMem < availMemory) ? (availMemory - reduceMem) : 0,
                max_partition_size,
                total_gpu_size);
            // No UINT_MAX cap needed - SIZE_TYPE is now 64-bit
            unsigned long temp = max_partition_size % fragmentSize;
            max_partition_size = max_partition_size - temp;
            max_static_node = 0;
            SIZE_TYPE edgesInStatic = 0;
            for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                if (nodePointers[i] < max_partition_size && (nodePointers[i] + degree[i] - 1) < max_partition_size) {
                    isInStatic[i] = true;
                    if (i > max_static_node) max_static_node = i;
                    edgesInStatic += degree[i];
                } else {
                    isInStatic[i] = false;
                }
            }

            //cout << "max_partition_size " << max_partition_size << " nodePointers[vertexArrSize-1]" << nodePointers[vertexArrSize-1] << " edgesInStatic " << edgesInStatic << endl;

            partOverloadSize = total_gpu_size - max_partition_size;
            overloadSize = edgeArrSize - edgesInStatic;

    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    cout << " max staticnode is "<<max_static_node<<endl;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initLableAndValue() {

    label = new bool[vertexArrSize];
    if (algType == PR) {
        valuePr = new float[vertexArrSize];
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            label[i] = 1;
            valuePr[i] = 0.15f;
        }
    } else {
        value = new SIZE_TYPE[vertexArrSize];
        switch (algType) {
            case BFS:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = vertexArrSize + 1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case SSSP:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = vertexArrSize + 1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case CC:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = i;
                }
        }
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::refreshLabelAndValue() {
    cout << "refreshLabelAndValue()" << endl;
    if (algType == PR) {
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            label[i] = 1;
            valuePr[i] = 0.15f;
        }
        //cout << "refreshLabelAndValue() end1" << endl;
        cudaMemcpy(valuePrD, valuePr, vertexArrSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        
        //cout << "refreshLabelAndValue() end2" << endl;
        gpuErrorcheck(cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool)));
        //cout << "refreshLabelAndValue() end3" << endl;
    } else {
        switch (algType) {
            case BFS:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = vertexArrSize+1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                cout << "sourceNode " << sourceNode << endl;
                break;
            case SSSP:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = vertexArrSize + 1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case CC:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = vertexArrSize+1;
                }

        }
        cudaMemcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        gpuErrorcheck(cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool)));
    }

    activeLablingThrust = thrust::device_ptr<bool>(isActiveD);
    actStaticLablingThrust = thrust::device_ptr<bool>(isStaticActive);
    actOverLablingThrust = thrust::device_ptr<bool>(isOverloadActive);
    actOverDegreeThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(activeOverloadDegreeD);
    
}

template<class EdgeType>
void GraphMeta<EdgeType>::fillEdgeArrByMultiThread(SIZE_TYPE overloadNodeSize) {
    //cout << "fillEdgeArrByMultiThread" << endl;
    int threadNum = 20;
    if (overloadNodeSize < 50) {
        threadNum = 1;
    }
    thread runThreads[threadNum];

    for (int threadIndex = 0; threadIndex < threadNum; threadIndex++) {
        //cout << "======= threadIndex " << threadIndex << endl;
        runThreads[threadIndex] = thread([&, threadIndex] {
            float waitToHandleNum = overloadNodeSize;
            float numThreadsF = threadNum;
            SIZE_TYPE chunkSize = ceil(waitToHandleNum / numThreadsF);
            SIZE_TYPE left, right;
            //cout << "======= threadIndex " << threadIndex << endl;
            left = threadIndex * chunkSize;
            right = min(left + chunkSize, overloadNodeSize);
            SIZE_TYPE thisNode;
            SIZE_TYPE thisDegree;
            EDGE_POINTER_TYPE fromHere = 0;
            EDGE_POINTER_TYPE fromThere = 0;
            //cout << left << "=======" << right << endl;
            for (SIZE_TYPE i = left; i < right; i++) {
                thisNode = overloadNodeList[i];
                thisDegree = degree[thisNode];
                fromHere = activeOverloadNodePointers[i];
                fromThere = nodePointers[thisNode];

                // if(activeOverloadNodePointers[i] > overloadSize) {
                //     cout << "activeOverloadNodePointers[" << i << "] is " << activeOverloadNodePointers[i] << endl;
                //     break;
                // }
                for (SIZE_TYPE j = 0; j < thisDegree; j++) {
                    overloadEdgeList[fromHere + j] = edgeArray[fromThere + j];
                    //cout << fromHere + j << " : " << overloadEdgeList[fromHere + j] << endl;
                }

            }
        });
    }
    for (int t = 0; t < threadNum; t++) {
        runThreads[t].join();
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::caculatePartInfoForEdgeList(SIZE_TYPE overloadNodeNum, EDGE_POINTER_TYPE overloadEdgeNum) {
    partEdgeListInfoArr.clear();
    if (partOverloadSize < overloadEdgeNum) {
        SIZE_TYPE left = 0;
        SIZE_TYPE right = overloadNodeNum - 1;
        while ((activeOverloadNodePointers[right] + degree[overloadNodeList[right]] -
                activeOverloadNodePointers[left]) >
               partOverloadSize) {

            //cout << "left " << left << " right " << right << endl;
            //cout << "activeOverloadNodePointers[right] + degree[overloadNodeList[right]] "<< activeOverloadNodePointers[right] + degree[overloadNodeList[right]] <<" activeOverloadNodePointers[left] " << activeOverloadNodePointers[left] << endl;

            SIZE_TYPE start = left;
            SIZE_TYPE end = right;
            SIZE_TYPE mid;
            while (start <= end) {
                mid = (start + end) / 2;
                EDGE_POINTER_TYPE headDistance = activeOverloadNodePointers[mid] - activeOverloadNodePointers[left];
                EDGE_POINTER_TYPE tailDistance =
                        activeOverloadNodePointers[mid] + degree[overloadNodeList[mid]] -
                        activeOverloadNodePointers[left];
                if (headDistance <= partOverloadSize && tailDistance > partOverloadSize) {
                    //cout << "left " << left << " mid " << mid << endl;
                    //cout << "activeOverloadNodePointers[mid] "<< activeOverloadNodePointers[mid] <<" activeOverloadNodePointers[left] " << activeOverloadNodePointers[left] << endl;

                    break;
                } else if (tailDistance <= partOverloadSize) {
                    start = mid + 1;
                } else if (headDistance > partOverloadSize) {
                    end = mid - 1;
                }
            }
            
            PartEdgeListInfo info;
            info.partActiveNodeNums = mid - left;
            info.partEdgeNums = activeOverloadNodePointers[mid] - activeOverloadNodePointers[left];
            info.partStartIndex = left;
            partEdgeListInfoArr.push_back(info);
            left = mid;
            //cout << "left " << left << " right " << right << endl;
            //cout << "activeOverloadNodePointers[right] + degree[overloadNodeList[right]] "<< activeOverloadNodePointers[right] + degree[overloadNodeList[right]] <<" activeOverloadNodePointers[left] " << activeOverloadNodePointers[left] << endl;

        }

        //cout << "left " << left << " right " << right << endl;
        //cout << "activeOverloadNodePointers[right] + degree[overloadNodeList[right]] "<< activeOverloadNodePointers[right] + degree[overloadNodeList[right]] <<" activeOverloadNodePointers[left] " << activeOverloadNodePointers[left] << endl;


        PartEdgeListInfo info;
        info.partActiveNodeNums = right - left + 1;
        info.partEdgeNums =
                activeOverloadNodePointers[right] + degree[overloadNodeList[right]] - activeOverloadNodePointers[left];
        info.partStartIndex = left;
        partEdgeListInfoArr.push_back(info);
    } else {
        PartEdgeListInfo info;
        info.partActiveNodeNums = overloadNodeNum;
        info.partEdgeNums = overloadEdgeNum;
        info.partStartIndex = 0;
        partEdgeListInfoArr.push_back(info);
    }
}
template<class EdgeType>
void GraphMeta<EdgeType>::writevalue(string filename){
    std::ofstream outfile(filename);
    unsigned int num = 0;
    switch (algType){
        case PR:
        cudaMemcpy(valuePr,valuePrD,vertexArrSize*sizeof(float),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        
        for(unsigned long long i=0;i<vertexArrSize;i++){
            if(valuePr[i]==0.15f)
            num++;
            outfile << valuePr[i] <<std::endl;
        }
        cout<<"0.15 num: "<<num<<endl;
        break;
        default:
        cudaMemcpy(value,valueD,vertexArrSize*sizeof(SIZE_TYPE),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        for(unsigned long long i=0;i<vertexArrSize;i++){
            outfile << value[i] <<std::endl;
        }
        break;
    }
    outfile.close();
    
}
#endif //PTGRAPH_GRAPHMETA_CUH
