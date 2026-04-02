#ifndef GLOBALS_CUH
#define GLOBALS_CUH
#include <cstdint>
#include <climits>
enum ALG_TYPE {
    BFS, SSSP, CC, PR
};
typedef unsigned long long SIZE_TYPE;
typedef unsigned long long EDGE_POINTER_TYPE;

//typedef uint EDGE_POINTER_TYPE;

struct EdgeWithWeight {
    unsigned long long toNode;
    unsigned long long weight;
};
struct FragmentData {
    unsigned long long startVertex = UINT64_MAX - 1;
    unsigned long long vertexNum = 0;
    bool isIn = false;
    bool isVisit = false;
    FragmentData() {
        startVertex = UINT64_MAX - 1;
        vertexNum = 0;
        isIn = false;
        isVisit = false;
    }
};

#endif