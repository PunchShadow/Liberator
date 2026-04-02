#ifndef PTGRAPH_COMMON_CUH
#define PTGRAPH_COMMON_CUH

#include <iostream>
#include <chrono>
#include <fstream>
#include <math.h>
#include "gpu_kernels.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <algorithm>
#include <thread>
#include "ArgumentParser.cuh"

using namespace std;
#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

const static SIZE_TYPE fragment_size = 4096;
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007-04/output.txt";
const static string converPath = "/home/gxl/labproject/subway/uk-2007-04/uk-2007-04.bcsr";
const static string testGraphPath = "/home/gxl/dataset/friendster/friendster.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007Restruct.bcsr";
//const static string testGraphPath = "/home/gxl/labproject/subway/friendster.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/friendsterRestruct.bcsr";
const static string testWeightGraphPath = "/home/gxl/labproject/subway/sk-2005.bwcsr";
const static string randomDataPath = "/home/gxl/labproject/subway/friendsterChange.random";
const static string prGraphPath = "/home/gxl/dataset/friendster/friendster.bcsc";
const static string ssspGraphPath = "/home/gxl/dataset/friendster/friendster.bwcsr";

const static SIZE_TYPE DIST_INFINITY = std::numeric_limits<unsigned long long>::max() - 1;
const static SIZE_TYPE trunk_size = 1 << 24;

struct CommonPartitionInfo {
    SIZE_TYPE startVertex;
    SIZE_TYPE endVertex;
    SIZE_TYPE nodePointerOffset;
    SIZE_TYPE partitionEdgeSize;
};
struct PartEdgeListInfo {
    SIZE_TYPE partActiveNodeNums;
    SIZE_TYPE partEdgeNums;
    SIZE_TYPE partStartIndex;
};

void
checkNeedTransferPartition(bool *needTransferPartition, CommonPartitionInfo *partitionInfoList, bool *isActiveNodeList,
                           int partitionNum, SIZE_TYPE testNumNodes, SIZE_TYPE &activeNum);

void checkNeedTransferPartitionOpt(bool *needTransferPartition, CommonPartitionInfo *partitionInfoList,
                                   bool *isActiveNodeList,
                                   int partitionNum, SIZE_TYPE testNumNodes, SIZE_TYPE &activeNum);


void getMaxPartitionSize(unsigned long &max_partition_size, unsigned long &totalSize, SIZE_TYPE testNumNodes, float param,
                         int edgeSize, int nodeParamSize = 15);


void getMaxPartitionSize(unsigned long &max_partition_size, unsigned long &totalSize, SIZE_TYPE testNumNodes, float param,
                         int edgeSize, SIZE_TYPE edgeListSize, int nodeParamsSize = 15);

void caculatePartInfoForEdgeList(SIZE_TYPE *overloadNodePointers, SIZE_TYPE *overloadNodeList, SIZE_TYPE *degree,
                                 vector<PartEdgeListInfo> &partEdgeListInfoArr, SIZE_TYPE overloadNodeNum,
                                 SIZE_TYPE overloadMemorySize, SIZE_TYPE overloadEdgeNum);

static void fillDynamic(int tId,
                        int numThreads,
                        SIZE_TYPE overloadNodeBegin,
                        SIZE_TYPE numActiveNodes,
                        SIZE_TYPE *outDegree,
                        SIZE_TYPE *activeNodesPointer,
                        SIZE_TYPE *nodePointer,
                        SIZE_TYPE *activeNodes,
                        SIZE_TYPE *edgeListOverload,
                        SIZE_TYPE *edgeList) {
    float waitToHandleNum = numActiveNodes - overloadNodeBegin;
    float numThreadsF = numThreads;
    SIZE_TYPE chunkSize = ceil(waitToHandleNum / numThreadsF);
    SIZE_TYPE left, right;
    left = tId * chunkSize + overloadNodeBegin;
    right = min(left + chunkSize, numActiveNodes);
    SIZE_TYPE thisNode;
    SIZE_TYPE thisDegree;
    SIZE_TYPE fromHere;
    SIZE_TYPE fromThere;
    for (SIZE_TYPE i = left; i < right; i++) {
        thisNode = activeNodes[i];
        thisDegree = outDegree[thisNode];
        fromHere = activeNodesPointer[i];
        fromThere = nodePointer[thisNode];
        for (SIZE_TYPE j = 0; j < thisDegree; j++) {
            edgeListOverload[fromHere + j] = edgeList[fromThere + j];
        }
    }
}


static void writeTrunkVistInIteration(vector<vector<SIZE_TYPE>> recordData, const string& outputPath) {
    ofstream fout(outputPath);
    for (int i = 0; i < recordData.size(); i++) {
        // output by iteration
        for (int j = 0; j < recordData[i].size(); j++) {
            fout << recordData[i][j] << "\t";
        }
        fout << endl;
    }
    fout.close();
}

static vector<SIZE_TYPE> countDataByIteration(SIZE_TYPE edgeListSize, SIZE_TYPE nodeListSize, SIZE_TYPE* nodePointers, SIZE_TYPE* degree, int *isActive) {
    SIZE_TYPE partSizeCursor = 0;
    SIZE_TYPE partSize = trunk_size / sizeof(SIZE_TYPE);
    SIZE_TYPE partNum = edgeListSize / partSize;
    vector<SIZE_TYPE> thisIterationVisit(partNum + 1);
    for (SIZE_TYPE i = 0; i < nodeListSize; i++) {
        SIZE_TYPE edgeStartIndex = nodePointers[i];
        SIZE_TYPE edgeEndIndex = nodePointers[i] + degree[i];
        SIZE_TYPE maxPartIndex = partSizeCursor * partSize + partSize;

        if (edgeStartIndex < maxPartIndex && edgeEndIndex < maxPartIndex) {
            if(isActive[i]) thisIterationVisit[partSizeCursor] += degree[i];
        } else if (edgeStartIndex < maxPartIndex && edgeEndIndex >= maxPartIndex) {
            if(isActive[i]) thisIterationVisit[partSizeCursor] += (maxPartIndex - edgeStartIndex);
            partSizeCursor += 1;
            if(isActive[i]) thisIterationVisit[partSizeCursor] += (edgeEndIndex - maxPartIndex);
        } else {
            partSizeCursor += 1;
            if(isActive[i]) thisIterationVisit[partSizeCursor] += degree[i];
        }
    }
    return thisIterationVisit;
}

static vector<SIZE_TYPE> countDataByIteration(SIZE_TYPE edgeListSize, SIZE_TYPE nodeListSize, SIZE_TYPE* nodePointers, SIZE_TYPE* degree, SIZE_TYPE *isActive) {
    SIZE_TYPE partSizeCursor = 0;
    SIZE_TYPE partSize = trunk_size / sizeof(SIZE_TYPE);
    SIZE_TYPE partNum = edgeListSize / partSize;
    vector<SIZE_TYPE> thisIterationVisit(partNum + 1);
    for (SIZE_TYPE i = 0; i < nodeListSize; i++) {
        SIZE_TYPE edgeStartIndex = nodePointers[i];
        SIZE_TYPE edgeEndIndex = nodePointers[i] + degree[i];
        SIZE_TYPE maxPartIndex = partSizeCursor * partSize + partSize;

        if (edgeStartIndex < maxPartIndex && edgeEndIndex < maxPartIndex) {
            if(isActive[i]) thisIterationVisit[partSizeCursor] += degree[i];
        } else if (edgeStartIndex < maxPartIndex && edgeEndIndex >= maxPartIndex) {
            if(isActive[i]) thisIterationVisit[partSizeCursor] += (maxPartIndex - edgeStartIndex);
            partSizeCursor += 1;
            if(isActive[i]) thisIterationVisit[partSizeCursor] += (edgeEndIndex - maxPartIndex);
        } else {
            partSizeCursor += 1;
            if(isActive[i]) thisIterationVisit[partSizeCursor] += degree[i];
        }
    }
    return thisIterationVisit;
}

static void calculateDegree(SIZE_TYPE nodesSize, SIZE_TYPE* nodePointers, SIZE_TYPE edgesSize, SIZE_TYPE* degree) {
    for (SIZE_TYPE i = 0; i < nodesSize - 1; i++) {
        if (nodePointers[i] > edgesSize) {
            cout << i << "   " << nodePointers[i] << endl;
            break;
        }
        degree[i] = nodePointers[i + 1] - nodePointers[i];
    }
    degree[nodesSize - 1] = edgesSize - nodePointers[nodesSize - 1];
}



#endif //PTGRAPH_COMMON_CUH