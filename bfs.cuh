//
// Created by gxl on 2020/12/30.
//

#ifndef PTGRAPH_BFS_CUH
#define PTGRAPH_BFS_CUH

#include "common.cuh"

void conventionParticipateBFS(string bfsPath, int sampleSourceNode);
void bfsShare(string bfsPath, int sampleSourceNode);
void bfsOpt(string bfsPath, int sampleSourceNode, float adviseK);
long bfsCaculateInShare(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode);

long
bfsCaculateInShareReturnValue(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode,
                              SIZE_TYPE **bfsValue, int index);

long
bfsCaculateInAsyncNoUVMSwap(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode);

long
bfsCaculateInAsyncNoUVM(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode, float adviseK);

long
bfsCaculateInAsyncNoUVMVisitRecord(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode,
                                   float adviseK);
long bfsCaculateInShareTrace(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode);
void bfsShareTrace(string bfsPath, int sampleSourceNode);
long
bfsCaculateInAsyncNoUVMRandom(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode,
                              float adviseK);
#endif //PTGRAPH_BFS_CUH
