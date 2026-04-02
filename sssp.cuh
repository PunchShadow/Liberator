//
// Created by gxl on 2021/1/6.
//

#ifndef PTGRAPH_SSSP_CUH
#define PTGRAPH_SSSP_CUH
#include "common.cuh"
void conventionParticipateSSSP(SIZE_TYPE sourceNodeSample, string ssspPath);
void ssspShare(SIZE_TYPE sourceNodeSample, string ssspPath);
void ssspOpt(SIZE_TYPE sourceNodeSample, string ssspPath, float adviseK);
void ssspOptSwap();
long ssspCaculateInShare(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, EdgeWithWeight *edgeList,
                         SIZE_TYPE sourceNode);
long
ssspCaculateCommonMemoryInnerAsync(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, EdgeWithWeight *edgeList,
                                   SIZE_TYPE sourceNode, float adviseK);

long
ssspCaculateCommonMemoryInnerAsyncVisitRecord(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, EdgeWithWeight *edgeList,
                                              SIZE_TYPE sourceNode, float adviseK);
long
ssspCaculateUVM(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, EdgeWithWeight *edgeList,
                SIZE_TYPE sourceNode);
void ssspShareTrace(SIZE_TYPE sourceNodeSample, string ssspPath);
long ssspCaculateInShareTrace(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, EdgeWithWeight *edgeList,
                              SIZE_TYPE sourceNode);
long
ssspCaculateCommonMemoryInnerAsyncRandom(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, EdgeWithWeight *edgeList,
                                         SIZE_TYPE sourceNode, float adviseK);
#endif //PTGRAPH_SSSP_CUH
