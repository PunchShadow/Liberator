//
// Created by gxl on 2021/1/5.
//

#ifndef PTGRAPH_CC_CUH
#define PTGRAPH_CC_CUH
#include "common.cuh"
void conventionParticipateCC(string ccPath);
void ccShare(string ccPath);
void ccOpt(string ccPath, float adviseK);
void ccOptSwap();
long ccCaculateInShare(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList);
long ccCaculateCommonMemoryInnerAsync(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, float adviseK);
void conventionParticipateCCInLong();
long ccCaculateCommonMemoryInnerAsyncRecordVisit(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList,
                                      float adviseK);
long ccCaculateInShareTrace(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList);
void ccShareTrace(string ccPath);
long ccCaculateCommonMemoryInnerAsyncRandom(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList,
                                            float adviseK);
#endif //PTGRAPH_CC_CUH
