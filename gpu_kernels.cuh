#ifndef GPU_KERNELS_CUH
#define GPU_KERNELS_CUH

#include "range.cuh"
#include "globals.cuh"
#include<device_launch_parameters.h>
#include<cuda.h>
#include<cuda_runtime.h>
using namespace util::lang;

// type alias to simplify typing...
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template<typename T>
__device__
step_range<T> grid_stride_range(T begin, T end) {
    begin += blockDim.x * blockIdx.x + threadIdx.x;
    //return range(begin, end).step(32);
    return range(begin, end).step(gridDim.x * blockDim.x);
}

template<typename Predicate>
__device__ void streamVertices(SIZE_TYPE vertices, Predicate p) {
    for (auto i : grid_stride_range(SIZE_TYPE(0), vertices)) {
        p(i);
    }
}

SIZE_TYPE reduceBool(SIZE_TYPE* resultD, bool* isActiveD, SIZE_TYPE vertexSize, dim3 grid, dim3 block);
__device__ void reduceStreamVertices(SIZE_TYPE vertices, bool *rawData, SIZE_TYPE *result);
__global__ void reduceByBool(SIZE_TYPE vertexSize, bool *rawData, SIZE_TYPE *result);

template <int blockSize> __global__ void reduceResult(SIZE_TYPE *result);
__global__ void
bfs_kernel(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
           bool *labelD);

__global__ void
cc_kernel(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
          bool *labelD);

template <typename T>
__global__ void
sssp_kernel(SIZE_TYPE activeNum, const SIZE_TYPE *activeNodesD, const EDGE_POINTER_TYPE *nodePointersD, const SIZE_TYPE *degreeD, EdgeWithWeight *edgeListD,
            SIZE_TYPE *valueD,
            T *labelD);

template <typename T, typename E>
__global__ void
sssp_kernelDynamic(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                   const SIZE_TYPE *degreeD,
                   SIZE_TYPE *valueD,
                   T *isActiveD, const EdgeWithWeight *edgeListOverloadD,
                   const E *activeOverloadNodePointersD);

__global__ void
bfs_kernelOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
              bool *labelD, SIZE_TYPE overloadNode, SIZE_TYPE *overloadEdgeListD,
              SIZE_TYPE *nodePointersOverloadD);

__global__ void
bfs_kernelStatic2Label(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                       SIZE_TYPE *valueD,
                       SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2);

__global__ void
bfs_kernelDynamic2Label(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                        const SIZE_TYPE *degreeD,
                        SIZE_TYPE *valueD,
                        SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2, const SIZE_TYPE *edgeListOverloadD,
                        const SIZE_TYPE *activeOverloadNodePointersD);

template<typename T, typename E>
__global__ void
bfs_kernelDynamicPart(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                      const SIZE_TYPE *degreeD,
                      SIZE_TYPE *valueD,
                      T *isActiveD, const SIZE_TYPE *edgeListOverloadD,
                      const E *activeOverloadNodePointersD);

__global__ void
setStaticAndOverloadNodePointer(SIZE_TYPE vertexNum, SIZE_TYPE *staticNodes, SIZE_TYPE *overloadNodes, SIZE_TYPE *overloadNodePointers,
                                SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel,
                                SIZE_TYPE *staticPrefix, SIZE_TYPE *overloadPrefix, SIZE_TYPE *degreeD);

__global__ void
sssp_kernelStaticSwapOpt2Label(SIZE_TYPE activeNodesNum, const SIZE_TYPE *activeNodeListD,
                               const SIZE_TYPE *staticNodePointerD, const SIZE_TYPE *degreeD,
                               EdgeWithWeight *edgeListD, SIZE_TYPE *valueD, SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2,
                               bool *isFinish);

__global__ void
sssp_kernelDynamicSwap2Label(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                             const SIZE_TYPE *degreeD,
                             SIZE_TYPE *valueD,
                             SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2, const EdgeWithWeight *edgeListOverloadD,
                             const SIZE_TYPE *activeOverloadNodePointersD, bool *finished);

template<class T>
__global__ void
bfs_kernelStatic(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, EDGE_POINTER_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
                 T *labelD, T* isInStaticD);

__global__ void
bfs_kernelStaticSwap(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                     SIZE_TYPE *valueD,
                     SIZE_TYPE *labelD, bool *isInD);

__global__ void
bfs_kernelStaticSwap(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                     SIZE_TYPE *valueD,
                     SIZE_TYPE *labelD, bool *isInD, SIZE_TYPE *fragmentRecordsD, SIZE_TYPE fragment_size);

__global__ void
bfs_kernelStaticSwap(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                     SIZE_TYPE *valueD,
                     SIZE_TYPE *labelD, SIZE_TYPE *fragmentRecordsD, SIZE_TYPE fragment_size);

__global__ void
bfs_kernelStaticSwap(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                     SIZE_TYPE *valueD,
                     SIZE_TYPE *labelD, SIZE_TYPE *fragmentRecordsD, SIZE_TYPE fragment_size, SIZE_TYPE maxpartionSize, SIZE_TYPE testNumNodes);

__global__ void
bfs_kernelDynamic(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *degreeD, SIZE_TYPE *valueD,
                  SIZE_TYPE *labelD, SIZE_TYPE overloadNode, SIZE_TYPE *overloadEdgeListD,
                  SIZE_TYPE *nodePointersOverloadD);

__global__ void
bfs_kernelDynamicSwap(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *degreeD, SIZE_TYPE *valueD,
                      SIZE_TYPE *labelD, SIZE_TYPE *overloadEdgeListD,
                      SIZE_TYPE *nodePointersOverloadD);

__global__ void
sssp_kernelOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, EdgeWithWeight *edgeListD,
               SIZE_TYPE *valueD,
               bool *labelD, SIZE_TYPE overloadNode, EdgeWithWeight *overloadEdgeListD, SIZE_TYPE *nodePointersOverloadD);
__global__ void
sssp_kernelDynamicUvm(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, EdgeWithWeight *edgeListD,
                      SIZE_TYPE *valueD,
                      SIZE_TYPE *labelD1, SIZE_TYPE *labelD2);

__global__ void
cc_kernelOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
             bool *labelD, SIZE_TYPE overloadNode, SIZE_TYPE *overloadEdgeListD, SIZE_TYPE *nodePointersOverloadD);

template <typename T, typename E>
__global__ void
cc_kernelDynamicSwap(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD, const SIZE_TYPE *degreeD,
                     SIZE_TYPE *valueD,
                     T *isActiveD, const SIZE_TYPE *edgeListOverloadD,
                     const E *activeOverloadNodePointersD);

__global__ void
cc_kernelDynamicSwap2Label(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                           const SIZE_TYPE *degreeD,
                           SIZE_TYPE *valueD,
                           SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2, const SIZE_TYPE *edgeListOverloadD,
                           const SIZE_TYPE *activeOverloadNodePointersD, bool *finished);

__global__ void
cc_kernelDynamicAsync(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD, const SIZE_TYPE *degreeD,
                      SIZE_TYPE *valueD, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2, const SIZE_TYPE *edgeListOverloadD,
                      const SIZE_TYPE *activeOverloadNodePointersD, bool *finished);

template <typename T>
__global__ void
cc_kernelStaticSwap(SIZE_TYPE activeNodesNum, SIZE_TYPE *activeNodeListD,
                    EDGE_POINTER_TYPE *staticNodePointerD, SIZE_TYPE *degreeD,
                    SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, T *isActiveD, bool *isInStaticD);

__global__ void
cc_kernelStaticAsync(SIZE_TYPE activeNodesNum, const SIZE_TYPE *activeNodeListD,
                     const SIZE_TYPE *staticNodePointerD, const SIZE_TYPE *degreeD,
                     const SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2, const bool *isInStaticD,
                     bool *finished, int *atomicValue);

__global__ void
bfs_kernelOptOfSorted(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                      SIZE_TYPE *edgeListOverload, SIZE_TYPE *valueD, bool *labelD, bool *isInListD, SIZE_TYPE *nodePointersOverloadD);

__global__ void
bfs_kernelShareOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                   SIZE_TYPE *edgeListShare, SIZE_TYPE *valueD, bool *labelD, SIZE_TYPE overloadNode);

__global__ void
setLabelDefault(SIZE_TYPE activeNum, SIZE_TYPE *activeNodes, bool *labelD);

template<class T>
__global__ void
setLabelDefaultOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodes, T *labelD);

__global__ void
mixStaticLabel(SIZE_TYPE activeNum, SIZE_TYPE *activeNodes, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2, bool *isInD);

__global__ void
mixDynamicPartLabel(SIZE_TYPE overloadPartNodeNum, SIZE_TYPE startIndex, const SIZE_TYPE *overloadNodes, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2);

__global__ void
setDynamicPartLabelTrue(SIZE_TYPE overloadPartNodeNum, SIZE_TYPE startIndex, const SIZE_TYPE *overloadNodes, SIZE_TYPE *labelD1,
                        SIZE_TYPE *labelD2);

__global__ void
mixCommonLabel(SIZE_TYPE testNodeNum, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2);

__global__ void
cleanStaticAndOverloadLabel(SIZE_TYPE vertexNum, SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel);

__global__ void
setStaticAndOverloadLabel(SIZE_TYPE vertexNum, SIZE_TYPE *activeLabel, SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel, bool *isInD);

__global__ void
setStaticAndOverloadLabelBool(SIZE_TYPE vertexNum, bool *activeLabel, bool *staticLabel, bool *overloadLabel, bool *isInD);

__global__ void
setStaticAndOverloadLabel4Pr(SIZE_TYPE vertexNum, SIZE_TYPE *activeLabel, SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel, bool *isInD,
                             SIZE_TYPE *fragmentRecordD, SIZE_TYPE *nodePointersD, SIZE_TYPE fragment_size, SIZE_TYPE *degreeD,
                             bool *isFragmentActiveD);

__global__ void
setOverloadActiveNodeArray(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *overloadLabel,
                           SIZE_TYPE *activeLabelPrefix);
template <typename T>
__global__ void
setStaticActiveNodeArray(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, T *staticLabel,
                         SIZE_TYPE *activeLabelPrefix);

__global__ void
cc_kernelStaticSwapOpt(SIZE_TYPE activeNodesNum, SIZE_TYPE *activeNodeListD,
                       SIZE_TYPE *staticNodePointerD, SIZE_TYPE *degreeD,
                       SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, SIZE_TYPE *isActiveD);

__global__ void
cc_kernelStaticSwapOpt2Label(SIZE_TYPE activeNodesNum, SIZE_TYPE *activeNodeListD,
                             SIZE_TYPE *staticNodePointerD, SIZE_TYPE *degreeD,
                             SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2, bool *isFinish);

__global__ void
setLabeling(SIZE_TYPE vertexNum, bool *labelD, SIZE_TYPE *labelingD);

__global__ void
setActiveNodeArray(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, bool *activeLabel, SIZE_TYPE *activeLabelPrefix);

__global__ void
setActiveNodeArrayAndNodePointer(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *activeNodePointers, bool *activeLabel,
                                 SIZE_TYPE *activeLabelPrefix, SIZE_TYPE overloadVertex, SIZE_TYPE *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerBySortOpt(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *activeOverloadDegree,
                                          bool *activeLabel, SIZE_TYPE *activeLabelPrefix, bool *isInList, SIZE_TYPE *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerOpt(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *activeNodePointers, SIZE_TYPE *activeLabel,
                                    SIZE_TYPE *activeLabelPrefix, SIZE_TYPE overloadVertex, SIZE_TYPE *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerSwap(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *activeLabel,
                                     SIZE_TYPE *activeLabelPrefix, bool *isInD);

template <class T, typename E>
__global__ void
setOverloadNodePointerSwap(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, E *activeNodePointers, T *activeLabel,
                           SIZE_TYPE *activeLabelPrefix, SIZE_TYPE *degreeD);

__global__ void
setFragmentData(SIZE_TYPE activeNodeNum, SIZE_TYPE *activeNodeList, SIZE_TYPE *staticNodePointers, SIZE_TYPE *staticFragmentData,
                SIZE_TYPE staticFragmentNum, SIZE_TYPE fragmentSize, bool *isInStatic);

__global__ void
setStaticFragmentData(SIZE_TYPE staticFragmentNum, SIZE_TYPE *canSwapFragmentD, SIZE_TYPE *canSwapFragmentPrefixD,
                      SIZE_TYPE *staticFragmentDataD);

__global__ void
setFragmentDataOpt(SIZE_TYPE *staticFragmentData, SIZE_TYPE staticFragmentNum, SIZE_TYPE *staticFragmentVisitRecordsD);

__global__ void
recordFragmentVisit(SIZE_TYPE *activeNodeListD, SIZE_TYPE activeNodeNum, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE fragment_size,
                    SIZE_TYPE *fragmentRecordsD);

__global__ void
bfsKernel_CommonPartition(SIZE_TYPE startVertex, SIZE_TYPE endVertex, SIZE_TYPE offset, const bool *isActiveNodeListD,
                          const SIZE_TYPE *nodePointersD,
                          const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, SIZE_TYPE *valueD, bool *nextActiveNodeListD);

__global__ void
prSumKernel_CommonPartition(SIZE_TYPE startVertex, SIZE_TYPE endVertex, SIZE_TYPE offset, const bool *isActiveNodeListD,
                            const SIZE_TYPE *nodePointersD,
                            const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const float *valueD,
                            float *sumD);

__global__ void
prKernel_CommonPartition(SIZE_TYPE nodeNum, float *valueD, float *sumD, bool *isActiveNodeList);

__global__ void
prSumKernel_UVM(SIZE_TYPE vertexNum, const int *isActiveNodeListD, const SIZE_TYPE *nodePointersD,
                const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, const float *valueD, float *sumD);

__global__ void
prKernel_UVM(SIZE_TYPE nodeNum, float *valueD, float *sumD, int *isActiveListD);

__global__ void
prSumKernel_UVM_Out(SIZE_TYPE vertexNum, int *isActiveNodeListD, const SIZE_TYPE *nodePointersD,
                    const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, float *valueD);

__global__ void
prKernel_UVM_outDegree(SIZE_TYPE nodeNum, float *valueD, float *sumD, int *isActiveListD);

template<typename T, typename E>
__global__ void
prSumKernel_static(SIZE_TYPE activeNum, const SIZE_TYPE *activeNodeList,
                   const EDGE_POINTER_TYPE *nodePointersD,
                   const E *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const T *valueD,
                   T *sumD);

template <typename E, typename K>
__global__ void
prSumKernel_dynamic(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                    const unsigned long long *nodePointersD,
                    const E *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const K *valueD,
                    K *sumD);

template <typename T, typename K>
__global__ void prKernel_Opt(SIZE_TYPE nodeNum, K *valueD, K *sumD, T *isActiveNodeList, K *diffD);

__global__ void
setFragmentDataOpt4Pr(SIZE_TYPE *staticFragmentData, SIZE_TYPE fragmentNum, SIZE_TYPE *fragmentVisitRecordsD,
                      bool *isActiveFragmentD, SIZE_TYPE* fragmentNormalMap2StaticD, SIZE_TYPE maxStaticFragment);

__global__ void
ccKernel_CommonPartition(SIZE_TYPE startVertex, SIZE_TYPE endVertex, SIZE_TYPE offset, const bool *isActiveNodeListD,
                         const SIZE_TYPE *nodePointersD,
                         const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, SIZE_TYPE *valueD, bool *nextActiveNodeListD);

__global__ void
ssspKernel_CommonPartition(SIZE_TYPE startVertex, SIZE_TYPE endVertex, SIZE_TYPE offset, const bool *isActiveNodeListD,
                           const SIZE_TYPE *nodePointersD,
                           const EdgeWithWeight *edgeListD, const SIZE_TYPE *degreeD, SIZE_TYPE *valueD,
                           bool *nextActiveNodeListD);
__global__ void
setStaticAndOverloadLabelAndRecord(SIZE_TYPE vertexNum, SIZE_TYPE *activeLabel, SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel,
                                   bool *isInD, SIZE_TYPE *vertexVisitRecordD);

template<int NT>
__device__ int reduceInWarp(int idInWarp, bool data);

template <typename T>
__global__ void
setActiveNodeList(SIZE_TYPE vertexnum, bool* activeLabel,T* activeNodes, T *activeLabelPrefix);

template <typename T, typename E>
__global__ void
NEW_cc_kernelDynamicSwap_test(SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD, const SIZE_TYPE *degreeD,
                     SIZE_TYPE *valueD, SIZE_TYPE numwarps,
                     T *isActiveD,
                     const SIZE_TYPE *edgeList, const E *NodePointers);

template<typename E>
__global__ void
scanstaticmem(SIZE_TYPE nodeNum, E* staticEdgelistD){
    printf("start scan static mem\n");
    for(SIZE_TYPE i=0;i<nodeNum;i++){
        E temp = staticEdgelistD[i];
        temp++;
    }
}



template<typename T, typename E>
__global__ void
bfs_kernelDynamicPart(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                      const SIZE_TYPE *degreeD,
                      SIZE_TYPE *valueD,
                      T *isActiveD, const SIZE_TYPE *edgeListOverloadD,
                      const E *activeOverloadNodePointersD) {
    streamVertices(overloadNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE traverseIndex = overloadStartNode + index;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue = sourceValue + 1;
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            SIZE_TYPE vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                              activeOverloadNodePointersD[overloadStartNode] + i];

            //printf("source node %d dest node %d set true finalValue %d valueD[vertexId] %d\n", id, vertexId, finalValue, valueD[vertexId]);

            if (finalValue < valueD[vertexId]) {
                //printf("source node %d dest node %d set true finalValue %d valueD[vertexId] %d\n", id, vertexId, finalValue, valueD[vertexId]);
                isActiveD[vertexId] = 1;
                atomicMin(&valueD[vertexId], finalValue);
            } else {
                //printf("source node %d dest node %d set false\n", id, vertexId);
                //isActiveD[vertexId] = 0;
            }
        }
    });
}
//try vertex-to-warp mapping for overload
#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32
#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)
#define MEM_ALIGN (~(0xfULL))
//#define MEM_ALIGN (~(0x1fULL))
template<typename T, typename E>
__global__ void
bfs_kernelDynamicPart_test(SIZE_TYPE overloadNodeNum,SIZE_TYPE numwarps, const SIZE_TYPE *overloadNodeListD,
    SIZE_TYPE *valueD, const SIZE_TYPE *degreeD,
    T *isActiveD, const SIZE_TYPE *edgeListOverloadD,
    const E *activeOverloadNodePointersD){
    
        const SIZE_TYPE tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
        SIZE_TYPE warpIdx = tid >> WARP_SHIFT;
        const SIZE_TYPE laneIdx = tid & ((1 << WARP_SHIFT) - 1);
        
        uint64_t chunk_size = CHUNK_SIZE;
        
        for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
            const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
            if((chunkIdx + CHUNK_SIZE) > overloadNodeNum) {
                if ( overloadNodeNum > chunkIdx )
                    chunk_size = overloadNodeNum - chunkIdx;
            }
            for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
                //SIZE_TYPE traverseIndex = overloadStartNode + i;
                const uint64_t traverseIndex = i;
                SIZE_TYPE id = overloadNodeListD[traverseIndex];
                SIZE_TYPE sourceValue = valueD[id];
                SIZE_TYPE finalValue = sourceValue + 1;

                const uint64_t start = activeOverloadNodePointersD[id];
                const uint64_t shift_start = start & MEM_ALIGN;
                const uint64_t end = activeOverloadNodePointersD[id]+degreeD[id];
               
                for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                    if (j >= start) {
                        SIZE_TYPE vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] + j];
                        if(finalValue < valueD[vertexId]) {
                            atomicMin(&valueD[vertexId], finalValue);
                            isActiveD[vertexId] = 1;
                        }
                    }
                }
            }
        }
        
}
template<typename T, typename E>
__global__ void
bfs_kernelDynamicPart_test2(SIZE_TYPE overloadNodeNum, SIZE_TYPE numwarps, const SIZE_TYPE *overloadNodeListD,
    SIZE_TYPE *valueD,const SIZE_TYPE *degreeD,
    T *isActiveD, const SIZE_TYPE *edgeListOverloadD,
    const E *activeOverloadNodePointersD){
    
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
        const uint64_t traverseIndex = warpIdx;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue = sourceValue + 1;
        const uint64_t start = activeOverloadNodePointersD[traverseIndex];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = activeOverloadNodePointersD[traverseIndex]+degreeD[id];
        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE){
            finalValue = sourceValue + 1;
            if(i>=start){
                SIZE_TYPE vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] + i];
                if(finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD[vertexId] = 1;
                }
            }
        }
    }
    //__syncthreads();
}

//new bfs kernel for overload
template<typename T, typename E>
__global__ void
New_bfs_kernelDynamicPart(SIZE_TYPE overloadNodeNum, SIZE_TYPE numwarps, const SIZE_TYPE *overloadNodeListD,
    SIZE_TYPE *valueD,const SIZE_TYPE *degreeD,
    T *isActiveD, const SIZE_TYPE *edgeList,
    const E *NodePointersD){
    
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    
    for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
        const uint64_t traverseIndex = warpIdx;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue = sourceValue + 1;
        const uint64_t start = NodePointersD[id];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = NodePointersD[id]+degreeD[id];
        
        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE){
            if(i>=start){
                SIZE_TYPE vertexId = edgeList[i];
                if(finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD[vertexId] = 1;
                }
            }
        }
    }
}



template<class T>
__global__ void
bfs_kernelStatic(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, EDGE_POINTER_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
                 T *labelD, T* isInStaticD) {
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        if(isInStaticD[id]){
            SIZE_TYPE edgeIndex = nodePointersD[id];
            SIZE_TYPE sourceValue = valueD[id];
            SIZE_TYPE finalValue;
            
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                SIZE_TYPE vertexId;
                vertexId = edgeListD[edgeIndex + i];
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    labelD[vertexId] = 1;
                }
            }
        }
    });
}
template<class T>
__global__ void
bfs_kernelStatic_test1(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
                 T *labelD, bool* changed) {
    const SIZE_TYPE tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const SIZE_TYPE warpIdx = tid >> WARP_SHIFT;
    const SIZE_TYPE laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;
    if((chunkIdx + CHUNK_SIZE) > nodeNum) {
        if ( nodeNum > chunkIdx )
            chunk_size = nodeNum - chunkIdx;
        else
            return;
    }
    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        SIZE_TYPE id = activeNodesD[i];
        const uint64_t start = nodePointersD[id];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = nodePointersD[id+1];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
            finalValue = sourceValue + 1;
            if (j >= start) {
                const SIZE_TYPE next = edgeListD[j];
                if(finalValue < valueD[next]) {
                    atomicMin(&valueD[next], finalValue);
                    *changed = true;
                    labelD[next] = 1;
                }
            }
        }
    }
                
}
template<class T>
__global__ void
bfs_kernelStatic_test2(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
                 T *labelD){
    SIZE_TYPE index = blockDim.x * blockIdx.x + threadIdx.x;
    for(index;index<nodeNum;index+=gridDim.x*blockDim.x){
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            SIZE_TYPE vertexId;
            vertexId = edgeListD[edgeIndex + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    }
}
template<class T>
__global__ void
setLabelDefaultOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodes, T *labelD) {
    streamVertices(activeNum, [&](SIZE_TYPE vertexId) {
        if (labelD[activeNodes[vertexId]]) {
            labelD[activeNodes[vertexId]] = 0;
            //printf("vertex%d index %d true to %d \n", vertexId, activeNodes[vertexId], labelD[activeNodes[vertexId]]);
        }
    });
}


template <class T, typename E>
__global__ void
setOverloadNodePointerSwap(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, E *activeNodePointers, T *activeLabel,
                           SIZE_TYPE *activeLabelPrefix, SIZE_TYPE *degreeD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
            activeLabel[vertexId] = 0;
            //if (vertexId == 1) {
            //    printf("activeLabel %d activeLabelPrefix[vertexId] %d degreeD[vertexId] %d\n", activeLabel[vertexId], activeLabelPrefix[vertexId], activeNodePointers[activeLabelPrefix[vertexId]]);
            //}
        }
    });
}

template <typename T>
__global__ void
setActiveNodeList(SIZE_TYPE vertexnum, bool* activeLabel,T* activeNodes, T *activeLabelPrefix){
    streamVertices(vertexnum,[&](SIZE_TYPE vertexId){
        if(activeLabel[vertexId]){
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            activeLabel[vertexId] = 0;
        }
    });
}

template <typename T>
__global__ void
setStaticActiveNodeArray(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, T *staticLabel,
                         SIZE_TYPE *activeLabelPrefix) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (staticLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            staticLabel[vertexId] = 0;
        }
    });
}


template <typename T>
__global__ void
cc_kernelStaticSwap(SIZE_TYPE activeNodesNum, SIZE_TYPE *activeNodeListD,
                    EDGE_POINTER_TYPE *staticNodePointerD, SIZE_TYPE *degreeD,
                    SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, T *isActiveD, bool *isInStaticD) {
    streamVertices(activeNodesNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodeListD[index];
        if (isInStaticD[id]) {
            EDGE_POINTER_TYPE edgeIndex = staticNodePointerD[id];
            SIZE_TYPE sourceValue = valueD[id];
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                SIZE_TYPE vertexId = edgeListD[edgeIndex + i];
                SIZE_TYPE destValue = valueD[vertexId];
                if (sourceValue < destValue) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    isActiveD[vertexId] = 1;
                } else if (destValue < sourceValue) {
                    atomicMin(&valueD[id], destValue);
                    isActiveD[id] = 1;
                }
            }
        }
    });
}


template <typename T, typename E>
__global__ void
cc_kernelDynamicSwap(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD, const SIZE_TYPE *degreeD,
                     SIZE_TYPE *valueD,
                     T *isActiveD, const SIZE_TYPE *edgeListOverloadD,
                     const E *activeOverloadNodePointersD) {
    streamVertices(overloadNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE traverseIndex = overloadStartNode + index;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        SIZE_TYPE sourceValue = valueD[id];
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            SIZE_TYPE vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                              activeOverloadNodePointersD[overloadStartNode] + i];
            SIZE_TYPE destValue = valueD[vertexId];
            if (sourceValue < destValue) {
                atomicMin(&valueD[vertexId], sourceValue);
                isActiveD[vertexId] = 1;
            } else if (destValue < sourceValue) {
                atomicMin(&valueD[id], destValue);
                isActiveD[id] = 1;
            }
        }
    });
}
template <typename T, typename E>
__global__ void
cc_kernelDynamicSwap_test(SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD, const SIZE_TYPE *degreeD,
                     SIZE_TYPE *valueD, SIZE_TYPE numwarps,
                     T *isActiveD, const SIZE_TYPE *edgeListOverloadD,
                     const E *activeOverloadNodePointersD){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
        SIZE_TYPE traverseIndex = warpIdx;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        SIZE_TYPE sourceValue = valueD[id];
        const uint64_t start = activeOverloadNodePointersD[traverseIndex];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = activeOverloadNodePointersD[traverseIndex]+degreeD[id];
        for(SIZE_TYPE i = laneIdx+shift_start;i<end;i+=WARP_SIZE){
            if(i>=start){
                SIZE_TYPE vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] + i];
                SIZE_TYPE destValue = valueD[vertexId];
                if (sourceValue < destValue) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    isActiveD[vertexId] = 1;
                } else if (destValue < sourceValue) {
                    atomicMin(&valueD[id], destValue);
                    isActiveD[id] = 1;
                }
            }
        }
    }
}
template <typename T, typename E>
__global__ void
NEW_cc_kernelDynamicSwap_test(SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD, const SIZE_TYPE *degreeD,
                     SIZE_TYPE *valueD, SIZE_TYPE numwarps,
                     T *isActiveD, const SIZE_TYPE *edgeList,
                     const E *NodePointers){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    
    for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
        SIZE_TYPE traverseIndex = warpIdx;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
            SIZE_TYPE sourceValue = valueD[id];
            const uint64_t start = NodePointers[id];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = start+degreeD[id];

            for(SIZE_TYPE i = laneIdx+shift_start;i<end;i+=WARP_SIZE){
                SIZE_TYPE vertexId = edgeList[i];
                if(i>=start&&i<end){
                    SIZE_TYPE tar_value = valueD[vertexId];
                    if (sourceValue < tar_value) {
                        atomicMin(&valueD[vertexId], sourceValue);
                        isActiveD[vertexId] = 1;
                    } else if (tar_value < sourceValue) {
                        atomicMin(&valueD[id], tar_value);
                        isActiveD[id] = 1;
                    }
                }
            }
    }
}
template <typename T>
__global__ void
sssp_kernel(SIZE_TYPE activeNum, const SIZE_TYPE *activeNodesD, const EDGE_POINTER_TYPE *nodePointersD, const SIZE_TYPE *degreeD, EdgeWithWeight *edgeListD,
            SIZE_TYPE *valueD,
            T *labelD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        EDGE_POINTER_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;

        for (EDGE_POINTER_TYPE i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            finalValue = sourceValue + edgeListD[i].weight;
            SIZE_TYPE vertexId = edgeListD[i].toNode;
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

template <typename T, typename E>
__global__ void
sssp_kernelDynamic(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                   const SIZE_TYPE *degreeD,
                   SIZE_TYPE *valueD,
                   T *isActiveD, const EdgeWithWeight *edgeListOverloadD,
                   const E *activeOverloadNodePointersD) {
    streamVertices(overloadNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE traverseIndex = overloadStartNode + index;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            EdgeWithWeight checkNode{};
            checkNode = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                          activeOverloadNodePointersD[overloadStartNode] + i];
            finalValue = sourceValue + checkNode.weight;
            SIZE_TYPE vertexId = checkNode.toNode;
            if (finalValue < valueD[vertexId]) {
                //printf("source node %d dest node %d set true\n", id, vertexId);
                atomicMin(&valueD[vertexId], finalValue);
                isActiveD[vertexId] = 1;
            }
        }
    });
}
template <typename T, typename E>
__global__ void sssp_kernelDynamic_test(SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                   const SIZE_TYPE *degreeD, SIZE_TYPE numwarps,
                   SIZE_TYPE *valueD,
                   T *isActiveD, const EdgeWithWeight *edgeListOverload,
                   const E *activeOverloadNodePointersD){
    // const uint64_t blockId = blockIdx.y * gridDim.x + blockIdx.x;  
    // const uint64_t tid = blockId * blockDim.x + threadIdx.x;
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    
    uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
        SIZE_TYPE traverseIndex = warpIdx;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        const uint64_t start = activeOverloadNodePointersD[id];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = activeOverloadNodePointersD[id]+degreeD[id];
        for(uint64_t i = laneIdx+shift_start;i<end;i+=WARP_SIZE){
            if(i>=start){
                EdgeWithWeight checkNode{};
                checkNode= edgeListOverload[i];
                finalValue = sourceValue+checkNode.weight;
                SIZE_TYPE vertexId = checkNode.toNode;
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD[vertexId] = 1;
                }
            }
        }
    }

}
template <typename T, typename E>
__global__ void NEW_sssp_kernelDynamic_test(SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                   const SIZE_TYPE *degreeD, SIZE_TYPE numwarps,
                   SIZE_TYPE *valueD,
                   T *isActiveD, const EdgeWithWeight *edgeList,
                   const E *NodePointers){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    //if(warpIdx<overloadNodeNum){
    for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
        SIZE_TYPE id = overloadNodeListD[warpIdx];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        const uint64_t start = NodePointers[id];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = NodePointers[id]+degreeD[id];
        for(uint64_t i = laneIdx+shift_start;i<end;i+=WARP_SIZE){
            EdgeWithWeight checkNode{};
            checkNode= edgeList[i];
            if(i>=start){
                finalValue = sourceValue+checkNode.weight;
                SIZE_TYPE vertexId = checkNode.toNode;
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD[vertexId] = 1;
                }
            }
        }
    }

}

template <typename T, typename K>
__global__ void
prKernel_Opt(SIZE_TYPE nodeNum, K *valueD, K *sumD, T *isActiveNodeList, K *diffD) {
    // Option 1 semantics: every vertex is processed every iteration.
    // The active flag is kept permanently set so setStaticAndOverloadLabelBool
    // buckets every vertex each iter; the main loop terminates on the global
    // max of diffD, not on the active-vertex count.
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        K tempValue = 0.15 + 0.85 * sumD[index];
        K diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);
        valueD[index] = tempValue;
        diffD[index] = diff;
        isActiveNodeList[index] = 1;
        sumD[index] = 0;
    });
}
template <typename T, typename K>
__global__ void
NEW_prKernel_Opt(SIZE_TYPE nodeNum, K *valueD, K *sumD, T *isActiveNodeList){
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        if (isActiveNodeList[index]) {
            //K tempValue = 0.15 + 0.85 * sumD[index];
            K tempValue = 0.15*valueD[index] + 0.85*sumD[index];
            K diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);
            valueD[index] = tempValue;
            if (diff > 0.01) {
                isActiveNodeList[index] = 1;
            } else {
                isActiveNodeList[index] = 0;
            }
            sumD[index] = 0;
        }

    });
}
template <typename E, typename K>
__global__ void
prSumKernel_dynamic(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                    const unsigned long long *nodePointersD,
                    const E *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const K *valueD,
                    K *sumD) {
    streamVertices(overloadNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE traverseIndex = overloadStartNode + index;
        SIZE_TYPE nodeIndex = overloadNodeListD[traverseIndex];

        SIZE_TYPE edgeOffset = nodePointersD[traverseIndex] - nodePointersD[overloadStartNode];
        K tempSum = 0;
        for (SIZE_TYPE i = edgeOffset; i < edgeOffset + degreeD[nodeIndex]; i++) {
            SIZE_TYPE srcNodeIndex = edgeListD[i];
            if(outDegreeD[srcNodeIndex]!=0){
                K tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
            //printf("src %d dest %d value %f \n", srcNodeIndex,nodeIndex )
                tempSum += tempValue;
            }
            // else{
            //     tempSum += 0;
            // }
        }
        sumD[nodeIndex] = tempSum;

    });
}
template <typename E, typename K>
__global__ void
prSumKernel_dynamic_test(SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                    const E *nodePointersD, SIZE_TYPE numwarps,
                    const SIZE_TYPE *edgeList, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const K *valueD,
                    K *sumD){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    K tempSum = 0;
    for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
        SIZE_TYPE traverseIndex = warpIdx;
        SIZE_TYPE nodeIndex = overloadNodeListD[traverseIndex];
        const uint64_t start = nodePointersD[traverseIndex];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = nodePointersD[traverseIndex]+degreeD[nodeIndex];
        for(SIZE_TYPE i = laneIdx+shift_start;i<end;i+=WARP_SIZE){
            if(i>=start){
                SIZE_TYPE srcNodeIndex = edgeList[i];
                if(outDegreeD[srcNodeIndex]!=0){
                    K tempValue = (valueD[srcNodeIndex] / outDegreeD[srcNodeIndex]);
                    tempSum += tempValue;
                }
                else{
                    tempSum+=0;
                }
            }
        }
        sumD[nodeIndex] = tempSum;
    }
}

template <typename E, typename K>
__global__ void
NEW_prSumKernel_dynamic_test(SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                    const EDGE_POINTER_TYPE *nodePointers, SIZE_TYPE numwarps,
                    const E *edgeList, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const K *valueD,
                    K *sumD){
    //uint64_t blockId = blockIdx.y * gridDim.x + blockIdx.x; 
    //uint64_t tid = blockId * blockDim.x + threadIdx.x; 
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    if(warpIdx<overloadNodeNum){
    //for(warpIdx;warpIdx < overloadNodeNum;warpIdx+=numwarps){
        SIZE_TYPE traverseIndex = warpIdx;
        SIZE_TYPE nodeIndex = overloadNodeListD[traverseIndex];
        const uint64_t start = nodePointers[nodeIndex];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = nodePointers[nodeIndex]+degreeD[nodeIndex];
        K tempSum = 0;
        for(SIZE_TYPE i = laneIdx+shift_start;i<end;i+=WARP_SIZE){
            SIZE_TYPE srcNodeIndex = edgeList[i];
            if(i>=start){
                if(outDegreeD[srcNodeIndex]!=0){
                    K tempValue = (valueD[srcNodeIndex] / outDegreeD[srcNodeIndex]);
                    atomicAdd(&sumD[nodeIndex],tempValue);
                    //tempSum += tempValue;
                }
                // else{
                //     atomicAdd(&sumD[nodeIndex],0);
                //     // tempSum+=0;
                // }
            }
        }
        //sumD[nodeIndex] = tempSum;
    //}
    }
}

template<typename T, typename E>
__global__ void
prSumKernel_static(SIZE_TYPE activeNum, const SIZE_TYPE *activeNodeList,
                   const EDGE_POINTER_TYPE *nodePointersD,
                   const E *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const T *valueD,
                   T *sumD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE nodeIndex = activeNodeList[index];
        SIZE_TYPE edgeIndex = nodePointersD[nodeIndex];
        T tempSum = 0;
        for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
            SIZE_TYPE srcNodeIndex = edgeListD[i];
            if(outDegreeD[srcNodeIndex]!=0){
                T tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
                tempSum += tempValue;
            }
            // else{
            //     tempSum+=0;
            // }
        }
        sumD[nodeIndex] = tempSum;
    });
}
template<typename T,typename E>
__global__ void
NEW_prSumKernel_static(SIZE_TYPE activeNum, const SIZE_TYPE *activeNodeList,
                   const SIZE_TYPE *nodePointersD,
                   const E *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const T *valueD,
                   T *sumD, T *Diff, T Add) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE nodeIndex = activeNodeList[index];
        SIZE_TYPE edgeIndex = nodePointersD[nodeIndex];
        T tempSum = 0;
        for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
            SIZE_TYPE srcNodeIndex = edgeListD[i];
            if(outDegreeD[srcNodeIndex]!=0){
                T tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
                tempSum += tempValue;
            }
            else{
                tempSum+=0;
            }
        }
        sumD[nodeIndex] = tempSum;
        sumD[nodeIndex] += Add;
        Diff[nodeIndex] = sumD[nodeIndex] - valueD[nodeIndex];
    });
}
#endif