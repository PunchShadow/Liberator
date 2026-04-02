#include "gpu_kernels.cuh"




__device__ void gpu_sync(int goalVal, volatile int *arrayIn, volatile int *arrayOut) {
    int tidInBlock = blockDim.y * threadIdx.x + threadIdx.y;
    int blockId = blockIdx.x * gridDim.y + blockIdx.y;
    int nBlockNum = gridDim.x * gridDim.y;
    if (tidInBlock == 0) {
        arrayIn[blockId] = goalVal;
    }
    if (blockId == 0) {
        if (tidInBlock < nBlockNum) {
            while (arrayIn[tidInBlock] != goalVal) {
            }
        }
        __syncthreads();
        if (tidInBlock < nBlockNum) {
            arrayOut[tidInBlock] = goalVal;
        }
    }
    if (tidInBlock == 0) {
        while (arrayOut[blockId] != goalVal) {
        }
        if (blockId == 0) {
        }
    }
    __syncthreads();
}
//__device__ vola int g_mutex;
__device__ void gpu_sync(int goalVal, volatile int *g_mutex) {
    int tidInBlock = blockDim.y * threadIdx.x + threadIdx.y;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidInBlock == 0) {
        atomicAdd((int *) g_mutex, 1);
        while ((*g_mutex) != goalVal) {

        }
    }
    __syncthreads();
}

__global__ void
setLabeling(SIZE_TYPE vertexNum, bool *labelD, SIZE_TYPE *labelingD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (labelD[vertexId]) {
            labelingD[vertexId] = 1;
            //printf("vertex[%d] set 1\n", vertexId);
        } else {
            labelingD[vertexId] = 0;
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerOpt(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *activeNodePointers, SIZE_TYPE *activeLabel,
                                    SIZE_TYPE *activeLabelPrefix, SIZE_TYPE overloadVertex, SIZE_TYPE *degreeD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (vertexId > overloadVertex) {
                activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
                activeLabel[vertexId] = 1;
            } else {
                activeNodePointers[activeLabelPrefix[vertexId]] = 0;
                activeLabel[vertexId] = 0;
            }
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerSwap(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *activeLabel,
                                     SIZE_TYPE *activeLabelPrefix, bool *isInD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (!isInD[vertexId]) {
                activeLabel[vertexId] = 1;
            } else {
                activeLabel[vertexId] = 0;
            }
        }
    });
}

__global__ void
setOverloadActiveNodeArray(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *overloadLabel,
                           SIZE_TYPE *activeLabelPrefix) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (overloadLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
        }
    });
}

__global__ void
setStaticAndOverloadLabel(SIZE_TYPE vertexNum, SIZE_TYPE *activeLabel, SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel, bool *isInD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            if (isInD[vertexId]) {
                staticLabel[vertexId] = 1;
            } else {
                overloadLabel[vertexId] = 1;
            }
        }
    });
}

__global__ void
setStaticAndOverloadLabelBool(SIZE_TYPE vertexNum, bool *activeLabel, bool *staticLabel, bool *overloadLabel, bool *isInD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            if (isInD[vertexId]) {
                staticLabel[vertexId] = 1;
            } else {
                overloadLabel[vertexId] = 1;
            }
        }
    });
}

__global__ void
setStaticAndOverloadLabelAndRecord(SIZE_TYPE vertexNum, SIZE_TYPE *activeLabel, SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel,
                                   bool *isInD, SIZE_TYPE *vertexVisitRecordD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            if (isInD[vertexId]) {
                staticLabel[vertexId] = 1;
            } else {
                overloadLabel[vertexId] = 1;
            }
            atomicAdd(&vertexVisitRecordD[vertexId], 1);
        }
    });
}

__global__ void
setStaticAndOverloadLabel4Pr(SIZE_TYPE vertexNum, SIZE_TYPE *activeLabel, SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel, bool *isInD,
                             SIZE_TYPE *fragmentRecordD, SIZE_TYPE *nodePointersD, SIZE_TYPE fragment_size, SIZE_TYPE *degreeD,
                             bool *isFragmentActiveD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            if (isInD[vertexId]) {
                staticLabel[vertexId] = 1;
            } else {
                overloadLabel[vertexId] = 1;
            }
        } else {
            SIZE_TYPE edgeIndex = nodePointersD[vertexId];
            SIZE_TYPE fragmentIndex = edgeIndex / fragment_size;
            SIZE_TYPE fragmentMoreIndex = (edgeIndex + degreeD[vertexId]) / fragment_size;
            if (isFragmentActiveD[fragmentIndex]) {
                if (fragmentMoreIndex > fragmentIndex) {
                    atomicAdd(&fragmentRecordD[fragmentIndex],
                              fragmentIndex * fragment_size + fragment_size - edgeIndex);
                } else {
                    atomicAdd(&fragmentRecordD[fragmentIndex], degreeD[vertexId]);
                }
            }
        }
    });
}

__global__ void
cleanStaticAndOverloadLabel(SIZE_TYPE vertexNum, SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        staticLabel[vertexId] = 0;
        overloadLabel[vertexId] = 0;
    });
}

__global__ void
setStaticAndOverloadNodePointer(SIZE_TYPE vertexNum, SIZE_TYPE *staticNodes, SIZE_TYPE *overloadNodes, SIZE_TYPE *overloadNodeDegrees,
                                SIZE_TYPE *staticLabel, SIZE_TYPE *overloadLabel,
                                SIZE_TYPE *staticPrefix, SIZE_TYPE *overloadPrefix, SIZE_TYPE *degreeD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (overloadLabel[vertexId]) {
            overloadNodes[overloadPrefix[vertexId]] = vertexId;
            overloadNodeDegrees[overloadPrefix[vertexId]] = degreeD[vertexId];
            overloadLabel[vertexId] = 0;
        }
        if (staticLabel[vertexId]) {
            staticNodes[staticPrefix[vertexId]] = vertexId;
            staticLabel[vertexId] = 0;
        }
    });
}

__global__ void
setActiveNodeArray(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, bool *activeLabel,
                   SIZE_TYPE *activeLabelPrefix) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            //printf("activeNodes %d set %d %d \n", activeLabelPrefix[vertexId], vertexId, activeLabel[vertexId]);
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointer(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *activeNodePointers, bool *activeLabel,
                                 SIZE_TYPE *activeLabelPrefix, SIZE_TYPE overloadVertex, SIZE_TYPE *degreeD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (vertexId > overloadVertex) {
                activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
            } else {
                activeNodePointers[activeLabelPrefix[vertexId]] = 0;
            }
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerBySortOpt(SIZE_TYPE vertexNum, SIZE_TYPE *activeNodes, SIZE_TYPE *activeOverloadDegree,
                                          bool *activeLabel, SIZE_TYPE *activeLabelPrefix, bool *isInList, SIZE_TYPE *degreeD) {
    streamVertices(vertexNum, [&](SIZE_TYPE vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (!isInList[vertexId]) {
                activeOverloadDegree[activeLabelPrefix[vertexId]] = degreeD[vertexId];
            } else {
                activeOverloadDegree[activeLabelPrefix[vertexId]] = 0;
            }
        }
    });
}

__global__ void
setLabelDefault(SIZE_TYPE activeNum, SIZE_TYPE *activeNodes, bool *labelD) {
    streamVertices(activeNum, [&](SIZE_TYPE vertexId) {
        if (labelD[activeNodes[vertexId]]) {
            labelD[activeNodes[vertexId]] = 0;
            //printf("vertex%d index %d true to %d \n", vertexId, activeNodes[vertexId], labelD[activeNodes[vertexId]]);
        }
    });
}

__global__ void
mixStaticLabel(SIZE_TYPE activeNum, SIZE_TYPE *activeNodes, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2, bool *isInD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE vertexId = activeNodes[index];
        if (labelD1[vertexId]) {
            labelD1[vertexId] = 0;
        }
        if (isInD[vertexId]) {
            labelD1[vertexId] = 1;
        }
        labelD2[vertexId] = 0;
    });
}

__global__ void
mixDynamicPartLabel(SIZE_TYPE overloadPartNodeNum, SIZE_TYPE startIndex, const SIZE_TYPE *overloadNodes, SIZE_TYPE *labelD1,
                    SIZE_TYPE *labelD2) {
    streamVertices(overloadPartNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE vertexId = overloadNodes[startIndex + index];
        labelD1[vertexId] = labelD1[vertexId] || labelD2[vertexId];
        labelD2[vertexId] = 0;
    });
}

__global__ void
mixCommonLabel(SIZE_TYPE testNodeNum, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2) {
    streamVertices(testNodeNum, [&](SIZE_TYPE vertexId) {
        labelD1[vertexId] = labelD1[vertexId] || labelD2[vertexId];
        labelD2[vertexId] = 0;
    });
}

__global__ void
setDynamicPartLabelTrue(SIZE_TYPE overloadPartNodeNum, SIZE_TYPE startIndex, const SIZE_TYPE *overloadNodes, SIZE_TYPE *labelD1,
                        SIZE_TYPE *labelD2) {
    streamVertices(overloadPartNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE vertexId = overloadNodes[startIndex + index];
        labelD1[vertexId] = true;
        labelD2[vertexId] = false;
    });
}

__global__ void
bfs_kernel(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
           bool *labelD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            if (finalValue < valueD[edgeListD[i]]) {
                atomicMin(&valueD[edgeListD[i]], finalValue);
                labelD[edgeListD[i]] = true;
            }
        }
    });
}

__global__ void
bfsKernel_CommonPartition(SIZE_TYPE startVertex, SIZE_TYPE endVertex, SIZE_TYPE offset, const bool *isActiveNodeListD,
                          const SIZE_TYPE *nodePointersD,
                          const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, SIZE_TYPE *valueD, bool *nextActiveNodeListD) {
    streamVertices(endVertex - startVertex + 1, [&](SIZE_TYPE index) {
        SIZE_TYPE nodeIndex = startVertex + index;
        if (isActiveNodeListD[nodeIndex]) {
            SIZE_TYPE edgeIndex = nodePointersD[nodeIndex] - offset;
            SIZE_TYPE sourceValue = valueD[nodeIndex];
            SIZE_TYPE finalValue;
            //printf("node %d edgeIndex %d sourceValue %d degreeD[nodeIndex] %d\n", nodeIndex, edgeIndex, sourceValue, degreeD[nodeIndex]);
            for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {

                //printf("node %d dest node %d set true \n", nodeIndex, edgeListD[i]);
                finalValue = sourceValue + 1;
                if (finalValue < valueD[edgeListD[i]]) {
                    atomicMin(&valueD[edgeListD[i]], finalValue);
                    nextActiveNodeListD[edgeListD[i]] = true;
                }
            }
        }
    });
}

__global__ void
prSumKernel_CommonPartition(SIZE_TYPE startVertex, SIZE_TYPE endVertex, SIZE_TYPE offset, const bool *isActiveNodeListD,
                            const SIZE_TYPE *nodePointersD,
                            const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, const float *valueD,
                            float *sumD) {
    streamVertices(endVertex - startVertex + 1, [&](SIZE_TYPE index) {
        SIZE_TYPE nodeIndex = startVertex + index;
        if (isActiveNodeListD[nodeIndex]) {
            SIZE_TYPE edgeIndex = nodePointersD[nodeIndex] - offset;
            float tempSum = 0;
            for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                SIZE_TYPE srcNodeIndex = edgeListD[i];
                if (outDegreeD[srcNodeIndex] != 0) {
                    float tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
                    tempSum += tempValue;
                }
            }
            sumD[nodeIndex] = tempSum;
        }
    });
}

__global__ void
prKernel_CommonPartition(SIZE_TYPE nodeNum, float *valueD, float *sumD, bool *isActiveNodeList) {
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        if (isActiveNodeList[index]) {
            float tempValue = 0.15 + 0.85 * sumD[index];
            float diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);
            /*if (index == 1) {
                printf("tempValue %f \n", tempValue);
            }*/
            if (diff > 0.01) {
                isActiveNodeList[index] = true;
                valueD[index] = tempValue;
                sumD[index] = 0;
            } else {
                isActiveNodeList[index] = false;
                sumD[index] = 0;
            }
        }

    });
}


__global__ void
prSumKernel_UVM(SIZE_TYPE vertexNum, const int *isActiveNodeListD, const SIZE_TYPE *nodePointersD,
                const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, const float *valueD, float *sumD) {
    streamVertices(vertexNum, [&](SIZE_TYPE index) {
        SIZE_TYPE nodeIndex = index;
        if (isActiveNodeListD[nodeIndex] > 0) {
            SIZE_TYPE edgeIndex = nodePointersD[nodeIndex];
            float sourceValue = (degreeD[nodeIndex] != 0) ? valueD[nodeIndex] / degreeD[nodeIndex] : 0.0f;
            //printf("node %d edgeIndex %d sourceValue %d degreeD[nodeIndex] %d\n", nodeIndex, edgeIndex, sourceValue, degreeD[nodeIndex]);
            for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {

                //printf("node %d dest node %d set true \n", nodeIndex, edgeListD[i]);
                atomicAdd(&sumD[edgeListD[i]], sourceValue);
            }
        }
    });
}

__global__ void
prSumKernel_UVM_Out(SIZE_TYPE vertexNum, int *isActiveNodeListD, const SIZE_TYPE *nodePointersD,
                    const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, const SIZE_TYPE *outDegreeD, float *valueD) {
    streamVertices(vertexNum, [&](SIZE_TYPE index) {
        SIZE_TYPE nodeIndex = index;
        if (isActiveNodeListD[nodeIndex] > 0) {
            SIZE_TYPE edgeIndex = nodePointersD[nodeIndex];
            float tempSum = 0;
            for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                SIZE_TYPE srcNodeIndex = edgeListD[i];
                if (outDegreeD[srcNodeIndex] != 0) {
                    float tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
                    tempSum += tempValue;
                }
            }

            float tempValue = 0.15 + 0.85 * tempSum;
            float diff =
                    tempValue > valueD[nodeIndex] ? (tempValue - valueD[nodeIndex]) : (valueD[nodeIndex] - tempValue);
            if (diff > 0.01) {
                isActiveNodeListD[nodeIndex] = 1;
                valueD[index] = tempValue;
                //sumD[index] = 0;
            } else {
                isActiveNodeListD[nodeIndex] = 0;
                valueD[index] = tempValue;
                //sumD[index] = 0;
            }

            if (index >= 0 && index <= 10) {
                printf("value %d is %f \n", index, valueD[index]);
            }
        }
    });
}

__global__ void
prKernel_UVM(SIZE_TYPE nodeNum, float *valueD, float *sumD, int *isActiveListD) {
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        float tempValue = 0.15 + 0.85 * sumD[index];
        float diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);
        if (diff > 0.01) {
            isActiveListD[index] = 1;
            valueD[index] = tempValue;
            sumD[index] = 0;
        } else {
            isActiveListD[index] = 1;
            valueD[index] = tempValue;
            sumD[index] = 0;
        }

        if (index >= 0 && index <= 10) {
            printf("value %d is %f \n", index, valueD[index]);
        }
    });
}

__global__ void
prKernel_UVM_outDegree(SIZE_TYPE nodeNum, float *valueD, float *sumD, int *isActiveListD) {
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        float tempValue = 0.15 + 0.85 * sumD[index];
        float diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);
        if (diff > 0.01) {
            isActiveListD[index] = 1;
            valueD[index] = tempValue;
            sumD[index] = 0;
        } else {
            isActiveListD[index] = 0;
            //valueD[index] = tempValue;
            sumD[index] = 0;
        }

        if (index >= 0 && index <= 10) {
            printf("value %d is %f \n", index, valueD[index]);
        }
    });
}

__global__ void
cc_kernel(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
          bool *labelD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE sourceValue = valueD[id];
        for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            SIZE_TYPE destValue = valueD[edgeListD[i]];
            if (sourceValue < destValue) {
                atomicMin(&valueD[edgeListD[i]], sourceValue);
                labelD[edgeListD[i]] = true;
            } else if (destValue < sourceValue) {
                atomicMin(&valueD[id], destValue);
                labelD[id] = true;
            }
        }
    });
}

__global__ void
ccKernel_CommonPartition(SIZE_TYPE startVertex, SIZE_TYPE endVertex, SIZE_TYPE offset, const bool *isActiveNodeListD,
                         const SIZE_TYPE *nodePointersD,
                         const SIZE_TYPE *edgeListD, const SIZE_TYPE *degreeD, SIZE_TYPE *valueD, bool *nextActiveNodeListD) {
    streamVertices(endVertex - startVertex + 1, [&](SIZE_TYPE index) {
        SIZE_TYPE nodeIndex = startVertex + index;
        if (isActiveNodeListD[nodeIndex]) {
            SIZE_TYPE edgeIndex = nodePointersD[nodeIndex] - offset;
            SIZE_TYPE sourceValue = valueD[nodeIndex];
            for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                SIZE_TYPE destValue = valueD[edgeListD[i]];
                if (sourceValue < destValue) {
                    atomicMin(&valueD[edgeListD[i]], sourceValue);
                    nextActiveNodeListD[edgeListD[i]] = true;
                } else if (destValue < sourceValue) {
                    atomicMin(&valueD[nodeIndex], destValue);
                    nextActiveNodeListD[nodeIndex] = true;
                }
            }
        }
    });
}

__global__ void
ssspKernel_CommonPartition(SIZE_TYPE startVertex, SIZE_TYPE endVertex, SIZE_TYPE offset, const bool *isActiveNodeListD,
                           const SIZE_TYPE *nodePointersD,
                           const EdgeWithWeight *edgeListD, const SIZE_TYPE *degreeD, SIZE_TYPE *valueD,
                           bool *nextActiveNodeListD) {
    streamVertices(endVertex - startVertex + 1, [&](SIZE_TYPE index) {
        SIZE_TYPE nodeIndex = startVertex + index;
        if (isActiveNodeListD[nodeIndex]) {
            SIZE_TYPE edgeIndex = nodePointersD[nodeIndex] - offset;
            SIZE_TYPE sourceValue = valueD[nodeIndex];
            SIZE_TYPE finalValue;
            for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                finalValue = sourceValue + edgeListD[i].weight;
                SIZE_TYPE vertexId = edgeListD[i].toNode;
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    nextActiveNodeListD[vertexId] = true;
                }
            }
        }
    });
}

__global__ void
bfs_kernelShareOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                   SIZE_TYPE *edgeListShare, SIZE_TYPE *valueD, bool *labelD, SIZE_TYPE overloadNode) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        if (id >= overloadNode) {
            SIZE_TYPE edgeIndex = nodePointersD[id];
            SIZE_TYPE sourceValue = valueD[id];
            SIZE_TYPE finalValue;
            for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                if (finalValue < valueD[edgeListShare[i]]) {
                    atomicMin(&valueD[edgeListShare[i]], finalValue);
                    labelD[edgeListShare[i]] = true;
                    //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
                }
            }
        } else {
            SIZE_TYPE edgeIndex = nodePointersD[id];
            SIZE_TYPE sourceValue = valueD[id];
            SIZE_TYPE finalValue;
            for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                if (finalValue < valueD[edgeListD[i]]) {
                    atomicMin(&valueD[edgeListD[i]], finalValue);
                    labelD[edgeListD[i]] = true;
                    //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
                }
            }
        }

        //printf("index %d vertex %d edgeIndex %d degree %d sourcevalue %d \n", index, id, edgeIndex, degreeD[id], sourceValue);
    });
}

__global__ void
bfs_kernelStatic2Label(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                       SIZE_TYPE *valueD,
                       SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2) {
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            SIZE_TYPE edgeIndex = nodePointersD[id];
            SIZE_TYPE sourceValue = valueD[id];
            SIZE_TYPE finalValue;
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                SIZE_TYPE vertexId;
                vertexId = edgeListD[edgeIndex + i];
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD2[vertexId] = 1;
                }
            }
        }
    });
}


__global__ void
bfs_kernelDynamic2Label(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                        const SIZE_TYPE *degreeD,
                        SIZE_TYPE *valueD,
                        SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2, const SIZE_TYPE *edgeListOverloadD,
                        const SIZE_TYPE *activeOverloadNodePointersD) {
    streamVertices(overloadNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE traverseIndex = overloadStartNode + index;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue = sourceValue + 1;
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                SIZE_TYPE vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                                  activeOverloadNodePointersD[overloadStartNode] + i];
                if (finalValue < valueD[vertexId]) {
                    //printf("source node %d dest node %d set true\n", id, vertexId);
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD2[vertexId] = 1;
                }
            }
        }
    });
}


__global__ void
sssp_kernelDynamicSwap2Label(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                             const SIZE_TYPE *degreeD,
                             SIZE_TYPE *valueD,
                             SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2, const EdgeWithWeight *edgeListOverloadD,
                             const SIZE_TYPE *activeOverloadNodePointersD, bool *finished) {
    streamVertices(overloadNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE traverseIndex = overloadStartNode + index;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
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
                    isActiveD2[vertexId] = 1;
                    *finished = false;
                }
            }
        }
    });
}

__global__ void
sssp_kernelDynamicUvm(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, EdgeWithWeight *edgeListD,
                      SIZE_TYPE *valueD,
                      SIZE_TYPE *labelD1, SIZE_TYPE *labelD2) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        if (labelD1[id]) {
            labelD1[id] = 0;
        }
        SIZE_TYPE edgeIndex = nodePointersD[index];
        /*if (isTest) {
            printf("index %d source vertex %d, edgeIndex is %d degree %d \n", index, id, edgeIndex, degreeD[id]);
        }*/
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        for (SIZE_TYPE i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            finalValue = sourceValue + edgeListD[i].weight;
            SIZE_TYPE vertexId = edgeListD[i].toNode;
            //printf("source vertex %d, edgeindex is %d destnode is %d \n", id, i, edgeListD[i].toNode);
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD2[vertexId] = 1;
            }
        }
    });
}


__global__ void
bfs_kernelStaticSwap(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                     SIZE_TYPE *valueD,
                     SIZE_TYPE *labelD, SIZE_TYPE *fragmentRecordsD, SIZE_TYPE fragment_size, SIZE_TYPE maxpartionSize, SIZE_TYPE testNumNodes) {
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;

        SIZE_TYPE fragmentIndex = edgeIndex / fragment_size;
        SIZE_TYPE fragmentMoreIndex = (edgeIndex + degreeD[id]) / fragment_size;
        if (fragmentMoreIndex > fragmentIndex) {
            atomicAdd(&fragmentRecordsD[fragmentIndex], fragmentIndex * fragment_size + fragment_size - edgeIndex);
        } else {
            atomicAdd(&fragmentRecordsD[fragmentIndex], degreeD[id]);
        }

        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            SIZE_TYPE vertexId;
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelStaticSwap(SIZE_TYPE nodeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                     SIZE_TYPE *valueD,
                     SIZE_TYPE *labelD, bool *isInD, SIZE_TYPE *fragmentRecordsD, SIZE_TYPE fragment_size) {
    streamVertices(nodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        if (isInD[id]) {
            SIZE_TYPE edgeIndex = nodePointersD[id];
            SIZE_TYPE sourceValue = valueD[id];
            SIZE_TYPE finalValue;
            SIZE_TYPE fragmentIndex = edgeIndex / fragment_size;
            SIZE_TYPE fragmentMoreIndex = (edgeIndex + degreeD[id]) / fragment_size;
            if (fragmentMoreIndex > fragmentIndex) {
                atomicAdd(&fragmentRecordsD[fragmentIndex], fragmentIndex * fragment_size + fragment_size - edgeIndex);
            } else {
                atomicAdd(&fragmentRecordsD[fragmentIndex], degreeD[id]);
            }
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

__global__ void
recordFragmentVisit(SIZE_TYPE *activeNodeListD, SIZE_TYPE activeNodeNum, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE fragment_size,
                    SIZE_TYPE *fragmentRecordsD) {
    streamVertices(activeNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodeListD[index];
        SIZE_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE fragmentIndex = edgeIndex / fragment_size;
        SIZE_TYPE fragmentMoreIndex = (edgeIndex + degreeD[id]) / fragment_size;
        if (fragmentMoreIndex > fragmentIndex) {
            atomicAdd(&fragmentRecordsD[fragmentIndex], fragmentIndex * fragment_size + fragment_size - edgeIndex);
        } else {
            atomicAdd(&fragmentRecordsD[fragmentIndex], degreeD[id]);
        }
    });
}

/*__global__ void
bfs_kernelDynamic(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *degreeD, SIZE_TYPE *valueD,
                  SIZE_TYPE *labelD, SIZE_TYPE overloadNode, SIZE_TYPE *overloadEdgeListD,
                  SIZE_TYPE *nodePointersOverloadD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        if (id > overloadNode) {
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                SIZE_TYPE vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    labelD[vertexId] = 1;
                }
            }
        }
    });
}*/

__global__ void
bfs_kernelDynamic(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *degreeD, SIZE_TYPE *valueD,
                  SIZE_TYPE *labelD, SIZE_TYPE overloadNode, SIZE_TYPE *overloadEdgeListD,
                  SIZE_TYPE *nodePointersOverloadD) {
    streamVertices(overloadNode, [&](SIZE_TYPE index) {
        SIZE_TYPE theIndex = activeNum - overloadNode + index;
        SIZE_TYPE id = activeNodesD[theIndex];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            SIZE_TYPE vertexId = overloadEdgeListD[nodePointersOverloadD[theIndex] + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelDynamicSwap(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *degreeD, SIZE_TYPE *valueD,
                      SIZE_TYPE *labelD, SIZE_TYPE *overloadEdgeListD,
                      SIZE_TYPE *nodePointersOverloadD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            SIZE_TYPE vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
              SIZE_TYPE *valueD,
              bool *labelD, SIZE_TYPE overloadNode, SIZE_TYPE *overloadEdgeListD, SIZE_TYPE *nodePointersOverloadD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];

        SIZE_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            SIZE_TYPE vertexId;
            if (id > overloadNode) {
                vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                vertexId = edgeListD[edgeIndex + i];
            }
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
            }
        }
    });
}

__global__ void
cc_kernelOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD, SIZE_TYPE *valueD,
             bool *labelD, SIZE_TYPE overloadNode, SIZE_TYPE *overloadEdgeListD, SIZE_TYPE *nodePointersOverloadD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE sourceValue = valueD[id];
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            SIZE_TYPE vertexId;
            if (id > overloadNode) {
                vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                vertexId = edgeListD[edgeIndex + i];
            }
            SIZE_TYPE destValue = valueD[vertexId];
            if (sourceValue < destValue) {
                atomicMin(&valueD[vertexId], sourceValue);
                labelD[vertexId] = true;
            } else if (destValue < sourceValue) {
                atomicMin(&valueD[id], destValue);
                labelD[id] = true;
            }
        }
    });
}

__global__ void
cc_kernelStaticSwapOpt2Label(SIZE_TYPE activeNodesNum, SIZE_TYPE *activeNodeListD,
                             SIZE_TYPE *staticNodePointerD, SIZE_TYPE *degreeD,
                             SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2, bool *isFinish) {
    streamVertices(activeNodesNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodeListD[index];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            SIZE_TYPE edgeIndex = staticNodePointerD[id];
            SIZE_TYPE sourceValue = valueD[id];
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                SIZE_TYPE vertexId = edgeListD[edgeIndex + i];
                SIZE_TYPE destValue = valueD[vertexId];
                if (sourceValue < destValue) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    isActiveD2[vertexId] = 1;
                    *isFinish = false;
                } else if (destValue < sourceValue) {
                    atomicMin(&valueD[id], destValue);
                    isActiveD2[id] = 1;
                    *isFinish = false;
                }
            }
        }

    });
}

__global__ void
cc_kernelStaticSwapOpt(SIZE_TYPE activeNodesNum, SIZE_TYPE *activeNodeListD,
                       SIZE_TYPE *staticNodePointerD, SIZE_TYPE *degreeD,
                       SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, SIZE_TYPE *isActiveD) {
    streamVertices(activeNodesNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodeListD[index];
        SIZE_TYPE edgeIndex = staticNodePointerD[id];
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
    });
}


__global__ void
cc_kernelStaticAsync(SIZE_TYPE activeNodesNum, const SIZE_TYPE *activeNodeListD,
                     const SIZE_TYPE *staticNodePointerD, const SIZE_TYPE *degreeD,
                     const SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2, const bool *isInStaticD,
                     bool *finished, int *atomicValue) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tidInBlock = blockDim.y * threadIdx.x + threadIdx.y;
    int blockId = blockIdx.x * gridDim.y + blockIdx.y;
    int iter = 0;
    SIZE_TYPE *checkLabel = iter % 2 == 0 ? labelD1 : labelD2;
    SIZE_TYPE *targetLabel = iter % 2 == 0 ? labelD2 : labelD1;
    int syncIndex = 1;
    volatile bool *testFinish = (bool *) finished;
    *testFinish = false;
    gpu_sync(gridDim.x, atomicValue);
    //__syncthreads();
    if (tidInBlock == 0) {
        printf("tid %d blockid %d testFinish %d \n", tid, blockId, *testFinish);
    }
    if (tid == 0) {
        *testFinish = false;
    }
    gpu_sync(2 * gridDim.x, atomicValue);
    if (*testFinish) {
        if (tidInBlock == 0) {
            printf("1 tid %d blockid %d testFinish %d \n", tid, blockId, *testFinish);
        }
        return;
    }
    if (tid == 0) {
        *testFinish = true;
    }
    gpu_sync(2 * gridDim.x, atomicValue);
    if (*testFinish) {
        if (tidInBlock == 0) {
            printf("2 tid %d blockid %d testFinish %d \n", tid, blockId, *testFinish);
        }
        return;
    }
    printf("================\n");
}

__global__ void
cc_kernelDynamicAsync(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD, const SIZE_TYPE *degreeD,
                      SIZE_TYPE *valueD, SIZE_TYPE *labelD1, SIZE_TYPE *labelD2, const SIZE_TYPE *edgeListOverloadD,
                      const SIZE_TYPE *activeOverloadNodePointersD, bool *finished) {
    streamVertices(overloadNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE traverseIndex = overloadStartNode + index;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        if (labelD1[id]) {
            labelD1[id] = 0;
            SIZE_TYPE sourceValue = valueD[id];
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                SIZE_TYPE vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                                  activeOverloadNodePointersD[overloadStartNode] + i];
                SIZE_TYPE destValue = valueD[vertexId];
                if (sourceValue < destValue) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    *finished = false;
                    labelD2[vertexId] = 1;
                } else if (destValue < sourceValue) {
                    atomicMin(&valueD[id], destValue);
                    *finished = false;
                    labelD2[id] = 1;
                }
            }
        }
    });
}


__global__ void
cc_kernelDynamicSwap2Label(SIZE_TYPE overloadStartNode, SIZE_TYPE overloadNodeNum, const SIZE_TYPE *overloadNodeListD,
                           const SIZE_TYPE *degreeD,
                           SIZE_TYPE *valueD,
                           SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2, const SIZE_TYPE *edgeListOverloadD,
                           const SIZE_TYPE *activeOverloadNodePointersD, bool *finished) {
    streamVertices(overloadNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE traverseIndex = overloadStartNode + index;
        SIZE_TYPE id = overloadNodeListD[traverseIndex];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            SIZE_TYPE sourceValue = valueD[id];
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                SIZE_TYPE vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                                  activeOverloadNodePointersD[overloadStartNode] + i];
                SIZE_TYPE destValue = valueD[vertexId];
                if (sourceValue < destValue) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    isActiveD2[vertexId] = 1;
                    *finished = false;
                } else if (destValue < sourceValue) {
                    atomicMin(&valueD[id], destValue);
                    isActiveD2[id] = 1;
                    *finished = false;
                }
            }
        }
    });
}

__global__ void
sssp_kernelStaticSwapOpt2Label(SIZE_TYPE activeNodesNum, const SIZE_TYPE *activeNodeListD,
                               const SIZE_TYPE *staticNodePointerD, const SIZE_TYPE *degreeD,
                               EdgeWithWeight *edgeListD, SIZE_TYPE *valueD, SIZE_TYPE *isActiveD1, SIZE_TYPE *isActiveD2,
                               bool *isFinish) {

    streamVertices(activeNodesNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodeListD[index];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            SIZE_TYPE edgeIndex = staticNodePointerD[id];
            SIZE_TYPE sourceValue = valueD[id];
            SIZE_TYPE finalValue;
            for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
                EdgeWithWeight checkNode{};
                checkNode = edgeListD[edgeIndex + i];
                finalValue = sourceValue + checkNode.weight;
                SIZE_TYPE vertexId = checkNode.toNode;
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD2[vertexId] = 1;
                    *isFinish = false;
                }
            }
        }

    });
}


__global__ void
sssp_kernelOpt(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, EdgeWithWeight *edgeListD,
               SIZE_TYPE *valueD,
               bool *labelD, SIZE_TYPE overloadNode, EdgeWithWeight *overloadEdgeListD, SIZE_TYPE *nodePointersOverloadD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE edgeIndex = nodePointersD[id];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            EdgeWithWeight checkNode{};
            if (id > overloadNode) {
                checkNode = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                checkNode = edgeListD[edgeIndex + i];
            }
            finalValue = sourceValue + checkNode.weight;
            SIZE_TYPE vertexId = checkNode.toNode;
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
                //printf("source vertex %d, toNode is %d \n", id, vertexId);
            }
        }
    });
}

__global__ void
bfs_kernelOptOfSorted(SIZE_TYPE activeNum, SIZE_TYPE *activeNodesD, SIZE_TYPE *nodePointersD, SIZE_TYPE *degreeD, SIZE_TYPE *edgeListD,
                      SIZE_TYPE *edgeListOverload, SIZE_TYPE *valueD, bool *labelD, bool *isInListD,
                      SIZE_TYPE *nodePointersOverloadD) {
    streamVertices(activeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodesD[index];
        SIZE_TYPE sourceValue = valueD[id];
        SIZE_TYPE finalValue;
        SIZE_TYPE edgeIndex;
        SIZE_TYPE *edgeList;
        if (!isInListD[id]) {
            edgeIndex = nodePointersOverloadD[index];
            edgeList = edgeListOverload;
        } else {
            edgeIndex = nodePointersD[id];
            edgeList = edgeListD;
        }

        for (SIZE_TYPE i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            SIZE_TYPE vertexId = edgeList[edgeIndex + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
                //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
            }
        }
    });
}

__global__ void
setFragmentData(SIZE_TYPE activeNodeNum, SIZE_TYPE *activeNodeList, SIZE_TYPE *staticNodePointers, SIZE_TYPE *staticFragmentData,
                SIZE_TYPE staticFragmentNum, SIZE_TYPE fragmentSize, bool *isInStatic) {
    streamVertices(activeNodeNum, [&](SIZE_TYPE index) {
        SIZE_TYPE vertexId = activeNodeList[index];
        if (isInStatic[vertexId]) {
            SIZE_TYPE staticFragmentIndex = staticNodePointers[vertexId] / fragmentSize;
            if (staticFragmentIndex < staticFragmentNum) {
                staticFragmentData[staticFragmentIndex] = 1;
            }
        }
    });
}

__global__ void
setStaticFragmentData(SIZE_TYPE staticFragmentNum, SIZE_TYPE *canSwapFragmentD, SIZE_TYPE *canSwapFragmentPrefixD,
                      SIZE_TYPE *staticFragmentDataD) {
    streamVertices(staticFragmentNum, [&](SIZE_TYPE index) {
        if (canSwapFragmentD[index] > 0) {
            staticFragmentDataD[canSwapFragmentPrefixD[index]] = index;
            canSwapFragmentD[index] = 0;
        }
    });
}

__global__ void
setFragmentDataOpt(SIZE_TYPE *staticFragmentData, SIZE_TYPE staticFragmentNum, SIZE_TYPE *staticFragmentVisitRecordsD) {
    streamVertices(staticFragmentNum, [&](SIZE_TYPE index) {
        SIZE_TYPE fragmentId = index;
        if (staticFragmentVisitRecordsD[fragmentId] > 3600) {
            staticFragmentData[fragmentId] = 1;
            staticFragmentVisitRecordsD[fragmentId] = 0;
        } else {
            staticFragmentData[fragmentId] = 0;
        }
    });
}

__global__ void
setFragmentDataOpt4Pr(SIZE_TYPE *staticFragmentData, SIZE_TYPE fragmentNum, SIZE_TYPE *fragmentVisitRecordsD,
                      bool *isActiveFragmentD, SIZE_TYPE *fragmentNormalMap2StaticD, SIZE_TYPE maxStaticFragment) {
    streamVertices(fragmentNum, [&](SIZE_TYPE fragmentId) {
        /*if (fragmentId == 887550) {
            printf("fragmentId 887550 record %d \n", fragmentVisitRecordsD[fragmentId]);
        }*/
        if (fragmentVisitRecordsD[fragmentId] > 3200) {
            isActiveFragmentD[fragmentId] = false;
            //fragmentVisitRecordsD[fragmentId] = 0;
        } else {
            isActiveFragmentD[fragmentId] = true;
            fragmentVisitRecordsD[fragmentId] = 0;
        }
        if (!isActiveFragmentD[fragmentId]) {
            SIZE_TYPE staticFragmentIndex = fragmentNormalMap2StaticD[fragmentId];
            if (staticFragmentIndex < maxStaticFragment) {
                staticFragmentData[staticFragmentIndex] = 1;
            }
        }
    });
}

SIZE_TYPE reduceBool(SIZE_TYPE *resultD, bool *isActiveD, SIZE_TYPE vertexSize, dim3 grid, dim3 block) {
    //printf("reduceBool \n");
    SIZE_TYPE activeNodesNum = 0;
    int blockSize = block.x;
    reduceByBool<<<grid, block, block.x * sizeof(SIZE_TYPE)>>>(vertexSize, isActiveD, resultD);
    reduceResult<56><<<1, 64, block.x * sizeof(SIZE_TYPE)>>>(resultD);
    cudaMemcpy(&activeNodesNum, resultD, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
    return activeNodesNum;
}

__device__ void reduceStreamVertices(SIZE_TYPE vertices, bool *rawData, SIZE_TYPE *result) {

    extern __shared__ SIZE_TYPE sdata[];
    SIZE_TYPE tid = threadIdx.x;
    sdata[tid] = 0;
    for (auto i : grid_stride_range(SIZE_TYPE(0), vertices)) {
        sdata[tid] += rawData[i];
    }
    __syncthreads();
    if (blockDim.x > 512 && tid < 512) { sdata[tid] += sdata[tid + 512]; }
    __syncthreads();
    if (blockDim.x > 256 && tid < 256) { sdata[tid] += sdata[tid + 256]; }
    __syncthreads();
    if (blockDim.x > 128 && tid < 128) { sdata[tid] += sdata[tid + 128]; }
    __syncthreads();
    if (blockDim.x > 64 && tid < 64) { sdata[tid] += sdata[tid + 64]; }
    __syncthreads();
    if (tid < 32) { sdata[tid] += sdata[tid + 32]; }
    __syncthreads();

    if (tid < 16) { sdata[tid] += sdata[tid + 16]; }
    __syncthreads();
    if (tid < 8) { sdata[tid] += sdata[tid + 8]; }
    __syncthreads();
    if (tid < 4) { sdata[tid] += sdata[tid + 4]; }
    __syncthreads();
    if (tid < 2) { sdata[tid] += sdata[tid + 2]; }
    if (tid < 1) { sdata[tid] += sdata[tid + 1]; }
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceByBool(SIZE_TYPE vertexSize, bool *rawData, SIZE_TYPE *result) {
    reduceStreamVertices(vertexSize, rawData, result);
}

template<int blockSize>
__global__ void reduceResult(SIZE_TYPE *result) {
    extern __shared__ SIZE_TYPE sdata[];
    SIZE_TYPE tid = threadIdx.x;
    sdata[tid] = 0;
    if (tid < blockSize) {
        sdata[tid] = result[tid];
    }
    __syncthreads();
    if (tid < 32) { sdata[tid] += sdata[tid + 32]; }
    __syncthreads();
    if (tid < 16) { sdata[tid] += sdata[tid + 16]; }
    __syncthreads();
    if (tid < 8) { sdata[tid] += sdata[tid + 8]; }
    __syncthreads();
    if (tid < 4) { sdata[tid] += sdata[tid + 4]; }
    __syncthreads();
    if (tid < 2) { sdata[tid] += sdata[tid + 2]; }
    if (tid < 1) { sdata[tid] += sdata[tid + 1]; }
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

template<int BS> __global__ void scanWarpReduceInBlock(int n, bool* in, SIZE_TYPE* out) {
    extern __shared__ SIZE_TYPE sdata[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadIdx.x / 32;
    int idInWarp = threadIdx.x % 32;
    bool data = in[id];
    int sumInWarp = reduceInWarp<32>(idInWarp, data);
    if (idInWarp == 0) sdata[warpId] = sumInWarp;
    __syncthreads();
}

template<int NT>
__device__ int reduceInWarp(int idInWarp, bool data) {
    int ret = data;
    for (int i = NT / 2; i > 0; i /= 2) {
        data = __shfl_down(ret, i, NT);
        if (idInWarp < i) ret += data;
    }
    return ret;
}


