#include"globals.cuh"
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
#include "TimeRecord.cuh"
#include"gpu_kernels.cuh"
#include "cpu_verify.cuh"
#include "cache_density.cuh"
#include "edge_path.cuh"
#pragma once
struct StaticRegionInfo
{
    SIZE_TYPE max_node;
    SIZE_TYPE max_partion_size;
};
EDGE_POINTER_TYPE vertexArrSize, edgeArrSize;

StaticRegionInfo getMaxPartionSize(int paramSize, unsigned long long edgeArrSize, EDGE_POINTER_TYPE vertexArrSize, EDGE_POINTER_TYPE* nodePointers, SIZE_TYPE*degree,
                       bool* isInStatic, size_t gpuMemoryLimitBytes = 0){
    unsigned long max_partition_size;
    unsigned long max_static_node;
    unsigned long total_gpu_size;
    SIZE_TYPE fragmentSize = 4096;
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
    g_gpuMemTracker.totalAllocated = 0;
    size_t reduceMem;
    reduceMem = 6*sizeof(SIZE_TYPE)*(size_t)vertexArrSize;
    reduceMem += 4 * sizeof(bool) * (size_t) vertexArrSize;
    reduceMem += (size_t)vertexArrSize*sizeof(EDGE_POINTER_TYPE);
    cout << "reduceMem " << reduceMem << " testNumNodes " << vertexArrSize << " edgeArrSize " << edgeArrSize << " ParamsSize " << paramSize << endl;
    if (reduceMem >= availMemory) {
        cout << "WARNING: per-vertex arrays (" << reduceMem / (1024.0*1024.0) << " MB) exceed available GPU memory ("
             << availMemory / (1024.0*1024.0) << " MB). Setting edge partition to minimum." << endl;
        total_gpu_size = 0;
    } else {
        total_gpu_size = (availMemory - reduceMem) / sizeof(SIZE_TYPE);
    }
    max_partition_size = total_gpu_size;
    if (max_partition_size > edgeArrSize) {
        max_partition_size = edgeArrSize;
    }
    cout << "availMemory " << availMemory << " totalMemory " << totalMemory << endl;
    printf("static memory is %zu totalGlobalMem is %zu, max static edge size is %lu\n gpu total edge size %lu \n multiprocessors %d \n",
                (reduceMem < availMemory) ? (availMemory - reduceMem) : (size_t)0,
                dev.totalGlobalMem, max_partition_size, total_gpu_size, dev.multiProcessorCount);
    SIZE_TYPE temp = max_partition_size % fragmentSize;
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
    StaticRegionInfo info;
    info.max_node = max_static_node;
    info.max_partion_size = max_partition_size;
    return info;
}
void refreshLableAndValue(bool* isActiveD, bool *isStaticActiveD, bool* isoverloadActiveD,SIZE_TYPE* value, SIZE_TYPE* valueD){
    cudaMemset(isActiveD,1,sizeof(bool)*vertexArrSize);
    cudaMemset(isStaticActiveD,0,sizeof(bool)*vertexArrSize);
    cudaMemset(isoverloadActiveD,0,sizeof(bool)*vertexArrSize);
    cudaMemcpy(valueD,value,vertexArrSize*sizeof(SIZE_TYPE),cudaMemcpyHostToDevice);
}

__global__ 
void cc_kernelStatic(SIZE_TYPE activeNodesNum, SIZE_TYPE *activeNodeListD,
                    SIZE_TYPE *staticNodePointerD, SIZE_TYPE *degreeD,
                    SIZE_TYPE *edgeListD, SIZE_TYPE *valueD, bool *isActiveD, bool *isInStaticD) {
    streamVertices(activeNodesNum, [&](SIZE_TYPE index) {
        SIZE_TYPE id = activeNodeListD[index];
        if (isInStaticD[id]) {
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
        }
    });
}

void New_CC_opt(string fileName,int model,int testTimes, double gpuMemoryLimit = 0.0, bool verify = false, string cacheCsv = "", string pathCsv = ""){
    if(model!=7){
        cout<<"model not match"<<endl;
        return;
    }
    EDGE_POINTER_TYPE *nodePointers;
    SIZE_TYPE* edgeArray; 
    
    StaticRegionInfo StaticInfo;
    //ReadDataFile
    cout << "readDataFromFile" << endl;
    auto startTime = chrono::steady_clock::now();
    bool isBCSR = endsWith(fileName, ".bcsr") || endsWith(fileName, ".bwcsr");
    ifstream infile(fileName, ios::in | ios::binary);
    if (isBCSR) {
        // Subway bcsr format: uint32 header + uint32 nodePointers + uint32 edges
        uint32_t num_nodes, num_edges;
        infile.read((char *) &num_nodes, sizeof(uint32_t));
        infile.read((char *) &num_edges, sizeof(uint32_t));
        vertexArrSize = num_nodes;
        edgeArrSize = num_edges;
        cout << "vertex num: " << vertexArrSize << " edge num: " << edgeArrSize << endl;
        // Read 32-bit nodePointers and widen to 64-bit
        uint32_t *nodePointersU32 = new uint32_t[num_nodes];
        infile.read((char *) nodePointersU32, sizeof(uint32_t) * num_nodes);
        nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
        for (SIZE_TYPE i = 0; i < num_nodes; i++) {
            nodePointers[i] = (EDGE_POINTER_TYPE) nodePointersU32[i];
        }
        delete[] nodePointersU32;
        // Read 32-bit edges and widen to 64-bit
        gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(SIZE_TYPE)*edgeArrSize));
        {
            const size_t CHUNK = 1 << 20;
            uint32_t *buf = new uint32_t[CHUNK];
            EDGE_POINTER_TYPE offset = 0;
            EDGE_POINTER_TYPE remaining = edgeArrSize;
            while (remaining > 0) {
                size_t n = (remaining < (EDGE_POINTER_TYPE)CHUNK) ? (size_t)remaining : CHUNK;
                infile.read((char *)buf, sizeof(uint32_t) * n);
                for (size_t i = 0; i < n; i++) {
                    edgeArray[offset + i] = (SIZE_TYPE)buf[i];
                }
                offset += n;
                remaining -= n;
            }
            delete[] buf;
        }
    } else {
        infile.read((char *) &vertexArrSize, sizeof(EDGE_POINTER_TYPE));
        infile.read((char *) &edgeArrSize, sizeof(EDGE_POINTER_TYPE));
        cout << "vertex num: " << vertexArrSize << " edge num: " << edgeArrSize << endl;
        nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
        infile.read((char *) nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
        gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(SIZE_TYPE)*edgeArrSize));
        infile.read((char *) edgeArray, sizeof(SIZE_TYPE) * edgeArrSize);
    }
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << endl;

    SIZE_TYPE* degree;
    bool* isInStatic;
    SIZE_TYPE* overloadNodeList;
    bool* isActive;
    SIZE_TYPE* value;
    SIZE_TYPE* staticnodepointers;
    cout << "initGraphHost()" << endl;
    degree = new SIZE_TYPE[vertexArrSize];
    isInStatic = new bool[vertexArrSize];
    overloadNodeList = new SIZE_TYPE[vertexArrSize];
    isActive = new bool[vertexArrSize];
    value = new SIZE_TYPE[vertexArrSize];
    
    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        if (nodePointers[i] > edgeArrSize) {
            cout << i << "   " << nodePointers[i] << endl;
            break;
        }
        degree[i] = nodePointers[i + 1] - nodePointers[i];
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    size_t gpuMemLimitBytes = 0;
    if(gpuMemoryLimit > 0){
        gpuMemLimitBytes = (size_t)(gpuMemoryLimit * 1024ULL * 1024ULL * 1024ULL);
    }
    StaticInfo = getMaxPartionSize(11,edgeArrSize,vertexArrSize,nodePointers,degree,isInStatic, gpuMemLimitBytes);
    for(SIZE_TYPE i=0;i<vertexArrSize;i++){
        isActive[i] = 1;
        value[i] = i;
    }

    SIZE_TYPE max_static_node = StaticInfo.max_node+1;
    SIZE_TYPE max_partition_size = StaticInfo.max_partion_size;
    staticnodepointers = new SIZE_TYPE[max_static_node];
    for(SIZE_TYPE i=0;i<max_static_node;i++){
        staticnodepointers[i] = (SIZE_TYPE)nodePointers[i];
    }

    cudaStream_t StreamStatic, StreamDynamic;
    EDGE_POINTER_TYPE* nodePointersD;
    SIZE_TYPE* prefixSumTemp;
    SIZE_TYPE* staticNodePointersD;
    SIZE_TYPE* staticEdgeListD;
    bool* isInStaticD;
    SIZE_TYPE* overloadNodeListD;
    SIZE_TYPE* staticNodeListD;
    SIZE_TYPE* degreeD;
    bool* isActiveD;
    bool* isStaticActive;
    bool* isOverloadActive;
    SIZE_TYPE* valueD;
    cout<<"initGraphDevice()"<<endl;
    GPU_MALLOC(&prefixSumTemp, vertexArrSize * sizeof(SIZE_TYPE), "CC:prefixSumTemp");
    gpuErrorcheck(cudaStreamCreate(&StreamStatic));
    gpuErrorcheck(cudaStreamCreate(&StreamDynamic));

    TimeRecord<chrono::milliseconds> preProcess("pre move data");
    preProcess.startRecord();
    GPU_MALLOC(&nodePointersD, vertexArrSize*sizeof(EDGE_POINTER_TYPE), "CC:nodePointersD");
    gpuErrorcheck(cudaMemcpy(nodePointersD,nodePointers,vertexArrSize*sizeof(EDGE_POINTER_TYPE),cudaMemcpyHostToDevice));
    GPU_MALLOC(&staticNodePointersD, max_static_node*sizeof(SIZE_TYPE), "CC:staticNodePointersD");
    gpuErrorcheck(cudaMemcpy(staticNodePointersD, staticnodepointers, max_static_node*sizeof(SIZE_TYPE),cudaMemcpyHostToDevice));
    GPU_MALLOC(&staticEdgeListD, max_partition_size * sizeof(SIZE_TYPE), "CC:staticEdgeListD");
    gpuErrorcheck(cudaMemcpy(staticEdgeListD, edgeArray, max_partition_size * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    preProcess.endRecord();
    long preMoveDataTime = preProcess.getDuration();
    preProcess.print();
    preProcess.clearRecord();
    GPU_MALLOC(&isInStaticD, vertexArrSize * sizeof(bool), "CC:isInStaticD");
    cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    GPU_MALLOC(&overloadNodeListD, vertexArrSize * sizeof(SIZE_TYPE), "CC:overloadNodeListD");
    GPU_MALLOC(&staticNodeListD, vertexArrSize * sizeof(SIZE_TYPE), "CC:staticNodeListD");
    GPU_MALLOC(&degreeD, vertexArrSize * sizeof(SIZE_TYPE), "CC:degreeD");
    GPU_MALLOC(&isActiveD, vertexArrSize * sizeof(bool), "CC:isActiveD");
    GPU_MALLOC(&isStaticActive, vertexArrSize * sizeof(bool), "CC:isStaticActive");
    GPU_MALLOC(&isOverloadActive, vertexArrSize * sizeof(bool), "CC:isOverloadActive");
    cudaMemcpy(degreeD, degree, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(isActiveD, isActive, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool));
    cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool));
    GPU_MALLOC(&valueD, vertexArrSize * sizeof(SIZE_TYPE), "CC:valueD");
    cudaMemcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    thrust::device_ptr<bool> activeLablingThrust;
    thrust::device_ptr<bool> actStaticLablingThrust;
    thrust::device_ptr<bool> actOverLablingThrust;
    activeLablingThrust = thrust::device_ptr<bool>(isActiveD);
    actStaticLablingThrust = thrust::device_ptr<bool>(isStaticActive);
    actOverLablingThrust = thrust::device_ptr<bool>(isOverloadActive);
    gpuErrorcheck(cudaPeekAtLastError());
    g_gpuMemTracker.printSummary();
    cout << "initGraphDevice() end" << endl;

    cudaDeviceSynchronize();
    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    totalProcess.startRecord();
    SIZE_TYPE activeNodesNum;
    activeNodesNum = thrust::reduce(activeLablingThrust, activeLablingThrust + vertexArrSize, 0,
                                    thrust::plus<SIZE_TYPE>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    uint64_t numthreads = 1024;
    long totalduration = 0;
    long overloadduration = 0;
    long staticduration = 0;
    dim3 staticgrid(56,1,1);
    dim3 staticblock(1024,1,1);
    CacheDensityRecorder cacheRec(vertexArrSize, isInStaticD, "cc", fileName, -1LL, cacheCsv);
    EdgePathRecorder pathRec(vertexArrSize, "cc", fileName, -1LL, pathCsv);
    for (int testIndex = 0; testIndex < testTimes; testIndex++){
        //
        cudaDeviceSynchronize();
        cout<<"================="<<"testIndex "<<testIndex<<"================="<<endl;
        SIZE_TYPE nodeSum = activeNodesNum;
        int iter = 0;
        totalProcess.startRecord();
        double overloadsize = 0;
        while(activeNodesNum){
            iter++;
            // Cache-density snapshot: before setLabelDefaultOpt clears
            // processed vertices.
            cacheRec.record(isActiveD, isInStaticD);
            //cout<<"iter "<<iter<<" activeNodeNum is "<<activeNodesNum<<" ";
            setStaticAndOverloadLabelBool<<<staticgrid,staticblock>>>(vertexArrSize, isActiveD, isStaticActive, isOverloadActive,
                                                        isInStaticD);
            // Edge-path snapshot
            pathRec.record(isStaticActive, isOverloadActive, degreeD);
            SIZE_TYPE staticNodeNum = thrust::reduce(actStaticLablingThrust,
                                                actStaticLablingThrust + vertexArrSize, 0,
                                                thrust::plus<SIZE_TYPE>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(prefixSumTemp);
                    
                thrust::exclusive_scan(actStaticLablingThrust, actStaticLablingThrust + vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setStaticActiveNodeArray<<<staticgrid,staticblock, 0, StreamStatic>>>(vertexArrSize, staticNodeListD, isStaticActive,
                                                                                      prefixSumTemp);
            }
            SIZE_TYPE overloadNodeNum = thrust::reduce(actOverLablingThrust,
                                                  actOverLablingThrust + vertexArrSize, 0,
                                                  thrust::plus<SIZE_TYPE>());
            //cout<<"staticNodeNum is "<<staticNodeNum<<" overloadNodeNum is "<<overloadNodeNum<<endl;
            if(overloadNodeNum>0){
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(prefixSumTemp);
                thrust::exclusive_scan(actOverLablingThrust, actOverLablingThrust + vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setActiveNodeList<<<staticgrid,staticblock, 0, StreamStatic>>>(vertexArrSize, isOverloadActive, overloadNodeListD,
                                                                               prefixSumTemp);
            }
            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<staticgrid, staticblock, 0, StreamStatic>>>(staticNodeNum, staticNodeListD, isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<staticgrid, staticblock, 0, StreamDynamic>>>(overloadNodeNum, overloadNodeListD, isActiveD);
            }
            cudaDeviceSynchronize();
            //cout<<"launch static kernel"<<endl;
            
            staticProcess.startRecord();
            cc_kernelStatic<<<staticgrid, staticblock , 0, StreamStatic>>>(staticNodeNum, staticNodeListD, staticNodePointersD,
                                                                degreeD, staticEdgeListD, valueD, 
                                                                isActiveD, isInStaticD);
            
            if(overloadNodeNum > 0){
                overloadProcess.startRecord();
                // uint64_t numblocks = ((overloadNodeNum * WARP_SIZE + numthreads) / numthreads);
                // dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                // SIZE_TYPE numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                SIZE_TYPE numwarps = (1024*56+32)/32;
                //cout<<"launch overload kernel"<<endl;
                NEW_cc_kernelDynamicSwap_test<<<staticgrid, staticblock ,0,StreamDynamic>>>(overloadNodeNum, overloadNodeListD,
                                                                                        degreeD, valueD, numwarps,
                                                                                        isActiveD,
                                                                                        edgeArray, nodePointersD);

                cudaStreamSynchronize(StreamDynamic);
                overloadProcess.endRecord(); 
                cudaStreamSynchronize(StreamStatic);
                staticProcess.endRecord();
            }
            else{
                cudaDeviceSynchronize();
                staticProcess.endRecord();
                gpuErrorcheck(cudaPeekAtLastError());
            }
            if(staticProcess._isStart()){
                staticProcess.endRecord();
            }
            activeNodesNum = thrust::reduce(activeLablingThrust, activeLablingThrust + vertexArrSize,
                                            0,
                                            thrust::plus<SIZE_TYPE>());
            nodeSum += activeNodesNum;
            // SIZE_TYPE overloadedges = 0;
            // SIZE_TYPE* overloadnodes = new SIZE_TYPE[overloadNodeNum];
            // cudaMemcpy(overloadnodes,overloadNodeListD,sizeof(SIZE_TYPE)*overloadNodeNum,cudaMemcpyDeviceToHost);
            // for(SIZE_TYPE i=0;i<overloadNodeNum;i++){
            //     overloadedges += degree[overloadnodes[i]];
            // }
            // double temp = (double)(overloadedges)*sizeof(SIZE_TYPE)/1024;
            // cout<<temp<<endl;
            // overloadsize += temp;
        }
        totalProcess.endRecord();
        cout<<"total iter: "<<iter<<endl;
        totalProcess.print();
        staticProcess.print();
        overloadProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        //cout<< "overloadSize: "<< overloadsize/1024/1024 <<"GB"<<endl;
        totalduration+=totalProcess.getDuration();
        staticduration+=staticProcess.getDuration();
        overloadduration+=overloadProcess.getDuration();
        totalProcess.clearRecord();
        staticProcess.clearRecord();
        overloadProcess.clearRecord();
        if (verify && testIndex == testTimes - 1) {
            // Verify before refresh resets the values
            cudaMemcpy(value, valueD, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cpu_verify_cc(nodePointers, edgeArray, vertexArrSize, edgeArrSize, value);
        }
        refreshLableAndValue(isActiveD,isStaticActive,isOverloadActive,value,valueD);
        activeNodesNum = thrust::reduce(activeLablingThrust, activeLablingThrust + vertexArrSize, 0,
                                    thrust::plus<SIZE_TYPE>());
    }
    cout<<"========TEST OVER========"<<endl;
    cout<<"pre move data time: "<<preMoveDataTime<<"ms"<<endl;
    cout<<"Test over, average total process time (including pre move data): "<<totalduration/testTimes + preMoveDataTime<<"ms"<<endl;
    cout<<"average static process time: "<<staticduration/testTimes<<"ms"<<endl;
    cout<<"average overload process time: "<<overloadduration/testTimes<<"ms"<<endl;
    cacheRec.printSummary();
    cacheRec.writeCsv(cacheCsv, 0);
    pathRec.printSummary();
    pathRec.writeCsv(pathCsv, 0);
    gpuErrorcheck(cudaPeekAtLastError());

    return;
}
