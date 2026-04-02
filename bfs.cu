//
// Created by gxl on 2020/12/30.
//
#include "bfs.cuh"

void conventionParticipateBFS(string bfsPath, int sampleSourceNode) {
    cout << "===============conventionParticipateBFS==============" << endl;
    SIZE_TYPE testNumNodes = 0;
    ulong testNumEdge = 0;
    ulong traverseSum = 0;
    SIZE_TYPE *nodePointersI;
    SIZE_TYPE *edgeList;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(bfsPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(SIZE_TYPE));
    SIZE_TYPE numEdge = 0;
    infile.read((char *) &numEdge, sizeof(SIZE_TYPE));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    nodePointersI = new SIZE_TYPE[testNumNodes];
    infile.read((char *) nodePointersI, sizeof(SIZE_TYPE) * testNumNodes);
    edgeList = new SIZE_TYPE[testNumEdge];
    infile.read((char *) edgeList, sizeof(SIZE_TYPE) * testNumEdge);
    infile.close();
    unsigned long max_partition_size;
    unsigned long total_gpu_size;
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, 0.9, sizeof(SIZE_TYPE), 5);
    SIZE_TYPE partitionNum;
    if (testNumEdge > max_partition_size) {
        partitionNum = testNumEdge / max_partition_size + 1;
    } else {
        partitionNum = 1;
    }

    SIZE_TYPE *degree = new SIZE_TYPE[testNumNodes];
    SIZE_TYPE *value = new SIZE_TYPE[testNumNodes];
    bool *isActiveNodeList = new bool[testNumNodes];
    CommonPartitionInfo *partitionInfoList = new CommonPartitionInfo[partitionNum];
    bool *needTransferPartition = new bool[partitionNum];
    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        isActiveNodeList[i] = false;
        value[i] = UINT_MAX;
        if (i + 1 < testNumNodes) {
            degree[i] = nodePointersI[i + 1] - nodePointersI[i];
        } else {
            degree[i] = testNumEdge - nodePointersI[i];
        }
        if (degree[i] > max_partition_size) {
            cout << "node " << i << " degree > maxPartition " << endl;
            return;
        }
    }
    for (SIZE_TYPE i = 0; i < partitionNum; i++) {
        partitionInfoList[i].startVertex = -1;
        partitionInfoList[i].endVertex = -1;
        partitionInfoList[i].nodePointerOffset = -1;
        partitionInfoList[i].partitionEdgeSize = -1;
    }
    int tempPartitionIndex = 0;
    SIZE_TYPE tempNodeIndex = 0;
    while (tempNodeIndex < testNumNodes) {
        if (partitionInfoList[tempPartitionIndex].startVertex == -1) {
            partitionInfoList[tempPartitionIndex].startVertex = tempNodeIndex;
            partitionInfoList[tempPartitionIndex].endVertex = tempNodeIndex;
            partitionInfoList[tempPartitionIndex].nodePointerOffset = nodePointersI[tempNodeIndex];
            partitionInfoList[tempPartitionIndex].partitionEdgeSize = degree[tempNodeIndex];
            tempNodeIndex++;
        } else {
            if (partitionInfoList[tempPartitionIndex].partitionEdgeSize + degree[tempNodeIndex] > max_partition_size) {
                tempPartitionIndex++;
            } else {
                partitionInfoList[tempPartitionIndex].endVertex = tempNodeIndex;
                partitionInfoList[tempPartitionIndex].partitionEdgeSize += degree[tempNodeIndex];
                tempNodeIndex++;
            }
        }
    }

    SIZE_TYPE *degreeD;
    bool *isActiveNodeListD;
    bool *nextActiveNodeListD;
    SIZE_TYPE *nodePointerListD;
    SIZE_TYPE *partitionEdgeListD;
    SIZE_TYPE *valueD;

    cudaMalloc(&degreeD, testNumNodes * sizeof(SIZE_TYPE));
    cudaMalloc(&valueD, testNumNodes * sizeof(SIZE_TYPE));
    cudaMalloc(&isActiveNodeListD, testNumNodes * sizeof(bool));
    cudaMalloc(&nextActiveNodeListD, testNumNodes * sizeof(bool));
    cudaMalloc(&nodePointerListD, testNumNodes * sizeof(SIZE_TYPE));
    cudaMalloc(&partitionEdgeListD, max_partition_size * sizeof(SIZE_TYPE));

    cudaMemcpy(degreeD, degree, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(nodePointerListD, nodePointersI, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemset(nextActiveNodeListD, 0, testNumNodes * sizeof(bool));
    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    int testTimes = 1;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        SIZE_TYPE sourceNode = rand() % testNumNodes;
        sourceNode = sampleSourceNode;
        //sourceNode = 25838548;
        //sourceNode = 26890152;
        //sourceNode = 47235513;
        cout << "sourceNode " << sourceNode << endl;
        for (int j = 0; j < testNumNodes; j++) {
            isActiveNodeList[j] = false;
            value[j] = UINT_MAX;
        }
        isActiveNodeList[sourceNode] = true;
        value[sourceNode] = 1;
        cudaMemcpy(valueD, value, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        SIZE_TYPE activeSum = 0;
        int iteration = 0;

        auto startProcessing = std::chrono::steady_clock::now();
        while (true) {
            SIZE_TYPE activeNodeNum = 0;
            checkNeedTransferPartitionOpt(needTransferPartition, partitionInfoList, isActiveNodeList, partitionNum,
                                          testNumNodes, activeNodeNum);
            if (activeNodeNum <= 0) {
                break;
            } else {
                //cout << "iteration " << iteration << " activeNodes " << activeNodeNum << endl;
                activeSum += activeNodeNum;
            }
            cudaMemcpy(isActiveNodeListD, isActiveNodeList, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
            for (int j = 0; j < partitionNum; j++) {
                if (needTransferPartition[j]) {
                    cudaMemcpy(partitionEdgeListD, edgeList + partitionInfoList[j].nodePointerOffset,
                               partitionInfoList[j].partitionEdgeSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
                    traverseSum += partitionInfoList[j].partitionEdgeSize * sizeof(SIZE_TYPE);
                    bfsKernel_CommonPartition<<<grid, block>>>(partitionInfoList[j].startVertex,
                                                               partitionInfoList[j].endVertex,
                                                               partitionInfoList[j].nodePointerOffset,
                                                               isActiveNodeListD, nodePointerListD,
                                                               partitionEdgeListD, degreeD, valueD,
                                                               nextActiveNodeListD);
                    cudaDeviceSynchronize();
                    gpuErrorcheck(cudaPeekAtLastError())
                }
            }
            cudaMemcpy(isActiveNodeList, nextActiveNodeListD, testNumNodes * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemset(nextActiveNodeListD, 0, testNumNodes * sizeof(bool));
            iteration++;
        }
        cout << " activeSum " << activeSum << endl;
        auto endRead = std::chrono::steady_clock::now();
        long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        cout << " finish time : " << durationRead << " ms" << endl;
        cout << "traverseSum " << traverseSum << endl;
    }

    free(nodePointersI);
    free(edgeList);
    free(degree);
    free(isActiveNodeList);
    cudaFree(isActiveNodeListD);
    cudaFree(nextActiveNodeListD);
    cudaFree(nodePointerListD);
    cudaFree(partitionEdgeListD);
    //todo free partitionInfoList needTransferPartition

}


long
bfsCaculateInShareReturnValue(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode,
                              SIZE_TYPE **bfsValue, int valueIndex) {
    auto start = std::chrono::steady_clock::now();
    SIZE_TYPE *degree;
    SIZE_TYPE *value;
    SIZE_TYPE sourceCode = 0;
    gpuErrorcheck(cudaMallocManaged(&degree, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMallocManaged(&value, testNumNodes * sizeof(SIZE_TYPE)));

    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate degree
    for (SIZE_TYPE i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    sourceCode = sourceNode;
    cout << "sourceNode " << sourceNode << " degree " << degree[sourceNode] << endl;
    bool *label;
    gpuErrorcheck(cudaMallocManaged(&label, testNumNodes * sizeof(bool)));
    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();

    label[sourceCode] = true;
    value[sourceCode] = 0;
    SIZE_TYPE *activeNodeList;
    cudaMallocManaged(&activeNodeList, testNumNodes * sizeof(SIZE_TYPE));
    //cacaulate the active node And make active node array
    SIZE_TYPE *activeNodeLabelingD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    SIZE_TYPE *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    SIZE_TYPE activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    SIZE_TYPE nodeSum = activeNodesNum;
    auto startProcessing = std::chrono::steady_clock::now();
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeList, label, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeList, label);
        bfs_kernel<<<grid, block>>>(activeNodesNum, activeNodeList, nodePointersI, degree, edgeList, value, label);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < testNumNodes; i++) {
        bfsValue[valueIndex][i] = value[i];
    }

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cudaFree(degree);
    cudaFree(label);
    cudaFree(value);
    cudaFree(activeNodeList);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    return durationRead;
}

void bfsShare(string bfsPath, int sampleSourceNode) {
    SIZE_TYPE testNumNodes = 0;
    ulong testNumEdge = 0;
    SIZE_TYPE *nodePointersI;
    SIZE_TYPE *edgeList;
    bool isUseShare = true;

    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(bfsPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(SIZE_TYPE));
    SIZE_TYPE numEdge = 0;
    infile.read((char *) &numEdge, sizeof(SIZE_TYPE));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(SIZE_TYPE)));
    infile.read((char *) nodePointersI, sizeof(SIZE_TYPE) * testNumNodes);
    gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(SIZE_TYPE)));
    cudaMemAdvise(nodePointersI, (testNumNodes + 1) * sizeof(SIZE_TYPE), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(edgeList, (numEdge) * sizeof(SIZE_TYPE), cudaMemAdviseSetReadMostly, 0);
    infile.read((char *) edgeList, sizeof(SIZE_TYPE) * testNumEdge);
    infile.close();
    //preprocessData(nodePointersI, edgeList, testNumNodes, testNumEdge);
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    int testTimes = 1;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        SIZE_TYPE sourceNode = rand() % testNumNodes;
        sourceNode = sampleSourceNode;
        cout << "sourceNode " << sourceNode << endl;
        timeSum += bfsCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList, sourceNode);
        //timeSum += bfsCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList, 53037907);
        break;
    }
}

void bfsOpt(string bfsPath, int sampleSourceNode, float adviseK) {
    SIZE_TYPE testNumNodes = 0;
    ulong testNumEdge = 0;
    SIZE_TYPE *nodePointersI;
    SIZE_TYPE *edgeList;
    bool isUseShare = true;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(bfsPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(SIZE_TYPE));
    SIZE_TYPE numEdge = 0;
    infile.read((char *) &numEdge, sizeof(SIZE_TYPE));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;

    nodePointersI = new SIZE_TYPE[testNumNodes + 1];
    infile.read((char *) nodePointersI, sizeof(SIZE_TYPE) * testNumNodes);
    edgeList = new SIZE_TYPE[testNumEdge + 1];
    infile.read((char *) edgeList, sizeof(SIZE_TYPE) * testNumEdge);
    infile.close();
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    int testTimes = 1;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        SIZE_TYPE sourceNode = rand() % testNumNodes;
        sourceNode = 47235513;
        sourceNode = 25838548;
        cout << "sourceNode " << sourceNode << endl;
        //timeSum += bfsCaculateInOpt(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);

        //caculateInOptChooseByDegree(testNumNodes, testNumEdge, nodePointersI, edgeList);
        //timeSum += bfsCaculateInAsync(testNumNodes, testNumEdge, nodePointersI, edgeList, 53037907);
        timeSum += bfsCaculateInAsyncNoUVMRandom(testNumNodes, testNumEdge, nodePointersI, edgeList, sourceNode,
                                                 adviseK);

        //timeSum += bfsCaculateInAsyncSwapOpt(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);

        //timeSum += bfsCaculateInAsyncSwapOptWithOverload(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);
        //timeSum += bfsCaculateInAsyncSwapManage(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);

        //break;
        cout << i << "========================================" << endl;
    }
    cudaFree(nodePointersI);
    cudaFree(edgeList);
}

void testBFS() {
    SIZE_TYPE testNumNodes = 0;
    ulong testNumEdge = 0;
    SIZE_TYPE *nodePointersI;
    SIZE_TYPE *edgeList;
    bool isUseShare = true;

    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(testGraphPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(SIZE_TYPE));
    SIZE_TYPE numEdge = 0;
    infile.read((char *) &numEdge, sizeof(SIZE_TYPE));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;

    if (isUseShare) {
        gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(SIZE_TYPE)));
        infile.read((char *) nodePointersI, sizeof(SIZE_TYPE) * testNumNodes);
        gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(SIZE_TYPE)));
        cudaMemAdvise(nodePointersI, (testNumNodes + 1) * sizeof(SIZE_TYPE), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(edgeList, (numEdge) * sizeof(SIZE_TYPE), cudaMemAdviseSetReadMostly, 0);
        infile.read((char *) edgeList, sizeof(SIZE_TYPE) * testNumEdge);
        infile.close();
        //preprocessData(nodePointersI, edgeList, testNumNodes, testNumEdge);

    } else {
        nodePointersI = new SIZE_TYPE[testNumNodes + 1];
        infile.read((char *) nodePointersI, sizeof(SIZE_TYPE) * testNumNodes);
        edgeList = new SIZE_TYPE[testNumEdge + 1];
        infile.read((char *) edgeList, sizeof(SIZE_TYPE) * testNumEdge);
        infile.close();
    }
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    int testTimes = 1;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        SIZE_TYPE sourceNode = rand() % testNumNodes;
        cout << "sourceNode " << sourceNode << endl;
        if (isUseShare) {
            //timeSum += bfsCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);
            timeSum += bfsCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList, 53037907);
            break;
        } else {
            //timeSum += bfsCaculateInOpt(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);

            //caculateInOptChooseByDegree(testNumNodes, testNumEdge, nodePointersI, edgeList);
            //timeSum += bfsCaculateInAsync(testNumNodes, testNumEdge, nodePointersI, edgeList, 53037907);
            timeSum += bfsCaculateInAsyncNoUVMSwap(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);

            //timeSum += bfsCaculateInAsyncSwapOpt(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);

            //timeSum += bfsCaculateInAsyncSwapOptWithOverload(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);
            //timeSum += bfsCaculateInAsyncSwapManage(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);

            //break;
        }
        cout << i << "========================================" << endl;
    }
    if (isUseShare) {
        cudaFree(nodePointersI);
        cudaFree(edgeList);
    } else {
        delete[]nodePointersI;
        delete[]edgeList;
    }
}


long bfsCaculateInShare(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode) {
    auto start = std::chrono::steady_clock::now();
    SIZE_TYPE *degree;
    SIZE_TYPE *value;
    SIZE_TYPE sourceCode = 0;
    gpuErrorcheck(cudaMallocManaged(&degree, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMallocManaged(&value, testNumNodes * sizeof(SIZE_TYPE)));

    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate degree
    for (SIZE_TYPE i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    sourceCode = sourceNode;
    cout << "sourceNode " << sourceNode << " degree " << degree[sourceNode] << endl;
    bool *label;
    gpuErrorcheck(cudaMallocManaged(&label, testNumNodes * sizeof(bool)));
    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    label[sourceCode] = true;
    value[sourceCode] = 1;
    SIZE_TYPE *activeNodeList;
    cudaMallocManaged(&activeNodeList, testNumNodes * sizeof(SIZE_TYPE));
    //cacaulate the active node And make active node array
    SIZE_TYPE *activeNodeLabelingD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    SIZE_TYPE *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    SIZE_TYPE activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    SIZE_TYPE nodeSum = activeNodesNum;
    auto startProcessing = std::chrono::steady_clock::now();
    //vector<vector<SIZE_TYPE>> visitRecordByIteration;
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeList, label, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeList, label);
        bfs_kernel<<<grid, block>>>(activeNodesNum, activeNodeList, nodePointersI, degree, edgeList, value, label);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        //visitRecordByIteration.push_back(countDataByIteration(testNumEdge, testNumNodes, nodePointersI, degree, activeNodeLabelingD));
        setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();
    //writeTrunkVistInIteration(visitRecordByIteration, "./CountByIterationbfs.txt");
    cout << "nodeSum: " << nodeSum << endl;

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "iter sum is " << iter << " finish time : " << durationRead << " ms" << endl;
    //cout << "range min " << rangeMin << " range max " << rangeMax << " range sum " << rangeSum << endl;
    cout << "source node pointer  " << nodePointersI[sourceNode] << endl;
    cudaFree(degree);
    cudaFree(label);
    cudaFree(value);
    cudaFree(activeNodeList);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    return durationRead;
}

long
bfsCaculateInAsyncNoUVMSwap(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode) {
    cout << "=========bfsCaculateInAsyncNoUVM========" << endl;
    auto start = std::chrono::steady_clock::now();
    auto startPreCaculate = std::chrono::steady_clock::now();
    //CPU
    long durationRead;
    SIZE_TYPE fragmentNum = testNumEdge / fragment_size;
    unsigned long max_partition_size;
    unsigned long total_gpu_size;
    SIZE_TYPE staticFragmentNum;
    SIZE_TYPE maxStaticNode = 0;
    SIZE_TYPE *degree;
    SIZE_TYPE *value;
    SIZE_TYPE *label;
    SIZE_TYPE *staticFragmentToNormalMap;
    bool *isInStatic;
    SIZE_TYPE *overloadNodeList;
    SIZE_TYPE *staticNodePointer;
    SIZE_TYPE *staticFragmentData;
    SIZE_TYPE *overloadFragmentData;
    SIZE_TYPE *activeNodeList;
    SIZE_TYPE *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    /*
     * overloadEdgeList overload edge list in every iteration
     * */
    SIZE_TYPE *overloadEdgeList;
    FragmentData *fragmentData;
    bool isFromTail = false;
    //GPU
    SIZE_TYPE *staticEdgeListD;
    SIZE_TYPE *overloadEdgeListD;
    bool *isInStaticD;
    SIZE_TYPE *overloadNodeListD;
    SIZE_TYPE *staticNodePointerD;
    SIZE_TYPE *staticFragmentVisitRecordsD;
    SIZE_TYPE *staticFragmentDataD;
    SIZE_TYPE *canSwapStaticFragmentDataD;
    SIZE_TYPE *canSwapFragmentPrefixSumD;
    SIZE_TYPE *degreeD;
    // async need two labels
    SIZE_TYPE *isActiveD1;
    SIZE_TYPE *isStaticActive;
    SIZE_TYPE *isOverloadActive;
    SIZE_TYPE *valueD;
    SIZE_TYPE *activeNodeListD;
    SIZE_TYPE *activeNodeLabelingPrefixD;
    SIZE_TYPE *overloadLabelingPrefixD;
    SIZE_TYPE *activeOverloadNodePointersD;
    SIZE_TYPE *activeOverloadDegreeD;

    degree = new SIZE_TYPE[testNumNodes];
    value = new SIZE_TYPE[testNumNodes];
    label = new SIZE_TYPE[testNumNodes];
    isInStatic = new bool[testNumNodes];
    overloadNodeList = new SIZE_TYPE[testNumNodes];
    staticNodePointer = new SIZE_TYPE[testNumNodes];
    activeNodeList = new SIZE_TYPE[testNumNodes];
    activeOverloadNodePointers = new SIZE_TYPE[testNumNodes];
    fragmentData = new FragmentData[fragmentNum];

    //getMaxPartitionSize(max_partition_size, testNumNodes);
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, 0.97, sizeof(SIZE_TYPE));
    staticFragmentNum = max_partition_size / fragment_size;
    staticFragmentToNormalMap = new SIZE_TYPE[staticFragmentNum];
    staticFragmentData = new SIZE_TYPE[staticFragmentNum];
    overloadFragmentData = new SIZE_TYPE[fragmentNum];
    //caculate degree
    SIZE_TYPE meanDegree = testNumEdge / testNumNodes;
    cout << " meanDegree " << meanDegree << endl;
    SIZE_TYPE degree0Sum = 0;
    for (SIZE_TYPE i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(SIZE_TYPE));

    //caculate static staticEdgeListD
    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(SIZE_TYPE)));
    auto startmove = std::chrono::steady_clock::now();
    gpuErrorcheck(
            cudaMemcpy(staticEdgeListD, edgeList, max_partition_size * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    auto endMove = std::chrono::steady_clock::now();
    long testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endMove - startmove).count();
    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(SIZE_TYPE)))
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, nodePointersI, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));

    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        label[i] = 0;
        value[i] = UINT_MAX - 1;

        SIZE_TYPE pointStartFragmentIndex = nodePointersI[i] / fragment_size;
        SIZE_TYPE pointEndFragmentIndex =
                degree[i] == 0 ? pointStartFragmentIndex : (nodePointersI[i] + degree[i] - 1) / fragment_size;
        if (pointStartFragmentIndex == pointEndFragmentIndex && pointStartFragmentIndex >= 0 &&
            pointStartFragmentIndex < fragmentNum) {
            if (fragmentData[pointStartFragmentIndex].vertexNum == 0) {
                fragmentData[pointStartFragmentIndex].startVertex = i;
            } else if (fragmentData[pointStartFragmentIndex].startVertex > i) {
                fragmentData[pointStartFragmentIndex].startVertex = i;
            }
            fragmentData[pointStartFragmentIndex].vertexNum++;
        }

        if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > maxStaticNode) maxStaticNode = i;
        } else {
            isInStatic[i] = false;
        }
    }
    label[sourceNode] = 1;
    value[sourceNode] = 1;
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    cout << "max_partition_size: " << max_partition_size << "  maxStaticNode: " << maxStaticNode << endl;
    cout << "fragmentNum " << fragmentNum << " staticFragmentNum " << staticFragmentNum << endl;
    for (int i = 0; i < staticFragmentNum; i++) {
        fragmentData[i].isIn = true;
    }
    for (SIZE_TYPE i = 0; i < staticFragmentNum; i++) {
        staticFragmentToNormalMap[i] = i;
    }
    //SIZE_TYPE partOverloadSize = max_partition_size / 2;
    SIZE_TYPE partOverloadSize = total_gpu_size - max_partition_size;
    SIZE_TYPE overloadSize = testNumEdge - nodePointersI[maxStaticNode + 1];
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    overloadEdgeList = (SIZE_TYPE *) malloc(overloadSize * sizeof(SIZE_TYPE));
    if (overloadEdgeList == NULL) {
        cout << "overloadEdgeList is null" << endl;
        return 0;
    }
    gpuErrorcheck(cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(SIZE_TYPE)));
    //gpuErrorcheck(cudaMallocManaged(&edgeListOverloadManage, overloadSize * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&staticFragmentDataD, staticFragmentNum * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&staticFragmentVisitRecordsD, staticFragmentNum * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&canSwapStaticFragmentDataD, staticFragmentNum * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&canSwapFragmentPrefixSumD, staticFragmentNum * sizeof(SIZE_TYPE)));
    thrust::device_ptr<unsigned int> ptr_canSwapFragment(canSwapStaticFragmentDataD);
    thrust::device_ptr<unsigned int> ptr_canSwapFragmentPrefixSum(canSwapFragmentPrefixSumD);
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isActiveD1, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isStaticActive, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isOverloadActive, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&overloadLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegreeD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(isActiveD1, label, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(SIZE_TYPE)));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD1);
    thrust::device_ptr<unsigned int> ptr_labeling_static(isStaticActive);
    thrust::device_ptr<unsigned int> ptr_labeling_overload(isOverloadActive);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegreeD);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(overloadLabelingPrefixD);

    SIZE_TYPE activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    SIZE_TYPE nodeSum = activeNodesNum;
    SIZE_TYPE overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startOverloadGpuProcessing = std::chrono::steady_clock::now();
    auto endOverloadGpuProcessing = std::chrono::steady_clock::now();
    long durationOverloadGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);
    auto startMemoryTraverse = std::chrono::steady_clock::now();
    auto endMemoryTraverse = std::chrono::steady_clock::now();
    long durationMemoryTraverse = 0;
    auto startProcessing = std::chrono::steady_clock::now();
    SIZE_TYPE cursorStartSwap = isFromTail ? fragmentNum - 1 : staticFragmentNum + 1;
    //SIZE_TYPE cursorStartSwap = staticFragmentNum + 1;
    SIZE_TYPE swapValidNodeSum = 0;
    SIZE_TYPE swapValidEdgeSum = 0;
    SIZE_TYPE swapNotValidNodeSum = 0;
    SIZE_TYPE swapNotValidEdgeSum = 0;
    SIZE_TYPE visitEdgeSum = 0;
    SIZE_TYPE swapInEdgeSum = 0;
    SIZE_TYPE partOverloadSum = 0;
    while (activeNodesNum > 0) {
        startPreGpuProcessing = std::chrono::steady_clock::now();
        iter++;
        cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
        setStaticAndOverloadLabel<<<grid, block>>>(testNumNodes, isActiveD1, isStaticActive, isOverloadActive,
                                                   isInStaticD);
        SIZE_TYPE staticNodeNum = thrust::reduce(ptr_labeling_static, ptr_labeling_static + testNumNodes);
        if (staticNodeNum > 0) {
            cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;
            thrust::exclusive_scan(ptr_labeling_static, ptr_labeling_static + testNumNodes, ptr_labeling_prefixsum);
            setStaticActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, isStaticActive,
                                                      activeNodeLabelingPrefixD);
        }

        SIZE_TYPE overloadNodeNum = thrust::reduce(ptr_labeling_overload, ptr_labeling_overload + testNumNodes);
        SIZE_TYPE overloadEdgeNum = 0;
        if (overloadNodeNum > 0) {
            cout << "iter " << iter << " overloadNodeNum " << overloadNodeNum << endl;
            thrust::exclusive_scan(ptr_labeling_overload, ptr_labeling_overload + testNumNodes, ptrOverloadPrefixsum);
            setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegreeD,
                                                        isOverloadActive,
                                                        overloadLabelingPrefixD, degreeD);
            thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum, activeOverloadNodePointersD);
            overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                             ptrOverloadDegree + overloadNodeNum, 0);
            cout << "iter " << iter << " overloadEdgeNum " << overloadEdgeNum << endl;
            overloadEdgeSum += overloadEdgeNum;

        }
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
        startGpuProcessing = std::chrono::steady_clock::now();
        if (staticNodeNum > 0) {
            setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD, isActiveD1);
        }
        if (overloadNodeNum > 0) {
            setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(overloadNodeNum, overloadNodeListD, isActiveD1);
        }
        bfs_kernelStaticSwap<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                              staticNodePointerD, degreeD,
                                                              staticEdgeListD, valueD, isActiveD1,
                                                              staticFragmentVisitRecordsD, fragment_size,
                                                              max_partition_size, testNumNodes);
        /*mixDynamicPartLabel<<<grid, block, 0, steamStatic>>>(staticNodeNum, 0, activeNodeListD, isActiveD1,
                                                             isActiveD2);

        bfs_kernelStatic2Label<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                                        staticNodePointerD, degreeD,
                                                                        staticEdgeListD, valueD, isActiveD1, isActiveD2);*/
        if (overloadNodeNum > 0) {
            startCpu = std::chrono::steady_clock::now();
            /*cudaMemcpyAsync(staticActiveNodeList, activeNodeListD, activeNodesNum * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost,
                            streamDynamic);*/
            cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost,
                            streamDynamic);
            cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(SIZE_TYPE),
                            cudaMemcpyDeviceToHost, streamDynamic);
            int threadNum = 20;
            if (overloadNodeNum < 50) {
                threadNum = 1;
            }
            thread runThreads[threadNum];

            for (int i = 0; i < threadNum; i++) {
                runThreads[i] = thread(fillDynamic,
                                       i,
                                       threadNum,
                                       0,
                                       overloadNodeNum,
                                       degree,
                                       activeOverloadNodePointers,
                                       nodePointersI,
                                       overloadNodeList,
                                       overloadEdgeList,
                                       edgeList);
            }

            for (unsigned int t = 0; t < threadNum; t++) {
                runThreads[t].join();
            }
            caculatePartInfoForEdgeList(activeOverloadNodePointers, overloadNodeList, degree, partEdgeListInfoArr,
                                        overloadNodeNum, partOverloadSize, overloadEdgeNum);

            endReadCpu = std::chrono::steady_clock::now();
            durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
            SIZE_TYPE canSwapFragmentNum;
            setFragmentDataOpt<<<grid, block, 0, steamStatic>>>(canSwapStaticFragmentDataD, staticFragmentNum,
                                                                staticFragmentVisitRecordsD);
            canSwapFragmentNum = thrust::reduce(ptr_canSwapFragment, ptr_canSwapFragment + staticFragmentNum);
            if (canSwapFragmentNum > 0) {
                thrust::exclusive_scan(ptr_canSwapFragment, ptr_canSwapFragment + staticFragmentNum,
                                       ptr_canSwapFragmentPrefixSum);
                setStaticFragmentData<<<grid, block, 0, steamStatic>>>(staticFragmentNum, canSwapStaticFragmentDataD,
                                                                       canSwapFragmentPrefixSumD, staticFragmentDataD);
                cudaMemcpyAsync(staticFragmentData, staticFragmentDataD, canSwapFragmentNum * sizeof(SIZE_TYPE),
                                cudaMemcpyDeviceToHost, steamStatic);
            }
            cudaDeviceSynchronize();
            //gpuErrorcheck(cudaPeekAtLastError())
            endGpuProcessing = std::chrono::steady_clock::now();
            durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endGpuProcessing - startGpuProcessing).count();
            startMemoryTraverse = std::chrono::steady_clock::now();
            partOverloadSum += partEdgeListInfoArr.size();
            for (int i = 0; i < partEdgeListInfoArr.size(); i++) {
                gpuErrorcheck(cudaMemcpy(overloadEdgeListD, overloadEdgeList +
                                                            activeOverloadNodePointers[partEdgeListInfoArr[i].partStartIndex],
                                         partEdgeListInfoArr[i].partEdgeNums * sizeof(SIZE_TYPE),
                                         cudaMemcpyHostToDevice))
                startOverloadGpuProcessing = std::chrono::steady_clock::now();
                /*mixDynamicPartLabel<<<grid, block, 0, streamDynamic>>>(partEdgeListInfoArr[i].partActiveNodeNums,
                                                                       partEdgeListInfoArr[i].partStartIndex,
                                                                       overloadNodeListD, isActiveD1,
                                                                       isActiveD2);*/
                bfs_kernelDynamicPart<<<grid, block, 0, streamDynamic>>>(
                        partEdgeListInfoArr[i].partStartIndex,
                        partEdgeListInfoArr[i].partActiveNodeNums,
                        overloadNodeListD, degreeD,
                        valueD, isActiveD1,
                        overloadEdgeListD,
                        activeOverloadNodePointersD);
                if (canSwapFragmentNum > 0) {
                    startSwap = std::chrono::steady_clock::now();
                    SIZE_TYPE canSwapStaticFragmentIndex = 0;
                    SIZE_TYPE swapSum = 0;
                    //for (SIZE_TYPE i = cursorStartSwap; i > 0; i--) {
                    for (SIZE_TYPE i = cursorStartSwap; i < fragmentNum; i++) {
                        if (cudaSuccess == cudaStreamQuery(streamDynamic) ||
                            canSwapStaticFragmentIndex >= canSwapFragmentNum) {
                            if (i < fragmentNum) {
                                cursorStartSwap = i + 1;
                            }
                            cout << "iter " << iter << " swapSum " << swapSum << " swap to " << i << endl;
                            swapInEdgeSum += swapSum * fragment_size;
                            endSwap = std::chrono::steady_clock::now();
                            durationSwap += std::chrono::duration_cast<std::chrono::milliseconds>(
                                    endSwap - startOverloadGpuProcessing).count();
                            break;
                        }
                        if (cudaErrorNotReady == cudaStreamQuery(streamDynamic)) {
                            const FragmentData swapFragmentData = fragmentData[i];

                            if (!swapFragmentData.isVisit && !swapFragmentData.isIn && swapFragmentData.vertexNum > 0) {
                                SIZE_TYPE swapStaticFragmentIndex = staticFragmentData[canSwapStaticFragmentIndex++];
                                SIZE_TYPE beSwappedFragmentIndex = staticFragmentToNormalMap[swapStaticFragmentIndex];
                                fragmentData[beSwappedFragmentIndex].isVisit = true;
                                fragmentData[beSwappedFragmentIndex].isIn = false;
                                FragmentData beSwappedFragment = fragmentData[beSwappedFragmentIndex];
                                SIZE_TYPE moveFrom = testNumEdge;
                                SIZE_TYPE moveTo = testNumEdge;
                                SIZE_TYPE moveNum = testNumEdge;
                                if (beSwappedFragment.vertexNum > 0 && beSwappedFragmentIndex > 0 &&
                                    beSwappedFragmentIndex < fragmentNum) {
                                    for (SIZE_TYPE j = beSwappedFragment.startVertex - 1;
                                         j < beSwappedFragment.startVertex + beSwappedFragment.vertexNum + 1 &&
                                         j < testNumNodes; j++) {
                                        isInStatic[j] = false;
                                    }
                                    for (SIZE_TYPE j = swapFragmentData.startVertex;
                                         j < swapFragmentData.startVertex + swapFragmentData.vertexNum; j++) {
                                        isInStatic[j] = true;
                                        staticNodePointer[j] =
                                                nodePointersI[j] - i * fragment_size +
                                                swapStaticFragmentIndex * fragment_size;
                                    }
                                    moveFrom = nodePointersI[swapFragmentData.startVertex];
                                    moveTo = staticNodePointer[swapFragmentData.startVertex];
                                    moveNum = nodePointersI[swapFragmentData.startVertex + swapFragmentData.vertexNum] -
                                              nodePointersI[swapFragmentData.startVertex];
                                    cudaMemcpyAsync(staticEdgeListD + moveTo, edgeList + moveFrom,
                                                    moveNum * sizeof(SIZE_TYPE),
                                                    cudaMemcpyHostToDevice, steamStatic);
                                    cudaMemcpyAsync(isInStaticD + beSwappedFragment.startVertex - 1,
                                                    isInStatic + beSwappedFragment.startVertex - 1,
                                                    (beSwappedFragment.vertexNum + 2) * sizeof(bool),
                                                    cudaMemcpyHostToDevice, steamStatic);
                                    cudaMemcpyAsync(isInStaticD + swapFragmentData.startVertex,
                                                    isInStatic + swapFragmentData.startVertex,
                                                    swapFragmentData.vertexNum * sizeof(bool),
                                                    cudaMemcpyHostToDevice, steamStatic);

                                    cudaMemcpyAsync(staticNodePointerD + swapFragmentData.startVertex,
                                                    staticNodePointer + swapFragmentData.startVertex,
                                                    swapFragmentData.vertexNum * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice,
                                                    steamStatic);
                                    staticFragmentToNormalMap[swapStaticFragmentIndex] = i;
                                    fragmentData[i].isIn = true;
                                    swapSum++;
                                }
                            }
                        }
                    }
                }
                cudaDeviceSynchronize();
                endOverloadGpuProcessing = std::chrono::steady_clock::now();
                durationOverloadGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endOverloadGpuProcessing - startOverloadGpuProcessing).count();
            }
            endMemoryTraverse = std::chrono::steady_clock::now();
            durationMemoryTraverse += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endMemoryTraverse - startMemoryTraverse).count();
            //gpuErrorcheck(cudaPeekAtLastError())
        } else {
            cudaDeviceSynchronize();
            endGpuProcessing = std::chrono::steady_clock::now();
            durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endGpuProcessing - startGpuProcessing).count();
        }

        cudaDeviceSynchronize();
        startPreGpuProcessing = std::chrono::steady_clock::now();
        /*mixCommonLabel<<<grid, block, 0, streamDynamic>>>(testNumNodes, isActiveD1, isActiveD2);
        cudaDeviceSynchronize();*/
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
    }
    //cudaDeviceSynchronize();
    auto endRead = std::chrono::steady_clock::now();
    durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
    cout << "durationOverloadGpuProcessing time : " << durationOverloadGpuProcessing << " ms" << endl;
    cout << "durationMemoryTraverse time : " << durationMemoryTraverse << " ms" << endl;

    cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cout << "partOverloadSum : " << partOverloadSum << " " << endl;
    cout << "nodeSum: " << nodeSum << endl;
    cudaFree(staticEdgeListD);
    cudaFree(degreeD);
    cudaFree(isActiveD1);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegreeD);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(staticFragmentVisitRecordsD);
    cudaFree(staticFragmentDataD);
    cudaFree(canSwapStaticFragmentDataD);
    cudaFree(canSwapFragmentPrefixSumD);
    cudaFree(overloadEdgeListD);
    cudaFree(isStaticActive);
    cudaFree(isOverloadActive);
    cudaFree(overloadLabelingPrefixD);
    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] staticFragmentData;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] staticNodePointer;
    delete[] staticFragmentToNormalMap;
    delete[] fragmentData;
    delete[] overloadFragmentData;
    delete[] overloadEdgeList;
    partEdgeListInfoArr.clear();
    return durationRead;
}

long
bfsCaculateInAsyncNoUVM(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode,
                        float adviseK) {
    cout << "=========bfsCaculateInAsyncNoUVM========" << endl;
    ulong edgeIterationMax = 0;
    auto start = std::chrono::steady_clock::now();
    auto startPreCaculate = std::chrono::steady_clock::now();
    //CPU
    long durationRead;
    ulong transferSum = 0;
    unsigned long max_partition_size;
    unsigned long total_gpu_size;
    SIZE_TYPE maxStaticNode = 0;
    SIZE_TYPE *degree;
    SIZE_TYPE *value;
    SIZE_TYPE *label;
    bool *isInStatic;
    SIZE_TYPE *overloadNodeList;
    SIZE_TYPE *staticNodePointer;
    SIZE_TYPE *activeNodeList;
    SIZE_TYPE *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    /*
     * overloadEdgeList overload edge list in every iteration
     * */
    SIZE_TYPE *overloadEdgeList;
    bool isFromTail = false;
    //GPU
    SIZE_TYPE *staticEdgeListD;
    SIZE_TYPE *overloadEdgeListD;
    bool *isInStaticD;
    SIZE_TYPE *overloadNodeListD;
    SIZE_TYPE *staticNodePointerD;
    SIZE_TYPE *degreeD;
    // async need two labels
    SIZE_TYPE *isActiveD1;
    SIZE_TYPE *isStaticActive;
    SIZE_TYPE *isOverloadActive;
    SIZE_TYPE *valueD;
    SIZE_TYPE *activeNodeListD;
    SIZE_TYPE *activeNodeLabelingPrefixD;
    SIZE_TYPE *overloadLabelingPrefixD;
    SIZE_TYPE *activeOverloadNodePointersD;
    SIZE_TYPE *activeOverloadDegreeD;

    degree = new SIZE_TYPE[testNumNodes];
    value = new SIZE_TYPE[testNumNodes];
    label = new SIZE_TYPE[testNumNodes];
    isInStatic = new bool[testNumNodes];
    overloadNodeList = new SIZE_TYPE[testNumNodes];
    staticNodePointer = new SIZE_TYPE[testNumNodes];
    activeNodeList = new SIZE_TYPE[testNumNodes];
    activeOverloadNodePointers = new SIZE_TYPE[testNumNodes];

    //getMaxPartitionSize(max_partition_size, testNumNodes);
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, adviseK, sizeof(SIZE_TYPE), testNumEdge, 15);
    //caculate degree
    SIZE_TYPE meanDegree = testNumEdge / testNumNodes;
    cout << " meanDegree " << meanDegree << endl;
    SIZE_TYPE degree0Sum = 0;

    for (SIZE_TYPE i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(SIZE_TYPE));

    //caculate static staticEdgeListD
    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(SIZE_TYPE)));
    auto startmove = std::chrono::steady_clock::now();
    gpuErrorcheck(
            cudaMemcpy(staticEdgeListD, edgeList, max_partition_size * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    auto endMove = std::chrono::steady_clock::now();
    long testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endMove - startmove).count();
    cout << "move duration " << testDuration << endl;
    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(SIZE_TYPE)))
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, nodePointersI, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));

    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        label[i] = 0;
        value[i] = UINT_MAX - 1;

        if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > maxStaticNode) maxStaticNode = i;
        } else {
            isInStatic[i] = false;
        }
    }
    label[sourceNode] = 1;
    value[sourceNode] = 1;
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    cout << "max_partition_size: " << max_partition_size << "  maxStaticNode: " << maxStaticNode << endl;

    //SIZE_TYPE partOverloadSize = max_partition_size / 2;
    SIZE_TYPE partOverloadSize = total_gpu_size - max_partition_size;
    SIZE_TYPE overloadSize = testNumEdge - nodePointersI[maxStaticNode + 1];
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    overloadEdgeList = (SIZE_TYPE *) malloc(overloadSize * sizeof(SIZE_TYPE));
    if (overloadEdgeList == NULL) {
        cout << "overloadEdgeList is null" << endl;
        return 0;
    }
    gpuErrorcheck(cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(SIZE_TYPE)));
    //gpuErrorcheck(cudaMallocManaged(&edgeListOverloadManage, overloadSize * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isActiveD1, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isStaticActive, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isOverloadActive, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&overloadLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegreeD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(isActiveD1, label, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(SIZE_TYPE)));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD1);
    thrust::device_ptr<unsigned int> ptr_labeling_static(isStaticActive);
    thrust::device_ptr<unsigned int> ptr_labeling_overload(isOverloadActive);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegreeD);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(overloadLabelingPrefixD);

    SIZE_TYPE activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    SIZE_TYPE nodeSum = activeNodesNum;
    ulong overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startOverloadGpuProcessing = std::chrono::steady_clock::now();
    auto endOverloadGpuProcessing = std::chrono::steady_clock::now();
    long durationOverloadGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);
    auto startMemoryTraverse = std::chrono::steady_clock::now();
    auto endMemoryTraverse = std::chrono::steady_clock::now();
    long durationMemoryTraverse = 0;
    auto startProcessing = std::chrono::steady_clock::now();
    //SIZE_TYPE cursorStartSwap = staticFragmentNum + 1;
    SIZE_TYPE swapValidNodeSum = 0;
    SIZE_TYPE swapValidEdgeSum = 0;
    SIZE_TYPE swapNotValidNodeSum = 0;
    SIZE_TYPE swapNotValidEdgeSum = 0;
    SIZE_TYPE visitEdgeSum = 0;
    SIZE_TYPE swapInEdgeSum = 0;
    SIZE_TYPE partOverloadSum = 0;
    while (activeNodesNum > 0) {
        startPreGpuProcessing = std::chrono::steady_clock::now();
        iter++;
        //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
        setStaticAndOverloadLabel<<<grid, block>>>(testNumNodes, isActiveD1, isStaticActive, isOverloadActive,
                                                   isInStaticD);
        SIZE_TYPE staticNodeNum = thrust::reduce(ptr_labeling_static, ptr_labeling_static + testNumNodes);
        if (staticNodeNum > 0) {
            //cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;
            thrust::exclusive_scan(ptr_labeling_static, ptr_labeling_static + testNumNodes, ptr_labeling_prefixsum);
            setStaticActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, isStaticActive,
                                                      activeNodeLabelingPrefixD);
        }

        SIZE_TYPE overloadNodeNum = thrust::reduce(ptr_labeling_overload, ptr_labeling_overload + testNumNodes);
        SIZE_TYPE overloadEdgeNum = 0;
        if (overloadNodeNum > 0) {
            //cout << "iter " << iter << " overloadNodeNum " << overloadNodeNum << endl;
            thrust::exclusive_scan(ptr_labeling_overload, ptr_labeling_overload + testNumNodes, ptrOverloadPrefixsum);
            setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegreeD,
                                                        isOverloadActive,
                                                        overloadLabelingPrefixD, degreeD);
            thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum, activeOverloadNodePointersD);
            overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                             ptrOverloadDegree + overloadNodeNum, 0);
            //cout << "iter " << iter << " overloadEdgeNum " << overloadEdgeNum << endl;
            overloadEdgeSum += overloadEdgeNum;
            if (overloadEdgeNum > edgeIterationMax) {
                edgeIterationMax = overloadEdgeNum;
            }
        }
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
        startGpuProcessing = std::chrono::steady_clock::now();
        if (staticNodeNum > 0) {
            setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD, isActiveD1);
        }
        if (overloadNodeNum > 0) {
            setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(overloadNodeNum, overloadNodeListD, isActiveD1);
        }
        bfs_kernelStatic<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                          staticNodePointerD, degreeD,
                                                          staticEdgeListD, valueD, isActiveD1);
        //cudaDeviceSynchronize();
        if (overloadNodeNum > 0) {
            startCpu = std::chrono::steady_clock::now();
            cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost,
                            streamDynamic);
            cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(SIZE_TYPE),
                            cudaMemcpyDeviceToHost, streamDynamic);
            int threadNum = 20;
            if (overloadNodeNum < 50) {
                threadNum = 1;
            }
            thread runThreads[threadNum];

            for (int i = 0; i < threadNum; i++) {
                runThreads[i] = thread(fillDynamic,
                                       i,
                                       threadNum,
                                       0,
                                       overloadNodeNum,
                                       degree,
                                       activeOverloadNodePointers,
                                       nodePointersI,
                                       overloadNodeList,
                                       overloadEdgeList,
                                       edgeList);
            }

            for (unsigned int t = 0; t < threadNum; t++) {
                runThreads[t].join();
            }
            caculatePartInfoForEdgeList(activeOverloadNodePointers, overloadNodeList, degree, partEdgeListInfoArr,
                                        overloadNodeNum, partOverloadSize, overloadEdgeNum);

            endReadCpu = std::chrono::steady_clock::now();
            durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
            cudaDeviceSynchronize();
            //gpuErrorcheck(cudaPeekAtLastError())
            endGpuProcessing = std::chrono::steady_clock::now();
            durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endGpuProcessing - startGpuProcessing).count();
            partOverloadSum += partEdgeListInfoArr.size();
            for (int i = 0; i < partEdgeListInfoArr.size(); i++) {
                startMemoryTraverse = std::chrono::steady_clock::now();
                gpuErrorcheck(cudaMemcpy(overloadEdgeListD, overloadEdgeList +
                                                            activeOverloadNodePointers[partEdgeListInfoArr[i].partStartIndex],
                                         partEdgeListInfoArr[i].partEdgeNums * sizeof(SIZE_TYPE),
                                         cudaMemcpyHostToDevice))
                transferSum += partEdgeListInfoArr[i].partEdgeNums;
                endMemoryTraverse = std::chrono::steady_clock::now();
                durationMemoryTraverse += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endMemoryTraverse - startMemoryTraverse).count();
                startOverloadGpuProcessing = std::chrono::steady_clock::now();
                bfs_kernelDynamicPart<<<grid, block, 0, streamDynamic>>>(
                        partEdgeListInfoArr[i].partStartIndex,
                        partEdgeListInfoArr[i].partActiveNodeNums,
                        overloadNodeListD, degreeD,
                        valueD, isActiveD1,
                        overloadEdgeListD,
                        activeOverloadNodePointersD);
                cudaDeviceSynchronize();
                endOverloadGpuProcessing = std::chrono::steady_clock::now();
                durationOverloadGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endOverloadGpuProcessing - startOverloadGpuProcessing).count();
            }
            //gpuErrorcheck(cudaPeekAtLastError())
        } else {
            cudaDeviceSynchronize();
            endGpuProcessing = std::chrono::steady_clock::now();
            durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endGpuProcessing - startGpuProcessing).count();
        }

        cudaDeviceSynchronize();
        startPreGpuProcessing = std::chrono::steady_clock::now();
        /*mixCommonLabel<<<grid, block, 0, streamDynamic>>>(testNumNodes, isActiveD1, isActiveD2);
        cudaDeviceSynchronize();*/
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
    }
    //cudaDeviceSynchronize();
    auto endRead = std::chrono::steady_clock::now();
    durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    transferSum += max_partition_size;

    cout << "iterationSum " << iter << endl;
    double edgeIterationAvg = (double) overloadEdgeSum / (double) testNumEdge / iter;
    double edgeIterationMaxAvg = (double) edgeIterationMax / (double) testNumEdge;
    cout << "edgeIterationAvg " << edgeIterationAvg << " edgeIterationMaxAvg " << edgeIterationMaxAvg << endl;

    cout << "transferSum : " << transferSum * 4 << " byte" << endl;
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "total time : " << durationRead + testDuration << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
    cout << "durationOverloadGpuProcessing time : " << durationOverloadGpuProcessing << " ms" << endl;
    cout << "durationMemoryTraverse time : " << durationMemoryTraverse << " mskail" << endl;

    cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cout << "partOverloadSum : " << partOverloadSum << " " << endl;
    cout << "nodeSum: " << nodeSum << endl;
    cudaFree(staticEdgeListD);
    cudaFree(degreeD);
    cudaFree(isActiveD1);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegreeD);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(overloadEdgeListD);
    cudaFree(isStaticActive);
    cudaFree(isOverloadActive);
    cudaFree(overloadLabelingPrefixD);
    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] staticNodePointer;
    delete[] overloadEdgeList;
    partEdgeListInfoArr.clear();
    return durationRead;
}

long
bfsCaculateInAsyncNoUVMVisitRecord(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList,
                                   SIZE_TYPE sourceNode,
                                   float adviseK) {
    cout << "=========bfsCaculateInAsyncNoUVM========" << endl;
    ulong edgeIterationMax = 0;
    auto start = std::chrono::steady_clock::now();
    auto startPreCaculate = std::chrono::steady_clock::now();
    //CPU
    long durationRead;
    ulong transferSum = 0;
    unsigned long max_partition_size;
    unsigned long total_gpu_size;
    SIZE_TYPE maxStaticNode = 0;
    SIZE_TYPE *degree;
    SIZE_TYPE *value;
    SIZE_TYPE *label;
    bool *isInStatic;
    SIZE_TYPE *overloadNodeList;
    SIZE_TYPE *staticNodePointer;
    SIZE_TYPE *activeNodeList;
    SIZE_TYPE *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    /*
     * overloadEdgeList overload edge list in every iteration
     * */
    SIZE_TYPE *overloadEdgeList;
    bool isFromTail = false;
    //GPU
    SIZE_TYPE *staticEdgeListD;
    SIZE_TYPE *overloadEdgeListD;
    bool *isInStaticD;
    SIZE_TYPE *overloadNodeListD;
    SIZE_TYPE *staticNodePointerD;
    SIZE_TYPE *degreeD;
    // async need two labels
    SIZE_TYPE *isActiveD1;
    SIZE_TYPE *isStaticActive;
    SIZE_TYPE *isOverloadActive;
    SIZE_TYPE *valueD;
    SIZE_TYPE *activeNodeListD;
    SIZE_TYPE *activeNodeLabelingPrefixD;
    SIZE_TYPE *overloadLabelingPrefixD;
    SIZE_TYPE *activeOverloadNodePointersD;
    SIZE_TYPE *activeOverloadDegreeD;

    degree = new SIZE_TYPE[testNumNodes];
    value = new SIZE_TYPE[testNumNodes];
    label = new SIZE_TYPE[testNumNodes];
    isInStatic = new bool[testNumNodes];
    overloadNodeList = new SIZE_TYPE[testNumNodes];
    staticNodePointer = new SIZE_TYPE[testNumNodes];
    activeNodeList = new SIZE_TYPE[testNumNodes];
    activeOverloadNodePointers = new SIZE_TYPE[testNumNodes];
    SIZE_TYPE *vertexVisitRecord;
    SIZE_TYPE *vertexVisitRecordD;
    vertexVisitRecord = new SIZE_TYPE[testNumNodes];
    cudaMalloc(&vertexVisitRecordD, testNumNodes * sizeof(SIZE_TYPE));
    cudaMemset(vertexVisitRecordD, 0, testNumNodes * sizeof(SIZE_TYPE));

    //getMaxPartitionSize(max_partition_size, testNumNodes);
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, adviseK, sizeof(SIZE_TYPE), testNumEdge, 15);
    //caculate degree
    SIZE_TYPE meanDegree = testNumEdge / testNumNodes;
    cout << " meanDegree " << meanDegree << endl;
    SIZE_TYPE degree0Sum = 0;
    for (SIZE_TYPE i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(SIZE_TYPE));

    //caculate static staticEdgeListD
    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(SIZE_TYPE)));
    auto startmove = std::chrono::steady_clock::now();
    gpuErrorcheck(
            cudaMemcpy(staticEdgeListD, edgeList, max_partition_size * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    auto endMove = std::chrono::steady_clock::now();
    long testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endMove - startmove).count();
    cout << "move duration " << testDuration << endl;
    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(SIZE_TYPE)))
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, nodePointersI, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));

    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        label[i] = 0;
        value[i] = UINT_MAX - 1;

        if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > maxStaticNode) maxStaticNode = i;
        } else {
            isInStatic[i] = false;
        }
    }
    label[sourceNode] = 1;
    value[sourceNode] = 1;
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    cout << "max_partition_size: " << max_partition_size << "  maxStaticNode: " << maxStaticNode << endl;

    //SIZE_TYPE partOverloadSize = max_partition_size / 2;
    SIZE_TYPE partOverloadSize = total_gpu_size - max_partition_size;
    SIZE_TYPE overloadSize = testNumEdge - nodePointersI[maxStaticNode + 1];
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    overloadEdgeList = (SIZE_TYPE *) malloc(overloadSize * sizeof(SIZE_TYPE));
    if (overloadEdgeList == NULL) {
        cout << "overloadEdgeList is null" << endl;
        return 0;
    }
    gpuErrorcheck(cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(SIZE_TYPE)));
    //gpuErrorcheck(cudaMallocManaged(&edgeListOverloadManage, overloadSize * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isActiveD1, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isStaticActive, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isOverloadActive, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&overloadLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegreeD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(isActiveD1, label, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(SIZE_TYPE)));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD1);
    thrust::device_ptr<unsigned int> ptr_labeling_static(isStaticActive);
    thrust::device_ptr<unsigned int> ptr_labeling_overload(isOverloadActive);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegreeD);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(overloadLabelingPrefixD);

    SIZE_TYPE activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    SIZE_TYPE nodeSum = activeNodesNum;
    ulong overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startOverloadGpuProcessing = std::chrono::steady_clock::now();
    auto endOverloadGpuProcessing = std::chrono::steady_clock::now();
    long durationOverloadGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);
    auto startMemoryTraverse = std::chrono::steady_clock::now();
    auto endMemoryTraverse = std::chrono::steady_clock::now();
    long durationMemoryTraverse = 0;
    auto startProcessing = std::chrono::steady_clock::now();
    //SIZE_TYPE cursorStartSwap = staticFragmentNum + 1;
    SIZE_TYPE swapValidNodeSum = 0;
    SIZE_TYPE swapValidEdgeSum = 0;
    SIZE_TYPE swapNotValidNodeSum = 0;
    SIZE_TYPE swapNotValidEdgeSum = 0;
    SIZE_TYPE visitEdgeSum = 0;
    SIZE_TYPE swapInEdgeSum = 0;
    SIZE_TYPE partOverloadSum = 0;
    while (activeNodesNum > 0) {
        startPreGpuProcessing = std::chrono::steady_clock::now();
        iter++;
        //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
        setStaticAndOverloadLabelAndRecord<<<grid, block>>>(testNumNodes, isActiveD1, isStaticActive, isOverloadActive,
                                                            isInStaticD, vertexVisitRecordD);
        SIZE_TYPE staticNodeNum = thrust::reduce(ptr_labeling_static, ptr_labeling_static + testNumNodes);
        if (staticNodeNum > 0) {
            //cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;
            thrust::exclusive_scan(ptr_labeling_static, ptr_labeling_static + testNumNodes, ptr_labeling_prefixsum);
            setStaticActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, isStaticActive,
                                                      activeNodeLabelingPrefixD);
        }

        SIZE_TYPE overloadNodeNum = thrust::reduce(ptr_labeling_overload, ptr_labeling_overload + testNumNodes);
        SIZE_TYPE overloadEdgeNum = 0;
        if (overloadNodeNum > 0) {
            //cout << "iter " << iter << " overloadNodeNum " << overloadNodeNum << endl;
            thrust::exclusive_scan(ptr_labeling_overload, ptr_labeling_overload + testNumNodes, ptrOverloadPrefixsum);
            setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegreeD,
                                                        isOverloadActive,
                                                        overloadLabelingPrefixD, degreeD);
            thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum, activeOverloadNodePointersD);
            overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                             ptrOverloadDegree + overloadNodeNum, 0);
            //cout << "iter " << iter << " overloadEdgeNum " << overloadEdgeNum << endl;
            overloadEdgeSum += overloadEdgeNum;
            if (overloadEdgeNum > edgeIterationMax) {
                edgeIterationMax = overloadEdgeNum;
            }
        }
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
        startGpuProcessing = std::chrono::steady_clock::now();
        if (staticNodeNum > 0) {
            setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD, isActiveD1);
        }
        if (overloadNodeNum > 0) {
            setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(overloadNodeNum, overloadNodeListD, isActiveD1);
        }
        bfs_kernelStatic<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                          staticNodePointerD, degreeD,
                                                          staticEdgeListD, valueD, isActiveD1);
        cudaDeviceSynchronize();
        if (overloadNodeNum > 0) {
            startCpu = std::chrono::steady_clock::now();
            cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost,
                            streamDynamic);
            cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(SIZE_TYPE),
                            cudaMemcpyDeviceToHost, streamDynamic);
            int threadNum = 20;
            if (overloadNodeNum < 50) {
                threadNum = 1;
            }
            thread runThreads[threadNum];

            for (int i = 0; i < threadNum; i++) {
                runThreads[i] = thread(fillDynamic,
                                       i,
                                       threadNum,
                                       0,
                                       overloadNodeNum,
                                       degree,
                                       activeOverloadNodePointers,
                                       nodePointersI,
                                       overloadNodeList,
                                       overloadEdgeList,
                                       edgeList);
            }

            for (unsigned int t = 0; t < threadNum; t++) {
                runThreads[t].join();
            }
            caculatePartInfoForEdgeList(activeOverloadNodePointers, overloadNodeList, degree, partEdgeListInfoArr,
                                        overloadNodeNum, partOverloadSize, overloadEdgeNum);

            endReadCpu = std::chrono::steady_clock::now();
            durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
            cudaDeviceSynchronize();
            //gpuErrorcheck(cudaPeekAtLastError())
            endGpuProcessing = std::chrono::steady_clock::now();
            durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endGpuProcessing - startGpuProcessing).count();
            startMemoryTraverse = std::chrono::steady_clock::now();
            partOverloadSum += partEdgeListInfoArr.size();
            for (int i = 0; i < partEdgeListInfoArr.size(); i++) {
                gpuErrorcheck(cudaMemcpy(overloadEdgeListD, overloadEdgeList +
                                                            activeOverloadNodePointers[partEdgeListInfoArr[i].partStartIndex],
                                         partEdgeListInfoArr[i].partEdgeNums * sizeof(SIZE_TYPE),
                                         cudaMemcpyHostToDevice))
                transferSum += partEdgeListInfoArr[i].partEdgeNums;
                startOverloadGpuProcessing = std::chrono::steady_clock::now();
                bfs_kernelDynamicPart<<<grid, block, 0, streamDynamic>>>(
                        partEdgeListInfoArr[i].partStartIndex,
                        partEdgeListInfoArr[i].partActiveNodeNums,
                        overloadNodeListD, degreeD,
                        valueD, isActiveD1,
                        overloadEdgeListD,
                        activeOverloadNodePointersD);
                cudaDeviceSynchronize();
                endOverloadGpuProcessing = std::chrono::steady_clock::now();
                durationOverloadGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endOverloadGpuProcessing - startOverloadGpuProcessing).count();
            }
            endMemoryTraverse = std::chrono::steady_clock::now();
            durationMemoryTraverse += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endMemoryTraverse - startMemoryTraverse).count();
            //gpuErrorcheck(cudaPeekAtLastError())
        } else {
            cudaDeviceSynchronize();
            endGpuProcessing = std::chrono::steady_clock::now();
            durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endGpuProcessing - startGpuProcessing).count();
        }

        cudaDeviceSynchronize();
        startPreGpuProcessing = std::chrono::steady_clock::now();
        /*mixCommonLabel<<<grid, block, 0, streamDynamic>>>(testNumNodes, isActiveD1, isActiveD2);
        cudaDeviceSynchronize();*/
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
    }
    //cudaDeviceSynchronize();
    auto endRead = std::chrono::steady_clock::now();
    durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    transferSum += max_partition_size;
    cudaMemcpy(vertexVisitRecord, vertexVisitRecordD, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
    SIZE_TYPE partNum = 50;
    SIZE_TYPE partSize = testNumEdge / partNum;
    vector<SIZE_TYPE> partVistRecordList(partNum + 1);
    SIZE_TYPE partSizeCursor = 0;
    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        SIZE_TYPE edgeStartIndex = nodePointersI[i];
        SIZE_TYPE edgeEndIndex = nodePointersI[i] + degree[i];
        SIZE_TYPE maxPartIndex = partSizeCursor * partSize + partSize;

        if (edgeStartIndex < maxPartIndex && edgeEndIndex < maxPartIndex) {
            partVistRecordList[partSizeCursor] += vertexVisitRecord[i] * degree[i];
        } else if (edgeStartIndex < maxPartIndex && edgeEndIndex >= maxPartIndex) {
            partVistRecordList[partSizeCursor] += vertexVisitRecord[i] * (maxPartIndex - edgeStartIndex);
            partSizeCursor += 1;
            partVistRecordList[partSizeCursor] += vertexVisitRecord[i] * (edgeEndIndex - maxPartIndex);
        } else {
            partSizeCursor += 1;
            partVistRecordList[partSizeCursor] += vertexVisitRecord[i] * degree[i];
        }
    }
    for (SIZE_TYPE i = 0; i < partNum + 1; i++) {
        cout << "part " << i << " is " << partVistRecordList[i] << endl;
    }
    for (SIZE_TYPE i = 0; i < partNum + 1; i++) {
        cout << partVistRecordList[i] << "\t";
    }
    cout << "iterationSum " << iter << endl;
    double edgeIterationAvg = (double) overloadEdgeSum / (double) testNumEdge / iter;
    double edgeIterationMaxAvg = (double) edgeIterationMax / (double) testNumEdge;
    cout << "edgeIterationAvg " << edgeIterationAvg << " edgeIterationMaxAvg " << edgeIterationMaxAvg << endl;

    cout << "transferSum : " << transferSum * 4 << " byte" << endl;
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "total time : " << durationRead + testDuration << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
    cout << "durationOverloadGpuProcessing time : " << durationOverloadGpuProcessing << " ms" << endl;
    cout << "durationMemoryTraverse time : " << durationMemoryTraverse << " mskail" << endl;

    cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cout << "partOverloadSum : " << partOverloadSum << " " << endl;
    cout << "nodeSum: " << nodeSum << endl;
    cudaFree(staticEdgeListD);
    cudaFree(degreeD);
    cudaFree(isActiveD1);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegreeD);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(overloadEdgeListD);
    cudaFree(isStaticActive);
    cudaFree(isOverloadActive);
    cudaFree(overloadLabelingPrefixD);
    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] staticNodePointer;
    delete[] overloadEdgeList;
    partEdgeListInfoArr.clear();
    return durationRead;
}


void bfsShareTrace(string bfsPath, int sampleSourceNode) {
    SIZE_TYPE testNumNodes = 0;
    ulong testNumEdge = 0;
    SIZE_TYPE *nodePointersI;
    SIZE_TYPE *edgeList;
    bool isUseShare = true;

    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(bfsPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(SIZE_TYPE));
    SIZE_TYPE numEdge = 0;
    infile.read((char *) &numEdge, sizeof(SIZE_TYPE));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    nodePointersI = new SIZE_TYPE[testNumNodes];
    infile.read((char *) nodePointersI, sizeof(SIZE_TYPE) * testNumNodes);
    gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(SIZE_TYPE)));
    cudaMemAdvise(nodePointersI, (testNumNodes + 1) * sizeof(SIZE_TYPE), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(edgeList, (numEdge) * sizeof(SIZE_TYPE), cudaMemAdviseSetReadMostly, 0);
    infile.read((char *) edgeList, sizeof(SIZE_TYPE) * testNumEdge);
    infile.close();
    //preprocessData(nodePointersI, edgeList, testNumNodes, testNumEdge);
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    int testTimes = 1;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        SIZE_TYPE sourceNode = rand() % testNumNodes;
        sourceNode = sampleSourceNode;
        cout << "sourceNode " << sourceNode << endl;
        timeSum += bfsCaculateInShareTrace(testNumNodes, testNumEdge, nodePointersI, edgeList, sourceNode);
        //timeSum += bfsCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList, 53037907);
        break;
    }
}


long
bfsCaculateInShareTrace(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode) {
    auto start = std::chrono::steady_clock::now();
    SIZE_TYPE *degree = new SIZE_TYPE[testNumNodes];
    SIZE_TYPE *value = new SIZE_TYPE[testNumNodes];
    SIZE_TYPE sourceCode = 0;

    auto startPreCaculate = std::chrono::steady_clock::now();
    for (SIZE_TYPE i = 0; i < testNumNodes - 1; i++) {
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }

    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    sourceCode = sourceNode;
    bool *label = new bool[testNumNodes];
    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }

    label[sourceCode] = true;
    value[sourceCode] = 1;
    SIZE_TYPE *activeNodeListD;
    SIZE_TYPE *degreeD;
    SIZE_TYPE *valueD;
    bool *labelD;
    SIZE_TYPE *nodePointersD;
    cudaMalloc(&activeNodeListD, testNumNodes * sizeof(SIZE_TYPE));
    cudaMalloc(&nodePointersD, testNumNodes * sizeof(SIZE_TYPE));
    cudaMalloc(&degreeD, testNumNodes * sizeof(SIZE_TYPE));
    cudaMalloc(&valueD, testNumNodes * sizeof(SIZE_TYPE));
    cudaMalloc(&labelD, testNumNodes * sizeof(bool));
    cudaMemcpy(degreeD, degree, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(valueD, value, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(labelD, label, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(nodePointersD, nodePointersI, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    //cacaulate the active node And make active node array
    SIZE_TYPE *activeNodeLabelingD;
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    SIZE_TYPE *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    SIZE_TYPE activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    SIZE_TYPE nodeSum = activeNodesNum;
    auto startProcessing = std::chrono::steady_clock::now();
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, labelD, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeListD, labelD);
        bfs_kernel<<<grid, block>>>(activeNodesNum, activeNodeListD, nodePointersD, degreeD, edgeList, valueD, labelD);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        long temp = 0;
        for (SIZE_TYPE j = 0; j < testNumEdge; j++) {
            temp += edgeList[j] % 10;
        }
        cout << "iter " << iter << " " << temp;
        setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();

    cout << "nodeSum: " << nodeSum << endl;

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "iter sum is " << iter << " finish time : " << durationRead << " ms" << endl;
    //cout << "range min " << rangeMin << " range max " << rangeMax << " range sum " << rangeSum << endl;
    cout << "source node pointer  " << nodePointersI[sourceNode] << endl;
    return durationRead;
}

long
bfsCaculateInAsyncNoUVMRandom(SIZE_TYPE testNumNodes, SIZE_TYPE testNumEdge, SIZE_TYPE *nodePointersI, SIZE_TYPE *edgeList, SIZE_TYPE sourceNode,
                              float adviseK) {
    cout << "=========bfsCaculateInAsyncNoUVM========" << endl;
    ulong edgeIterationMax = 0;
    auto start = std::chrono::steady_clock::now();
    auto startPreCaculate = std::chrono::steady_clock::now();
    //CPU
    long durationRead;
    ulong transferSum = 0;
    unsigned long max_partition_size;
    unsigned long total_gpu_size;
    SIZE_TYPE maxStaticNode = 0;
    SIZE_TYPE *degree;
    SIZE_TYPE *value;
    SIZE_TYPE *label;
    bool *isInStatic;
    SIZE_TYPE *overloadNodeList;
    SIZE_TYPE *staticNodePointer;
    SIZE_TYPE *activeNodeList;
    SIZE_TYPE *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    /*
     * overloadEdgeList overload edge list in every iteration
     * */
    SIZE_TYPE *overloadEdgeList;
    bool isFromTail = false;
    //GPU
    SIZE_TYPE *staticEdgeListD;
    SIZE_TYPE *overloadEdgeListD;
    bool *isInStaticD;
    SIZE_TYPE *overloadNodeListD;
    SIZE_TYPE *staticNodePointerD;
    SIZE_TYPE *degreeD;
    // async need two labels
    SIZE_TYPE *isActiveD1;
    SIZE_TYPE *isStaticActive;
    SIZE_TYPE *isOverloadActive;
    SIZE_TYPE *valueD;
    SIZE_TYPE *activeNodeListD;
    SIZE_TYPE *activeNodeLabelingPrefixD;
    SIZE_TYPE *overloadLabelingPrefixD;
    SIZE_TYPE *activeOverloadNodePointersD;
    SIZE_TYPE *activeOverloadDegreeD;

    degree = new SIZE_TYPE[testNumNodes];
    value = new SIZE_TYPE[testNumNodes];
    label = new SIZE_TYPE[testNumNodes];
    isInStatic = new bool[testNumNodes];
    overloadNodeList = new SIZE_TYPE[testNumNodes];
    staticNodePointer = new SIZE_TYPE[testNumNodes];
    activeNodeList = new SIZE_TYPE[testNumNodes];
    activeOverloadNodePointers = new SIZE_TYPE[testNumNodes];

    //getMaxPartitionSize(max_partition_size, testNumNodes);
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, adviseK, sizeof(SIZE_TYPE), testNumEdge, 15);
    calculateDegree(testNumNodes, nodePointersI, testNumEdge, degree);
    //memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(SIZE_TYPE));
    SIZE_TYPE edgesInStatic = 0;
    float startRate = (1 - (float) max_partition_size / (float) testNumEdge) / 2;
    SIZE_TYPE startIndex = (float) testNumNodes * startRate;
    /*SIZE_TYPE tempStaticSum = 0;
    for (SIZE_TYPE i = testNumNodes - 1; i >= 0; i--) {
        tempStaticSum += degree[i];
        if (tempStaticSum > max_partition_size) {
            startIndex = i;
            break;
        }
    }*/
    startIndex = 0;
    if (nodePointersI[startIndex] + max_partition_size > testNumEdge) {
        startIndex = (float) testNumNodes * 0.1f;
    }
    for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
        label[i] = 0;
        value[i] = UINT_MAX - 1;
        if (i >= startIndex && nodePointersI[i] < nodePointersI[startIndex] + max_partition_size - degree[i]) {
            isInStatic[i] = true;
            staticNodePointer[i] = nodePointersI[i] - nodePointersI[startIndex];
            if (i > maxStaticNode) {
                maxStaticNode = i;
            }
            edgesInStatic += degree[i];
        } else {
            isInStatic[i] = false;
        }
    }
    label[sourceNode] = 1;
    value[sourceNode] = 1;

    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(SIZE_TYPE)));
    auto startmove = std::chrono::steady_clock::now();
    gpuErrorcheck(
            cudaMemcpy(staticEdgeListD, edgeList + nodePointersI[startIndex], max_partition_size * sizeof(SIZE_TYPE),
                       cudaMemcpyHostToDevice));
    auto endMove = std::chrono::steady_clock::now();
    long testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endMove - startmove).count();
    cout << "move duration " << testDuration << endl;

    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(SIZE_TYPE)))
    gpuErrorcheck(
            cudaMemcpy(staticNodePointerD, staticNodePointer, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    label[sourceNode] = 1;
    value[sourceNode] = 1;
    cout << "max_partition_size: " << max_partition_size << "  maxStaticNode: " << maxStaticNode << endl;

    //SIZE_TYPE partOverloadSize = max_partition_size / 2;
    SIZE_TYPE partOverloadSize = total_gpu_size - max_partition_size;
    SIZE_TYPE overloadSize = testNumEdge - edgesInStatic;
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    overloadEdgeList = (SIZE_TYPE *) malloc(overloadSize * sizeof(SIZE_TYPE));
    if (overloadEdgeList == NULL) {
        cout << "overloadEdgeList is null" << endl;
        return 0;
    }
    gpuErrorcheck(cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(SIZE_TYPE)));
    //gpuErrorcheck(cudaMallocManaged(&edgeListOverloadManage, overloadSize * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isActiveD1, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isStaticActive, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&isOverloadActive, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&overloadLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegreeD, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(isActiveD1, label, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(SIZE_TYPE)));
    gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(SIZE_TYPE)));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD1);
    thrust::device_ptr<unsigned int> ptr_labeling_static(isStaticActive);
    thrust::device_ptr<unsigned int> ptr_labeling_overload(isOverloadActive);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegreeD);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(overloadLabelingPrefixD);

    SIZE_TYPE activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    SIZE_TYPE nodeSum = activeNodesNum;
    ulong overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startOverloadGpuProcessing = std::chrono::steady_clock::now();
    auto endOverloadGpuProcessing = std::chrono::steady_clock::now();
    long durationOverloadGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);
    auto startMemoryTraverse = std::chrono::steady_clock::now();
    auto endMemoryTraverse = std::chrono::steady_clock::now();
    long durationMemoryTraverse = 0;
    //SIZE_TYPE cursorStartSwap = staticFragmentNum + 1;
    SIZE_TYPE swapValidNodeSum = 0;
    SIZE_TYPE swapValidEdgeSum = 0;
    SIZE_TYPE swapNotValidNodeSum = 0;
    SIZE_TYPE swapNotValidEdgeSum = 0;
    SIZE_TYPE visitEdgeSum = 0;
    SIZE_TYPE swapInEdgeSum = 0;
    SIZE_TYPE partOverloadSum = 0;

    long TIME = 0;
    int testTimes = 1;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {

        for (SIZE_TYPE i = 0; i < testNumNodes; i++) {
            label[i] = 0;
            value[i] = UINT_MAX - 1;
        }
        label[sourceNode] = 1;
        value[sourceNode] = 1;
        cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
        gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(isActiveD1, label, testNumNodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(SIZE_TYPE)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(SIZE_TYPE)));
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        iter = 0;

        auto startProcessing = std::chrono::steady_clock::now();
        auto startTest = std::chrono::steady_clock::now();
        auto endTest = std::chrono::steady_clock::now();
        long durationTest = 0;
        while (activeNodesNum > 0) {
            startPreGpuProcessing = std::chrono::steady_clock::now();
            iter++;
            cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            setStaticAndOverloadLabel<<<grid, block>>>(testNumNodes, isActiveD1, isStaticActive, isOverloadActive,
                                                       isInStaticD);
            SIZE_TYPE staticNodeNum = thrust::reduce(ptr_labeling_static, ptr_labeling_static + testNumNodes);
            if (staticNodeNum > 0) {
                cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;
                thrust::exclusive_scan(ptr_labeling_static, ptr_labeling_static + testNumNodes, ptr_labeling_prefixsum);
                setStaticActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, isStaticActive,
                                                          activeNodeLabelingPrefixD);
            }

            SIZE_TYPE overloadNodeNum = thrust::reduce(ptr_labeling_overload, ptr_labeling_overload + testNumNodes);
            SIZE_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                cout << "iter " << iter << " overloadNodeNum " << overloadNodeNum << endl;
                thrust::exclusive_scan(ptr_labeling_overload, ptr_labeling_overload + testNumNodes,
                                       ptrOverloadPrefixsum);
                setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegreeD,
                                                            isOverloadActive,
                                                            overloadLabelingPrefixD, degreeD);
                thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum,
                                       activeOverloadNodePointersD);
                overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                                 ptrOverloadDegree + overloadNodeNum, 0);
                cout << "iter " << iter << " overloadEdgeNum " << overloadEdgeNum << endl;
                overloadEdgeSum += overloadEdgeNum;
                if (overloadEdgeNum > edgeIterationMax) {
                    edgeIterationMax = overloadEdgeNum;
                }
            }
            endPreGpuProcessing = std::chrono::steady_clock::now();
            durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endPreGpuProcessing - startPreGpuProcessing).count();
            startGpuProcessing = std::chrono::steady_clock::now();
            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD, isActiveD1);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(overloadNodeNum, overloadNodeListD, isActiveD1);
            }
            bfs_kernelStatic<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                              staticNodePointerD, degreeD,
                                                              staticEdgeListD, valueD, isActiveD1);
            //cudaDeviceSynchronize();
            if (overloadNodeNum > 0) {
                startCpu = std::chrono::steady_clock::now();
                cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(SIZE_TYPE),
                                cudaMemcpyDeviceToHost,
                                streamDynamic);
                cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(SIZE_TYPE),
                                cudaMemcpyDeviceToHost, streamDynamic);
                int threadNum = 20;
                if (overloadNodeNum < 50) {
                    threadNum = 1;
                }
                thread runThreads[threadNum];

                for (int i = 0; i < threadNum; i++) {
                    runThreads[i] = thread(fillDynamic,
                                           i,
                                           threadNum,
                                           0,
                                           overloadNodeNum,
                                           degree,
                                           activeOverloadNodePointers,
                                           nodePointersI,
                                           overloadNodeList,
                                           overloadEdgeList,
                                           edgeList);
                }

                for (unsigned int t = 0; t < threadNum; t++) {
                    runThreads[t].join();
                }
                caculatePartInfoForEdgeList(activeOverloadNodePointers, overloadNodeList, degree, partEdgeListInfoArr,
                                            overloadNodeNum, partOverloadSize, overloadEdgeNum);

                endReadCpu = std::chrono::steady_clock::now();
                durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
                cudaDeviceSynchronize();
                //gpuErrorcheck(cudaPeekAtLastError())
                endGpuProcessing = std::chrono::steady_clock::now();
                durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endGpuProcessing - startGpuProcessing).count();
                partOverloadSum += partEdgeListInfoArr.size();
                for (int i = 0; i < partEdgeListInfoArr.size(); i++) {
                    startMemoryTraverse = std::chrono::steady_clock::now();
                    gpuErrorcheck(cudaMemcpy(overloadEdgeListD, overloadEdgeList +
                                                                activeOverloadNodePointers[partEdgeListInfoArr[i].partStartIndex],
                                             partEdgeListInfoArr[i].partEdgeNums * sizeof(SIZE_TYPE),
                                             cudaMemcpyHostToDevice))
                    transferSum += partEdgeListInfoArr[i].partEdgeNums;
                    endMemoryTraverse = std::chrono::steady_clock::now();
                    durationMemoryTraverse += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endMemoryTraverse - startMemoryTraverse).count();
                    startOverloadGpuProcessing = std::chrono::steady_clock::now();
                    activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
                    cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
                    bfs_kernelDynamicPart<<<grid, block, 0, streamDynamic>>>(
                            partEdgeListInfoArr[i].partStartIndex,
                            partEdgeListInfoArr[i].partActiveNodeNums,
                            overloadNodeListD, degreeD,
                            valueD, isActiveD1,
                            overloadEdgeListD,
                            activeOverloadNodePointersD);
                    cudaDeviceSynchronize();
                    endOverloadGpuProcessing = std::chrono::steady_clock::now();
                    durationOverloadGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endOverloadGpuProcessing - startOverloadGpuProcessing).count();
                }
                //gpuErrorcheck(cudaPeekAtLastError())
            } else {
                cudaDeviceSynchronize();
                endGpuProcessing = std::chrono::steady_clock::now();
                durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endGpuProcessing - startGpuProcessing).count();
            }

            cudaDeviceSynchronize();
            startPreGpuProcessing = std::chrono::steady_clock::now();
            /*mixCommonLabel<<<grid, block, 0, streamDynamic>>>(testNumNodes, isActiveD1, isActiveD2);
            cudaDeviceSynchronize();*/
            activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
            nodeSum += activeNodesNum;
            endPreGpuProcessing = std::chrono::steady_clock::now();
            durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endPreGpuProcessing - startPreGpuProcessing).count();
            cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            if (iter == 2) {
                break;
            }
        }
        //cudaDeviceSynchronize();
        auto endRead = std::chrono::steady_clock::now();
        durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        transferSum += max_partition_size;

        cout << "iterationSum " << iter << endl;
        double edgeIterationAvg = (double) overloadEdgeSum / (double) testNumEdge / iter;
        double edgeIterationMaxAvg = (double) edgeIterationMax / (double) testNumEdge;
        cout << "edgeIterationAvg " << edgeIterationAvg << " edgeIterationMaxAvg " << edgeIterationMaxAvg << endl;

        cout << "transferSum : " << transferSum * 4 << " byte" << endl;
        cout << "finish time : " << durationRead << " ms" << endl;
        cout << "total time : " << durationRead + testDuration << " ms" << endl;
        cout << "cpu time : " << durationReadCpu << " ms" << endl;
        cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
        cout << "durationOverloadGpuProcessing time : " << durationOverloadGpuProcessing << " ms" << endl;
        cout << "durationMemoryTraverse time : " << durationMemoryTraverse << " mskail" << endl;

        cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
        cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
        cout << "partOverloadSum : " << partOverloadSum << " " << endl;
        cout << "nodeSum: " << nodeSum << endl;
        TIME += durationRead;

    }
    cout << "TIME " << (float) TIME / (float) testTimes << endl;
    cudaFree(staticEdgeListD);
    cudaFree(degreeD);
    cudaFree(isActiveD1);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegreeD);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(overloadEdgeListD);
    cudaFree(isStaticActive);
    cudaFree(isOverloadActive);
    cudaFree(overloadLabelingPrefixD);
    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] staticNodePointer;
    delete[] overloadEdgeList;
    partEdgeListInfoArr.clear();
    return durationRead;
}