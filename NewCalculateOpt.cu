#include "CalculateOpt.cuh"
#include "cpu_verify.cuh"

void newbfs_opt(string path, SIZE_TYPE sourceNode, double adviseRate,int model, int testTimes, double gpuMemoryLimit, bool verify){
    cout << "======NEW_bfs_opt=======" << endl;
    cout<<"sourceNode: "<<sourceNode<<endl;
    GraphMeta<SIZE_TYPE> graph;
    graph.setAlgType(BFS);
    graph.setmodel(model);
    graph.setGpuMemoryLimit(gpuMemoryLimit);
    if(model!=7)
    {
        cout<<"model not match"<<endl;
        return;
    }
    graph.setSourceNode(sourceNode);
    graph.readGraph(path, false);

    graph.setPrestoreRatio(adviseRate, 11);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();
    cout << "1test!!!!!!!!!" <<endl;
    
    graph.checkNode(sourceNode);

    cout << "2test!!!!!!!!!" <<endl;
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");

    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    
    totalProcess.startRecord();
    cout << "graph.vertexArrSize " << graph.vertexArrSize << endl;
    SIZE_TYPE activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<SIZE_TYPE>());
    //SIZE_TYPE staticmem = graph.ret_max_partition_size();

    cout << "3test!!!!!!!!!" <<endl;
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.endRecord();
    totalProcess.print();
    totalProcess.clearRecord();
    cout<<endl;
    //EDGE_POINTER_TYPE overloadEdges = 0; 

    cout<<"Start test new bfs, total test time: "<<testTimes<<endl;
    SIZE_TYPE src=graph.sourceNode;
    long totalduration = 0;
    long overloaduration = 0;
    long staticduration = 0;
    double overloadsize = 0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++){
        if(src<graph.vertexArrSize)
        graph.setSourceNode(src);
        else
        graph.setSourceNode(sourceNode);
        unsigned long long overloadedges = 0;
        cout<<"======iter "<<testIndex<<"======"<<endl;
        time_t now = time(0); 
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<SIZE_TYPE>());
        SIZE_TYPE nodeSum = activeNodesNum;

        int iter = 0;
        totalProcess.startRecord();
        while(activeNodesNum){
            iter++;
            //cout <<"iter "<<iter;
            //cout <<"iter "<<iter<< " activeNodesNum is " << activeNodesNum << " ";
            preProcess.startRecord();
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            SIZE_TYPE staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<SIZE_TYPE>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(graph.prefixSumTemp);
                    
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
            }
            //cout<<"staticNodeNum is "<<staticNodeNum;

            SIZE_TYPE overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<SIZE_TYPE>());
            

            if(overloadNodeNum>0){
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setActiveNodeList<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isOverloadActive,
                                                                graph.overloadNodeListD,
                                                                graph.prefixSumTemp);
            }
            
            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }
            preProcess.endRecord();
            
            staticProcess.startRecord();
            
            bfs_kernelStatic<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                graph.staticNodePointerD, graph.degreeD,
                                                                                graph.staticEdgeListD, graph.valueD,
                                                                                graph.isActiveD,graph.isInStaticD);
            
            gpuErrorcheck(cudaPeekAtLastError());
            
            if(overloadNodeNum > 0){
                
                overloadProcess.startRecord();
                uint64_t numthreads = 1024;
                uint64_t numblocks = ((overloadNodeNum * WARP_SIZE + numthreads) / numthreads);
                dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                SIZE_TYPE numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                New_bfs_kernelDynamicPart<<<blockDim, numthreads , 0, graph.streamDynamic>>>(overloadNodeNum, numwarps, graph.overloadNodeListD,
                                                                                            graph.valueD, graph.degreeD,
                                                                                            graph.isActiveD, graph.edgeArray,
                                                                                            graph.nodePointersD);
                gpuErrorcheck(cudaPeekAtLastError());
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(graph.streamDynamic);
                overloadProcess.endRecord(); 
                cudaStreamSynchronize(graph.steamStatic);
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
            preProcess.startRecord();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<SIZE_TYPE>());
            nodeSum += activeNodesNum;
            preProcess.endRecord();
            //overloadedges = 0; 
            // SIZE_TYPE *overloadnodes = new SIZE_TYPE[graph.vertexArrSize];
            // cudaMemcpy(overloadnodes,graph.overloadNodeListD,sizeof(SIZE_TYPE)*overloadNodeNum,cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            // for(SIZE_TYPE i=0;i<overloadNodeNum;i++){
            //     overloadedges += graph.degree[overloadnodes[i]];
            // }
            // double temp = (double)(overloadedges)*sizeof(SIZE_TYPE)/1024/1024;
            // cout<<temp<<endl;
        }
        totalProcess.endRecord();
        cout<<"total iter: "<<iter<<endl;
        totalProcess.print();
        preProcess.print();
        staticProcess.print();
        overloadProcess.print();

        cout << "nodeSum : " << nodeSum << endl;
        //cout << "move overload size : " << overloadEdges * sizeof(SIZE_TYPE) << endl;

        src+=graph.vertexArrSize/testTimes;
        totalduration+=totalProcess.getDuration();
        staticduration+=staticProcess.getDuration();
        overloaduration+=overloadProcess.getDuration();
        totalProcess.clearRecord();
        preProcess.clearRecord();
        staticProcess.clearRecord();
        overloadProcess.clearRecord();
        // double temp = (double)(overloadedges)*sizeof(SIZE_TYPE)/1024/1024/1024;
        // overloadsize+=temp;
        // cout<<"transfer "<<temp<<" GB"<<endl;

    }
    gpuErrorcheck(cudaPeekAtLastError());
    cout<<"========TEST OVER========"<<endl;
    cout<<"pre move data time: "<<graph.preMoveDataTime<<"ms"<<endl;
    cout<<"Test over, average total process time (including pre move data): "<<totalduration/testTimes + graph.preMoveDataTime<<"ms"<<endl;
    cout<<"average static process time: "<<staticduration/testTimes<<"ms"<<endl;
    cout<<"average overload process time: "<<overloaduration/testTimes<<"ms"<<endl;
    //cout<<"average transfer data: "<<overloadsize/testTimes<<" GB"<<endl;
    if (verify) {
        cudaMemcpy(graph.value, graph.valueD, graph.vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cpu_verify_bfs(graph.nodePointers, (SIZE_TYPE *)graph.edgeArray, graph.vertexArrSize, graph.edgeArrSize, sourceNode, graph.value);
    }
}

void newcc_opt(string path, double adviseRate,int model,int testTimes, double gpuMemoryLimit, bool verify){
    cout << "======NEW_cc_opt=======" << endl;
    //cout<<"sourceNode: "<<sourceNode<<endl;
    GraphMeta<SIZE_TYPE> graph;
    graph.setAlgType(CC);
    graph.setmodel(model);
    graph.setGpuMemoryLimit(gpuMemoryLimit);
    if(model!=7)
    {
        cout<<"model not match"<<endl;
        return;
    }
    //graph.setSourceNode(sourceNode);
    graph.readGraph(path, false);

    graph.setPrestoreRatio(adviseRate, 11);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();
    cout << "1test!!!!!!!!!" <<endl;
    
    //graph.checkNode(sourceNode);

    cout << "2test!!!!!!!!!" <<endl;
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");

    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    
    totalProcess.startRecord();
    cout << "graph.vertexArrSize " << graph.vertexArrSize << endl;
    SIZE_TYPE activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<SIZE_TYPE>());
    //SIZE_TYPE staticmem = graph.ret_max_partition_size();

    cout << "3test!!!!!!!!!" <<endl;
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.endRecord();
    totalProcess.print();
    totalProcess.clearRecord();
    cout<<endl;
    //EDGE_POINTER_TYPE overloadEdges = 0; 

    cout<<"Start test new cc, total test time: "<<testTimes<<endl;
    //SIZE_TYPE src=graph.sourceNode;
    long totalduration = 0;
    long overloaduration = 0;
    long staticduration = 0;
    double overloadsize = 0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++){
        
        unsigned long long overloadedges = 0;
        cout<<"======iter "<<testIndex<<"======"<<endl;
        time_t now = time(0); 
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<SIZE_TYPE>());
        SIZE_TYPE nodeSum = activeNodesNum;

        int iter = 0;
        totalProcess.startRecord();
        while(activeNodesNum){
            iter++;
            //cout <<"iter "<<iter;
            cout <<"iter "<<iter<< " activeNodesNum is " << activeNodesNum << " ";
            preProcess.startRecord();
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            SIZE_TYPE staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<SIZE_TYPE>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(graph.prefixSumTemp);
                    
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
            }
            cout<<"staticNodeNum is "<<staticNodeNum;

            SIZE_TYPE overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<SIZE_TYPE>());
            

            if(overloadNodeNum>0){
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setActiveNodeList<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isOverloadActive,
                                                                graph.overloadNodeListD,
                                                                graph.prefixSumTemp);
            }
            cudaDeviceSynchronize();
            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }
            cudaDeviceSynchronize();
            preProcess.endRecord();
            
            staticProcess.startRecord();
            
            cc_kernelStaticSwap<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                graph.staticNodePointerD, graph.degreeD,
                                                                                graph.staticEdgeListD, graph.valueD,
                                                                                graph.isActiveD,graph.isInStaticD);
            
            gpuErrorcheck(cudaPeekAtLastError());
            
            cout<<"overloadNodeNum is "<<overloadNodeNum<<endl;
            if(overloadNodeNum > 0){
                
                overloadProcess.startRecord();
                uint64_t numthreads = 1024;
                uint64_t numblocks = ((overloadNodeNum * WARP_SIZE + numthreads) / numthreads);
                dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                SIZE_TYPE numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                NEW_cc_kernelDynamicSwap_test<<<blockDim, numthreads , 0, graph.streamDynamic>>>(overloadNodeNum,  graph.overloadNodeListD,
                                                                                            graph.degreeD, graph.valueD, numwarps,
                                                                                            graph.isActiveD, graph.edgeArray,
                                                                                            graph.nodePointersD);
                gpuErrorcheck(cudaPeekAtLastError());
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(graph.streamDynamic);
                overloadProcess.endRecord(); 
                //cudaStreamSynchronize(graph.steamStatic);
                //staticProcess.endRecord();
            }
            //else{
            //cudaDeviceSynchronize();
            cudaStreamSynchronize(graph.steamStatic);
            staticProcess.endRecord();
            gpuErrorcheck(cudaPeekAtLastError());
            //}
            
            if(staticProcess._isStart()){
                staticProcess.endRecord();
            }
            preProcess.startRecord();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<SIZE_TYPE>());
            cout<<"next iter activenodesnum: "<<activeNodesNum<<endl;
            nodeSum += activeNodesNum;
            preProcess.endRecord();
            //overloadedges = 0; 
            // SIZE_TYPE *overloadnodes = new SIZE_TYPE[graph.vertexArrSize];
            // cudaMemcpy(overloadnodes,graph.overloadNodeListD,sizeof(SIZE_TYPE)*overloadNodeNum,cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            // for(SIZE_TYPE i=0;i<overloadNodeNum;i++){
            //     overloadedges += graph.degree[overloadnodes[i]];
            // }
            // double temp = (double)(overloadedges)*sizeof(SIZE_TYPE)/1024/1024;
            // cout<<temp<<endl;
        }
        totalProcess.endRecord();
        cout<<"total iter: "<<iter<<endl;
        totalProcess.print();
        preProcess.print();
        staticProcess.print();
        overloadProcess.print();

        cout << "nodeSum : " << nodeSum << endl;
        //cout << "move overload size : " << overloadEdges * sizeof(SIZE_TYPE) << endl;

        //src+=graph.vertexArrSize/testTimes;
        totalduration+=totalProcess.getDuration();
        staticduration+=staticProcess.getDuration();
        overloaduration+=overloadProcess.getDuration();
        totalProcess.clearRecord();
        preProcess.clearRecord();
        staticProcess.clearRecord();
        overloadProcess.clearRecord();
        // double temp = (double)(overloadedges)*sizeof(SIZE_TYPE)/1024/1024/1024;
        // overloadsize+=temp;
        // cout<<"transfer "<<temp<<" GB"<<endl;

    }
    gpuErrorcheck(cudaPeekAtLastError());
    cout<<"========TEST OVER========"<<endl;
    cout<<"pre move data time: "<<graph.preMoveDataTime<<"ms"<<endl;
    cout<<"Test over, average total process time (including pre move data): "<<totalduration/testTimes + graph.preMoveDataTime<<"ms"<<endl;
    cout<<"average static process time: "<<staticduration/testTimes<<"ms"<<endl;
    cout<<"average overload process time: "<<overloaduration/testTimes<<"ms"<<endl;
    //cout<<"average transfer data: "<<overloadsize/testTimes<<" GB"<<endl;
    if (verify) {
        cudaMemcpy(graph.value, graph.valueD, graph.vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cpu_verify_cc(graph.nodePointers, (SIZE_TYPE *)graph.edgeArray, graph.vertexArrSize, graph.edgeArrSize, graph.value);
    }
}

void newsssp_opt(string path, SIZE_TYPE sourceNode, double adviseRate,int model,int testTimes, double gpuMemoryLimit, bool verify){
    cout << "========NEW_sssp_opt==========" << endl;
    GraphMeta<EdgeWithWeight> graph;
    graph.setAlgType(SSSP);
    graph.setmodel(model);
    graph.setGpuMemoryLimit(gpuMemoryLimit);
    graph.setSourceNode(sourceNode);
    graph.readGraph(path, false);
    graph.setPrestoreRatio(adviseRate, 13);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();
    gpuErrorcheck(cudaPeekAtLastError());
    graph.checkNodeforSSSP(sourceNode);

    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    totalProcess.startRecord();
    SIZE_TYPE activeNodesNum;
    activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<SIZE_TYPE>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    SIZE_TYPE src = graph.sourceNode;
    long totalduration = 0;
    long staticduration = 0;
    long overloaduration = 0;
    uint64_t numthreads = 1024;
    uint64_t numblocks = ((graph.vertexArrSize * WARP_SIZE + numthreads) / numthreads);
    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    //double overloadGB =0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++){
        cout<<"================="<<"testIndex "<<testIndex<<"================="<<endl;
        cout<<"source: "<<src<<endl;
        //unsigned long long overloadedges=0;
        graph.setSourceNode(src);
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<SIZE_TYPE>());
        SIZE_TYPE nodeSum = activeNodesNum;
        int iter = 0;
        totalProcess.startRecord();

        while(activeNodesNum){
            iter++;
            //cout<<"iter "<<iter<<" activeNodeNum is "<<activeNodesNum<<" ";
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            SIZE_TYPE staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<SIZE_TYPE>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
            }
            cout << "iter " << iter << " staticNodeNum is " << staticNodeNum;
            SIZE_TYPE overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<SIZE_TYPE>());
            
            if(overloadNodeNum > 0){
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setActiveNodeList<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isOverloadActive,
                                                                graph.overloadNodeListD,
                                                                graph.prefixSumTemp);
                
            }
            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }
            staticProcess.startRecord();
            sssp_kernel<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                           graph.staticNodePointerD, graph.degreeD,
                                                                           graph.staticEdgeListD, graph.valueD,
                                                                           graph.isActiveD);
            printf(" overloadnum %ld\n", overloadNodeNum);
            if(overloadNodeNum>0){
                overloadProcess.startRecord();
                    SIZE_TYPE numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                    printf("iter %d numblocks %ld numwarps %ld overloadnum %ld\n",iter,numblocks, numwarps, overloadNodeNum);
                    NEW_sssp_kernelDynamic_test<<<blockDim,numthreads,0,graph.streamDynamic>>>(overloadNodeNum,graph.overloadNodeListD,graph.degreeD,
                                                                                                numwarps, graph.valueD, graph.isActiveD,
                                                                                                graph.edgeArray, graph.nodePointersD);
                cudaStreamSynchronize(graph.streamDynamic);
                cudaStreamSynchronize(graph.steamStatic);
                overloadProcess.endRecord();
                staticProcess.endRecord();
            }
            else{
                cudaDeviceSynchronize();
                staticProcess.endRecord();
            }
            if(staticProcess._isStart()){
                staticProcess.endRecord();
            }

            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<SIZE_TYPE>());
            nodeSum += activeNodesNum;
            // overloadedges = 0;
            // SIZE_TYPE *overloadnodes = new SIZE_TYPE[graph.vertexArrSize];
            // cudaMemcpy(overloadnodes,graph.overloadNodeListD,sizeof(SIZE_TYPE)*overloadNodeNum,cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            // for(SIZE_TYPE i=0;i<overloadNodeNum;i++){
            //     overloadedges += graph.degree[overloadnodes[i]];
            // }
            // double temp = (double)overloadedges*sizeof(SIZE_TYPE)/1024/1024;
            // cout<<temp<<endl;
        }
        totalProcess.endRecord();
        cout<<"total iter: "<<iter<<endl;
        totalProcess.print();
        staticProcess.print();
        overloadProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        //overloadGB += (double)overloadedges*sizeof(SIZE_TYPE)/1024/1024/1024;
        //cout<<" transfer data: "<<(double)overloadedges*sizeof(SIZE_TYPE)/1024/1024/1024<<" GB"<<endl;
        src+=graph.vertexArrSize/testTimes;
        totalduration+=totalProcess.getDuration();
        staticduration+=staticProcess.getDuration();
        overloaduration+=overloadProcess.getDuration();
        totalProcess.clearRecord();
        staticProcess.clearRecord();
        overloadProcess.clearRecord();
    }
    //cout<<"Total overloadsize: "<<overloadGB<<" GB"<<endl;
    cout<<"========TEST OVER========"<<endl;
    cout<<"pre move data time: "<<graph.preMoveDataTime<<"ms"<<endl;
    cout<<"Test over, average total process time (including pre move data): "<<totalduration/testTimes + graph.preMoveDataTime<<"ms"<<endl;
    cout<<"average static process time: "<<staticduration/testTimes<<"ms"<<endl;
    cout<<"average overload process time: "<<overloaduration/testTimes<<"ms"<<endl;
    gpuErrorcheck(cudaPeekAtLastError());
    //graph.writevalue("newsssp.txt");
    if (verify) {
        cudaMemcpy(graph.value, graph.valueD, graph.vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cpu_verify_sssp(graph.nodePointers, graph.edgeArray, graph.vertexArrSize, graph.edgeArrSize, sourceNode, graph.value);
    }
}

void newpr_opt(string path, double adviseRate,int model,int testTimes, double gpuMemoryLimit, bool verify){
    cout<<"========NEW_pr_opt==========="<<endl;
    GraphMeta<unsigned long long> graph;
    graph.setAlgType(PR);
    graph.setmodel(model);
    graph.setGpuMemoryLimit(gpuMemoryLimit);
    graph.readGraph(path, true);
    graph.setPrestoreRatio(adviseRate, 16);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();
    graph.checkNodeforPR(1024);

    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    long Total = 0;
    long Static = 0;
    long Overload = 0;
    totalProcess.startRecord();

    graph.refreshLabelAndValue();
    
    SIZE_TYPE activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<SIZE_TYPE>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    cout<<endl;
    cout<<"=================PR test start================="<<endl;
    double overloadsize=0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++){
        cout<<"====="<<testIndex<<" test====="<<endl;
        graph.refreshLabelAndValue();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                thrust::plus<SIZE_TYPE>());
        gpuErrorcheck(cudaPeekAtLastError());
        SIZE_TYPE nodeSum = activeNodesNum;
        int iter = 0;
        double totalSum = thrust::reduce(thrust::device, graph.valuePrD, graph.valuePrD + graph.vertexArrSize) / graph.vertexArrSize;
        cout << "totalSum " << totalSum << endl;
        totalProcess.startRecord();
        gpuErrorcheck(cudaPeekAtLastError());
        double totalDiff = 0.0;
        uint64_t numthreads = 1024;
        uint64_t numblocks = ((graph.vertexArrSize * WARP_SIZE + numthreads) / numthreads);
        dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
        //unsigned long long overloadedges=0;
        while(activeNodesNum > 0 ){
            //overloadedges = 0;
            iter++;
            //cout<<"iter "<<iter;
            preProcess.startRecord();
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            SIZE_TYPE staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<SIZE_TYPE>());
            //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum;
            if (staticNodeNum > 0) {
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
                
            }

            SIZE_TYPE overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<SIZE_TYPE>());
            //cout<<" overloadNodeNum: "<<overloadNodeNum<<endl;
            if(overloadNodeNum > 0){
                thrust::device_ptr<SIZE_TYPE> tempTestPrefixThrust = thrust::device_ptr<SIZE_TYPE>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<SIZE_TYPE>());
                setActiveNodeList<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isOverloadActive,
                                                                graph.overloadNodeListD,
                                                                graph.prefixSumTemp);
            }
            preProcess.endRecord();
            staticProcess.startRecord();
            prSumKernel_static<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                  graph.staticNodePointerD,
                                                                                  graph.staticEdgeListD, graph.degreeD,
                                                                                  graph.outDegreeD, graph.valuePrD,
                                                                                  graph.sumD);
            //cudaDeviceSynchronize();
            //staticProcess.endRecord();
            if(overloadNodeNum > 0){
                overloadProcess.startRecord();
                    SIZE_TYPE numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                    NEW_prSumKernel_dynamic_test<<<blockDim,numthreads,0,graph.streamDynamic>>>(
                        overloadNodeNum, graph.overloadNodeListD, graph.nodePointersD,
                        numwarps, graph.edgeArray, 
                        graph.degreeD, graph.outDegreeD,
                        graph.valuePrD, graph.sumD);
                cudaStreamSynchronize(graph.streamDynamic);
                overloadProcess.endRecord();
                cudaStreamSynchronize(graph.steamStatic);
                staticProcess.endRecord();
            }
            if(staticProcess._isStart()){
                staticProcess.endRecord();
            }
            preProcess.startRecord();
            //totalDiff = thrust::reduce(thrust::device, graph.DiffDThrust, graph.DiffDThrust+graph.vertexArrSize);
            prKernel_Opt<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.valuePrD, graph.sumD, graph.isActiveD);
            // double oldsum = thrust::reduce(graph.sumDThrust, graph.sumDThrust + graph.vertexArrSize,0, thrust::plus<double>());
            // double tempAdd = (1.0-oldsum)/graph.vertexArrSize;
            //prKernel_Opt<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.valuePrD, graph.sumD, graph.isActiveD);
            cudaDeviceSynchronize();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<SIZE_TYPE>());
            nodeSum += activeNodesNum;
            //totalDiff = thrust::reduce(thrust::device, graph.DiffDThrust, graph.DiffDThrust+graph.vertexArrSize);
            preProcess.endRecord();
            //cout << "iter " << iter+1 << " activeNodesNum " << activeNodesNum << endl;
            // SIZE_TYPE *overloadnodes = new SIZE_TYPE[graph.vertexArrSize];
            // cudaMemcpy(overloadnodes,graph.overloadNodeListD,sizeof(SIZE_TYPE)*overloadNodeNum,cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            // for(SIZE_TYPE i=0;i<overloadNodeNum;i++){
            //     overloadedges += graph.degree[overloadnodes[i]];
            // }
            // double temp = (double)overloadedges*sizeof(SIZE_TYPE)/1024/1024/1024;
            // cout<<"transfer "<<temp<<" GB"<<endl;
        }
        cout<<"total iter "<<iter<<endl;
        totalProcess.endRecord();
        totalProcess.print();
        Total+=totalProcess.getDuration();
        preProcess.print();
        staticProcess.print();
        Static+=staticProcess.getDuration();
        overloadProcess.print();
        Overload+=overloadProcess.getDuration();
        cout << "nodeSum : " << nodeSum << endl;
        
        //double temp = (double)overloadedges*sizeof(SIZE_TYPE)/1024/1024/1024;
        //cout<<"transfer "<<temp<<" GB"<<endl;
        //overloadsize+=temp;
    }
    cout<<"========TEST OVER========"<<endl;
    cout<<"pre move data time: "<<graph.preMoveDataTime<<"ms"<<endl;
    cout<<"Test over, average total process time (including pre move data): "<<Total/testTimes + graph.preMoveDataTime<<"ms"<<endl;
    cout<<"average static process time: "<<Static/testTimes<<"ms"<<endl;
    cout<<"average overload process time: "<<Overload/testTimes<<"ms"<<endl;
    //cout<<"Average overloadSize: "<<overloadsize/testTimes<<endl;
    gpuErrorcheck(cudaPeekAtLastError());
    //graph.writevalue("Liberator_PR_GSHll_res.txt");
    if (verify) {
        cudaMemcpy(graph.valuePr, graph.valuePrD, graph.vertexArrSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        // Convert unsigned long long edge list to SIZE_TYPE for CPU verification
        SIZE_TYPE *edgeListUint = new SIZE_TYPE[graph.edgeArrSize];
        #pragma omp parallel for
        for (EDGE_POINTER_TYPE i = 0; i < graph.edgeArrSize; i++)
            edgeListUint[i] = (SIZE_TYPE)graph.edgeArray[i];
        cpu_verify_pr(graph.nodePointers, edgeListUint, graph.outDegree, graph.vertexArrSize, graph.edgeArrSize, graph.valuePr);
        delete[] edgeListUint;
    }
}