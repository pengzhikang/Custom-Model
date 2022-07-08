#include "runtime/engine.hpp"
#include "runtime/op/allops.hpp"


namespace OCLEngine{
    /* 输入Tensor的对应cpu-mem */
    std::unordered_map<size_t, void*> cpu_mem;
    std::vector<struct TensorsS> input_tensors;
    std::vector<struct TensorsS> output_tensors;

    struct IoInfo{
        std::vector<uint32_t> input_id;
        std::vector<uint32_t> output_id;
    };

    /* 事件依赖的相关结构体 */
    struct DependOfEvent{
        /* 当事件依赖的id为-1时，表示没有依赖 */
        uint32_t depend_event_ids_num = 0;
        int32_t depend_event_ids_ptr = -1;
        int32_t this_event_id_ptr = -1;
    };

    /* 返回输入连接 */
    std::unordered_map<uint32_t, IoInfo> ReturnLayersIoId(PzkM& model){
        std::unordered_map<uint32_t, IoInfo> IoIds;
        std::unordered_map<uint32_t, bool> rTensorsDyn;
        /* 1.获取相应的实时Tensor的id */
        for (size_t i = 0; i < model.rTensors.size(); i++){
            if (model.rTensors[i].tensor_type == TensorType_DYNAMIC){
                rTensorsDyn[model.rTensors[i].id] = true;
            }else{
                rTensorsDyn[model.rTensors[i].id] = false;
            }
        }
        /* 2.获取全部的层的输入和输出 */
        for (size_t i = 0; i < model.rLayers.size(); i++){
            struct IoInfo one_layer_id;
            std::vector<uint32_t> input_id = model.rLayers[i].return_id(model.rLayers[i].input_id);
            std::vector<uint32_t> output_id = model.rLayers[i].return_id(model.rLayers[i].output_id);
            for(auto j:input_id){
                if (rTensorsDyn[j]){
                    one_layer_id.input_id.push_back(j);
                }
            }
            for(auto j:output_id){
                if (rTensorsDyn[j]){
                    one_layer_id.output_id.push_back(j);
                }
            }
            IoIds[model.rLayers[i].layer_id] = one_layer_id;
        }
        return IoIds;
    }

    void ShowDirectedGraph(std::vector<std::vector<bool>> DirectedGraph){
        printf("<-------------------------------------->\n");
        printf("DirectedGraph Mat:\n");
        for(auto i: DirectedGraph){
            for(auto j:i){
                printf("%s, ", j?"True ":"False");
            }
            printf("\n");
        }
        printf("<-------------------------------------->\n");
    }

    /* 获取模型有向图函数 */
    std::vector<std::vector<bool>> GetDirectedGraph(PzkM& model){
        /* 1.创造一个二维的bool矩阵 
            其标号顺序分别是输入节点，层节点，输出节点
        */
        size_t input_base_index = 0;
        size_t rLayers_base_index = model.model_runtime_input_id.size();
        size_t output_base_index = rLayers_base_index + model.rLayers.size();
        size_t NodeNum = model.model_runtime_input_id.size() + model.rLayers.size() + model.model_runtime_output_id.size();
        
        std::vector<std::vector<bool>> DirectedGraph;
        std::vector<bool> OneLine;
        for(size_t i = 0; i < NodeNum; i++){
            OneLine.push_back(false);
        }
        for(size_t i = 0; i < NodeNum; i++){
            DirectedGraph.push_back(OneLine);
        }
        /* 2.根据模型，得到所有连接信息 */
        std::unordered_map<uint32_t, IoInfo> IoIds = ReturnLayersIoId(model);
        for (size_t i = 0; i < NodeNum; i++){
            for (size_t j = 0; j < NodeNum; j++){
                if (i < j){
                    /* 只对右上角的连接关系进行处理操作，当遇到是层与层之间的关系时考虑到双向的连接
                        0-输入节点，1-层节点，2-输出节点
                        i:表示第i各个节点
                        j:表示第j个节点
                        DirectedGraph[i][j]=False表示j不依赖于i节点，否则就是依赖i节点
                    */
                   uint32_t i_type = (i < rLayers_base_index)?0:((i < output_base_index)?1:2);
                   uint32_t j_type = (j < rLayers_base_index)?0:((j < output_base_index)?1:2);
                   /*
                   整体存在的组合情况如下所示
                   <i,j>:
                   <入，层>=<0,1>
                   <入，出>=<0,2>
                   <层，层>=<1,1>
                   <层，出>=<1,2>
                   */
                  if (i_type == 0 && j_type == 1){
                    std::vector<uint32_t> j_input_index = IoIds[model.rLayers[j-rLayers_base_index].layer_id].input_id;
                    if (std::find(j_input_index.begin(), j_input_index.end(), model.model_runtime_input_id[i]) != j_input_index.end()){
                        /* 输入连接到了该层 */
                        DirectedGraph[i][j] = true;
                    }
                  }else if (i_type == 0 && j_type == 2){
                    if (model.model_runtime_input_id[i] == model.model_runtime_output_id[j-output_base_index]){
                        /* 输出即输入 */
                        DirectedGraph[i][j] = true;
                    }
                  }else if (i_type == 1 && j_type == 1){
                    bool link_flag = false;
                    /* 首先检查i-->j的情况，如果连接，则设link_flag=true */
                    std::vector<uint32_t> i_ouput_index = IoIds[model.rLayers[i-rLayers_base_index].layer_id].output_id;
                    std::vector<uint32_t> j_input_index = IoIds[model.rLayers[j-rLayers_base_index].layer_id].input_id;
                    for(auto i_id:i_ouput_index){
                        if(std::find(j_input_index.begin(), j_input_index.end(), i_id) != j_input_index.end()){
                            /* 此时i-->j */
                            DirectedGraph[i][j] = true;
                            link_flag = true;
                            break;
                        }
                    }
                    /* 再检查j--->i的情况，需要保证link_flag == false，否则表示该模型不是一个有向无环图 */

                    std::vector<uint32_t> i_input_index = IoIds[model.rLayers[i-rLayers_base_index].layer_id].input_id;
                    std::vector<uint32_t> j_output_index = IoIds[model.rLayers[j-rLayers_base_index].layer_id].output_id;
                    for(auto j_id:j_output_index){
                        if(std::find(i_input_index.begin(), i_input_index.end(), j_id) != i_input_index.end()){
                            /* 此时j-->i */
                            if (!link_flag){
                                DirectedGraph[j][i] = true;
                                break;
                            }else{
                                printf("This model is not a Directed Acyclic Graph\n");
                                return DirectedGraph;
                            }
                        }
                    } 
                  }else if (i_type == 1 & j_type == 2){
                    std::vector<uint32_t> i_output_index = IoIds[model.rLayers[i-rLayers_base_index].layer_id].output_id;
                    if (std::find(i_output_index.begin(), i_output_index.end(), model.model_runtime_output_id[j-output_base_index]) != i_output_index.end()){
                        /* 输入连接到了该层 */
                        DirectedGraph[i][j] = true;
                    }
                  }
                   
                }
            }
        }
        return DirectedGraph;
    }

    /* 剥离出层的连接关系 */
    std::vector<std::vector<bool>> MaskDirectedGraph(std::vector<std::vector<bool>> DirectedGraph, size_t begin, size_t end){
        std::vector<std::vector<bool>> MDirectedGraph;
        for(size_t i = begin; i < end; i++){
            std::vector<bool> OneLine;
            for(size_t j = begin; j < end; j++){
                OneLine.push_back(DirectedGraph[i][j]);
            }
            MDirectedGraph.push_back(OneLine);
        }
        return MDirectedGraph;
    }

    /* 去除掉某个节点 */
    std::vector<std::vector<bool>> RemoveDirectedGraph(std::vector<std::vector<bool>> DirectedGraph, std::vector<size_t> points){
        std::vector<std::vector<bool>> RDirectedGraph;
        for(size_t i = 0; i < DirectedGraph.size(); i++){
            std::vector<bool> OneLine;
            if (std::find(points.begin(), points.end(), i) != points.end()){
                continue;
            }else{
                for(size_t j = 0; j < DirectedGraph[i].size(); j++){
                    if (std::find(points.begin(), points.end(), j) == points.end()){
                        OneLine.push_back(DirectedGraph[i][j]);
                    }
                }
                RDirectedGraph.push_back(OneLine);
            }
        }
        return RDirectedGraph;
        
    }

    /* 判断哪些是根节点 */
    std::vector<size_t> JudgeRootNode(std::vector<std::vector<bool>> DirectedGraph){
        std::vector<size_t> RootNode;
        size_t MatLen = DirectedGraph.size();
        for(size_t i = 0; i < MatLen; i++){
            bool HasInput = false;
            for(size_t j = 0; j < MatLen; j++){
                if (DirectedGraph[j][i]){
                    HasInput = true;
                    break;
                }
            }
            if (!HasInput){
                RootNode.push_back(i);
            }
        }
        return RootNode;
    }

    /* 判断哪些是孤立节点,也就是没有输入和输出的节点 */
    std::vector<size_t> JudgeIsolatedNode(std::vector<std::vector<bool>> DirectedGraph){
        std::vector<size_t> IsolatedNode;
        size_t MatLen = DirectedGraph.size();
        for(size_t i = 0; i < MatLen; i++){
            bool HasLink = false;
            for(size_t j = 0; j < MatLen; j++){
                if (DirectedGraph[j][i] || DirectedGraph[i][j]){
                    HasLink = true;
                    break;
                }
            }
            if (!HasLink){
                IsolatedNode.push_back(i);
            }
        }
        return IsolatedNode;
    } 

    /* 获取某个节点的输入 */
    std::vector<size_t> GetInputNumOfNode(std::vector<std::vector<bool>> DirectedGraph, size_t index){
        std::vector<size_t> input_num;
        for(size_t i = 0; i < DirectedGraph.size(); i++){
            if (DirectedGraph[i][index]){
                input_num.push_back(i);
            }
        }
        return input_num;
    }

    /* 获取某个节点的输出 */
    std::vector<size_t> GetOutputNumOfNode(std::vector<std::vector<bool>> DirectedGraph, size_t index){
        std::vector<size_t> output_num;
        for(size_t i = 0; i < DirectedGraph.size(); i++){
            if (DirectedGraph[index][i]){
                output_num.push_back(i);
            }
        }
        return output_num;
    }

    /* 集合减法 C = A - B */
    std::vector<size_t> MinusSet(std::vector<size_t> A, std::vector<size_t> B){
        std::vector<size_t> C;
        for(auto i: A){
            if (std::find(B.begin(), B.end(), i) == B.end()){
                /* i属于A,但是不属于B,则需要被推入C中*/
                C.push_back(i);
            }
        }
        return C;
    }

    /* 返回重排结果的标号信息 */
    std::vector<size_t> ReSortByDirectedGraph(std::vector<std::vector<bool>> DirectedGraph){
        /* 运用的主要原理是根节点只有输出没有输入的特性;
            通过不断去除掉根节点，更新有向图，然后进行操作的时候
        */
        std::vector<size_t> ReSortIndex;
        std::vector<size_t> RemainIndex;
        std::vector<size_t> RegIndex;
        std::vector<size_t> Reg2Index;
        std::vector<std::vector<bool>> BakDirectedGraph = DirectedGraph;
        for(size_t i = 0; i < DirectedGraph.size(); i++){
            RemainIndex.push_back(i);
        }
        /* 1.开始进行根节点获取操作 */
        ReSortIndex = JudgeRootNode(DirectedGraph);
        /* 2. 移除RemainIndex中的重复点 */
        RemainIndex = MinusSet(RemainIndex, ReSortIndex);
        BakDirectedGraph = RemoveDirectedGraph(BakDirectedGraph, ReSortIndex);
        /* 3. 重复上述两个步骤,直到BakDirectedGraph中不存在节点或者是RemainIndex中没有值 */
        while(RemainIndex.size() > 0 && BakDirectedGraph.size() > 0 && ReSortIndex.size() < DirectedGraph.size()){
            Reg2Index.clear();
            RegIndex = JudgeRootNode(BakDirectedGraph);
            /* 加入到ReSortIndex中 */
            for(auto i:RegIndex){
                ReSortIndex.push_back(RemainIndex[i]);
                Reg2Index.push_back(RemainIndex[i]);
            }
            RemainIndex = MinusSet(RemainIndex, Reg2Index);
            BakDirectedGraph = RemoveDirectedGraph(BakDirectedGraph, RegIndex);
        }
        return ReSortIndex;
    }


    void PrintDependOfEvent(std::vector<DependOfEvent> d){
        printf("depend of event is \n");
        for(size_t i = 0; i < d.size(); i++){
            printf("node=%d--->[dpnum=%d,dphead=%d,thisid=%d]\n",
                    i, d[i].depend_event_ids_num,
                    d[i].depend_event_ids_ptr,
                    d[i].this_event_id_ptr);
        }
        return;
    }

    /* 对重排后的model得到其事件依赖标号图 */
    std::vector<DependOfEvent> DependIndex(std::vector<std::vector<bool>> AllDirectedGraph, size_t input_num, size_t layer_num){
        size_t output_num = AllDirectedGraph.size() - input_num - layer_num;
        size_t input_base_index = 0;
        size_t rLayers_base_index = input_num;
        size_t output_base_index = rLayers_base_index + layer_num;
        /* 运用的主要原理是根节点只有输出没有输入的特性;
            通过不断去除掉根节点，更新有向图，然后进行操作的时候
        */
        std::vector<std::vector<bool>> DirectedGraph = MaskDirectedGraph(AllDirectedGraph, 
                                                                        rLayers_base_index, 
                                                                        output_base_index);
        std::vector<size_t> ReSortIndex;
        std::vector<size_t> RemainIndex;
        std::vector<size_t> RegIndex;
        std::vector<size_t> Reg2Index;
        std::vector<std::vector<bool>> BakDirectedGraph = DirectedGraph;
        
        std::vector<std::vector<size_t>> AllRootIndex;
        for(size_t i = 0; i < DirectedGraph.size(); i++){
            RemainIndex.push_back(i);
        }
        /* 1. 获取成组的层集合 */
        ReSortIndex = JudgeRootNode(DirectedGraph);
        AllRootIndex.push_back(ReSortIndex);
        RemainIndex = MinusSet(RemainIndex, ReSortIndex);
        BakDirectedGraph = RemoveDirectedGraph(BakDirectedGraph, ReSortIndex);
        while(RemainIndex.size() > 0 && BakDirectedGraph.size() > 0 && ReSortIndex.size() < DirectedGraph.size()){
            Reg2Index.clear();
            RegIndex = JudgeRootNode(BakDirectedGraph);
            /* 加入到ReSortIndex中 */
            for(auto i:RegIndex){
                ReSortIndex.push_back(RemainIndex[i]);
                Reg2Index.push_back(RemainIndex[i]);
            }
            AllRootIndex.push_back(Reg2Index);
            RemainIndex = MinusSet(RemainIndex, Reg2Index);
            BakDirectedGraph = RemoveDirectedGraph(BakDirectedGraph, RegIndex);
        }
        /* AllRootIndex内的标号需要集体反推回AllDirectedGraph中的标号 */
        for(size_t i = 0; i < AllRootIndex.size(); i++)
            for(size_t j = 0; j < AllRootIndex[i].size(); j++)
                AllRootIndex[i][j] += input_num;
        /* 
            获取的AllRootIndex从前往后走是表示一系列的根节点
            把一组根节点当作一个新的节点，那么就序列化了整网
            不同层。组合成了顺序执行的命令队列。
            可能存在的AllRootIndex.size()情况：
            《1》2:只有输入和输出，也就是输入和输出一块
         */
        /* 2. 根据层集合，进行后续的事件依赖操作 */
        std::vector<DependOfEvent> AllDependOfEvent;
        /* 2.1. 首先组织起输入的依赖 */
        for(size_t i = 0; i < input_num; i++){
            DependOfEvent OneDepend;
            OneDepend.depend_event_ids_num = 0;
            OneDepend.depend_event_ids_ptr = -1;
            OneDepend.this_event_id_ptr = i;
            AllDependOfEvent.push_back(OneDepend);
        }
        /* 2.2. 然后进行每个层的依赖 */
        int32_t depend_event_id_head = 0;
        size_t depend_event_num = input_num;
        for (size_t i = 0; i < AllRootIndex.size(); i++){
            std::vector<size_t> OneRootIndexSet = AllRootIndex[i];
            for (size_t j = 0; j < OneRootIndexSet.size(); j++){
                DependOfEvent OneDepend;
                OneDepend.depend_event_ids_num = depend_event_num;
                OneDepend.depend_event_ids_ptr = depend_event_id_head;
                OneDepend.this_event_id_ptr = depend_event_id_head + depend_event_num + j;
                AllDependOfEvent.push_back(OneDepend);
            }
            /* 更新头 */
            depend_event_num = OneRootIndexSet.size();
            depend_event_id_head +=OneRootIndexSet.size();
        }
        /* 2.3. 组织输出的依赖 */
        for(size_t i = 0; i < output_num; i++){
            DependOfEvent OneDepend;
            OneDepend.depend_event_ids_num = depend_event_num;
            OneDepend.depend_event_ids_ptr = depend_event_id_head;
            OneDepend.this_event_id_ptr = -1;
            AllDependOfEvent.push_back(OneDepend);
        }
        return AllDependOfEvent;

    }



    /* 根据model构建有向图，重排所有的层 */
    bool SortLayers(PzkM& model){
        /* 1.获取有向图 */
        std::vector<std::vector<bool>> DirectedGraph = GetDirectedGraph(model);
        /* 2.打印有向图结果 */
        ShowDirectedGraph(DirectedGraph);
        /* 3.剥离出只关乎层的有向图 */
        std::vector<std::vector<bool>> LayersDirectedGraph = MaskDirectedGraph(DirectedGraph, 
                                                                                model.model_runtime_input_id.size(), 
                                                                                model.model_runtime_input_id.size() + model.rLayers.size());
        ShowDirectedGraph(LayersDirectedGraph);
        /* 4.进行层顺序重排操作 */
        std::vector<size_t> ReSortIndex = ReSortByDirectedGraph(LayersDirectedGraph);
        std::vector<layer_maker> NewLayer;
        for(auto i : ReSortIndex){
            NewLayer.push_back(model.rLayers[i]);
        }
        model.rLayers  = NewLayer;
        /*测试依赖标号图是否正常*/
        std::vector<std::vector<bool>> DG = GetDirectedGraph(model);
        std::vector<DependOfEvent> DI = DependIndex(DG, model.model_runtime_input_id.size(), model.rLayers.size());
        PrintDependOfEvent(DI);
        return true;
    }


    /* 通过PzkM来创建一个OCL后端运行时 */
    bool CreateNetWork(PzkM model){
        /* 0. 重排model的层顺序 */
        if (!SortLayers(model)){
            printf("ReSort Layers Failed");
            return false;
        }

        /* 构建相应的节点依赖对象 */
        std::vector<std::vector<bool>> DG = GetDirectedGraph(model);
        std::vector<DependOfEvent> DI = DependIndex(DG, model.model_runtime_input_id.size(), model.rLayers.size());
        CreateOrgEvents(model.model_runtime_input_id.size() + model.rLayers.size());
        for(auto i : DI){
            CreateNodeEventByOut(i.depend_event_ids_num, i.depend_event_ids_ptr, i.this_event_id_ptr);
        }

        /* 1.首先构建所有的Tensor */
        for(size_t i = 0; i < model.rTensors.size(); i++){
            if(CreateClMem(&model.rTensors[i]) == false){
                printf("create clmem faided in id = %d\n", model.rTensors[i].id);
                return false;
            }
        }
        /* 2.设置Tensor作为输入 */
        std::vector<struct TensorsS*> inputs;
        std::vector<size_t> input_indexs;
        for(size_t i = 0; i < model.model_runtime_input_id.size(); i++){
            for(size_t j = 0; j < model.rTensors.size(); j++){
                if (model.rTensors[j].id == model.model_runtime_input_id[i]){
                    inputs.push_back(&(model.rTensors[j]));
                }
            }
            input_indexs.push_back(i);
        }
        SetAsInputs(inputs, input_indexs);
        for(size_t i = 0; i < inputs.size(); i++){
            input_tensors.push_back(*inputs[i]);
        }
        /* 3.进行网络层的运行时构建 */
        if(BuildLayers(model) == false){
            printf("failed build runtime layers\n");
            return false;
        }
        /* 4.设置Tensor作为输出 */
        std::vector<struct TensorsS*> outputs;
        std::vector<size_t> output_indexs;
        size_t output_indexs_base = model.model_runtime_input_id.size() + model.rLayers.size();
        for(size_t i = 0; i < model.model_runtime_output_id.size(); i++){
            for(size_t j = 0; j < model.rTensors.size(); j++){
                if (model.rTensors[j].id == model.model_runtime_output_id[i]){
                    outputs.push_back(&(model.rTensors[j]));
                }
            }
            output_indexs.push_back(i + output_indexs_base);
        }
        SetAsOutputs(outputs, output_indexs);
        for(size_t i = 0; i < outputs.size(); i++){
            output_tensors.push_back(*outputs[i]);
        }
        return true;
    }
    /* 绑定cpu内存到此处 */
    void AttachMem(size_t id, void* cpu_mem1){
        if(cpu_mem1 != NULL){
            cpu_mem[id] = cpu_mem1;
        }
    }
    /* 进行推理的接口 */
    bool Inference(){
        /* 首先input:cpu->device */
        /* 错误修正：因为这里的input_tensors是一个unordered_map，不能之前的方式遍历 
                    但是用unorder_map无法正常取地址，所以改变了对应类型*/
        for(size_t i = 0; i < input_tensors.size(); i++){
            if(WriteCLMem(&input_tensors[i], cpu_mem[input_tensors[i].id]) == false){
                printf("write CLmem in inference, which id = %d\n", input_tensors[i].id);
                return false;
            }
        }
        // for(auto it=input_tensors.begin(); it != input_tensors.end(); it++){
        //     if(WriteCLMem(&it->second, cpu_mem[it->second.id]) == false){
        //         printf("write CLmem in inference, which id = %d\n", it->second.id);
        //         return false;
        //     }
        // }
        /* 然后进行推理 */
        for(size_t i = 0; i < AllLayers.size(); i++){
            AllLayers[i]->run();
        }
        /* 最后ouput:device->cpu */
        for(size_t i = 0; i < output_tensors.size(); i++){
            if(ReadCLMem(&output_tensors[i], cpu_mem[output_tensors[i].id]) == false){
                printf("read CLmem in inference, which id = %d\n", output_tensors[i].id);
                return false;
            }
        }
        // for(auto it1=output_tensors.begin(); it1 != output_tensors.end(); it1++){
        //     if(ReadCLMem(&output_tensors[it1->first], cpu_mem[it1->second.id]) == false){
        //         printf("read CLmem in inference, which id = %d\n", it1->second.id);
        //         return false;
        //     }
        // }
        return true;
    }
}