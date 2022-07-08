#ifndef ENGINE_HPP_
#define ENGINE_HPP_
#include "model/pzk.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#ifdef _APPLE_
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
/* opencl推理引擎的命令空间
    想要通过效仿acl-opencl的推理流程来构建自己的推理引擎
 */
/*
1、希望opencl平台的设备管理等方面由命令空间管理
2、提供了对CLmem的操作手段，包括创建、读写等
3、提供了CLKernel排对进入命令队列的操作、以及对CLKernel的管理
4、命令队列查询、以及等待操作等
*/
namespace OCLEngine {
    /*---------------------------------------------opencl基本变量------------------------------------*/
    /* opencl平台持久化变量 */
    cl_bool CLSynchronize = CL_TRUE;
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_device_id device = 0;
    std::unordered_map<uint32_t, cl_mem> clmem;
    cl_int errNum;
    /*---------------------------------为了能够进行事件同步而设置前向链表结构--------------------------*/
    struct EventWithNum{
        cl_uint num = 0;
        cl_event* event = NULL;
        bool NewMalloc = false;/* 表示新malloc出来的空间，需要进行free */
    };
    /* 重构这些节点内容 */
    struct NodeEvent{
        EventWithNum wait_event;
        EventWithNum this_event;
    };

    /* 生成EventWithNum结构体的函数 */
    struct EventWithNum CreateEventWithNum(cl_uint num){
        struct EventWithNum event;
        event.num = num;
        event.event = (cl_event*)malloc(sizeof(cl_event) * num);
        event.NewMalloc = true;
        return event;
    }

    struct EventWithNum CreateEventWithNum(cl_uint num, cl_event* this_event){
        struct EventWithNum event;
        if (num >= 0 && this_event != NULL){
            event.num = num;
            event.event = this_event;
        }else{
            event.num = 0;
            event.event = NULL;
        }
        return event;
    }

    /* 销毁EventWithNum */
    void DestoryEventWithNum(struct EventWithNum event){
        if (event.num > 0 && event.event != NULL && event.NewMalloc == false){
            free(event.event);
        }
    }

    /* 生成NodeEvent函数 */
    struct NodeEvent CreateNodeEvent(cl_uint num){
        struct NodeEvent Node;
        Node.wait_event = CreateEventWithNum(num);
        Node.this_event = CreateEventWithNum((cl_uint)1);
        return Node;
    }
    struct NodeEvent CreateNodeEvent(cl_uint num, cl_event* wait_event, cl_event* this_event){
        struct NodeEvent Node;
        Node.wait_event = CreateEventWithNum(num, wait_event);
        Node.this_event = CreateEventWithNum(1, this_event);
        return Node;
    }

    /* 销毁 */
    void DestoryEventWithNum(struct NodeEvent Node){
        DestoryEventWithNum(Node.wait_event);
        DestoryEventWithNum(Node.this_event);
    }

    /*
     class wait_event{
     public:
         cl_uint num = 0;
         cl_event* event = NULL;
         wait_event* next = NULL;
         wait_event(){};
         ~wait_event(){

         };
     }
     */
    /* 事件同步产生的变量 */
    cl_event* all_events = NULL;
    uint32_t all_events_num = 0;
    std::vector<NodeEvent> all_node_events;
    std::unordered_map<uint32_t, NodeEvent> event_of_tensor;
    /* 创建事件 */
    void CreateOrgEvents(uint32_t num){
        if (all_events != NULL){
            free(all_events);
        }
        all_events = (cl_event*)malloc(sizeof(cl_event) * num);
        all_events_num = num;
    }
    /* 创建NodeEvent */
    bool CreateNodeEventByOut(uint32_t wait_events_num, int32_t wait_events_offset, int32_t this_events_offset){
        /* 记得这里需要把uint32_t的all_events_num转换成int32_t类型的数值，否则会比较出错 */
        int32_t int32_all_events_num = (int32_t)all_events_num;
        uint32_t real_wait_events_num = wait_events_num;
        cl_event* wait_events_offset_ptr = NULL;
        cl_event* this_events_ptr = NULL;
        if (wait_events_num <= 0 || wait_events_offset < 0){
            real_wait_events_num = 0;
        }else if (wait_events_offset < int32_all_events_num && (wait_events_offset + (int32_t)wait_events_num) <= int32_all_events_num){
            wait_events_offset_ptr = &(all_events[wait_events_offset]);
        }else{
            return false;
        }
        if (this_events_offset >= 0 && this_events_offset < int32_all_events_num){
            this_events_ptr = &(all_events[this_events_offset]);
        }else if(this_events_offset >= int32_all_events_num){
            return false;
        }
        NodeEvent ThisNodeEvent = CreateNodeEvent((cl_uint)real_wait_events_num, wait_events_offset_ptr, this_events_ptr);
        all_node_events.push_back(ThisNodeEvent); 
        return true;
    }

    /*
    1.创建平台
    2.创建设备
    3.根据设备创建上下文
    */
    cl_context CreateContext(cl_device_id *device){
        cl_int errNum;
        cl_uint numPlatforms;
        cl_platform_id firstPlatformId;
        cl_context context = NULL;
        /* 默认初始化一个opencl平台，其平台id放到firstPlatformId中，同时得到了支持opencl的平台数目 */
        errNum = clGetPlatformIDs(1, &firstPlatformId,&numPlatforms);
        if (errNum!= CL_SUCCESS || numPlatforms <=  0)
        {
            printf( "Failed to find any OpenCL  platforms.\n" );
            return NULL;
        }
        errNum = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU,1,
                                device,NULL);
        if (errNum!= CL_SUCCESS)
        {
            printf( "There is no GPU,trying CPU……\n" );
            errNum = clGetDeviceIDs(firstPlatformId,
                                    CL_DEVICE_TYPE_CPU, 1,device,NULL);
        }
        if (errNum!= CL_SUCCESS)
        {
            printf( "There is  NO GPU or CPU\n" );
            return NULL;
        }
        context = clCreateContext(NULL,1, device, NULL,NULL,& errNum);
        if (errNum!= CL_SUCCESS)
        {
            printf( " create context error\n" );
            return NULL;
        }
        return context;
    }
    /*
    @在上下文可用的第一个设备中创建命令队列
    */
    cl_command_queue CreateCommandQueue(){
        cl_int errNum;
        cl_command_queue commandQueue = NULL;
        commandQueue = clCreateCommandQueue(context, device,0,NULL);
        if (commandQueue == NULL)
        {
            printf("Failed to create commandQueue  for device 0\n");
            return NULL;
        }
        return commandQueue;
    }

    /* 添加节点 */
    // void add_wait_event_node(cl_uint event_num){
    //     if(wehead == NULL){
    //         wehead = (wait_event*)malloc(sizeof(wait_event));
    //         wehead->num = event_num;
    //         wehead->event = (cl_event*)malloc(sizeof(cl_event) * wehead->num);
    //         weend = wehead;
    //     }
    //     else{
    //         weend->next = (wait_event*)malloc(sizeof(wait_event));
    //         weend = weend->next;
    //         weend->num = event_num;
    //         weend->event = (cl_event*)malloc(sizeof(cl_event) * wehead->num);
    //     }
    // }
    /* 清除事件同步链表 */
    // void CleanEvent(){
    //     wait_event* ptr = wehead;
    //     while(ptr != NULL){
    //         struct wait_event* ptr1 = ptr->next;
    //         if(ptr->event != NULL && ptr->num > 0){
    //             for(size_t i = 0; i < ptr->num; i++){
    //                 clReleaseEvent(*(ptr->event + i));
    //             }
    //             free(ptr->event);
    //         }
    //         free(ptr);
    //         ptr = ptr1;
    //     }
    //     return;
    // }
    /* 只是单纯释放 */
    // void ReleaseEvent(){
    //     struct wait_event* ptr = wehead;
    //     while(ptr != NULL){
    //         struct wait_event* ptr1 = ptr->next;
    //         if(ptr->event != NULL && ptr->num > 0){
    //             for(size_t i = 0; i < ptr->num; i++){
    //                 clReleaseEvent(*(ptr->event + i));
    //                 *(ptr->event + i) = NULL;
    //             }
    //         }
    //         ptr = ptr1;
    //     }
    //     return;
    // }

    /* 用于管理相关的程序对象，防止多余的程序对象的生成 */
    struct ProgramBuildData{
        cl_program program;
        std::string fileName;
        std::vector<std::string> buildOption;
    };
    class ClProgramManager{
    private:
        std::string root_path;
        std::vector<ProgramBuildData> all_program;
    public:
        ClProgramManager() = default;
        ClProgramManager(std::string cl_root_path){
            if (cl_root_path.c_str()[cl_root_path.size() - 1] == '/' || cl_root_path.c_str()[cl_root_path.size() - 1] == '\\'){
                root_path = cl_root_path;
            }else{
                root_path = cl_root_path + "/";
            }
        };
        ~ClProgramManager(){
            /* 销毁函数 */

        };
        /*
            用于从文本中读取出文本文件
        */
        char* ReadKernelSourceFile(const char *filename, size_t *length)
        {
            FILE *file = NULL;
            size_t sourceLength;
            char *sourceString;
            int ret;
            file = fopen(filename,"rb");
            if(file == NULL)
            {
                printf("Can't open %s\n",filename);
                return NULL;
            }
            fseek(file,0,SEEK_END);
            sourceLength = ftell(file);
            fseek(file,0,SEEK_SET);
            sourceString = (char *)malloc(sourceLength + 1);
            sourceString[0] = '\0';
            ret = fread(sourceString,sourceLength,1, file);
            if(ret == 0)
            {
                printf("Can't read source %s\n", filename);
                return NULL;
            }
            fclose(file);
            if(length!= 0)
            {
                *length = sourceLength;
            }
            sourceString[sourceLength] = '\0';
            return sourceString;
        }
        /*
        @读取内核源码创建OpenCL程序
        第一个表示文件名字，第二个表示编译优化指令
        */
        cl_program CreateProgram(const char *fileName, const char* buildOption = NULL){
            cl_int errNum;
            cl_program program;
            size_t program_length;
            //从.cl文件中获取cl代码
            char *const source = ReadKernelSourceFile(fileName, &program_length);
            // 创建程序对象
            program = clCreateProgramWithSource(context, 1,
                                                (const  char **)&source,
                                                NULL, NULL);
            if (program == NULL)
            {
                printf("Failed to create CL program from  source.\n" );
                free(source);
                return NULL;
            }
            // 编译程序对象
            errNum = clBuildProgram(program,0,NULL, buildOption,NULL,NULL);
            if (errNum!= CL_SUCCESS)
            {
                char buildLog[16384];
                clGetProgramBuildInfo(program,device,
                                        CL_PROGRAM_BUILD_LOG,
                                        sizeof(buildLog),
                                        buildLog,NULL);
                printf("Error in kernel：%s \n",buildLog);
                clReleaseProgram(program);
                return NULL;
            }
            free(source);
            return program;
        }
        /* 用来是否满足每一列或者是每一个行有且只有一个true */
        static bool RightMap(std::vector<std::vector<bool>> Map){
            bool A = true;
            size_t MapSize = Map.size();
            for (size_t i = 0; i < MapSize; i++){
                /*行*/
                int htrueNum = 0;
                for (size_t k = 0; k < MapSize; k++){
                    if (Map[i][k]){
                        htrueNum++;
                    }
                }
                /*列*/
                int ltrueNum = 0;
                for (size_t k = 0; k < MapSize; k++){
                    if (Map[k][i]){
                        ltrueNum++;
                    }
                }
                if (htrueNum == 1 && ltrueNum == 1){
                    continue;
                }else{
                    A = false;
                    break;
                }
            }
            return A;
        };

        cl_program GetProgram(std::string fileName, std::vector<std::string> buildOption){
            /* 首先查看目前是否存在相应的已经编译好的程序对象 */
            for(size_t i = 0; i < all_program.size(); i++){
                if (all_program[i].fileName == fileName){
                    std::vector<std::vector<bool>> Map;
                    for(size_t j = 0; j < buildOption.size(); j++){
                        std::vector<bool> OneMap;
                        for (size_t k = 0; k < all_program[i].buildOption.size(); k++){
                            OneMap.push_back(all_program[i].buildOption[k] == buildOption[j]);
                        }
                        Map.push_back(OneMap);
                    }
                    if (RightMap(Map)){
                        /* 此时找到了对应的相同的程序对象 */
                        return all_program[i].program;
                    }

                }
            }
            std::string AllBuildOptions;
            for(auto i: buildOption){
                AllBuildOptions += " " + i;
            }
            cl_program clp = CreateProgram((root_path + fileName).c_str(), AllBuildOptions.c_str());
            if (clp != NULL){
                ProgramBuildData One;
                One.program = clp;
                One.fileName = fileName;
                One.buildOption = buildOption;
                all_program.push_back(One);
            }
            return clp;
        }


        cl_kernel GetKernel(std::string fileName, std::vector<std::string> buildOption, std::string kernelName){
            /* 从这里进行操作 */
            cl_program clp = GetProgram(fileName, buildOption);

            if (clp == NULL){
                return NULL;
            }
            /* 创建OpenCL内核 */
            cl_kernel kernel = clCreateKernel(clp, kernelName.c_str(),NULL);
            if (kernel == NULL)
            {
                printf( "Failed to create kernel\n");
            }
            return kernel;

        };
    };


    ClProgramManager ProgramManager(std::string("/home/opencl/ocl-engine/clkernel/"));
    /*---------------------------------------------基类ClFunction------------------------------------*/
    class CLFunction{
    public:
        CLFunction(){};
        virtual ~CLFunction(){

        };
        virtual void run(){
            printf("Run Base CLFunction run()-----Nothing Happen\n");
        };
        virtual void cpu_run(){
            printf("Run Base CPU        run()-----Nothing Happen\n");
        };
    };
    /*---------------------------------------------推理函数------------------------------------*/

    // /*
    // @创建内存对象
    // */
    // bool CreateMemObjects(cl_mem  memObjects[3], float * a){
    //     // clCreateBuffer会在上下文中创建存储器对象，也就是内存对象，而是否设置主机内存、全局、全局常量、局部、私有内存等，通过设置cl_mem_flags(eg:CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR)
    //     memObjects[0] = clCreateBuffer(context,
    //                         CL_MEM_READ_WRITE |  CL_MEM_COPY_HOST_PTR,
    //                         sizeof(float) *  ARRAY_SIZE,a,NULL);
    //     memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
    //                                     sizeof(float)  * col_chw,
    //                                     NULL,NULL);
    //     if (memObjects[0] == NULL || memObjects[1]  == NULL)
    //     {
    //         printf("Error creating memory objects.\n");
    //         return false;
    //     }
    //     return true;
    // }
    /*
    @根据传入的TensorsS对象t创建clmem对象
    */
    bool CreateClMem(struct TensorsS* one){
        if(clmem.find(one->id) != clmem.end()){
            // 表示已经存在id对应的clmem
            return true;
        }
        else{
            cl_mem_flags memf;
            void* databuffer = NULL;
            if (one->tensor_type == TensorType_CONST){
                memf = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
                databuffer = one->weights.buffer.data();
            }
            else{
                memf = CL_MEM_READ_WRITE;
            }
            cl_int errNum;
            clmem[one->id] = clCreateBuffer(context, memf, 
                                            PzkM::shape2size(one->shape.dims)*datalen(one->data_type), 
                                            databuffer, &errNum);
            if (errNum != CL_SUCCESS || clmem[one->id] == NULL){
                printf("Error create memory when id = %d\n", one->id);
                return false;
            }
            return true;
        }
    }
    // 以后的TensorsS读取函数增加了事件同步的功能
    /*
    @根据传入的TensorsS对象读取clmem对象到特定的位置
    */
    bool ReadCLMem(TensorsS* one, void* dest){
        if (clmem.find(one->id) == clmem.end()){
           // 未创建过对应的cl_mem对象
           return false; 
        }
        else{
            cl_event* clevents = NULL;
            cl_uint events_num = 0;
            cl_event* this_events = NULL;
            if (event_of_tensor.find(one->id) != event_of_tensor.end()){
                clevents = event_of_tensor[one->id].wait_event.event;
                events_num = event_of_tensor[one->id].wait_event.num;
                this_events = event_of_tensor[one->id].this_event.event;
            }
            errNum = clEnqueueReadBuffer(commandQueue, clmem[one->id], CLSynchronize, 0,
                                        PzkM::shape2size(one->shape.dims)*datalen(one->data_type),
                                        dest, events_num, clevents, this_events);
            if (errNum != CL_SUCCESS){
                printf("Error reading clmem buffer (errCode=%d)\n", errNum);
                return false;
            }
            return true;
        }
    }
    /*
    @设置模型输入one
    */
    bool SetAsInputs(std::vector<struct TensorsS*> mores, std::vector<size_t> index){
        size_t num = mores.size();
        if (num <= 0)
            return true;
        // add_wait_event_node((cl_uint)num);
        for(size_t i = 0; i < num; i++){
            struct TensorsS* one = mores[i];
            event_of_tensor[one->id] = all_node_events[index[i]];
        }
        return true;
    }
    /* 设置模型输出 */
    bool SetAsOutputs(std::vector<struct TensorsS*> mores, std::vector<size_t> index){
        size_t num = mores.size();
        if (num <= 0)
            return true;
        // add_wait_event_node(num);
        for(size_t i = 0; i < num; i++){
            struct TensorsS* one = mores[i];
            event_of_tensor[one->id] = all_node_events[index[i]];
        }
        return true;
    }
    /* 根据传入的TensorsS对象写入特定的cpu源数据到cl_mem中 */
    bool WriteCLMem(struct TensorsS* one, void* src){
        if (clmem.find(one->id) == clmem.end()){
           /* 未创建过对应的cl_mem对象 */
           return false; 
        }
        else{
            cl_event* clevents = NULL;
            cl_uint events_num = 0;
            cl_event* this_events = NULL;
            if (event_of_tensor.find(one->id) != event_of_tensor.end()){
                clevents = event_of_tensor[one->id].wait_event.event;
                events_num = event_of_tensor[one->id].wait_event.num;
                this_events = event_of_tensor[one->id].this_event.event;
            }
            errNum = clEnqueueWriteBuffer(commandQueue, clmem[one->id], CLSynchronize, 0,
                                        PzkM::shape2size(one->shape.dims)*datalen(one->data_type),
                                        src, events_num, clevents, this_events);
            if (errNum != CL_SUCCESS){
                printf("Error write clmem buffer\n");
                return false;
            }
            return true;
        }
    }

    enum ExecutionModel{
        SYNCHRONIZE = 0,
        ASYNCHRONOUS = 1,
    };
    /* 设置同步还是异步执行方式 */
    void SetExecutionModel(int exemodel = ExecutionModel::SYNCHRONIZE){
        if (exemodel == ExecutionModel::SYNCHRONIZE){
            CLSynchronize = CL_TRUE;
        }else{
            CLSynchronize = CL_FALSE;
        }
    }

    /* 等待进行 */
    bool WaitFinish(){
        if (CLSynchronize == CL_FALSE)
            return clFinish(commandQueue)==CL_SUCCESS?true:false;
        else
            return true;
    }

    /* 清除OpenCL资源 */
    void Cleanup(){
        for (auto& kv:clmem)
        {
            if (kv.second!= 0)
                clReleaseMemObject(kv.second);
        }
        for (int i = 0; i < all_events_num; i++){
            clReleaseEvent(all_events[i]);
        }
        
        for (auto& kv:event_of_tensor){
            if (kv.second.this_event.NewMalloc && kv.second.this_event.event != NULL  && kv.second.this_event.num > 0){
                clReleaseEvent(*(kv.second.this_event.event));
                free(kv.second.this_event.event);
            }
            if (kv.second.wait_event.NewMalloc && kv.second.wait_event.event != NULL  && kv.second.wait_event.num > 0){
                clReleaseEvent(*(kv.second.wait_event.event));
                free(kv.second.wait_event.event);
            }
        }
        if (commandQueue!= 0)
            clReleaseCommandQueue(commandQueue);
        if (context!= 0)
            clReleaseContext(context);
        
        /* 回复成最初的初始值 */
    }
    /* 初始化函数 */
    void InitCL(){
        /* 创建OpenCL上下文 */
        context = CreateContext(&device);
        if (context == NULL)
        {
            printf("Failed to create OpenCL  context.\n" );
            return;
        }
        /* 获得OpenCL设备,并创建命令队列 */
        commandQueue = CreateCommandQueue();
        if (commandQueue == NULL)
        {
            printf("Failed to create commandQueue.\n");
            Cleanup();
            return;
        }
    }
}
#endif