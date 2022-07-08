#include "runtime/builder.hpp"
#include "model/pzk.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

#ifdef _WIN32
static double get_current_time()
{
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
}
#else  // _WIN32

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif // _WIN32


/* 一个简单的使用builder的例子 */
void simple_using(char** argv){
    /* 1.初始化 */
    OCLEngine::InitCL();
    /* 2.使用自定义模型创建一个模型表示 */
    PzkM smodel = PzkM(std::string(argv[1]));
    smodel.ReadModel(argv[2]);
    /* 3.进行opencl的运行时网络构建 */
    bool ret = OCLEngine::CreateNetWork(smodel);
    if (ret == false){
        printf("Failed CreateNetWork!\n");
        return;
    }
    /* 开辟cpu空间(input和output) */
    std::unordered_map<size_t, void*> io_tensor;
    for(size_t i = 0; i < OCLEngine::input_tensors.size(); i++){
        io_tensor[OCLEngine::input_tensors[i].id] = malloc(PzkM::shape2size(OCLEngine::input_tensors[i].shape.dims)*datalen(OCLEngine::input_tensors[i].data_type));
    }
//    std::vector<void*> outputs;
    for(size_t i = 0; i < OCLEngine::output_tensors.size(); i++){
        io_tensor[OCLEngine::output_tensors[i].id] = malloc(PzkM::shape2size(OCLEngine::output_tensors[i].shape.dims)*datalen(OCLEngine::output_tensors[i].data_type));
    }
    /* 4.进行绑定 */
    for(auto iter = io_tensor.begin(); iter != io_tensor.end(); iter++){
        OCLEngine::AttachMem(iter->first, iter->second);
    }
    /* 5.自行对上述空间导入输入
        5.进行推理 */
    OCLEngine::SetExecutionModel(OCLEngine::ExecutionModel::ASYNCHRONOUS);/* 设置为 */
    size_t loopnum = 10000;
    double time1 = get_current_time();
    for(size_t i = 0; i < loopnum; i++){
        OCLEngine::Inference();
        if(OCLEngine::WaitFinish() == false){
            printf("error when waiting finish\n");
        }
    }
    double time2 = get_current_time();
    printf("inference time is %.3f fps\n", 1000.f/((time2 - time1)/loopnum));
    OCLEngine::Cleanup();
    /* 6.然后就得到了outputs对应的所有输出信息 */
    return;
}
int main(int argc, char** argv){
    if (argc >= 3){
        simple_using(argv);
    }
    else{
        printf("Usage: ./test json_file_path model_file_path\n");
    }
    return 0;
}