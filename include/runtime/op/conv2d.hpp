#include "runtime/engine.hpp"

namespace OCLEngine{
    struct Conv2dCfg{
        cl_mem* input = NULL;
        cl_mem* weights = NULL;
        cl_mem* biases = NULL;
        cl_mem* output = NULL;
        NodeEvent event;
        uint batchSize = 1;
        uint inputChannels;
        uint inputWidth;
        uint inputHeight;
        uint kernelWidth;
        uint kernelHeight;
        uint padTop = 0;
        uint padRight = 0;
        uint padBottom = 0;
        uint padLeft = 0;
        uint strideX;
        uint strideY;
        size_t outputChannels;
        size_t outputHeight;
        size_t outputWeight;
    };

    class Conv2dLayer : public CLFunction{
    private:
        Conv2dCfg cfg;
        cl_kernel kernel = NULL;
        size_t* globalWorkSize = NULL;
        size_t* localWorkSize = NULL;
        cl_uint work_dim = 0;
        bool useful = false;
        cl_int Conv2derrNum = CL_SUCCESS;
    public:
        Conv2dLayer() = default;
        ~Conv2dLayer(){
            if (this->globalWorkSize != NULL){
                free(this->globalWorkSize);
            }
            if (this->localWorkSize != NULL){
                free(this->localWorkSize);
            }
            if (this->kernel != NULL){
                clReleaseKernel(kernel);
            }
        };
        // 配置函数
        bool configure(Conv2dCfg conf){
            this->cfg  = conf;
            std::vector<std::string> buildOptions;
            if (this->cfg.biases != NULL){
                /* 如果有bias，则进行如下所示的编译命令 */
                buildOptions.push_back(std::string("-D HASBIAS"));
            }
            /* 1、获取对应的核心 */
            this->kernel = ProgramManager.GetKernel(std::string("conv2d.cl"), buildOptions, std::string("convolutionNaive"));
            if (this->kernel == NULL){
                printf("Get convolutionNaive kernel of conv2d.cl Failed\n");
                return false;
            }
            /* 2、对核心进行相应的参数设置 */
            cl_uint arg_idx = 0;
            Conv2derrNum = clSetKernelArg(kernel,arg_idx,sizeof (cl_mem),
                                    this->cfg.input);
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint), this->cfg.weights);
            if (this->cfg.biases != NULL)
                Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (cl_mem),this->cfg.biases);
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.batchSize));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.inputChannels));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.inputWidth));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.inputHeight));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.kernelWidth));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.kernelHeight));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.padTop));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.padRight));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.padBottom));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.padLeft));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.strideX));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.strideY));
            Conv2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (cl_mem),this->cfg.output);
            if (Conv2derrNum != CL_SUCCESS){
                printf("Set kernel Arguments Failed On Conv2d Layers\n");
                return false;
            }
            /* 3、 设置全局尺寸和局部尺寸大小，以便于后续的加入命令队列操作 */
            this->work_dim = 3;
            this->globalWorkSize = (size_t*)malloc(sizeof(size_t) * this->work_dim);
            this->globalWorkSize[0] = this->cfg.batchSize * this->cfg.outputChannels;
            this->globalWorkSize[1] = this->cfg.outputHeight;
            this->globalWorkSize[2] = this->cfg.outputWeight;
            this->localWorkSize = (size_t*)malloc(sizeof(size_t) * this->work_dim);
            this->localWorkSize[0] = 1;
            this->localWorkSize[1] = 1;
            this->localWorkSize[2] = 1;
            useful = true;
            return true;
        };
        // 重载函数，主要的run函数
        void run() override{
            if (this->useful){
                this->Conv2derrNum = clEnqueueNDRangeKernel(commandQueue,
                                                            this->kernel,
                                                            this->work_dim,
                                                            NULL,
                                                            this->globalWorkSize,
                                                            this->localWorkSize,
                                                            this->cfg.event.wait_event.num,
                                                            this->cfg.event.wait_event.event,
                                                            this->cfg.event.this_event.event);
                if (this->Conv2derrNum != CL_SUCCESS){
                    printf("Inference Conv2d Layers Failed\n");
                    return;
                }
            }else{
                printf("This Conv2d Layers is useless\n");
                return;
            }
        };
        /* cpu推理函数，主要用于测试 
            此时，因为其父类拥有这个
        */
        void cpu_run() override{

        };
    };
}