#include "runtime/engine.hpp"

namespace OCLEngine{
    enum Pool2dType{
        POOLMIN = 0,
        POOLMAX = 1,
        POOLMEAN = 2
    };
    struct Pool2dCfg{
        cl_mem* input = NULL;
        cl_mem* output = NULL;
        NodeEvent event;
        int pooltype = 0;
        float poolvalue = 0.f;
        uint batchSize = 1;
        uint inputWidth;
        uint inputHeight;
        uint poolWidth;
        uint poolHeight;
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

    class Pool2dLayer : public CLFunction{
    private:
        Pool2dCfg cfg;
        cl_kernel kernel = NULL;
        size_t* globalWorkSize = NULL;
        size_t* localWorkSize = NULL;
        cl_uint work_dim = 0;
        bool useful = false;
        cl_int Pool2derrNum = CL_SUCCESS;
    public:
        Pool2dLayer() = default;
        ~Pool2dLayer(){
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
        bool configure(Pool2dCfg conf){
            this->cfg  = conf;
            std::vector<std::string> buildOptions;
            /* 设置pool pad对应的值是多少 */
            buildOptions.push_back(std::string("-D POOLVALUE=") + std::to_string(this->cfg.poolvalue));
            /* 设置pool池化的计算类型 */
            if (this->cfg.pooltype == Pool2dType::POOLMIN){
                buildOptions.push_back(std::string("-D POOLTYPE=POOLMIN"));
            }else if (this->cfg.pooltype == Pool2dType::POOLMAX)
            {
                buildOptions.push_back(std::string("-D POOLTYPE=POOLMAX"));
            }else if (this->cfg.pooltype == Pool2dType::POOLMEAN)
            {
                buildOptions.push_back(std::string("-D POOLTYPE=POOLMEAN"));
            }
            
            /* 1、获取对应的核心 */
            this->kernel = ProgramManager.GetKernel(std::string("pool2d.cl"), buildOptions, std::string("pooling2d"));
            if (this->kernel == NULL){
                printf("Get pooling2d kernel of pool2d.cl Failed\n");
                return false;
            }
            /* 2、对核心进行相应的参数设置 */
            cl_uint arg_idx = 0;
            Pool2derrNum = clSetKernelArg(kernel,arg_idx,sizeof (cl_mem),
                                    this->cfg.input);
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.batchSize));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.inputWidth));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.inputHeight));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.poolWidth));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.poolHeight));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.padTop));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.padRight));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.padBottom));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.padLeft));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.strideX));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (uint),&(this->cfg.strideY));
            Pool2derrNum |= clSetKernelArg(kernel,++arg_idx,sizeof (cl_mem),this->cfg.output);
            if (Pool2derrNum != CL_SUCCESS){
                printf("Set kernel Arguments Failed On Pool2d Layers\n");
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
                this->Pool2derrNum = clEnqueueNDRangeKernel(commandQueue,
                                                            this->kernel,
                                                            this->work_dim,
                                                            NULL,
                                                            this->globalWorkSize,
                                                            this->localWorkSize,
                                                            this->cfg.event.wait_event.num,
                                                            this->cfg.event.wait_event.event,
                                                            this->cfg.event.this_event.event);
                if (this->Pool2derrNum != CL_SUCCESS){
                    printf("Inference Pool2d Layers Failed\n");
                    return;
                }
            }else{
                printf("This Pool2d Layers is useless\n");
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