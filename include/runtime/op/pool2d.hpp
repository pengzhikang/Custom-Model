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
        /* cpu推理函数，主要用于测试核的精度
            此时，因为其父类拥有这个
        */
        void cpu_run() override{
            size_t input_num = sizeof(float) * this->cfg.batchSize * this->cfg.outputChannels * this->cfg.inputWidth * this->cfg.inputHeight;
            size_t output_num = sizeof(float) * this->cfg.batchSize * this->cfg.outputChannels * this->cfg.outputHeight * this->cfg.outputWeight;
            float* input = (float*)malloc(input_num);
            float* output = (float*)malloc(output_num);
            float* cl_output = (float*)malloc(output_num);
            float POOLVALUE = this->cfg.poolvalue;
            /* 读取cl_mem到cpu内存上 */
            ReadCLMem(this->cfg.input, input, input_num);
            ReadCLMem(this->cfg.output, cl_output, output_num);
            /* 进行cpu conv2d操作 */
            for(size_t b = 0; b < this->cfg.batchSize; b++){
                for (size_t oc = 0; oc < this->cfg.outputChannels; oc++){
                    for(size_t ohx = 0; ohx < this->cfg.outputHeight; ohx++){
                        for(size_t owy = 0; owy < this->cfg.outputWeight; owy++){
                            uint output_offset = b * this->cfg.outputChannels * this->cfg.outputHeight * this->cfg.outputWeight + oc * this->cfg.outputHeight * this->cfg.outputWeight + ohx * this->cfg.outputWeight + owy;
                            uint input_feature_map_size = this->cfg.inputHeight * this->cfg.inputWidth;
                            uint input_one_size = this->cfg.outputChannels * input_feature_map_size;
                            float result = 0.0;
                            if (this->cfg.pooltype == POOLMIN){
                                result = (CL_FLT_MAX);
                            }else if(this->cfg.pooltype == POOLMAX){
                                result = - (CL_FLT_MAX);
                            }else if(this->cfg.pooltype == POOLMEAN){
                                result = 0.0;
                            }
                            int padinputWidthMax = this->cfg.padLeft + this->cfg.inputWidth;
                            int padinputHeightMax = this->cfg.padBottom + this->cfg.inputHeight;
                            int ihx = ohx * this->cfg.strideX;
                            int iwy = owy * this->cfg.strideY;
                            /* 进行取数和得分操作操作 */
                            // #ifndef COMPAREDATANUM
                            // #define COMPAREDATANUM 10
                            // #endif
                            //   float compareData[COMPAREDATANUM];
                            //   uint usefulnum = 0;
                            for (uint i = 0; i < this->cfg.poolHeight; i++){
                                if (ihx + i < this->cfg.padTop || ihx + i >= padinputHeightMax){
                                    if (this->cfg.pooltype == POOLMIN){
                                        result = (result > POOLVALUE)?POOLVALUE:result;
                                    }else if(this->cfg.pooltype == POOLMAX){
                                        result = (result > POOLVALUE)?result:POOLVALUE;
                                    }else if(this->cfg.pooltype == POOLMEAN){
                                        result += POOLVALUE;
                                    }
                                }else{
                                    for (uint j = 0; j < this->cfg.poolWidth; j++){
                                        if (iwy + j < this->cfg.padRight || iwy + j >= padinputWidthMax){
                                            if (this->cfg.pooltype == POOLMIN){
                                                result = (result > POOLVALUE)?POOLVALUE:result;
                                            }else if(this->cfg.pooltype == POOLMAX){
                                                result = (result > POOLVALUE)?result:POOLVALUE;
                                            }else if(this->cfg.pooltype == POOLMEAN){
                                                result += POOLVALUE;
                                            }
                                        }else{
                                            /* 此时表示没有超出对应的尺寸范围之外，所以需要进行池化操作 */
                                            uint one_featuremap_offset = (ihx + i - this->cfg.padTop) * this->cfg.inputWidth + (iwy + j - this->cfg.padRight);
                                            for (uint ic = 0; ic < this->cfg.outputChannels; ic++){
                                                uint input_ptr = b * input_one_size + ic * input_feature_map_size + one_featuremap_offset;
                                                if (this->cfg.pooltype == POOLMIN){
                                                    result = (result > input[input_ptr])?input[input_ptr]:result;
                                                }else if(this->cfg.pooltype == POOLMAX){
                                                    result = (result > input[input_ptr])?result:input[input_ptr];
                                                }else if(this->cfg.pooltype == POOLMEAN){
                                                    result += input[input_ptr];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (this->cfg.pooltype == POOLMEAN)
                                result /= (this->cfg.poolWidth * this->cfg.poolHeight);
                            output[output_offset] = result;
                        }
                    }
                }
            }

            /* 进行对比操作 */
            double cos_sim = CosineSimilarity<float>(cl_output, output, output_num/sizeof(float));
            printf("pool2d CosineSimilarity value = %.10lf\n", cos_sim * 100.0);
            free(input);
            free(output);
            free(cl_output);
            return;
        };
    };
}