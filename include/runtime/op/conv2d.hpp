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
            size_t input_num = sizeof(float) * this->cfg.batchSize * this->cfg.inputChannels * this->cfg.inputWidth * this->cfg.inputHeight;
            size_t weight_num = sizeof(float) * this->cfg.outputChannels * this->cfg.inputChannels * this->cfg.kernelWidth * this->cfg.kernelHeight;
            size_t bias_num = (this->cfg.biases == NULL) ? 0: this->cfg.outputChannels * sizeof(float);
            size_t output_num = sizeof(float) * this->cfg.batchSize * this->cfg.outputChannels * this->cfg.outputHeight * this->cfg.outputWeight;
            float* input = (float*)malloc(input_num);
            float* weights = (float*)malloc(weight_num);
            float* biases = (bias_num == 0) ? NULL:(float*)malloc(bias_num);
            float* output = (float*)malloc(output_num);
            float* cl_output = (float*)malloc(output_num);
            /* 读取cl_mem到cpu内存上 */
            ReadCLMem(this->cfg.input, input, input_num);
            ReadCLMem(this->cfg.weights, weights, weight_num);
            if (biases != NULL)
                ReadCLMem(this->cfg.biases, biases, bias_num);
            ReadCLMem(this->cfg.output, cl_output, output_num);
            /* 进行cpu conv2d操作 */
            for(size_t b = 0; b < this->cfg.batchSize; b++){
                for (size_t oc = 0; oc < this->cfg.outputChannels; oc++){
                    for(size_t ohx = 0; ohx < this->cfg.outputHeight; ohx++){
                        for(size_t owy = 0; owy < this->cfg.outputWeight; owy++){
                            uint output_offset = b * this->cfg.outputChannels * this->cfg.outputHeight * this->cfg.outputWeight + oc * this->cfg.outputHeight * this->cfg.outputWeight + ohx * this->cfg.outputWeight + owy;
                            uint input_feature_map_size = this->cfg.inputHeight * this->cfg.inputWidth;
                            uint input_one_size = this->cfg.inputChannels * input_feature_map_size;
                            uint weight_feature_map_size = this->cfg.kernelWidth * this->cfg.kernelHeight;
                            uint weight_one_size = this->cfg.inputChannels * weight_feature_map_size;
                            // /* 定义一次卷积的长度=kernelWidth乘kernelHeight */
                            // #define CalSize 10
                            //   local float input_reg[CalSize];
                            //   local float weights_reg[CalSize];
                            /*
                            [ohx, owy]表示输出特征图的x,y点坐标
                            我们需要从输出映射到输入的坐标值，需要考虑到Pad的偏移等因素。
                            */
                            float result = 0.0;
                            int padinputWidthMax = this->cfg.padLeft + this->cfg.inputWidth;
                            int padinputHeightMax = this->cfg.padBottom + this->cfg.inputHeight;
                            int ihx = ohx * this->cfg.strideX;
                            int iwy = owy * this->cfg.strideY;
                            /* 首先只进行卷积的weight乘加 */
                            for (uint i = 0; i < this->cfg.kernelHeight; i++){
                                if (ihx + i < this->cfg.padTop || ihx + i >= padinputHeightMax){
                                    continue;
                                }else{
                                    for (uint j = 0; j < this->cfg.kernelWidth; j++){
                                        if (iwy + j < this->cfg.padRight || iwy + j >= padinputWidthMax){
                                            continue;
                                        }else{
                                            /* 此时表示没有超出卷积的尺寸范围之外，所以需要进行卷积操作 */
                                            uint one_featuremap_offset = (ihx + i - this->cfg.padTop) * this->cfg.inputWidth + (iwy + j - this->cfg.padRight);
                                            uint one_weight_offset = i * this->cfg.kernelWidth + j;
                                            for (uint ic = 0; ic < this->cfg.inputChannels; ic++){
                                                uint input_ptr = b * input_one_size + ic * input_feature_map_size + one_featuremap_offset;
                                                uint weight_ptr = oc * weight_one_size + ic * weight_one_size + one_weight_offset;
                                                result += (input[input_ptr] * weights[weight_ptr]);
                                            }
                                        }
                                    }
                                }
                            }
                            /* 然后进行bias的相加 */
                            if (biases != NULL)
                                result += biases[oc];
                            output[output_offset] = result;
                        }
                    }
                }
            }

            /* 进行对比操作 */
            double cos_sim = CosineSimilarity<float>(cl_output, output, output_num/sizeof(float));
            printf("conv2d CosineSimilarity value = %.10lf\n", cos_sim * 100.0);
            free(input);
            free(weights);
            if (biases != NULL)
                free(biases);
            free(output);
            free(cl_output);
            return;
        };
    };
}