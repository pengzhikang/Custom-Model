#include "runtime/op/img2col.hpp"
#include "runtime/op/conv2d.hpp"
#include "runtime/op/pool2d.hpp"
#include <memory>
namespace OCLEngine{
    std::vector<std::shared_ptr<CLFunction>> AllLayers;
    int find_tensor_by_id(std::vector<TensorsS> Tensor, uint32_t id){
        int index = -1;
        for(size_t i = 0; i < Tensor.size(); i++){
            if (Tensor[i].id == id){
                return i;
            }
        }
        return index;
    }
    /* img2col网络层的构建 */
    bool add_img2col_layer(layer_maker l, NodeEvent node_event, std::vector<TensorsS> input, std::vector<TensorsS> output){

        return true;
    }
    /* conv2d网络层的构建 */
    bool add_conv2d_layer(layer_maker l, NodeEvent node_event, std::vector<TensorsS> input, std::vector<TensorsS> output){
        Conv2dCfg cfg;
        cfg.input = l.get_input_id("input") != -1 ? &(clmem[l.get_input_id("input")]):NULL;
        cfg.weights = l.get_input_id("weights") != -1 ? &(clmem[l.get_input_id("weights")]):NULL;
        cfg.biases = l.get_input_id("biases") != -1 ? &(clmem[l.get_input_id("biases")]):NULL;
        cfg.output = l.get_output_id("conv2d-output") != -1 ? &(clmem[l.get_output_id("conv2d-output")]):NULL;
        if (cfg.input == NULL || cfg.weights == NULL || cfg.output == NULL){
            return false;
        }
        cfg.event = node_event;
        TensorsS input_tensor = input[find_tensor_by_id(input, l.get_input_id("input"))];
        cfg.batchSize = input_tensor.shape.dims[0];
        cfg.inputChannels = input_tensor.shape.dims[1];
        cfg.inputHeight = input_tensor.shape.dims[2];
        cfg.inputWidth = input_tensor.shape.dims[3];
        TensorsS weight_tensor = input[find_tensor_by_id(input, l.get_input_id("weights"))];
        cfg.kernelHeight = weight_tensor.shape.dims[2];
        cfg.kernelWidth = weight_tensor.shape.dims[3];
        cfg.padTop = l.get_attr<uint>(std::string("padTop")).size() == 0 ? 0:l.get_attr<uint>(std::string("padTop"))[0];
        cfg.padRight = l.get_attr<uint>(std::string("padRight")).size() == 0 ? 0:l.get_attr<uint>(std::string("padRight"))[0];
        cfg.padBottom = l.get_attr<uint>(std::string("padBottom")).size() == 0 ? 0:l.get_attr<uint>(std::string("padBottom"))[0];
        cfg.padLeft = l.get_attr<uint>(std::string("padLeft")).size() == 0 ? 0:l.get_attr<uint>(std::string("padLeft"))[0];
        cfg.strideX = l.get_attr<uint>(std::string("strideX")).size() == 0 ? cfg.kernelWidth:l.get_attr<uint>(std::string("strideX"))[0];
        cfg.strideY = l.get_attr<uint>(std::string("strideY")).size() == 0 ? cfg.kernelHeight:l.get_attr<uint>(std::string("strideY"))[0];
        TensorsS output_tensor = output[find_tensor_by_id(output, l.get_output_id("conv2d-output"))];
        cfg.outputChannels = output_tensor.shape.dims[1];
        cfg.outputHeight = output_tensor.shape.dims[2];
        cfg.outputWeight = output_tensor.shape.dims[3];
        /* 正式构建卷积层 */
        std::shared_ptr<Conv2dLayer> conv2d = std::make_shared<Conv2dLayer>();
        if (conv2d->configure(cfg) == false){
            printf("conv2d make failed\n");
            return false;
        }else{
            AllLayers.push_back(conv2d);
            return true;
        }
    }
    bool add_pool2d_layer(layer_maker l, NodeEvent node_event, std::vector<TensorsS> input, std::vector<TensorsS> output){
        Pool2dCfg cfg;
        cfg.input = l.get_input_id("input") != -1 ? &(clmem[l.get_input_id("input")]):NULL;
        cfg.output = l.get_output_id("output") != -1 ? &(clmem[l.get_output_id("output")]):NULL;
        if (cfg.input == NULL || cfg.output == NULL){
            return false;
        }
        cfg.event = node_event;
        TensorsS input_tensor = input[find_tensor_by_id(input, l.get_input_id("input"))];
        cfg.batchSize = input_tensor.shape.dims[0];
        cfg.inputHeight = input_tensor.shape.dims[2];
        cfg.inputWidth = input_tensor.shape.dims[3];
        cfg.poolHeight = l.get_attr<uint>(std::string("poolHeight")).size() == 0 ? 2:l.get_attr<uint>(std::string("poolHeight"))[0];
        cfg.poolWidth = l.get_attr<uint>(std::string("poolWidth")).size() == 0 ? 2:l.get_attr<uint>(std::string("poolWidth"))[0];
        cfg.padTop = l.get_attr<uint>(std::string("padTop")).size() == 0 ? 0:l.get_attr<uint>(std::string("padTop"))[0];
        cfg.padRight = l.get_attr<uint>(std::string("padRight")).size() == 0 ? 0:l.get_attr<uint>(std::string("padRight"))[0];
        cfg.padBottom = l.get_attr<uint>(std::string("padBottom")).size() == 0 ? 0:l.get_attr<uint>(std::string("padBottom"))[0];
        cfg.padLeft = l.get_attr<uint>(std::string("padLeft")).size() == 0 ? 0:l.get_attr<uint>(std::string("padLeft"))[0];
        cfg.strideX = l.get_attr<uint>(std::string("strideX")).size() == 0 ? cfg.poolWidth:l.get_attr<uint>(std::string("strideX"))[0];
        cfg.strideY = l.get_attr<uint>(std::string("strideY")).size() == 0 ? cfg.poolHeight:l.get_attr<uint>(std::string("strideY"))[0];
        TensorsS output_tensor = output[find_tensor_by_id(output, l.get_output_id("output"))];
        cfg.outputChannels = output_tensor.shape.dims[1];
        cfg.outputHeight = output_tensor.shape.dims[2];
        cfg.outputWeight = output_tensor.shape.dims[3];
        /* 正式构建卷积层 */
        std::shared_ptr<Pool2dLayer> pool2d = std::make_shared<Pool2dLayer>();
        if (pool2d->configure(cfg) == false){
            printf("conv2d make failed\n");
            return false;
        }else{
            AllLayers.push_back(pool2d);
            return true;
        }
    }
    /* upsample网络层的构建 */

    /* 构建运行时的网络层 */
    bool BuildLayers(PzkM model){
        bool ret = true;
        for (size_t i = 0; i < model.rLayers.size(); i++){
            /* 进行各种不同类型的选择 */
            layer_maker onelayer = model.rLayers[i];
            NodeEvent node_event = all_node_events[model.model_runtime_input_id.size() + i];
            std::vector<TensorsS> input_tensor = model.get_layer_input(onelayer);
            std::vector<TensorsS> output_tensor = model.get_layer_output(onelayer);
            if (onelayer.type == "img2col"){
                ret = add_img2col_layer(onelayer, node_event, input_tensor, output_tensor);
            }else if (onelayer.type == "Convolution2dLayer"){
                ret = add_conv2d_layer(onelayer, node_event, input_tensor, output_tensor);
            }else if (onelayer.type == "Pooling2dLayer"){
                ret = add_pool2d_layer(onelayer, node_event, input_tensor, output_tensor);
            }
            else{
                printf("unknown type = %s layer, cant't finish it\n", onelayer.type.c_str());
                return false;
            }
            /* 查看是否正确与否 */
            if (!ret){
                printf("failed to build type=%s, name=%s Layers\n", onelayer.type.c_str(), onelayer.name.c_str());
                return false;
            }
        }
        return true;
    }
}