#include "pzk.hpp"
#include <iostream>

std::vector<float> rand_weight(uint32_t num=100)
{
    srand(num);
    std::vector<float> weight;
    for (size_t i = 0; i < num; i++)
    {
        weight.push_back( ((rand() % 10) - 4.5f) / 4.5f);
    }
    return weight;
}
template<class T>
std::vector<uint8_t> fp2ubyte(std::vector<T> w1)
{
    std::vector<uint8_t> buf;
    for (size_t i = 0; i < w1.size(); i++)
    {
        T* one = &w1[i];
        uint8_t* charp = reinterpret_cast<uint8_t*>(one); 
        buf.push_back(charp[0]);
        buf.push_back(charp[1]);
        buf.push_back(charp[2]);
        buf.push_back(charp[3]);
    }
    return buf;
    
}

void test_save()
{
    const char* a = "123123131";
    flatbuffers::SaveFile("1.bin", a, 9, true);
}


int main(int argc, char **argv) {
    if(argc == 3 && argv[1] == std::string("--json"))
    {
        // 1. init PzkM by one json file
        PzkM smodel(argv[2]);
        //PzkM smodel("/home/pack/custom-model/model-flatbuffer/pzk-metadata.json");
        // 2. add some model info
        smodel.add_info("pengzhikang", "v2.1", "holly-model");
        smodel.create_time();
        // 3. add inputs for model
        std::vector<uint32_t> input_dims = {1,3,416,416};
        uint32_t input_id = smodel.add_input(input_dims);
        // 4. add layer for model
        // 4.1 get one empty by layer_type
        layer_maker l = smodel.make_empty_layer("\"Convolution2dLayer\"", "conv2d-index-1");
        // 4.2 make weight tensor for conv2d
        std::vector<float> org_weight  = rand_weight(10*3*4*4);
        std::vector<uint32_t> wdims;
        wdims.push_back(10);
        wdims.push_back(3);
        wdims.push_back(4);
        wdims.push_back(4);
        uint32_t weight_id = smodel.add_tensor(wdims,
                                            fp2ubyte<float>(org_weight));
        // 4.3 make bias tensor for conv2d
        std::vector<float> org_bias = rand_weight(10);
        uint32_t bias_id = smodel.add_tensor(std::vector<uint32_t>({10}),
                                            fp2ubyte<float>(org_bias));
        // 4.4 make output layer for conv2d
        uint32_t output_id = smodel.add_tensor(std::vector<uint32_t>({1,10,416/4,416/4}),
                                                std::vector<uint8_t>(), DataLayout_NCHW, TensorType_DYNAMIC);
        // 4.5 Configuration conv2d layer
        l.add_input(input_id, "\"input\"");
        l.add_input(weight_id, "\"weights\"");
        l.add_input(bias_id, "\"biases\"");
        l.add_output(output_id, "\"conv2d-output\"");
        l.add_attr("\"padTop\"", fp2ubyte<uint32_t>(std::vector<uint32_t>({0})));
        // 4.6 add this layer to smodel
        smodel.add_layer(l);
        // 4.7 set the conv2d output is model output
        smodel.set_as_output(output_id);
        // 4.8 generate smodel ro model file
        smodel.model2file("first.PZKM");
        
    }
    else{
        test_save();
    }
    return 0;
}