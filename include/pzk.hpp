#include "pzk-schema_generated.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include <iostream> // C++ header file for printing
#include <fstream> // C++ header file for file access
#include "json11.hpp" // for read json file
using namespace PzkModel;

// if this funcation is necessary
bool has_key(json11::Json j, std::string key)
{
    if (j[key].is_null())
        return false;
    else
        return true;
}

// one layer describe from json file
class min_meta
{
private:
    void _Rubber();
    /* data */
public:
    min_meta(){};
    min_meta(json11::Json onelayer);
    ~min_meta();
    void print();
    // remove \"
    std::string remove(std::string a, char rp = 34);
    std::string name;
    std::string category;
    std::map<std::string, std::string> attributes;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string nkey = "name";
    std::string ckey = "category";
    std::string akey = "attributes";
    std::string ikey = "inputs";
    std::string okey = "outputs";
};

min_meta::min_meta(json11::Json onelayer)
{
    if(onelayer.is_null())
    {
        std::cout << " min_meta is null\n" << std::endl;
        return;
    }
    if(onelayer.is_object())
    {
        // get layer name
        assert(!onelayer[nkey].is_null());
        assert(onelayer[nkey].is_string());
        name = onelayer[nkey].dump();
        // get layer category
        if (!onelayer[ckey].is_null())
        {
            category = onelayer[ckey].dump();
        }
        // get layer attributes
        if (!onelayer[akey].is_null() && onelayer[akey].is_array())
        {
            auto alist = onelayer[akey].array_items();
            for (size_t i = 0; i < alist.size(); i++)
            {
                assert(!alist[i].is_null() && alist[i].is_object());
                attributes[alist[i]["name"].dump()] = alist[i]["type"].dump();
            }
        }
        // get layer inputs
        if(!onelayer[ikey].is_null() && onelayer[ikey].is_array())
        {
            auto ilist = onelayer[ikey].array_items();
            for (size_t i = 0; i < ilist.size(); i++)
            {
                assert(!ilist[i].is_null() && ilist[i].is_object());
                inputs.push_back(ilist[i]["name"].dump());
            }
            
        }
        // if (std::find(inputs.begin(), inputs.end(), "input") == inputs.end() || inputs.size() == 0)
        // {
        //     inputs.push_back("input");
        // }
        // get layer outputs
        if(!onelayer[okey].is_null() && onelayer[okey].is_array())
        {
            auto olist = onelayer[okey].array_items();
            for (size_t i = 0; i < olist.size(); i++)
            {
                assert(!olist[i].is_null() && olist[i].is_object());
                outputs.push_back(olist[i]["name"].dump());
            }
            
        }

    }
    this->_Rubber();
}

min_meta::~min_meta()
{
    attributes.clear();
    inputs.clear();
    outputs.clear();
}

std::string min_meta::remove(std::string a, char rp){

    std::string::iterator it;
    std::string str = a;
    for (it = str.begin(); it < str.end(); it++)
    {
        if (*it == rp)
        {
            str.erase(it);
            it--;
            /*
            it--很重要，因为使用erase()删除it指向的字符后，后面的字符就移了过来，
            it指向的位置就被后一个字符填充了，而for语句最后的it++，又使it向后移
            了一个位置，所以就忽略掉了填充过来的这个字符。在这加上it--后就和for
            语句的it++抵消了，使迭代器能够访问所有的字符。
            */
        }
    }
    return str;
}

void min_meta::_Rubber(){
    this->name = this->remove(this->name);
    this->category = this->remove(this->category);
    std::map<std::string, std::string> new_attr;
    for(auto &k: this->attributes){
        new_attr[this->remove(k.first)] = this->remove(k.second);
    }
    this->attributes.clear();
    this->attributes = new_attr;
    for(size_t i = 0; i < this->inputs.size(); i++){
        this->inputs[i] = this->remove(this->inputs[i]);
    }
    for(size_t j = 0; j < this->outputs.size(); j++){
        this->outputs[j] = this->remove(this->outputs[j]);
    }
}

void min_meta::print()
{
    std::cout << nkey << " : " << name << std::endl;
    if (category != "")
    {
        std::cout << ckey << " : " << category << std::endl;
    }
    if (attributes.size() > 0)
    {
        std::cout << akey << ":" << std::endl;
        for (auto i:attributes)
        {
            std::cout << i.first << " : " << i.second << std::endl;   
        }
    }
    if (inputs.size() > 0)
    {
        std::string input_buf = ikey + " : ";
        for (size_t i = 0; i < inputs.size(); i++)
        {
            input_buf += inputs[i] + " , ";
        }
        std::cout << input_buf << std::endl;
    }
    if (outputs.size() > 0)
    {
        std::string output_buf = okey + " : ";
        for (size_t i = 0; i < outputs.size(); i++)
        {
            output_buf += outputs[i] + " , ";
        }
        std::cout << output_buf << std::endl;
    }
}

// for describe the json file to class
class jsonmeta
{
private:
    void _getinfo();
public:
    jsonmeta(){};
    jsonmeta(json11::Json jmeta);
    ~jsonmeta();
    void printinfo();
    bool has_layer(std::string);
    min_meta get_meta(std::string layer);
    void updata(json11::Json jmeta);
    std::vector<min_meta> meta;
    std::map<std::string, size_t> laycategory;
    std::vector<std::string> layname;
};

jsonmeta::jsonmeta(json11::Json jmeta)
{
    updata(jmeta);
}

bool jsonmeta::has_layer(std::string layer)
{
    return std::find(layname.begin(), layname.end(), layer) != layname.end();
}

min_meta jsonmeta::get_meta(std::string layer)
{
    if(has_layer(layer))
    {
        for(size_t i = 0; i < meta.size(); i++)
        {
            if(meta[i].name == layer)
            {
                return meta[i];
            }
        }
    }
    return min_meta();
}

void jsonmeta::updata(json11::Json jmeta)
{
    meta.clear();
    laycategory.clear();
    layname.clear();
    assert(!jmeta.is_null() && jmeta.is_array());
    auto jarray = jmeta.array_items();
    for (size_t i = 0; i < jarray.size(); i++)
    {
        meta.push_back(min_meta(jarray[i]));
    }
    _getinfo();
}
void jsonmeta::_getinfo()
{
    if (meta.size() > 0)
    {
        for (size_t i = 0; i < meta.size(); i++)
        {
            min_meta a = meta[i];
            assert(std::find(layname.begin(), layname.end(), a.name) == layname.end());
            layname.push_back(a.name);
            if (laycategory.find(a.category) != laycategory.end())
            {
                laycategory[a.category] += 1;
            }
            else
            {
                laycategory[a.category] = 1;
            }
 
        }
    }
}

void jsonmeta::printinfo()
{
    std::cout << "Summary:" << std::endl;
    std::cout << "LayerCategory:" << laycategory.size() << std::endl;
    for (auto i:laycategory)
    {
        std::cout << i.first << " : " << i.second << std::endl;
    }
    std::cout << "LayerNum:" << layname.size() << std::endl;
    std::string buf = "LayerName : ";
    for (size_t i = 0; i < layname.size(); i++)
    {
        buf += layname[i] + ",";
    }
    std::cout << buf << std::endl;
    for (size_t i = 0; i < meta.size(); i++)
    {
        std::cout << "<------------" << meta[i].name << "------------>" << std::endl;
        meta[i].print();
    }
    
    
}
jsonmeta::~jsonmeta()
{
    meta.clear();
    laycategory.clear();
    layname.clear();
}
// AttrMeta
struct AMeta{
    std::string key;
    bool require;
    DataType buffer_data;
    uint32_t buffer_ele_num;
    std::vector<uint8_t> buffer;    
};
// Attributes
struct Attrs{
    std::string type;
    uint32_t meta_num;
    uint32_t meta_require_num;
    std::vector<struct AMeta> buffer;
};
// Connect
struct Conn{
    std::string name;
    bool seted;
    bool necesary;
    uint32_t tensor_id;
};
// for build the layer
class layer_maker
{
private:
    /* data */
public:
    layer_maker();
    layer_maker(min_meta layer_meta, uint32_t layerid, std::string layername);
    ~layer_maker();
    bool add_input(uint32_t id, std::string input_name = "");    
    bool add_output(uint32_t id, std::string output_name = "" , bool force_set = true);
    bool add_attr(std::string key, std::vector<uint8_t> buf);
    static DataType string2datatype(std::string a);
    static std::vector<uint32_t> return_id(std::vector<Conn> a);
    min_meta meta_info;
    uint32_t layer_id;
    std::string type;
    std::string name;
    uint8_t input_num = 0;
    uint8_t output_num = 0;
    bool require_attrs = false;
    std::vector<struct Conn> input_id;
    std::vector<struct Conn> output_id;
    struct Attrs attrs;
};

// main class for build pzkmodel
class PzkM
{
private:
    /* data */
public:
    PzkM();
    PzkM(std::string jsonfile);
    ~PzkM();

    void add_info(std::string author="pzk", std::string version="v1.0", std::string model_name="Model");
    void create_time();
    uint32_t layout_len(DataLayout layout);
    std::vector<uint32_t> remark_dims(std::vector<uint32_t> dims, DataLayout layout);
    uint32_t add_input(std::vector<uint32_t> dims, DataLayout layout = DataLayout_NCHW, DataType datatype = DataType_FP32);
    uint32_t add_tensor(std::vector<uint32_t> dims, std::vector<uint8_t> weight, DataLayout layout = DataLayout_NCHW ,TensorType tensor_type = TensorType_CONST , DataType datatype = DataType_FP32);
    bool add_layer(layer_maker layerm);
    bool set_as_output(uint32_t id);
    bool has_tensor(uint32_t id);
    bool has_layer(uint32_t id);
    bool model2file(std::string filepath);
    layer_maker make_empty_layer(std::string layertype, std::string layername = "");
    static uint32_t datatype_len(DataType datatype);
    static json11::Json ReadJson(std::string file);
    static uint32_t shape2size(std::vector<uint32_t> dims);
    jsonmeta meta;
    std::string author;
    std::string version;
    std::string model_name;
    struct tm * target_time;
    uint32_t model_runtime_input_num = 0;
    uint32_t model_runtime_output_num = 0;
    std::vector<uint32_t> model_runtime_input_id;
    std::vector<uint32_t> model_runtime_output_id;
    std::vector<flatbuffers::Offset<Tensor>> tensors;
    std::vector<uint32_t> all_tensor_id;
    std::vector<flatbuffers::Offset<Layer>> layers;
    std::vector<uint32_t> all_layer_id;
    flatbuffers::FlatBufferBuilder builder;
    uint32_t tensor_id = 0;
    uint32_t layer_id = 0;
    
};





layer_maker::layer_maker(min_meta layer_meta, uint32_t layerid, std::string layername = "")
{
    layer_id = layerid;
    meta_info = layer_meta;

    type = meta_info.name;
    if (layername == "")
        name = type;
    else
        name = layername;
    for (size_t i = 0; i < meta_info.inputs.size(); i++)
    {
        struct Conn a;
        a.name = meta_info.inputs[i];
        a.necesary = true;
        a.seted = false;
        input_id.push_back(a);
    }
    for (size_t j = 0; j < meta_info.outputs.size(); j++)
    {
        struct  Conn b;
        b.name = meta_info.okey[j];
        b.necesary = true;
        b.seted = false;
    }
    attrs.type = type + "-" + "Attrs";
    attrs.meta_num = meta_info.attributes.size();
    attrs.meta_require_num = attrs.meta_num;
    for (auto onea:meta_info.attributes)
    {
        struct AMeta m;
        m.key = onea.first;
        m.require = true;
        m.buffer_ele_num = 0;
        m.buffer_data = string2datatype(onea.second);
        attrs.buffer.push_back(m);
    }
    if (attrs.meta_require_num > 0)
    {
        require_attrs = true;
    }
    input_num = input_id.size();
    output_num = output_id.size();
}
// set input for layer
bool layer_maker::add_input(uint32_t id, std::string input_name)
{
    if (input_name == "")
    {
        for (size_t i = 0; i < input_id.size(); i++)
        {
            if (input_id[i].seted == false)
            {
                input_id[i].tensor_id = id;
                return true;
            }
        }
    }
    else
    {
        for (size_t j = 0; j < input_id.size(); j++)
        {
            if (input_id[j].name == input_name)
            {
                input_id[j].tensor_id = id;
                return true;
            }
        }
    }
    return false;
}

bool layer_maker::add_output(uint32_t id, std::string output_name, bool force_set)
{
    bool seted_flag = false;
    if (output_name == "")
    {
        for (size_t i = 0; i < output_id.size(); i++)
        {
            if (output_id[i].seted == false)
            {
                output_id[i].tensor_id = id;
                seted_flag = true;
            }
        }
    }
    else
    {
        for (size_t j = 0; j < output_id.size(); j++)
        {
            if (output_id[j].name == output_name)
            {
                output_id[j].tensor_id = id;
                seted_flag =  true;
            }
        }
    }
    if (seted_flag == false)
    {
        struct Conn one;
        one.name = output_name;
        one.seted = true;
        one.necesary = true;
        one.tensor_id = id;
        output_id.push_back(one);
        seted_flag = true;
    }
    this->output_num = output_id.size();
    if (seted_flag)
        return true;
    return false;
}

bool layer_maker::add_attr(std::string key, std::vector<uint8_t> buf)
{
    if(require_attrs)
    {
        for (size_t i = 0; i < attrs.meta_num; i++)
        {
            if(attrs.buffer[i].key == key)
            {
                uint32_t datalen = PzkM::datatype_len(attrs.buffer[i].buffer_data);
                uint32_t ele_num = buf.size()/datalen; 
                attrs.buffer[i].buffer_ele_num = ele_num;
                for (size_t j = 0; j < ele_num * datalen; j++)
                {
                    attrs.buffer[i].buffer.push_back(buf[j]);
                }
                return true;
            }
        }
    }
    return false;
}

std::vector<uint32_t> layer_maker::return_id(std::vector<Conn> a)
{
    std::vector<uint32_t> b;
    for (size_t i = 0; i < a.size(); i++)
    {
        b.push_back(a[i].tensor_id);
    }
    return b;
}

DataType layer_maker::string2datatype(std::string a)
{
    if (a == "uint32")
    {
        return DataType_UINT32;
    }
    else if (a == "int32")
    {
        return DataType_UINT32;
    }
    else if (a == "uint16")
    {
        return DataType_UINT16;
    }
    else if (a == "int16")
    {
        return DataType_INT16;
    }
    else if (a == "uint8")
    {
        return DataType_UINT8;
    }
    else if (a == "int8")
    {
        return DataType_INT8;
    }
    else if (a == "quint8")
    {
        return DataType_QASYMMEUINT8;
    }
    else if (a == "qint8")
    {
        return DataType_QSYMMEINT8;
    }
    else if (a == "float32")
    {
        return DataType_FP32;
    }
    else if (a == "float16")
    {
        return DataType_FP16;
    }
    else
    {
        return DataType_CHAR;
    }
}
layer_maker::~layer_maker()
{
}



PzkM::PzkM()
{

}

PzkM::PzkM(std::string jsonfile)
{
    json11::Json modeljson = ReadJson(jsonfile);
    meta = jsonmeta(modeljson);
}

void PzkM::add_info(std::string author, std::string version, std::string model_name)
{
    this->author = author;
    this->version = version;
    this->model_name = model_name;
    create_time();
}
void PzkM::create_time()
{
    time_t rawtime;
    std::time(&rawtime);
    this->target_time  = localtime(&rawtime);
}

uint32_t PzkM::layout_len(DataLayout layout)
{
    uint32_t layoutlen = 4;
    switch (layout)
    {
    case DataLayout_NCHW:
    case DataLayout_NHWC:
        layoutlen = 4;
        break;
    case DataLayout_ND:
        layoutlen = 2;
        break;
    case DataLayout_NCD:
        layoutlen = 3;
        break;
    default:
        break;
    }
    return layoutlen;
}

uint32_t PzkM::datatype_len(DataType datatype)
{
    uint32_t datatypelen = 4;
    switch (datatype)
    {
    case DataType_FP32:
    case DataType_INT32:
    case DataType_UINT32:
        datatypelen = 4;
        break;
    case DataType_FP16:
    case DataType_UINT16:
        datatypelen = 2;
        break;
    default:
        datatypelen = 1;
        break;
    }
    return datatypelen;
}

uint32_t PzkM::shape2size(std::vector<uint32_t> dims)
{
    uint32_t a = 1;
    if (dims.size() == 0)
    {
        a = 0;
    }
    for (size_t i = 0; i < dims.size(); i++)
    {
        a = a * dims[i];
    }
    return a;
}

std::vector<uint32_t> PzkM::remark_dims(std::vector<uint32_t> dims, DataLayout layout)
{
    std::vector<uint32_t> dims1(dims);
    std::vector<uint32_t> dims2;
    if (dims1.size() < layout_len(layout))
    {
        size_t align_num = layout_len(layout) - dims1.size();
        for (size_t i = 0; i < align_num; i++)
        {
            dims1.push_back(1);
        }
        dims2 = dims1;
    }
    else if (dims1.size() > layout_len(layout))
    {
        for (size_t i = 0; i < layout_len(layout); i++)
        {
            dims2.push_back(dims1[i]);
        }
        for (size_t j = 0; j < (dims1.size() - layout_len(layout)); j++)
        {
            dims2[dims2.size() - 1] = dims2[dims2.size() - 1] * dims1[dims2.size() + j];
        }
    }
    else{
        dims2 = dims1;
    }
    assert(dims2.size() == layout_len(layout));
    return dims2;
}

// make model input and return the tensor id
uint32_t PzkM::add_input(std::vector<uint32_t> dims, DataLayout layout, DataType datatype)
{
    uint32_t old_id = tensor_id;
    // remark dims by layout
    std::vector<uint32_t> dims1 = remark_dims(dims, layout);
    std::string name = "model_input_" + std::to_string((unsigned int)model_runtime_input_num);
    uint8_t dims_len = dims1.size();
    auto tensor_shape = CreateTensorShape(builder, dims_len, builder.CreateVector(dims1));
    auto tensor = CreateTensor(builder, tensor_id, builder.CreateString(name), 
                                TensorType_DYNAMIC, datatype, layout, tensor_shape);
    tensors.push_back(tensor);
    model_runtime_input_id.push_back(tensor_id);
    all_tensor_id.push_back(tensor_id);
    model_runtime_input_num += 1;
    tensor_id += 1;
    return old_id;
}

// make model tensor and return the tensor id
uint32_t PzkM::add_tensor(std::vector<uint32_t> dims, std::vector<uint8_t> weight, DataLayout layout,TensorType tensor_type, DataType datatype)
{
    uint32_t old_id = tensor_id;
    std::vector<uint32_t> dims1 = remark_dims(dims, layout);
    if (tensor_type == TensorType_CONST)
    {
        assert(weight.size()  == shape2size(dims1) * datatype_len(datatype));
    }
    std::string name = "tensor_" + std::to_string((unsigned int) tensor_id);
    auto tensor_shape = CreateTensorShape(builder, dims1.size(), builder.CreateVector(dims1));
    if (tensor_type == TensorType_CONST)
    {
        auto weight_data = CreateWeights(builder, datatype_len(datatype), shape2size(dims1), builder.CreateVector(weight));
        auto tensor = CreateTensor(builder, tensor_id, builder.CreateString(name), 
                                    tensor_type, datatype, layout, tensor_shape, weight_data);
        tensors.push_back(tensor);
    }
    else
    {
        auto tensor = CreateTensor(builder, tensor_id, builder.CreateString(name), 
                                    tensor_type, datatype, layout, tensor_shape);
        tensors.push_back(tensor);
    }
    all_tensor_id.push_back(tensor_id);
    tensor_id += 1;
    return old_id;
}

// set output by id
bool PzkM::set_as_output(uint32_t id)
{
    if(has_tensor(id))
    {
        model_runtime_output_id.push_back(id);
        model_runtime_output_num += 1;
        return true;
    }
    return false;
}

// if this model has this tensor who's id is this one
bool PzkM::has_tensor(uint32_t id)
{
    return std::find(all_tensor_id.begin(), all_tensor_id.end(), id) != all_tensor_id.end();
}
// if the model has this layer who's id is this one
bool PzkM::has_layer(uint32_t id)
{
    return std::find(all_layer_id.begin(), all_layer_id.end(), id) != all_layer_id.end();
}
// uint32_t PzkM::add_layer(std::vector<uint32_t> input_id, std::vector<uint32_t>)

layer_maker PzkM::make_empty_layer(std::string layertype, std::string layername)
{
    uint32_t old_id = layer_id;
    assert(meta.has_layer(layertype));
    if (layername == "")
        layername = layertype + "_" +std::to_string(old_id);
    layer_id += 1;
    return layer_maker(meta.get_meta(layertype), old_id, layername);
}

// add one layer to this model
bool PzkM::add_layer(layer_maker layerm)
{
    assert(has_layer(layerm.layer_id) == false);
    std::vector<flatbuffers::Offset<Connect>> input_c;
    std::vector<flatbuffers::Offset<Connect>> output_c;
    flatbuffers::Offset<Attributes> attrs = 0;
    for (size_t i = 0; i < layerm.input_id.size(); i++)
    {
        auto con = CreateConnect(builder, builder.CreateString(layerm.input_id[i].name), 
                                    layerm.input_id[i].necesary, layerm.input_id[i].tensor_id);
        input_c.push_back(con);
    }
    for (size_t i = 0; i < layerm.output_id.size(); i++)
    {
        auto con = CreateConnect(builder, builder.CreateString(layerm.output_id[i].name),
                                    layerm.output_id[i].necesary, layerm.output_id[i].tensor_id);
        output_c.push_back(con);
    }
    // set all the attrs
    if (layerm.require_attrs && layerm.attrs.buffer.size() > 0)
    {
        std::vector<flatbuffers::Offset<AttrMeta>> all_attrs;
        for (size_t i = 0; i < layerm.attrs.buffer.size(); i++)
        {
            AMeta a = layerm.attrs.buffer[i];
            auto one_attr = CreateAttrMeta(builder, builder.CreateString(a.key), a.require,
                                            a.buffer_data, a.buffer_ele_num,
                                            builder.CreateVector(a.buffer));
            all_attrs.push_back(one_attr);
        }
        auto layer_attrs = CreateAttributes(builder, builder.CreateString(layerm.attrs.type),
                                            layerm.attrs.meta_num, layerm.attrs.meta_require_num, 
                                            builder.CreateVector(all_attrs));
        auto one_layer = CreateLayer(builder, layerm.layer_id, builder.CreateString(layerm.name),
                                        builder.CreateString(layerm.type), layerm.input_num, layerm.output_num,
                                        builder.CreateVector(input_c), builder.CreateVector(output_c), layerm.require_attrs,
                                        layer_attrs);
        this->layers.push_back(one_layer);
    }
    else
    {
        auto one_layer = CreateLayer(builder, layerm.layer_id, builder.CreateString(layerm.name),
                                builder.CreateString(layerm.type), layerm.input_num, layerm.output_num,
                                builder.CreateVector(input_c), builder.CreateVector(output_c));
        this->layers.push_back(one_layer);
    }
    this->all_layer_id.push_back(layerm.layer_id);
}

// generate the bin file from model
bool PzkM::model2file(std::string filepath)
{
    auto model_time = Createtime(builder, 1900 + target_time->tm_year, 1 + target_time->tm_mon,
                                    target_time->tm_mday, target_time->tm_hour, target_time->tm_min,
                                    target_time->tm_sec);
    auto last_model = CreatePModel(builder, builder.CreateString(author), model_time, 
                                    builder.CreateString(version), builder.CreateString(model_name),
                                    model_runtime_input_num, model_runtime_output_num,
                                    builder.CreateVector(model_runtime_input_id),
                                    builder.CreateVector(model_runtime_output_id),
                                    all_tensor_id.size(), builder.CreateVector(tensors),
                                    all_layer_id.size(), builder.CreateVector(layers));
    builder.Finish(last_model);
    const char* all_buf = reinterpret_cast<char* >(builder.GetBufferPointer());
    int size = builder.GetSize();
    std::cout << "this model size is " << size << std::endl;
    flatbuffers::SaveFile(filepath.c_str(), all_buf, builder.GetSize(), true);
    return true;
}
json11::Json PzkM::ReadJson(std::string file)
{

    std::ifstream in;
    in.open(file, std::ios::in);
    if(!in.is_open())
    {
        std::cout << "open File Failed" << std::endl;
        exit(-1);
    }
    std::string buf;
    std::string line;
    while (getline(in, line))
    {
        buf += line + "\n";
    }
    in.close();
    std::string err;
    auto json = json11::Json::parse(buf, err);
    if (!err.empty()) {
        printf("Failed: %s when open %s\n", err.c_str(), file.c_str());
    } else {
        printf("Result: open %s success\n", file.c_str());
    }
    return json;
}
PzkM::~PzkM()
{
}


