
__kernel
void convolutionNaive(__global const float* input,
            __global const float* weights,
#ifdef HASBIAS
            __global const float* biases,
#endif
            const uint batchSize,
            const uint inputChannels,
            const uint inputWidth,
            const uint inputHeight,
            const uint kernelWidth,
            const uint kernelHeight,
            const uint padTop,
            const uint padRight,
            const uint padBottom,
            const uint padLeft,
            const uint strideX,
            const uint strideY,
            __global float* output
            ){
  int outputChannels = get_global_size(0) / batchSize;
  int outputHeight = get_global_size(1);
  int outputWeight = get_global_size(2);
  /* NC融合进行之后，如何拆分出相应维度
    错误示例如下：
    解释：会导致实际分配不正确，当batchSize=1，OutputChannels=10的时候，发现oc===0，明显出错。
    int b = get_global_id(0) / batchSize;
    int oc = get_global_id(0) % batchSize;
   */
  int b = get_global_id(0) / outputChannels;/* batchSize被融入到第一个并行度中，N*C */
  int oc = get_global_id(0) % outputChannels;
  int ohx = get_global_id(1); // [0, col_chw)
  int owy = get_global_id(2);
  uint output_offset = b * outputChannels * outputHeight * outputWeight + oc * outputHeight * outputWeight + ohx * outputWeight + owy;
  uint input_feature_map_size = inputHeight * inputWidth;
  uint input_one_size = inputChannels * input_feature_map_size;
  uint weight_feature_map_size = kernelWidth * kernelHeight;
  uint weight_one_size = inputChannels * weight_feature_map_size;
// /* 定义一次卷积的长度=kernelWidth乘kernelHeight */
// #define CalSize 10
//   local float input_reg[CalSize];
//   local float weights_reg[CalSize];
  /*
  [ohx, owy]表示输出特征图的x,y点坐标
  我们需要从输出映射到输入的坐标值，需要考虑到Pad的偏移等因素。
  */
  float result = 0.0;
  int padinputWidthMax = padLeft + inputWidth;
  int padinputHeightMax = padBottom + inputHeight;
  int ihx = ohx * strideX;
  int iwy = owy * strideY;
  /* 首先只进行卷积的weight乘加 */
  for (uint i = 0; i < kernelHeight; i++){
    if (ihx + i < padTop || ihx + i >= padinputHeightMax){
        continue;
    }else{
        for (uint j = 0; j < kernelWidth; j++){
            if (iwy + j < padRight || iwy + j >= padinputWidthMax){
                continue;
            }else{
                /* 此时表示没有超出卷积的尺寸范围之外，所以需要进行卷积操作 */
                uint one_featuremap_offset = (ihx + i - padTop) * inputWidth + (iwy + j - padRight);
                uint one_weight_offset = i * kernelWidth + j;
                for (uint ic = 0; ic < inputChannels; ic++){
                    uint input_ptr = b * input_one_size + ic * input_feature_map_size + one_featuremap_offset;
                    uint weight_ptr = oc * weight_one_size + ic * weight_one_size + one_weight_offset;
                    result += (input[input_ptr] * weights[weight_ptr]);
                }
            }
        }
    }
  }
  /* 然后进行bias的相加 */
#ifdef HASBIAS
  result += biases[oc];
#endif
  output[output_offset] = result;
}