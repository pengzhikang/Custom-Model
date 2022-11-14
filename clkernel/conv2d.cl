
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
                #ifdef SUPPORT_FLOAT_VECTOR
                // 矢量数据类型进行加速
                // 编写simd加速代码
                #ifdef VECTOR_SIZE_128
                #define FLOAT_PER_VECTOR 4
                #define UINT_PER_VECTOR 4
                uint4 input_ptr_bias = (uint4)(b * input_one_size + one_featuremap_offset);
                uint4 input_ptr_scale = (uint4)(input_feature_map_size);
                uint4 input_ptr = (uint4)(0);

                uint4 weight_ptr_bias = (uint4)(oc * weight_one_size + one_weight_offset);
                uint4 weight_ptr_scale = (uint4)(weight_one_size);
                uint4 weight_ptr = (uint4)(0);

                uint4 vector_ic = (uint4)(0, 1, 2, 3);//ic矢量

                // 计算的数值
                float4 input_vector;
                float4 weight_vector;
                float4 tmp_result;
                uint loop_time = (uint)floor((float)inputChannels/FLOAT_PER_VECTOR);
                // 首先进行simd部分
                while(loop_time>0){
                  // 地址计算
                  input_ptr = input_ptr_bias + input_ptr_scale * vector_ic;
                  weight_ptr = weight_ptr_bias + weight_ptr_scale * vector_ic;
                  // 数据加载
                  input_vector.s0 = input[input_ptr.s0];
                  input_vector.s1 = input[input_ptr.s1];
                  input_vector.s2 = input[input_ptr.s2];
                  input_vector.s3 = input[input_ptr.s3];
                  weight_vector.s0 = weights[weight_ptr.s0];
                  weight_vector.s1 = weights[weight_ptr.s1];
                  weight_vector.s2 = weights[weight_ptr.s2];
                  weight_vector.s3 = weights[weight_ptr.s3];
                  // 向量乘
                  tmp_result = input_vector * weight_vector;
                  // 归约+
                  result += (tmp_result.s0 + tmp_result.s1 + tmp_result.s2 + tmp_result.s3);
                  // 循环
                  loop_time--;
                  vector_ic = vector_ic + FLOAT_PER_VECTOR;

                }
                #endif
                #ifdef VECTOR_SIZE_256
                #define FLOAT_PER_VECTOR 8
                #define UINT_PER_VECTOR 4
                uint8 input_ptr_bias = (uint8)(b * input_one_size + one_featuremap_offset);
                uint8 input_ptr_scale = (uint8)(input_feature_map_size);
                uint8 input_ptr = (uint8)(0);

                uint8 weight_ptr_bias = (uint8)(oc * weight_one_size + one_weight_offset);
                uint8 weight_ptr_scale = (uint8)(weight_one_size);
                uint8 weight_ptr = (uint8)(0);

                uint8 vector_ic = (uint8)(0, 1, 2, 3, 4, 5, 6, 7);//ic矢量

                // 计算的数值
                float8 input_vector;
                float8 weight_vector;
                float8 tmp_result;
                uint loop_time = (uint)floor((float)inputChannels/FLOAT_PER_VECTOR);
                // 首先进行simd部分
                while(loop_time>0){
                  // 地址计算
                  input_ptr = input_ptr_bias + input_ptr_scale * vector_ic;
                  weight_ptr = weight_ptr_bias + weight_ptr_scale * vector_ic;
                  // 数据加载
                  input_vector.s0 = input[input_ptr.s0];
                  input_vector.s1 = input[input_ptr.s1];
                  input_vector.s2 = input[input_ptr.s2];
                  input_vector.s3 = input[input_ptr.s3];
                  input_vector.s4 = input[input_ptr.s4];
                  input_vector.s5 = input[input_ptr.s5];
                  input_vector.s6 = input[input_ptr.s6];
                  input_vector.s7 = input[input_ptr.s7];
                  weight_vector.s0 = weights[weight_ptr.s0];
                  weight_vector.s1 = weights[weight_ptr.s1];
                  weight_vector.s2 = weights[weight_ptr.s2];
                  weight_vector.s3 = weights[weight_ptr.s3];
                  weight_vector.s4 = weights[weight_ptr.s4];
                  weight_vector.s5 = weights[weight_ptr.s5];
                  weight_vector.s6 = weights[weight_ptr.s6];
                  weight_vector.s7 = weights[weight_ptr.s7];
                  // 向量乘
                  tmp_result = input_vector * weight_vector;
                  // 归约+
                  result += (tmp_result.s0 + tmp_result.s1 + tmp_result.s2 + tmp_result.s3 + tmp_result.s4 + tmp_result.s5 + tmp_result.s6 + tmp_result.s7);
                  // 循环
                  loop_time--;
                  vector_ic = vector_ic + FLOAT_PER_VECTOR;

                }
                #endif
                // 剩余部分
                for (uint leftover_start_ptr = loop_time * FLOAT_PER_VECTOR;; leftover_start_ptr < inputChannels; leftover_start_ptr++){
                    uint input_ptr = b * input_one_size + leftover_start_ptr * input_feature_map_size + one_featuremap_offset;
                    uint weight_ptr = oc * weight_one_size + leftover_start_ptr * weight_one_size + one_weight_offset;
                    result += (input[input_ptr] * weights[weight_ptr]);
                }
                #else
                //原始的c代码
                for (uint ic = 0; ic < inputChannels; ic++){
                    uint input_ptr = b * input_one_size + ic * input_feature_map_size + one_featuremap_offset;
                    uint weight_ptr = oc * weight_one_size + ic * weight_one_size + one_weight_offset;
                    result += (input[input_ptr] * weights[weight_ptr]);
                }
                #endif
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
