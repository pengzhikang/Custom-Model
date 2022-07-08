
__kernel
void pooling2d(__global const float* input,
            const uint batchSize,
            const uint inputWidth,
            const uint inputHeight,
            const uint poolWidth,
            const uint poolHeight,
            const uint padTop,
            const uint padRight,
            const uint padBottom,
            const uint padLeft,
            const uint strideX,
            const uint strideY,
            __global float* output
            ){
  int Channels = get_global_size(0) / batchSize;
  int outputHeight = get_global_size(1);
  int outputWeight = get_global_size(2);
  int b = get_global_id(0) / Channels;/* batchSize被融入到第一个并行度中，N*C */
  int oc = get_global_id(0) % Channels;
  int ohx = get_global_id(1); // [0, col_chw)
  int owy = get_global_id(2);
  uint output_offset = b * Channels * outputHeight * outputWeight + oc * outputHeight * outputWeight + ohx * outputWeight + owy;
  uint input_feature_map_size = inputHeight * inputWidth;
  uint input_one_size = Channels * input_feature_map_size;
// /* 定义一次pool的长度=poolWidth乘poolHeight */
// #define CalSize 10
//   local float input_reg[CalSize];
//   local float weights_reg[CalSize];
  /*
  [ohx, owy]表示输出特征图的x,y点坐标
  我们需要从输出映射到输入的坐标值，需要考虑到Pad的偏移等因素。
  */

#ifndef POOLVALUE
#define POOLVALUE 0.0
#endif

#define POOLMIN 0 
#define POOLMAX 1
#define POOLMEAN 2
#ifndef POOLTYPE
#define POOLTYPE POOLMIN
#endif


#if POOLTYPE==POOLMIN
  float result = FLT_MAX;
#elif POOLTYPE==POOLMAX;
  float result = -FLT_MAX;
#elif POOLTYPE==POOLMEAN
  float result = 0.0;
#endif
  int padinputWidthMax = padLeft + inputWidth;
  int padinputHeightMax = padBottom + inputHeight;
  int ihx = ohx * strideX;
  int iwy = owy * strideY;
  /* 进行取数和得分操作操作 */
// #ifndef COMPAREDATANUM
// #define COMPAREDATANUM 10
// #endif
//   float compareData[COMPAREDATANUM];
//   uint usefulnum = 0;
  for (uint i = 0; i < poolHeight; i++){
    if (ihx + i < padTop || ihx + i >= padinputHeightMax){
#if POOLTYPE==POOLMIN
        result = (result > POOLVALUE)?POOLVALUE:result;
#elif POOLTYPE==POOLMAX;
        result = (result > POOLVALUE)?result:POOLVALUE;
#elif POOLTYPE==POOLMEAN
        result += POOLVALUE;
#endif
    }else{
        for (uint j = 0; j < poolWidth; j++){
            if (iwy + j < padRight || iwy + j >= padinputWidthMax){
#if POOLTYPE==POOLMIN
                result = (result > POOLVALUE)?POOLVALUE:result;
#elif POOLTYPE==POOLMAX;
                result = (result > POOLVALUE)?result:POOLVALUE;
#elif POOLTYPE==POOLMEAN
                result += POOLVALUE;
#endif
            }else{
                /* 此时表示没有超出对应的尺寸范围之外，所以需要进行池化操作 */
                uint one_featuremap_offset = (ihx + i - padTop) * inputWidth + (iwy + j - padRight);
                for (uint ic = 0; ic < Channels; ic++){
                    uint input_ptr = b * input_one_size + ic * input_feature_map_size + one_featuremap_offset;
#if POOLTYPE==POOLMIN
                    result = (result > input[input_ptr])?input[input_ptr]:result;
#elif POOLTYPE==POOLMAX;
                    result = (result > input[input_ptr])?result:input[input_ptr];
#elif POOLTYPE==POOLMEAN
                    result += input[input_ptr];
#endif
                }
            }
        }
    }
  }
#if POOLTYPE==POOLMEAN
  result /= (poolWidth * poolHeight);
#endif
  output[output_offset] = result;
}