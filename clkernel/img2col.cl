__kernel
void im2col2(__global const float* data_im,
            const int col_chw,
            const int height, const int width,
            const int kernel_c, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int height_col, const int width_col,
            __global float* col_data) {
  int index = get_global_id(0); // [0, col_chw)

  int kernel_hw = kernel_h * kernel_w;
  int kernel_chw = kernel_c * kernel_hw;

  if(index < col_chw) {
    int h_out = index / kernel_chw;
    int w_out = index % kernel_chw;

    int h_step = h_out / width_col;
    int w_step = h_out % width_col;

    int hw_kernel = w_out % kernel_hw;
    int h_kernel = hw_kernel / kernel_w;
    int w_kernel = hw_kernel % kernel_w;

    int c_in = w_out / kernel_hw;
    int h_in = mad24(h_step, stride_h, - pad_h + h_kernel);
    int w_in = mad24(w_step, stride_w, - pad_w + w_kernel);

    int out_c = height_col * width_col;
    int out_index = w_out * out_c + h_out;
    col_data[out_index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width)
        ? data_im[c_in * height * width + h_in * width + w_in]
        : 0;
  }
}

/*
    data_in:img的首地址
    col_chw:进行img2col后的矩阵元素个数
    height:img的高度
    width:img的宽度
    kernel_c:输入的channel数目
    kernel_h:kernel的高
    kernel_w:kernel的宽
    pad_h:高填充
    pad_w:宽填充
    stride_h:高步长
    stride_w:宽步长
    height_col:输出特征图的高
    width_col:输出特征图的宽
    col_data：col的首地址
*/
__kernel
void im2col(__global const float* data_im,
            const int col_chw,
            const int height, const int width,
            const int kernel_c, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int height_col, const int width_col,
            __global float* col_data) {        
    // 根据输出col的序列号绑定工作组
    int index = get_global_id(0); // [0, col_chw)
    // 保证在col_data范围内
    if (index < col_chw)
    {
        /*
        算出index对应的(x,y)
        x决定了对应输出特征图第几个输出点
        x,y决定了其对应输入特征图的第几个channel的哪个坐标上的数
        */
        int col_wp = kernel_h * kernel_w;
        int col_w = kernel_c * col_wp;
        int out_x = index / col_w;
        int out_y = index % col_w;
        /*
        (one_x,one_y)代表在输出特征图上对应的坐标
        */
        int out_one_x = out_x / width_col;
        int out_one_y = out_x % width_col;
        // 计算跟源特征图的对应坐标顶点
        int in_one_x_p = out_one_x * stride_h;
        int in_one_y_p = out_one_y * stride_w;
        // 计算对应哪个kernel，具体哪个位置
        int in_one_kernel_index = out_y / col_wp;
        int in_one_xy = out_y % col_wp;
        //相对位置
        int in_one_dx = in_one_xy / kernel_w;
        int in_one_dy = in_one_xy % kernel_w;
        // 在源上的绝对位置
        int in_one_x = in_one_x_p + in_one_dx - pad_h;
        int in_one_y = in_one_y_p + in_one_dy - pad_w;
        col_data[index] = (in_one_x >= 0 && in_one_x < height && in_one_y >= 0 && in_one_y < width)?data_im[in_one_x*width+in_one_y + in_one_kernel_index * height * width]:0;
    }
}