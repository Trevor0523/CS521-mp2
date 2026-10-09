#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8   
#define TILE_W 8   
#define TILE_C 16  

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    extern __shared__ float shmem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int n = blockIdx.z / C_out;
    int oc = blockIdx.z % C_out;
    int base_oh = blockIdx.y * blockDim.y;
    int base_ow = blockIdx.x * blockDim.x;
    int oh = base_oh + ty;
    int ow = base_ow + tx;

    int shared_h = (blockDim.y - 1) * stride + KH;
    int shared_w = (blockDim.x - 1) * stride + KW;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    float acc = 0.0f;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int idx = tid; idx < shared_h * shared_w; idx += num_threads) {
            int ih = base_oh * stride - pad + idx / shared_w;
            int iw = base_ow * stride - pad + idx % shared_w;
            float value = 0.0f;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                value = x[((n * C_in + ic) * H + ih) * W + iw];
            }
            shmem[idx] = value;
        }
        __syncthreads();

        if (oh < out_h && ow < out_w) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    float input_value =
                        shmem[(ty * stride + kh) * shared_w + tx * stride + kw];
                    float weight_value =
                        w[((oc * C_in + ic) * KH + kh) * KW + kw];
                    acc += input_value * weight_value;
                }
            }
        }

        __syncthreads();
    }

    if (oh < out_h && ow < out_w) {
        out[((n * C_out + oc) * out_h + oh) * out_w + ow] = acc;
    }
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    int out_h = (H + 2 * pad - KH) / stride + 1;
    int out_w = (W + 2 * pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    dim3 block(TILE_W, TILE_H);
    dim3 grid((out_w + block.x - 1)/block.x,
              (out_h + block.y - 1)/block.y,
              N * C_out);

    int shared_h = (TILE_H - 1) * stride + KH;
    int shared_w = (TILE_W - 1) * stride + KW;
    size_t shared_bytes = static_cast<size_t>(shared_h) * shared_w * sizeof(float);

    gemm_gpu_o4_kernel<<<grid, block, shared_bytes>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
