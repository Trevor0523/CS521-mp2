import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here. 

        # Precompute output size
        self.out_h = (H + 2 * padding - kernel_size) // stride + 1
        self.out_w = (W + 2 * padding - kernel_size) // stride + 1

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
        # Refer to Lecture 3 for implementing this operation.

        slices = []
        for kh in range(KH):
            for kw in range(KW):
                slices.append(x_pad[
                    :, :,
                    kh : kh + out_h * S : S,
                    kw : kw + out_w * S : S,
                ])
        patches = torch.stack(slices, dim=2)  # (N, C, KH*KW, out_h, out_w)
        patches = patches.permute(0, 3, 4, 1, 2)
        patches = patches.reshape(N, out_h * out_w, C * KH * KW)
        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        KH = KW = self.kernel_size

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)          

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        weight_flat = self.weight.view(C_out, -1)

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        L = cols.shape[1]
        K = cols.shape[2]
        weight_t = weight_flat.t()
        tile_dim = 32

        out = cols.new_zeros((N, L, C_out))
        for i in range(0, L, tile_dim):
            for j in range(0, C_out, tile_dim):
                for k in range(0, K, tile_dim):
                    i_end = min(i + tile_dim, L)
                    j_end = min(j + tile_dim, C_out)
                    k_end = min(k + tile_dim, K)
                    out[:, i:i_end, j:j_end] += torch.matmul(
                        cols[:, i:i_end, k:k_end],
                        weight_t[k:k_end, j:j_end]
                    )

        # TO DO: 4) Add bias.
        out += self.bias.view(1, 1, C_out)

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        out = out.permute(0, 2, 1).contiguous().view(N, C_out, self.out_h, self.out_w)



        return out

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    # N, C, H, W = 2, 4, 22, 22
    N, C, H, W = 4, 3, 25, 25
    x = torch.randn(N, C, H, W).cuda()
    # out_channels = 8
    # kernel_size = 7
    out_channels = 4
    kernel_size = 6
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).cuda()
    # Warm up before recording steady-state execution.
    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for i in range(3):
                with record_function(f"pytorch_conv_iter_{i}"):
                    out = model(x)
                    torch.cuda.synchronize()

    prof.export_chrome_trace("traces/pytorch_trace_4x3x25x25.json")

    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
