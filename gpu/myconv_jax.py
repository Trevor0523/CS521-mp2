import time
import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel
import jax.profiler

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))

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
    patches = jnp.stack(slices, axis=2)  # (N, C, KH*KW, out_h, out_w)
    patches = patches.transpose(0, 3, 4, 1, 2)
    patches = patches.reshape(N, out_h * out_w, C * KH * KW)
    return patches

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h = (H + 2 * padding - KH) // stride + 1
    out_w = (W + 2 * padding - KW) // stride + 1
    
    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    weight_flat = weight.reshape(C_out, -1)

    # TO DO: 3) perform tiled matmul after required reshaping is done.
    L = cols.shape[1]
    K = cols.shape[2]
    weight_t = weight_flat.T
    tile_dim = 32

    out = jnp.zeros((N, L, C_out))
    for i in range(0, L, tile_dim):
        for j in range(0, C_out, tile_dim):
            for k in range(0, K, tile_dim):
                i_end = min(i + tile_dim, L)
                j_end = min(j + tile_dim, C_out)
                k_end = min(k + tile_dim, K)
                out = out.at[:, i:i_end, j:j_end].add(
                    jnp.matmul(
                        cols[:, i:i_end, k:k_end],
                        weight_t[k:k_end, j:j_end]
                    )
                )

    # TO DO: 4) Add bias.
    out = out + bias.reshape(1, 1, C_out)

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    out = out.transpose(0, 2, 1).reshape(N, C_out, out_h, out_w)

    return out

if __name__ == "__main__":
    print("JAX backend:", jax.default_backend())

    # Instantiate PyTorch model
    # H, W = 33, 33
    # model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
    N, C, H, W = 4, 3, 25, 25
    model = ConvModel(H, W, in_channels=C, out_channels=4, kernel_size=6, stride=1, padding=1)
    model.eval()

    # Example input
    # x_torch = torch.randn(1, 3, H, W)
    x_torch = torch.randn(N, C, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # Exclude input transfer from host lowering/compilation measurements.
    jax.block_until_ready((x_jax, weight_jax, bias_jax))
    compile_start = time.perf_counter()
    conv2d_manual_jax_jit = jit(conv2d_manual_jax).lower(x_jax, weight_jax, bias_jax).compile()
    print(f"JAX --- host lowering + compilation: {time.perf_counter() - compile_start:.6f} s")

    # call your JAX function
    for _ in range(5):
        out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)
        out_jax.block_until_ready()

    torch.cuda.init()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(3):
            with record_function(f"jax_conv_iter_{i}"):
                out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)
                out_jax.block_until_ready()

    prof.export_chrome_trace("traces/jax_trace_4x3x25x25.json")

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(torch.from_numpy(np.array(out_jax)), conv_ref, atol=1e-1))
