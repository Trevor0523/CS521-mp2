import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    channels = nl.tile_size.pmax
    n_in = in_channels // channels
    n_out = out_channels // channels
    tile_rows = min(out_height, nl.tile_size.gemm_moving_fmax // out_width)

    strip_budget_rows = max(tile_rows, (4 * 1024 * 1024) //
                            (in_channels * input_width * 4) - filter_height + 1)
    tiles_per_strip = min(8, max(1, strip_budget_rows // tile_rows),
                          (out_height + tile_rows - 1) // tile_rows)
    strip_rows = tile_rows * tiles_per_strip
    load_rows = strip_rows + filter_height - 1
    pixels = tile_rows * out_width

    p = nl.arange(channels)[:, None]
    q = nl.arange(channels)[None, :]
    weights = nl.ndarray((n_out, n_in, filter_height, filter_width,
                          nl.par_dim(channels), channels),
                         dtype=W.dtype, buffer=nl.sbuf)
    biases = nl.ndarray((n_out, nl.par_dim(channels), 1),
                        dtype=nl.float32, buffer=nl.sbuf)
    for oc in nl.affine_range(n_out):
        biases[oc] = nl.load(bias[oc * channels + p])
        for ic in nl.affine_range(n_in):
            raw_w = nl.load(W[oc * channels:(oc + 1) * channels,
                              ic * channels:(ic + 1) * channels, :, :])
            for kh in nl.affine_range(filter_height):
                for kw in nl.affine_range(filter_width):
                    w = raw_w[p, q, kh, kw]
                    weights[oc, ic, kh, kw] = nl.copy(
                        nisa.nc_transpose(w), dtype=W.dtype)

    cp = nl.arange(channels)[:, None, None]
    load_r = nl.arange(load_rows)[None, :, None]
    load_c = nl.arange(input_width)[None, None, :]
    r = nl.arange(tile_rows)[None, :, None]
    c = nl.arange(out_width)[None, None, :]

    for b in nl.sequential_range(batch_size):
        for strip in nl.affine_range((out_height + strip_rows - 1) // strip_rows):
            first_row = strip * strip_rows
            inputs = nl.ndarray((n_in, nl.par_dim(channels), load_rows, input_width),
                                dtype=X.dtype, buffer=nl.sbuf)
            for ic in nl.affine_range(n_in):
                inputs[ic] = nl.load(
                    X[b, ic * channels + cp, first_row + load_r, load_c],
                    mask=first_row + load_r < input_height)

            for oc in nl.affine_range(n_out):
                for tile in nl.affine_range(tiles_per_strip):
                    acc = nl.zeros((channels, pixels), dtype=nl.float32,
                                   buffer=nl.psum)
                    for ic in nl.affine_range(n_in):
                        for kh in nl.affine_range(filter_height):
                            for kw in nl.affine_range(filter_width):
                                patch = nl.copy(inputs[
                                    ic, cp, tile * tile_rows + kh + r, kw + c])
                                acc += nisa.nc_matmul(
                                    weights[oc, ic, kh, kw],
                                    patch.reshape((channels, pixels)))

                    result = nl.add(nl.copy(acc, dtype=nl.float32), biases[oc])
                    output_row = first_row + tile * tile_rows + r
                    nl.store(X_out[b, oc * channels + cp, output_row, c],
                             value=result.reshape((channels, tile_rows, out_width)),
                             mask=output_row < out_height)

    return X_out

