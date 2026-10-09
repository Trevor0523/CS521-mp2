import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

if __name__ == "__main__":
    torch.manual_seed(0)

    # Instantiate your PyTorch model
    # N, C, H, W = 2, 3, 19, 19
    N, C, H, W = 4, 3, 25, 25
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=C, out_channels=4, kernel_size=6, stride=1, padding=1).cuda().eval()

    # Includes tracing, compilation and first execution; not pure compiler time.
    torch.cuda.synchronize()
    with torch.no_grad():
        compile_start = time.perf_counter()
        scripted_model = torch.compile(model, backend="inductor")
        scripted_model(x)
        torch.cuda.synchronize()
        print(f"Inductor --- compile + first execution: {time.perf_counter() - compile_start:.6f} s")

        # Warm up outside the recorded region.
        for _ in range(5):
            scripted_model(x)
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for i in range(3):
                with record_function(f"inductor_conv_iter_{i}"):
                    out = scripted_model(x)
                    torch.cuda.synchronize()

    prof.export_chrome_trace("traces/inductor_trace_4x3x25x25.json")
    
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
