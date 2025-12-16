import time, torch

def enable_tf32():
    # Global policy for float32 matmuls
    torch.set_float32_matmul_precision('high')   # default = 'highest' => FP32 matmuls use strictly FP32, thus cannot get Tensor Core acceleration
    torch.backends.cuda.matmul.allow_tf32 = True # allow TF32 for matmuls
    torch.backends.cudnn.allow_tf32 = True       # allow TF32 for convolutions

def bench_gemm(m=4096, n=4096, k=4096, iters=10):
    a = torch.randn(m, k, device='cuda', dtype=torch.float32)
    b = torch.randn(k, n, device='cuda', dtype=torch.float32)
    # warmup
    for _ in range(5): (a @ b).norm().item()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): (a @ b).norm().item()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000 / iters  # ms

def demo_tf32_speedup():
    iters = 10
    # run once with strict FP32
    torch.set_float32_matmul_precision('highest')
    t_fp32 = bench_gemm(iters=iters)
    # run once with TF32 allowed
    enable_tf32()
    t_tf32 = bench_gemm(iters=iters)
    spd = t_fp32 / t_tf32 if t_tf32 > 0 else float('inf')
    print(f"[GEMM 4096^2] FP32={t_fp32:.1f} ms, TF32={t_tf32:.1f} ms, speedup x{spd:.2f} ({iters} runs each)")


demo_tf32_speedup()
# expected output: [GEMM 4096^2] FP32=2.8 ms, TF32=0.4 ms, speedup x6.81 (10 runs each)
