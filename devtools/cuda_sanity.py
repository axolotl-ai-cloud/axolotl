import json, time
import torch

info = {
    "torch": torch.__version__,
    "cuda": getattr(torch.version, "cuda", None),
    "is_available": torch.cuda.is_available(),
    "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
}
print(json.dumps(info))

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")

# Simple matmul to exercise CUDA
x = torch.randn(2048, 2048, device="cuda")
start = time.time()
y = x @ x.t()
torch.cuda.synchronize()
elapsed = time.time() - start
print(json.dumps({"cuda_matmul_ok": list(y.shape), "elapsed_sec": round(elapsed, 4)}))
