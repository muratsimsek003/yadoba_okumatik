#!/usr/bin/env python3
"""
check_gpu.py — kurulumdan ÖNCE çalıştır. CUDA + VRAM + kütüphaneleri doğrular
ve minik bir GPU forward/backward ile her şeyin çalıştığını kanıtlar.

  python check_gpu.py
"""
def main():
    import importlib
    print("== Kütüphaneler ==")
    for pkg in ["torch", "transformers", "datasets", "evaluate", "jiwer", "accelerate", "peft"]:
        try:
            m = importlib.import_module(pkg)
            print(f"  ✓ {pkg:14s} {getattr(m,'__version__','?')}")
        except Exception as e:
            print(f"  ✗ {pkg:14s} EKSİK ({e})")

    import torch
    print("\n== GPU ==")
    print(f"  torch            : {torch.__version__}")
    print(f"  CUDA derlemesi   : {torch.version.cuda}")
    print(f"  cuda.is_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n[HATA] CUDA görünmüyor. Çözüm:")
        print("  pip uninstall -y torch")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU              : {p.name}")
    print(f"  VRAM             : {p.total_memory/1e9:.1f} GB")
    print(f"  bf16 destekli    : {torch.cuda.is_bf16_supported()}")

    print("\n== Minik GPU testi (matmul + backward) ==")
    x = torch.randn(2048, 2048, device="cuda", requires_grad=True)
    y = (x @ x).sum()
    y.backward()
    torch.cuda.synchronize()
    used = torch.cuda.max_memory_allocated() / 1e9
    print(f"  ✓ Başarılı. Tepe VRAM kullanımı: {used:.2f} GB")
    print("\nHazırsınız. Sonraki adım:")
    print("  python finetune_whisper_tr.py --baseline_only")


if __name__ == "__main__":
    main()
