# -*- coding: utf-8 -*-
import torch
import sys

print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print("=" * 50)

cuda_ok = torch.cuda.is_available()
print(f"CUDA 可用: {cuda_ok}")

if cuda_ok:
    print(f"GPU 名称:   {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    total_mem = round(props.total_memory / 1024**3, 1)
    print(f"显存总量:   {total_mem} GB")
    print(f"CUDA 版本:  {torch.version.cuda}")
    print(f"设备数量:   {torch.cuda.device_count()}")

    # 做一次简单的张量运算测试
    print("\n运行 GPU 张量测试...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print(f"  矩阵乘法结果 shape: {y.shape} -- GPU 工作正常！")
else:
    print("\n未检测到 CUDA，常见原因：")
    print("  1. 安装的是 CPU 版 PyTorch（最常见）")
    print("  2. NVIDIA 驱动版本过旧")
    print("  3. CUDA Toolkit 未安装或版本不匹配")
    print()
    print("修复方法（RTX 5080 建议用 CUDA 12.4）：")
    print("  pip uninstall torch torchvision torchaudio -y")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print()
    # 检查驱动
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # 提取驱动版本
            for line in result.stdout.split('\n'):
                if 'Driver Version' in line or 'CUDA Version' in line:
                    print(f"nvidia-smi 信息: {line.strip()}")
        else:
            print("nvidia-smi 运行失败，请确认已安装 NVIDIA 驱动")
    except Exception:
        print("无法运行 nvidia-smi")
