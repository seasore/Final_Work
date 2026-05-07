# -*- coding: utf-8 -*-
"""
设备检测工具
自动检测 GPU 是否真正可用（包含 CUDA kernel 架构兼容性测试）。
对于 RTX 5080 (Blackwell, sm_120) + 旧版 PyTorch (≤2.6) 的不兼容情况，
自动回退到 CPU 并给出升级提示。
"""

import torch


def get_best_device(verbose: bool = True) -> str:
    """
    返回最优可用设备字符串 ('cuda' 或 'cpu')。
    若 CUDA 被识别但 kernel 运行失败（架构不兼容），自动回退 CPU。
    """
    if not torch.cuda.is_available():
        if verbose:
            print("使用设备: cpu（未检测到 CUDA）")
        return 'cpu'

    gpu_name = torch.cuda.get_device_name(0)

    # 测试是否真的能运行 CUDA kernel
    try:
        _test = torch.zeros(4, device='cuda')
        _ = _test + 1
        torch.cuda.synchronize()
        del _test
        if verbose:
            print(f"使用设备: cuda")
            print(f"  GPU: {gpu_name}")
            mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            print(f"  显存: {mem_gb} GB  |  CUDA: {torch.version.cuda}")
        return 'cuda'

    except RuntimeError as e:
        if 'no kernel image' in str(e) or 'not compatible' in str(e).lower():
            cap = torch.cuda.get_device_capability(0)
            sm = f"sm_{cap[0]}{cap[1]}"
            if verbose:
                print(f"\n{'='*60}")
                print(f"警告: GPU ({gpu_name}) 架构 {sm} 与当前 PyTorch 不兼容！")
                print(f"  当前 PyTorch: {torch.__version__}")
                print(f"  RTX 5080 需要 PyTorch 2.7+ 配合 CUDA 12.8 支持 {sm}")
                print(f"\n  升级命令（运行后重新训练）：")
                print(f"    pip uninstall torch torchvision torchaudio -y")
                print(f"    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
                print(f"\n  当前已自动回退到 CPU 运行（速度较慢，功能完整）")
                print(f"{'='*60}\n")
            return 'cpu'
        else:
            if verbose:
                print(f"GPU 初始化失败: {e}\n已回退到 CPU")
            return 'cpu'
