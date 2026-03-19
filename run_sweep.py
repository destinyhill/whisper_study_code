#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遍历条件的批量测试脚本
自动从配置矩阵中生成所有测试组合并依次运行
"""

import argparse
import itertools
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# 测试矩阵配置：按需修改
# ─────────────────────────────────────────────

# 要测试的组合矩阵
SWEEP_MATRIX = {
    # ── 后端 ──────────────────────────────────
    # 每一组 (backend, native_isa, onednn_isa) 代表一个有意义的配置
    # 格式: (backend, native_isa, onednn_isa)
    #   native_isa: "default" | "avx2" | "avx512"
    #   onednn_isa: "auto" | "avx2" | "avx512_core" | "avx512_core_vnni" | "avx512_core_bf16" | "avx512_core_amx"
    "backend_isa_combos": [
        # --- native 后端 ---
        ("native", "default", "auto"),          # 原生默认（回退到 SSE 等基础 SIMD）
        ("native", "avx2",    "auto"),          # 强制 AVX2
        ("native", "avx512",  "auto"),          # 强制 AVX512

        # --- mkldnn 后端 ---
        ("mkldnn", "default", "auto"),           # mkldnn 默认（native用基础SIMD）
        ("mkldnn", "default", "avx512_core_vnni"),  # native基础SIMD，oneDNN限VNNI
        ("mkldnn", "avx2",    "avx2"),           # 均限 AVX2
        ("mkldnn", "avx512",  "avx512_core"),    # 均限 AVX512_core
        ("mkldnn", "avx512",  "avx512_core_vnni"),   # VNNI 加速
        ("mkldnn", "avx512",  "avx512_core_bf16"),   # BF16 加速
        ("mkldnn", "avx512",  "avx512_core_amx"),    # AMX 加速（Sapphire Rapids）
    ],

    # ── 线程数 ────────────────────────────────
    # 填入要测试的线程数列表；None 表示不指定（系统默认）
    "threads": [None, 4, 8],

    # ── interop 线程数 ────────────────────────
    # 控制算子间并行度；None 表示不指定（系统默认）
    "interop_threads": [None, 2, 4],

    # ── 模型 ──────────────────────────────────
    "models": ["small"],

    # ── 音频文件 ──────────────────────────────
    "audio_files": ["zh_long.wav"],

    # ── 语言 ──────────────────────────────────
    "language": "zh",

    # ── 任务 ──────────────────────────────────
    "task": "transcribe",

    # ── 测试范围 ──────────────────────────────
    "sections": "encoder,decoder,full",

    # ── decoder 前缀 token 长度 ──────────────
    "decoder_tokens": [16],

    # ── without_timestamps ────────────────────
    "without_timestamps": False,

    # ── warmup / repeat ───────────────────────
    "warmup": 3,
    "repeat": 5,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="批量遍历条件的 Whisper CPU benchmark runner"
    )
    p.add_argument(
        "--audio-dir",
        default=".",
        help="音频文件所在目录（与 SWEEP_MATRIX.audio_files 配合使用）",
    )
    p.add_argument(
        "--json-dir",
        default="./sweep_results",
        help="JSON 结果输出目录（默认 ./sweep_results）",
    )
    p.add_argument(
        "--bench-script",
        default="whisper_cpu_bench_allinone.py",
        help="benchmark 主脚本路径",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python 解释器路径（默认当前环境）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不实际运行",
    )
    p.add_argument(
        "--sections",
        default=None,
        help="覆盖 SWEEP_MATRIX 中的 sections 设置",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="覆盖 SWEEP_MATRIX 中的 warmup 设置",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="覆盖 SWEEP_MATRIX 中的 repeat 设置",
    )
    p.add_argument(
        "--skip-failed",
        action="store_true",
        help="某个组合运行失败时跳过继续，而非中止",
    )
    return p.parse_args()


def build_commands(args):
    """根据配置矩阵生成所有测试命令"""
    cfg = SWEEP_MATRIX
    sections = args.sections or cfg["sections"]
    warmup = args.warmup if args.warmup is not None else cfg["warmup"]
    repeat = args.repeat if args.repeat is not None else cfg["repeat"]

    commands = []

    for model, audio_file, (backend, native_isa, onednn_isa), threads, interop_threads, decoder_tokens in itertools.product(
        cfg["models"],
        cfg["audio_files"],
        cfg["backend_isa_combos"],
        cfg["threads"],
        cfg["interop_threads"],
        cfg["decoder_tokens"],
    ):
        audio_path = Path(args.audio_dir) / audio_file

        cmd = [
            args.python,
            args.bench_script,
            "--audio", str(audio_path),
            "--model", model,
            "--backend", backend,
            "--native-isa", native_isa,
            "--onednn-isa", onednn_isa,
            "--sections", sections,
            "--warmup", str(warmup),
            "--repeat", str(repeat),
            "--decoder-tokens", str(decoder_tokens),
            "--json-auto",
            "--json-dir", args.json_dir,
        ]

        if threads is not None:
            cmd += ["--threads", str(threads)]

        if interop_threads is not None:
            cmd += ["--interop-threads", str(interop_threads)]

        if cfg.get("language"):
            cmd += ["--language", cfg["language"]]

        if cfg.get("task") and cfg["task"] != "transcribe":
            cmd += ["--task", cfg["task"]]

        if cfg.get("without_timestamps"):
            cmd += ["--without-timestamps"]

        # 描述标签（用于日志打印）
        label_parts = [
            f"model={model}",
            f"backend={backend}",
            f"native_isa={native_isa}",
            f"onednn_isa={onednn_isa}",
            f"threads={threads if threads is not None else 'default'}",
            f"interop={interop_threads if interop_threads is not None else 'default'}",
            f"dec_tokens={decoder_tokens}",
            f"audio={audio_file}",
        ]
        label = " | ".join(label_parts)

        commands.append((label, cmd))

    return commands


def run_all(commands, dry_run=False, skip_failed=False):
    total = len(commands)
    passed = 0
    failed = 0
    skipped = 0
    failed_labels = []

    print(f"\n{'='*60}")
    print(f"  共 {total} 个测试组合")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for idx, (label, cmd) in enumerate(commands, 1):
        print(f"[{idx:>3}/{total}] {label}")
        print(f"         CMD: {' '.join(cmd)}")

        if dry_run:
            print("         [dry-run] 跳过执行\n")
            skipped += 1
            continue

        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # 保持终端输出可见
                text=True,
                timeout=3600  # 1小时超时保护
            )
        except subprocess.TimeoutExpired:
            print(f"         ⏱️  超时（>1小时）\n")
            failed += 1
            failed_labels.append(f"{label} (timeout)")
            if not skip_failed:
                print("测试中止（使用 --skip-failed 可跳过失败继续运行）")
                break
            continue
        except Exception as e:
            print(f"         💥 异常: {e}\n")
            failed += 1
            failed_labels.append(f"{label} (exception)")
            if not skip_failed:
                print("测试中止（使用 --skip-failed 可跳过失败继续运行）")
                break
            continue
        
        elapsed = time.perf_counter() - t0

        if result.returncode == 0:
            print(f"         ✅ 成功  耗时 {elapsed:.1f}s\n")
            passed += 1
        else:
            print(f"         ❌ 失败  returncode={result.returncode}  耗时 {elapsed:.1f}s\n")
            failed += 1
            failed_labels.append(label)
            if not skip_failed:
                print("测试中止（使用 --skip-failed 可跳过失败继续运行）")
                break

    print(f"\n{'='*60}")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总计: {total}  成功: {passed}  失败: {failed}  跳过: {skipped}")
    print(f"{'='*60}")

    if failed_labels:
        print("\n❌ 失败的组合：")
        for label in failed_labels:
            print(f"   - {label}")

    return failed == 0


def main():
    args = parse_args()

    # 创建输出目录
    if not args.dry_run:
        Path(args.json_dir).mkdir(parents=True, exist_ok=True)

    # 检查主脚本是否存在
    bench_script = Path(args.bench_script)
    if not bench_script.exists():
        print(f"❌ benchmark 脚本不存在: {bench_script}")
        sys.exit(1)

    # 检查音频文件是否存在
    audio_dir = Path(args.audio_dir)
    missing_files = []
    for audio_file in SWEEP_MATRIX["audio_files"]:
        audio_path = audio_dir / audio_file
        if not audio_path.exists():
            missing_files.append(str(audio_path))
    
    if missing_files:
        print("❌ 以下音频文件不存在：")
        for f in missing_files:
            print(f"   - {f}")
        sys.exit(1)

    # 构建所有命令
    commands = build_commands(args)

    if not commands:
        print("❌ 未生成任何测试命令，请检查 SWEEP_MATRIX 配置")
        sys.exit(1)

    # 打印测试矩阵概览
    print("\n📋 测试矩阵概览：")
    print(f"  模型       : {SWEEP_MATRIX['models']}")
    print(f"  音频文件   : {SWEEP_MATRIX['audio_files']}")
    print(f"  后端/ISA组合: {len(SWEEP_MATRIX['backend_isa_combos'])} 种")
    
    # 检查并警告不合理的 ISA 配置
    has_warning = False
    for combo in SWEEP_MATRIX["backend_isa_combos"]:
        backend, native_isa, onednn_isa = combo
        print(f"             - backend={backend:<8} native_isa={native_isa:<8} onednn_isa={onednn_isa}")
        
        # 警告：native 后端设置了非 auto 的 onednn_isa
        if backend == "native" and onednn_isa != "auto":
            if not has_warning:
                print("\n⚠️  配置警告：")
                has_warning = True
            print(f"   - native 后端下 onednn_isa={onednn_isa} 不会生效（已关闭 mkldnn）")
    
    if has_warning:
        print()
    
    print(f"  线程数     : {SWEEP_MATRIX['threads']}")
    print(f"  interop线程: {SWEEP_MATRIX['interop_threads']}")
    print(f"  decoder_tok: {SWEEP_MATRIX['decoder_tokens']}")
    print(f"  no_timestamp: {SWEEP_MATRIX['without_timestamps']}")
    print(f"  sections   : {args.sections or SWEEP_MATRIX['sections']}")
    print(f"  warmup     : {args.warmup or SWEEP_MATRIX['warmup']}")
    print(f"  repeat     : {args.repeat or SWEEP_MATRIX['repeat']}")
    print(f"  输出目录   : {args.json_dir}")
    print(f"  总组合数   : {len(commands)}")

    success = run_all(commands, dry_run=args.dry_run, skip_failed=args.skip_failed)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
