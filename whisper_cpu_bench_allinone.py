#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark official Whisper on CPU with native PyTorch path and/or MKLDNN/oneDNN path."
    )

    # 基本参数
    p.add_argument("--audio", required=True, help="音频文件路径")
    p.add_argument("--model", default="small", help="Whisper 模型名，如 tiny/base/small/medium/large")
    p.add_argument("--language", default=None, help="语言代码，如 zh/en/ja")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])

    # 后端选择
    p.add_argument(
        "--backend",
        default="native",
        choices=["native", "mkldnn"],
        help="native=关闭mkldnn走原生路径；mkldnn=开启mkldnn",
    )

    # 原生 PyTorch CPU ISA
    p.add_argument(
        "--native-isa",
        default="auto",
        choices=["auto", "default", "avx2", "avx512"],
        help="限制 PyTorch 原生 ATen CPU capability（ATEN_CPU_CAPABILITY）",
    )

    # oneDNN ISA
    p.add_argument(
        "--onednn-isa",
        default="auto",
        choices=[
            "auto",
            "avx2",
            "avx512_core",
            "avx512_core_vnni",
            "avx512_core_bf16",
            "avx512_core_amx",
        ],
        help="限制 oneDNN 最高 ISA；auto 表示不限制",
    )

    p.add_argument(
        "--mkldnn-verbose",
        action="store_true",
        help="开启 oneDNN verbose",
    )

    # 线程
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads()")
    p.add_argument("--interop-threads", type=int, default=None, help="torch.set_num_interop_threads()")

    # benchmark 控制
    p.add_argument(
        "--sections",
        default="encoder,decoder,full",
        help="要 benchmark 的部分：encoder,decoder,full，逗号分隔",
    )
    p.add_argument("--warmup", type=int, default=3, help="正式计时前 warmup 次数")
    p.add_argument("--repeat", type=int, default=5, help="正式计时次数")
    p.add_argument("--decoder-tokens", type=int, default=16, help="decoder_step 的前缀 token 长度")
    p.add_argument(
        "--without-timestamps",
        action="store_true",
        help="full transcribe 时使用 without_timestamps=True",
    )
    p.add_argument("--json-out", default=None, help="导出 benchmark JSON 路径")

    # profiler 控制
    p.add_argument(
        "--profile-sections",
        default="",
        help="要做 profiler 的部分：encoder,decoder,full，逗号分隔；空表示不做",
    )
    p.add_argument("--profile-topk", type=int, default=30, help="profiler 表显示前多少行")
    p.add_argument(
        "--profile-sort-by",
        default="self_cpu_time_total",
        help="profiler 排序字段，如 self_cpu_time_total / cpu_time_total",
    )
    p.add_argument("--profile-record-shapes", action="store_true", help="记录输入 shape")
    p.add_argument(
        "--profile-group-by-input-shape",
        action="store_true",
        help="profiler 按 op + input shape 分组",
    )
    p.add_argument("--profile-memory", action="store_true", help="记录 profiler memory 信息")
    p.add_argument("--profile-with-stack", action="store_true", help="记录 Python stack（开销更大）")
    p.add_argument("--profile-txt-dir", default=None, help="profiler 表格文本输出目录")
    p.add_argument("--profile-trace-dir", default=None, help="Chrome trace 输出目录")

    return p.parse_args()


args = parse_args()

# ---------------------------
# 必须在 import torch 之前设置
# ---------------------------

# 原生 PyTorch CPU ISA
if args.native_isa != "auto":
    os.environ["ATEN_CPU_CAPABILITY"] = args.native_isa

# oneDNN ISA
if args.onednn_isa != "auto":
    isa = args.onednn_isa.upper()
    os.environ["ONEDNN_MAX_CPU_ISA"] = isa
    # 兼容部分环境的旧变量名
    os.environ["DNNL_MAX_CPU_ISA"] = isa

# 线程环境变量
if args.threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

if args.warmup < 0:
    raise ValueError("warmup must be >= 0")
if args.repeat <= 0:
    raise ValueError("repeat must be > 0")
if args.decoder_tokens <= 0:
    raise ValueError("decoder-tokens must be > 0")

import torch  # noqa: E402
import whisper  # noqa: E402
from torch.profiler import profile, ProfilerActivity, record_function  # noqa: E402
from whisper.audio import SAMPLE_RATE  # noqa: E402
from whisper.tokenizer import get_tokenizer  # noqa: E402


def setup_runtime():
    if args.backend == "native":
        torch.backends.mkldnn.enabled = False
    elif args.backend == "mkldnn":
        torch.backends.mkldnn.enabled = True

    if args.backend == "native" and args.onednn_isa != "auto":
        print("note                  : onednn isa is set but mkldnn backend is disabled in this run")

    torch.set_grad_enabled(False)

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    interop_set_status = "unused"
    if args.interop_threads is not None:
        try:
            torch.set_num_interop_threads(args.interop_threads)
            interop_set_status = "ok"
        except RuntimeError:
            interop_set_status = "failed(already initialized)"
    print(f"interop set status     : {interop_set_status}")
    return interop_set_status


def maybe_mkldnn_verbose():
    if torch.backends.mkldnn.enabled and args.mkldnn_verbose and hasattr(torch.backends.mkldnn, "verbose"):
        return torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON)
    return contextlib.nullcontext()


def mean_ms(xs):
    return statistics.mean(xs) * 1000.0


def median_ms(xs):
    return statistics.median(xs) * 1000.0


def min_ms(xs):
    return min(xs) * 1000.0


def max_ms(xs):
    return max(xs) * 1000.0


def stdev_ms(xs):
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs) * 1000.0


def percentile_ms(xs, p):
    ys = sorted(xs)
    if not ys:
        return 0.0
    if len(ys) == 1:
        return ys[0] * 1000.0
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f] * 1000.0
    return (ys[f] + (ys[c] - ys[f]) * (k - f)) * 1000.0


def maybe_mkdir(path_str):
    if not path_str:
        return None
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def print_section_summary(name, times, extra=None):
    print(f"\n[{name}]")
    print(f"  repeat              : {len(times)}")
    print(f"  mean (ms)           : {mean_ms(times):.3f}")
    print(f"  median (ms)         : {median_ms(times):.3f}")
    print(f"  min (ms)            : {min_ms(times):.3f}")
    print(f"  max (ms)            : {max_ms(times):.3f}")
    print(f"  p90 (ms)            : {percentile_ms(times, 0.90):.3f}")
    print(f"  p95 (ms)            : {percentile_ms(times, 0.95):.3f}")
    print(f"  stdev (ms)          : {stdev_ms(times):.3f}")
    if extra:
        for k, v in extra.items():
            print(f"  {k:<20}: {v}")


def time_fn(fn, warmup, repeat):
    last = None

    for i in range(warmup):
        with torch.inference_mode():
            # 仅最后一次 warmup 开启 verbose，避免刷屏
            if i == warmup - 1:
                with maybe_mkldnn_verbose():
                    last = fn()
            else:
                last = fn()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with torch.inference_mode():
            last = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times, last


def run_profile_once(section_name, fn):
    activities = [ProfilerActivity.CPU]

    with profile(
        activities=activities,
        record_shapes=args.profile_record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.profile_with_stack,
    ) as prof:
        with record_function(f"whisper_{section_name}"):
            with maybe_mkldnn_verbose():
                with torch.inference_mode():
                    out = fn()

    events = prof.key_averages(
        group_by_input_shape=args.profile_group_by_input_shape,
        group_by_stack_n=5 if args.profile_with_stack else 0,
    )
    table = events.table(
        sort_by=args.profile_sort_by,
        row_limit=args.profile_topk,
    )

    print(f"\n[profiler::{section_name}]")
    print(table)

    txt_dir = maybe_mkdir(args.profile_txt_dir)
    if txt_dir is not None:
        txt_path = txt_dir / f"{sanitize_name(section_name)}.txt"
        txt_path.write_text(table, encoding="utf-8")
        print(f"profiler table saved: {txt_path}")

    trace_dir = maybe_mkdir(args.profile_trace_dir)
    if trace_dir is not None:
        trace_path = trace_dir / f"{sanitize_name(section_name)}.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"chrome trace saved : {trace_path}")

    return out, table


def build_decoder_prefix(tokenizer, length: int):
    prefix = list(getattr(tokenizer, "sot_sequence", []))
    if not prefix:
        prefix = [getattr(tokenizer, "sot", 0)]

    if args.without_timestamps:
        no_ts = getattr(tokenizer, "no_timestamps", None)
        if no_ts is not None and no_ts not in prefix:
            prefix.append(no_ts)

    seed_text = " this is a whisper cpu benchmark sample"
    try:
        extra_ids = tokenizer.encode(seed_text)
    except Exception:
        extra_ids = []

    if not extra_ids:
        fallback = getattr(tokenizer, "eot", prefix[-1])
        extra_ids = [fallback]

    out = prefix[:]
    while len(out) < length:
        need = length - len(out)
        out.extend(extra_ids[:need] if need <= len(extra_ids) else extra_ids)

    return out[:length]


def main():
    interop_set_status = setup_runtime()

    sections = {x.strip() for x in args.sections.split(",") if x.strip()}
    profile_sections = {x.strip() for x in args.profile_sections.split(",") if x.strip()}
    valid_sections = {"encoder", "decoder", "full"}
    invalid_sections = sections - valid_sections
    if invalid_sections:
        raise ValueError(f"invalid sections: {sorted(invalid_sections)}")

    invalid_profile_sections = profile_sections - valid_sections
    if invalid_profile_sections:
        raise ValueError(f"invalid profile sections: {sorted(invalid_profile_sections)}")

    if not sections:
        raise ValueError("at least one section must be selected")

    if not profile_sections.issubset(sections):
        raise ValueError(
            f"profile sections must be a subset of sections: {sorted(profile_sections - sections)}"
        )
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")

    print("=== Environment ===")
    print(f"audio path            : {audio_path}")
    print(f"model                 : {args.model}")
    print(f"language              : {args.language}")
    print(f"task                  : {args.task}")
    print(f"backend requested     : {args.backend}")
    print(f"native isa requested  : {args.native_isa}")
    print(f"ATEN_CPU_CAPABILITY   : {os.environ.get('ATEN_CPU_CAPABILITY', '<unset>')}")
    print(f"mkldnn available      : {torch.backends.mkldnn.is_available()}")
    print(f"mkldnn enabled        : {torch.backends.mkldnn.enabled}")
    print(f"onednn isa requested  : {args.onednn_isa}")
    print(f"ONEDNN_MAX_CPU_ISA    : {os.environ.get('ONEDNN_MAX_CPU_ISA', '<unset>')}")
    print(f"DNNL_MAX_CPU_ISA      : {os.environ.get('DNNL_MAX_CPU_ISA', '<unset>')}")
    print(f"mkldnn verbose        : {args.mkldnn_verbose}")
    print(f"cpu capability        : {torch.backends.cpu.get_cpu_capability()}")
    print(f"torch num threads     : {torch.get_num_threads()}")
    print(f"torch interop threads : {torch.get_num_interop_threads()}")
    print(f"warmup/repeat         : {args.warmup}/{args.repeat}")
    print(f"profile sections      : {','.join(sorted(profile_sections)) if profile_sections else '<disabled>'}")

    model = whisper.load_model(args.model, device="cpu")
    model.eval()

    audio = whisper.load_audio(str(audio_path))
    audio_duration_sec = max(len(audio) / SAMPLE_RATE, 1e-6)

    # encoder benchmark 使用固定 30s chunk
    audio_30s = whisper.pad_or_trim(audio)
    mel_30s = whisper.log_mel_spectrogram(audio_30s, n_mels=model.dims.n_mels).unsqueeze(0).cpu()

    tokenizer = get_tokenizer(model.is_multilingual, language=args.language, task=args.task)
    prefix_ids = build_decoder_prefix(tokenizer, args.decoder_tokens)
    decoder_tokens = torch.tensor([prefix_ids], dtype=torch.long, device="cpu")

    results = {
        "env": {
            "audio": str(audio_path),
            "audio_duration_sec": audio_duration_sec,
            "model": args.model,
            "language": args.language,
            "task": args.task,
            "backend_requested": args.backend,
            "native_isa_requested": args.native_isa,
            "env_ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
            "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "onednn_isa_requested": args.onednn_isa,
            "env_ONEDNN_MAX_CPU_ISA": os.environ.get("ONEDNN_MAX_CPU_ISA"),
            "env_DNNL_MAX_CPU_ISA": os.environ.get("DNNL_MAX_CPU_ISA"),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "warmup": args.warmup,
            "repeat": args.repeat,
            "decoder_tokens": args.decoder_tokens,
            "profile_sections": sorted(profile_sections),
            "profile_sort_by": args.profile_sort_by,
            "interop_set_status": interop_set_status,
        },
        "bench": {},
    }

    # ---------------- encoder ----------------
    if "encoder" in sections:
        def run_encoder():
            return model.embed_audio(mel_30s)

        encoder_times, encoder_out = time_fn(run_encoder, args.warmup, args.repeat)
        enc_shape = tuple(encoder_out.shape)

        print_section_summary(
            "encoder_30s_chunk",
            encoder_times,
            extra={
                "output_shape": enc_shape,
                "chunk_sec": 30.0,
                "rtf(mean)": f"{statistics.mean(encoder_times) / 30.0:.6f}",
                "rtf(median)": f"{statistics.median(encoder_times) / 30.0:.6f}",
            },
        )

        results["bench"]["encoder_30s_chunk"] = {
            "times_sec": encoder_times,
            "output_shape": enc_shape,
            "chunk_sec": 30.0,
            "rtf_mean": statistics.mean(encoder_times) / 30.0,
            "rtf_median": statistics.median(encoder_times) / 30.0,
        }

        if "encoder" in profile_sections:
            _, table = run_profile_once("encoder_30s_chunk", run_encoder)
            results["bench"]["encoder_30s_chunk"]["profiler_table"] = table

    # ---------------- decoder ----------------
    if "decoder" in sections:
        with torch.inference_mode():
            audio_features = model.embed_audio(mel_30s)

        def run_decoder_step():
            logits = model.logits(decoder_tokens, audio_features)
            next_token = logits[:, -1, :].argmax(dim=-1)
            return logits, next_token

        decoder_times, decoder_out = time_fn(run_decoder_step, args.warmup, args.repeat)
        logits, next_token = decoder_out

        print_section_summary(
            "decoder_step",
            decoder_times,
            extra={
                "prefix_len": decoder_tokens.shape[1],
                "logits_shape": tuple(logits.shape),
                "steps_per_sec(mean)": f"{1.0 / statistics.mean(decoder_times):.3f}",
                "next_token": next_token.tolist(),
            },
        )

        results["bench"]["decoder_step"] = {
            "times_sec": decoder_times,
            "prefix_len": int(decoder_tokens.shape[1]),
            "logits_shape": tuple(logits.shape),
            "steps_per_sec_mean": 1.0 / statistics.mean(decoder_times),
            "next_token": next_token.tolist(),
        }

        if "decoder" in profile_sections:
            _, table = run_profile_once("decoder_step", run_decoder_step)
            results["bench"]["decoder_step"]["profiler_table"] = table

    # ---------------- full transcribe ----------------
    if "full" in sections:
        transcribe_kwargs = {
            "task": args.task,
            "fp16": False,
            "verbose": False,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        }
        if args.language is not None:
            transcribe_kwargs["language"] = args.language
        if args.without_timestamps:
            transcribe_kwargs["without_timestamps"] = True

        def run_full():
            return model.transcribe(audio, **transcribe_kwargs)

        full_times, full_out = time_fn(run_full, args.warmup, args.repeat)
        text = (full_out.get("text", "") or "").strip()

        print_section_summary(
            "full_transcribe",
            full_times,
            extra={
                "audio_sec": f"{audio_duration_sec:.3f}",
                "rtf(mean)": f"{statistics.mean(full_times) / audio_duration_sec:.6f}",
                "rtf(median)": f"{statistics.median(full_times) / audio_duration_sec:.6f}",
                "text_preview": text[:120].replace("\n", " "),
            },
        )

        results["bench"]["full_transcribe"] = {
            "times_sec": full_times,
            "audio_duration_sec": audio_duration_sec,
            "rtf_mean": statistics.mean(full_times) / audio_duration_sec,
            "rtf_median": statistics.median(full_times) / audio_duration_sec,
            "text_preview": text[:500],
            "text_len": len(text),
            "num_segments": len(full_out.get("segments", [])),
        }

        if "full" in profile_sections:
            _, table = run_profile_once("full_transcribe", run_full)
            results["bench"]["full_transcribe"]["profiler_table"] = table

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON saved to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
