# Whisper CPU Benchmark 工具

这是一个用于在 CPU 上对 OpenAI Whisper 模型进行性能基准测试的工具，支持原生 PyTorch 和 MKLDNN/oneDNN 后端性能对比测试。

## 功能特性

- ✅ **多后端支持**：原生 PyTorch 或 MKLDNN/oneDNN 加速后端
- ✅ **ISA 指令集控制**：精确控制 AVX2、AVX512 等指令集使用
- ✅ **分段性能测试**：可单独测试 encoder、decoder 或完整转录流程
- ✅ **详细性能统计**：均值、中位数、P90/P95、标准差等指标
- ✅ **PyTorch Profiler 集成**：深入分析性能瓶颈
- ✅ **智能结果导出**：自动生成包含配置信息的 JSON 文件名，批量测试更便捷
- ✅ **批量测试友好**：适合大规模性能对比和自动化测试

## 环境要求

### 必需依赖
- Python 3.8+
- PyTorch (建议 1.13+ 以获得更好的 oneDNN 支持)
- openai-whisper

### 安装步骤

```bash
# 安装 PyTorch (CPU 版本)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 安装 Whisper
pip install openai-whisper

# 或者从源码安装最新版
pip install git+https://github.com/openai/whisper.git
```

## 快速开始

### 基本使用

```bash
# 使用默认参数测试（small 模型，原生后端）
python whisper_cpu_bench_allinone.py --audio test.wav

# 指定模型大小
python whisper_cpu_bench_allinone.py --audio test.wav --model base

# 开启 MKLDNN 加速
python whisper_cpu_bench_allinone.py --audio test.wav --backend mkldnn

# 保存结果到 JSON（自动生成文件名）
python whisper_cpu_bench_allinone.py --audio test.wav --json-auto
```

### 完整示例

```bash
# 完整配置示例：测试中文音频，使用 small 模型，开启 MKLDNN，限制 AVX512
python whisper_cpu_bench_allinone.py \
  --audio chinese_sample.wav \
  --model small \
  --language zh \
  --backend mkldnn \
  --onednn-isa avx512_core_vnni \
  --threads 8 \
  --warmup 5 \
  --repeat 10 \
  --json-auto \
  --json-dir ./results

# 输出示例：
# ./results/whisper_bench_model_small_backend_mkldnn_onednn_avx512_core_vnni_t8_audio_chinese_sample_20260316_143052.json
```

## 参数详解

### 基本参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--audio` | ✅ | - | 输入音频文件路径（支持 WAV、MP3 等格式） |
| `--model` | ❌ | `small` | Whisper 模型大小：`tiny`/`base`/`small`/`medium`/`large` |
| `--language` | ❌ | `None` | 语言代码（如 `zh`/`en`/`ja`），不指定则自动检测 |
| `--task` | ❌ | `transcribe` | 任务类型：`transcribe`（转录）或 `translate`（翻译为英文） |

### 后端与指令集控制

| 参数 | 默认值 | 选项 | 说明 |
|------|--------|------|------|
| `--backend` | `native` | `native`/`mkldnn` | `native`=关闭 MKLDNN；`mkldnn`=开启 MKLDNN/oneDNN 加速 |
| `--native-isa` | `auto` | `auto`/`default`/`avx2`/`avx512` | 限制原生 PyTorch CPU 指令集（`ATEN_CPU_CAPABILITY`） |
| `--onednn-isa` | `auto` | `auto`/`avx2`/`avx512_core`/`avx512_core_vnni`/`avx512_core_bf16`/`avx512_core_amx` | 限制 oneDNN 最高指令集 |
| `--mkldnn-verbose` | `False` | - | 开启 oneDNN 详细日志（调试用） |

**ISA 选择建议：**
- **Intel 10th Gen+ (Ice Lake/Cascade Lake)**：`avx512_core_vnni`
- **Intel 11th Gen+ (Tiger Lake/Rocket Lake)**：`avx512_core_vnni` 或 `avx512_core_bf16`
- **Intel 12th Gen+ (Alder Lake) / 4th Gen Xeon (Sapphire Rapids)**：`avx512_core_amx`（支持 AMX 指令）
- **AMD Zen 3/4**：`avx2`（AMD 无 AVX512）

### 线程控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--threads` | 系统默认 | 设置线程数（影响 `torch.set_num_threads()` 和 `OMP_NUM_THREADS` 等） |
| `--interop-threads` | 系统默认 | 设置线程池间并行度（`torch.set_num_interop_threads()`） |

**线程数建议：**
- **物理核心数**：通常是最优选择（如 8 核 CPU 用 `--threads 8`）
- **避免超线程数**：使用逻辑核心数（如 16 线程）通常性能更差
- **测试对比**：不同线程数下性能差异可能很大，建议对比测试

### Benchmark 控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sections` | `encoder,decoder,full` | 要测试的部分（逗号分隔）：<br>• `encoder`：仅测试 mel-spectrogram → audio features<br>• `decoder`：仅测试单步 decoder 推理<br>• `full`：完整端到端转录 |
| `--warmup` | `3` | 预热次数（正式计时前执行，避免冷启动影响） |
| `--repeat` | `5` | 正式计时重复次数 |
| `--decoder-tokens` | `16` | decoder 测试时的前缀 token 长度 |
| `--without-timestamps` | `False` | 完整转录时禁用时间戳预测（可能更快） |
| `--json-out` | `None` | 手动指定 JSON 结果文件路径 |
| `--json-auto` | `False` | 🌟 自动生成 JSON 文件名（包含模型、后端、ISA、线程数等信息） |
| `--json-dir` | `.` | 配合 `--json-auto` 使用，指定输出目录（默认当前目录） |

### Profiler 控制（性能分析）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--profile-sections` | `""` | 要进行 profiler 分析的部分（逗号分隔），空字符串=不分析 |
| `--profile-topk` | `30` | Profiler 表格显示前 K 个操作 |
| `--profile-sort-by` | `self_cpu_time_total` | 排序字段（`self_cpu_time_total`/`cpu_time_total`/`cpu_time` 等） |
| `--profile-record-shapes` | `False` | 记录输入 tensor shape |
| `--profile-group-by-input-shape` | `False` | 按操作 + input shape 分组 |
| `--profile-memory` | `False` | 记录内存使用信息（有额外开销） |
| `--profile-with-stack` | `False` | 记录 Python 调用栈（开销更大，但更详细） |
| `--profile-txt-dir` | `None` | Profiler 表格文本输出目录 |
| `--profile-trace-dir` | `None` | Chrome trace JSON 输出目录（可用 chrome://tracing 查看） |

## 使用示例

### 示例 1：快速性能测试

```bash
# 测试不同模型大小
python whisper_cpu_bench_allinone.py --audio test.wav --model tiny
python whisper_cpu_bench_allinone.py --audio test.wav --model base
python whisper_cpu_bench_allinone.py --audio test.wav --model small
```

### 示例 2：原生 vs MKLDNN 对比（自动生成文件名）

```bash
# 原生 PyTorch（AVX2）- 自动生成文件名
python whisper_cpu_bench_allinone.py \
  --audio test.wav \
  --model small \
  --backend native \
  --native-isa avx2 \
  --json-auto \
  --json-dir ./results
# 输出：./results/whisper_bench_model_small_backend_native_native_avx2_audio_test_20260316_143052.json

# MKLDNN/oneDNN（AVX512 VNNI）- 自动生成文件名
python whisper_cpu_bench_allinone.py \
  --audio test.wav \
  --model small \
  --backend mkldnn \
  --onednn-isa avx512_core_vnni \
  --json-auto \
  --json-dir ./results
# 输出：./results/whisper_bench_model_small_backend_mkldnn_onednn_avx512_core_vnni_audio_test_20260316_143053.json
```

### 示例 3：线程数调优（批量测试）

```bash
# Linux/macOS - 自动生成文件名包含线程数信息
for threads in 1 2 4 8 16; do
  python whisper_cpu_bench_allinone.py \
    --audio test.wav \
    --model small \
    --backend mkldnn \
    --threads $threads \
    --json-auto \
    --json-dir ./thread_test
done
# 输出文件示例：
# whisper_bench_model_small_backend_mkldnn_t1_audio_test_20260316_143052.json
# whisper_bench_model_small_backend_mkldnn_t2_audio_test_20260316_143053.json
# ...

# Windows PowerShell
foreach ($t in @(1,2,4,8,16)) {
  python whisper_cpu_bench_allinone.py `
    --audio test.wav `
    --model small `
    --backend mkldnn `
    --threads $t `
    --json-auto `
    --json-dir ./thread_test
}
```

### 示例 4：仅测试 encoder（最快）

```bash
# 只测试 encoder，适合快速对比不同配置
python whisper_cpu_bench_allinone.py \
  --audio test.wav \
  --model small \
  --sections encoder \
  --warmup 10 \
  --repeat 50
```

### 示例 5：深度性能分析

```bash
# 开启 profiler，输出详细表格和 Chrome trace
python whisper_cpu_bench_allinone.py \
  --audio test.wav \
  --model small \
  --backend mkldnn \
  --sections full \
  --profile-sections full \
  --profile-topk 50 \
  --profile-record-shapes \
  --profile-memory \
  --profile-txt-dir ./profiler_tables \
  --profile-trace-dir ./profiler_traces

# 然后在 Chrome 浏览器打开 chrome://tracing，加载 ./profiler_traces/full_transcribe.json
```

### 示例 6：多语言支持

```bash
# 中文
python whisper_cpu_bench_allinone.py --audio chinese.wav --language zh --model medium

# 日语
python whisper_cpu_bench_allinone.py --audio japanese.wav --language ja --model small

# 英语（带翻译）
python whisper_cpu_bench_allinone.py --audio spanish.wav --task translate --model base
```

### 示例 7：批量测试所有组合（自动化）

```bash
#!/bin/bash
# 测试多个模型 + 后端组合，自动生成文件名

AUDIO="test.wav"
MODELS="tiny base small"
BACKENDS="native mkldnn"
OUTPUT_DIR="./benchmark_results"

for model in $MODELS; do
  for backend in $BACKENDS; do
    echo "Testing: $model with $backend"
    python whisper_cpu_bench_allinone.py \
      --audio $AUDIO \
      --model $model \
      --backend $backend \
      --json-auto \
      --json-dir $OUTPUT_DIR
  done
done

echo "All results saved to: $OUTPUT_DIR"
echo "Files generated:"
ls -1 $OUTPUT_DIR
```

**Windows PowerShell 版本：**

```powershell
# 批量测试脚本
$audio = "test.wav"
$models = @("tiny", "base", "small")
$backends = @("native", "mkldnn")
$outputDir = "./benchmark_results"

foreach ($model in $models) {
    foreach ($backend in $backends) {
        Write-Host "Testing: $model with $backend"
        python whisper_cpu_bench_allinone.py `
            --audio $audio `
            --model $model `
            --backend $backend `
            --json-auto `
            --json-dir $outputDir
    }
}

Write-Host "All results saved to: $outputDir"
Get-ChildItem $outputDir -Name
```

## 输出说明

### 终端输出

```
=== Environment ===
audio path            : test.wav
model                 : small
backend requested     : mkldnn
mkldnn enabled        : True
torch num threads     : 8
...

[encoder_30s_chunk]
  repeat              : 5
  mean (ms)           : 1234.567
  median (ms)         : 1230.123
  min (ms)            : 1210.456
  max (ms)            : 1256.789
  p90 (ms)            : 1245.678
  p95 (ms)            : 1250.234
  stdev (ms)          : 15.678
  output_shape        : (1, 1500, 512)
  chunk_sec           : 30.0
  rtf(mean)           : 0.041152  # Real-Time Factor：值越小越快
  rtf(median)         : 0.041004

[decoder_step]
  ...

[full_transcribe]
  ...
  text_preview        : 这是一段测试音频的转录结果...
```

### JSON 输出

#### 方式 1：自动生成文件名（推荐用于批量测试）

使用 `--json-auto` 自动生成包含配置信息的文件名：

```bash
python whisper_cpu_bench_allinone.py \
  --audio test.wav \
  --model small \
  --backend mkldnn \
  --onednn-isa avx512_core_vnni \
  --threads 8 \
  --json-auto \
  --json-dir ./results
```

**自动生成的文件名格式：**
```
whisper_bench_model_{模型}_backend_{后端}_[native_{native_isa}]_[onednn_{onednn_isa}]_[t{线程数}]_audio_{音频名}_{时间戳}.json
```

**说明：**
- `native_{native_isa}` - 仅当设置 `--native-isa` 时出现
- `onednn_{onednn_isa}` - 仅当设置 `--onednn-isa` 时出现
- 两个 ISA 可以同时设置，会同时出现在文件名中

**示例：**
```bash
# 仅设置 oneDNN ISA
whisper_bench_model_small_backend_mkldnn_onednn_avx512_core_vnni_t8_audio_test_20260316_143052.json

# 仅设置 native ISA
whisper_bench_model_small_backend_native_native_avx2_t8_audio_test_20260316_143052.json

# 同时设置两个 ISA（高级用法）
whisper_bench_model_small_backend_mkldnn_native_avx2_onednn_avx512_core_vnni_t8_audio_test_20260316_143052.json
```

**优势：**
- 📁 **批量测试友好**：无需手动管理文件名
- 📝 **信息丰富**：文件名包含所有关键配置
- ⏰ **避免覆盖**：时间戳确保唯一性
- 🔍 **易于查找**：可按模型/后端/ISA 筛选

#### 方式 2：手动指定文件名

使用 `--json-out results.json` 手动指定文件路径：

```json
{
  "env": {
    "audio": "test.wav",
    "model": "small",
    "backend_requested": "mkldnn",
    "mkldnn_enabled": true,
    "torch_num_threads": 8,
    ...
  },
  "bench": {
    "encoder_30s_chunk": {
      "times_sec": [1.2345, 1.2301, ...],
      "rtf_mean": 0.041152,
      "rtf_median": 0.041004,
      ...
    },
    "decoder_step": {...},
    "full_transcribe": {
      "times_sec": [...],
      "audio_duration_sec": 45.6,
      "text_preview": "转录结果...",
      "text_len": 234,
      ...
    }
  }
}
```

## 性能指标说明

### RTF (Real-Time Factor)

**实时率**：处理时间与音频时长的比值

- `RTF = 0.1`：处理 1 秒音频需要 0.1 秒（**10x 实时速度**）
- `RTF = 1.0`：处理 1 秒音频需要 1 秒（**实时速度**）
- `RTF = 2.0`：处理 1 秒音频需要 2 秒（慢于实时）

**目标**：RTF 越小越好，通常 < 0.5 为可接受范围

### 统计指标

- **mean**：平均值（综合性能）
- **median**：中位数（更稳定，不受异常值影响）
- **p90/p95**：90%/95% 分位数（尾延迟）
- **stdev**：标准差（稳定性，越小越稳定）
- **min/max**：最快/最慢

## 常见问题

### Q1：为什么 MKLDNN 比 native 慢？

可能原因：
1. **CPU 不支持高级指令集**：在老旧 CPU 上 MKLDNN 优化可能失效
2. **线程设置不当**：尝试调整 `--threads` 参数
3. **模型太小**：对于 tiny/base 模型，MKLDNN 开销可能大于收益
4. **需要预热**：增加 `--warmup` 次数

### Q2：如何选择合适的线程数？

```bash
# 查看 CPU 物理核心数
lscpu | grep "Core(s) per socket"  # Linux
wmic cpu get NumberOfCores          # Windows

# 建议从物理核心数开始测试
```

### Q3：Profiler 输出太多怎么办？

```bash
# 增加 topk 值，只看关键操作
--profile-topk 10

# 按不同字段排序，找瓶颈
--profile-sort-by cpu_time_total
```

### Q4：音频文件格式支持？

Whisper 内部使用 FFmpeg，支持常见格式：
- ✅ WAV, MP3, M4A, FLAC, OGG
- ✅ 视频文件（自动提取音轨）：MP4, MKV, AVI

### Q5：为什么 encoder 固定测试 30 秒？

Whisper 的设计是将音频切分为 30 秒 chunks 分别处理，因此 encoder benchmark 统一使用 30 秒切片以保证对比公平性。

## 进阶用法

### 批量测试脚本（自动化）

使用 `--json-auto` 简化批量测试，无需手动管理文件名：

```bash
#!/bin/bash
# 测试所有模型 + 后端组合，自动生成有意义的文件名

AUDIO="test.wav"
MODELS="tiny base small"
BACKENDS="native mkldnn"
OUTPUT_DIR="./benchmark_results"

mkdir -p $OUTPUT_DIR

for model in $MODELS; do
  for backend in $BACKENDS; do
    echo "Testing: $model with $backend"
    python whisper_cpu_bench_allinone.py \
      --audio $AUDIO \
      --model $model \
      --backend $backend \
      --json-auto \
      --json-dir $OUTPUT_DIR
  done
done

echo "All results saved to: $OUTPUT_DIR"
ls -lh $OUTPUT_DIR
```

**Windows PowerShell 版本：**

```powershell
# 批量测试脚本
$audio = "test.wav"
$models = @("tiny", "base", "small")
$backends = @("native", "mkldnn")
$outputDir = "./benchmark_results"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

foreach ($model in $models) {
    foreach ($backend in $backends) {
        Write-Host "Testing: $model with $backend"
        python whisper_cpu_bench_allinone.py `
            --audio $audio `
            --model $model `
            --backend $backend `
            --json-auto `
            --json-dir $outputDir
    }
}

Write-Host "All results saved to: $outputDir"
Get-ChildItem $outputDir | Format-Table Name, Length
```

### 结果分析脚本

```python
import json
import glob
from pathlib import Path

# 读取所有结果（支持自动生成的文件名）
results = []
result_dir = "./benchmark_results"

for f in Path(result_dir).glob("whisper_bench_*.json"):
    with open(f, encoding='utf-8') as fp:
        data = json.load(fp)
        
        # 提取关键信息
        env = data["env"]
        bench = data["bench"]
        
        result = {
            "file": f.name,
            "model": env["model"],
            "backend": env["backend_requested"],
            "threads": env.get("torch_num_threads", "default"),
            "rtf_mean": None,
            "rtf_median": None,
        }
        
        # 尝试获取 full_transcribe 结果
        if "full_transcribe" in bench:
            result["rtf_mean"] = bench["full_transcribe"].get("rtf_mean")
            result["rtf_median"] = bench["full_transcribe"].get("rtf_median")
        # 否则使用 encoder 结果
        elif "encoder_30s_chunk" in bench:
            result["rtf_mean"] = bench["encoder_30s_chunk"].get("rtf_mean")
            result["rtf_median"] = bench["encoder_30s_chunk"].get("rtf_median")
        
        results.append(result)

# 按 RTF 排序（越小越好）
results.sort(key=lambda x: x["rtf_median"] if x["rtf_median"] else 999)

# 输出结果表格
print(f"{'Model':<10} {'Backend':<10} {'Threads':<10} {'RTF(median)':<12} {'RTF(mean)':<12}")
print("-" * 60)
for r in results:
    rtf_median = f"{r['rtf_median']:.6f}" if r['rtf_median'] else "N/A"
    rtf_mean = f"{r['rtf_mean']:.6f}" if r['rtf_mean'] else "N/A"
    threads = str(r['threads'])
    print(f"{r['model']:<10} {r['backend']:<10} {threads:<10} {rtf_median:<12} {rtf_mean:<12}")

# 找出最快的配置
if results and results[0]["rtf_median"]:
    best = results[0]
    print(f"\n🏆 Best configuration:")
    print(f"   Model: {best['model']}, Backend: {best['backend']}, "
          f"Threads: {best['threads']}, RTF: {best['rtf_median']:.6f}")
    print(f"   File: {best['file']}")
```

**运行分析脚本：**

```bash
# 生成测试结果后，运行分析
python analyze_results.py
```

**输出示例：**

```
Model      Backend    Threads    RTF(median)  RTF(mean)   
------------------------------------------------------------
small      mkldnn     8          0.035234     0.035678    
small      native     8          0.042156     0.042890    
base       mkldnn     8          0.028901     0.029234    
base       native     8          0.035678     0.036123    
tiny       mkldnn     8          0.018456     0.018789    
tiny       native     8          0.023456     0.023890    

🏆 Best configuration:
   Model: tiny, Backend: mkldnn, Threads: 8, RTF: 0.018456
   File: whisper_bench_model_tiny_backend_mkldnn_t8_audio_test_20260316_143052.json
```

## 贡献与反馈

如有问题或建议，欢迎提交 Issue 或 Pull Request。

## 许可证

本工具基于 MIT 许可证开源。
