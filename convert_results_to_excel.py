#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 data 目录下的 JSON benchmark 结果整理成 Excel 文件
"""

import json
from pathlib import Path
import pandas as pd
from datetime import datetime


def extract_benchmark_data(json_path):
    """从 JSON 文件提取关键信息"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    env = data.get('env', {})
    bench = data.get('bench', {})
    
    # 基础信息
    result = {
        'file_name': json_path.name,
        'audio_file': Path(env.get('audio', '')).name,
        'audio_duration_sec': env.get('audio_duration_sec', 0),
        'model': env.get('model', ''),
        'language': env.get('language', ''),
        'task': env.get('task', ''),
    }
    
    # 后端配置
    result.update({
        'backend': env.get('backend_requested', ''),
        'native_isa': env.get('native_isa_requested', 'auto'),
        'onednn_isa': env.get('onednn_isa_requested', 'auto'),
        'mkldnn_enabled': env.get('mkldnn_enabled', False),
        'cpu_capability': env.get('cpu_capability', ''),
    })
    
    # 线程配置
    result.update({
        'threads': env.get('torch_num_threads', 0),
        'interop_threads': env.get('torch_num_interop_threads', 0),
        'warmup': env.get('warmup', 0),
        'repeat': env.get('repeat', 0),
    })
    
    # Encoder 性能
    if 'encoder_30s_chunk' in bench:
        encoder = bench['encoder_30s_chunk']
        result.update({
            'encoder_rtf_mean': encoder.get('rtf_mean', None),
            'encoder_rtf_median': encoder.get('rtf_median', None),
            'encoder_mean_ms': sum(encoder.get('times_sec', [])) / len(encoder.get('times_sec', [1])) * 1000 if encoder.get('times_sec') else None,
        })
    else:
        result.update({
            'encoder_rtf_mean': None,
            'encoder_rtf_median': None,
            'encoder_mean_ms': None,
        })
    
    # Decoder 性能
    if 'decoder_step' in bench:
        decoder = bench['decoder_step']
        result.update({
            'decoder_steps_per_sec': decoder.get('steps_per_sec_mean', None),
            'decoder_mean_ms': sum(decoder.get('times_sec', [])) / len(decoder.get('times_sec', [1])) * 1000 if decoder.get('times_sec') else None,
        })
    else:
        result.update({
            'decoder_steps_per_sec': None,
            'decoder_mean_ms': None,
        })
    
    # Full transcribe 性能
    if 'full_transcribe' in bench:
        full = bench['full_transcribe']
        result.update({
            'full_rtf_mean': full.get('rtf_mean', None),
            'full_rtf_median': full.get('rtf_median', None),
            'full_mean_ms': sum(full.get('times_sec', [])) / len(full.get('times_sec', [1])) * 1000 if full.get('times_sec') else None,
            'text_len': full.get('text_len', 0),
            'num_segments': full.get('num_segments', 0),
        })
    else:
        result.update({
            'full_rtf_mean': None,
            'full_rtf_median': None,
            'full_mean_ms': None,
            'text_len': 0,
            'num_segments': 0,
        })
    
    return result


def main():
    # 数据目录
    data_dir = Path(__file__).parent / 'data'
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 查找所有 JSON 文件
    json_files = list(data_dir.glob('whisper_bench_*.json'))
    
    if not json_files:
        print(f"❌ 未找到任何 JSON 文件: {data_dir}")
        return
    
    print(f"📁 找到 {len(json_files)} 个 JSON 文件")
    
    # 提取数据
    results = []
    for json_file in json_files:
        try:
            result = extract_benchmark_data(json_file)
            results.append(result)
            print(f"✅ 处理: {json_file.name}")
        except Exception as e:
            print(f"❌ 处理失败 {json_file.name}: {e}")
    
    if not results:
        print("❌ 没有成功提取任何数据")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 按条件排序（backend, native_isa, onednn_isa, full_rtf_median）
    sort_columns = []
    if 'backend' in df.columns:
        sort_columns.append('backend')
    if 'native_isa' in df.columns:
        sort_columns.append('native_isa')
    if 'onednn_isa' in df.columns:
        sort_columns.append('onednn_isa')
    if 'full_rtf_median' in df.columns:
        sort_columns.append('full_rtf_median')
    
    if sort_columns:
        df = df.sort_values(by=sort_columns)
    
    # 生成输出文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(__file__).parent / f'benchmark_results_{timestamp}.xlsx'
    
    # 写入 Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 主表：所有数据
        df.to_excel(writer, sheet_name='All Results', index=False)
        
        # 调整列宽
        worksheet = writer.sheets['All Results']
        for idx, col in enumerate(df.columns, 1):
            try:
                max_length = max(
                    df[col].astype(str).str.len().max(),
                    len(str(col))
                )
                column_letter = chr(64 + idx) if idx <= 26 else f"A{chr(64 + idx - 26)}"
                worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
            except:
                # 如果出错，使用默认宽度
                pass
        
        # 汇总表：按配置分组
        if 'full_rtf_median' in df.columns:
            summary_cols = ['backend', 'native_isa', 'onednn_isa', 'threads', 
                           'full_rtf_median', 'full_rtf_mean', 'encoder_rtf_median']
            summary_df = df[summary_cols].copy() if all(c in df.columns for c in summary_cols) else df
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 调整汇总表列宽
            worksheet = writer.sheets['Summary']
            for idx, col in enumerate(summary_df.columns, 1):
                try:
                    max_length = max(
                        summary_df[col].astype(str).str.len().max(),
                        len(str(col))
                    )
                    column_letter = chr(64 + idx) if idx <= 26 else f"A{chr(64 + idx - 26)}"
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 30)
                except:
                    pass
    
    print(f"\n✅ Excel 文件已生成: {output_file}")
    print(f"📊 共整理 {len(results)} 条记录")
    
    # 显示统计信息
    print("\n📈 统计信息:")
    if 'backend' in df.columns:
        print(f"  - 后端类型: {df['backend'].unique().tolist()}")
    if 'model' in df.columns:
        print(f"  - 模型: {df['model'].unique().tolist()}")
    if 'threads' in df.columns:
        print(f"  - 线程数: {sorted(df['threads'].unique().tolist())}")
    
    # 显示性能最优的配置
    if 'full_rtf_median' in df.columns:
        best_row = df.loc[df['full_rtf_median'].idxmin()]
        print("\n🏆 性能最优配置 (Full RTF Median):")
        print(f"  - 后端: {best_row.get('backend', 'N/A')}")
        print(f"  - Native ISA: {best_row.get('native_isa', 'N/A')}")
        print(f"  - oneDNN ISA: {best_row.get('onednn_isa', 'N/A')}")
        print(f"  - 线程数: {best_row.get('threads', 'N/A')}")
        print(f"  - RTF (median): {best_row.get('full_rtf_median', 'N/A'):.6f}")
        print(f"  - 文件: {best_row.get('file_name', 'N/A')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
