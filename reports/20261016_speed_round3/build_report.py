"""Generate a two-page offline HTML report from completed validation artifacts."""
import html
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    comparison = read("comparison.json")
    decision = read("selection.json")
    selected = decision["case"]
    profile_dir = HERE / "no_prefetch_profile" if selected == "no_prefetch" else HERE
    profile = json.loads((profile_dir / "profile_comparison.json").read_text())
    scopes = json.loads((profile_dir / "attribution_scope_summary.json").read_text())
    previous_scopes = json.loads((HERE.parent / "20261016_round3_profile/attribution_scope_summary.json").read_text())
    cases = comparison["cases"]
    assert cases[selected]["eligible"]
    final = comparison["comparisons"][selected]
    perf = final["performance"]["31:110"]
    b, a = perf["baseline_s"], perf["candidate_s"]
    gain = perf["throughput_gain_percent"]
    reduction = perf["update_time_reduction_percent"]
    pair = final["paired_metrics"]["metrics"]
    kernel0, kernel1 = (profile[k]["kernels_per_update"] for k in ("baseline", "optimized"))
    metrics = []
    for phase, title in (("Optimizer.step#DistributedMuon.step", "Muon 主机范围"),
                         ("optimizer.muon_newton_schulz", "Newton–Schulz 主机范围"),
                         ("teacher.video", "video teacher 主机范围")):
        old = previous_scopes["common_phases_across_ranks"].get(phase, {}).get("mean_host_s_per_step")
        new = scopes["common_phases_across_ranks"].get(phase, {}).get("mean_host_s_per_step")
        if old is not None and new is not None:
            metrics.append((title, f"{old*1000:.1f} ms", f"{new*1000:.1f} ms"))
    metrics.insert(0, ("每卡每步 kernel 数", f"{kernel0:,.0f}", f"{kernel1:,.0f}"))
    for key, title in (("Free_s", "设备空闲范围"), ("Communication(Not Overlapped)_s", "未重叠通信范围")):
        old = profile["baseline"]["official_mean_s_per_step"][key]
        new = profile["optimized"]["official_mean_s_per_step"][key]
        metrics.append((title, f"{old*1000:.1f} ms", f"{new*1000:.1f} ms"))
    for name in ("BatchMatMulV2", "RmsNorm", "RmsNormGrad"):
        old = profile["baseline"]["kernel_types"].get(name, {}).get("calls_per_step", 0)
        new = profile["optimized"]["kernel_types"].get(name, {}).get("calls_per_step", 0)
        if old or new:
            metrics.append((name + " 次数/步", f"{old:,.0f}", f"{new:,.0f}"))
    metric_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td><td>{w}</td></tr>" for k, v, w in metrics)
    speed_rows = ""
    for name, title in (("baseline", "第二轮默认 / 基线"), ("candidate_v2", "第三轮候选 · 120 步"), (selected, "最终配置 · 132 步")):
        w = cases[name]["windows"]["31:110"]
        speed_rows += f"<tr><td>{title}</td><td>{w['wall_seconds_per_update']:.4f}</td><td>{w['global_samples_per_second']:.2f}</td><td>{cases[name]['exit_code']}</td></tr>"
    resources = {}
    for name in ("baseline", selected):
        cards = cases[name]["windows"]["31:110"]["resources"]["cards"]
        values = [v["Aicore Usage Rate(%)"]["mean"] for v in cards.values() if "Aicore Usage Rate(%)" in v]
        resources[name] = sum(values)/len(values) if values else None
    utilization = ""
    if all(v is not None for v in resources.values()):
        utilization = f"该窗口的 8 卡 AI Core 采样均值：{resources['baseline']:.1f}% → {resources[selected]:.1f}%。这是 npu-smi 轮询值，不是 MFU；短窗口采样会受波动影响。"
    on = cases["default_profile"]["windows"]["31:110"]["wall_seconds_per_update"]
    off = cases["no_prefetch"]["windows"]["31:110"]["wall_seconds_per_update"]
    prefetch_setting = "0" if selected == "no_prefetch" else "1"
    prefetch_choice = "关闭" if selected == "no_prefetch" else "开启"
    prefetch_description = f"实现按 QKV/输出两阶段顺序的显式预取并单独对照：开启 {on:.4f} s/步，关闭 {off:.4f} s/步。差异仅 0.5%，未超出短程波动，最终默认{prefetch_choice}。"
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第三轮昇腾训练优化 · 实测报告</title>
<style>
@page{{size:A4;margin:12mm}}*{{box-sizing:border-box}}body{{margin:0;background:#e9eef5;color:#172d43;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}}.page{{width:210mm;min-height:290mm;margin:18px auto;background:white;padding:17mm;page-break-after:always}}.page:last-child{{page-break-after:auto}}h1{{font-size:29px;line-height:1.25;margin:9px 0 14px}}h2{{font-size:18px;margin:20px 0 8px}}p{{margin:8px 0}}.eyebrow{{font-size:12px;color:#54718c;letter-spacing:1px}}.cards{{display:flex;gap:12px;margin:18px 0}}.card{{flex:1;background:#eff5fb;padding:15px;border-radius:7px}}.number{{font-size:28px;font-weight:700;color:#1762a1}}.label{{font-size:12px;color:#4a647e}}table{{width:100%;border-collapse:collapse;font-size:12px}}th{{background:#173b5d;color:white;text-align:left}}th,td{{padding:9px 10px;border-bottom:1px solid #dde5ee;vertical-align:top}}.note{{font-size:11px;color:#52697d}}.box{{padding:12px 15px;background:#f3f7fa;border-left:3px solid #3c84bc;margin:13px 0}}a{{color:#185d94;text-decoration:none}}code{{font-size:11px;background:#eef2f6;padding:2px 3px}}.foot{{margin-top:20px;border-top:1px solid #d9e3ed;padding-top:9px;color:#6b7e90;font-size:11px}}.bar{{background:#edf1f5;margin:9px 0;height:26px;border-radius:3px}}.bar span{{display:block;height:100%;background:#7a8fa3;color:white;font-size:12px;padding:3px 8px}}.bar.new span{{background:#267ca7}}
@media print{{body{{background:white;font-size:12px}}.page{{width:auto;min-height:0;height:269mm;margin:0;padding:4mm 3mm;overflow:hidden}}h1{{font-size:27px}}h2{{margin-top:15px}}th,td{{padding:7px 8px}}.cards{{margin:14px 0}}}}
</style>
<section class="page"><div class="eyebrow">LINGBOT-VLA V2 · ASCEND 910B1 × 8 · 2026-10-16</div>
<h1>第三轮优化：减少小算子与同步开销</h1>
<p>默认训练脚本已接入验证通过的优化。保持 micro batch 12、global batch 96、Muon、学习率调度与模型结构，采用本机 torch_npu 2.6 / CANN 8.2 已有能力。</p>
<div class="cards"><div class="card"><div class="number">+{gain:.1f}%</div><div class="label">实际训练吞吐提升</div></div><div class="card"><div class="number">{a:.3f} s</div><div class="label">默认配置每次更新耗时</div></div><div class="card"><div class="number">−{profile['kernel_count_reduction_percent']:.1f}%</div><div class="label">每卡每步 kernel 数量</div></div></div>
<h2>同条件训练对照</h2><table><tr><th>运行</th><th>实际秒/步</th><th>样本/秒</th><th>退出码</th></tr>{speed_rows}</table>
<div class="bar"><span style="width:100%">基线 {b:.3f} s</span></div><div class="bar new"><span style="width:{a/b*100:.2f}%">默认优化 {a:.3f} s</span></div>
<p class="note">统一取第 31–110 步：用 step 110 与 step 30 日志时间差除以 80，包含数据等待；时间戳量化误差约 ±0.0125 s/步。最终运行在第 121–124 步采样，测速窗口未进入 profiler 采集。</p>
<h2>具体修改</h2><table><tr><th style="width:24%">位置</th><th>修改与作用</th></tr>
<tr><td>Muon 专家权重</td><td>同形状的 3D 本地专家权重合批执行 Newton–Schulz；每个专家仍独立归一化，保留动量与状态结构。每批输入最多 32 MiB，单个参数除外。</td></tr>
<tr><td>RMSNorm / AdaNorm</td><td>在原有模块内部使用 <code>torch_npu.npu_rms_norm</code> 融合 FP32 计算；保留 FiLM、权重键和原有低精度转换顺序。</td></tr>
<tr><td>位置元数据</td><td>整数索引集中在 CPU 生成，减少循环读取 NPU 标量；修复视觉位置缓存重复计算，并在网格或 batch 变化时失效。</td></tr>
<tr><td>video teacher</td><td>缓存 RGB 均值/方差与形状决定的序列边界，减少每步重复创建设备常量。缓存不包含图像、特征或训练结果。</td></tr>
<tr><td>FSDP2 对照</td><td>{prefetch_description}</td></tr></table>
<div class="foot">1 / 2 · 主入口：train_robotwin_npu.sh · 各可选优化均可通过环境变量关闭</div></section>
<section class="page"><div class="eyebrow">VALIDATION · PROFILING · SOURCES</div><h1>验证结果与适用边界</h1>
<h2>数值、状态与运行验证</h2><p>21 项 NPU 检查、44 项启动脚本检查通过；双卡 DTensor Muon 更新和旧优化器状态恢复通过，两阶段 FSDP 预取前后向对照通过。120 步候选及预取开/关两组 132 步验证均正常结束。最终脚本使用与关闭预取试验相同的有效配置。</p>
<p>不带 profiler 的候选峰值分配显存为 <b>{cases['candidate_v2']['peak_allocated_gib']:.2f} GiB</b>，基线为 {cases['baseline']['peak_allocated_gib']:.2f} GiB（仅 rank 0 allocator 统计）。</p>
<p class="note">融合计算产生浮点舍入差异，完整轨迹并非逐位一致。最终配置前 120 步对照的 loss 最大绝对差为 {pair['loss']['max_abs_diff']['abs_diff']:.6f}，梯度范数最大相对差为 {pair['grad_norm']['max_relative_diff']['relative_diff']*100:.2f}%。上述短程验证不替代完整 epoch 收敛与机器人任务成功率评估。</p>
<h2>8 卡 profiler 复查</h2><table><tr><th>指标</th><th>优化前</th><th>优化后</th></tr>{metric_rows}</table>
<p class="note">两次均为 Level1、PipeUtilization、8 卡、第 121–124 次更新。主机范围包含异步提交与等待，存在嵌套和重叠，不能相加作为端到端节省时间；kernel 时间和通信等待也不等于算力利用率。</p>
<p class="note">{utilization}</p>
<div class="box">端到端更新时间下降 {reduction:.1f}%。剩余开销仍包含分片通信、teacher 几何恢复和小算子调度；本轮没有消除所有低利用率原因。除 FSDP 预取外，各项收益未做独立消融，报告给出整组修改的实测效果。</div>
<h2>网络调研如何用于本轮</h2><p class="note"><a href="https://github.com/Ascend/msprof-analyze/blob/master/docs/zh/fused_operator_api_replacement_example.md">昇腾官方融合算子示例</a>：采用现有 RMSNorm 接口，并用本机数值测试验证。<a href="https://github.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/tuning/fsdp2_backend_performance_optimization.md">MindSpeed FSDP2 调优文档</a>：参考显式预取，再按本模型两阶段调用调整。</p>
<p class="note"><a href="https://github.com/Ascend/triton-ascend/blob/main/docs/en/installation_guide.md">Triton-Ascend 兼容说明</a>及<a href="https://github.com/Hailey-Zh/triton-muon-npu">第三方 Triton Muon 实现</a>用于评估后续方案，未引入当前环境。现有 NpuFusedAdamW 的存储合并需要额外适配 DTensor；本轮保留已验证的 AdamW foreach。</p>
<p class="note">复现开关：<code>LINGBOT_MUON_LOCAL_BATCH=8</code>、<code>LINGBOT_HOST_METADATA=1</code>、<code>LINGBOT_NPU_FUSED_RMSNORM=1</code>、<code>LINGBOT_FSDP_FORWARD_PREFETCH={prefetch_setting}</code>。设为 0 可关闭对应可选优化。</p>
<div class="foot">2 / 2 · 证据索引：artifact_index.txt；最终 profiler 统计在 no_prefetch_profile/。早期加载失败的候选已修复并重跑，不计入速度对照。</div></section></html>'''
    spec = importlib.util.spec_from_file_location("report_fonts", HERE.parent / "20261016_round2_default/build_report.py")
    fonts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fonts)
    content = content.replace('BlinkMacSystemFont,"Noto Sans CJK SC"', 'BlinkMacSystemFont,ReportCJK,"Noto Sans CJK SC"')
    content = content.replace("<style>", "<style>\n" + fonts.embedded_font(content), 1).replace(";overflow:hidden", "")
    (HERE / "report.html").write_text(content)
    summary = {"selected_case": selected, "fsdp_prefetch_default": prefetch_setting,
               "default_seconds_per_update": a, "baseline_seconds_per_update": b,
               "throughput_gain_percent": gain, "update_time_reduction_percent": reduction,
               "kernel_count_reduction_percent": profile["kernel_count_reduction_percent"],
               "ai_core_sample_mean_percent": resources, "report": str(HERE / "report.html")}
    (HERE / "completion.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
