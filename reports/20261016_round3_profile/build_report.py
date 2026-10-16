"""Build a two-page Chinese profiling report from completed measured artifacts.

No training imports, device access, or invented example measurements. Only writes
report.html beside this script, and requires all eight-rank inputs to exist.
"""
from __future__ import annotations

import html
import importlib.util
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("round2_report", HERE.parent / "20261016_round2_default/build_report.py")
old_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_report)


def esc(value):
    return html.escape(str(value), quote=True)


def read(path):
    return json.loads(path.read_text())


def build(run, analysis, attribution, host, findings, ranks):
    assert run["status"] == "passed" and run["exit_code"] == 0
    assert run["all_numeric_values_finite"] and run["snapshot_hash_check"]["passed"]
    assert analysis["aggregate"]["rank_count"] == len(ranks) == len(attribution["ranks"]) == 8
    assert all(r["recorded_steps"] == 4 for r in ranks)
    assert all(r["captured_steps"] == 4 for r in attribution["ranks"].values())
    assert all(r["sync_union"]["hccl_overlap_s_per_step"] > 0 for r in attribution["ranks"].values()), "HCOM classification must be corrected"
    baseline = run["windows"]["81-116"]
    capture = run["windows"]["121-124"]
    assert baseline["usable_for_speed_comparison"] and not capture["usable_for_speed_comparison"]
    seconds = baseline["wall_seconds_per_update"]
    overhead = run["capture_logged_time_overhead_pct_vs_unprofiled"]
    aggregate = analysis["aggregate"]
    official = {k: v["mean"] for k, v in aggregate["official_rank_statistics_s_per_step"].items()}
    stage = official["Stage_s"]
    kernel_count = aggregate["kernel_calls_per_rank_step_mean"]
    small_count = sum(r["noncommunication_kernel_duration_us"]["under_50us_count"] for r in ranks)
    total_count = sum(r["noncommunication_kernel_duration_us"]["count"] for r in ranks)
    small_pct = 100 * small_count / total_count
    grouped_gemm = statistics.mean(sum(k["duration_sum_s_per_step"] for k in r["kernel_types"]
                                       if k["name"] in ("GroupedMatmul", "GroupedMatmulAdd")) for r in ranks)
    collectives = {}
    for name in ("hcom_allGather_", "hcom_reduceScatter_", "hcom_allReduce_"):
        collectives[name] = statistics.mean(next(k["calls_per_step"] for k in r["collectives"] if k["name"] == name) for r in ranks)
    phases = attribution["common_phases_across_ranks"]
    phase = lambda name: phases[name]["mean_host_s_per_step"]
    sync_mean = statistics.mean(r["sync_union"]["host_union_s_per_step"] for r in attribution["ranks"].values())
    sync_device = statistics.mean(r["sync_union"]["device_active_overlap_s_per_step"] for r in attribution["ranks"].values())
    retention = [100 * host["ranks"][str(rank)]["sampling_quality"]["main_thread_retained_sample_fraction"] for rank in (0, 4)]
    date = run["completed_at"][:10]
    categories = [
        ("非通信设备任务", "Computing_s", "矩阵、向量及其他设备任务区间", "compute"),
        ("未与计算重叠的通信", "Communication(Not Overlapped)_s", "HCCL 区间；包含依赖与等待的可能", "comm"),
        ("未记录设备任务的空档", "Free_s", "不能直接等同于 CPU 瓶颈或数据等待", "free"),
    ]
    parts = "".join(f'<span class="{style}" style="width:{100*official[key]/stage:.5f}%">{100*official[key]/stage:.1f}%</span>' for label, key, note, style in categories)
    rows = "".join(f'<tr><td><i class="dot {style}"></i>{label}</td><td>{official[key]:.3f} 秒</td><td>{note}</td></tr>' for label, key, note, style in categories)
    host_rows = "".join(f'<tr><td>{label}</td><td>{phase(name):.3f} 秒</td><td>{note}</td></tr>' for label, name, note in [
        ("模型前向", "train.forward", "含 FSDP 调度、算子提交及等待"),
        ("模型反向", "train.backward", "进入 autograd 后的主机范围"),
        ("深度与视频 teacher", "train.teachers", "在线生成辅助监督目标"),
        ("优化器", "train.optimizer", "Muon 与 AdamW 的主机范围"),
    ])
    recommendations = findings["recommendations"]
    communication_example = findings["communication_wait_example"]
    wait_callout = (
        f'<b>一次通信的等待证据：</b>归一化相对序号并核对类型、通信组与记录流量后，'
        f'第 {communication_example["training_update"]} 步一次 reduce-scatter 中，rank 0 持续 '
        f'{communication_example["rank0_duration_ms"]:.1f} 毫秒、rank 2 仅 '
        f'{communication_example["rank2_duration_ms"]:.3f} 毫秒，各卡结束时间相差 '
        f'{communication_example["end_skew_ms"]:.3f} 毫秒。这支持到达不齐 / 等待，'
        '不能据此推算纯传输时间或无采集时的等待量。'
    )
    recommendation_html = "".join(
        f'<div class="priority"><b>{esc(item["priority"])} · {esc(item["title"])}</b><p>{esc(item["detail"])}</p></div>'
        for item in recommendations[:4]
    )
    limits = [item for item in findings.get("limitations", [])
              if not item.startswith("采集窗口日志步时") and "同名HCOM" not in item]
    limitations = "".join(f'<li>{esc(item)}</li>' for item in limits[:2])
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LingBot-VLA｜优化后全面 Profiling</title>
<meta name="description" content="当前默认八卡训练的算子、通信、CPU采样和同步开销剖析。">
<style>
/* EMBEDDED_FONT */
:root{{--ink:#172b45;--muted:#5c6d7d;--accent:#147b72;--line:#dce5eb;--soft:#eaf5f2}}
*{{box-sizing:border-box}}body{{margin:0;background:#e9eef2;color:var(--ink);font:13px/1.6 ReportCJK,"Noto Sans SC","Microsoft YaHei",sans-serif}}
.page{{width:210mm;min-height:297mm;margin:22px auto;padding:14mm 15mm 15mm;background:white;box-shadow:0 8px 35px #17324d14}}
.meta{{display:flex;justify-content:space-between;gap:15px;border-bottom:1px solid var(--line);padding-bottom:9px;margin-bottom:16px;color:var(--muted);font-size:10px}}
.eyebrow{{color:var(--accent);font-size:10px;letter-spacing:.08em;font-weight:bold}}h1{{font-size:28px;line-height:1.35;letter-spacing:-.025em;margin:7px 0 12px}}h2{{font-size:16px;line-height:1.4;margin:18px 0 9px}}h3{{font-size:12px;margin:0 0 4px}}p{{margin:6px 0 10px}}.lead{{color:var(--muted);font-size:12px;line-height:1.8}}
.hero{{display:flex;gap:10px;margin:16px 0}}.metric{{flex:1;background:#f7fafb;border:1px solid var(--line);border-radius:9px;padding:12px}}.metric:first-child{{background:var(--soft);border-color:#cde5df}}.metric strong{{font-size:29px;letter-spacing:-.04em;display:block;color:var(--accent);line-height:1.25}}.metric small{{font-size:10px;color:var(--muted);display:block;margin-top:5px}}
.timeline{{display:flex;width:100%;height:34px;overflow:hidden;border-radius:6px;margin:11px 0;color:white;text-align:center;font-size:12px;line-height:34px}}.compute{{background:#167e75}}.comm{{background:#64879e}}.free{{background:#bac6cf;color:#273e50}}.dot{{display:inline-block;width:7px;height:7px;border-radius:2px;margin-right:7px}}
table{{width:100%;border-collapse:collapse;font-size:11px;line-height:1.55}}th{{font-size:10px;color:var(--muted);background:#f3f6f8;text-align:left;font-weight:600}}td,th{{padding:8px;border-bottom:1px solid var(--line)}}td:nth-child(2){{white-space:nowrap}}.note{{font-size:10px;line-height:1.7;color:var(--muted)}}
.cards{{display:flex;flex-wrap:wrap;gap:10px}}.card{{width:calc(50% - 5px);border:1px solid var(--line);border-radius:8px;padding:11px 12px;font-size:11px;line-height:1.65}}.card p{{margin:0}}.card.wide{{width:100%}}.callout{{border-left:3px solid var(--accent);background:#f1f7f5;padding:10px 12px;font-size:11px;line-height:1.75;margin:12px 0}}
.details{{display:flex;gap:12px;margin-top:10px}}.detail{{flex:1;padding:9px 11px;background:#f6f8fa;border-radius:6px;font-size:10.5px;line-height:1.7}}.detail b{{color:var(--accent)}}.priority{{border-bottom:1px solid var(--line);padding:8px 0;font-size:11px;line-height:1.6}}.priority b{{color:var(--accent)}}.priority p{{margin:3px 0 0}}.recommendations{{display:grid;grid-template-columns:1fr 1fr;gap:0 16px}}
.sources{{font-size:9px;line-height:1.7;color:var(--muted);margin-top:10px;overflow-wrap:anywhere}}.footer{{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding-top:8px;margin-top:14px;color:var(--muted);font-size:9px}}a{{color:#186f73;text-decoration:none}}code{{font-family:ui-monospace,"DejaVu Sans Mono",monospace;font-size:.9em}}ul{{margin:5px 0 8px;padding-left:17px}}li{{margin:3px 0}}.second h2{{margin-top:15px}}
@page{{size:A4;margin:0}}@media print{{body{{background:white}}.page{{width:210mm;min-height:297mm;margin:0;box-shadow:none;break-after:page;page-break-after:always}}.page:last-child{{break-after:auto;page-break-after:auto}}*{{print-color-adjust:exact;-webkit-print-color-adjust:exact}}}}
@media screen and (max-width:820px){{.page{{width:calc(100% - 24px);min-height:0;padding:24px;margin:12px}}h1{{font-size:25px}}.recommendations{{grid-template-columns:1fr}}.card{{width:100%}}.metric strong{{font-size:25px}}.metric{{padding:9px}}.details{{display:block}}.detail{{margin:6px 0}}td,th{{padding:7px 5px}}}}
</style></head><body>
<main class="page" aria-label="第1页：瓶颈证据">
<div class="meta"><span>LINGBOT-VLA · ROBOTWIN · 当前默认 8 卡配置</span><span>{esc(date)} · 01 / 02</span></div>
<div class="eyebrow">第二轮优化后 · 全面 PROFILING</div>
<h1>通信与细碎任务仍然显著，<br>专家 GMM 已非首要瓶颈。</h1>
<p class="lead">{esc(findings['summary'])}</p>
<div class="hero"><div class="metric"><strong>{seconds:.2f} 秒</strong><small>无原生采集 · 每次更新实际耗时<br>第 81–116 步，全局 batch 96</small></div><div class="metric"><strong>+{overhead:.1f}%</strong><small>采集窗口日志步时增加幅度<br>约 {baseline['logged_step_seconds']['mean']:.2f} → {capture['logged_step_seconds']['mean']:.2f} 秒</small></div><div class="metric"><strong>{kernel_count/10000:.2f} 万</strong><small>每卡每步记录的设备任务数<br>非通信任务 {small_pct:.1f}% 短于 50 微秒</small></div></div>
<h2>8 卡设备时间线：通信与空档各约三成</h2>
<div class="timeline" aria-label="采集窗口官方设备时间分类">{parts}</div>
<table><thead><tr><th>官方时间分类</th><th>每卡每步均值</th><th>正确解释</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">来自全 8 rank、每 rank 4 个采集步；分母为官方设备 Stage 均值 {stage:.3f} 秒。分类由设备区间并集计算，通信重叠部分已去重。这些比例不是 AI Core 算力利用率，不能套到无采集的 {seconds:.2f} 秒上。</p>
<h2>低利用率的三个具体线索</h2>
<div class="cards"><div class="card"><h3>01 · 大量短任务与逐张量操作</h3><p>每卡每步约 {kernel_count:,.0f} 个设备任务，{small_pct:.1f}% 的非通信任务不足 50 微秒。Cast、Slice、Mul、Add 等调用很多，矩阵单元难以持续忙碌；但空档不能全部归因于 CPU 发射。</p></div>
<div class="card"><h3>02 · 分布式通信显著，带宽瓶颈未证实</h3><p>每步约 {collectives['hcom_allGather_']:.0f} 次 all-gather、{collectives['hcom_reduceScatter_']:.0f} 次 reduce-scatter、{collectives['hcom_allReduce_']:.0f} 次 all-reduce。当前 EP=1，无专家 all-to-all；HCCL 区间可能含依赖等待，不能直接解释为线路传输耗时。</p></div>
<div class="card wide"><h3>03 · MoE 的 GMM 本体已很小，架构相关的其他开销仍需细分</h3><p>GroupedMatmul 与 GroupedMatmulAdd 设备时长合计平均仅 <b>{grouped_gemm*1000:.1f} 毫秒 / 卡 / 步</b>（任务时长求和，非排他墙钟时间）。这不包含路由、重排、监控、优化器和 FSDP；不能把整个 MoE 的成本视为这一个数字。<b>MoE 的大权重仍可能通过 FSDP 间接增加通信开销。</b></p></div></div>
<div class="callout">{wait_callout}</div>
<div class="footer"><span>设备证据：<a href="analysis/summary.json">全 8 卡汇总</a> · <a href="analysis/rank0.json">rank 0 完整算子数据</a></span><span>设备与通信 / 1</span></div>
</main>
<section class="page second" aria-label="第2页：优化优先级与测量范围">
<div class="meta"><span>主机时间范围 · 同步归因 · 下一步优化</span><span>{esc(date)} · 02 / 02</span></div>
<h1>先针对具体路径，<br>再用对照实验确认收益。</h1>
<p class="lead">主机范围定位代码在提交或等待什么，设备时间线说明当时有哪些任务。两种观测互相补充；主机等待不是可以额外加到设备计算上的独立开销。</p>
<table><thead><tr><th>采集窗口主机范围</th><th>8 卡平均每步</th><th>范围含义</th></tr></thead><tbody>{host_rows}</tbody></table>
<div class="details"><div class="detail"><b>Teacher 与 CPU 几何拟合</b><br>geometry_recover {phase('teacher.geometry_recover'):.3f} 秒，包含 SciPy 拟合 {phase('teacher.geometry_scipy'):.3f} 秒；两者嵌套，不能相加。video teacher 主机范围 {phase('teacher.video'):.3f} 秒。</div><div class="detail"><b>Muon 与同步</b><br>Newton–Schulz 主机范围 {phase('optimizer.muon_newton_schulz'):.3f} 秒，已包含在优化器范围内。同步范围并集约 {sync_mean:.3f} 秒，其中 {sync_device:.3f} 秒与记录的设备活动重叠。</div></div>
<p class="note">上述均为有采集开销的主机包含式范围，不能作为生产步时的可加分解，也不能直接视为某优化的可节省上限。同步与设备活动重叠只表示同时发生，不证明因果归属。</p>
<h2>建议的实验顺序</h2><div class="recommendations">{recommendation_html}</div>
<h2>本轮实际覆盖与限制</h2>
<p class="note">完成 {run['updates']} 步八卡训练，日志无 NaN/Inf；实际默认配置与前轮复验一致。全 8 卡采集训练第 121–124 步，profiler 内部编号 120–123；第 120 步预热，采集与导出步不计入正常吞吐。额外对 rank 0/4 进行 20 Hz、30 秒 Python 栈采样，基线排除第 50–64 步。</p>
<ul class="note"><li>CPU 非阻塞栈采样有效保留率 {retention[0]:.1f}% / {retention[1]:.1f}%；遗漏可能有偏，只作定位佐证，不是 CPU/NPU 利用率。</li><li>+{overhead:.1f}% 是不同窗口的日志步时增加幅度，不是严格 A/B 的 profiler 开销；不含窗口外 trace 导出与启动停止成本。</li>{limitations}</ul>
<div class="sources">可复核数据：<a href="run_summary.json">运行与速度窗口</a> · <a href="analysis/summary.json">设备与通信</a> · <a href="attribution_scope_summary.json">主机范围与同步</a> · <a href="host_sampling_analysis.json">Python 栈及采样质量</a> · <a href="host_config_audit.json">配置审计</a> · <a href="findings.json">结论与建议</a> · <a href="cross_rank_sequence_diagnostic.json">通信序号核验</a> · <a href="analysis.txt">详细分析</a><br>完整实验目录：<code>{esc(run['run'])}</code></div>
<div class="footer"><span>仅独立快照添加计时范围；生产训练算法与默认配置未改变</span><span>归因与后续实验 / 2</span></div>
</section></body></html>'''
    return document.replace("/* EMBEDDED_FONT */", old_report.embedded_font(document))


def main():
    run = read(HERE / "run_summary.json")
    analysis = read(HERE / "analysis/summary.json")
    attribution = read(HERE / "attribution_scope_summary.json")
    host = read(HERE / "host_sampling_analysis.json")
    findings = read(HERE / "findings.json")
    ranks = [read(HERE / "analysis" / name) for name in analysis["per_rank_files"].values()]
    document = build(run, analysis, attribution, host, findings, ranks)
    path = HERE / "report.html"
    path.write_text(document, encoding="utf-8")
    print(json.dumps({"report": str(path), "bytes": path.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
