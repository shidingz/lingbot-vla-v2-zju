"""Build a two-page HTML report from completed, independently audited results.

Run after summarize.py has produced completion.json. This command never starts
training or imports torch. The embedded font is optional and generated locally.
"""
from __future__ import annotations

import base64
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def escape(value):
    return html.escape(str(value), quote=True)


def embedded_font(document):
    """Subset a local CJK font when available; do not require network access."""
    font_path = Path("/tmp/lingbot_report_fonts/NotoSansSC-Regular.ttf")
    if not font_path.is_file():
        return ""
    try:
        from fontTools import subset
        from fontTools.ttLib.woff2 import haveBrotli
    except ImportError:
        return ""

    class Text(HTMLParser):
        def __init__(self):
            super().__init__()
            self.chunks = []

        def handle_data(self, data):
            self.chunks.append(data)

    text = Text()
    text.feed(document)
    options = subset.Options()
    options.flavor = "woff2" if haveBrotli else None
    font = subset.load_font(str(font_path), options)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text="".join(text.chunks))
    subsetter.subset(font)
    stream = io.BytesIO()
    subset.save_font(font, stream, options)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    mime, flavor = ("woff2", "woff2") if haveBrotli else ("ttf", "truetype")
    return ('@font-face{font-family:ReportCJK;src:url("data:font/' + mime + ';base64,'
            + encoded + '") format("' + flavor + '");font-weight:100 900;font-display:swap;}')


def build_report(completion, audit, launch, historical):
    assert completion["status"] == "passed", "Training validation has not passed"
    assert audit["passed"] and launch["passed"], "Configuration/launcher audit failed"
    assert completion["all_numeric_log_values_finite"]
    assert completion["launcher_exit_code"] == 0
    assert completion["performance_flags_from_main_defaults"]
    assert audit["no_model_data_optimizer_precision_batch_loss_difference"]
    assert all(audit["main_default_provenance_checks"].values())
    assert completion["launcher_checks_passed"] == launch["check_count"]
    history = {row["case"]: row for row in historical["results"]}
    old = completion["historical_controlled_ab"]["baseline"]
    new = completion["historical_controlled_ab"]["optimized"]
    assert old == history["baseline_repeat"]
    assert new == history["combined_no_checkpoint"]
    low_memory = history["combined"]
    gain = (new["samples_s"] / old["samples_s"] - 1) * 100
    reduction = (1 - new["wall_s"] / old["wall_s"]) * 100
    updates = completion["updates"]
    peak = completion["peak_allocated_rank0_gib"]
    date = escape(completion["completed_at"][:10])
    rows = []
    for name, window in completion["windows"].items():
        assert window["available"] and window["usable_for_speed_comparison"]
        first, last = window["steps_inclusive"]
        purpose = ("与历史实验相同窗口" if (first, last) == (13, 60)
                   else "跨过预热后的稳定性" if first == 115
                   else "剔除启动阶段的整体表现")
        rows.append(f'<tr><td>第 {first}–{last} 步<span class="cell-note">{purpose}</span></td>'
                    f'<td>{window["wall_seconds_per_update"]:.3f} 秒</td>'
                    f'<td>{window["global_samples_per_second"]:.2f}</td></tr>')
    current_rows = "".join(rows)
    historical_rows = "".join(
        f'<tr class="{style}"><td>{label}</td><td>{row["wall_s"]:.2f} 秒</td>'
        f'<td>{row["samples_s"]:.2f}</td><td>{row["peak_allocated_rank0_gib"]:.2f} GiB</td></tr>'
        for label, row, style in [
            ("第一轮 GMM 基线（复测）", old, ""),
            ("第二轮组合 · 保留重计算", low_memory, ""),
            ("第二轮组合 · 关闭重计算", new, "selected"),
        ]
    )
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="LingBot-VLA 第二轮 NPU 训练优化：历史性能对照、默认入口接入与长窗口验证。">
<title>LingBot-VLA｜第二轮优化完成报告</title>
<style>
/* EMBEDDED_FONT */
:root{{--ink:#172b45;--muted:#5c6d7d;--accent:#147b72;--soft:#eaf5f2;--line:#dce5eb}}
*{{box-sizing:border-box}} body{{margin:0;background:#e9eef2;color:var(--ink);font:14px/1.65 ReportCJK,"Noto Sans SC","Microsoft YaHei",sans-serif}}
.page{{position:relative;width:210mm;min-height:297mm;margin:22px auto;padding:14mm 15mm 16mm;background:#fff;box-shadow:0 8px 35px #17324d14}}
.eyebrow{{font-size:10px;letter-spacing:.1em;color:var(--accent);font-weight:bold;text-transform:uppercase}}
.meta{{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding-bottom:9px;margin-bottom:19px;color:var(--muted);font-size:11px}}
h1{{font-size:30px;line-height:1.3;letter-spacing:-.035em;margin:7px 0 10px}} h2{{font-size:17px;line-height:1.4;margin:20px 0 10px}} h3{{font-size:13px;margin:0 0 5px}}
p{{margin:6px 0 10px}} .lead{{font-size:13px;color:var(--muted);line-height:1.8}}
.hero{{display:flex;gap:12px;margin:18px 0}} .metric{{flex:1;border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:#f8fafb}}
.metric:first-child{{background:var(--soft);border-color:#cde5df}} .value{{display:block;font-size:37px;line-height:1.15;font-weight:bold;letter-spacing:-.04em;color:var(--accent)}} .metric small{{display:block;font-size:11px;margin-top:6px;color:var(--muted)}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:11px}} .card{{border:1px solid var(--line);border-radius:8px;padding:12px 13px;font-size:12px;line-height:1.7}} .card p{{margin:0}} .num{{color:var(--accent);margin-right:6px;font-size:11px}} .file{{display:block;font-size:10px;color:var(--muted);margin-top:6px;overflow-wrap:anywhere}}
table{{width:100%;border-collapse:collapse;font-size:12px;line-height:1.55}} th{{text-align:left;color:var(--muted);font-size:10px;background:#f3f6f8;font-weight:600}} td,th{{padding:9px 9px;border-bottom:1px solid var(--line)}} td:not(:first-child){{white-space:nowrap}} .selected{{background:var(--soft);font-weight:bold}} .cell-note{{display:block;font-size:10px;font-weight:normal;color:var(--muted)}}
.note{{font-size:10.5px;color:var(--muted);line-height:1.7}} .callout{{border-left:3px solid var(--accent);padding:10px 13px;background:#f1f7f5;font-size:12px;margin:13px 0}}
.checks{{display:grid;grid-template-columns:1fr 1fr;gap:8px 17px;font-size:11.5px;line-height:1.7}} .checks p{{margin:0}} .checks b{{color:var(--accent)}}
.command{{background:#142c3c;border-radius:7px;color:#e8f3f4;padding:10px 13px;margin:8px 0;font:10.5px/1.6 ui-monospace,"DejaVu Sans Mono",monospace;white-space:pre-wrap;overflow-wrap:anywhere}} .command-label{{color:#a4c6c5}}
.footer{{border-top:1px solid var(--line);padding-top:8px;margin-top:16px;display:flex;justify-content:space-between;gap:10px;font-size:9px;color:var(--muted)}} a{{color:#186f73;text-decoration:none}} code{{font-family:ui-monospace,"DejaVu Sans Mono",monospace;font-size:.92em}} .sources{{font-size:9.5px;line-height:1.7;margin-top:10px;color:var(--muted)}}
.verification h2{{margin:14px 0 8px}} .verification .lead{{font-size:12px;line-height:1.7}} .verification .checks{{font-size:11px;line-height:1.6}} .verification .command{{padding:8px 13px}} .verification .note{{line-height:1.6}}
@page{{size:A4;margin:0}} @media print{{body{{background:#fff}} .page{{width:210mm;min-height:297mm;margin:0;box-shadow:none;break-after:page;page-break-after:always}} .page:last-child{{break-after:auto;page-break-after:auto}} *{{-webkit-print-color-adjust:exact;print-color-adjust:exact}} a{{color:inherit}}}}
@media screen and (max-width:820px){{.page{{width:calc(100% - 24px);min-height:0;margin:12px;padding:24px}} h1{{font-size:26px}} .cards{{grid-template-columns:1fr}} .hero{{gap:8px}} .metric{{padding:12px}} .value{{font-size:30px}} .checks{{grid-template-columns:1fr}} td,th{{padding:8px 5px;font-size:10px}}}}
</style></head><body>
<main class="page" aria-label="第1页：修改内容与历史对照">
<div class="meta"><span>LINGBOT-VLA · ROBOTWIN · 8 × ASCEND 910B1</span><span>{date} · 01 / 02</span></div>
<div class="eyebrow">第二轮优化 · 已接入默认入口</div>
<h1>减少等待与重复计算，<br>让同样的训练更快完成。</h1>
<p class="lead">在第一轮 GMM 优化基础上，采用注意力融合、监控与优化器合批，并利用显存余量关闭动作层重计算。本次统一主入口，再完成 <b>{updates} 步八卡训练验证</b>。</p>
<div class="hero"><div class="metric"><span class="value">+{gain:.0f}%</span><small>有效样本吞吐提升 · 历史同条件 A/B</small></div><div class="metric"><span class="value">−{reduction:.1f}%</span><small>每步实际耗时 · {old['wall_s']:.2f} → {new['wall_s']:.2f} 秒</small></div></div>
<h2>具体修改了什么</h2>
<div class="cards">
<div class="card"><h3><span class="num">01</span>联合注意力使用 FP32 SDPA 接口</h3><p>融合注意力计算，减少中间张量与调度开销；保留原有可见性规则、GQA 及全遮罩行行为。</p><a class="file" href="../../lingbotvla/ops/joint_attention.py">ops/joint_attention.py</a></div>
<div class="card"><h3><span class="num">02</span>MoE 监控计数合并回传</h3><p>将逐层读取计数改为一次合批回传，减少 CPU 等待 NPU 的同步次数；监控统计保持原语义。</p><a class="file" href="../../lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py">modeling_lingbot_vla_v2.py</a></div>
<div class="card"><h3><span class="num">03</span>既有 AdamW 分支启用 foreach</h3><p>相同更新操作按张量批量提交。Muon 算法和参数分组沿用原方案，并处理续训后的开关恢复。</p><a class="file" href="../../lingbotvla/optim/optimizer.py">optim/optimizer.py</a></div>
<div class="card"><h3><span class="num">04</span>关闭动作层梯度重计算</h3><p>注意力优化节约显存后，保留动作层中间结果供反向使用，减少重复前向计算；可一键恢复省显存模式。</p><a class="file" href="../../train_robotwin_npu.sh">train_robotwin_npu.sh</a></div>
</div>
<h2>提升多少：已有 60 步同条件对照</h2>
<table><thead><tr><th>配置</th><th>每步实际耗时</th><th>样本 / 秒</th><th>rank 0 峰值分配</th></tr></thead><tbody>{historical_rows}</tbody></table>
<p class="note">各组运行 60 步，统计第 13–60 步；采用较快的 GMM 基线复测作比较。均为八卡、每卡 12 样本、全局 batch 96，保留 Muon、训练精度策略、teacher 与完整学习率调度。速度按日志时间戳间隔计算，采集期间未运行原生 profiler。</p>
<div class="callout"><b>这次补齐的是默认入口和更长复验。</b>直接运行主脚本即可启用已验证组合；兼容入口统一委托主脚本，避免两套默认参数逐渐分叉。</div>
<div class="footer"><span>历史性能来源：<a href="../20261016_speed_round2/final_summary.json">第二轮 A/B 汇总</a> · <a href="../20261016_speed_round2/report.html">原理与数值验证明细</a></span><span>性能结果 / 1</span></div>
</main>
<section class="page verification" aria-label="第2页：默认入口复验与使用方法">
<div class="meta"><span>默认入口集成验收 · 真实八卡训练</span><span>{date} · 02 / 02</span></div>
<h1>默认配置通过 {updates} 步复验。</h1>
<p class="lead">本次由主脚本自身提供性能默认值，运行连续 {updates} 次更新，退出码 0；所有日志数值有限。源码快照与当前文件哈希一致，启动配置通过 <b>{launch['check_count']} 项检查</b>。</p>
<h2>本次实测：独立列出三个窗口</h2>
<table><thead><tr><th>统计窗口与用途</th><th>每步实际耗时</th><th>全局样本 / 秒</th></tr></thead><tbody>{current_rows}</tbody></table>
<p class="note">本次 rank 0 峰值已分配显存 <b>{peak:.2f} GiB</b>。第 115 步起已跨过 114 步学习率预热；调度总长度仍为 {completion['lr_horizon']} 步。实际耗时包含窗口内迭代间隔，剔除初始化；时间戳精度为秒。</p>
<div class="callout">本次用于确认<b>真实默认配置和跨预热运行稳定性</b>。它与历史实验分时段执行，不能把两次结果的波动解释为新的优化收益；页面 1 的 +36% 来自已有同条件 A/B。</div>
<h2>正确性与适用范围</h2>
<div class="checks">
<p><b>✓ 配置一致性</b><br>审计确认模型、数据、优化器、精度、batch 和 loss 配置与已验证组合一致；预取参数同为 4。</p>
<p><b>✓ 数值与梯度</b><br>已有组件测试覆盖注意力输出及梯度、监控统计、八卡 AdamW 更新和恢复开关；本次日志无 NaN/Inf。</p>
<p><b>✓ 使用条件</b><br>当前证据覆盖 8 × 910B1、每卡 12 样本、仅训练动作专家。单独使用 YAML 仍保留 eager 与重计算。</p>
<p><b>待完整训练验证</b><br>{updates} 步不等于完整 epoch 或机器人成功率验证；本次未进行完整大小的 DCP 保存/恢复，也未验证所有卡的峰值显存。</p>
</div>
<h2>直接使用、节省显存与回退</h2>
<pre class="command"><span class="command-label"># 默认训练；工作目录为仓库根目录</span>
bash train_robotwin_npu.sh
<span class="command-label"># 更省显存：保留同一组合，开启动作层重计算</span>
ENABLE_GRAD_CKPT=true bash train_robotwin_npu.sh</pre>
<pre class="command"><span class="command-label"># 回退到第一轮 GMM 配置</span>
LINGBOT_JOINT_ATTN_BACKEND=eager \\
LINGBOT_MOE_MONITOR_BATCH_COUNTS=0 \\
LINGBOT_ADAMW_FOREACH_NPU=0 ENABLE_GRAD_CKPT=true \\
bash train_robotwin_npu.sh</pre>
<p class="note">默认实验名为 <code>{escape(completion['new_default_run_name'])}</code>。旧实验产物保留；需要恢复旧目录时显式设置 <code>RUN_NAME=robotwin_1ep_fast</code>。主脚本保留保存/恢复能力，关闭梯度重计算不影响训练存档；本次短跑另行关闭了保存/上传。</p>
<div class="sources">可复核证据：<a href="completion.json">本次完成与窗口数据</a> · <a href="config_audit.json">有效配置审计</a> · <a href="launcher_validation.json">启动与回退检查</a> · <a href="../20261016_speed_round2/final_summary.json">历史 A/B 原始汇总</a><br>复验日志目录：<code>{escape(completion['run'])}</code></div>
<div class="footer"><span>生成依据：已完成的训练记录与审计 JSON · 无外部字体或脚本请求</span><span>使用与验证 / 2</span></div>
</section></body></html>'''
    return document.replace("/* EMBEDDED_FONT */", embedded_font(document))


def main():
    read = lambda name: json.loads((HERE / name).read_text())
    completion = read("completion.json")
    audit = read("config_audit.json")
    launch = read("launcher_validation.json")
    historical = read(completion["historical_controlled_ab"]["source"])
    output = HERE / "report.html"
    output.write_text(build_report(completion, audit, launch, historical), encoding="utf-8")
    print(json.dumps({"report": str(output), "bytes": output.stat().st_size,
                      "updates": completion["updates"], "status": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
