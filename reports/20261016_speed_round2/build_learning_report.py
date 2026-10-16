#!/usr/bin/env python3
"""Build a beginner-friendly Chinese report from audited experiment results.

Reads JSON/patch/test artifacts; writes report.html only. No torch or NPU import.
Run again after analyze_round2.py refreshes round2_analysis.json.
"""

import argparse
import html
import json
import math
import re
from pathlib import Path


# Final conclusions are reviewed against completed experiments and the integrated entry point.
FINAL_REVIEW = {
    "adoption": "已合入监控统计合批、FP32联合注意力和AdamW批量更新；新增带中文注释的优化入口，默认启用三项并关闭动作重计算。原入口保留原默认开关。本轮没有启动新的完整epoch。",
    "combined": "7组有效实验各完成60步并正常退出。以较快的原代码复测为对照，最终方案3.54→2.60秒/步，每步时间减少26.5%，吞吐27.11→36.86样本/秒（增加36.0%）；rank 0峰值已分配显存53.31→50.61 GiB。保留重计算时为3.06秒/步、46.68 GiB，可作为更省显存的选项。",
    "launch_command": "cd /home/ma-user/work/lingbot-vla-v2\n# 先预览实际设置，不启动训练\nDRY_RUN=1 bash train_robotwin_npu_optimized.sh\n# 正式训练时执行；本轮尚未执行此命令\nbash train_robotwin_npu_optimized.sh\n# 如需更省显存，用下面这条代替上面的正式启动命令\nENABLE_GRAD_CKPT=true bash train_robotwin_npu_optimized.sh",
    "rollback_command": "# 退回本轮开始时已有的GMM基线；沿用原学习率、batch和优化器算法\nLINGBOT_JOINT_ATTN_BACKEND=eager LINGBOT_MOE_MONITOR_BATCH_COUNTS=0 LINGBOT_ADAMW_FOREACH_NPU=0 ENABLE_GRAD_CKPT=true bash train_robotwin_npu_optimized.sh",
    "next_validation": "当前证据支持在本机8×910B1、每卡12样本、仅训练动作专家的配置下采用此性能方案。60步仍处在学习率预热期；损失和梯度没有NaN/Inf，轨迹接近但并非逐位相同，长期收敛和机器人任务成功率仍需完整训练与评测确认。增大batch或解冻视觉/语言模型后，需要重新检查显存。",
}

HERE = Path(__file__).resolve().parent
OP_COMMIT = "480bc4311abb3be02a5e204caf3890bc02e32e10"
OP_URL = f"https://github.com/Ascend/op-plugin/blob/{OP_COMMIT}/op_plugin/ops/opapi/"
LABELS = {
    "baseline": "本轮基准：已有 MoE 合批",
    "baseline_repeat": "原代码复测（控制波动）",
    "monitor": "只改监控查询",
    "attention": "只改联合注意力",
    "adamw_foreach": "只改 AdamW 批量更新",
    "combined": "组合实验（保留重计算）",
    "combined_no_checkpoint": "同一组合，关闭动作重计算",
}


def esc(value):
    return html.escape(str(value), quote=True)


def read_json(path):
    if not path.exists():
        return None
    # Malformed existing evidence is an error, not silently treated as missing.
    return json.loads(path.read_text(encoding="utf-8"))


def at(data, *keys, default=None):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def num(value, digits=2, unit=""):
    return f"{value:.{digits}f}{unit}" if finite(value) else "—"


def small(value):
    return f"{value:.3g}" if finite(value) else "—"


def link(path, label):
    path = Path(path) if not str(path).startswith("https://") else str(path)
    if isinstance(path, Path):
        target = path if path.is_absolute() else HERE / path
        if not target.exists():
            return f'<span class="muted">{esc(label)}（尚未生成）</span>'
    return f'<a href="{esc(path)}">{esc(label)}</a>'


def table(headers, rows):
    head = "".join(f"<th scope='col'>{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def role_of(name, case, baseline):
    if name == baseline:
        return "baseline"
    if name.startswith("baseline_repeat"):
        return "baseline_repeat"
    if name.startswith("combined_no_checkpoint"):
        return "combined_no_checkpoint"
    if name in LABELS:
        return name
    settings = at(case, "provenance", "manifest_settings", default={})
    flags = [settings.get("monitor") == "1",
             settings.get("attention") == "sdpa_fp32",
             settings.get("adamw_foreach") == "1"]
    if (sum(flags) >= 2 or "combined" in name) and settings.get("gradient_checkpointing") == "false":
        return "combined_no_checkpoint"
    if sum(flags) >= 2 or "combined" in name:
        return "combined"
    if flags[0] or "monitor" in name:
        return "monitor"
    if flags[1] or "attention" in name or "sdpa" in name:
        return "attention"
    if flags[2] or "foreach" in name or "adamw" in name:
        return "adamw_foreach"
    return None


def status(case):
    if case is None:
        return '<span class="pending">未完成：尚无汇总结果</span>'
    if case.get("eligible"):
        return '<span class="good">已完成，可进入比较</span>'
    if case.get("exit_code") not in (None, 0):
        return f'<span class="bad">失败：退出码 {esc(case["exit_code"])}</span>'
    return f'<span class="pending">未完成：已有 {esc(case.get("updates", 0))} 步</span>'


def checkpoint_pair(data):
    cases = data.get("cases", {})
    baseline = data.get("baseline", "gmm_baseline_v2")
    def find(role):
        found = [(n, c) for n, c in cases.items() if role_of(n, c, baseline) == role]
        return found[-1] if found else (None, None)
    on_name, on = find("combined")
    off_name, off = find("combined_no_checkpoint")
    valid = all(c and c.get("eligible")
                and at(c, "windows", "13-60", "usable_for_speed_comparison") for c in (on, off))
    before = at(on, "windows", "13-60", "wall_seconds_per_update")
    after = at(off, "windows", "13-60", "wall_seconds_per_update")
    valid = bool(valid and finite(before) and before > 0 and finite(after))
    return {"on_name": on_name, "off_name": off_name, "on": on, "off": off,
            "valid": valid, "before": before, "after": after,
            "reduction_pct": 100 * (before - after) / before if valid else None}


def render_result_tables(data):
    cases = data.get("cases", {})
    comparisons = data.get("comparisons", {})
    baseline = data.get("baseline", "gmm_baseline_v2")
    display = []
    assigned = set()
    for role in LABELS:
        matches = [(name, c) for name, c in cases.items() if role_of(name, c, baseline) == role]
        if not matches:
            display.append((role, None, None))
        for name, case in matches:
            display.append((role, name, case))
            assigned.add(name)
    display += [(None, name, case) for name, case in cases.items() if name not in assigned]
    results, numerical, resources = [], [], []
    for role, name, case in display:
        label = LABELS.get(role, "其他/失败尝试")
        label += f'<small>{esc(name)}</small>' if name else ""
        window = at(case, "windows", "13-60", default={})
        tail = at(case, "windows", "31-60", default={})
        usable = bool(case and case.get("eligible") and window.get("usable_for_speed_comparison"))
        compare = comparisons.get(name, {})
        pct = at(compare, "windows", "13-60", "wall_time_reduction_pct") if usable else None
        if role == "combined_no_checkpoint":
            pct = checkpoint_pair(data)["reduction_pct"]
        reduction = "基准" if role == "baseline" and usable else "—"
        if finite(pct):
            reduction = f'<strong>{abs(pct):.1f}% {"更省时" if pct >= 0 else "更耗时"}</strong>'
            if role == "baseline_repeat":
                reduction += "<small>原代码重复测量，反映波动</small>"
            elif role == "combined_no_checkpoint":
                reduction += "<small>对照：同一组合，保留重计算</small>"
            else:
                reduction += "<small>对照：本轮基准</small>"
        results.append([label, status(case), num(window.get("wall_seconds_per_update") if usable else None, 3, " 秒"),
                        reduction, num(window.get("global_samples_per_second") if usable else None, 2),
                        num(tail.get("wall_seconds_per_update") if usable else None, 3, " 秒")])
        if not usable:
            continue
        metrics = at(compare, "windows", "13-60", "paired_numerics", "metrics", default={})
        loss_delta = at(metrics, "loss", "max_abs_diff", "abs_diff")
        grad_delta = at(metrics, "grad_norm", "max_relative_diff", "relative_diff")
        numerical.append([esc(name), "是" if case.get("all_required_metrics_finite") else "否",
                          num(at(window, "loss", "mean"), 5), small(loss_delta),
                          num(100 * grad_delta if finite(grad_delta) else None, 2, "%")])
        cards = at(window, "resources", "cards", default={})
        def card_pair(field):
            return " / ".join(num(at(cards, card, field, "mean"), 1, "%") for card in ("0", "4"))
        resources.append([esc(name), card_pair("Aicore Usage Rate(%)"), card_pair("HBM Usage Rate(%)"),
                          card_pair("HBM Bandwidth Usage Rate(%)"), num(case.get("peak_allocated_gib"), 2)])
    return (table(["实验", "状态", "每步实际时间 ↓", "相对指定对照", "全机样本/秒 ↑", "31–60 步复核 ↓"], results),
            table(["实验", "loss/梯度记录均有限", "平均 loss", "逐步 loss 最大绝对差", "梯度范数最大相对差"], numerical),
            table(["实验", "AI Core 0/4 卡", "HBM 占用 0/4 卡", "HBM 带宽 0/4 卡", "rank 0 峰值已分配 GiB"], resources))


def validation_summary():
    monitor = read_json(HERE / "monitor_cpu_validation.json") or {}
    attn = read_json(HERE / "attention_npu_checks_v2.json")
    attn_text = "NPU 数值验证尚未完成。"
    benchmark = ""
    if attn:
        checks = attn.get("checks", [])
        attn_text = f'{len(checks)} 项 NPU 用例，结果：{"通过" if attn.get("passed") else "未通过"}。'
        maxima = [m.get("max_abs") for c in checks for m in c.get("metrics", {}).values() if finite(m.get("max_abs"))]
        if maxima:
            attn_text += f"输出及 Q/K/V 梯度的最大绝对差为 {max(maxima):.3g}。"
        eager = at(attn, "benchmarks", "eager", "median_forward_backward_ms")
        sdpa = at(attn, "benchmarks", "sdpa", "median_forward_backward_ms")
        if finite(eager) and finite(sdpa):
            benchmark = (f"真实形状的单层前向＋反向：{eager:.3f} → {sdpa:.3f} 毫秒。"
                         "这只计一个注意力模块，整模型结果应看上面的短跑对照表。")
    npu_files = sorted(HERE.glob("adamw_foreach_npu_validation*.rank*.json"))
    npu = [read_json(p) for p in npu_files]
    resume = read_json(HERE / "adamw_foreach_resume_cpu.json")
    combined = read_json(HERE / "combined_cpu_validation.json")
    optimizer_text = f"NPU 已生成 {len(npu)}/8 个 rank 的验证报告。"
    if len(npu) == 8:
        optimizer_text += "全部通过。" if all(x.get("passed") for x in npu) else "存在未通过项，需查原始记录。"
    else:
        optimizer_text += "多卡验证尚未齐全；CPU 通过不代表 NPU 已通过。"
    rows = [
        ["监控批量查询", f'{len(monitor.get("cases", []))} 项 CPU 边界用例；'
         "验证监控输出、辅助损失及梯度。原36次标量读取改成一次长度36的计数传输。", link("monitor_cpu_validation.json", "监控验证")],
        ["FP32 接口注意力", esc(attn_text + benchmark), link("attention_npu_checks_v2.json", "NPU 修正后验证")],
        ["AdamW foreach", esc(optimizer_text) + " CPU还检查参数、动量状态、跳过梯度与保存恢复。",
         link("adamw_foreach_cpu_final24.rank0.json", "CPU 最终验证") + " · " +
         " · ".join(link(p.name, p.name) for p in npu_files)],
    ]
    resume_text = "续训开关验证尚未生成。"
    if resume:
        resume_text = (f'{len(resume.get("checks", []))} 项 CPU 切换用例，'
                       f'结果：{"通过" if resume.get("passed") else "未通过"}。'
                       "恢复后遵守本次 foreach 开关；学习率、betas、衰减、步数和动量状态保持。")
    rows.append(["AdamW 续训开关", esc(resume_text), link("adamw_foreach_resume_cpu.json", "续训兼容验证")])
    combined_text = "组合快照的 CPU 组件验证尚未完成。"
    if combined:
        combined_text = (f'组合快照 CPU 组件检查：{"通过" if combined.get("passed") else "未通过"}。'
                         f'监控{at(combined, "monitor", "cases", default="—")}项、'
                         f'注意力{at(combined, "attention", "cases", default="—")}项、'
                         f'AdamW更新{at(combined, "adamw_postload", "cases", default="—")}项、'
                         f'续训开关{at(combined, "adamw_resume", "cases", default="—")}项。'
                         "这不是组合方案的8卡整模型短跑验证；短跑状态见主表。")
    rows.append(["组合源码快照", esc(combined_text), link("combined_cpu_validation.json", "组合快照 CPU 验证")])
    return table(["修改", "已检查什么", "证据"], rows)


def attention_observation(data):
    baseline = at(data, "cases", data.get("baseline", "gmm_baseline_v2"), default={})
    candidates = [(name, c) for name, c in data.get("cases", {}).items()
                  if role_of(name, c, data.get("baseline")) == "attention" and c.get("eligible")]
    if not candidates:
        return '<p class="muted">联合注意力整模型短跑对照尚未完成；当前不报告整步收益或显存变化。</p>'
    name, candidate = candidates[-1]
    before = at(baseline, "windows", "13-60", "wall_seconds_per_update")
    after = at(candidate, "windows", "13-60", "wall_seconds_per_update")
    before_memory = baseline.get("peak_allocated_gib")
    after_memory = candidate.get("peak_allocated_gib")
    return (f'<div class="callout"><p><b>单独改注意力的整模型短跑记录：</b>'
            f'{num(before, 3)} → {num(after, 3)} 秒/步；'
            f'<b>rank 0 峰值已分配显存</b> {num(before_memory)} → {num(after_memory)} GiB。'
            '显存指标只针对rank 0的allocated峰值，不能解释为每张卡已经释放了同样大小的物理显存。</p></div>')


def evidence_strength(data):
    """Use the faster unchanged-code run to qualify small one-off gains."""
    baseline = data.get("baseline", "gmm_baseline_v2")
    observed = {}
    for name, case in data.get("cases", {}).items():
        window = at(case, "windows", "13-60", default={})
        value = window.get("wall_seconds_per_update")
        if case.get("eligible") and window.get("usable_for_speed_comparison") and finite(value) and value > 0:
            observed[role_of(name, case, baseline)] = value
    required = ("baseline", "baseline_repeat", "monitor", "attention", "adamw_foreach")
    if not all(key in observed for key in required):
        return '<p class="muted">单项收益还需结合原代码复测判断；相关组齐全后再给出数值比较。</p>'
    first, repeat = observed["baseline"], observed["baseline_repeat"]
    faster = min(first, repeat)
    drift = 100 * abs(first - repeat) / first
    reductions = {key: 100 * (faster - observed[key]) / faster
                  for key in ("monitor", "attention", "adamw_foreach")}
    return (f'<div class="callout"><b>小幅提速，要先和自然波动比较。</b>'
            f'<p>相同原代码两次测得 {num(first, 3)} 和 {num(repeat, 3)} 秒/步，相差约 {drift:.1f}%。'
            f'改用较快的原代码结果作对照，单独监控查询只省 {reductions["monitor"]:.1f}%，'
            f'单独注意力只省 {reductions["attention"]:.1f}%。'
            '这两项整步时间变化与已观察到的运行波动接近，不能把表里的小幅改善全部认定为稳定收益。'
            '注意力的显存下降更明确，也为关闭重计算留出了空间。</p>'
            f'<p>AdamW 批量更新相对较快原代码仍省 {reductions["adamw_foreach"]:.1f}% 时间，'
            '本轮提速证据更明确。只有两次原代码测量、每个候选一次短跑，尚不足以给出长期速度范围或统计显著性结论。'
            '顶部最终方案也采用较快原代码作对照，避免只挑较慢基线报告收益。</p></div>')


def checkpoint_observation(data):
    pair = checkpoint_pair(data)
    if not pair["valid"]:
        return ('<p class="pending"><b>关闭重计算的增量收益：待验证。</b>'
                '需要“组合＋保留重计算”和“同一组合＋关闭重计算”都完成60步且满足比较条件，'
                '再比较第13–60步。排队中或尚未完成时，不报告提速结果。</p>')
    pct = pair["reduction_pct"]
    return (f'<div class="callout"><b>只关闭重计算的增量：</b>'
            f'{num(pair["before"], 3)} → {num(pair["after"], 3)} 秒/步，'
            f'{abs(pct):.1f}% {"更省时" if pct >= 0 else "更耗时"}。'
            f'<p>rank 0峰值已分配显存：{num(pair["on"].get("peak_allocated_gib"))} → '
            f'{num(pair["off"].get("peak_allocated_gib"))} GiB。'
            '这是同一组合代码、只改变激活重计算开关的对照；短跑结果不能代替长跑稳定性验证。</p></div>')


def previous_section():
    previous = read_json(HERE / "previous_ab_audit.json") or {}
    original = at(previous, "cases", "torch", "wall_s_per_step_13_29")
    optimized = at(previous, "cases", "gmm", "wall_s_per_step_13_29")
    pair = f"{num(original)} → {num(optimized)} 秒/步"
    return f"""<div class="callout"><b>此前的 MoE 改进：{pair}</b>
    <p>这是此前独立 A/B 的第 13–29 步窗口。把许多专家的小矩阵计算合批后，观察到明显提速。
    本轮已经以这项改进为起点，重新测量基准约 3.65 秒/步。<b>跨轮数字的变化不能记作某个新候选的收益。</b></p>
    <p class="muted">旧后端已有40步训练记录，但启动脚本最终退出码为2；此处仅引用已完成的训练窗口。
    旧实验还存在配置记录和源码快照的局限，见 {link('previous_ab_audit.json', '此前 A/B 独立审计')}。</p></div>"""


def failure_section(data):
    rows = []
    for name, case in data.get("cases", {}).items():
        if case.get("exit_code") not in (None, 0):
            reason = "未进入速度收益比较。"
            if name == "gmm_baseline":
                reason = "实验快照遗漏归一化输入文件，训练前失败；补齐快照后使用新目录重跑。原数据未改。"
            rows.append([esc(name), status(case), esc(reason)])
    rows += [
        ["BF16 注意力", "本轮未采纳", "CPU 大 logit 压力用例中，Q/K 梯度相对 L2 差约3.6%/3.9%，未通过预设门槛；没有为通过而调高门槛。"],
        ["注意力第一次 NPU 验证", "验证规则已修正，原记录保留", "只允许看一个 key 时，理想 Q/K 梯度为0。约1e-6的舍入差被除以1e-12，产生无意义的巨大相对误差。改为仅零参考使用原绝对门槛1e-4，非零参考规则不变。"],
        ["Tensor/DTensor 混合 foreach", "保留兼容保护", "同一批次混合普通张量与分片张量会报错。候选检查真实参数组，这类组继续逐个更新；不更换优化算法或参数分组。"],
        ["续训覆盖 foreach 开关", "已修复并独立验证", "旧存档中的foreach设置会覆盖本次开关。加载完成后重新应用本次执行策略，同时保留学习率、步数、动量等训练状态。"],
        ["FSDP/通信、选择性重计算", "本轮尚无这些调整的完整对照", "已完成的重计算实验只比较当前动作层全部开/关；更细的选择性重计算、FSDP和通信改动尚未完成新A/B，不计入本轮收益。"],
    ]
    return table(["问题/候选", "处理状态", "原因与学习点"], rows)


def build(data):
    results, numerics, resources = render_result_tables(data)
    updated = data.get("generated_at", "结果尚未汇总")
    artifacts = [
        ("round2_analysis.json", "本轮完整汇总"), ("analyze_round2.py", "统计与窗口计算脚本"),
        ("run_case.py", "隔离短跑启动器"), ("launcher_fairness_audit.json", "可比性审计"),
        ("baseline_repeat_audit.json", "原代码复测与自然波动审计"),
        ("final_experiment_audit.json", "最终实验独立复核"), ("final_code_review.json", "最终代码审阅"),
        ("integration_manifest.json", "主代码合入清单"), ("final_launcher_validation.json", "最终启动与回退验证"),
        ("final_resource_state.json", "实验结束后的进程状态"), ("final_npu_smi.txt", "实验结束后的八卡资源快照"),
        ("snapshot_manifest.json", "源码快照清单"), ("monitor_candidate.patch", "监控补丁"),
        ("check_monitor_candidate.py", "监控验证脚本"), ("attention_candidate.patch", "注意力补丁"),
        ("test_attention_candidate.py", "注意力验证脚本"), ("attention_npu_checks.json", "注意力原始失败记录"),
        ("attention_error_rule_audit.txt", "零参考误差规则说明"), ("attention_cpu_checks.json", "含BF16压力失败的CPU记录"),
        ("adamw_foreach_candidate.patch", "AdamW批量更新补丁"), ("check_adamw_foreach.py", "AdamW验证脚本"),
        ("adamw_foreach_resume_cpu.json", "续训开关12项验证"), ("combined_cpu_validation.json", "组合快照CPU组件验证"),
        ("checkpoint_scope_semantics_audit.json", "重计算作用范围与专家计数审计"),
        ("muon_trace_readonly_audit.json", "优化器瓶颈审计"),
    ]
    evidence = "<ul class='evidence'>" + "".join(f"<li>{link(p, label)}</li>" for p, label in artifacts) + "</ul>"
    launcher = HERE.parents[1] / "train_robotwin_npu_optimized.sh"
    checkpoint_default = "待核对"
    if launcher.exists():
        match = re.search(r"ENABLE_GRAD_CKPT:-([^}]+)", launcher.read_text(encoding="utf-8"))
        if match:
            checkpoint_default = match.group(1)
    learning_links = table(["代码入口", "读它时重点看什么"], [
        [link("../../train_robotwin_npu_optimized.sh", "train_robotwin_npu_optimized.sh"),
         "先读中文开关注释，理解每个环境变量对应哪项实验。报告生成时重计算默认值为 "
         f"<code>{esc(checkpoint_default)}</code>；当前设置及适用范围见最终采用方案。"],
        [link("../../lingbotvla/ops/joint_attention.py", "joint_attention.py"),
         "对照输入形状、mask方向、GQA，以及整行不可见时恢复原均值结果的处理。"],
        [link("../../lingbotvla/optim/optimizer.py", "optimizer.py"),
         "看哪些参数走Muon、哪些走既有AdamW；再看foreach开关、混合张量保护和存档加载后的开关恢复。"],
        [link("../../tests/test_moe_monitor_batch.py", "test_moe_monitor_batch.py"),
         "把测试当作小例子：正常计数、空计数、缺失监控状态时，优化前后应满足什么条件。"],
    ])
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>LingBot 训练提速：实验与学习记录</title>
<style>
:root{{--ink:#172e3b;--muted:#526675;--line:#dce5e9;--accent:#006a68;--paper:#fff;--bg:#edf2f4}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}}
main{{max-width:1100px;margin:30px auto;padding:42px 48px;background:var(--paper);border-radius:14px}}
h1{{font-size:32px;line-height:1.35;margin:0 0 12px}} h2{{font-size:23px;margin:36px 0 12px}}h3{{font-size:18px;margin:20px 0 8px}}
p{{margin:10px 0}}a{{color:var(--accent);text-underline-offset:3px}}small,.muted{{color:var(--muted);font-size:13px}}small{{display:block}}
.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:1px;font-size:13px}}.lede{{font-size:18px;max-width:900px}}
.callout{{background:#edf7f5;padding:18px 22px;border-left:4px solid var(--accent);border-radius:4px;margin:20px 0}}
.table-scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}}th,td{{border-bottom:1px solid var(--line);padding:12px 10px;text-align:left;vertical-align:top}}th{{background:#f0f5f6;font-size:13px;white-space:nowrap}}td:first-child{{min-width:130px}}tr:last-child td{{border-bottom:1px solid var(--line)}}
.good{{color:#087348}}.pending{{color:#86601b}}.bad{{color:#a13232}}.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}.step{{padding:16px;background:#f4f7f8;border-top:3px solid #87b9b1;border-radius:4px}}.step b{{display:block}}.step p{{font-size:14px}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}code{{background:#eef3f5;padding:2px 4px;border-radius:3px;font-size:.9em}}pre{{padding:16px;background:#172e3b;color:#f2f7f9;white-space:pre-wrap;overflow-wrap:anywhere;border-radius:5px;font-size:13px}}
.evidence{{columns:2;column-gap:30px;padding-left:20px}}.evidence li{{break-inside:avoid;margin:5px 0}}details{{border:1px solid var(--line);padding:14px 18px;margin:16px 0;border-radius:6px}}summary{{cursor:pointer;font-weight:600}}footer{{margin-top:32px;padding-top:18px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}}
@media(max-width:760px){{main{{margin:0;padding:24px 18px;border-radius:0}}h1{{font-size:27px}}.steps{{grid-template-columns:1fr 1fr}}.evidence{{columns:1}}}}@media print{{body{{background:white}}main{{margin:0;padding:0;max-width:none}}h2{{break-after:avoid}}tr,.step{{break-inside:avoid}}details{{display:block}}}}
</style></head><body><main>
<div class="eyebrow">LINGBOT · 8 × 昇腾 910B1 · 2026-10-16</div>
<h1>训练为什么慢，怎样一步步变快</h1>
<p class="lede">把优化拆成小实验：每次只改一件事，先检查算得对不对，再用整模型短跑测每步快了多少。你可以把这份记录当作代码修改说明，也可以按下面的解释学习训练的基本过程。</p>
<p class="muted">统计更新时间：{esc(updated)}（时间含 +08:00 时为北京时间）。本页是生成时的结果快照；重新运行报告脚本可读取最新汇总。缺失实验始终标为未完成。</p>

<h2>先看结果：本轮各项修改有没有省时间</h2>
<div class="callout"><p><b>最终采用：</b>{esc(FINAL_REVIEW["combined"])}</p></div>
<p>主要看<b>每秒完成多少有效训练样本</b>，以及固定batch下等价的每步时间。取第13–60步，共48次更新，目的是避开最初编译、首次执行等热身干扰。单项优化和组合方案与本轮基准比较；<b>“关闭动作重计算”单独与同一组合的保留重计算版本比较</b>。第31–60步提供另一段窗口，帮助判断效果是否稳定。</p>
{results}
<p class="muted">未完成、退出异常或受到 profiler 采集影响的窗口不参与收益比较。实际时间由第60步时间戳减去第12步时间戳，再除以48；日志按整秒记时，单步时间的量化误差小于约0.021秒。模型加载、首次编译、最初12步、保存检查点和上传时间另计。</p>
<p><b>这与学习率预热是两回事：</b>学习率预热是在训练刚开始时逐步增大学习率，避免一开始更新过猛。本配置调度长度5717步、预热比例0.02，约前114步都在学习率预热中；因此<b>60步短跑仍处于学习率预热</b>，没有验证后期专家分布、长期收敛或任务质量。</p>
<p class="muted">“原代码复测”再次运行本轮基准代码，用来观察机器状态、缓存和调度造成的自然波动；它不是新优化。候选的小幅变化需要结合这组对照判断，见 {link('baseline_repeat_audit.json', '原代码复测审计')}。</p>
{evidence_strength(data)}
{previous_section()}

<h2>读懂一步训练：96 个样本怎样更新一次模型</h2>
<p>本实验每张卡一次处理12个样本，8张卡合计<b>96个样本</b>。梯度累积次数为1，因此这里的一个 <code>step</code> 就是用这96个样本完成一次参数更新。比如每步3.6秒，全机吞吐约为96÷3.6≈26.7个样本/秒。</p>
<div class="steps"><div class="step"><b>1 · 前向：做一次题</b><p>模型读入图像、文字和状态，预测动作，并完成辅助任务。</p></div><div class="step"><b>2 · loss：给答案打分</b><p>损失衡量预测与目标的差距。它是学习信号，不能单独代表机器人实际任务成功率。</p></div><div class="step"><b>3 · 反向：找修改方向</b><p>梯度描述每个参数往哪个方向变化能减小损失。梯度范数是这许多方向数值的整体大小。</p></div><div class="step"><b>4 · optimizer：真正改参数</b><p>优化器结合梯度、学习率和历史状态更新参数。多卡训练还需要交换、汇总相关数据。</p></div></div>
<p><b>为什么要测完整一步：</b>某个算子即使快一倍，读数据、其他层、更新参数和卡间通信仍然要花时间，所以整步不一定快一倍。</p>

<h2>这次具体在改哪里</h2>
<h3>① 监控查询：把逐层问进度，改成集中问一次</h3>
<p>模型有36层 MoE。旧代码为了统计每层专家有没有收到数据，多次让 CPU 等待 NPU 给出一个计数结果。像工人每做一点事，就被叫停汇报一次。</p>
<p>新代码把36层的计数集中取回，再生成原来的监控指标；不删除参与训练的辅助损失。这样减少往返等待，同时保留学习过程需要的信息。本轮已合入，但单独整步提速很小，仍需结合原代码复测波动理解。</p>
<h3>② 联合注意力：减少中间大表的搬运</h3>
<p>注意力让每个位置判断“其他哪些位置的信息值得参考”。原实现分开打分、计算比例、汇总信息；融合入口让 NPU 用专门算法完成这些步骤。实际序列长度337、32个查询头时，一张 FP32 得分表就约166 MiB。</p>
<p>32个查询头会共用8组 key/value 资料，这叫 <code>GQA</code>。新入口能直接处理这种共享关系，避免先复制完整资料。这里保持输入、输出及调用前位置编码的 FP32 类型，<b>没有启用 BF16 候选</b>。</p>
<p><b>遮罩（mask）就是“哪些信息可以看”。</b>动作、图像和任务位置之间原有的可见规则必须保留。还有一个容易漏掉的边界：某一行全部被遮住时，旧实现返回value信息的均值，新SDPA入口显式保留这个结果及对应梯度行为；不能为了提速顺手把它改成零或其他默认结果。</p>
<p class="muted">“FP32 接口”不等于已经证明每条内部硬件指令都使用完整 FP32。对应版本包装代码没有把 QKV 降成半精度，本机 CANN 也有 float 输入/中间类型模板；此次尚未用实际 tiling key 和硬件数学模式证明全部内部精度。结论依据应包括下面的数值对照。</p>
{attention_observation(data)}
<h3>③ AdamW foreach：同样的更新规则，一次安排更多参数</h3>
<p>当前优化器组合里，本来就有一部分参数由 AdamW 更新。旧路径逐个张量发出更新指令；<code>foreach</code> 把一批张量的相同操作一起提交，减少调度次数。</p>
<p>这项候选只改变现有 AdamW 分支的执行方式，<b>不把 Muon 换成 AdamW</b>。学习率、权重衰减、参数分组和 Muon 更新算法沿用原方案。混合普通张量与分片张量的组有兼容保护，仍使用原路径。</p>
<p><b>续训也要检查：</b>训练存档里带着当时的优化器开关，恢复时可能把今天想启用的设置盖回去。新增加载完成后的处理，先恢复原有学习状态，再采用本次配置的批量执行方式；因此从旧存档继续训练、或关闭开关回退，都需要有对应验证。</p>

<h2>显存很满，为什么计算单元还不忙</h2>
<p>可以把<b>显存理解成仓库、计算单元理解成工人</b>。模型权重、中间结果和优化器状态会占仓库；仓库装满，不代表工人一直在计算。若任务太碎、反复等待 CPU、搬数据或等其他卡，工人仍可能闲着。</p>
<p><b>让工人忙不是最终目标，让同样的训练任务更快完成才是。</b>减少重复计算后，利用率读数可能变化不大，甚至下降，但有效样本/秒仍可能提高。重算出来的中间结果不算新样本；不能为了追求100%的利用率而增加无用计算。</p>
{resources}
<p class="muted">这里只采样0号和4号卡，而且查询有先后、间隔约数秒。AI Core 数字是短窗读数，不是整轮平均效率，也不是理论算力利用率 MFU。HBM占用、HBM带宽和计算忙碌度是不同指标。rank 0峰值已分配显存只统计该rank，不等于各卡总占用或缓存预留量；融合后分配需求变小，缓存分配器也可能暂不归还显存。</p>
<h3>把省下的显存用于少做“重计算”</h3>
<p>训练原本可以把中间结果当作草稿留着，反向时直接使用。显存紧张时，就少留一部分草稿，需要时再算一遍；这样省内存，但增加计算。融合注意力省下内存后，可以测试保留更多草稿、减少重算，以内存余量换速度。</p>
<p><b>两个 checkpoint 不是一回事：</b><code>gradient checkpointing</code> 指本段的“计算草稿少留、需要时重算”；保存训练进度的 <code>checkpoint</code> 则是把模型、优化器和训练步数写进存档。<b>关闭前者不妨碍存档和恢复训练。</b>本轮性能短跑是否保存文件，仍由独立的保存开关控制。</p>
<p class="muted">本轮60步短跑关闭了保存与恢复，没有验证八卡真实存档的端到端恢复；AdamW跨开关恢复的结论来自下面单列的组件验证。</p>
{checkpoint_observation(data)}
<p>源码审计表明，在<b>本次只训练动作专家</b>的配置下，这个开关实际影响动作专家的36层。冻结的视觉/语言模型保持eval，teacher独立构建，联合注意力也位于被重算区域之外。这个结论依赖当前配置，不能直接推广到解冻视觉/语言模型或启用激活卸载的训练。</p>
<details><summary>关闭重算后，专家计数可能减半：为什么并非少训练一半样本</summary>
<p>现有计数器每执行一次专家前向就记一次，重算时也会再记。源码推导：每层全机专家分配次数，可能从包含重算的约39,168降到19,584。后一个数来自96个样本 × 每样本51个动作位置 × 每个位置分给4个专家。它统计的是<b>专家分配次数</b>，不是不同训练样本的数量；每次更新仍然使用96个样本。</p>
<p>如果重算选择的专家相同、每层计数统一缩放，归一化后的负载比例、偏差和死专家判断应保持一致；原始条形图高度会变化。接近top-k选择边界的浮点差异可能改变路由，因此“比例相同”有条件，不能无条件保证。</p>
<p>本配置的专家偏置更新速度为0、偏置居中关闭；用于训练的平衡损失读取另外计算的路由信息，不读取这个监控计数。因此计数重复本身不表示训练目标翻倍。本轮保留原始计数实现，避免把计数修正和速度实验混在一起。这些是源码与配置推导，详见 {link('checkpoint_scope_semantics_audit.json', '重计算与计数语义审计')}。</p>
</details>

<h2>怎样判断“跑快了，也没有明显算错”</h2>
{validation_summary()}
<p>单模块测试使用相同输入检查输出和梯度；优化器测试还检查更新后的参数与历史状态。整模型短跑则继续观察损失、梯度是否有限，以及同一步的差异。</p>
{numerics}
<p class="muted">上表数值差异来自分析器的第13–60步逐步配对结果，当前均以本轮基准为数值对照；关闭重算的单项速度收益则使用保留重算的组合作对照。“有限”只表示没有 NaN/Inf，不表示数值完全一致。梯度范数是一个汇总标量，不能代替全部梯度逐元素比较。短跑会受浮点运算顺序和多卡执行影响；数值测试通过也不等于长期收敛或机器人任务质量已经验证。</p>

<h2>失败和暂未采用的方向也要留下来</h2>
{failure_section(data)}
<p>保留失败能帮助区分代码问题、测试规则问题和实验环境问题。下次升级软件或改变数据形状时，可以复用这些边界用例。</p>

<h2>想边改边学，可以从这些代码开始</h2>
<p>先看启动入口的中文注释，再挑一个算子或测试读。建议每次只跟踪一项改动：它改了哪次等待或计算、哪些数学行为要保留、对应测试怎样证明这些条件。</p>
{learning_links}

<h2>最终采用方案与下一步验证</h2>
<div class="callout"><p><b>合入状态：</b>{esc(FINAL_REVIEW['adoption'])}</p><p><b>组合实验：</b>{esc(FINAL_REVIEW['combined'])}</p><p>{esc(FINAL_REVIEW['next_validation'])}</p></div>
<p>启动与省显存选项（已通过命令预览验证）：</p><pre>{esc(FINAL_REVIEW['launch_command'])}</pre>
<p>回退到本轮基线：</p><pre>{esc(FINAL_REVIEW['rollback_command'])}</pre>

<details><summary>比较方法与可复核证据</summary>
<p>本轮使用独立源码快照和新实验目录；同一初始化路径、随机种子、数据清单、全局batch96、学习率与5717步调度长度。性能短跑关闭profiler采集、检查点保存与上传。配置、数据清单和源码差异记录在审计文件中；没有逐batch数据指纹和完整初始权重哈希，所以不宣称逐位可复现。</p>
{evidence}
<p>报告生成命令：<code>python reports/20261016_speed_round2/build_learning_report.py</code>。它只读取已有证据、重建本页，不启动训练，也不使用 NPU。</p>
</details>
<details><summary>官方源码与精度说明：需要时再看</summary>
<p>安装的 torch_npu 2.6.0 提交为 <code>41d2476c22b22f484ad1f07b5a6d011200638b6d</code>，对应 op-plugin 固定提交为 <code>{OP_COMMIT}</code>。以下链接固定版本，方便以后对照。</p>
<ul><li>{link(OP_URL+'ScaledDotProductAttentionKernelNpuOpApi.cpp#L223', 'SDPA 分发条件，第223–257行')}：支持FP32、BNSD四维、bool二/四维mask和整倍数GQA；当前32/8头、D128符合这些条件。第98行转换mask方向。</li>
<li>{link(OP_URL+'FlashAttentionKernelNpuOpApi.cpp#L1667', 'FlashAttention 包装，第1667–1720行')}：QKV只变换存储格式，输出沿用query类型，再调用CANN；包装层未显式降为BF16/FP16。</li>
<li>本机CANN8.2.RC1的 <code>flash_attention_score.cpp:8347</code> 有带mask的BNSD float/float模板；它是源码证据，不是此次具体kernel与数学模式的完整证明。<code>inner_precise=0</code> 也不能单独当作“内部所有计算都为FP32”的保证。</li></ul>
</details>
<footer>阅读顺序建议：先看整模型短跑的有效样本/秒和每步时间，再看某项修改的原理，最后看验证和局限。优化的价值是同样的训练任务更省时间，且有足够证据支持继续使用。</footer>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=HERE / "round2_analysis.json")
    parser.add_argument("--output", type=Path, default=HERE / "report.html")
    args = parser.parse_args()
    data = read_json(args.analysis) or {"cases": {}, "comparisons": {}}
    args.output.write_text(build(data), encoding="utf-8")
    print(f"Wrote {args.output}; {len(data.get('cases', {}))} recorded cases; no NPU used.")


if __name__ == "__main__":
    main()
