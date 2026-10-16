import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lingbot_npu_moe_report_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

P = Path(__file__).resolve().parent
repo = P.parents[1]
D = json.loads((P / "comparison.json").read_text())
G = json.loads((P / "fsdp_gradients.json").read_text())
root = Path(D["cases"]["torch"]["root"]).parent
rows = {name: [json.loads(line) for line in (root / name / "logs/loss.jsonl").read_text().splitlines()]
        for name in ["torch", "gmm"]}
font = Path("/tmp/lingbot_report_fonts/NotoSansSC-Regular.ttf")
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
fig.suptitle("LingBot-VLA · NPU MoE 优化的 8 卡完整模型对照", fontsize=19)
colors = {"torch": "#315EB5", "gmm": "#0B8C7E"}
labels = {"torch": "逐专家兼容实现", "gmm": "原生 GMM + 显式反向"}
ax = axes[0, 0]
for name in ["torch", "gmm"]:
    data = [r for r in rows[name] if 13 <= r["step"] <= 29]
    ax.plot([r["step"] for r in data], [r["step_time"] for r in data], "o-", label=labels[name], color=colors[name])
ax.set(title="稳定区间第 13–29 步（无 profiler 采集）", xlabel="训练步", ylabel="秒 / 步", ylim=(0, 8))
ax.legend()
ax = axes[0, 1]
for name in ["torch", "gmm"]:
    ax.plot([r["step"] for r in rows[name]], [r["loss"] for r in rows[name]], label=labels[name], color=colors[name], alpha=.8)
ax.set(title="相同初始权重、数据顺序和 LR 调度", xlabel="训练步", ylabel="总 loss")
ax.legend()
ax = axes[1, 0]
metrics = ["Aicore Usage Rate(%)", "Aivector Usage Rate(%)", "HBM Bandwidth Usage Rate(%)"]
for i, name in enumerate(["torch", "gmm"]):
    vals = [sum(card[key]["mean"] for card in D["resources"][name].values()) / len(D["resources"][name]) for key in metrics]
    bars = ax.bar([j + (i - .5) * .35 for j in range(3)], vals, width=.35, color=colors[name], label=labels[name])
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .5, f"{value:.1f}%", ha="center", fontsize=9)
ax.set(xticks=range(3), xticklabels=["AI Core", "Vector", "HBM 带宽"], ylabel="采样均值 %", title="卡 0/4 稳定区间采样（非 MFU）")
ax.legend()
ax = axes[1, 1]; ax.axis("off")
a, b = D["cases"]["torch"], D["cases"]["gmm"]
ax.text(0, .97, f"端到端吞吐提升 {D['speedup_wall']:.2f} 倍", fontsize=19, color=colors["gmm"], va="top")
ax.text(0, .82,
        f"• 实际步间隔：{a['steady_wall_seconds_per_update']:.2f} → {b['steady_wall_seconds_per_update']:.2f} s\n"
        f"• 吞吐：{a['global_samples_per_second']:.2f} → {b['global_samples_per_second']:.2f} 样本/s\n"
        f"• 峰值已分配显存：{a['peak_allocated_gib']:.2f} → {b['peak_allocated_gib']:.2f} GiB\n"
        "• 两种后端均完成 40 个更新，loss/梯度有限\n"
        "• 10 项 NPU 数值测试通过\n"
        "• 8 卡 FSDP2 + 检查点重计算梯度对照通过\n\n"
        "完整长跑与 RoboTwin 成功率尚未评测。",
        fontsize=12, linespacing=1.8, va="top")
for ax in axes.flat:
    ax.grid(alpha=.14)
fig.savefig(P / "optimization_overview.png", dpi=170)
fig.savefig(P / "optimization_overview.pdf")

lines = [
    "2026-10-16 NPU MoE 优化及验证",
    "",
    "默认 8 卡脚本和 NPU 非 EP fused 路径现使用原生 grouped GEMM + 显式一阶反向。保留 stacked 专家权重的形状、名称和现有检查点格式；路由器仍使用原来的 FP32 计算。",
    "实现：lingbotvla/ops/npu_moe.py。三次前向 GEMM 及输入梯度使用 npu_grouped_matmul；权重梯度使用 npu_grouped_matmul_add_ 的 FP32 累加，再恢复原计算 dtype 与连续布局。计数和 cumsum 留在 NPU；精确 FP32 专家 ID 排序替代整数 ArgSort 的 AICPU 回退。",
    "",
    "完整模型 A/B：相同预训练权重、随机种子、数据、8×12=96、Muon、FP32/BF16 策略、teacher 和梯度检查点；各跑 40 步，LR horizon 仍为 5717 步。速度比较仅使用第 13–29 步，排除初始化、profiler 预热/采集/导出和之后的通信等待。",
    f"  原实现：日志内均值 {a['steady_step_time_mean_s']:.4f} s/步，实际步间隔 {a['steady_wall_seconds_per_update']:.4f} s，吞吐 {a['global_samples_per_second']:.4f} 样本/s，峰值已分配显存 {a['peak_allocated_gib']:.2f} GiB。",
    f"  GMM：日志内均值 {b['steady_step_time_mean_s']:.4f} s/步，实际步间隔 {b['steady_wall_seconds_per_update']:.4f} s，吞吐 {b['global_samples_per_second']:.4f} 样本/s，峰值已分配显存 {b['peak_allocated_gib']:.2f} GiB。",
    f"  吞吐倍率 {D['speedup_wall']:.4f}，每步耗时降低 {D['step_time_reduction_pct']:.2f}%。",
    f"  40 步配对 loss 最大绝对差 {D['max_loss_abs_diff']:.6g}；梯度范数最大绝对差 {D['max_grad_norm_abs_diff']:.6g}。两轮 loss 与梯度均有限；该短跑不能证明完整训练的收敛或任务成功率。",
    "",
    "数值验证：10 项硬件测试覆盖随机路由、空专家、top1/top4、冻结输入或权重、实际训练尺寸、非重入检查点以及后端切换。输出和所有输入、路由及权重梯度与原实现对照通过。",
    "8 卡梯度验证：真实 FSDP2 分片、非重入检查点、同名参考权重严格载入；每个参数的完整梯度与各 rank 参考梯度的全局均值比较；专家和上游 FP32 路由器梯度均非零且有限，AdamW 为全部参数建立优化器状态。",
    json.dumps(G, ensure_ascii=False, indent=2),
    "",
    "稳定区间资源采样（卡 0/4；不等同 MFU 或全 epoch 平均）：",
    json.dumps(D["resources"], ensure_ascii=False, indent=2),
    "",
    "基线启动器说明：40 步训练和结束清理正常完成后，外层 shell 因运行期间源脚本更新造成读取偏移，报 unexpected EOF，原始 exit_code=2 已保留。比较使用的是错误发生前完整记录的训练步骤。GMM 启动器使用不可变快照，退出码为 0。详细审计见 baseline_launcher_note.json。",
    "",
    "默认运行：bash train_robotwin_npu.sh",
    "切回原实现：LINGBOT_NPU_MOE_IMPL=torch bash train_robotwin_npu.sh",
    "数值测试：OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_npu_moe.py",
    "8 卡梯度测试：OMP_NUM_THREADS=1 torchrun --nproc-per-node 8 --master-port 62519 tools/check_npu_moe_fsdp.py --output /cache/fsdp_gradients.json",
    "完整模型对照：bash tools/bench_robotwin_npu_moe.sh",
    f"原始日志、采样与优化后 trace：{root}",
    "",
    "限制：当前验证针对默认 910B1、torch_npu 2.6、非 EP、BF16 专家计算和一阶反向。未测试其他 Ascend 型号、专家并行、高阶梯度、完整 epoch 或 RoboTwin 成功率。",
]
(P / "analysis.txt").write_text("\n".join(lines) + "\n")
manifest = json.loads((P / "manifest.json").read_text())
manifest.update({"status": "validated", "gmm_completed_updates": 40, "gmm_launcher_exit_code": 0,
                 "fsdp_gradient_check_world_size": G["world_size"], "fsdp_gradient_check_passed": True,
                 "end_to_end_speedup": D["speedup_wall"]})
for path in manifest["source_sha256"]:
    manifest["source_sha256"][path] = hashlib.sha256((repo / path).read_bytes()).hexdigest()
(P / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Wrote optimization report, charts and final manifest")
