import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lingbot_npu_profile_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

P = Path(__file__).resolve().parent
repo = P.parents[1]
D = json.loads((P / "profile_summary.json").read_text())
F = json.loads((P / "findings.json").read_text())
B = json.loads((P / "moe_gmm_benchmark.json").read_text())["results"]
T = json.loads((P / "run_timing.json").read_text())
font = Path("/tmp/lingbot_report_fonts/NotoSansSC-Regular.ttf")
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
fig.suptitle("LingBot-VLA · 8×910B1 实测 profiling 与 MoE 优化实验", fontsize=20)
rank_names = list(D)
phase_keys = [("train.backward", "反向（含重计算/等待）", "#315EB5"),
              ("train.forward", "前向", "#0B8C7E"),
              ("train.optimizer", "优化器", "#D29231"),
              ("train.teachers", "teacher", "#8C6BB1")]
ax = axes[0, 0]
left = [0.] * len(D)
for key, label, color in phase_keys:
    values = [D[r]["phases"][key]["per_step_s"] for r in rank_names]
    ax.barh(rank_names, values, left=left, label=label, color=color)
    left = [a + b for a, b in zip(left, values)]
rest = [max(0., D[r]["host_step_mean_s"] - x) for r, x in zip(rank_names, left)]
ax.barh(rank_names, rest, left=left, label="其他", color="#C8CFD7")
ax.invert_yaxis(); ax.set(title="各 rank 耗时一致，反向约占 56%", xlabel="采集窗口内秒 / 步")
ax.legend(fontsize=8, loc="lower right")
ax = axes[0, 1]; left = [0.] * len(D)
for key, label, color in [("Computing", "计算执行", "#0B8C7E"),
                          ("Communication(Not Overlapped)", "未覆盖通信", "#D29231"),
                          ("Free", "设备时间线空档", "#C8CFD7")]:
    values = [D[r]["step_trace_mean_s"][key] for r in rank_names]
    ax.barh(rank_names, values, left=left, label=label, color=color)
    left = [a + b for a, b in zip(left, values)]
ax.invert_yaxis(); ax.set(title="设备计算约 1.92 秒，空档约 6.06 秒", xlabel="采集窗口内秒 / 步")
ax.legend(fontsize=8, loc="lower right")
ax = axes[1, 0]
names = ["原实现", "权重 unbind", "unbind + cat", "NPU GMM + 显式反向"]
values = [x["median"]["total_ms"] for x in B.values()]
bars = ax.barh(names, values, color=["#A6AFBA", "#8DAACA", "#315EB5", "#0B8C7E"])
for bar, value in zip(bars, values):
    ax.text(value + .35, bar.get_y() + bar.get_height() / 2, f"{value:.2f} ms", va="center")
ax.invert_yaxis(); ax.set(xlim=(0, 40), title="单层 MoE 前向 + 反向（固定随机路由）", xlabel="毫秒；无 FSDP / teacher / 优化器")
ax = axes[1, 1]; ax.axis("off")
ax.text(0, .98, "采集与验证范围", fontsize=15, weight="bold", va="top")
ax.text(0, .85,
        "• 相同训练配置：8 卡 × micro 12，global 96\n"
        "• 共跑 20 步，每 rank 采集第 13–16 步\n"
        f"• 无采集时 {T['warm_baseline_mean_seconds']:.2f} s/步；采集时 {T['profiled_mean_seconds']:.2f} s/步\n"
        "• profiler 带来约 30% 耗时扰动\n"
        "• MoE 的前向与重计算合计约 2.48 s/步\n"
        "• GMM 原型验证了随机、变化路由和空专家\n"
        "• 梯度最大相对 L2 误差 < 2e-4\n\n"
        "固定路由单层加速约 11.6 倍，不能外推整轮训练。\n"
        "下一步：验证 8 卡 FSDP、重计算、显存和恢复。",
        fontsize=12, linespacing=1.8, va="top")
for ax in axes.flat:
    ax.grid(axis="x", alpha=.15)
fig.savefig(P / "profile_overview.png", dpi=170)
fig.savefig(P / "profile_overview.pdf")

lines = [
    "2026-10-16 LingBot-VLA 默认 8 卡 NPU profiling 结果",
    "",
    "结论：最大阶段开销是反向传播（含重计算和等待）。优先优化 NPU MoE 的逐专家循环、细碎算子调度及其反向路径。数据加载不是本次窗口中的主要瓶颈。",
    "",
    "方法：使用原始预训练权重、原数据集和训练配置，8×12=96，Muon、FP32/BF16 设置和梯度检查点不变。独立短跑 20 步，第 13–16 步在全部 8 个 rank 采集 torch_npu.profiler Level1 CPU/NPU/HCCL。关闭 OBS 上传与检查点输出。",
    f"无采集的第 2–11 步均值 {T['warm_baseline_mean_seconds']:.4f} 秒，采集第 13–16 步均值 {T['profiled_mean_seconds']:.4f} 秒，额外开销 {T['profiling_time_increase_pct']:.1f}%。以下阶段数字是采集窗口内的测量值，不是无 profiler 的原始耗时。",
    "",
    "8 个 rank × 4 步的阶段均值：",
]
labels = {"train.backward": "反向", "train.forward": "前向", "train.optimizer": "优化器", "train.teachers": "teacher", "train.clip_grad": "梯度裁剪", "train.data_wait": "数据等待", "train.h2d": "H2D 下发", "train.metrics_reduce": "统计量归约"}
for row in F["phase_wall_times"]:
    lines.append(f"  {labels[row['phase']]}: {row['mean_seconds']:.6f} 秒/步，占 {row['share_pct']:.3f}%")
lines += [
    "",
    "MoE 专家范围每步调用 72 次（36 层正常前向 + 36 层检查点重计算），合计 2.4803 秒/步。该数字是前向和反向的子集，不能再与上表相加；实际 MoE 梯度计算未全部包含在该范围内。",
    "设备时间线均值：计算执行 1.9223 秒/步，未被计算覆盖的通信 1.2370 秒/步，Free 6.0568 秒/步。通信包含同步和等待，不能直接等同链路传输；Free 也不能直接等同某一种 CPU 开销。",
    "rank 0 的 4 步有 339528 个计算 kernel；62248 次 MatMulV2 平均仅 8.624 微秒。主机 aclnnMatmulGetWorkspaceSize 合计 4.0928 秒，MatMulV2_Tiling 合计 1.7364 秒，launch 合计 3.3339 秒。这些主机项可嵌套或位于不同线程，不可直接相加。",
    "rank 0 的 aten::matmul_backward 为 15661 次，Host Total 3.5343 秒；aten::_local_scalar_dense 为 18432 次，Host Total 2.1566 秒；empty_tensor 为 240502 次。细碎调度、标量读取和分配是具体优化线索，不应将整段 backward 都解释成大矩阵计算。",
    "",
    "解决方案实验（仅隔离单层 MoE；32 专家、top4、612 tokens、D=768、H=512、FP32 主权重转 BF16）：",
]
for key, value in B.items():
    e = value["errors_vs_baseline"]
    lines.append(f"  {key}: 前向 {value['median']['forward_ms']:.3f} ms，反向 {value['median']['backward_ms']:.3f} ms，合计 {value['median']['total_ms']:.3f} ms；最大相对 L2 误差 {max(x['relative_l2'] for x in e):.3g}")
lines += [
    "",
    "原生 npu_grouped_matmul 在当前 torch_npu 2.6 环境中直接调用不会产生 x/weight 梯度（能力探针已验证），因此不能直接替换训练代码。研究原型用显式 autograd backward：输入梯度走 grouped GEMM，权重梯度走 npu_grouped_matmul_add_ 的 FP32 累加，再转回原计算 dtype。",
    "原型在 NPU 内完成专家计数与分组，避免 counts.tolist()；0..31 的专家 ID 转成精确 FP32 值排序，避免整数 ArgSort 回退 AICPU。",
    "额外验证了变化随机路由、仅 4 个活跃专家且头尾/中间均有空组的情形。所有输出和梯度有限；三个 GMM 用例的最大相对 L2 误差小于 2e-4。详细时序和误差见对应 JSON。",
    "固定路由时单层 GMM 前后向约 2.89 ms，相比原实现约 33.49 ms，加速约 11.6 倍。变化路由测试是探索性微基准，候选按固定顺序运行，存在 tiling 缓存及主机状态影响；不能用其倍率预测训练吞吐。",
    "",
    "实施优先级：",
    "1. 优先将原生 grouped GEMM + 显式 backward 作为 NPU 专用候选后端；保留现有 stacked 专家权重和 checkpoint 格式。先验证 FSDP2 的 DTensor 梯度传递、梯度检查点重计算、路由/损失、峰值 HBM 与恢复，再以无 profiler 的固定数据短跑对比端到端吞吐。",
    "2. 若要分阶段推进，unbind + cat 是已有数值对照的较小改动，但其收益随活跃专家数变化，稀疏用例不保证加速。",
    "3. MoE 优化后再采集剩余 FSDP 通信和 Muon 热点，研究通信重叠与选择性重计算；FP32 eager attention 的融合作为独立候选，需保留 mask 语义并做数值/任务成功率验证。",
    "",
    "本轮完成的是 profiling、归因与隔离原型研究；默认训练计算后端仍使用原实现。原型单层加速不能换算为整轮训练加速，未进行完整训练或 RoboTwin 成功率评测。",
    "",
    "复现采集：bash tools/profile_robotwin_npu.sh",
    "汇总：python3 tools/analyze_npu_profile.py /cache/lingbot/profiling/robotwin_20261016_1530 --output reports/20261016_npu_profile",
    "微基准：OMP_NUM_THREADS=1 python3 tools/bench_npu_moe_candidates.py --native-gmm --output /cache/moe_benchmark.json",
    "原始 trace、日志和资源采样：/cache/lingbot/profiling/robotwin_20261016_1530（约 22 GiB）",
    "",
    "官方研究参考：",
    "https://github.com/Ascend/MindSpeed/blob/master/mindspeed/ops/gmm.py",
    "https://github.com/Ascend/MindSpeed/blob/master/docs/zh/ops/gmm.md",
    "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha002/devaids/Profiling/atlasprofiling_16_0033.html",
]
(P / "analysis.txt").write_text("\n".join(lines) + "\n")
manifest = json.loads((P / "run_manifest.json").read_text())
manifest.update({"exit_code": 0, "completed_steps": 20, "all_ranks_captured": 8,
                 "captured_steps_per_rank": 4, "checkpoints_written": 0,
                 "finite_loss_and_gradients": True,
                 "research_only_prototype": "tools/bench_npu_moe_candidates.py"})
for path in ["tools/profile_robotwin_npu.sh", "tools/analyze_npu_profile.py", "tools/bench_npu_moe_candidates.py", "lingbotvla/utils/training_profiler.py", "tasks/vla/train_lingbotvla.py"]:
    manifest["source_sha256"][path] = hashlib.sha256((repo / path).read_bytes()).hexdigest()
(P / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Wrote analysis.txt, profile_overview.png, profile_overview.pdf and final run manifest")
