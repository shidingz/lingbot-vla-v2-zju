"""One-page strategy and measured-throughput report; does not touch training."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lingbot_report_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
SNAPSHOT = json.loads((ROOT / "training_snapshot.json").read_text())
S = SNAPSHOT["summary"]
ROWS = SNAPSHOT["rows"]
FONT = "/tmp/lingbot_report_fonts/NotoSansSC-Regular.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(),
                     "pdf.fonttype": 42, "axes.unicode_minus": False, "font.size": 8})
W, H = 595.276, 841.89
INK, MUTED, TEAL, BLUE, LINE = "#172F43", "#536777", "#078477", "#206BBA", "#DAE4EB"
fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
page = fig.add_axes([0, 0, 1, 1])
page.set(xlim=(0, W), ylim=(H, 0))
page.axis("off")


def txt(x, y, s, size=9, color=INK, weight="normal", **kw):
    return page.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                     va="top", ha="left", **kw)


def section(y, n, title):
    txt(32, y, n, size=9.5, color=TEAL, weight="bold")
    txt(54, y - .5, title, size=10.5, weight="bold")


def panel(x, y, w, h, color="#F0F5F8"):
    page.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=5",
                                 fc=color, ec="none"))


txt(32, 29, "LingBot-VLA 2.0", size=22, weight="bold")
txt(225, 34, "训练策略 · 进度 · 速度", size=16)
txt(32, 62, "团队讨论版  /  2026-10-16 00:04:37（北京时间）  /  运行中 · 第 1 / 5 轮", size=8.5, color=MUTED)
page.add_patch(Rectangle((32, 77), W - 64, 2, fc=TEAL, ec="none"))

cards = [
    ("核心预算 · 单轮计算", "19–20 小时", "500 步实测外推：19.73 小时"),
    ("当前进度 · 第 1 轮", "504 / 5,717 步", "首轮 8.82%；总计划 1.76%"),
    ("动作 loss · 百步均值", "0.332 → 0.105", "当前较前 100 步下降 68.4%"),
    ("5 轮计算总预算", "约 4.1 天", "另加保存、上传与评测时间"),
]
cw = (W - 64 - 24) / 4
for i, (cap, value, note) in enumerate(cards):
    x = 32 + i * (cw + 8)
    panel(x, 88, cw, 55, "#E8F4F0" if i == 0 else "#F0F5F8")
    txt(x + 9, 95, cap, size=7.5, color=MUTED)
    txt(x + 9, 109, value, size=12.3 if i != 1 else 11.3, weight="bold", color=TEAL if i == 0 else INK)
    txt(x + 9, 128, note, size=6.9, color=MUTED)

section(158, "01", "本轮训练策略：先建立 5e-5 基线，每轮观察效果")
plan = [
    "模型与数据：指定 LingBot-VLA 2.0 6B；50 任务 × 50 轨迹、548,893 帧；目标为 clean 训练→randomized 评测。",
    "训练范围：冻结 VLM，训练动作专家及深度/视频查询、共享投影与对齐头；保持 TRAIN_EXPERT_ONLY=true。",
    "学习目标：从图像、指令和状态预测未来 50 步动作；动作损失 + 深度/未来深度/视频辅助损失（各 0.004）。",
    "训练批量：8 × Ascend 910B（64 GiB/卡）；每卡 batch 12 × 梯度累积 1 = 全局 96，使用 FSDP2。",
    "优化设置：Muon；基础 LR 5e-5→1e-5（cosine），warmup 571 步 / 2%，WD 0.001；MoE 专家 LR ×2.83。",
    "训练预算：5 epochs，共 28,585 步；每轮保存一次检查点并上传 OBS，供后续评测和恢复。",
]
for i, line in enumerate(plan):
    txt(32, 179 + i * 14, line, size=8.35)

section(278, "02", "当前观察：训练持续推进，loss 下降；尚无任务成功率")
steps = np.array([r["step"] for r in ROWS])
loss = np.array([r["loss"] for r in ROWS])
vla = np.array([r["vla_loss"] for r in ROWS])
times = np.array([r["step_time"] for r in ROWS])


def chart(x, y, w, h):
    ax = fig.add_axes([x / W, (H - y - h) / H, w / W, h / H])
    ax.set_facecolor("#F8FAFC")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color(LINE)
    ax.tick_params(labelsize=7, colors=MUTED, length=2, pad=3)
    ax.grid(axis="y", color=LINE, alpha=.6, lw=.6)
    ax.set_axisbelow(True)
    ax.set_xlim(1, S["step"])
    ax.set_xticks([1, 100, 200, 300, 400, S["step"]])
    return ax


txt(32, 300, "训练 loss（25 步滑动平均；浅线为原始动作 loss）", size=7.4, color=MUTED)
txt(310, 300, "单步耗时 / 秒（剔除首步 97 秒初始化开销）", size=7.4, color=MUTED)
ax = chart(54, 316, 235, 84)
ax.plot(steps, vla, color=TEAL, lw=.65, alpha=.18)
kernel = np.ones(25) / 25
ax.plot(steps[24:], np.convolve(loss, kernel, "valid"), color=BLUE, lw=1.5, label="总 loss")
ax.plot(steps[24:], np.convolve(vla, kernel, "valid"), color=TEAL, lw=1.5, label="动作 loss")
ax.set_ylim(0, .53)
ax.set_yticks([0, .2, .4])
ax.legend(loc="upper right", fontsize=7, frameon=False, handlelength=1.5)
ax2 = chart(332, 316, 229, 84)
ax2.plot(steps[1:], times[1:], lw=.65, color=BLUE)
ax2.axhline(S["last500"]["wall_seconds_per_step"], color=TEAL, ls="--", lw=.8,
            label="500 步实际均值 12.42 秒")
ax2.set_ylim(0, 62)
ax2.set_yticks([0, 20, 40, 60])
ax2.legend(loc="upper right", fontsize=6.5, frameon=False, handlelength=1.5)
txt(252, 412, "更新步数", size=6.5, color=MUTED)
txt(524, 412, "更新步数", size=6.5, color=MUTED)
txt(32, 429, "总 loss 均值 0.3614→0.1157，动作 loss 0.3321→0.1048（步骤 1–100 对比 405–504）。", size=8.35)
txt(32, 443, "训练循环已运行 1 小时 47 分；仍在 warmup（504/571）；未见 OOM/非有限 loss，尚无首轮检查点及闭环评测。", size=8.1, color=MUTED)

section(467, "03", "速度评估：采用实际总耗时，单轮按 19–20 小时安排")
xs = [32, 164, 267, 367, 469]
headers = ["统计窗口", "实际秒 / 步", "全局样本 / 秒", "外推小时 / 轮", "说明"]
page.add_patch(Rectangle((32, 489), W - 64, 18, fc=INK, ec="none"))
for x, h in zip(xs, headers):
    txt(x + 7, 492, h, size=8, color="white")
for i, n in enumerate([100, 300, 500]):
    m = S[f"last{n}"]
    y = 507 + 20 * i
    page.add_patch(Rectangle((32, y), W - 64, 20, fc="#E8F4F0" if n == 500 else ("#F0F5F8" if i == 0 else "white"), ec="none"))
    cells = [f"最近 {n} 步（{505-n}–504）", f"{m['wall_seconds_per_step']:.2f}",
             f"{m['wall_samples_per_sec']:.2f}", f"{m['epoch_hours']:.2f}", "主要估算口径" if n == 500 else "短窗口交叉核对"]
    for x, val in zip(xs, cells):
        txt(x + 7, y + 4, val, size=8.1, color=TEAL if n == 500 else INK)
txt(32, 575, "为什么不能按 8.7 秒估时：最近 500 步中，41 步超过 30 秒，占步数 8.2%，却占 StepTime 总和 25.3%。", size=8.15)
txt(32, 589, "中位数 8.72 秒只对应约 13.8 小时/轮，会低估约 30%；当前瓶颈来源仍需定位，不能仅凭曲线归因。", size=8.15, color=MUTED)

panel(32, 612, W - 64, 69, "#E8F4F0")
txt(44, 622, "首轮预计完成计算：10 月 16 日 18:00 左右", size=12, color=TEAL, weight="bold")
txt(44, 642, "从快照起约再需 18 小时；5 轮计算约 98.6 小时（4.1 天），当前剩余约 96.9 小时（4.0 天）。", size=8.6)
txt(44, 659, "假设后续速度与最近 500 步相近；不含保存/OBS 上传、评测与中断。首轮完整保存耗时尚未测得。", size=8.1, color=MUTED)

section(697, "04", "团队讨论重点：用评测判断收益，用吞吐控制实验成本")
discussion = [
    "策略：先观察本轮 5e-5 基线；每轮用相同任务与 seed 做 randomized 评测，再判断继续训练的收益。",
    "效率：优先减少慢步骤；目前 HBM 约 96–97%，不宜直接加大 batch。后续可讨论短程筛选实验的预算。",
    "数据：当前是朋友提供的转换数据；clean 来源尚未独立核验，需补齐来源依据再解释正式评测结果。",
]
for i, line in enumerate(discussion):
    txt(32, 717 + 14 * i, line, size=8.25)

page.plot([32, W - 32], [771, 771], color=LINE, lw=.65)
txt(32, 779, "估算口径", size=7.1, color=TEAL, weight="bold")
txt(84, 779, "实际秒/步＝窗口首尾日志时间差÷更新数，包含等待；样本/秒＝96÷实际秒/步，并非视频 FPS。", size=7.0, color=MUTED)
txt(84, 791, "单轮小时＝5,717×实际秒/步÷3,600。速度窗口不含首步；数值为当前早期观测外推，非完成时限。", size=7.0, color=MUTED)
txt(32, 804, "数据来源", size=7.1, color=TEAL, weight="bold")
txt(84, 804, "logs/robotwin_5ep_lr5e5/{train.log, loss.jsonl} 与当前配置；固定快照随本报告保存。", size=7.0, color=MUTED)
txt(32, 823, "LINGBOT-VLA 2.0  /  TRAINING STRATEGY & THROUGHPUT", size=6.2, color="#7C8F9E")
txt(540, 821, "1 / 1", size=7, color=MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for t in page.texts:
    b = t.get_window_extent(renderer)
    if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5:
        raise RuntimeError(f"Text outside page: {t.get_text()}")
pdf = ROOT / "lingbot_strategy_progress_speed_20261016.pdf"
fig.savefig(pdf, metadata={"Title": "LingBot-VLA 2.0 训练策略、进度与速度", "Author": "LingBot training team",
                          "Subject": "Snapshot 2026-10-16 00:04:37 +08:00, step 504. One-page measured-throughput brief."})
fig.savefig(ROOT / "lingbot_strategy_progress_speed_20261016.png", dpi=160)
print(pdf)
print(f"Size: {pdf.stat().st_size:,} bytes")
