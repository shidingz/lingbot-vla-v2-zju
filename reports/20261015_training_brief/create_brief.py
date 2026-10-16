"""Render a one-page Chinese training brief from a frozen, read-only snapshot."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lingbot_report_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
SNAPSHOT = json.loads((ROOT / "training_snapshot.json").read_text())
ROWS = SNAPSHOT["rows"]
FONT = "/tmp/lingbot_report_fonts/NotoSansSC-Regular.ttf"
font_manager.fontManager.addfont(FONT)
FAMILY = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({
    "font.family": FAMILY,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "font.size": 8,
})

W, H = 595.276, 841.89  # A4, points
INK = "#172F43"
MUTED = "#536777"
BLUE = "#206BBA"
TEAL = "#078477"
LINE = "#DAE4EB"
fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
canvas = fig.add_axes([0, 0, 1, 1])
canvas.set(xlim=(0, W), ylim=(H, 0))
canvas.axis("off")


def text(x, y, s, size=9, color=INK, weight="normal", **kwargs):
    return canvas.text(x, y, s, fontsize=size, color=color,
                       fontweight=weight, va="top", ha="left", **kwargs)


def rule(y):
    canvas.plot([32, W - 32], [y, y], color=LINE, lw=0.65)


def label(y, number, title):
    text(32, y, number, size=9.5, color=TEAL, weight="bold")
    text(54, y - 0.5, title, size=10.5, weight="bold")


text(32, 28, "LingBot-VLA 2.0", size=22, weight="bold")
text(225, 33, "训练方案与阶段结果", size=16)
text(32, 61, "团队讨论简报  /  2026-10-15 23:47（北京时间）  /  robotwin_5ep_lr5e5", size=8.5, color=MUTED)
canvas.add_patch(Rectangle((32, 76), W - 64, 2, color=TEAL, lw=0))

cards = [
    ("当前进度 · 第 1 / 5 轮", "419 / 5,717 步", "总计划已完成 1.47%"),
    ("动作 loss · 百步均值", "0.332 → 0.131", "前 100 步 vs 最近 100 步"),
    ("最近 100 步 · 平均耗时", "12.19 秒 / 步", "中位数 8.69 秒；存在长尾"),
    ("训练资源", "8 × Ascend 910B", "每卡 64 GiB；HBM 约 96–97%"),
]
gap = 8
cw = (W - 64 - 3 * gap) / 4
for i, (cap, value, note) in enumerate(cards):
    x = 32 + i * (cw + gap)
    canvas.add_patch(FancyBboxPatch((x, 87), cw, 53,
                    boxstyle="round,pad=0,rounding_size=5", fc="#F0F5F8", ec="none"))
    text(x + 9, 94, cap, size=7.4, color=MUTED)
    text(x + 9, 108, value, size=11.4, weight="bold")
    text(x + 9, 126, note, size=6.8, color=MUTED)

label(154, "01", "训练什么、怎么训练")
plan = [
    "起点：指定的 Robbyant/lingbot-vla-v2-6b；用图像、指令和机器人状态学习未来 50 步动作。",
    "范围：TRAIN_EXPERT_ONLY=true，冻结 VLM；动作专家、深度/视频查询、共享投影和对齐头可训练。",
    "监督：L1 flow-matching 动作损失 + 深度/未来深度/视频辅助损失（权重各 0.004）；teacher 固定。",
    "批量：8 卡 FSDP2；每卡 batch 12 × 梯度累积 1 = 全局 96；5 epochs，共 28,585 次更新。",
    "优化：Muon；基础 LR 5e-5→1e-5（cosine）；warmup 571 步 / 2%；WD 0.001；MoE 专家 LR ×2.83。",
    "数据：RoboTwin LeRobot v3，50 任务 × 50 轨迹，548,893 帧，15 FPS；目标为 clean 训练→randomized 评测。",
]
for i, line in enumerate(plan):
    text(32, 174 + 14 * i, line, size=8.5)

label(270, "02", "已观察到：损失下降；仍在 warmup，尚无闭环成功率")
steps = np.array([r["step"] for r in ROWS])
loss = np.array([r["loss"] for r in ROWS])
vla = np.array([r["vla_loss"] for r in ROWS])
times = np.array([r["step_time"] for r in ROWS])


def axes_at(x, y, w, h):
    ax = fig.add_axes([x / W, (H - y - h) / H, w / W, h / H])
    ax.set_facecolor("#F8FAFC")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(labelsize=7, colors=MUTED, length=2, pad=3)
    ax.grid(axis="y", alpha=0.5, color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(1, 419)
    ax.set_xticks([1, 100, 200, 300, 419])
    return ax


ax1 = axes_at(54, 307, 235, 83)
ax1.plot(steps, vla, color=TEAL, alpha=0.18, lw=0.65)
kernel = np.ones(25) / 25
ax1.plot(steps[24:], np.convolve(loss, kernel, mode="valid"), color=BLUE, lw=1.5, label="总 loss")
ax1.plot(steps[24:], np.convolve(vla, kernel, mode="valid"), color=TEAL, lw=1.5, label="动作 loss")
ax1.set_ylim(0, 0.55)
ax1.set_yticks([0, 0.2, 0.4])
ax1.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.5)
text(32, 292, "损失趋势（25 步滑动平均；浅色为原始动作 loss）", size=7.6, color=MUTED)

ax2 = axes_at(333, 307, 228, 83)
ax2.plot(steps[1:], times[1:], color=BLUE, lw=0.65, alpha=0.8)
ax2.axhline(np.median(times[-100:]), color=TEAL, ls="--", lw=0.8)
ax2.set_ylim(0, 60)
ax2.set_yticks([0, 20, 40, 60])
text(311, 292, "单步耗时 / 秒（不含首步初始化 97 秒）", size=7.6, color=MUTED)
text(251, 403, "更新步数", size=6.5, color=MUTED)
text(523, 403, "更新步数", size=6.5, color=MUTED)

text(32, 419, "总 loss 百步均值 0.3614 → 0.1427；动作 loss 下降 60.7%。比较区间为步骤 1–100 与 320–419。", size=8.2)
text(32, 432, "419 步未见非有限 loss / OOM；基础 LR 当前 3.669e-5。尚未完成首轮，尚无本次训练的 epoch 检查点。", size=8.2, color=MUTED)

label(454, "03", "相对原仓库：改了什么，为什么")
left, widths, top = 32, [101, 257, W - 64 - 101 - 257], 474
xs = [left, left + widths[0], left + widths[0] + widths[1]]
canvas.add_patch(Rectangle((left, top), W - 64, 17, fc=INK, ec="none"))
for x, s in zip(xs, ["改动位置", "改动内容", "原因 / 意义"]):
    text(x + 7, top + 3, s, size=8.2, color="white")
table = [
    ("已有 · NPU 适配", "HCCL；注意力 / MoE 兼容实现；\n梯度检查点，关闭 compile。", "适配 Ascend；保持预训练\n专家权重结构，降低显存。"),
    ("本次 · 训练配置", "固定 5 轮、基础 LR 5e-5、全局 batch 96；\n移除固定 step 上限，每轮保存一次。", "与本轮训练预算一致；\n避免提前截断或漏存。"),
    ("本次 · OBS 存储", "数据缓存 /cache；每轮保存 DCP + HF；\n上传校验完成再清理本地，保留小日志。", "work 仅 50 GB；模型与\n恢复状态主要存入 OBS。"),
    ("本次 · 恢复与观测", "恢复优化器 / 调度器 / 数据进度 / RNG；\n记录 loss.jsonl，通过 tmux 持续运行。", "便于续训、追踪与诊断；\n恢复失败显式报错。"),
    ("本次 · 视频边界", "按轨迹/相机限制取帧；尾部缺 1–2 帧\n复用该轨迹末帧，严重异常继续报错。", "避免越界和跨轨迹取帧；\n仅调整读取，不改原始数据。"),
]
for i, row in enumerate(table):
    y = top + 17 + i * 31
    canvas.add_patch(Rectangle((left, y), W - 64, 31, fc="#F0F5F8" if i % 2 == 0 else "#FFFFFF", ec="none"))
    for j, (x, s) in enumerate(zip(xs, row)):
        text(x + 7, y + 4, s, size=8.0 if j else 8.2, color=INK if j < 2 else MUTED, linespacing=1.45)

label(659, "04", "下次讨论：先验收，再做少量参数实验")
discuss = [
    "① 数据：补齐官方 clean 来源证明；当前为朋友提供的转换数据，clean 来源尚未独立核验。",
    "② 效果：首轮模型与指定预训练模型做同任务、同 seed 的 randomized 闭环评测；loss ≠ 成功率。",
    "③ 速度：定位约 9 秒短步与 30–45 秒长尾的取数/通信/计算耗时；HBM 已紧张，先不增大 batch。",
    "④ 交付：验收首轮完整 OBS 检查点与恢复；已有小检查点上传测试，不能替代完整训练链路验证。",
]
for i, line in enumerate(discuss):
    text(32, 679 + i * 14, line, size=8.3)

rule(745)
text(32, 753, "存放位置", size=7.3, color=TEAL, weight="bold")
text(84, 753, "OBS 根目录：obs://yw-2030-extern/Partner_Qin/robot_1014/", size=7.1, color=MUTED)
text(84, 765, "数据：datasets/RoboTwin_lerobot_v30/    训练产物：robotwin_5ep_lr5e5/（均相对上方根目录）", size=7.1, color=MUTED)
text(32, 782, "证据与口径", size=7.3, color=TEAL, weight="bold")
text(84, 782, "本地 logs/robotwin_5ep_lr5e5/{loss.jsonl, train.log, lingbotvla_cli.yaml}；固定快照随报告保存。", size=7.0, color=MUTED)
text(84, 794, "原仓库基线：Robbyant/lingbot-vla-v2 @ be969b8；本报告为训练早期观察，非最终效果结论。", size=7.0, color=MUTED,
     url="https://github.com/Robbyant/lingbot-vla-v2/tree/be969b8fd117fb70550c5d4bf4bc328211b5b1b6")
text(32, 818, "LINGBOT-VLA 2.0  /  ROBOTWIN  /  EXPERIMENT BRIEF", size=6.3, color="#7C8F9E")
text(539, 816, "1 / 1", size=7, color=MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
overflows = []
for t in canvas.texts:
    box = t.get_window_extent(renderer)
    if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
        overflows.append(t.get_text())
if overflows:
    raise RuntimeError(f"Text outside page: {overflows}")

pdf = ROOT / "lingbot_training_brief_20261015.pdf"
fig.savefig(pdf, metadata={
    "Title": "LingBot-VLA 2.0 训练方案与阶段结果",
    "Author": "LingBot training team",
    "Subject": "2026-10-15 23:47 +08:00; step 419; one-page training brief",
})
fig.savefig(ROOT / "lingbot_training_brief_20261015.png", dpi=160)
print(pdf)
print(f"PDF size: {pdf.stat().st_size:,} bytes; fixed snapshot step {ROWS[-1]['step']}")
