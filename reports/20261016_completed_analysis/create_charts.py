import os,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/lingbot_report_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
P=Path(__file__).resolve().parent
D=json.loads((P/'training_summary.json').read_text())
rows=[json.loads(s) for s in (P.parent.parent/'logs/robotwin_1ep_fast/loss.jsonl').read_text().splitlines()]
font='/tmp/lingbot_report_fonts/NotoSansSC-Regular.ttf';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
x=np.array([r['step'] for r in rows]); colors=['#0B8C7E','#315EB5','#D29231'];ink='#183047'
def arr(k):return np.array([r[k] for r in rows])
def smooth(v,n=100):return np.convolve(v,np.ones(n)/n,mode='valid')
fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained');fig.set_facecolor('#F5F8FB')
fig.suptitle('LingBot-VLA 2.0｜1 epoch 完整训练复盘',fontsize=20,color=ink)
for ax in axs.ravel():ax.set_facecolor('white');ax.grid(alpha=.16);ax.tick_params(labelsize=9)
a=axs[0,0];a.plot(x,arr('vla_loss'),color=colors[0],alpha=.15,lw=.5)
for k,c,lab in [('vla_loss',colors[0],'动作 loss'),('loss',colors[1],'总 loss')]:a.plot(x[99:],smooth(arr(k)),color=c,label=lab+'（100步均值）')
a.set(title='训练 loss 持续下降，末段趋缓',xlabel='训练步',ylabel='loss');a.legend(fontsize=9)
a=axs[0,1]
for k,c,lab in zip(['depth_loss','future_depth_loss','future_video_loss'],colors,['当前深度','未来深度','未来视频']):a.plot(x[99:],smooth(arr(k)),label=lab,color=c)
a.set(title='辅助目标均下降（未乘 0.004 权重）',xlabel='训练步',ylabel='辅助 loss');a.set_yscale('log');a.legend(fontsize=9)
a=axs[0,2];a.plot(x,arr('grad_norm'),alpha=.25,lw=.5,color=colors[1]);a.plot(x[99:],smooth(arr('grad_norm')),color=colors[1],label='100步均值');a.axhline(1,ls='--',color=colors[2],label='梯度裁剪阈值 1.0');a.set(title='梯度有限，未见持续爆炸',xlabel='训练步',ylabel='裁剪前梯度范数');a.legend(fontsize=9)
a=axs[1,0];a.plot(x[1:],arr('step_time')[1:],alpha=.25,lw=.6,color=colors[1]);a.plot(x[100:],smooth(arr('step_time')[1:])),
a.lines[-1].set_color(colors[0]);a.lines[-1].set_label('100步均值');a.set(title='单步耗时稳定（图中排除首步初始化）',xlabel='训练步',ylabel='秒 / 步');a.text(.03,.95,'首步 64.0 s；4 次周期性短暂升至 13–14 s',transform=a.transAxes,va='top',fontsize=9);a.legend(loc='center right',fontsize=9)
a=axs[1,1];a.plot(x,arr('lr')*1e5,color=colors[0],label='基础 LR');a.plot(x,arr('expert_lr')*1e5,color=colors[1],label='路由专家 LR');a.set(title='2% warmup 后余弦衰减',xlabel='训练步',ylabel='学习率（单位 1e-5）');a.legend(fontsize=9)
a=axs[1,2];labels=['AI Core\n8卡均值','AI Vector\n2卡均值','HBM带宽\n2卡均值','显存占用\n每卡约'];vals=[6.84375,10.55,9.85,96.5];bars=a.bar(labels,vals,color=[colors[2],colors[1],colors[1],colors[0]],width=.55)
for b,v in zip(bars,vals):a.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}%',ha='center',fontsize=10)
a.set(title='资源仅有早期短窗口采样，非全程均值',ylabel='监控读数 %',ylim=(0,112));a.grid(axis='x',visible=False);a.text(.02,.94,'AI Core：00:57–00:58；其余计算指标：01:11–01:12\n显存占用与算力利用率不是同一指标',transform=a.transAxes,va='top',fontsize=8.3)
fig.savefig(P/'training_overview.png',dpi=160,bbox_inches='tight')
fig.savefig(P/'training_overview.pdf',bbox_inches='tight')
plt.close(fig)
print(P/'training_overview.png')
