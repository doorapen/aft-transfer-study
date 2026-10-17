import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'reports/extended_30ep'
h=json.loads((p/'histories.json').read_text())
s=json.loads((p/'summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for arm,color,label in [('baseline','#2764a3','No-transfer baseline'),('aft_rbf','#d77427','RBF AFT')]:
    values=np.array([[r['validation']['accuracy']*100 for r in h[str(seed)][arm]['history']] for seed in (42,43,44)])
    axes[0].plot(range(1,31),values.mean(0),color=color,label=label)
    axes[0].fill_between(range(1,31),values.min(0),values.max(0),color=color,alpha=.15)
    test=[r[arm]['test_accuracy_percent'] for r in s['seeds']]
    axes[1].plot([42,43,44],test,'o-',color=color,label=label)
axes[0].set(title='Validation: mean and seed range',xlabel='Epoch',ylabel='Accuracy (%)')
axes[1].set(title='Test: validation-selected checkpoint',xlabel='Training seed',ylabel='Accuracy (%)',xticks=[42,43,44])
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('CIFAR-100: 10k train / 2k validation / 10k test; fixed split')
fig.savefig(p/'comparison.png',dpi=180)
