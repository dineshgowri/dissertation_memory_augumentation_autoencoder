import matplotlib.pyplot as plt
import numpy as np

# Data (PED2)
models = ['FFP\n(PED2)', 'MemAE\n(PED2)']
published_auc = [95.4, 94.10]          # baselines from the papers
our_auc = [95.407, 93.0772]            # your results

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, published_auc, width, label='Published AUC', color='skyblue')
bars2 = ax.bar(x + width/2, our_auc, width, label='Our Testing AUC', color='salmon')

ax.set_title('Model Performance Comparison: Published vs Our Testing Results')
ax.set_xlabel('Model (Dataset)')
ax.set_ylabel('AUC Score (%)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(93.0, 96.5)
ax.legend()

# Label bars (3 decimal places to show 95.407, 93.077)
for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.3f}%',
                    xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
