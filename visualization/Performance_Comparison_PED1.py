import matplotlib.pyplot as plt
import numpy as np

# Data (PED2)
models = ['FFP\n(PED1)', 'MemAE\n(PED1)']
published_auc = [83.1, 0]          # baselines from the papers
our_auc = [82.826, 72.60]            # your results

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
ax.set_ylim(70.0, 84.5)
ax.legend()

# Label bars
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
