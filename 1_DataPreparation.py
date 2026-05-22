#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# ### Selecting the Timeseries Attributes 

# In[1]:


import pandas as pd
import os

# File list and label mapping
files = ['F.csv', 'Np.csv', 'P4.csv', 'P5.csv', 'P6.csv',
         'Tp.csv', 'V.csv', 'Vx.csv', 'Xl.csv', 'Xs.csv']

label_map = {
    'NSEP': 0,
    'gt10': 1,
    'gt30': 2,
    'gt60': 3,
    'gt100': 4
}

input_dir = './dataset'   # folder where your CSVs live
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

for filename in files:
    filepath = os.path.join(input_dir, filename)
    df = pd.read_csv(filepath, header=None)  # adjust header=0 if there's a header row

    # First column = label (string), last 288 columns = time series
    label_col = df.iloc[:, 0]
    timeseries_cols = df.iloc[:, -288:]

    # Map label strings to integers
    label_col = label_col.map(label_map)

    # Combine label + time series
    result = pd.concat([label_col.reset_index(drop=True),
                        timeseries_cols.reset_index(drop=True)], axis=1)

    # Reset column names to 0, 1, 2, ..., 288
    result.columns = range(result.shape[1])

    # Save
    out_path = os.path.join(output_dir, filename)
    result.to_csv(out_path, index=False, header=False)
    print(f"Saved: {out_path}  shape={result.shape}")

print("Done!")


# ### Missing value analysis 

# In[2]:


import pandas as pd
import numpy as np

files = ['F.csv', 'Np.csv', 'P4.csv', 'P5.csv', 'P6.csv',
         'Tp.csv', 'V.csv', 'Vx.csv', 'Xl.csv', 'Xs.csv']

input_dir = './output'

print(f"{'File':<12} {'Rows':>7} {'Total Cells':>12} {'Missing':>10} {'Missing %':>10} {'Rows w/ Missing':>16}")
print("-" * 72)

for filename in files:
    filepath = f"{input_dir}/{filename}"
    try:
        df = pd.read_csv(filepath, header=None, low_memory=False)
        df = df.iloc[1:].reset_index(drop=True)   # drop header row
        df = df.apply(pd.to_numeric, errors='coerce')

        total_cells   = df.size
        missing_cells = df.isna().sum().sum()
        missing_pct   = 100 * missing_cells / total_cells
        rows_affected = df.isna().any(axis=1).sum()

        print(f"{filename:<12} {len(df):>7} {total_cells:>12,} {missing_cells:>10,} {missing_pct:>9.2f}% {rows_affected:>16,}")

    except FileNotFoundError:
        print(f"{filename:<12} [FILE NOT FOUND]")

print("-" * 72)
print("\nDone!")


# ### tSNE plots of classes

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

files = ['F.csv', 'Np.csv', 'P4.csv', 'P5.csv', 'P6.csv',
         'Tp.csv', 'V.csv', 'Vx.csv', 'Xl.csv', 'Xs.csv']

input_dir = './output'
output_dir = './tsne_plots'
import os; os.makedirs(output_dir, exist_ok=True)

# Label metadata
label_names  = {0: 'NSEP', 1: 'gt10', 2: 'gt30', 3: 'gt60', 4: 'gt100'}
label_colors = {0: '#7e7e7e', 1: '#e6194b', 2: '#3cb44b', 3: '#4363d8', 4: '#f58231'}
DOT_SMALL    = 8    # NSEP
DOT_BIG      = 80  # all other classes (5x bigger)

for filename in files:
    filepath = f"{input_dir}/{filename}"
    if not os.path.exists(filepath):
        print(f"[SKIP] {filename} not found")
        continue

    print(f"Processing {filename} ...")

    df = pd.read_csv(filepath, header=None, low_memory=False)
    df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    labels = df.iloc[:, 0].astype(int).values
    X      = df.iloc[:, 1:].values

    # t-SNE
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30,
                  max_iter=1000, verbose=1)
    X_2d   = tsne.fit_transform(X)

    # Build per-point colors and sizes
    colors = [label_colors[l] for l in labels]
    sizes  = [DOT_SMALL if l == 0 else DOT_BIG for l in labels]

    # Plot — draw NSEP first so minority classes render on top
    fig, ax = plt.subplots(figsize=(10, 8))

    for label_id in sorted(label_names.keys()):
        mask = labels == label_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=label_colors[label_id],
            s=DOT_SMALL if label_id == 0 else DOT_BIG,
            label=f"{label_names[label_id]} (n={mask.sum()})",
            alpha=0.6 if label_id == 0 else 0.9,
            edgecolors='none' if label_id == 0 else 'black',
            linewidths=0 if label_id == 0 else 0.4,
            zorder=1 if label_id == 0 else 2   # minority classes on top
        )

    ax.set_title(f't-SNE — {filename.replace(".csv", "")}', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    out_path = f"{output_dir}/{filename.replace('.csv', '_tsne.png')}"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")

print("\nAll done!")


# ### Combining files to create an MVTS

# In[4]:


import pandas as pd
import numpy as np
import pickle
import os

files = ['F.csv', 'Np.csv', 'P4.csv', 'P5.csv', 'P6.csv',
         'Tp.csv', 'V.csv', 'Vx.csv', 'Xl.csv', 'Xs.csv']

input_dir  = './output'
output_dir = './mvts'
os.makedirs(output_dir, exist_ok=True)

all_X = []
all_y = []

for filename in files:
    filepath = f"{input_dir}/{filename}"
    if not os.path.exists(filepath):
        print(f"[SKIP] {filename} not found")
        continue

    print(f"Loading: {filename}")

    df = pd.read_csv(filepath, header=None, low_memory=False)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    labels = df.iloc[:, 0].astype(int).values   # (num_samples,)
    ts     = df.iloc[:, 1:].values               # (num_samples, 288)

    all_X.append(ts)
    all_y.append(labels)

    print(f"  Samples: {len(labels)}  |  Timesteps: {ts.shape[1]}")

# Stack all files
# Each file = one feature → shape becomes (num_samples, num_timesteps, num_features)
X = np.stack(all_X, axis=-1)   # (num_samples, 288, 10)
y = all_y[0]                    # labels are the same across all files

print(f"\nFinal dataset shape : {X.shape}  →  (samples, timesteps, features)")
print(f"Labels shape        : {y.shape}")
print(f"Label distribution  : {dict(zip(*np.unique(y, return_counts=True)))}")

# Save X as pickle
pickle_path = os.path.join(output_dir, 'X.pkl')
with open(pickle_path, 'wb') as f:
    pickle.dump(X, f)
print(f"\nSaved X → {pickle_path}")

# Save y as 1D csv
labels_path = os.path.join(output_dir, 'y.csv')
pd.Series(y).to_csv(labels_path, index=False, header=False)
print(f"Saved y → {labels_path}")

print("\nAll done!")


# ### tSNE plots of classes using MVTS

# In[5]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

input_dir  = './mvts'
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)

# Load data
with open(f'{input_dir}/X.pkl', 'rb') as f:
    X = pickle.load(f)

y = pd.read_csv(f'{input_dir}/y.csv', header=None).values.flatten().astype(int)

# Flatten (num_samples, 288, 10) → (num_samples, 2880)
X_flat = X.reshape(X.shape[0], -1)

label_names  = {0: 'NSEP', 1: 'gt10', 2: 'gt30', 3: 'gt60', 4: 'gt100'}
label_colors = {
    0: '#94A3B8',  # cool slate
    1: '#F43F5E',  # rose
    2: '#10B981',  # emerald
    3: '#F59E0B',  # amber
    4: '#8B5CF6',  # violet
}
DOT_SMALL = 8
DOT_BIG   = 100

def plot_embedding(X_2d, y, title, filename, xlabel, ylabel):
    sns.set_theme(style='whitegrid', context='talk', font='DejaVu Sans')
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F1F5F9')

    # Draw NSEP first so minority classes render on top
    for label_id in sorted(label_names.keys()):
        mask = y == label_id
        if mask.sum() == 0:
            continue

        is_nsep = label_id == 0
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=label_colors[label_id],
            s=DOT_SMALL if is_nsep else DOT_BIG,
            label=f"{label_names[label_id]}  (n={mask.sum():,})",
            alpha=0.4 if is_nsep else 0.9,
            edgecolors='none' if is_nsep else 'white',
            linewidths=0 if is_nsep else 0.5,
            zorder=1 if is_nsep else 2
        )

    ax.set_title(title, fontsize=20, fontweight='bold', color='#1E293B', pad=16)
    ax.set_xlabel(xlabel, fontsize=13, color='#475569', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=13, color='#475569', labelpad=10)
    ax.tick_params(colors='#475569', labelsize=11)

    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')
    ax.yaxis.grid(True, color='#CBD5E1', linewidth=0.6, linestyle='--')
    ax.xaxis.grid(True, color='#CBD5E1', linewidth=0.6, linestyle='--')

    legend = ax.legend(
        title='Class', title_fontsize=11,
        fontsize=10, loc='best',
        framealpha=0.95, edgecolor='#CBD5E1',
        facecolor='white'
    )
    legend.get_title().set_color('#1E293B')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {output_dir}/{filename}")

# ── PCA ───────────────────────────────────────────────────────────────────────
print("Running PCA...")
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_flat)
var   = pca.explained_variance_ratio_ * 100
print(f"  Explained variance: PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  Total={sum(var):.1f}%")

plot_embedding(
    X_pca, y,
    title=f'PCA — Multivariate Dataset  (PC1={var[0]:.1f}%  PC2={var[1]:.1f}%)',
    filename='pca.png',
    xlabel=f'PC1  ({var[0]:.1f}% variance)',
    ylabel=f'PC2  ({var[1]:.1f}% variance)'
)

# ── t-SNE ─────────────────────────────────────────────────────────────────────
print("\nRunning t-SNE (PCA 50 → t-SNE 2)...")
pca50   = PCA(n_components=50, random_state=42)
X_pca50 = pca50.fit_transform(X_flat)

tsne   = TSNE(n_components=2, random_state=42, perplexity=30,
              max_iter=1000, verbose=1)
X_tsne = tsne.fit_transform(X_pca50)

plot_embedding(
    X_tsne, y,
    title='t-SNE — Multivariate Dataset',
    filename='tsne.png',
    xlabel='t-SNE Dimension 1',
    ylabel='t-SNE Dimension 2'
)

print("\nAll done!")


# ### Class Distribution

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os

input_dir  = './mvts'
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)

y = pd.read_csv(f'{input_dir}/y.csv', header=None).values.flatten().astype(int)

label_names  = {0: 'NSEP', 1: 'gt10', 2: 'gt30', 3: 'gt60', 4: 'gt100'}
label_colors = {
    0: '#94A3B8',  # cool slate
    1: '#F43F5E',  # rose
    2: '#10B981',  # emerald
    3: '#F59E0B',  # amber
    4: '#8B5CF6',  # violet
}

unique, counts = np.unique(y, return_counts=True)
total = counts.sum()

df_plot = pd.DataFrame({
    'Class'  : [label_names[i] for i in unique],
    'Samples': counts,
    'Color'  : [label_colors[i] for i in unique]
})

# ── Canvas ───────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', context='talk', font='DejaVu Sans')
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F1F5F9')

# ── Bars ─────────────────────────────────────────────────────────────────────
sns.barplot(
    data=df_plot,
    x='Class', y='Samples',
    palette=df_plot['Color'].tolist(),
    edgecolor='white', linewidth=1.5,
    ax=ax, width=0.55
)

# Soft shadow behind each bar
for bar, label_id in zip(ax.patches, unique):
    cx = bar.get_x() + bar.get_width() / 2
    for alpha, extra_w in [(0.10, 0.30), (0.05, 0.55)]:
        ax.bar(cx, bar.get_height(),
               width=bar.get_width() + extra_w,
               color=label_colors[label_id],
               alpha=alpha, zorder=0, edgecolor='none')

# ── Labels on bars ───────────────────────────────────────────────────────────
for bar, count, label_id in zip(ax.patches, counts, unique):
    pct = 100 * count / total
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.004,
        f'{count:,}',
        ha='center', va='bottom',
        fontsize=13, fontweight='bold',
        color=label_colors[label_id],
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.004 + counts.max() * 0.045,
        f'{pct:.1f}%',
        ha='center', va='bottom',
        fontsize=10, color='#64748B',
    )

# ── Grid & spines ────────────────────────────────────────────────────────────
ax.yaxis.grid(True, color='#CBD5E1', linewidth=0.8, linestyle='--')
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color('#E2E8F0')

# ── Title & axis labels ──────────────────────────────────────────────────────
ax.set_title('Class Distribution', fontsize=22, fontweight='bold',
             color='#1E293B', pad=20)
ax.set_xlabel('Class', fontsize=14, color='#475569', labelpad=12)
ax.set_ylabel('Number of Samples', fontsize=14, color='#475569', labelpad=12)
ax.tick_params(colors='#475569', labelsize=12)
ax.set_ylim(0, counts.max() * 1.22)

# ── Total samples badge ──────────────────────────────────────────────────────
ax.text(0.99, 0.97, f'Total  {total:,} samples',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, color='#475569', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#CBD5E1', linewidth=1.2))

plt.tight_layout()
plt.savefig(f'{output_dir}/class_distribution.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {output_dir}/class_distribution.png")


# ### Zscore normalization 

# In[7]:


import numpy as np
import pandas as pd
import pickle
import os

# ── Paths ────────────────────────────────────────────────────────────────────
mvts_dir        = './mvts'                    # source (no normalization)
out_per_file    = './mvts_zscore_per_file'
out_per_row     = './mvts_zscore_per_row'

os.makedirs(out_per_file, exist_ok=True)
os.makedirs(out_per_row,  exist_ok=True)

# ── Load base MVTS ────────────────────────────────────────────────────────────
with open(os.path.join(mvts_dir, 'X.pkl'), 'rb') as f:
    X = pickle.load(f)                        # (samples, 288, 10)

y = pd.read_csv(os.path.join(mvts_dir, 'y.csv'), header=None).values.flatten()

print(f"Loaded X : {X.shape}  →  (samples, timesteps, features)")
print(f"Loaded y : {y.shape}")

# ── 1. Z-score per file ───────────────────────────────────────────────────────
# One global mean & std computed over ALL values across every sample,
# timestep, and feature — then applied uniformly.
X_pf = X.copy().astype(float)

global_mean = X_pf.mean()
global_std  = X_pf.std()
X_pf        = (X_pf - global_mean) / global_std

with open(os.path.join(out_per_file, 'X.pkl'), 'wb') as f:
    pickle.dump(X_pf, f)

pd.Series(y).to_csv(os.path.join(out_per_file, 'y.csv'), index=False, header=False)
print(f"\n[Per-file]  mean={global_mean:.4f}  std={global_std:.4f}")
print(f"  Saved → {out_per_file}/X.pkl  |  y.csv")

# ── 2. Z-score per row ────────────────────────────────────────────────────────
# Each sample (row) is normalised independently using its own mean & std,
# computed across all its timesteps and features (i.e. all 288×10 values).
X_pr = X.copy().astype(float)                # (samples, 288, 10)

row_mean = X_pr.mean(axis=(1, 2), keepdims=True)   # (samples, 1, 1)
row_std  = X_pr.std(axis=(1, 2),  keepdims=True)   # (samples, 1, 1)
row_std  = np.where(row_std == 0, 1, row_std)       # avoid division by zero

X_pr = (X_pr - row_mean) / row_std

with open(os.path.join(out_per_row, 'X.pkl'), 'wb') as f:
    pickle.dump(X_pr, f)

pd.Series(y).to_csv(os.path.join(out_per_row, 'y.csv'), index=False, header=False)
print(f"\n[Per-row]   normalised each sample over its own (288×10) values")
print(f"  Saved → {out_per_row}/X.pkl  |  y.csv")

print("\nAll done!")


# In[3]:


import pandas as pd
import numpy as np
from scipy.stats import zscore
import os

files = ['F.csv', 'Np.csv', 'P4.csv', 'P5.csv', 'P6.csv',
         'Tp.csv', 'V.csv', 'Vx.csv', 'Xl.csv', 'Xs.csv']

input_dir = './output'
out_per_column = './output_per_column'
out_per_file   = './output_per_file'
os.makedirs(out_per_column, exist_ok=True)
os.makedirs(out_per_file,   exist_ok=True)

for filename in files:
    filepath = f"{input_dir}/{filename}"
    if not os.path.exists(filepath):
        print(f"[SKIP] {filename} not found")
        continue

    print(f"Processing: {filename}")

    df = pd.read_csv(filepath, header=None, low_memory=False)
    df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    label_col = df.iloc[:, 0].astype(int)
    ts = df.iloc[:, 1:].copy().astype(float)

    # --- 1. Per-column z-score ---
    ts_per_col = ts.apply(zscore, nan_policy='omit').round(6)

    result_per_col = pd.concat([label_col.reset_index(drop=True),
                                ts_per_col.reset_index(drop=True)], axis=1)
    result_per_col.columns = range(result_per_col.shape[1])
    result_per_col.to_csv(os.path.join(out_per_column, filename), index=False, header=False)

    # --- 2. Per-file global z-score ---
    global_mean = ts.values.mean()
    global_std  = ts.values.std()
    ts_per_file = ((ts - global_mean) / global_std).round(6)

    result_per_file = pd.concat([label_col.reset_index(drop=True),
                                 ts_per_file.reset_index(drop=True)], axis=1)
    result_per_file.columns = range(result_per_file.shape[1])
    result_per_file.to_csv(os.path.join(out_per_file, filename), index=False, header=False)

    print(f"  Per-column → {out_per_column}/{filename}")
    print(f"  Per-file   → {out_per_file}/{filename}  (mean={global_mean:.4f}, std={global_std:.4f})")

print("\nAll done!")


# ### Multiclass to Binary 

# In[12]:


import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter

# ── Directories ───────────────────────────────────────────────────────────────
variants = [
    ('./mvts',                './mvts_binary'),
    ('./mvts_zscore_per_file', './mvts_zscore_per_file_binary'),
    ('./mvts_zscore_per_row',  './mvts_zscore_per_row_binary'),
]

for input_dir, output_dir in variants:
    print(f"\n{'='*55}")
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print('='*55)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    with open(f'{input_dir}/X.pkl', 'rb') as f:
        X = pickle.load(f)
    y = pd.read_csv(f'{input_dir}/y.csv', header=None).values.flatten().astype(int)
    print(f"Original Distribution : {Counter(y)}")

    # 2. Convert to Binary: Any label > 0 becomes 1
    # Class 0 = NSEP | Class 1 = SEP (gt10, gt30, gt60, gt100 combined)
    y_binary = np.where(y > 0, 1, 0)
    print(f"Binary Distribution   : {Counter(y_binary)}")

    # 3. Save
    with open(f'{output_dir}/X.pkl', 'wb') as f:
        pickle.dump(X, f)
    pd.Series(y_binary).to_csv(f'{output_dir}/y.csv', index=False, header=False)

    print(f"Saved X → {output_dir}/X.pkl")
    print(f"Saved y → {output_dir}/y.csv")

print("\nAll done!")


# ### Separate the classes 

# In[13]:


import numpy as np
import pickle
import pandas as pd
import os
from collections import Counter

# ── Directories ───────────────────────────────────────────────────────────────
variants = [
    './mvts_binary',
    './mvts_zscore_per_file_binary',
    './mvts_zscore_per_row_binary',
]

for input_dir in variants:
    print(f"\n{'='*55}")
    print(f"Input : {input_dir}")
    print('='*55)

    # 1. Load
    with open(f'{input_dir}/X.pkl', 'rb') as f:
        X = pickle.load(f)
    y = pd.read_csv(f'{input_dir}/y.csv', header=None).values.flatten().astype(int)
    print(f"Full Dataset Distribution : {Counter(y)}")

    # 2. Separate classes
    nsep_idx = np.where(y == 0)[0]
    sep_idx  = np.where(y == 1)[0]

    X_nsep = X[nsep_idx];  y_nsep = y[nsep_idx]
    X_sep  = X[sep_idx];   y_sep  = y[sep_idx]

    print(f"NSEP (Class 0) Samples    : {X_nsep.shape[0]:,}")
    print(f"SEP  (Class 1) Samples    : {X_sep.shape[0]:,}")

    # 3. Save separated arrays back into the same folder
    with open(f'{input_dir}/X_nsep.pkl', 'wb') as f:
        pickle.dump(X_nsep, f)
    with open(f'{input_dir}/X_sep.pkl', 'wb') as f:
        pickle.dump(X_sep, f)

    pd.Series(y_nsep).to_csv(f'{input_dir}/y_nsep.csv', index=False, header=False)
    pd.Series(y_sep).to_csv(f'{input_dir}/y_sep.csv',   index=False, header=False)

    print(f"Saved X_nsep, X_sep, y_nsep, y_sep → {input_dir}/")

print("\nAll done!")


# ### Train-test split (70 - 10 - 20) and shuffling

# In[ ]:


variants = [
    ('./mvts_binary',                './final_split_data_noNorm'),
    ('./mvts_zscore_per_row_binary',  './final_split_data_RowNorm'),
    ('./mvts_zscore_per_file_binary', './final_split_data_FileNorm'),
]


# In[15]:


import numpy as np
import pandas as pd
import pickle
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ── Variants: (binary input dir, output dir) ─────────────────────────────────
variants = [
    ('./mvts_binary',                './final_split_data_noNorm'),
    ('./mvts_zscore_per_row_binary',  './final_split_data_RowNorm'),
    ('./mvts_zscore_per_file_binary', './final_split_data_FileNorm'),
]

for input_dir, output_dir in variants:
    print(f"\n{'='*55}")
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print('='*55)

    os.makedirs(output_dir, exist_ok=True)

    # ── Load separated class files ────────────────────────────────────────────
    with open(f'{input_dir}/X_nsep.pkl', 'rb') as f:
        X_nsep = pickle.load(f)
    with open(f'{input_dir}/X_sep.pkl', 'rb') as f:
        X_sep = pickle.load(f)

    y_nsep = pd.read_csv(f'{input_dir}/y_nsep.csv', header=None).values.flatten().astype(int)
    y_sep  = pd.read_csv(f'{input_dir}/y_sep.csv',  header=None).values.flatten().astype(int)

    print(f"NSEP samples : {X_nsep.shape[0]:,}")
    print(f"SEP  samples : {X_sep.shape[0]:,}")

    # ── Step 1: Split NSEP 70-10-20 ──────────────────────────────────────────
    X_nsep_tv, X_nsep_test, y_nsep_tv, y_nsep_test = train_test_split(
        X_nsep, y_nsep, test_size=0.20, random_state=42, shuffle=False)

    X_nsep_train, X_nsep_val, y_nsep_train, y_nsep_val = train_test_split(
        X_nsep_tv, y_nsep_tv, test_size=0.125, random_state=42, shuffle=False)

    # ── Step 2: Split SEP 70-10-20 ───────────────────────────────────────────
    X_sep_tv, X_sep_test, y_sep_tv, y_sep_test = train_test_split(
        X_sep, y_sep, test_size=0.20, random_state=42, shuffle=False)

    X_sep_train, X_sep_val, y_sep_train, y_sep_val = train_test_split(
        X_sep_tv, y_sep_tv, test_size=0.125, random_state=42, shuffle=False)

    # ── Step 3: Combine ───────────────────────────────────────────────────────
    X_train = np.concatenate([X_nsep_train, X_sep_train], axis=0)
    y_train = np.concatenate([y_nsep_train, y_sep_train], axis=0)

    X_val   = np.concatenate([X_nsep_val,   X_sep_val],   axis=0)
    y_val   = np.concatenate([y_nsep_val,   y_sep_val],   axis=0)

    X_test  = np.concatenate([X_nsep_test,  X_sep_test],  axis=0)
    y_test  = np.concatenate([y_nsep_test,  y_sep_test],  axis=0)

    # ── Step 4: Shuffle all three splits ─────────────────────────────────────
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val,   y_val   = shuffle(X_val,   y_val,   random_state=42)
    X_test,  y_test  = shuffle(X_test,  y_test,  random_state=42)

    print(f"\nTrain : {X_train.shape[0]:,} samples  |  {Counter(y_train)}")
    print(f"Val   : {X_val.shape[0]:,}   samples  |  {Counter(y_val)}")
    print(f"Test  : {X_test.shape[0]:,}  samples  |  {Counter(y_test)}")

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    def save_bundle(X_data, y_data, filename):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump({'X': X_data, 'y': y_data}, f)
        print(f"Saved : {filepath}")

    save_bundle(X_train, y_train, 'train_set.pkl')
    save_bundle(X_val,   y_val,   'val_set.pkl')
    save_bundle(X_test,  y_test,  'test_set.pkl')

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del X_nsep, X_sep, X_train, X_val, X_test

print("\nAll done!")


# In[16]:


import shutil
import os

# ── Intermediate folders to delete ───────────────────────────────────────────
to_delete = [
    './output',
    './mvts',
    './mvts_binary',
    './mvts_zscore_per_file',
    './mvts_zscore_per_file_binary',
    './mvts_zscore_per_row',
    './mvts_zscore_per_row_binary'
]

for folder in to_delete:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"[DELETED]  {folder}")
    else:
        print(f"[SKIP]     {folder}  (not found)")

print("\nCleanup complete!")

