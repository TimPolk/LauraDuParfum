import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
import hdbscan
import umap

# Mute Windows resource tracker warnings
warnings.filterwarnings('ignore', category=UserWarning)
path = "./Data/fragrance_cleaned.csv"

df = pd.read_csv(path).copy()

# Feature Groups 
gender = df[[c for c in df.columns if c.startswith("gender_")]]
accords = df[[c for c in df.columns if c.startswith("accord_")]]
notes = df[[c for c in df.columns if c.startswith("note_")]]
rating = df[["Rating Value", "Rating Count"]]

# (T) TF-IDF-style reweighting on notes to downweight common notes like "musk"
idf = np.log(len(notes) / (notes.sum(axis=0) + 1))
notes_weighted = notes * idf

# (T) Weight feature groups so accords (strongest scent signal) dominate over raw column count
w_gender, w_accords, w_notes = 0.5, 1.5, 1.0
X_scent = pd.concat([
    gender * w_gender,
    accords * w_accords,
    notes_weighted * w_notes
], axis=1)

# 1. Dimensionality Reduction
# (T) Added n_jobs=-1 for parallel nearest-neighbor search (main UMAP bottleneck)
# (T) Lower n_neighbors (default 15) speeds up computation with minimal quality loss
reducer = umap.UMAP(
    n_components=10,
    metric='cosine',
    random_state=42,
    n_jobs=-1,
    n_neighbors=12
)
X_compressed = reducer.fit_transform(X_scent)

# 2. Clustering
# (T) Tuned for 64k dataset: eom selection merges small clusters up to reduce noise,
# cluster_selection_epsilon=0.3 allows nearby micro-clusters to merge into meaningful groups,
# min_samples=3 balances between too permissive (1) and too strict (5 which gave 58% noise)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=3,
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.3,
    metric='euclidean',
    core_dist_n_jobs=-1
)
df['Cluster'] = clusterer.fit_predict(X_compressed)

n_clusters = df['Cluster'].nunique() - 1 # Subtract 1 because -1 is the noise cluster
n_noise = (df['Cluster'] == -1).sum()
print(f"\nNumber of clusters found: {n_clusters}")
print(f"Noise points (-1): {n_noise} out of {len(df)} ({n_noise/len(df)*100:.1f}%)\n")

# (T) Precompute normalized popularity scores for use as a tiebreaker
scaler = MinMaxScaler()
df['popularity'] = scaler.fit_transform(df[['Rating Count']])

# 3. Recommendation Function
# (T) Computes similarity per-query on the 10-dim UMAP embedding (no RAM issue + fast)
def test_recommendation_by_name(perfume_name, top_n=5, cluster_boost=0.05, alpha=0.9):
        try:
            idx = df[df['Name'].str.contains(perfume_name, case=False, na=False)].index[0]
        except IndexError:
            print(f" -> ERROR: Could not find '{perfume_name}' in the dataset.")
            return
    
        target_cluster = df.iloc[idx]['Cluster']
        target_name = df.iloc[idx]['Name']
        
        # (T) Compute cosine similarity on the 10-dim UMAP embedding per query
        # On 10 dimensions this takes <10ms even for 64k rows — no need to precompute
        scores = cosine_similarity(X_compressed[[idx]], X_compressed)[0]
        
        # (T) Blend similarity + cluster boost + popularity into final ranking score
        # Cluster boost and popularity are applied to final_scores only, keeping scores as pure cosine similarity
        final_scores = scores.copy()
        if target_cluster != -1:
            same_cluster = (df['Cluster'] == target_cluster).values.astype(float)
            final_scores += same_cluster * cluster_boost
        final_scores = alpha * final_scores + (1 - alpha) * df['popularity'].values

        # Remove self-match and sort
        final_scores[idx] = -1
        top_idx = np.argsort(final_scores)[::-1][:top_n]
        
        # (T) Re-sort top results: similarity descending, then popularity as tiebreaker
        top_idx = sorted(top_idx, key=lambda i: (scores[i], df.iloc[i]['popularity']), reverse=True)
        
        # Print Results
        cluster_label = f"Cluster {target_cluster}" if target_cluster != -1 else "Noise/Unclustered"
        print(f"If you like '{target_name}' ({cluster_label}), you might like:")
        
        for rank, i in enumerate(top_idx, start=1):
            match = df.iloc[i]
            print(f" {rank}. {match['Name']} (Similarity: {scores[i]:.4f} | Popularity Score: {match['Rating Count']:.3f})")
    
# 4. Test the recommendation
test_recommendation_by_name("Light Blue Dolce&Gabbana")