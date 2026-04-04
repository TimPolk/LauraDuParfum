import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
import hdbscan
import umap
import joblib
import hashlib
import os

# Mute Windows resource tracker warnings
warnings.filterwarnings('ignore', category=UserWarning)
path = "./Data/fragrance_cleaned.csv"

# (TP) Cache directory for storing computed embeddings and cluster labels
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

df = pd.read_csv(path).copy()




# Feature Groups
# (TP) Columns are already TF-trimmed in clean.py — ultra-common and ultra-rare features removed
gender = df[[c for c in df.columns if c.startswith("gender_")]]
accords = df[[c for c in df.columns if c.startswith("accord_")]]
# (TP) Separate note pyramid levels for weighted contribution
top_notes = df[[c for c in df.columns if c.startswith("top_note_")]]
mid_notes = df[[c for c in df.columns if c.startswith("middle_note_")]]
base_notes = df[[c for c in df.columns if c.startswith("base_note_")]]
rating = df[["Rating Value", "Rating Count"]]

# (TP) Print feature counts to verify TF trimming from clean.py
print(f"Feature columns: {len(gender.columns)} gender | {len(accords.columns)} accords | "
      f"{len(top_notes.columns)} top | {len(mid_notes.columns)} mid | {len(base_notes.columns)} base")

# (TP) IDF reweighting on accords — TF trimming already removed ubiquitous accords
idf_accords = np.log(len(accords) / (accords.sum(axis=0) + 1))
accords_weighted = accords * idf_accords




# (TP) PYRAMID-AWARE SIMILARITY
# Instead of cosine similarity on flattened note vectors, this compares notes
# by their position in the pyramid. Same level = 100%, adjacent = 75%, far = 50%

# (TP) Build a unified note vocabulary across all three pyramid levels
top_name_set = {c.replace("top_note_", "") for c in top_notes.columns}
mid_name_set = {c.replace("middle_note_", "") for c in mid_notes.columns}
base_name_set = {c.replace("base_note_", "") for c in base_notes.columns}
unified_notes = sorted(top_name_set | mid_name_set | base_name_set)
print(f"Unified note vocabulary: {len(unified_notes)} unique notes across all pyramid levels")

# (TP) Build level matrix: -1 = absent, 0 = top, 1 = mid, 2 = base
# If a note appears at multiple levels, top takes priority (strongest scent signal)
note_level_matrix = np.full((len(df), len(unified_notes)), -1, dtype=np.int8)

for j, note in enumerate(unified_notes):
    base_col = f"base_note_{note}"
    mid_col = f"middle_note_{note}"
    top_col = f"top_note_{note}"

    # Assign base first, then mid, then top — so top overwrites if present at multiple levels
    if base_col in df.columns:
        note_level_matrix[df[base_col].values == 1, j] = 2
    if mid_col in df.columns:
        note_level_matrix[df[mid_col].values == 1, j] = 1
    if top_col in df.columns:
        note_level_matrix[df[top_col].values == 1, j] = 0

# (TP) Proximity weights by pyramid distance: same=1.0, adjacent=0.75, far=0.5
PROXIMITY = np.array([1.0, 0.75, 0.5])

def compute_note_similarity(idx):
    query_levels = note_level_matrix[idx]                    # (n_notes,)
    query_present = query_levels >= 0                        # where query has this note

    n_query_notes = query_present.sum()
    if n_query_notes == 0:
        return np.zeros(len(df))

    # Where both query and candidate have the note
    cand_present = note_level_matrix >= 0                    # (n_perfumes, n_notes)
    both_present = query_present & cand_present              # (n_perfumes, n_notes)

    # Pyramid distance between levels
    level_diffs = np.abs(note_level_matrix - query_levels)   # (n_perfumes, n_notes)
    level_diffs = np.clip(level_diffs, 0, 2)

    # Map distance to proximity weight
    weights = PROXIMITY[level_diffs]                         # (n_perfumes, n_notes)
    weights = weights * both_present                         # zero out non-shared notes

    # Normalize by the number of notes the query has
    return weights.sum(axis=1) / n_query_notes




# (TP) GENDER COMPATIBILITY
# Same gender = 1.0, unisex involved = 0.75, opposite = 0.5
# Applied as a multiplier so wrong-gender recommendations get halved

gender_types = gender.values  # (n_perfumes, 3) -> [unisex, women, men]

def compute_gender_compat(idx):
    """Computes gender compatibility between one perfume and all others."""
    query = gender_types[idx]  # [unisex, women, men]

    # Exact same gender category
    exact_match = (gender_types * query).sum(axis=1) == 1    # both women, both men, or both unisex

    # Either side is unisex
    query_is_unisex = query[0] == 1
    cand_is_unisex = gender_types[:, 0] == 1
    unisex_involved = query_is_unisex | cand_is_unisex

    # Same = 1.0, unisex involved = 0.75, opposite = 0.5
    compat = np.where(exact_match, 1.0,
             np.where(unisex_involved, 0.75, 0.5))

    return compat




# UMAP + HDBSCAN

# (TP) Build clustering features — still uses all weighted features for UMAP
idf_top = np.log(len(top_notes) / (top_notes.sum(axis=0) + 1))
idf_mid = np.log(len(mid_notes) / (mid_notes.sum(axis=0) + 1))
idf_base = np.log(len(base_notes) / (base_notes.sum(axis=0) + 1))

w_gender, w_accords = 0.5, 1.5
w_top, w_mid, w_base = 1.2, 1.0, 0.8

X_scent = pd.concat([
    gender * w_gender,
    accords_weighted * w_accords,
    (top_notes * idf_top) * w_top,
    (mid_notes * idf_mid) * w_mid,
    (base_notes * idf_base) * w_base,
], axis=1)

# (TP) L2-normalize so Euclidean distance in HDBSCAN approximates cosine geometry
X_scent = normalize(X_scent)

print(f"Final X_scent shape: {X_scent.shape}")

# (TP) Precompute confidence factor from rating count
df['confidence'] = np.log1p(df['Rating Count'])
df['confidence'] = df['confidence'] / df['confidence'].max()





# (TP) Save/load UMAP + HDBSCAN results
def load_or_compute():
    data_hash = hashlib.md5(X_scent.tobytes()).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, f"cluster_cache_{data_hash}.joblib")

    if os.path.exists(cache_file):
        print(f"Loading cached clustering results ({cache_file})...")
        cache = joblib.load(cache_file)
        return cache['X_compressed'], cache['labels']

    print("No matching cache found — computing fresh clustering results...")

    reducer = umap.UMAP(
        n_components=23,
        metric='cosine',
        random_state=42,
        n_jobs=-1,
        n_neighbors=20,
        min_dist=0.05
    )
    X_compressed = reducer.fit_transform(X_scent)

    # (TP) Euclidean on L2-normalized vectors behaves like cosine without BallTree issues
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=3,
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.1,
        metric='euclidean',
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(X_compressed)

    joblib.dump({
        'X_compressed': X_compressed,
        'labels': labels,
    }, cache_file)
    print(f"Cached results saved to {cache_file}")

    return X_compressed, labels

X_compressed, cluster_labels = load_or_compute()
df['Cluster'] = cluster_labels

n_clusters = df['Cluster'].nunique() - 1
n_noise = (df['Cluster'] == -1).sum()
print(f"\nNumber of clusters found: {n_clusters}")
print(f"Noise points (-1): {n_noise} out of {len(df)} ({n_noise/len(df)*100:.1f}%)\n")

scaler = MinMaxScaler()
df['popularity'] = scaler.fit_transform(df[['Rating Count']])






# 3. Recommendation Function
# (TP) Uses three custom similarity components:
#   - Accord similarity (cosine on IDF-weighted accords)
#   - Pyramid-aware note similarity (same=100%, adjacent=75%, far=50%)
#   - Gender compatibility multiplier (same=1.0, unisex=0.75, opposite=0.5)

def test_recommendation_by_name(perfume_name, top_n=5, cluster_boost=0.05, min_score=0.25):
        try:
            idx = df[df['Name'].str.contains(perfume_name, case=False, na=False)].index[0]
        except IndexError:
            print(f" -> ERROR: Could not find '{perfume_name}' in the dataset.")
            return
    
        target_cluster = df.iloc[idx]['Cluster']
        target_name = df.iloc[idx]['Name']
        
        # (TP) Accord similarity — cosine on IDF-weighted accords
        accord_sim = cosine_similarity(accords_weighted.iloc[[idx]], accords_weighted)[0]

        # (TP) Pyramid-aware note similarity — shared notes weighted by pyramid distance
        note_sim = compute_note_similarity(idx)

        # (TP) Gender compatibility — same=1.0, unisex=0.75, opposite=0.5
        gender_compat = compute_gender_compat(idx)

        # (TP) Blend accord and note similarity, then apply gender as a multiplier
        scores = 0.75 * accord_sim + 0.25 * note_sim
        scores = scores * gender_compat

        # (TP) Cluster boost scaled by similarity
        final_scores = scores.copy()
        if target_cluster != -1:
            same_cluster = (df['Cluster'] == target_cluster).values.astype(float)
            final_scores += same_cluster * cluster_boost * scores

        # (TP) Confidence multiplier
        confidence = 0.2 + 0.8 * df['confidence'].values
        final_scores = final_scores * confidence

        # Zero out unrated and low-confidence perfumes
        final_scores[df['Rating Count'] == 0] = -1
        final_scores[df['Rating Count'] < 0.005] = -1

        # (TP) Dampen near-duplicates (flankers/clones from the same brand line)
        target_first_word = target_name.split()[0]
        same_line = df['Name'].str.split().str[0] == target_first_word
        final_scores[same_line] *= 0.8

        # Remove self-match and sort
        final_scores[idx] = -1
        top_idx = np.argsort(final_scores)[::-1][:top_n]

        top_idx = sorted(top_idx, key=lambda i: final_scores[i], reverse=True)
        top_idx = [i for i in top_idx if final_scores[i] >= min_score]

        if not top_idx:
            print(f"No strong matches found for '{target_name}' above threshold {min_score}.")
            return

        # Print Results
        cluster_label = f"Cluster {target_cluster}" if target_cluster != -1 else "Noise/Unclustered"
        query_gender = "unisex" if gender_types[idx][0] else ("women" if gender_types[idx][1] else "men")
        print(f"If you like '{target_name}' ({cluster_label}, {query_gender}), you might like:")
        
        for rank, i in enumerate(top_idx, start=1):
            match = df.iloc[i]
            g = "unisex" if gender_types[i][0] else ("women" if gender_types[i][1] else "men")
            print(f" {rank}. {match['Name']} [{g}] "
                  f"(Score: {final_scores[i]:.4f} | "
                  f"Accords: {accord_sim[i]:.3f} | "
                  f"Notes: {note_sim[i]:.3f} | "
                  f"Gender: {gender_compat[i]:.2f})")





# (TP) Expanded cluster diagnostics 
def evaluate_clusters(sample_size=10000):
    mask = df['Cluster'] != -1
    X_clustered = X_compressed[mask]
    labels_clustered = df.loc[mask, 'Cluster']

    print("Running cluster evaluation...")

    # Core metrics
    sil = silhouette_score(X_clustered, labels_clustered, metric='euclidean', sample_size=sample_size, random_state=42)
    ch = calinski_harabasz_score(X_clustered, labels_clustered)
    db = davies_bouldin_score(X_clustered, labels_clustered)

    print(f"Silhouette Coefficient: {sil:.4f}  (range: -1 to 1, higher = better separated)")
    print(f"Calinski-Harabasz Index: {ch:.2f}  (higher = denser clusters, more separated)")
    print(f"Davies-Bouldin Index: {db:.4f}  (lower = better separated)\n")

    # (TP) Cluster size diagnostics
    cluster_sizes = df.loc[mask, 'Cluster'].value_counts().sort_values(ascending=False)
    print(f"Cluster count: {len(cluster_sizes)}")
    print(f"Median cluster size: {cluster_sizes.median():.0f}")
    print(f"Largest cluster: {cluster_sizes.iloc[0]} points (Cluster {cluster_sizes.index[0]})")
    print(f"Smallest cluster: {cluster_sizes.iloc[-1]} points")

    # (TP) Concentration check 
    top_10 = cluster_sizes.head(10).sum()
    total_clustered = mask.sum()
    print(f"Top 10 clusters hold: {top_10}/{total_clustered} ({top_10/total_clustered*100:.1f}%)\n")




# 4. Test the recommendation
evaluate_clusters()
test_recommendation_by_name("Light Blue Dolce&Gabbana")