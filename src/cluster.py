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

X_scent = pd.concat([gender, accords, notes], axis=1)

# 1. Dimensionality Reduction
reducer = umap.UMAP(n_components=10, metric='cosine', random_state=42)
X_compressed = reducer.fit_transform(X_scent)

# 2. Clustering
print("Running HDBSCAN Clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1, cluster_selection_epsilon=0.5, metric='euclidean', core_dist_n_jobs=-1)
df['Cluster'] = clusterer.fit_predict(X_compressed)

n_clusters = df['Cluster'].nunique() - 1 # Subtract 1 because -1 is the noise cluster
n_noise = (df['Cluster'] == -1).sum()
print(f"Number of clusters found: {n_clusters}")
print(f"Noise points (-1): {n_noise} out of {len(df)}\n")

# 3. RECOMMENDATION FUNCTION
def test_recommendation_by_name(perfume_name, top_n=5):
        try:
            idx = df[df['Name'].str.contains(perfume_name, case=False, na=False)].index[0]
        except IndexError:
            print(f" -> ERROR: Could not find '{perfume_name}' in the dataset.")
            return
    
        target_cluster = df.iloc[idx]['Cluster']
        target_name = df.iloc[idx]['Name']
        
        # Filter candidates to only those in the exact same cluster 
        # (If HDBSCAN labeled it as noise (-1), search the whole dataset as a fallback)
        if target_cluster != -1:
            candidates_idx = df[df['Cluster'] == target_cluster].index
        else:
            candidates_idx = df.index
            
        # Extract features for the target and the filtered candidates
        target_features = X_scent.iloc[[idx]]
        candidate_features = X_scent.iloc[candidates_idx]
        
        # Calculate Cosine Similarity ONLY on the scent features
        scores = cosine_similarity(target_features, candidate_features)[0] 
        
        # Zip the indices with their scores, remove the exact self-match, and sort
        scores_list = list(zip(candidates_idx, scores))
        scores_list = [x for x in scores_list if x[0] != idx] 
        sorted_scores = sorted(scores_list, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Print Results
        cluster_label = f"Cluster {target_cluster}" if target_cluster != -1 else "Noise/Unclustered"
        print(f"If you like '{target_name}' ({cluster_label}), you might like:")
        
        for i, score in sorted_scores:
            match = df.iloc[i]
            # displays the ratings as helpful text without them ruining the math
            print(f" - {match['Name']} (Similarity: {score:.4f} | Popularity Score: {match['Rating Count']:.3f})")
    
# 4. Test the recommendation
test_recommendation_by_name("9am Afnan")