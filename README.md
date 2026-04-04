# L'aura Du Parfum

## Preparing environment

### To continue with a virtual environment \(.venv\)
- Should have the latest Python installed or Python >= 3.4
- In terminal you will type out `python3 -m venv name_of_your_venv` (I suggest using a file that starts with a '.' to avoid git adding it to your commits and pushes)
- In the terminal type in `chmod +x packages.sh`
- To download all the packages type 
  - `./packages.sh` 
- This will run a script to download all the packages

### Downloading Packages without a venv
Run `pip install -r requirements.txt`
## Dataset download

### Option 1 - manually download
- Head to the kaggle dataset -- https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset?select=fra_cleaned.csv
- Click the download and look down until you see "Download dataset as zip."
- Unzip file into `Data` directory
### Option 2 - CLI
#### Prerequisites
- Log in to your Kaggle account
- Go to **Account** settings
- Scroll down to the "API" section 
- Select create Legacy API key, which will download a "kaggle.json" file
- In the terminal 
  - `mkdir -p ~/.kaggle`
  - `mv /path/to/downloaded/kaggle.json ~/.kaggle`
  - `chmod 600 ~/.kaggle/kaggle.json`
#### Downloading the files
- Once more into the terminal you will now type in:
  - `cd Data`
  - `kaggle datasets download olgagmiufana1/fragrantica-com-fragrance-dataset`
- To automatically unzip the files add `--unzip` to the end of it.

## Our teams cleaned dataset
To be able to see the results we are getting use our teams cleaned dataset. To obtain this run in the terminal:
- `python3 src/clean.py`
- You will then see it appear in the `Data` directory as the name `fragrance_cleaned.csv`.

### What clean.py does

The cleaning script takes the raw Fragrantica dataset and transforms it into a model-ready format through several steps.

Rows with no usable description are dropped entirely since we cannot extract note information from them. Gender labels embedded in the perfume name are stripped out and encoded separately as binary columns (`gender_unisex`, `gender_women`, `gender_men`).

Notes are extracted from the description text by parsing "top notes are", "middle notes are", and "base notes are" patterns. Instead of pooling all notes into a single flat list, they are separated into three pyramid levels with their own column prefixes: `top_note_`, `middle_note_`, and `base_note_`. This preserves where each note sits in the scent pyramid so the clustering model can weight them differently.

Main accords are parsed from the accords list and multi-hot encoded with the `accord_` prefix.

After encoding, a TF-based frequency trim is applied to all note and accord columns. Features that appear in more than 40% of perfumes are too common to be distinctive, and features that appear in less than 2% are too rare to be a reliable signal. Both are removed. This reduces dimensionality and removes noise on both ends before the data reaches the clustering pipeline.

Rating Value and Rating Count are normalized with MinMaxScaler so they sit on a 0–1 scale.

## How our clustering works

Our recommendation pipeline transforms cleaned perfume data into ranked suggestions through several stages.

### 1. Multi-hot encoding

Each perfume is represented as rows of binary features across separate groups: `gender_`, `accord_`, `top_note_`, `middle_note_`, and `base_note_`. The cleaning script (`clean.py`) handles this encoding using a shared vocabulary built from the entire dataset, with separate vocabularies per pyramid level so `top_note_lemon` and `base_note_lemon` are distinct columns.

### 2. IDF reweighting

Not all features are equally informative. A note like "musk" or an accord like "woody" appears in thousands of fragrances and tells you very little, while "saffron" is rare and distinctive. We apply an Inverse Document Frequency (IDF) weight to both accord and note columns:

```
idf_weight = log(total_perfumes / (perfumes_with_that_feature + 1))
```

Common features get pushed toward zero. Rare features are amplified. This is applied independently to accords and to each note pyramid level (top, middle, base), so rarity is measured within each group.

### 3. Feature group weighting

After IDF reweighting, we apply explicit group weights to control how much each feature type contributes. Notes are weighted by their position in the scent pyramid since top notes define first impression and base notes are the least distinctive for perceived similarity.

| Group        | Weight | Reasoning |
|--------------|--------|-----------|
| Gender       | 0.5    | Weak signal — many unisex fragrances cross gender lines |
| Accords      | 1.5    | Strongest signal — describes the overall scent character |
| Top notes    | 1.2    | First impression — most distinctive for perceived similarity |
| Middle notes | 1.0    | Heart of the fragrance — solid signal |
| Base notes   | 0.8    | Lingering dry-down — least distinctive for quick comparison |

The weighted features are concatenated into a single feature matrix and then L2-normalized. Normalizing the vectors means Euclidean distance in the embedding space approximates cosine geometry, which keeps UMAP and HDBSCAN aligned without compatibility issues.

### 4. UMAP dimensionality reduction

The combined feature matrix is compressed using UMAP (Uniform Manifold Approximation and Projection) down to 20 dimensions while preserving neighborhood structure. This makes clustering feasible on a dataset of 40k+ perfumes without hitting the curse of dimensionality where all points start to look equidistant in high-dimensional sparse binary data.

```
reducer = umap.UMAP(n_components=20, metric='cosine', n_jobs=-1, n_neighbors=20, min_dist=0.1)
X_compressed = reducer.fit_transform(X_scent)
```

Results are cached to disk using joblib with a hash of the input data. If the data or weights haven't changed, subsequent runs load the cached embedding instead of recomputing.

### 5. Clustering with HDBSCAN

Perfumes are grouped into scent families using HDBSCAN (Hierarchical Density-Based Spatial Clustering). Unlike k-means which requires a predefined cluster count, HDBSCAN discovers natural groupings by density and labels uncertain points as noise rather than forcing them into bad clusters. Perfumes in the same cluster receive a small similarity boost during ranking, but the boundary is soft — a strong match in a different cluster will still rank above a weak match in the same cluster.

### 6. Pyramid-aware note similarity

Instead of computing a single cosine similarity on a flattened note vector, we compare notes by their position in the pyramid. If two perfumes share a note at the same pyramid level, that match counts at full weight. If the note appears at adjacent levels it counts at 75%, and if it spans the full distance (top in one, base in another) it counts at 50%.

```
proximity_weights = [1.0, 0.75, 0.5]  # same level, adjacent, far
```

This captures how perfumers think about scent composition — two perfumes with cinnamon as a top note are more alike than one with cinnamon on top and another with cinnamon buried in the base.

### 7. Gender compatibility

Gender is applied as a multiplier on the final score rather than a small additive feature. Same gender matches get full weight, either side being unisex gets 75%, and opposite genders get halved. This prevents a men's cologne from dominating recommendations for a women's perfume just because they share some accords.

```
# same gender = 1.0, unisex involved = 0.75, opposite = 0.5
scores = scores * gender_compatibility
```

### 8. Final ranking

Each candidate perfume's final score is a blend of accord similarity (cosine on IDF-weighted accords) and pyramid-aware note similarity, multiplied by gender compatibility and a confidence factor derived from rating count. Perfumes with very few ratings get their scores scaled down so obscure unreviewed fragrances don't dominate over well-established ones.

```
scores = 0.60 * accord_similarity + 0.40 * note_similarity
scores = scores * gender_compatibility
final_scores = scores * confidence
```

Results are then filtered by a minimum score threshold so recommendations are only shown when the match quality is strong enough to be meaningful.

### Evaluation

Cluster quality is measured with three complementary metrics: Silhouette Coefficient (how well each point fits its cluster vs neighbors), Calinski-Harabasz Index (ratio of between-cluster to within-cluster spread), and Davies-Bouldin Index (average similarity between each cluster and its most similar neighbor). Alongside these, we track cluster size diagnostics like median size, largest cluster, and concentration in the top 10 clusters to catch structural problems the core metrics can miss. Recommendation quality is validated manually against a benchmark set of known perfumes and their expected scent families.