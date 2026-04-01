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
- In the terminal type in `chmod +x packages.sh`
- To download all the packages type 
  - `./packages.sh` 
- This will run a script to download all the packages
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

## How our clustering works

Our clustering pipeline transforms raw perfume data into similarity scores through five stages.
 
### 1. Multi-hot encoding
 
Each perfume is represented as rows of binary features across three separate groups: "gender_", "accords_", and "notes_". The cleaning script (`clean.py`) handles this encoding using a shared vocabulary built from the entire dataset.
 
### 2. IDF reweighting on notes
 
Not all notes are equally informative. A note like "musk" appears in thousands of fragrances and tells you very little, while "saffron" is rare and distinctive. We apply an Inverse Document Frequency (IDF) weight to each note column:
 
```
idf(note) = ln(total_perfumes / (perfumes_with_that_note + 1))
```
 
Common notes get pushed towards zero. Rare notes are amplified closer to 1. This ensures that rare notes contribute more to similarity than common ones.
 
### 3. Feature group weighting
 
After encoding, notes typically produce hundreds of columns while accords produce around 20-30. Without correction, notes would dominate the similarity calculation purely by column count. We apply explicit group weights before combining:
 
| Group   | Weight | Reasoning |
|---------|--------|-----------|
| Gender  | 0.5    | Weak signal — many unisex fragrances cross gender lines |
| Accords | 1.5    | Strongest signal — describes the overall scent character |
| Notes   | 1.0    | Useful but noisy — parsed from description text, not curated |
 
The weighted features are concatenated into a single feature matrix.
 
### 4. UMAP dimensionality reduction
 
The combined feature matrix has many columns, which is expensive and hard to compare (high dimensionality curse: in very high dimensions data points become muddled). UMAP (Uniform Manifold Approximation and Projection) compresses this down to 10 dimensions and still preserves the integrity of the data.
 
```
reducer = umap.UMAP(n_components=10, metric='cosine', n_jobs=-1)
X_compressed = reducer.fit_transform(X_scent)
```
 
The resulting 10 values per perfume are abstract coordinates that can't be interpreted individually, but distances between them are meaningful.
 
### 5. Cosine similarity
 
To compare two perfumes, we compute the cosine of the angle between their 10-dimensional vectors:
 
```
dot_product = (a[0]*b[0]) + (a[1]*b[1]) + ... + (a[9]*b[9])
magnitude_a = sqrt(a[0]**2 + a[1]**2 + ... + a[9]**2)
magnitude_b = sqrt(b[0]**2 + b[1]**2 + ... + b[9]**2)
 
cosine_similarity = dot_product / (magnitude_a * magnitude_b)
```
 
Cosine similarity ranges from -1 to 1 (1 being identical). It measures the direction of two vectors point, not looking at their magnitude, which means two perfumes with the same scent profile but different rating counts will still score as highly similar.
 
### Clustering with HDBSCAN
 
Before recommendation, we group perfumes into scent families using HDBSCAN (Hierarchical Density-Based Spatial Clustering). This identifies natural groupings (fresh citrus, warm orientals, woody ouds, etc.) unlike other clustering algorithms like k-means clustering which require us to specify a number for clusters before. Perfumes in the same cluster get a slight similarity boost for ranking, but the cluster boundary is soft allowing a different perfume in a different cluster with a high similarity score beat it out.
 
### Final ranking
 
Each candidate perfume receives a blended score:
 
```
alpha = 0.9
final_score = alpha * cosine_similarity + (1 - alpha) * popularity
```
 
Where `alpha = 0.9` keeps scent similarity dominant and popularity acts only as a tiebreaker. So the results are ordered by similarity first and foremost. 

