import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import hdbscan


path = "./Data/fragrance_cleaned.csv"

df = pd.read_csv(path)

# Feature Groups 
gender = df[[c for c in df.columns if c.startswith("gender_")]]
accords = df[[c for c in df.columns if c.startswith("accord_")]]
notes = df[[c for c in df.columns if c.startswith("note_")]]
rating = df[["Rating Value", "Rating Count"]]

