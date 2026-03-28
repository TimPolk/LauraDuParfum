import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

path = "./Data/fragrance_cleaned.csv"

df = pd.read_csv(path)

print(df)

