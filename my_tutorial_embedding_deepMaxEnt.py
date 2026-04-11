# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:38:38 2026

@author: Warong
"""
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Geospatial libraries
import rasterio
from rasterio.windows import from_bounds

# Machine Learning
import torch
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# DeepMaxent libraries
from librairies.model import deepmaxent_model, deepmaxent_embedding_model
from librairies.losses import deepmaxent_loss
from librairies.utils import set_seed

#%%
# Load the biodiversity dataset
df = pd.read_csv('data/custom/occurrences_data.csv', sep='\t', low_memory=False)

print(f"📊 Dataset shape: {df.shape[0]:,} observations × {df.shape[1]} columns")
print(f"\n🔤 Available columns:\n{df.columns.tolist()[:15]}...")

# Display first few rows
df.head(3)

# Clean coordinates and filter valid data
df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')

# Remove rows with missing coordinates or species
df_clean = df.dropna(subset=['decimalLatitude', 'decimalLongitude', 'species'])

print(f"📍 Valid observations with coordinates: {len(df_clean):,}")
print("\n📈 Distribution by Species:")
print(df_clean['species'].value_counts())
