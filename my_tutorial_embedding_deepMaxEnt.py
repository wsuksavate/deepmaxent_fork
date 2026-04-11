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

# Machine Learning
import torch
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#%% Data Import and Cleaning
# Load the biodiversity dataset
df = pd.read_csv('data/custom/occu_data.csv', low_memory=False)

print(f"📊 Dataset shape: {df.shape[0]:,} observations × {df.shape[1]} columns")
print(f"\n🔤 Available columns:\n{df.columns.tolist()[:15]}...")

# Display first few rows
df.head(3)

# Clean coordinates and filter valid data
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

# Remove rows with missing coordinates or species
df_clean = df.dropna(subset=['lat', 'lon', 'species'])

print(f"📍 Valid observations with coordinates: {len(df_clean):,}")
print("\n📋 Distribution by Species:")
print(df_clean['species'].value_counts())

#%% Create a beautiful map showing all occurrences
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Define map extent with padding
padding = 0.5
extent = [
    df_clean['lon'].min() - padding,
    df_clean['lon'].max() + padding,
    df_clean['lat'].min() - padding,
    df_clean['lat'].max() + padding
]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
ax.add_feature(cfeature.OCEAN, facecolor='#cce5ff')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#555555')
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5, edgecolor='#888888')
ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='#99ccff')

# Plot occurrence points
scatter = ax.scatter(
    df_clean['lon'], 
    df_clean['lat'],
    c='#2ecc71',
    s=15,
    alpha=0.6,
    edgecolor='#27ae60',
    linewidth=0.3,
    transform=ccrs.PlateCarree(),
    label='Occurrences'
)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Title and legend
ax.set_title('Species Occurrences', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower left', fontsize=11)

# Add observation count annotation
ax.text(0.02, 0.1, f'n = {len(df_clean):,} observations', 
        transform=ax.transAxes, fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

#%% Define study area extent based on occurrence data
# Setup folders
rasters_dir = 'data/custom/rasters'

# List available rasters
raster_files = sorted([f for f in os.listdir(rasters_dir) if f.endswith('.tif')])

# Check the cropped raster dimensions
# Get raster properties
with rasterio.open(os.path.join(rasters_dir, raster_files[0])) as src:
    raster_resolution = abs(src.transform[0])
    raster_transform = src.transform
    raster_crs = src.crs
    
    print(f"\n📐 Raster dimensions: {src.width} × {src.height} pixels")
    print(f"📐 Raster resolution: {raster_resolution:.4f}° (~{raster_resolution * 111:.1f} km)")
    print(f"📁 Number of environmental variables: {len(raster_files)}")
    print("\n📋 Variables:")
    for i, f in enumerate(raster_files, 1):
        print(f"   {i:2d}. {f}")


#%%
# Prepare Training Data
# Round 'lat' and 'lon' columns to 3 decimal places
df_clean[['lat', 'lon']] = df_clean[['lat', 'lon']].round(3)

# Aggregate observations by unique location
# This groups all species observations at each coordinate pair
df_unique = df_clean.groupby(['lon', 'lat']).agg({'species': list  # Collect all species observed at this location
}).reset_index()

# Get unique species list and create index mapping
species_list = df_clean['species'].unique()
num_species = len(species_list)
species_to_idx = {species: idx for idx, species in enumerate(species_list)}

print(f"📊 From {len(df_clean):,} individual observations")
print(f"📍 Unique locations: {len(df_unique):,}")
print(f"🌱 Unique species: {num_species:,}")
print(f"\n📐 Aggregation ratio: {len(df_clean) / len(df_unique):.1f} observations per location (avg)")

#%% Build Training Tensors
# X tensor (environmental features): shape (n_locations, n_variables)
# y tensor (occurrence counts): shape (n_locations, n_species)

# Initialize tensors
num_locations = len(df_unique)
num_variables = len(raster_files)

# Create zeros matrices for X and y
X_tensor = torch.zeros((num_locations, num_variables), dtype=torch.float32)
y_tensor = torch.zeros((num_locations, num_species), dtype=torch.float32)

# Build occurrence count matrix (y)
print("🔢 Building occurrence count matrix...")
for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Processing locations"):
    species_at_location = row['species']
    for sp in species_at_location:
        y_tensor[idx, species_to_idx[sp]] += 1

# Extract environmental values at each location
print("\n🌡️ Extracting environmental values...")
coords = list(zip(df_unique['lon'], df_unique['lat']))

for var_idx, filename in enumerate(tqdm(raster_files, desc="Processing rasters")):
    filepath = os.path.join(rasters_dir, filename)
    
    with rasterio.open(filepath) as src:
        nodata = src.nodata
        # Convert to float first to allow NaN assignment
        values = np.array([val[0] for val in src.sample(coords)], dtype=np.float32)
        
        # Handle nodata values
        if nodata is not None:
            values[values == nodata] = np.nan
        
        X_tensor[:, var_idx] = torch.tensor(values, dtype=torch.float32)

print("\n✅ Tensor creation complete!")
print(f"   X (environmental features): {X_tensor.shape}")
print(f"   y (occurrence counts):      {y_tensor.shape}")

#%% Visualize species occurrence distribution

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# Distribution of occurrences per species
ax1 = axes[0]
species_totals = y_tensor.sum(dim=0).numpy()
species_totals_nonzero = species_totals[species_totals > 0]

ax1.hist(species_totals_nonzero, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
ax1.axvline(np.median(species_totals_nonzero), color='#e74c3c', linestyle='--', linewidth=2, 
            label=f'Median: {np.median(species_totals_nonzero):.0f}')
ax1.set_xlabel('Number of Occurrences', fontsize=12)
ax1.set_ylabel('Number of Species', fontsize=12)
ax1.set_title('Distribution of Occurrences per Species', fontsize=13, fontweight='bold')
ax1.legend()
ax1.set_yscale('log')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Distribution of species richness per location
ax2 = axes[1]
richness_per_location = (y_tensor > 0).sum(dim=1).numpy()

ax2.hist(richness_per_location, bins=30, color='#2ecc71', edgecolor='white', alpha=0.8)
ax2.axvline(np.median(richness_per_location), color='#e74c3c', linestyle='--', linewidth=2,
            label=f'Median: {np.median(richness_per_location):.0f}')
ax2.set_xlabel('Number of Species', fontsize=12)
ax2.set_ylabel('Number of Locations', fontsize=12)
ax2.set_title('Species Richness per Location', fontsize=13, fontweight='bold')
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

#%%Map of Species Richness

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Calculate species richness per location
richness = (y_tensor > 0).sum(dim=1).numpy()

# Map extent
padding = 0.5
extent = [
    df_unique['lon'].min() - padding,
    df_unique['lon'].max() + padding,
    df_unique['lat'].min() - padding,
    df_unique['lat'].max() + padding
]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add features
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

# Create colormap
scatter = ax.scatter(
    df_unique['lon'],
    df_unique['lat'],
    c=richness,
    cmap='viridis',
    s=20 + richness * 5,  # Size proportional to richness
    alpha=0.7,
    edgecolor='white',
    linewidth=0.3,
    transform=ccrs.PlateCarree()
)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Number of Species', fontsize=11)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax.set_title('Species Richness per Location\n(Size proportional to richness)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

#%% Spatial Block Split for Train/Validation
# Spatial Block Split
# We divide the study area into longitudinal bands to reduce spatial autocorrelation
# Validation bands: 20th-30th percentile AND 70th-80th percentile

# Get coordinates from df_unique
longitudes = df_unique['lon'].values
latitudes = df_unique['lat'].values

# Calculate all necessary percentile thresholds simultaneously
p20, p30, p70, p80 = np.percentile(longitudes, [20, 30, 70, 80])

# Create validation mask using bitwise OR for the two distinct geographic bands
val_mask = ((longitudes >= p20) & (longitudes < p30)) | \
           ((longitudes >= p70) & (longitudes < p80))

# Create train/validation masks
train_mask = ~val_mask

# Split the tensors
X_train = X_tensor[train_mask]
y_train = y_tensor[train_mask]
X_val = X_tensor[val_mask]
y_val = y_tensor[val_mask]

# --- Analytical Validation Printouts ---
print("🔀 SPATIAL BAND SPLIT")
print("=" * 50)
print(f"📍 Band 1 (Lon): {p20:.4f}° to {p30:.4f}°")
print(f"📍 Band 2 (Lon): {p70:.4f}° to {p80:.4f}°")
print("\n📊 Split statistics:")
print(f"   Training set:   {train_mask.sum():,} locations ({100*train_mask.mean():.1f}%)")
print(f"   Validation set: {val_mask.sum():,} locations ({100*val_mask.mean():.1f}%)")

#%% Plot Train / Validation set

# Get coordinates for plotting
train_coords = df_unique[train_mask][['lon', 'lat']].values
val_coords = df_unique[val_mask][['lon', 'lat']].values

print("\n📐 Tensor shapes:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_val:   {X_val.shape}")
print(f"   y_val:   {y_val.shape}")

# Visualize the spatial split on a map
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Map extent
padding = 0.5
extent = [
    df_unique['lon'].min() - padding,
    df_unique['lon'].max() + padding,
    df_unique['lat'].min() - padding,
    df_unique['lat'].max() + padding
]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

# Plot training points (blue)
ax.scatter(train_coords[:, 0], train_coords[:, 1],
           c='#3498db', s=15, alpha=0.6, label=f'Training (n={len(train_coords):,})',
           transform=ccrs.PlateCarree(), edgecolor='white', linewidth=0.2)

# Plot validation points (red)
ax.scatter(val_coords[:, 0], val_coords[:, 1],
           c='#e74c3c', s=15, alpha=0.6, label=f'Validation (n={len(val_coords):,})',
           transform=ccrs.PlateCarree(), edgecolor='white', linewidth=0.2)

# Draw the split line using plot instead of axvline (works with cartopy projections)
lat_min, lat_max = df_unique['lat'].min() - padding, df_unique['lat'].max() + padding

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax.set_title('Block Split: Training vs Validation Sets\n(Based on Longitude)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.show()

#%% DeepMaxent Model Training
# Data Preprocessing
# Before training, we need to:

# Handle missing values (NaN) in the environmental features
# Normalize the features using StandardScaler

# Handle NaN values - remove locations with missing environmental data
train_nan_mask = torch.isnan(X_train).any(dim=1)
val_nan_mask = torch.isnan(X_val).any(dim=1)

X_train_clean = X_train[~train_nan_mask]
y_train_clean = y_train[~train_nan_mask]
X_val_clean = X_val[~val_nan_mask]
y_val_clean = y_val[~val_nan_mask]

print("🧹 Removing locations with NaN values...")
print(f"   Training: {train_nan_mask.sum().item()} locations removed → {X_train_clean.shape[0]:,} remaining")
print(f"   Validation: {val_nan_mask.sum().item()} locations removed → {X_val_clean.shape[0]:,} remaining")

# Normalize features using StandardScaler
scaler = StandardScaler()

# Fit on training data only
X_train_np = X_train_clean.numpy()
X_val_np = X_val_clean.numpy()

X_train_scaled = scaler.fit_transform(X_train_np)
X_val_scaled = scaler.transform(X_val_np)

# Convert back to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = y_train_clean
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = y_val_clean

print("\n✅ Data preprocessing complete!")
print(f"   X_train_tensor: {X_train_tensor.shape}")
print(f"   y_train_tensor: {y_train_tensor.shape}")
print(f"   X_val_tensor:   {X_val_tensor.shape}")
print(f"   y_val_tensor:   {y_val_tensor.shape}")

#%% Define Training Configuration and Model# Additional imports for training

# Training configuration using a simple namespace
class Args:
    def __init__(self):
        self.learning_rate = 0.001
        self.epoch = 3000
        self.hidden_nbr = 3  # Number of hidden layers
        self.weight_decay = 1e-4  # L2 regularization

args = Args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
print(torch.cuda.get_device_name(0))

# Model parameters
input_size = X_train_tensor.shape[1]
output_size = y_train_tensor.shape[1]
hidden_size = 30
batch_size = 150

print("\n🧠 Model Architecture:")
print(f"   Input size:  {input_size} (environmental variables)")
print(f"   Hidden size: {hidden_size}")
print(f"   Output size: {output_size} (species)")
print(f"   Hidden layers: {args.hidden_nbr}")

print("\n⚙️ Training Configuration:")
print(f"   Learning rate: {args.learning_rate}")
print(f"   Epochs: {args.epoch}")
print(f"   Weight decay: {args.weight_decay}")
print(f"   Batch size: {batch_size}")

#%% Training Loop with Validation AUC
# DeepMaxent loss for optimization
# Validation AUC computed at each epoch to monitor performance
# Best model selection based on validation loss

# DeepMaxent libraries
from librairies.train_models_custom import train_deepmodel

# Train model
results = train_deepmodel(
    X_train_tensor, 
    y_train_tensor, 
    X_val_tensor, 
    y_val_tensor, 
    args, 
    hidden_size=hidden_size, 
    device=device,
    sp_embedding = True
)