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

#%% Filter out species that have fewer than 5 observations
min_count = 100

# Calculate the frequency of each species
species_counts = df_clean['species'].value_counts()
# Identify species that meet the minimum threshold (>= min_count)
valid_species = species_counts[species_counts >= min_count].index

# Filter the dataframe to keep only those valid species and create a clean copy
df_clean = df_clean[df_clean['species'].isin(valid_species)].copy()

# Analytical printouts to verify the data reduction
species_removed = len(species_counts) - len(valid_species)
print("\n🧹 RARE SPECIES FILTERING")
print("=" * 50)
print(f"📉 Species removed (n < 5): {species_removed:,}")
print(f"🌿 Valid observations remaining: {len(df_clean):,}")
print(f"🌱 Unique species remaining: {df_clean['species'].nunique():,}")
print("\n📋 Distribution by Species (Top 10):")
print(df_clean['species'].value_counts().head(10))

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
from librairies.utils import Args, compute_auc

args = Args(learning_rate = 0.001,
            epoch = 2500,
            hidden_nbr = 4, # Number of hidden layers
            weight_decay = 1e-4) # L2 regularization

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
print(torch.cuda.get_device_name(0))

# Model parameters
input_size = X_train_tensor.shape[1]
output_size = y_train_tensor.shape[1]
hidden_size = 50 # size of each hidden layer
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

#%% Visualize training progress
# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1 = axes[0]
epochs = range(1, len(results['train_losses']) + 1)
ax1.plot(epochs, results['train_losses'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
ax1.plot(epochs, results['val_losses'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('DeepMaxent Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# AUC curves
ax2 = axes[1]
train_auc_epochs = [x[0] + 1 for x in results['train_aucs']]
train_auc_values = [x[1] for x in results['train_aucs']]
val_auc_epochs = [x[0] + 1 for x in results['val_aucs']]
val_auc_values = [x[1] for x in results['val_aucs']]

ax2.plot(train_auc_epochs, train_auc_values, 'b-o', linewidth=2, markersize=6, 
         label='Training AUC', alpha=0.8)
ax2.plot(val_auc_epochs, val_auc_values, 'r-o', linewidth=2, markersize=6, 
         label='Validation AUC', alpha=0.8)
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (AUC=0.5)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Mean AUC', fontsize=12)
ax2.set_title('AUC Evolution During Training', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.4, 1.0])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Print final metrics
print("\n📊 FINAL METRICS")
print("=" * 50)
print(f"   Final Training Loss:   {results['train_losses'][-1]:.4f}")
print(f"   Final Validation Loss: {results['val_losses'][-1]:.4f}")
print(f"   Best Validation Loss:  {results['best_val_loss']:.4f}")
print(f"\n   Final Training AUC:    {train_auc_values[-1]:.4f}")
print(f"   Final Validation AUC:  {val_auc_values[-1]:.4f}")

#%% Detailed AUC Analysis per Species

# Compute detailed AUC for validation set
val_mean_auc, val_species_aucs = compute_auc(results['model'], X_val_tensor, y_val_tensor, device)

# Distribution of AUC values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of AUC values
ax1 = axes[0]
ax1.hist(val_species_aucs, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
ax1.axvline(x=np.mean(val_species_aucs), color='#e74c3c', linestyle='--', linewidth=2,
            label=f'Mean AUC: {np.mean(val_species_aucs):.3f}')
ax1.axvline(x=np.median(val_species_aucs), color='#27ae60', linestyle='--', linewidth=2,
            label=f'Median AUC: {np.median(val_species_aucs):.3f}')
ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Random (0.5)')
ax1.set_xlabel('AUC Score', fontsize=12)
ax1.set_ylabel('Number of Species', fontsize=12)
ax1.set_title('Distribution of Validation AUC per Species', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Box plot
ax2 = axes[1]
box = ax2.boxplot(val_species_aucs, vert=True, patch_artist=True)
box['boxes'][0].set_facecolor('#3498db')
box['boxes'][0].set_alpha(0.7)
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')
ax2.set_ylabel('AUC Score', fontsize=12)
ax2.set_title('AUC Distribution (Box Plot)', fontsize=13, fontweight='bold')
ax2.set_xticklabels(['Validation Set'])
ax2.legend(fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n📊 VALIDATION AUC SUMMARY")
print("=" * 50)
print(f"   Number of species evaluated: {len(val_species_aucs)}")
print(f"   Mean AUC:   {np.mean(val_species_aucs):.4f}")
print(f"   Median AUC: {np.median(val_species_aucs):.4f}")
print(f"   Std AUC:    {np.std(val_species_aucs):.4f}")
print(f"   Min AUC:    {np.min(val_species_aucs):.4f}")
print(f"   Max AUC:    {np.max(val_species_aucs):.4f}")
print(f"\n   Species with AUC > 0.7: {sum(np.array(val_species_aucs) > 0.7)} ({100*sum(np.array(val_species_aucs) > 0.7)/len(val_species_aucs):.1f}%)")
print(f"   Species with AUC > 0.5: {sum(np.array(val_species_aucs) > 0.5)} ({100*sum(np.array(val_species_aucs) > 0.5)/len(val_species_aucs):.1f}%)")

#%% Generate Species Suitability Maps

# Load all rasters and stack them in the correct order
print("📂 Loading rasters in training order...")
print(f"   Raster directory: {rasters_dir}")
print(f"   Number of variables: {len(raster_files)}")

# Load the first raster to get metadata
with rasterio.open(os.path.join(rasters_dir, raster_files[0])) as src:
    raster_height = src.height
    raster_width = src.width
    raster_transform_map = src.transform
    raster_crs_map = src.crs
    raster_bounds = src.bounds

print(f"\n📐 Raster dimensions: {raster_width} × {raster_height} pixels")
print(f"   Total pixels: {raster_width * raster_height:,}")

# Stack all rasters in the same order as training
raster_stack = np.zeros((len(raster_files), raster_height, raster_width), dtype=np.float32)

for i, filename in enumerate(tqdm(raster_files, desc="Loading rasters")):
    filepath = os.path.join(rasters_dir, filename)
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        nodata_val = src.nodata
        
        # Replace nodata with NaN
        if nodata_val is not None:
            data[data == nodata_val] = np.nan
        # Also mask extreme values
        data[np.abs(data) > 1e10] = np.nan
        
        raster_stack[i] = data

print(f"\n✅ Raster stack shape: {raster_stack.shape}")
print("   Order of variables:")
for i, f in enumerate(raster_files):
    print(f"   {i}: {f}")
    
# Reshape raster stack for model prediction
# From (n_variables, height, width) to (n_pixels, n_variables)
n_pixels = raster_height * raster_width

# Reshape: (variables, height, width) -> (height*width, variables)
X_raster = raster_stack.reshape(len(raster_files), -1).T  # Shape: (n_pixels, n_variables)

print(f"📐 Reshaped for prediction: {X_raster.shape}")

# Create mask for valid pixels (no NaN in any variable)
valid_pixel_mask = ~np.isnan(X_raster).any(axis=1)
n_valid_pixels = valid_pixel_mask.sum()

print(f"   Valid pixels: {n_valid_pixels:,} / {n_pixels:,} ({100*n_valid_pixels/n_pixels:.1f}%)")

# Extract only valid pixels for prediction
X_valid = X_raster[valid_pixel_mask]

# Apply the same scaler used during training
print("\n🔄 Applying StandardScaler (same as training)...")
X_valid_scaled = scaler.transform(X_valid)

# Convert to tensor
X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)
print(f"   Tensor shape for prediction: {X_valid_tensor.shape}")

#%% Run model prediction on the raster data
print("🧠 Running model prediction on raster pixels...")

model = results['model']
model.eval()
model = model.to(device)

# Predict in batches to avoid memory issues
batch_size = 10000
n_batches = (len(X_valid_tensor) + batch_size - 1) // batch_size

predictions_list = []

with torch.no_grad():
    for i in tqdm(range(n_batches), desc="Predicting"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_valid_tensor))
        
        batch = X_valid_tensor[start_idx:end_idx].to(device)
        batch_pred = model(batch).cpu()
        predictions_list.append(batch_pred)

# Concatenate all predictions
predictions_valid = torch.cat(predictions_list, dim=0)

# Apply softmax to get suitability probabilities (along spatial dimension)
# This gives relative suitability across space for each species
suitability_valid = torch.softmax(predictions_valid, dim=0).numpy()

print("\n✅ Predictions complete!")
print(f"   Suitability shape: {suitability_valid.shape}")
print(f"   (n_valid_pixels, n_species) = ({n_valid_pixels:,}, {num_species:,})")

#%% Reconstruct full suitability maps (including NaN pixels)
print("🗺️ Reconstructing suitability maps...")

# Initialize full suitability array with NaN
suitability_maps = np.full((num_species, n_pixels), np.nan, dtype=np.float32)

# Fill in valid pixels
suitability_maps[:, valid_pixel_mask] = suitability_valid.T  # Transpose to (species, pixels)

# Reshape to (n_species, height, width)
suitability_maps = suitability_maps.reshape(num_species, raster_height, raster_width)

print(f"✅ Suitability maps shape: {suitability_maps.shape}")
print(f"   (n_species, height, width) = ({num_species:,}, {raster_height}, {raster_width})")

# Create species index to name mapping
idx_to_species = {idx: species for species, idx in species_to_idx.items()}

# Find top species by number of occurrences for visualization
species_occurrence_counts = y_tensor.sum(dim=0).numpy()
top_species_indices = np.argsort(species_occurrence_counts)[::-1][:10]

print("\n📊 Top 10 species by occurrence count:")
for rank, sp_idx in enumerate(top_species_indices, 1):
    sp_name = idx_to_species[sp_idx]
    count = int(species_occurrence_counts[sp_idx])
    print(f"   {rank:2d}. {sp_name}: {count} occurrences")
    
#%% Visulize suitabulity maps
# Plot suitability maps for top 4 species
fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

# Raster extent for plotting
extent_map = [raster_bounds.left, raster_bounds.right, raster_bounds.bottom, raster_bounds.top]

for ax, sp_idx in zip(axes, top_species_indices[:4]):
    sp_name = idx_to_species[sp_idx]
    suitability = suitability_maps[sp_idx]
    
    # Mask NaN values
    suitability_masked = np.ma.masked_invalid(suitability)
    
    # Normalize for better visualization (0-1 range based on this species)
    suit_min = np.nanmin(suitability)
    suit_max = np.nanmax(suitability)
    if suit_max > suit_min:
        suitability_norm = (suitability_masked - suit_min) / (suit_max - suit_min)
    else:
        suitability_norm = suitability_masked
    
    # Set map style (no ocean/lakes)
    ax.set_facecolor('honeydew')
    oceans = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='none', facecolor='lightblue')
    ax.add_feature(oceans, zorder=2)
    lakes = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='none', facecolor='lightblue')
    ax.add_feature(lakes, zorder=2)
    ax.add_feature(cfeature.BORDERS, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    
    # Plot suitability map
    im = ax.imshow(suitability_norm, extent=extent_map, origin='upper', 
                   cmap='YlOrRd', transform=ccrs.PlateCarree(), vmin=0, vmax=1, zorder=1)
    
    # Get occurrence points for this species
    sp_occurrences = y_tensor[:, sp_idx].numpy()
    has_occurrence = sp_occurrences > 0
    if has_occurrence.any():
        occ_lons = df_unique['lon'].values[has_occurrence]
        occ_lats = df_unique['lat'].values[has_occurrence]
        ax.scatter(occ_lons, occ_lats, c='blue', s=10, alpha=0.6, 
                   transform=ccrs.PlateCarree(), edgecolor='white', linewidth=0.3,
                   label=f'Occurrences (n={has_occurrence.sum()})', zorder=3)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Relative Suitability', fontsize=9)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 8}
    gl.ylabel_style = {'fontsize': 8}
    
    # Title with species name in italic
    space_char = r"\ " # Use raw string to avoid escape issues
    ax.set_title(f'$\it{{{sp_name.replace(" ", space_char)}}}$\n({int(species_occurrence_counts[sp_idx])} occurrences)')
    ax.legend(loc='lower left', fontsize=8)

plt.suptitle('Predicted Suitability Maps for Top 4 Species\n(Blue dots = observed occurrences)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

#%% Species Richness Map
# Compute predicted a sum of suitability across species(sum of suitabilities)
# This gives an indication of how many species are predicted to be suitable at each location

# Sum suitabilities across all species
predicted_richness = np.nansum(suitability_maps, axis=0)
predicted_richness_masked = np.ma.masked_invalid(predicted_richness)

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set map style (no ocean/lakes visible over land)
ax.set_facecolor('honeydew')
oceans = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='none', facecolor='lightblue')
ax.add_feature(oceans, zorder=2)
lakes = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='none', facecolor='lightblue')
ax.add_feature(lakes, zorder=2)
ax.add_feature(cfeature.BORDERS, zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)

# Plot predicted richness
im = ax.imshow(predicted_richness_masked, extent=extent_map, origin='upper', 
               cmap='Spectral_r', transform=ccrs.PlateCarree(), zorder=1)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Sum of Suitability Scores', fontsize=11)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax.set_title('Sum of suitability across all species', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("Predicted Richness Statistics:")
print(f"   Min: {np.nanmin(predicted_richness):.4f}")
print(f"   Max: {np.nanmax(predicted_richness):.4f}")
print(f"   Mean: {np.nanmean(predicted_richness):.4f}")

#%% Save Suitability Maps as GeoTIFF (Optional)

# Create output directory for suitability maps
output_suitability_dir = 'output/suitability_maps'
os.makedirs(output_suitability_dir, exist_ok=True)

# Save suitability maps for top 10 species
print("💾 Saving suitability maps as GeoTIFF...")

for sp_idx in tqdm(top_species_indices[:10], desc="Saving"):
    sp_name = idx_to_species[sp_idx]
    # Clean species name for filename
    sp_name_clean = sp_name.replace(' ', '_').replace('.', '')
    
    suitability = suitability_maps[sp_idx]
    
    # Define output path
    output_path = os.path.join(output_suitability_dir, f'suitability_{sp_name_clean}.tif')
    
    # Write GeoTIFF
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=raster_height,
        width=raster_width,
        count=1,
        dtype=np.float32,
        crs=raster_crs_map,
        transform=raster_transform_map,
        nodata=np.nan
    ) as dst:
        dst.write(suitability, 1)

# Also save the species richness map
richness_path = os.path.join(output_suitability_dir, 'predicted_species_richness.tif')
with rasterio.open(
    richness_path, 'w',
    driver='GTiff',
    height=raster_height,
    width=raster_width,
    count=1,
    dtype=np.float32,
    crs=raster_crs_map,
    transform=raster_transform_map,
    nodata=np.nan
) as dst:
    dst.write(predicted_richness.astype(np.float32), 1)

print(f"\n✅ Saved {len(top_species_indices[:10]) + 1} GeoTIFF files to: {output_suitability_dir}")
print("   - 10 species suitability maps")
print("   - 1 predicted species richness map")

#%% Get embedding layer
import pandas as pd

# Extract the trained model from your results dictionary
trained_model = results['model']

# Access the embedding layer's weights and convert to a NumPy array
# .detach()   -> Removes the tensor from the gradient computation graph
# .cpu()      -> Moves the tensor from the GPU (if used) to the CPU
# .numpy()    -> Converts the PyTorch tensor into a standard NumPy array
learned_embeddings = trained_model.species_embedding.weight.detach().cpu().numpy()

# Assuming you still have your 'species_to_idx' dictionary from data prep
# Create a list of species names ordered by their index
ordered_species_names = list(species_to_idx.keys())

# Create a DataFrame
embedding_df = pd.DataFrame(
    learned_embeddings, 
    columns=['Emb_Dim_1', 'Emb_Dim_2', 'Emb_Dim_3'],
    index=ordered_species_names
)

# Verify the shape
print(f"✅ Extracted embeddings shape: {learned_embeddings.shape}")
print(f"   (Expected: {trained_model.num_species} species × 3 embedding dimensions)")

#%% 3D plot of embedding latent
import matplotlib.pyplot as plt
import numpy as np

print("📊 Generating presentation-ready 3D embedding plot...")

# 1. Initialize figure with strict white background for presentations
fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

# 2. Extract dimensions
x = embedding_df['Emb_Dim_1']
y = embedding_df['Emb_Dim_2']
z = embedding_df['Emb_Dim_3']

# Optional: We can color the dots based on their depth (Z-axis) for better 3D perception
colors = plt.cm.viridis((z - z.min()) / (z.max() - z.min()))

# 3. Create the 3D scatter plot
scatter = ax.scatter(
    x, y, z, 
    c=colors, 
    s=60,               # Marker size
    alpha=0.8,          # Slight transparency to see overlaps
    edgecolor='black',  # Clean borders
    linewidth=0.5
)

# 4. Clean up axes and labels
ax.set_xlabel('Latent Trait 1', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Latent Trait 2', fontsize=12, fontweight='bold', labelpad=10)
ax.set_zlabel('Latent Trait 3', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Learned 3D Species Ecological Niche Space', fontsize=16, fontweight='bold', pad=20)

# Remove the default gray background panes for a cleaner slide look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

# 5. Annotate a few sample species (e.g., the first 5) to give context
for i in range(20):
    ax.text(x.iloc[i], y.iloc[i], z.iloc[i], 
            f' $\it{{{embedding_df.index[i]}}}$', 
            fontsize=9, zdir='x')

# Save the figure
plot_path = "data/tutorial/species_embedding_3d.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"✅ 3D Plot saved to: {plot_path}")
plt.show()

