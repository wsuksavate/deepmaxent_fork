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
input_folder = 'data/custom/rasters'

# List available rasters
raster_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.tif')])
print(f"\n📂 Found {len(raster_files)} raster files")
print(f"{raster_files}")

#%% Crop all rasters to the study extent
print("✂️ Cropping rasters to study area...")

for filename in tqdm(raster_files, desc="Processing"):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    with rasterio.open(input_path) as src:
        # Get the window for our extent
        window = from_bounds(
            min_longitude, min_latitude, 
            max_longitude, max_latitude, 
            transform=src.transform
        )
        
        # Read cropped data
        data = src.read(window=window)
        
        # Update metadata
        meta = src.meta.copy()
        meta.update({
            'height': int(window.height),
            'width': int(window.width),
            'transform': rasterio.windows.transform(window, src.transform)
        })
        
        # Write cropped raster
        if window.height > 0 and window.width > 0:
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(data)

print(f"\n✅ Successfully cropped {len(raster_files)} rasters!")

# Check the cropped raster dimensions
with rasterio.open(os.path.join(output_folder, raster_files[0])) as src:
    print(f"\n📐 Cropped raster dimensions: {src.width} × {src.height} pixels")
    print(f"   Resolution: {abs(src.transform[0]):.4f}° (~{abs(src.transform[0]) * 111:.1f} km)")
    
#%% Visuaize environmental variables
# Select key variables to visualize
variables_to_plot = [
    ('CHELSA_bio1_1981-2010_V.2.1.tif', 'BIO1: Annual Mean Temperature', '°C × 10', 'RdYlBu_r'),
    ('CHELSA_bio12_1981-2010_V.2.1.tif', 'BIO12: Annual Precipitation', 'mm', 'YlGnBu'),
    ('CHELSA_bio4_1981-2010_V.2.1.tif', 'BIO4: Temperature Seasonality', 'std × 100', 'Spectral_r'),
    ('CHELSA_bio15_1981-2010_V.2.1.tif', 'BIO15: Precipitation Seasonality', 'CV', 'BrBG')
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

for ax, (filename, title, unit, cmap) in zip(axes, variables_to_plot):
    filepath = os.path.join(output_folder, filename)
    
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        extent_raster = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        nodata = src.nodata
        
        # Mask nodata values and any extreme values that could cause overflow
        if nodata is not None:
            data = np.ma.masked_where(
                (data == nodata) | (np.abs(data) > 1e10) | ~np.isfinite(data), 
                data
            )
        else:
            data = np.ma.masked_where((np.abs(data) > 1e10) | ~np.isfinite(data), data)
    
    # Add features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.3)
    
    # Plot raster
    im = ax.imshow(data, extent=extent_raster, origin='upper', cmap=cmap, 
                   transform=ccrs.PlateCarree())
    
    # Overlay occurrences
    ax.scatter(df_clean['lon'], df_clean['lat'],
               c='black', s=3, alpha=0.3, transform=ccrs.PlateCarree())
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(unit, fontsize=10)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('Key Bioclimatic Variables with Species Occurrences (black dots)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

#%%
# Prepare Training Data
# Now we'll prepare the data for DeepMaxent by:
# 1 Extracting environmental values at each occurrence location
# 2 Aggregating occurrences by raster cell (multiple observations at the same location)
# 3 Building the occurrence count matrix (y) where each entry represents the number of observations per species per location

# Get the raster resolution info
rasters_dir = 'data/custom/cropped_rasters/'
raster_files = sorted([f for f in os.listdir(rasters_dir) if f.endswith('.tif')])

# Get raster properties
with rasterio.open(os.path.join(rasters_dir, raster_files[0])) as src:
    raster_resolution = abs(src.transform[0])
    raster_transform = src.transform
    raster_crs = src.crs

print(f"📐 Raster resolution: {raster_resolution:.4f}° (~{raster_resolution * 111:.1f} km)")
print(f"📁 Number of environmental variables: {len(raster_files)}")
print("\n📋 Variables:")
for i, f in enumerate(raster_files, 1):
    print(f"   {i:2d}. {f}")

#%% Aggregate Occurrences by Raster Cell
# Multiple species observations can occur at the same location. We need to:
# 1 Group observations by unique coordinates
# 2 Count occurrences per species at each location
# 3 This creates our occurrence count matrix (y)

# Aggregate observations by unique location
# This groups all species observations at each coordinate pair
df_unique = df_clean.groupby(['lon', 'lat']).agg({
    'species': list  # Collect all species observed at this location
}).reset_index()

# Get unique species list and create index mapping
species_list = df_clean['species'].unique()
num_species = len(species_list)
species_to_idx = {species: idx for idx, species in enumerate(species_list)}

print(f"📍 Unique locations: {len(df_unique):,}")
print(f"🌱 Unique species: {num_species:,}")
print(f"📊 From {len(df_clean):,} individual observations")
print(f"\n📐 Aggregation ratio: {len(df_clean) / len(df_unique):.1f} observations per location (avg)")

#%% Build Training Tensors
# X tensor (environmental features): shape (n_locations, n_variables)
# y tensor (occurrence counts): shape (n_locations, n_species)

# Initialize tensors
num_locations = len(df_unique)
num_variables = len(raster_files)

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

print(f"\n✅ Tensor creation complete!")
print(f"   X (environmental features): {X_tensor.shape}")
print(f"   y (occurrence counts):      {y_tensor.shape}")

#%%

# Visualize species occurrence distribution
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

# Map species richness
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
# We'll divide the study area into blocks based on longitude and assign blocks to train/validation

# Get coordinates from df_unique
longitudes = df_unique['lon'].values
latitudes = df_unique['lat'].values

# Define the split based on longitude (roughly 80% train, 20% validation)
# We'll use the western part for training and eastern part for validation
lon_threshold = np.percentile(longitudes, 80)

# Create train/validation masks
train_mask = longitudes < lon_threshold
val_mask = ~train_mask

print("🔀 SPATIAL BLOCK SPLIT")
print("=" * 50)
print(f"📍 Longitude threshold: {lon_threshold:.4f}°")
print(f"\n📊 Split statistics:")
print(f"   Training set:   {train_mask.sum():,} locations ({100*train_mask.mean():.1f}%)")
print(f"   Validation set: {val_mask.sum():,} locations ({100*val_mask.mean():.1f}%)")

# Split the tensors
X_train = X_tensor[train_mask]
y_train = y_tensor[train_mask]
X_val = X_tensor[val_mask]
y_val = y_tensor[val_mask]

# Get coordinates for plotting
train_coords = df_unique[train_mask][['lon', 'lat']].values
val_coords = df_unique[val_mask][['lon', 'lat']].values

print(f"\n📐 Tensor shapes:")
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
ax.plot([lon_threshold, lon_threshold], [lat_min, lat_max], 
        color='#2c3e50', linestyle='--', linewidth=2, 
        transform=ccrs.PlateCarree(), label=f'Split boundary ({lon_threshold:.2f}°)')

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax.set_title('patial Block Split: Training vs Validation Sets\n(Based on Longitude)', 
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

print(f"\n✅ Data preprocessing complete!")
print(f"   X_train_tensor: {X_train_tensor.shape}")
print(f"   y_train_tensor: {y_train_tensor.shape}")
print(f"   X_val_tensor:   {X_val_tensor.shape}")
print(f"   y_val_tensor:   {y_val_tensor.shape}")

#%% Define Training Configuration and Model# Additional imports for training
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import copy

# Training configuration using a simple namespace
class Args:
    def __init__(self):
        self.learning_rate = 0.001
        self.epoch = 5000
        self.hidden_nbr = 2  # Number of hidden layers
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

print(f"\n🧠 Model Architecture:")
print(f"   Input size:  {input_size} (environmental variables)")
print(f"   Hidden size: {hidden_size}")
print(f"   Output size: {output_size} (species)")
print(f"   Hidden layers: {args.hidden_nbr}")

print(f"\n⚙️ Training Configuration:")
print(f"   Learning rate: {args.learning_rate}")
print(f"   Epochs: {args.epoch}")
print(f"   Weight decay: {args.weight_decay}")
print(f"   Batch size: 250")

#%% Training Loop with Validation AUC
# DeepMaxent loss for optimization
# Validation AUC computed at each epoch to monitor performance
# Best model selection based on validation loss

def compute_auc(model, X, y, device):
    """
    Compute mean AUC across all species with sufficient data.
    
    Args:
        model: trained DeepMaxent model
        X: input features tensor
        y: target occurrence tensor
        device: computation device
    
    Returns:
        mean_auc: average AUC across species
        valid_aucs: list of AUC values for each valid species
    """
    model.eval()
    with torch.no_grad():
        X_dev = X.to(device)
        predictions = model(X_dev).cpu()
        # Apply softmax to get probabilities
        probs = torch.softmax(predictions, dim=0).numpy()
    
    y_np = y.numpy()
    
    # Convert to binary (presence/absence)
    y_binary = (y_np > 0).astype(int)
    
    valid_aucs = []
    for sp_idx in range(y_binary.shape[1]):
        # Only compute AUC if species has both presences and absences
        if y_binary[:, sp_idx].sum() > 0 and y_binary[:, sp_idx].sum() < len(y_binary):
            try:
                auc = roc_auc_score(y_binary[:, sp_idx], probs[:, sp_idx])
                valid_aucs.append(auc)
            except:
                pass
    
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    return mean_auc, valid_aucs


def train_deepmodel(X_train, y_train, X_val, y_val, args, hidden_size=50, device="cuda", sp_embedding = True):
    """
    Train DeepMaxent model with validation monitoring.
    
    Returns:
        dict with model, predictions, loss history, and AUC history
    """
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=250, shuffle=True)
    
    # Initialize model and loss
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    ### If using species embedding model
    if sp_embedding:
        model = deepmaxent_embedding_model(input_size, hidden_size, output_size, args.hidden_nbr)
        print("\nUse Species Embedding Model")
    else:
        model = deepmaxent_model(input_size, hidden_size, output_size, args.hidden_nbr)  
        print("\nUse Original Model")
    model = model.to(device)
    
    criterion = deepmaxent_loss().to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("🚀 Starting training...")
    print("=" * 60)
    
    for epoch in tqdm(range(args.epoch), desc="Training"):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            X_val_dev = X_val.to(device)
            y_val_dev = y_val.to(device)
            val_outputs = model(X_val_dev)
            val_loss = criterion(val_outputs, y_val_dev).item()
        val_losses.append(val_loss)
        
        # Compute AUC every 10 epochs (to save time)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_auc, _ = compute_auc(model, X_train, y_train, device)
            val_auc, _ = compute_auc(model, X_val, y_val, device)
            train_aucs.append((epoch, train_auc))
            val_aucs.append((epoch, val_auc))
            
            print(f"   Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_train.to(device)).cpu()
    
    print("=" * 60)
    print(f" Training complete! Best validation loss: {best_val_loss:.4f}")
    
    return {
        "model": model,
        "predictions": final_predictions,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_aucs": train_aucs,
        "val_aucs": val_aucs,
        "best_val_loss": best_val_loss
    }

#%%
# Train the model!
results = train_deepmodel(
    X_train_tensor, 
    y_train_tensor, 
    X_val_tensor, 
    y_val_tensor, 
    args, 
    hidden_size=hidden_size, 
    device=device
)