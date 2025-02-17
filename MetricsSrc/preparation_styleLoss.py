import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
from PIL import Image
import pickle
import re

# Base directory containing the ten cities
base_dir = "/Users/wangzhuoyulucas/SMART /data_server/urban_data/tencities/GenAI_density"

# List of cities
cities = [
    'Singapore',
    'HongKong',
    'Munich',
    'Stockholm',
    'Chicago',
    'Orlando',
    'Kinshasa',
    'SaoPaulo',
    'Mexico',
    'Kigali'
]

# Define which file extensions to consider as images
image_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# How many images to sample per city
sample_size = 1000

for city in cities:
    # Specifically look in {city}/satellite_images_z17
    city_satellite_dir = os.path.join(base_dir, city, "satellite_images_z17")
    if not os.path.isdir(city_satellite_dir):
        print(f"Warning: directory does not exist for {city}: {city_satellite_dir}")
        continue

    # Collect all image filepaths recursively from satellite_images_z17
    city_img_files = []
    for root, dirs, files in os.walk(city_satellite_dir):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(root, filename)
                city_img_files.append(full_path)

    # Randomly shuffle and take up to 1000
    random.shuffle(city_img_files)
    selected_files = city_img_files[:sample_size]

    # Write these image paths to a JSON file within the city folder
    out_json = os.path.join(base_dir, city, f"{city}_img_paths_sf.json")
    with open(out_json, 'w') as outfile:
        json.dump(selected_files, outfile, indent=2)

    print(f"Saved {len(selected_files)} image paths for {city} -> {out_json}")


showcase_dir = "/Users/wangzhuoyulucas/SMART /generatedImg/show-case2"
groupings_output = "/Users/wangzhuoyulucas/SMART /data_server/urban_data/tencities/GenAI_density/groupings.json"

# 1) Define multiple regex patterns
patterns = [
    # a) original pattern: something containing "_cv.jpg" and ending "_gt.png"
    re.compile(r'_cv\.jpg.*_gt\.png$', re.IGNORECASE),
    
    # b) pattern for e.g. "34_45_r0_d0_cv.jpggt-Chicago"
    #    or "61_52_r0_d1_cv.jpggt-Mexico.png"
    #    We look for "_cv.jpggt-" + (some city text) + optional extension
    #    NOTE: Adjust character class [^.]+ if city names can have spaces, etc.
    re.compile(r'_cv\.jpggt-[^.]+(\.png|\.jpg|\.jpeg|\.tiff|\.bmp)?$', re.IGNORECASE),
]

groupings_list = []

for fname in os.listdir(showcase_dir):
    # Skip directories
    if os.path.isdir(os.path.join(showcase_dir, fname)):
        continue
    
    # Check each pattern; if any matches, we collect it
    for pat in patterns:
        if pat.search(fname):
            groupings_list.append(fname)
            break  # stop checking other patterns once matched

# Save to JSON
with open(groupings_output, 'w') as f:
    json.dump(groupings_list, f, indent=2)

print(f"Found {len(groupings_list)} matching filenames.")
print(f"Groupings extracted and saved to: {groupings_output}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STYLE_LAYERS = ['conv_1','conv_2','conv_3','conv_4','conv_5']

class FeatureExtractor(nn.Module):
    """
    A module that:
      1) Normalizes the input
      2) Extracts feature maps at specific style layers
    We'll compute Gram matrices from these.
    """
    def __init__(self, style_layers=None):
        super().__init__()
        if style_layers is None:
            style_layers = STYLE_LAYERS
        
        # Load the VGG19 'features' sub-model once, on GPU if available
        self.base_cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        self.style_layers = set(style_layers)

        # Normalization constants (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(-1,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(-1,1,1)

        self.chosen_blocks = nn.ModuleDict()
        
        i = 0
        block = []
        for layer in self.base_cnn.children():
            block.append(layer)
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
                if name in self.style_layers:
                    # Store the sub-block up to this conv
                    self.chosen_blocks[name] = nn.Sequential(*block)
                    block = []

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        """
        Returns { 'conv_1': feat, 'conv_2': feat, ... } for the specified style layers.
        """
        x = self.normalize(x)
        feats = {}
        current_input = x
        for name, sub_model in self.chosen_blocks.items():
            out = sub_model(current_input)
            feats[name] = out
            current_input = out
        return feats


def gram_matrix(feat_map: torch.Tensor) -> torch.Tensor:
    """
    For a single image, feat_map shape: [1, C, H, W].
    Returns Gram matrix [C, C].
    """
    B, C, H, W = feat_map.shape
    # Flatten to [C, H*W]
    fm_flat = feat_map.view(C, H*W)
    G = fm_flat @ fm_flat.t()  # [C, C]
    return G / (C*H*W)


def compute_style_grams(img_tensor: torch.Tensor, extractor: nn.Module):
    """
    Pass 'img_tensor' through the extractor, returning {layer: GramMatrix}.
    We'll keep them on CPU to avoid big GPU memory usage.
    """
    with torch.no_grad():
        # Extract the feature maps on GPU
        feat_dict = extractor(img_tensor.to(device))

    grams = {}
    for layer_name, fm in feat_dict.items():
        # fm is on GPU, shape [1, C, H, W]
        G = gram_matrix(fm)
        grams[layer_name] = G.cpu()  # move to CPU to store
    return grams


# A small helper to load an image as a tensor (similar to load_image_as_tensor)
def load_img_as_tensor(path, size=512):
    """
    Reads the image from disk, returns a normalized Tensor [1,3,H,W].
    """
    tfms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tfms(img).unsqueeze(0)


def precompute_city_grams(city_name, real_image_paths, extractor, out_file):
    """
    For each real image, compute {layer: GramMatrix} dict,
    store in a dictionary keyed by the image path:
      city_grams[img_path] = { 'conv_1': G, 'conv_2': G, ... }
    Then pickle it to out_file.
    """
    city_grams = {}
    for p in tqdm(real_image_paths, desc=f"Precompute grams for {city_name}", leave=False):
        if not os.path.isfile(p):
            continue
        img_tensor = load_img_as_tensor(p)
        grams_dict = compute_style_grams(img_tensor, extractor)
        city_grams[p] = grams_dict
    
    # Save the dictionary as a .pkl for quick reuse
    with open(out_file, 'wb') as f:
        pickle.dump(city_grams, f)
    print(f"[INFO] Saved {len(city_grams)} real-image grams for {city_name} -> {out_file}")


# Suppose we have a list of city names and each has city_img_paths.json with up to 1000 paths
extractor = FeatureExtractor()  # loads VGG19 features once
data_dir = "/Users/wangzhuoyulucas/SMART /data_server/urban_data/tencities/GenAI_density/"
for city in cities:
    city_json = os.path.join(data_dir, city, f"{city}_img_paths_sf.json")
    if not os.path.isfile(city_json):
        continue
    with open(city_json,'r') as f:
        real_paths = json.load(f)
    if not real_paths:
        continue

    out_file = os.path.join(data_dir, city, f"{city}_grams.pkl")
    precompute_city_grams(city, real_paths, extractor, out_file)

