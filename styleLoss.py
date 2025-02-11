import os
import re
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
from tqdm import tqdm
import time

###############################################################################
# Regex patterns for parse_base_grouping
###############################################################################
pattern_cv_jpg    = re.compile(r'^(.*?)_cv\.jpg', re.IGNORECASE)
pattern_cv_jpggt  = re.compile(r'^(.*?)_cv\.jpggt-', re.IGNORECASE)
pattern_triple_us = re.compile(r'^(.*?)___', re.IGNORECASE)

def parse_base_grouping(fname):
    """
    Extract the 'base' portion (e.g. '10_64_r1_d1') from a grouping filename.
    Returns the base string if matched, else None.
    """
    m1 = pattern_cv_jpg.search(fname)
    if m1:
        return m1.group(1)
    m2 = pattern_cv_jpggt.search(fname)
    if m2:
        return m2.group(1)
    m3 = pattern_triple_us.search(fname)
    if m3:
        return m3.group(1)
    return None

###############################################################################
# Global GPU and Model
###############################################################################
# -- UPDATED to use cuda:0 (or CPU if no GPU):
device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
_base_cnn = None

class FeatureExtractor(nn.Module):
    """
    A module that:
      1) Normalizes the input
      2) Extracts feature maps at specific style layers
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


def get_vgg19_features():
    """
    Lazily loads and caches the VGG19 feature extractor.
    """
    global _base_cnn
    if _base_cnn is None:
        print("[INFO] Loading VGG19 'features' model to GPU for the first time...")
        _base_cnn = FeatureExtractor().to(device).eval()
    return _base_cnn


def gram_matrix(feat_map: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix of feature maps.
    """
    B, C, H, W = feat_map.shape
    fm_flat = feat_map.view(C, H*W)  # Flatten to [C, H*W]
    G = fm_flat @ fm_flat.t()        # [C, C]
    return G / (C * H * W)


def compute_gram_dict(img_tensor: torch.Tensor, extractor: nn.Module):
    """
    Runs the image tensor through the feature extractor and returns a dict
    {layer_name: gram_matrix_tensor}.
    """
    with torch.no_grad():
        feats = extractor(img_tensor.to(device))

    grams = {}
    for layer_name, fm in feats.items():
        G = gram_matrix(fm)
        grams[layer_name] = G.to(device)  # Move to CPU to avoid GPU overload
    return grams


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Loads an image, applies transforms & normalization, returns a 4D tensor: [1, C, H, W].
    """
    imsize = 512 if torch.cuda.is_available() else 128
    transform_pipeline = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        # Normalization here is optional if we're normalizing in the FeatureExtractor,
        # but it won't hurt. You may remove or keep depending on your preference.
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform_pipeline(img).unsqueeze(0).to(device, torch.float)


def load_precomputed_grams(pkl_path: str, device: torch.device):
    """
    Loads a dict from .pkl containing real_image_grams:
        { real_img_path: { layer_name: gram_matrix_tensor }, ... }
    Ensures all gram matrices are moved to 'device'.
    """
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"[ERROR] Precomputed Gram file not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        gram_dict = pickle.load(f)
    
    # Move everything to the specified device
    for real_path, layer_dict in gram_dict.items():
        for layer_name, G in layer_dict.items():
            gram_dict[real_path][layer_name] = G.to(device)

    print(f"[INFO] Loaded precomputed grams from: {pkl_path}  (count={len(gram_dict)})")
    return gram_dict


def compute_city_style_loss(gen_img_path, real_grams_dict, extractor):
    """
    Given a generated image and the dictionary of real-image Gram matrices,
    computes MSE vs each real image's grams, then returns the average.
    """
    if not os.path.isfile(gen_img_path):
        print(f"[WARN] Missing generated image: {gen_img_path}")
        return None

    # Compute Gram dict for the generated image
    gen_tensor = load_image_as_tensor(gen_img_path)
    gen_grams = compute_gram_dict(gen_tensor, extractor)

    losses = []
    for real_path, real_grams in real_grams_dict.items():
        layer_losses = []
        for layer_name in gen_grams:
            if layer_name in real_grams:
                if gen_grams[layer_name].numel() == 0 or real_grams[layer_name].numel() == 0:
                    continue
                loss = F.mse_loss(gen_grams[layer_name], real_grams[layer_name]).item()
                layer_losses.append(loss)

        if layer_losses:
            losses.append(sum(layer_losses))

    if not losses:
        return None
    else:
        return sum(losses) / len(losses)  # average

###############################################################################
# Main Workflow: Loop over groupings, then cities, then sub‐indices
###############################################################################
if __name__ == "__main__":
    # 1) Load your grouping JSON
    groupings_file = "./urban_data/tencities/GenAI_density/groupings_all.json"
    with open(groupings_file, 'r') as f:
        groupings_list = json.load(f)
    print(f"[INFO] Loaded {len(groupings_list)} groupings.")

    if not groupings_list:
        raise SystemExit("[ERROR] groupings_list is empty. Nothing to do.")

    # 2) Directory info
    showcase_dir = "./urban_data/tencities/GenAI_density/show-case2"
    output_csv   = "./urban_data/tencities/GenAI_density/output_results.csv"

    # 3) Mapping city -> pkl file
    city_pkl_map = {
        "Singapore": "./urban_data/tencities/GenAI_density/Singapore/Singapore_grams.pkl",
        "HongKong": "./urban_data/tencities/GenAI_density/HongKong/HongKong_grams.pkl",
        "Munich": "./urban_data/tencities/GenAI_density/Munich/Munich_grams.pkl",
        "Stockholm": "./urban_data/tencities/GenAI_density/Stockholm/Stockholm_grams.pkl",
        "Chicago": "./urban_data/tencities/GenAI_density/Chicago/Chicago_grams.pkl",
        "Orlando": "./urban_data/tencities/GenAI_density/Orlando/Orlando_grams.pkl",
        "Kinshasa": "./urban_data/tencities/GenAI_density/Kinshasa/Kinshasa_grams.pkl",
        "SaoPaulo": "./urban_data/tencities/GenAI_density/SaoPaulo/SaoPaulo_grams.pkl",
        "Mexico": "./urban_data/tencities/GenAI_density/Mexico/Mexico_grams.pkl",
        "Kigali": "./urban_data/tencities/GenAI_density/Kigali/Kigali_grams.pkl"
    }

    # 4) City list
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

    # 5) Initialize the VGG19 feature extractor once
    extractor = get_vgg19_features()

    # 6) Load each city's real-image Gram dict
    city_grams_dicts = {}
    for city in cities:
        pkl_path = city_pkl_map.get(city, None)
        if pkl_path and os.path.isfile(pkl_path):
            city_grams_dicts[city] = load_precomputed_grams(pkl_path, device)
        else:
            print(f"[WARN] No precomputed grams for {city} (skipping).")
            city_grams_dicts[city] = {}

    # 7) Main loop
    results_rows = []
    start_time = time.time()

    for fname in tqdm(groupings_list, desc="Processing groupings"):
        base_part = parse_base_grouping(fname) or fname
        row = {"GroupBase": base_part, "OriginalFilename": fname}

        # For each city and sub-index [0,1,2], compute style loss
        for city in cities:
            # If we have no grams for the city, skip
            if not city_grams_dicts[city]:
                row[f"{city}____0"] = None
                row[f"{city}____1"] = None
                row[f"{city}____2"] = None
                continue

            for i in [0,1,2]:
                gen_name = f"{base_part}___{city}____{i}.png"
                gen_path = os.path.join(showcase_dir, gen_name)
                if not os.path.isfile(gen_path):
                    print(f"[WARN] Generated image missing: {gen_path}")

                loss_val = compute_city_style_loss(
                    gen_img_path=gen_path,
                    real_grams_dict=city_grams_dicts[city],
                    extractor=extractor
                )

                col_name = f"{city}____{i}"
                row[col_name] = loss_val

        results_rows.append(row)

    end_time = time.time()
    elapsed_sec = end_time - start_time
    print(f"[INFO] Finished processing. Elapsed time: {elapsed_sec:.2f} seconds.")

    # 8) Convert results to DataFrame and save
    df = pd.DataFrame(results_rows)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Results saved to: {output_csv}")
