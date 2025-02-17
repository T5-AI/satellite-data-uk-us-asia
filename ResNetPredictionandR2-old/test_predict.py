import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import json
import os
import re
import time
from tqdm import tqdm

# Force the use of GPU #0
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:1")
print(f"Using device: {device}")

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define max_values for de-normalization (same as training)
max_values = [135.9, 5379.6, 38447.0]

# Corrected Test Dataset Class (returns image paths)
class TestImageRegressionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform
        with open(data_dir, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target_filename = item['target']  # Get the image path
        image = Image.open(target_filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target_filename  # Return image and its path

# Load test dataset
test_data_path = "./urban_data/tencities/test1.json"
test_dataset = TestImageRegressionDataset(test_data_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define model architecture (same as training)
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 3),
    nn.Sigmoid()
)

# Load trained weights and move to device
model.load_state_dict(torch.load("resnet50_regression.pth", map_location=device))
model = model.to(device)
model.eval()

# Perform predictions
predictions = []
start_time = time.time()

with torch.no_grad():
    for images, image_paths in tqdm(test_dataloader, desc="Processing Images"):
        images = images.to(device)
        outputs = model(images).squeeze().cpu().numpy()
        
        # De-normalize predictions
        predicted_values = [outputs[i] * max_values[i] for i in range(3)]
        
        predictions.append({
            "image": image_paths[0],  # Correct image path from dataset
            "predicted_values": predicted_values
        })

# Save predictions
output_json_path = "predictions.json"
with open(output_json_path, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"Predictions saved to {output_json_path}")
print(f"Total processing time: {time.time() - start_time:.2f} seconds")