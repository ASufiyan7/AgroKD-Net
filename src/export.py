import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. AUTOMATED EXTRACTION
# Change this to the name of the file you uploaded to Colab
ZIP_FILE_PATH = "/content/dataset-1.0.zip" 
EXTRACT_DIR = "/content/extracted_cwfid"

if not os.path.exists(EXTRACT_DIR):
    print(f"📦 Extracting {ZIP_FILE_PATH}...")
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("✅ Extraction Complete!")
else:
    print("✅ Dataset already extracted.")

# Dynamically find the images and annotations folders
image_dir_path, annotation_dir_path = None, None
for root, dirs, files in os.walk(EXTRACT_DIR):
    if 'images' in dirs and image_dir_path is None:
        image_dir_path = os.path.join(root, 'images')
    if 'annotations' in dirs and annotation_dir_path is None:
        annotation_dir_path = os.path.join(root, 'annotations')

if not image_dir_path or not annotation_dir_path:
    raise FileNotFoundError("Could not find 'images' or 'annotations' folders inside the ZIP.")

print(f"📂 Found Images at: {image_dir_path}")
print(f"📂 Found Annotations at: {annotation_dir_path}")

# 2. THE PYTORCH DATASET PIPELINE
class RealCWFIDDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(224, 224)):
        self.image_dir = img_dir
        self.mask_dir = mask_dir
        
        # Get list of image files
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg'))])
        self.img_size = img_size
        
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = img_name.replace('image', 'annotation') 
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB") 
        
        image_tensor = self.img_transform(image)
        
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask_np = np.array(mask)
        
        mask_tensor = torch.zeros(self.img_size, dtype=torch.long)
        
        # Color Mapping: Green (Crop), Red (Weed)
        crop_pixels = (mask_np[:,:,1] > 128) & (mask_np[:,:,0] < 128) 
        weed_pixels = (mask_np[:,:,0] > 128) & (mask_np[:,:,1] < 128) 
        
        mask_tensor[crop_pixels] = 1 # Class 1: Crop
        mask_tensor[weed_pixels] = 2 # Class 2: Weed
        
        return image_tensor, mask_tensor

# Initialize the DataLoader
real_dataset = RealCWFIDDataset(img_dir=image_dir_path, mask_dir=annotation_dir_path)
train_loader = DataLoader(real_dataset, batch_size=4, shuffle=True)

print(f"🚀 DataLoader Ready! Loaded {len(real_dataset)} field images for training.")