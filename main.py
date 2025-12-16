import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import argparse

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_DIR = r"D:\Sadman\Thesis_anomaly_detection\dataset"
TXT_FILE = r"D:\Sadman\Thesis_anomaly_detection\dataset\SHANGHAI_TRAIN\SHANGHAI_train.txt"
SAVE_DIR = r"D:\Sadman\Thesis_anomaly_detection\models"

# Create model save directory if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on Device: {DEVICE}")

# ==========================================
# 2. DATASET LOADER (Supervised: 0 vs 1)
# ==========================================
class ShanghaiDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.transform = transform
        self.data = [] 

        print(f"Parsing {txt_file}...")
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4: continue
            
            rel_path = parts[0].replace('/', os.sep)
            label = int(parts[3]) # 0 = Normal, 1 = Anomaly
            folder_path = os.path.join(root_dir, rel_path)
            
            # Grab all images in the folder
            images = glob.glob(os.path.join(folder_path, "*.jpg"))
            for img_path in images:
                self.data.append((img_path, label))

        print(f"Dataset Loaded: {len(self.data)} frames.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            return torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1])), torch.tensor(0.0)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. DYNAMIC MODEL FACTORY
# ==========================================
def get_model(model_name):
    print(f"Initializing Model: {model_name.upper()}...")
    
    if model_name == "custom_cnn":
        # Model 1: Simple Baseline CNN
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
        
    elif model_name == "resnet50":
        # Model 2: ResNet50
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 1), nn.Sigmoid()
        )
        
    elif model_name == "vgg16":
        # Model 3: VGG16 (Large, classic)
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 1), nn.Sigmoid()
        )
        
    elif model_name == "mobilenet_v2":
        # Model 4: MobileNetV2 (Lightweight/Edge)
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Sequential(
            nn.Linear(model.last_channel, 512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, 1), nn.Sigmoid()
        )
        
    elif model_name == "densenet121":
        # Model 5: DenseNet121 (High accuracy)
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 1), nn.Sigmoid()
        )
    else:
        raise ValueError(f"Model {model_name} not implemented")
        
    return model.to(DEVICE)

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_engine(model_name):
    # Load Data
    dataset = ShanghaiDataset(txt_file=TXT_FILE, root_dir=BASE_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load Model
    model = get_model(model_name)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"STARTING TRAINING FOR {model_name.upper()}")
    print("-" * 40)

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        model.train()
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        acc = 100 * correct / total
        print(f"✅ Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    # Save Model
    save_path = os.path.join(SAVE_DIR, f"{model_name}_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Model saved to: {save_path}")
    print("=" * 40)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Create an argument parser so you can select models from command line
    parser = argparse.ArgumentParser(description="Train Anomaly Detection Models")
    parser.add_argument("--model", type=str, default="resnet50", 
                        help="Choose from: custom_cnn, resnet50, vgg16, mobilenet_v2, densenet121")
    
    args = parser.parse_args()
    
    # Run the training
    train_engine(args.model)