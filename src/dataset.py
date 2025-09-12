import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # Only keep fracture and normal folders
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d)) and d in ["fracture","normal"]])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.transform = transform
        self.samples = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(cls_path, img_file), self.class_to_idx[cls]))

        print(f"Loaded {len(self.samples)} images from {root_dir}")
        print(f"Class mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
