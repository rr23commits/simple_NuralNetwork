import os
import shutil

# Change this to your dataset folder
base_dir = "/Users/rigelrai/Documents/Bennett/Year 2/Python/fracture/HBFMID/Bone Fractures Detection"

for split in ["train", "test"]:
    img_dir = os.path.join(base_dir, split, "images")
    label_dir = os.path.join(base_dir, split, "labels")
    
    # Create target folders
    fracture_dir = os.path.join(base_dir, split, "fracture")
    normal_dir = os.path.join(base_dir, split, "normal")
    os.makedirs(fracture_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    
    for img_file in os.listdir(img_dir):
        if not img_file.endswith((".jpg", ".png", ".jpeg")):
            continue
        
        label_file = os.path.join(label_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        # If label file exists and is not empty → fracture
        if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
            shutil.move(os.path.join(img_dir, img_file), os.path.join(fracture_dir, img_file))
        else:
            shutil.move(os.path.join(img_dir, img_file), os.path.join(normal_dir, img_file))
    
    print(f"✅ {split} split done: fracture vs normal")
