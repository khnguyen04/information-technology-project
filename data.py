import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer

# Danh sách các thư mục chứa ảnh
IMAGE_ROOTS = ["images_1", "images_2", "images_3"]
CSV_PATH = "merged_cleaned_images.csv"  
IMG_SIZE = 256
BATCH_SIZE = 8

class XrayPairedDataset(Dataset):
    """
    Dataset image-to-image cho X-ray, với cặp (No Finding -> Disease) và nhãn bệnh.
    File CSV đã lọc sẵn các ảnh tồn tại.
    Tìm ảnh trong nhiều thư mục.
    """
    def __init__(self, csv_path=CSV_PATH, root_dirs=IMAGE_ROOTS, img_size=256):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.img_size = img_size

        # 1. Đọc CSV
        self.df = pd.read_csv(csv_path)
        print(f"[INFO] {len(self.df)} cặp ảnh có trong CSV")

        # 2. Xử lý nhãn: one-hot encode
        # Lấy tất cả nhãn bệnh (không tính No Finding)
        all_labels = set()
        for labels in self.df['Finding Labels_disease']:
            for l in labels.split('|'):
                all_labels.add(l)
        self.label_names = sorted(list(all_labels))
        self.num_classes = len(self.label_names)

        self.mlb = MultiLabelBinarizer(classes=self.label_names)
        self.mlb.fit([self.label_names])

        # 3. Transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def _find_image_path(self, image_filename):
        """
        Tìm đường dẫn ảnh trong tất cả các thư mục
        """
        for root_dir in self.root_dirs:
            potential_path = os.path.join(root_dir, image_filename)
            if os.path.exists(potential_path):
                return potential_path
        
        # Nếu không tìm thấy trong bất kỳ thư mục nào
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_filename} trong các thư mục {self.root_dirs}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tìm đường dẫn ảnh trong các thư mục
        img_cond_path = self._find_image_path(row['Image Index_no'])
        img_target_path = self._find_image_path(row['Image Index_disease'])

        # Load ảnh
        img_cond = Image.open(img_cond_path).convert("L")
        img_target = Image.open(img_target_path).convert("L")

        img_cond = self.transform(img_cond)
        img_target = self.transform(img_target)

        # Nhãn bệnh one-hot
        labels = row['Finding Labels_disease'].split('|')
        condition_tensor = torch.tensor(self.mlb.transform([labels])[0], dtype=torch.float32)

        return img_cond, img_target, condition_tensor

def load_data(csv_path=CSV_PATH, root_dirs=IMAGE_ROOTS, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True):
    dataset = XrayPairedDataset(csv_path, root_dirs, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

if __name__ == "__main__":
    # Test dataset
    dataset = XrayPairedDataset(CSV_PATH, IMAGE_ROOTS, IMG_SIZE)
    print(f"Tên các nhãn bệnh: {dataset.label_names}")
    print(f"Số lớp: {dataset.num_classes}")
    
    # # Test tìm ảnh
    # sample_idx = 0
    # try:
    #     img_cond, img_target, condition = dataset[sample_idx]
    #     print(f"✅ Load ảnh thành công!")
    #     print(f"   Ảnh condition shape: {img_cond.shape}")
    #     print(f"   Ảnh target shape: {img_target.shape}")
    #     print(f"   Condition tensor: {condition}")
    # except FileNotFoundError as e:
    #     print(f"❌ Lỗi: {e}")
    
    # # Test dataloader
    # print("\n🧪 Test DataLoader:")
    # loader = load_data()
    # for batch_idx, (x_cond, x_target, cond) in enumerate(loader):
    #     print(f"Batch {batch_idx + 1}:")
    #     print(f"  x_cond shape: {x_cond.shape}")
    #     print(f"  x_target shape: {x_target.shape}")
    #     print(f"  cond shape: {cond.shape}")
    #     if batch_idx == 0:  
    #         break