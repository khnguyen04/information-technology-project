import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class ChestXrayDatasetLoader:
    def __init__(self, csv_path, image_dir=None):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.df = None

    def load_csv(self):
        df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        keep_cols = ['Image Index', 'Finding Labels', 'View Position']
        df = df[keep_cols]
        self.df = df
        return df

    def preprocess(self):
        if self.df is None:
            raise ValueError("Bạn cần gọi load_csv() trước khi preprocess()")

        df = self.df.copy()
        df['Finding Labels'] = df['Finding Labels'].apply(
            lambda x: 0 if isinstance(x, str) and x.strip() == "No Finding" else 1
        )

        if self.image_dir:
            df['Image Path'] = df['Image Index'].apply(lambda x: os.path.join(self.image_dir, x))
            df = df[df['Image Path'].apply(os.path.exists)].reset_index(drop=True)

        self.df = df
        return df

    def summary(self):
        if self.df is None:
            raise ValueError("Bạn cần gọi load_csv() trước khi xem summary")
        print("Total images:", len(self.df))
        print("Count labels:", self.df['Finding Labels'].value_counts().to_dict())
        print("View Position:", self.df['View Position'].value_counts(dropna=False).to_dict())


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['Image Path']
        label = self.data.iloc[idx]['Finding Labels']

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

    def get_subset(self, disease=True):
        subset_df = self.data[self.data['Finding Labels'] == (1 if disease else 0)]
        return ChestXrayDataset(subset_df, transform=self.transform)


def get_dataloader(csv_path, image_dir, batch_size=8, size=256, num_workers=2):
    loader = ChestXrayDatasetLoader(csv_path, image_dir)
    loader.load_csv()
    loader.preprocess()
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = ChestXrayDataset(loader.df, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader, loader.df


if __name__ == "__main__":
    csv_path = "Data_Entry_2017.csv"
    image_dir = "images_1"

    loader = ChestXrayDatasetLoader(csv_path, image_dir)
    loader.load_csv()
    loader.preprocess()
    loader.summary()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ChestXrayDataset(loader.df, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    images, labels = next(iter(dataloader))
    print("Batch shape:", images.shape)
    print("Labels:", labels)
