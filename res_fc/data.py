import config
import pandas as pd
from PIL import Image
import torch
import random
from torch.utils.data import Subset
from collections import defaultdict
import math

picFolder = config.picFolder
photosFolder = config.photosFolder
commentFile = config.commentFile
boxDataFile = config.boxDataFile

# ====================== dataset ==========================
class dataset():
    def __init__(self,
                 picFolder=config.picFolder,
                 photosFolder=config.photosFolder,
                 boxDataFile=config.boxDataFile,
                 commentFile=config.commentFile,
                 transform=config.transform):
        
        self.dataFrame = pd.read_excel(boxDataFile)
        self.transform = transform
        self.commentFile = commentFile
        
        self.imgFile = {}
        

        for row in self.dataFrame.itertuples():
            i = row.id  
            self.imgFile[i] = [
                f"{picFolder}/{i}.png",
                f"{photosFolder}/{i}_0.png",
                f"{photosFolder}/{i}_1.png",
                f"{photosFolder}/{i}_2.png"
            ]
            
    def __len__(self):
        return len(self.dataFrame)
    
    def __getitem__(self, idx):
        row = self.dataFrame.iloc[idx]
        
        label_val = row["票房_万"]
        label = torch.tensor(label_val, dtype=torch.float32)
        
        imgid = row["id"]
        imgList = self.imgFile[imgid]
        imgStack = []

        for img_path in imgList:
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                imgStack.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                raise e

        imgStack = torch.stack(imgStack)

        return imgStack, label
    
    
# ===================== 数据集划分 =====================

def split_within_year_month(dataset, train_ratio=0.7, random_seed=42):
    """
    在同年月的数据内部按比例划分
    每个年月的数据独立划分，训练集向上取整
    
    参数：
    - dataset: 已实例化的 dataset 对象
    - train_ratio: 训练集比例
    - random_seed: 随机种子
    """
    random.seed(random_seed)
    
    df = dataset.dataFrame.copy()

    df['_year_month'] = (
        df['year'].astype(str) + '-' + 
        df['month'].astype(str).str.zfill(2)
    )
    
    ym_to_indices = defaultdict(list)
    
    # 记录每个年月对应的“行号”
    for idx in range(len(df)):
        ym = df.iloc[idx]['_year_month']
        ym_to_indices[ym].append(idx)
    
    train_indices = []
    test_indices = []
    
    for ym, indices in ym_to_indices.items():
        n_samples = len(indices)
        n_train = math.ceil(n_samples * train_ratio) 

        random.shuffle(indices)

        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
 
    return train_dataset, test_dataset
