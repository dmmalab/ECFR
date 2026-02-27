import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from datasets.augmentations import get_tta_transforms

class StreamingMedicalDataset(Dataset):
    """
    Standardized dataset loader for Test-Time Adaptation streaming.
    Outputs the original image alongside its weakly and strongly augmented views.
    """
    def __init__(self, csv_path, input_size=224):
        super().__init__()
        self.data_info = pd.read_csv(csv_path)
        
        # Load the asymmetric augmentation pipelines
        self.base_tf, self.weak_tf, self.strong_tf = get_tta_transforms(input_size)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        img_path = row['filepath']
        label = row['label']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images in real-world streaming
            img = Image.new('RGB', (224, 224), color='black') # DermLIP

        # Generate the multi-view inputs required for consistency regularization
        x_clean = self.base_tf(img)
        x_weak = self.weak_tf(img)
        x_strong = self.strong_tf(img)

        return x_clean, x_weak, x_strong, label, img_path