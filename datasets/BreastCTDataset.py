from torch.utils.data import Dataset
import cv2
import numpy as np

class BreastCTDataset(Dataset):
    """
    This custom dataset has the following parameters:
    - dataframe: for the dataframe containing all the info
    - transform: expected to be a transforms object from the albumentations library
    - set: train/valid/test
    - fold: fold number
    """
    def __init__(self, dataframe, transform=None, set_type="train", fold=0):
        self.dataframe = dataframe[dataframe["set"] == set_type]
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        ct_path = self.dataframe.iloc[idx]["resized_ct_file"]
        mask_path = self.dataframe.iloc[idx]["resized_mask_file"]
        label = self.dataframe.iloc[idx]["label"]

        ct_image = cv2.imread(ct_path)        
        ct_image = (ct_image/255.0).astype(np.float32)
        
        mask_image = cv2.imread(mask_path)		
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        mask_image = np.expand_dims(mask_image, axis=0)
        mask_image = (mask_image/255.0).astype(np.float32)

        if self.transform:
            augmented_images = self.transform(image=ct_image, mask=mask_image)
            ct_image = augmented_images["image"]
            mask_image = augmented_images["mask"]

        return ct_image, mask_image, label