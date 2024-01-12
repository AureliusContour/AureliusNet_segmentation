from torch.utils.data import Dataset
import numpy as np
import os

class LungCTDataset(Dataset):
    # adjacent slices either 0,1,2
    def __init__(self, dataframe, num_adjacent_slices=1, segmented=False, set_type="train", fold=0):
        self.dataframe = dataframe[(dataframe["set"] == set_type) & (dataframe["fold"] == fold)]
        self.num_adjacent_slices = num_adjacent_slices
        self.segmented = segmented

    def __len__(self):
        return len(self.dataframe)
    
    def load_slice(self, idx, col_name):
        slice_path = self.dataframe.iloc[idx][col_name]
        if (slice_path == None):
            return np.zeros((1, 512, 512))
        
        full_slice_path = os.path.join("segmented_ct" if self.segmented else "ct", slice_path)
        slice_array = np.load(full_slice_path)
        slice_array = np.expand_dims(slice_array, axis=0)
        
        return slice_array
    
    def __getitem__(self, idx):
        impath = self.dataframe.iloc[idx]["path"]
        mask_path = os.path.join("mask", impath)
        ct_path = os.path.join("segmented_ct" if self.segmented else "ct", impath)
        ct_array = np.load(ct_path)
        ct_array = np.expand_dims(ct_array, axis=0)
        mask_array = np.load(mask_path)

        if (self.num_adjacent_slices == 0):
            return ct_array, mask_array
        
        else:
            # add top slices
            for i in range(self.num_adjacent_slices, 0, -1):
                col_name = f"adjacent_top_{i}"
                adjacent_slice = self.load_slice(idx, col_name)
                ct_array = np.concatenate([adjacent_slice, ct_array], axis=0)

            # add bottom slices
            for i in range(0, self.num_adjacent_slices, 1):
                col_name = f"adjacent_bot_{i+1}"
                adjacent_slice = self.load_slice(idx, col_name)
                ct_array = np.concatenate([ct_array, adjacent_slice], axis=0)

        return ct_array, mask_array