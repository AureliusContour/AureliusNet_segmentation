import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

transform = A.Compose([
    ToTensorV2(),
], is_check_shapes=False)