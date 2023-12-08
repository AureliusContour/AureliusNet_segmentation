import torch
from torch.utils.data import DataLoader
from config import DEVICE, SEGMENTATION_THRESHOLD, BATCH_SIZE
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


class ModelTester:
    def __init__(self, model: torch.nn.Module, metrics, device=DEVICE) -> None:
        self.__model = model.to(device)
        self.__metrics = metrics
        self.__device = device

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.__model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, image):
        self.__model.eval()

        image = image.to(self.__device)
        
        with torch.no_grad():
            output = self.__model(image)
        
        return output

    def visualize_result(self, image, predicted_mask):
        plt.figure(figsize=(18, 6))
        mask = np.where(predicted_mask.squeeze().detach().numpy() < SEGMENTATION_THRESHOLD, 0, 255)

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.title('Original Image')

        # Predicted Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
        plt.title('Predicted Mask')

        # Combined Result
        plt.subplot(1, 3, 3)
        plt.imshow(image + mask, cmap='gray', vmin=0, vmax=255)
        plt.title('Combined Result')

        plt.show()

    def evaluate(self, dataset, save_folder_path=None, batch_size=BATCH_SIZE):
        self.__model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
        
        result = {metric.__name__: 0.0 for metric in self.__metrics}
        num_samples = 0
        
        with torch.no_grad():
            for image, true_mask_image, label in tqdm(data_loader, desc="Evaluation", unit="batch", leave=False):
                # Predictions
                pred = self.predict(image)

                # Save predicted image to a folder
                if save_folder_path:
                    for i in range(len(pred)):
                        mask = np.where(pred.squeeze().detach().numpy() < SEGMENTATION_THRESHOLD, 0, 255)
                        mask = TF.to_pil_image(mask)

                        if not os.path.exists(save_folder_path):
                            os.makedirs(save_folder_path)

                        mask.save(os.path.join(save_folder_path, f"predicted_mask_{num_samples + i}.png"))

                # Calculating all the metrics
                for metric in self.__metrics:
                    result[metric.__name__] += torch.sum(metric(pred, true_mask_image)).item()
                
                num_samples += len(pred)

        result = {key: value / num_samples for key, value in result.items()}
        for metric_name, avg_result in result.items():
            print(f"Average {metric_name}: {avg_result:.4f}")