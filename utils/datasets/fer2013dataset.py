import os

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils.augmenters.augment import seg

EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class FER2013(Dataset):
    def __init__(self, stage, configs, tta=False, tta_size=48, transform=None):
        self._stage = stage
        self._configs = configs
        self._tta = tta
        self._tta_size = tta_size

        self._image_size = (configs["image_size"], configs["image_size"])

        self._data = pd.read_csv(
            os.path.join(configs["data_path"], "{}.csv".format(stage))
        )

        self._pixels = self._data["pixels"].tolist()
        self._emotions = pd.get_dummies(self._data["emotion"])

        # Use o transform fornecido, ou o transform padrão
        self._transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            ]
        )

    def is_tta(self):
        return self._tta == True

    def __len__(self):
        return len(self._pixels)

    def __getitem__(self, idx):
        pixels = self._pixels[idx]
        pixels = list(map(int, pixels.split(" ")))
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)

        image = cv2.resize(image, self._image_size)
        image = np.dstack([image] * 3)

        if self._stage == "train":
            image = seg(image=image)

        if self._stage == "test" and self._tta == True:
            images = [seg(image=image) for _ in range(self._tta_size)]
            images = torch.stack(list(map(self._transform, images)))  # Empilhar todas as imagens do TTA
            target = self._emotions.iloc[idx].idxmax()
            return images, target

        image = self._transform(image)
        target = self._emotions.iloc[idx].idxmax()
        return image, target


def fer2013(stage, configs=None, tta=False, tta_size=48, transform=None):
    """
    Wrapper function to initialize the FER2013 dataset with optional TTA and transform.
    
    Parameters:
    -----------
    stage : str
        Dataset stage, e.g., 'train', 'val', 'test'.
    configs : dict
        Configuration dictionary.
    tta : bool
        Whether to apply test-time augmentation (TTA).
    tta_size : int
        Number of augmentations for TTA.
    transform : torchvision.transforms.Compose
        Transformations to apply to the images.

    Returns:
    --------
    FER2013
        An instance of the FER2013 dataset.
    """
    return FER2013(stage, configs, tta, tta_size, transform)

if __name__ == "__main__":
    data = FER2013(
        "train",
        {
            "data_path": "/home/z/research/tee/saved/data/fer2013/",
            "image_size": 224,
            "in_channels": 3,
        },
    )
    import cv2

    targets = []

    for i in range(len(data)):
        image, target = data[i]
        cv2.imwrite("debug/{}.png".format(i), image)
        if i == 200:
            break
