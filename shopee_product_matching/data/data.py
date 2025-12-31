from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class ShopeeDataset(Dataset):
    """Кастомный датасет для Shopee"""

    def __init__(self, df, image_dir: str, augs: A.Compose):
        self.df = df
        self.augs = augs
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # Текстовые фичи (BERT эмбеддинги)
        text_features = {
            "input_ids": torch.tensor(self.df["input_ids"].iloc[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.df["attention_mask"].iloc[idx], dtype=torch.long),
        }

        # Изображение
        img_path = self.image_dir + self.df.iloc[idx]["image"]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.augs(image=image)["image"]

        # Целевая переменная
        target = torch.tensor(self.df["label_group"].iloc[idx], dtype=torch.long)

        return image, text_features, target


def make_augmentations(
    image_size: int = 420, is_train: bool = False, scale: float = 1.0
) -> A.Compose:
    """
    Создает аугментации для изображений

    Args:
        image_size: базовый размер изображения
        is_train: флаг тренировочных аугментаций
        scale: множитель для размера
    """
    im_size = int(round(scale * image_size))

    if is_train:
        transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=im_size, p=1.0),
                A.PadIfNeeded(
                    min_height=im_size,
                    min_width=im_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomGamma(p=0.5),
                    ],
                    p=0.3,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )
    else:
        transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=im_size, p=1.0),
                A.PadIfNeeded(
                    min_height=im_size,
                    min_width=im_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )

    return transforms


class PKSampler(torch.utils.data.Sampler):
    """PKSampler для метрического обучения"""

    def __init__(self, labels: List[int], P: int, K: int, seed: int = 42):
        """
        Args:
            labels: список меток для каждого примера
            P: количество классов в батче
            K: количество примеров каждого класса
            seed: random seed
        """
        super().__init__(None)
        self.P = P
        self.K = K
        self.batch_size = P * K
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Группируем индексы по классам
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        # Оставляем только классы с ≥K примерами
        self.valid_classes = [c for c, indices in self.class_indices.items() if len(indices) >= K]

        if len(self.valid_classes) < P:
            raise ValueError(
                f"Требуется минимум {P} классов с ≥{K} примерами, "
                f"но найдено только {len(self.valid_classes)}"
            )

        # Вычисляем количество батчей
        self.n_batches = (
            sum(
                len(indices) for c, indices in self.class_indices.items() if c in self.valid_classes
            )
            // self.batch_size
        )

        print(f"PKSampler: {len(self.valid_classes)} классов, {self.n_batches} батчей")

    def __iter__(self):
        for _ in range(self.n_batches):
            batch_indices = []
            # Выбираем P случайных классов
            selected_classes = self.rng.choice(self.valid_classes, size=self.P, replace=False)

            # Для каждого класса выбираем K примеров
            for cls in selected_classes:
                indices = self.class_indices[cls]
                # Если примеров больше чем K, выбираем случайные K
                if len(indices) > self.K:
                    selected = self.rng.choice(indices, size=self.K, replace=False)
                else:
                    # Если меньше, дублируем случайные
                    selected = self.rng.choice(indices, size=self.K, replace=True)
                batch_indices.extend(selected.tolist())

            yield batch_indices

    def __len__(self) -> int:
        return self.n_batches
