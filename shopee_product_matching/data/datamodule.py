from typing import Optional

import pytorch_lightning as pl
import torch
from data.data import PKSampler, ShopeeDataset, make_augmentations
from torch.utils.data import DataLoader


class ShopeeDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule для Shopee"""

    def __init__(
        self,
        train_df,
        val_df,
        image_dir: str,
        image_size: int = 224,
        train_scale: float = 1.0,
        val_scale: float = 1.0,
        P: int = 16,  # PK-Sampler параметры
        K: int = 4,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.image_dir = image_dir
        self.image_size = image_size
        self.train_scale = train_scale
        self.val_scale = val_scale
        self.P = P
        self.K = K
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        # Инициализация датасетов
        self.train_dataset = None
        self.val_dataset = None

        # Инициализация сэмплера
        self.train_sampler = None

    def prepare_data(self):
        """Скачивание/подготовка данных (если нужно)"""
        # Здесь можно добавить скачивание данных
        pass

    def setup(self, stage: Optional[str] = None):
        """Создание датасетов"""
        if stage == "fit" or stage is None:
            # Создаем аугментации для тренировки
            train_augs = make_augmentations(
                image_size=self.image_size, is_train=True, scale=self.train_scale
            )

            # Создаем тренировочный датасет
            self.train_dataset = ShopeeDataset(
                df=self.train_df, image_dir=self.image_dir, augs=train_augs
            )

            # Создаем PKSampler для тренировки
            self.train_sampler = PKSampler(
                labels=self.train_df["label_group"].values, P=self.P, K=self.K
            )

            # Создаем аугментации для валидации
            val_augs = make_augmentations(
                image_size=self.image_size, is_train=False, scale=self.val_scale
            )

            # Создаем валидационный датасет
            self.val_dataset = ShopeeDataset(
                df=self.val_df, image_dir=self.image_dir, augs=val_augs
            )

        if stage == "test" or stage is None:
            # Для теста используем те же аугментации что и для валидации
            test_augs = make_augmentations(
                image_size=self.image_size, is_train=False, scale=self.val_scale
            )

            self.test_dataset = ShopeeDataset(
                df=self.val_df,  # или отдельный test_df если есть
                image_dir=self.image_dir,
                augs=test_augs,
            )

    def train_dataloader(self) -> DataLoader:
        """DataLoader для тренировки"""
        if self.train_dataset is None:
            raise RuntimeError("Сначала вызовите setup(stage='fit')")

        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """DataLoader для валидации"""
        if self.val_dataset is None:
            raise RuntimeError("Сначала вызовите setup(stage='fit')")

        return DataLoader(
            self.val_dataset,
            batch_size=self.P * self.K,  # тот же размер батча что и в train
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """DataLoader для тестирования"""
        if self.test_dataset is None:
            self.setup(stage="test")

        return DataLoader(
            self.test_dataset,
            batch_size=self.P * self.K,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
        )

    def predict_dataloader(self) -> DataLoader:
        """DataLoader для инференса (аналогичен test)"""
        return self.test_dataloader()
