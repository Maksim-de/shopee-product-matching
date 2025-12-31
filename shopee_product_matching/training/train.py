import os

import hydra
import mlflow
from data.preprocess_data import preprocess_data_for_model
from models.joint_model import MultiModalLightningModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

# import sys


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.abspath(__file__))


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train_model(cfg: DictConfig):
    print(cfg.text_model)
    print(cfg.image_model)
    print(cfg.data)
    print(cfg.train)
    print(cfg.mlflow)

    if cfg.mlflow.get("tracking_uri"):
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    experiment_name = cfg.mlflow.get("experiment_name", "shopee_experiment")
    mlflow.set_experiment(experiment_name)

    model = MultiModalLightningModule(
        lr_image=cfg.image_model.lr_image,
        lr_text=cfg.text_model.lr_text,
        lr_joint=cfg.lr_joint,
        weight_decay=cfg.weight_decay,
        alpha=cfg.alpha,
        beta=cfg.beta,
        base=cfg.base,
        momentum=cfg.momentum,
        retrieval_k_values=cfg.metriks_k,
    )

    datamodule = preprocess_data_for_model(
        df_path=cfg.data.df_path,
        little_filter_count=cfg.data.little_filter_count,
        tokenizer=model.tokenizator,
        train_ratio=cfg.data.train_ratio,
        image_dir=cfg.data.image_dir,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        p_sampler=cfg.data.p_sampler,
        k_sampler=cfg.data.k_sampler,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_precision@10",
        mode="max",
        save_top_k=3,
        filename="best-{epoch:02d}-{val_precision@10:.4f}",
        save_last=True,
        verbose=True,
    )

    tb_logger = TensorBoardLogger(
        save_dir="./logs", name="multimodal_model", default_hp_metric=False
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=cfg.mlflow.get("run_name"),
        tracking_uri=cfg.mlflow.get("tracking_uri", "file:./mlruns"),
        tags={"project": "shopee", "user": os.environ.get("USER", "unknown")},
        log_model=True,
    )

    loggers = [tb_logger, mlflow_logger]

    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="mps",
        logger=loggers,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.train.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        enable_progress_bar=cfg.train.enable_progress_bar,
        deterministic=cfg.train.deterministic,
        # Добавляем для отладки
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
    )

    trainer.fit(model, datamodule=datamodule)

    best_model = MultiModalLightningModule.load_from_checkpoint(checkpoint_callback.best_model_path)

    return best_model


if __name__ == "__main__":
    # Отключаем multiprocessing для диагностики
    os.environ["OMP_NUM_THREADS"] = "1"

    model = train_model()
