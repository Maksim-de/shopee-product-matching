import madgrad
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.image_model import SimpleMobileNetV3
from models.text_models import TinyBERTWrapper
from pytorch_metric_learning.losses import MultiSimilarityLoss


def compute_retrieval_metrics(embeddings, labels, k_values=50):
    """
    Вычисляет метрики retrieval для Shopee с усреднением по запросам

    Args:
        embeddings: матрица эмбеддингов [n_samples, embedding_dim]
        labels: метки товаров [n_samples]
        k_values: значения K для recall@K, precision@K, F1@K

    Returns:
        metrics: словарь с метриками
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    n_samples = len(embeddings)

    # 1. Находим k ближайших соседей для каждого товара
    max_k = k_values + 1  # +1 потому что первый сосед - сам элемент
    knn = NearestNeighbors(n_neighbors=max_k, metric="cosine")
    knn.fit(embeddings)

    distances, indices = knn.kneighbors(embeddings)

    # 2. Вычисляем recall@K, precision@K, F1@K для каждого K
    metrics = {}

    all_recalls = []
    all_precisions = []
    all_f1 = []
    ndcg_scores = []

    for i in range(n_samples):
        neighbor_indices = indices[i, 1 : k_values + 1]
        neighbor_labels = labels[neighbor_indices]

        # Сколько соседей имеют тот же label?
        correct = np.sum(neighbor_labels == labels[i])
        total_relevant_items = np.sum(labels == labels[i]) - 1
        if total_relevant_items > 0:
            recall_i = correct / total_relevant_items
        else:
            recall_i = 0.0

        precision_i = correct / k_values

        if (precision_i + recall_i) > 0:
            f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
        else:
            f1_i = 0.0
        all_recalls.append(recall_i)
        all_precisions.append(precision_i)
        all_f1.append(f1_i)

        relevance_vector = (neighbor_labels == labels[i]).astype(float)
        dcg = 0.0
        for j, rel in enumerate(relevance_vector, start=1):
            dcg += rel / np.log2(j + 1)

        num_relevant = min(total_relevant_items, k_values)
        idcg = 0.0
        for j in range(1, num_relevant + 1):
            idcg += 1.0 / np.log2(j + 1)

        # NDCG = DCG / IDCG
        if idcg > 0:
            ndcg_i = dcg / idcg
        else:
            ndcg_i = 0.0
        ndcg_scores.append(ndcg_i)

        # Усредняем по всем запросам
    metrics[f"recall@{k_values}"] = np.mean(all_recalls)
    metrics[f"precision@{k_values}"] = np.mean(all_precisions)
    metrics[f"f1@{k_values}"] = np.mean(all_f1)
    metrics[f"ndcg@{k_values}"] = np.mean(ndcg_scores)

    return metrics


class JointModel(nn.Module):
    def __init__(self, text_model, image_model):
        super().__init__()
        self.text_model = text_model
        self.tokenizator = text_model.tokenizator
        self.image_model = image_model

    def forward(self, image_input, text_input):
        image_emb = self.image_model(image_input)
        text_emb = self.text_model(text_input)
        x = torch.cat((image_emb, text_emb), dim=1)
        x = F.normalize(x, p=2, dim=1)
        return x


# Lightning модуль
class MultiModalLightningModule(pl.LightningModule):
    def __init__(
        self,
        text_model=TinyBERTWrapper(),
        image_model=SimpleMobileNetV3(),
        lr_image=1e-4,
        lr_text=5e-5,
        lr_joint=1e-3,
        weight_decay=1e-5,
        alpha=2.0,
        beta=100.0,
        base=0.7,
        momentum=0.9,
        retrieval_k_values=[5, 10, 50],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["text_model", "image_model"])
        self.validation_step_outputs = []

        # Инициализация моделей
        self.model = JointModel(text_model, image_model)
        self.tokenizator = text_model.tokenizator
        self.criterion = MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
        self.momentum = momentum

    def forward(self, image_input, text_input):
        return self.model(image_input, text_input)

    def training_step(self, batch, batch_idx):
        images, texts, labels = batch
        embeddings = self(images, texts)
        loss = self.criterion(embeddings, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, labels = batch
        embeddings = self(images, texts)
        loss = self.criterion(embeddings, labels)

        self.validation_step_outputs.append(
            {
                "embeddings": embeddings.detach().cpu(),
                "labels": labels.detach().cpu(),
                "val_loss": loss.detach(),
            }
        )

        # Логируем loss
        self.log("val_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_end(self):
        # Проверяем, что есть данные
        if not self.validation_step_outputs:
            # Логируем нулевые значения
            for k in self.hparams.retrieval_k_values:
                self.log(f"val_recall@{k}", torch.tensor(0.0), prog_bar=(k == 50))
                self.log(f"val_precision@{k}", torch.tensor(0.0), prog_bar=(k == 50))
                self.log(f"val_F1@{k}", torch.tensor(0.0), prog_bar=(k == 50))
            self.validation_step_outputs.clear()
            return

        try:
            # Собираем данные
            all_embeddings = torch.cat(
                [x["embeddings"] for x in self.validation_step_outputs], dim=0
            )
            all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0)

            # Средний loss
            # avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
            # self.log("val_loss", avg_loss, prog_bar=True)

            # Вычисляем метрики
            if len(all_embeddings) > 0 and len(all_labels) > 0:
                embeddings_np = all_embeddings.numpy()
                labels_np = all_labels.numpy().astype(int)

                for k in self.hparams.retrieval_k_values:
                    try:
                        metrics = compute_retrieval_metrics(embeddings_np, labels_np, k)

                        # Логируем все метрики
                        for metric_name, metric_value in metrics.items():
                            log_name = f"val_{metric_name}"
                            self.log(
                                log_name, metric_value, prog_bar=(metric_name == f"recall@{k}")
                            )

                    except Exception as e:
                        print(f"Ошибка при вычислении метрик для k={k}: {e}")
                        # Логируем нулевые значения
                        self.log(f"val_recall@{k}", torch.tensor(0.0), prog_bar=(k == 50))
                        self.log(f"val_precision@{k}", torch.tensor(0.0), prog_bar=False)
                        self.log(f"val_f1@{k}", torch.tensor(0.0), prog_bar=False)
            else:
                for k in self.hparams.retrieval_k_values:
                    self.log(f"val_recall@{k}", torch.tensor(0.0), prog_bar=(k == 50))
                    self.log(f"val_precision@{k}", torch.tensor(0.0), prog_bar=False)
                    self.log(f"val_f1@{k}", torch.tensor(0.0), prog_bar=False)

        except Exception as e:
            print(f"Ошибка в on_validation_epoch_end: {e}")
            # Логируем нулевые значения
            # self.log("val_loss", torch.tensor(0.0), prog_bar=True)
            for k in self.hparams.retrieval_k_values:
                self.log(f"val_recall@{k}", torch.tensor(0.0), prog_bar=(k == 50))

        # Очищаем outputs
        self.validation_step_outputs.clear()

    def on_validation_epoch_start(self):
        # Очищаем outputs в начале каждой эпохи
        self.validation_step_outputs.clear()

    def validation_step_end(self, step_output):
        # Сохраняем output для дальнейшей обработки
        self.validation_outputs.append(step_output)
        return step_output

    def configure_optimizers(self):
        # Создаем группы параметров с разными learning rates
        param_groups = []

        # Параметры для image_model
        image_params = []
        for name, param in self.model.named_parameters():
            if "image_model" in name and param.requires_grad:
                image_params.append(param)

        if image_params:
            param_groups.append(
                {"params": image_params, "lr": self.hparams.lr_image, "name": "image_params"}
            )

        # Параметры для text_model
        text_params = []
        for name, param in self.model.named_parameters():
            if "text_model" in name and param.requires_grad:
                text_params.append(param)

        if text_params:
            param_groups.append(
                {"params": text_params, "lr": self.hparams.lr_text, "name": "text_params"}
            )

        # Остальные параметры (joint слои)
        other_params = []
        for name, param in self.model.named_parameters():
            if ("image_model" not in name) and ("text_model" not in name) and param.requires_grad:
                other_params.append(param)

        if other_params:
            param_groups.append(
                {"params": other_params, "lr": self.hparams.lr_joint, "name": "joint_params"}
            )

        # Создаем один оптимизатор с разными LR для разных групп
        optimizer = madgrad.MADGRAD(
            param_groups, momentum=self.momentum, weight_decay=self.hparams.weight_decay
        )

        return optimizer
