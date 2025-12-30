import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional


class TinyBERTWrapper(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.projection = nn.Linear(128, embedding_size)
        self.tokenizator = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    def forward(self, batch):
        # with torch.no_grad():
        #     output = self.bert(**batch)

        output = self.bert(**batch)
        cls_token = output.last_hidden_state[:, 0, :]
        x = self.projection(cls_token)
        return x
