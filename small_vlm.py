import torch
import torchvision
import numpy as np

from torch import nn
from transformers.models.bert.modeling_bert import BertEmbeddings


class SmallVLM(nn.Module):
    def __init__(self, num_answers: int, pretrained_embeddings, hidden_cls_dim: int = 1000, hidden_lstm_dim:int = 300, embedding_dimension: int = 300, pretrained_visual_encoder: bool = True,
                 train_emb: bool = False, device: str = "cuda") -> None:
        super().__init__()
        if pretrained_visual_encoder:
            weights = torchvision.models.VGG16_BN_Weights.DEFAULT
            self.visual_encoder = torchvision.models.vgg16_bn(weights=weights).features.to(device)
        else:
            self.visual_encoder = torchvision.models.vgg16_bn().features.to(device)

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=not train_emb).to(device)
        self.lstm = nn.LSTM(embedding_dimension, hidden_lstm_dim, bidirectional=True, batch_first=True).to(device)

        self.big_projection_layer = nn.Linear(25088, embedding_dimension).to(device)
        self.small_projection_layer = nn.Linear(1800, 300).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_cls_dim, bias=False),
            nn.BatchNorm1d(num_features=hidden_cls_dim),
            nn.ReLU(),
            nn.Linear(hidden_cls_dim, num_answers)
        ).to(device)

    def forward(self, batch_token_ids, lengths, batch_images):
        word_emb = self.embedding(batch_token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(word_emb, lengths, batch_first=True, enforce_sorted=False)
        _, (questions_features, _) = self.lstm(packed)
        questions_features = questions_features.mean(dim=0)

        visual_features = []
        for images in batch_images:
            features = self.visual_encoder(images)
            projected_features = self.big_projection_layer(features.flatten(start_dim=1))
            visual_features.append(projected_features.flatten())

        visual_features = torch.stack(visual_features)
        visual_features = self.small_projection_layer(visual_features)

        final_features = questions_features * visual_features

        return self.classifier(final_features)
