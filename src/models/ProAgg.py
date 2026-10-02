"""Released SaProt reward architecture used by AggStab checkpoints."""

import torch
import torch.nn as nn
from transformers import EsmForMaskedLM

from src.models.factory import register_proagg_model


@register_proagg_model("proagg_mlp_v36")
class ProAggMLPV36(torch.nn.Module):
    """SaProt encoder with attention pooling and score/delta-G heads."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_lora = False
        self.backbone = EsmForMaskedLM.from_pretrained(cfg.model.saprot_path)

        lora_cfg = cfg.model.get("lora", None)
        if lora_cfg and lora_cfg.get("enabled", False):
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.1),
                target_modules=lora_cfg.get("target_modules", ["query", "key", "value"]),
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.use_lora = True
        else:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.7),
        )
        self.score_head = nn.Linear(256, 1)
        self.delta_g_head = nn.Linear(256, 1)

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict):
        input_ids = batch_dict["input_ids"]
        attention_mask = batch_dict["attention_mask"]
        outputs = self._get_esm()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for row, eos_index in enumerate(eos_indices):
            if eos_index > 0:
                mask[row, eos_index] = 0

        attention_scores = self.attention(hidden_states).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=1)
        pooled = (hidden_states * attention_weights.unsqueeze(-1)).sum(dim=1)
        features = self.proj(pooled)

        return {
            "score": self.score_head(features),
            "deltaG": self.delta_g_head(features),
            "attention_weights": attention_weights,
        }
