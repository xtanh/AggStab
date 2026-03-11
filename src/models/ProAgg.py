import os
import sys
import torch
import torch.nn as nn

from transformers import EsmForMaskedLM

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.models.factory import register_proagg_model


@register_proagg_model("proagg_mlp_v1")
class ProAggMLPV1(torch.nn.Module):
    def __init__(self, cfg):
        super(ProAggMLPV1, self).__init__()
        self.cfg = cfg
        self.use_lora = False

        saprot_path = cfg.model.saprot_path
        self.backbone = EsmForMaskedLM.from_pretrained(saprot_path)

        lora_cfg = cfg.model.get("lora", None)
        if lora_cfg and lora_cfg.get("enabled", False):
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.1),
                target_modules=lora_cfg.get("target_modules", ["query", "key", "value"]),
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
            self.use_lora = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, 1280)

        # Mean pooling: exclude CLS (pos 0) and PAD/EOS tokens
        mask = attention_mask.clone()
        mask[:, 0] = 0  # CLS
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        protein_repr = self.proj(pooled)  # (B, 1)

        return {'score': protein_repr}


@register_proagg_model("proagg_mlp_v2")
class ProAggMLPV2(torch.nn.Module):
    """V2: 更强的正则化，防止过拟合"""
    def __init__(self, cfg):
        super(ProAggMLPV2, self).__init__()
        self.cfg = cfg
        self.use_lora = False

        saprot_path = cfg.model.saprot_path
        self.backbone = EsmForMaskedLM.from_pretrained(saprot_path)

        lora_cfg = cfg.model.get("lora", None)
        if lora_cfg and lora_cfg.get("enabled", False):
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.1),
                target_modules=lora_cfg.get("target_modules", ["query", "key", "value"]),
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
            self.use_lora = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(1280, 640),
            nn.LayerNorm(640),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(640, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, 1280)

        # Mean pooling: exclude CLS (pos 0) and PAD/EOS tokens
        mask = attention_mask.clone()
        mask[:, 0] = 0  # CLS
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        protein_repr = self.proj(pooled)  # (B, 1)

        return {'score': protein_repr}


@register_proagg_model("proagg_mlp_v3")
class ProAggMLPV3(torch.nn.Module):
    """V3: Attention-based pooling + 更强的正则化"""
    def __init__(self, cfg):
        super(ProAggMLPV3, self).__init__()
        self.cfg = cfg
        self.use_lora = False

        saprot_path = cfg.model.saprot_path
        self.backbone = EsmForMaskedLM.from_pretrained(saprot_path)

        lora_cfg = cfg.model.get("lora", None)
        if lora_cfg and lora_cfg.get("enabled", False):
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.1),
                target_modules=lora_cfg.get("target_modules", ["query", "key", "value"]),
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
            self.use_lora = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Attention pooling: 学习残基重要性
        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.proj = nn.Sequential(
            nn.Linear(1280, 640),
            nn.LayerNorm(640),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(640, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, 1280)

        # 准备mask: exclude CLS (pos 0) and PAD/EOS tokens
        mask = attention_mask.clone()
        mask[:, 0] = 0  # CLS
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        # Attention pooling: 学习每个残基的重要性
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L)

        # 加权平均
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 1280)

        protein_repr = self.proj(pooled)  # (B, 1)

        return {'score': protein_repr, 'attention_weights': attn_weights}


ProAgg = ProAggMLPV1


if __name__ == "__main__":
    from src.config.utils import load_yaml_config
    cfg = load_yaml_config('/home/xy_th/Project_protein_aggregation/configs/default.yaml')
    model = ProAgg(cfg)
    print(model)
