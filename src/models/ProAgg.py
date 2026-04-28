import os
import sys
import torch
import torch.nn as nn

from transformers import EsmForMaskedLM, EsmModel

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


@register_proagg_model("proagg_mlp_v4")
class ProAggMLPV4(torch.nn.Module):
    """V4: Attention pooling + 更宽网络 + 更强的正则化"""
    def __init__(self, cfg):
        super(ProAggMLPV4, self).__init__()
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

        # 更宽的网络: 1280 -> 1024 -> 256 -> 1
        self.proj = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.6),  # 更强的正则化
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.6),  # 更强的正则化
            nn.Linear(256, 1)
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




@register_proagg_model("proagg_mlp_v6")
class ProAggMLPV6(torch.nn.Module):
    """V6: 多层特征融合 (最后4层) + Attention pooling + 宽网络"""
    def __init__(self, cfg):
        super(ProAggMLPV6, self).__init__()
        self.cfg = cfg
        self.use_lora = False
        self.num_layers_to_fuse = 4  # 融合最后4层

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

        # 多层融合: 4层 * 1280 = 5120, 投影到1280
        self.layer_fusion = nn.Sequential(
            nn.Linear(1280 * self.num_layers_to_fuse, 1280),
            nn.LayerNorm(1280),
            nn.GELU(),
            nn.Dropout(0.6)
        )

        # Attention pooling: 学习残基重要性
        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # MLP head: 1280 -> 1024 -> 256 -> 1
        self.proj = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1)
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

        # 获取最后4层 hidden states
        hidden_states = torch.stack(outputs.hidden_states[-self.num_layers_to_fuse:], dim=0)
        # hidden_states: (4, B, L, 1280)

        B, L = input_ids.shape

        # 融合多层: 拼接后投影
        fused = hidden_states.permute(1, 2, 0, 3).reshape(B, L, -1)  # (B, L, 4*1280)
        fused = self.layer_fusion(fused)  # (B, L, 1280)

        # 准备mask
        mask = attention_mask.clone()
        mask[:, 0] = 0  # CLS
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        # Attention pooling
        attn_scores = self.attention(fused).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L)

        # 加权平均
        pooled = (fused * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 1280)

        protein_repr = self.proj(pooled)  # (B, 1)

        return {'score': protein_repr, 'attention_weights': attn_weights}


@register_proagg_model("proagg_mlp_v7")
class ProAggMLPV7(torch.nn.Module):
    """V7: Attention pooling + Residual MLP (Pre-LN结构) + 宽网络"""
    def __init__(self, cfg):
        super(ProAggMLPV7, self).__init__()
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

        # Pre-LN MLP with residual: 1280 -> 1024 -> 256 -> 1
        self.norm1 = nn.LayerNorm(1280)
        self.fc1 = nn.Linear(1280, 1024)
        self.dropout1 = nn.Dropout(0.6)

        self.norm2 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.6)

        # 投影用于残差连接
        self.proj_residual = nn.Linear(1280, 256)

        self.norm3 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 1)

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

        # 准备mask
        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        # Attention pooling
        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)

        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 1280)

        # Pre-LN Residual Block 1: 1280 -> 1024
        x = self.norm1(pooled)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.dropout1(x)

        # Pre-LN Residual Block 2: 1024 -> 256 (with residual from pooled)
        x = self.norm2(x)
        x = torch.nn.functional.gelu(self.fc2(x))
        x = self.dropout2(x)

        # Residual connection: project pooled (1280) to 256
        residual = self.proj_residual(pooled)
        x = x + residual  # Residual connection

        # Final layer: 256 -> 1
        x = self.norm3(x)
        protein_repr = self.fc3(x)

        return {'score': protein_repr, 'attention_weights': attn_weights}


@register_proagg_model("proagg_mlp_v8")
class ProAggMLPV8(torch.nn.Module):
    """V8: Attention pooling + Deeper MLP (4 layers) + moderate width"""
    def __init__(self, cfg):
        super(ProAggMLPV8, self).__init__()
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

        # Deeper MLP: 1280 -> 512 -> 256 -> 128 -> 1
        self.proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
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


@register_proagg_model("proagg_mlp_v22")
class ProAggMLPV22(torch.nn.Module):
    """V22: V8 with BatchNorm1d in the MLP head."""
    def __init__(self, cfg):
        super(ProAggMLPV22, self).__init__()
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

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc1 = nn.Linear(1280, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)

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
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        x = self.fc1(pooled)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.drop3(x)
        protein_repr = self.fc4(x)

        return {'score': protein_repr, 'attention_weights': attn_weights}


@register_proagg_model("proagg_mlp_v23")
class ProAggMLPV23(torch.nn.Module):
    """V23: Dual encoder with SaProt and ESM2 pooled fusion."""
    def __init__(self, cfg):
        super(ProAggMLPV23, self).__init__()
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

        esm2_path = cfg.model.esm2_path
        self.esm2 = EsmModel.from_pretrained(esm2_path)
        if not cfg.model.get("esm2_trainable", False):
            for param in self.esm2.parameters():
                param.requires_grad = False

        saprot_hidden = 1280
        esm2_hidden = self.esm2.config.hidden_size
        fusion_dim = cfg.model.get("fusion_dim", 256)

        self.attention = nn.Sequential(
            nn.Linear(saprot_hidden, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.saprot_proj = nn.Sequential(
            nn.Linear(saprot_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.5),
        )
        self.esm2_proj = nn.Sequential(
            nn.Linear(esm2_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.5),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1),
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']
        protein_input_ids = batch_dict['protein_input_ids']
        protein_attention_mask = batch_dict['protein_attention_mask']

        saprot_esm = self._get_esm()
        saprot_outputs = saprot_esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        saprot_hidden = saprot_outputs.hidden_states[-1]

        saprot_mask = attention_mask.clone()
        saprot_mask[:, 0] = 0
        eos_indices = saprot_mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                saprot_mask[i, eos_idx] = 0

        attn_scores = self.attention(saprot_hidden).squeeze(-1)
        attn_scores = attn_scores.masked_fill(saprot_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        saprot_pooled = (saprot_hidden * attn_weights.unsqueeze(-1)).sum(dim=1)

        esm2_outputs = self.esm2(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
        )
        esm2_hidden = esm2_outputs.last_hidden_state
        protein_mask = protein_attention_mask.clone()
        protein_mask[:, 0] = 0
        eos_indices = protein_mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                protein_mask[i, eos_idx] = 0
        protein_mask_expanded = protein_mask.unsqueeze(-1).float()
        esm2_pooled = (esm2_hidden * protein_mask_expanded).sum(dim=1) / protein_mask_expanded.sum(dim=1).clamp(min=1.0)

        saprot_feat = self.saprot_proj(saprot_pooled)
        esm2_feat = self.esm2_proj(esm2_pooled)
        gate = self.fusion_gate(torch.cat([saprot_feat, esm2_feat], dim=-1))
        fused = torch.cat([saprot_feat, gate * esm2_feat], dim=-1)
        score = self.head(fused)

        return {'score': score, 'attention_weights': attn_weights}


@register_proagg_model("proagg_mlp_v24")
class ProAggMLPV24(torch.nn.Module):
    """V24: V13-style model with an explicit latent bottleneck for contrastive shaping."""
    def __init__(self, cfg):
        super(ProAggMLPV24, self).__init__()
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

        latent_dim = int(cfg.model.get("latent_dim", 128))

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, latent_dim),
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, 1),
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict, return_embedding=False):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        latent = self.encoder(pooled)
        score = self.score_head(latent)

        out = {
            'score': score,
            'attention_weights': attn_weights,
        }
        if return_embedding:
            out['embedding'] = nn.functional.normalize(latent, dim=1)
        return out


@register_proagg_model("proagg_mlp_v25")
class ProAggMLPV25(torch.nn.Module):
    """V25: V24 with Multi-Head Self-Attention replacing MLP attention."""
    def __init__(self, cfg):
        super(ProAggMLPV25, self).__init__()
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

        latent_dim = int(cfg.model.get("latent_dim", 128))
        self.num_heads = cfg.model.get("num_attention_heads", 8)

        # Multi-Head Self-Attention for residue interaction
        self.self_attn = nn.MultiheadAttention(
            embed_dim=1280,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Feed-forward after attention
        self.attn_ffn = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Final attention pooling (per-position scoring)
        self.attention_pool = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.encoder = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, latent_dim),
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, 1),
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict, return_embedding=False):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, 1280)

        # Create key padding mask for attention
        # True = masked (ignored), False = attended
        mask = attention_mask.bool()  # (B, L)

        # Apply Self-Attention
        # query, key, value are all from hidden_states
        attn_output, attn_weights = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~mask,  # Invert: True = ignore
            need_weights=True,
            average_attn_weights=False  # Return per-head weights
        )  # attn_output: (B, L, 1280), attn_weights: (B, num_heads, L, L)

        # Feed-forward after attention
        ffn_output = self.attn_ffn(attn_output)  # (B, L, 128)

        # Attention pooling (per-position scoring)
        position_scores = self.attention_pool(ffn_output).squeeze(-1)  # (B, L)

        # Mask out special tokens
        mask_processed = attention_mask.clone()
        mask_processed[:, 0] = 0  # CLS
        eos_indices = mask_processed.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask_processed[i, eos_idx] = 0

        position_scores = position_scores.masked_fill(mask_processed == 0, float('-inf'))
        attn_weights_pool = torch.softmax(position_scores, dim=1)  # (B, L)

        # Weighted pooling
        pooled = (hidden_states * attn_weights_pool.unsqueeze(-1)).sum(dim=1)  # (B, 1280)

        latent = self.encoder(pooled)
        score = self.score_head(latent)

        out = {
            'score': score,
            'attention_weights': attn_weights_pool,
        }
        if return_embedding:
            out['embedding'] = nn.functional.normalize(latent, dim=1)
        return out


@register_proagg_model("proagg_mlp_v9")
class ProAggMLPV9(torch.nn.Module):
    """V9: Attention pooling + 对比学习 (Contrastive Learning)"""
    def __init__(self, cfg):
        super(ProAggMLPV9, self).__init__()
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

        # MLP head: 1280 -> 1024 -> 256 -> 1
        self.proj = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.6),
        )

        # 最终输出层
        self.fc_out = nn.Linear(256, 1)

        # 对比学习投影头 (将embedding投影到对比空间)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict, return_embedding=False):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, 1280)

        # 准备mask
        mask = attention_mask.clone()
        mask[:, 0] = 0  # CLS
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        # Attention pooling
        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)

        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 1280)

        # MLP特征
        features = self.proj(pooled)  # (B, 256)

        # 预测分数
        score = self.fc_out(features)  # (B, 1)

        if return_embedding:
            # 对比学习embedding
            emb = self.contrastive_proj(features)  # (B, 64)
            emb = nn.functional.normalize(emb, dim=1)
            return {'score': score, 'embedding': emb, 'attention_weights': attn_weights}

        return {'score': score, 'attention_weights': attn_weights}


@register_proagg_model("proagg_mlp_v13")
class ProAggMLPV13(torch.nn.Module):
    """V13: Attention pooling + 更强正则化 (attention dropout + weight decay)"""
    def __init__(self, cfg):
        super(ProAggMLPV13, self).__init__()
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

        # Attention pooling with dropout
        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),  # Attention层加dropout
            nn.Linear(128, 1)
        )

        # MLP with higher dropout
        self.proj = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.7),  # 从0.6提升到0.7
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.7),  # 从0.6提升到0.7
            nn.Linear(256, 1)
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

        # 准备mask
        mask = attention_mask.clone()
        mask[:, 0] = 0  # CLS
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        # Attention pooling with dropout
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L)

        # 加权平均
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 1280)

        protein_repr = self.proj(pooled)  # (B, 1)

        return {'score': protein_repr, 'attention_weights': attn_weights}


@register_proagg_model("proagg_mlp_v14")
class ProAggMLPV14(torch.nn.Module):
    """V14: Multi-task learning - predicts 50, 75, 4 clip aggregation scores."""
    def __init__(self, cfg):
        super(ProAggMLPV14, self).__init__()
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

        # Attention pooling with dropout (from v13)
        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Shared feature extractor (from v13)
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

        # Multi-task heads: 50_clip, 75_clip, 4_clip
        self.head_50 = nn.Linear(256, 1)
        self.head_75 = nn.Linear(256, 1)
        self.head_4 = nn.Linear(256, 1)

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

        # Prepare mask
        mask = attention_mask.clone()
        mask[:, 0] = 0  # CLS
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        # Attention pooling
        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted average
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 1280)

        # Shared features
        features = self.proj(pooled)  # (B, 256)

        # Multi-task predictions
        pred_50 = self.head_50(features)  # (B, 1)
        pred_75 = self.head_75(features)  # (B, 1)
        pred_4 = self.head_4(features)    # (B, 1)

        return {
            'score_50': pred_50,
            'score_75': pred_75,
            'score_4': pred_4,
            'score': pred_75,  # For compatibility (main task)
            'attention_weights': attn_weights
        }


@register_proagg_model("proagg_mlp_v35")
class ProAggMLPV35(torch.nn.Module):
    """V35: V13 backbone with 3 soft experts for tail/middle/upper regimes."""
    def __init__(self, cfg):
        super(ProAggMLPV35, self).__init__()
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

        self.num_experts = int(cfg.model.get("num_experts", 3))

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
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

        self.gate = nn.Linear(256, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )
            for _ in range(self.num_experts)
        ])

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
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)

        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        features = self.proj(pooled)

        gate_logits = self.gate(features)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        expert_scores = torch.cat([expert(features) for expert in self.experts], dim=-1)
        score = (gate_weights * expert_scores).sum(dim=-1, keepdim=True)

        return {
            'score': score,
            'score_bin_logits': gate_logits,
            'expert_scores': expert_scores,
            'gate_weights': gate_weights,
            'attention_weights': attn_weights,
        }


@register_proagg_model("proagg_mlp_v36")
class ProAggMLPV36(torch.nn.Module):
    """V36: V13 backbone with an auxiliary deltaG prediction head."""
    def __init__(self, cfg):
        super(ProAggMLPV36, self).__init__()
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

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
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
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)

        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        features = self.proj(pooled)

        score = self.score_head(features)
        delta_g = self.delta_g_head(features)

        return {
            'score': score,
            'deltaG': delta_g,
            'attention_weights': attn_weights,
        }


@register_proagg_model("proagg_mlp_v16")
class ProAggMLPV16(torch.nn.Module):
    """V16: V13 backbone with an auxiliary score-bin classification head."""
    def __init__(self, cfg):
        super(ProAggMLPV16, self).__init__()
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

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.feature_proj = nn.Sequential(
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

        bin_edges = cfg.train.get("bin_edges", [-3.0, -2.0, -1.0, 0.0])
        self.num_bins = len(bin_edges) + 1
        self.bin_head = nn.Linear(256, self.num_bins)

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
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        features = self.feature_proj(pooled)
        score = self.score_head(features)
        score_bin_logits = self.bin_head(features)

        return {
            'score': score,
            'score_bin_logits': score_bin_logits,
            'attention_weights': attn_weights,
        }


@register_proagg_model("proagg_mlp_v17")
class ProAggMLPV17(torch.nn.Module):
    """V17: V13 backbone with an auxiliary ordinal head."""
    def __init__(self, cfg):
        super(ProAggMLPV17, self).__init__()
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

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.feature_proj = nn.Sequential(
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

        ordinal_thresholds = cfg.train.get("ordinal_thresholds", [-3.0, -2.0, -1.0, 0.0])
        self.num_ordinal_tasks = len(ordinal_thresholds)
        self.ordinal_head = nn.Linear(256, self.num_ordinal_tasks)

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
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        features = self.feature_proj(pooled)
        score = self.score_head(features)
        ordinal_logits = self.ordinal_head(features)

        return {
            'score': score,
            'ordinal_logits': ordinal_logits,
            'attention_weights': attn_weights,
        }


@register_proagg_model("proagg_mlp_v21")
class ProAggMLPV21(torch.nn.Module):
    """V21: V17 ordinal model with BatchNorm1d instead of LayerNorm."""
    def __init__(self, cfg):
        super(ProAggMLPV21, self).__init__()
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

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.fc1 = nn.Linear(1280, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.7)
        self.drop2 = nn.Dropout(0.7)
        self.score_head = nn.Linear(256, 1)

        ordinal_thresholds = cfg.train.get("ordinal_thresholds", [-3.0, -2.0, -1.0, 0.0])
        self.num_ordinal_tasks = len(ordinal_thresholds)
        self.ordinal_head = nn.Linear(256, self.num_ordinal_tasks)

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
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        features = self.fc1(pooled)
        features = self.bn1(features)
        features = self.act(features)
        features = self.drop1(features)
        features = self.fc2(features)
        features = self.bn2(features)
        features = self.act(features)
        features = self.drop2(features)

        score = self.score_head(features)
        ordinal_logits = self.ordinal_head(features)

        return {
            'score': score,
            'ordinal_logits': ordinal_logits,
            'attention_weights': attn_weights,
        }


@register_proagg_model("proagg_mlp_v18")
class ProAggMLPV18(torch.nn.Module):
    """V18: V13 backbone with a lightweight contrastive projection head."""
    def __init__(self, cfg):
        super(ProAggMLPV18, self).__init__()
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

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.feature_proj = nn.Sequential(
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
        self.contrastive_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict, return_embedding=False):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        features = self.feature_proj(pooled)
        score = self.score_head(features)

        out = {
            'score': score,
            'attention_weights': attn_weights,
        }
        if return_embedding:
            emb = self.contrastive_proj(features)
            out['embedding'] = nn.functional.normalize(emb, dim=1)
        return out


@register_proagg_model("proagg_mlp_v19")
class ProAggMLPV19(torch.nn.Module):
    """V19: V13-style head with attention, mean, and max pooled features."""
    def __init__(self, cfg):
        super(ProAggMLPV19, self).__init__()
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

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.proj = nn.Sequential(
            nn.Linear(1280 * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(256, 1)
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
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        mask_expanded = mask.unsqueeze(-1).float()
        denom = mask_expanded.sum(dim=1).clamp(min=1.0)
        mean_pooled = (hidden_states * mask_expanded).sum(dim=1) / denom

        masked_hidden = hidden_states.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        max_pooled = masked_hidden.max(dim=1).values
        max_pooled = torch.where(torch.isfinite(max_pooled), max_pooled, torch.zeros_like(max_pooled))

        pooled = torch.cat([attn_pooled, mean_pooled, max_pooled], dim=-1)
        protein_repr = self.proj(pooled)
        return {'score': protein_repr, 'attention_weights': attn_weights}


@register_proagg_model("proagg_mlp_v20")
class ProAggMLPV20(torch.nn.Module):
    """V20: V13 backbone with residual local-hotspot pooling."""
    def __init__(self, cfg):
        super(ProAggMLPV20, self).__init__()
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

        self.topk_tokens = int(cfg.model.get("hotspot_topk_tokens", 8))

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Keep the original global path as the anchor and add only a gated local residual.
        self.local_proj = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.LayerNorm(1280),
            nn.GELU(),
        )
        self.local_gate = nn.Sequential(
            nn.Linear(1280 * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1280),
            nn.Sigmoid(),
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
            nn.Linear(256, 1)
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
        hidden_states = outputs.hidden_states[-1]

        mask = attention_mask.clone()
        mask[:, 0] = 0
        eos_indices = mask.sum(dim=1).long() - 1
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx > 0:
                mask[i, eos_idx] = 0

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        global_pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        local_pooled = []
        for i in range(hidden_states.size(0)):
            valid_idx = torch.nonzero(mask[i], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                local_pooled.append(torch.zeros_like(global_pooled[i]))
                continue

            k = min(self.topk_tokens, int(valid_idx.numel()))
            topk_pos = torch.topk(attn_scores[i, valid_idx], k=k, dim=0).indices
            selected_idx = valid_idx[topk_pos]
            local_vec = hidden_states[i, selected_idx].mean(dim=0)
            local_pooled.append(local_vec)

        local_pooled = torch.stack(local_pooled, dim=0)
        local_residual = self.local_proj(local_pooled)
        gate = self.local_gate(torch.cat([global_pooled, local_pooled], dim=-1))
        fused = global_pooled + gate * local_residual

        protein_repr = self.proj(fused)
        return {'score': protein_repr, 'attention_weights': attn_weights}


class HotspotEncoder(nn.Module):
    """Extract and encode key residue hotspots for aggregation prediction."""

    def __init__(self, top_k=5, min_window=3):
        super().__init__()
        self.top_k = top_k
        self.min_window = min_window

        self.hotspot_processor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1280),
        )

    def forward(self, hidden_states, attn_weights, mask):
        B, L, D = hidden_states.shape
        device = hidden_states.device

        attn_masked = attn_weights.masked_fill(~mask.bool(), float('-inf'))
        k = min(self.top_k, L)
        topk_values, topk_indices = torch.topk(attn_masked, k=k, dim=1)

        hotspot_feats = []
        for b in range(B):
            seq_hotspots = []
            for idx in topk_indices[b]:
                if attn_masked[b, idx] == float('-inf'):
                    continue
                start = max(0, idx.item() - self.min_window // 2)
                end = min(L, idx.item() + self.min_window // 2 + 1)
                window_feat = hidden_states[b, start:end, :].mean(dim=0)
                seq_hotspots.append(window_feat)

            if len(seq_hotspots) > 0:
                seq_hotspot = torch.stack(seq_hotspots).mean(dim=0)
            else:
                seq_hotspot = torch.zeros(D, device=device)
            hotspot_feats.append(seq_hotspot)

        hotspot_features = torch.stack(hotspot_feats)
        hotspot_features = self.hotspot_processor(hotspot_features)

        valid_mask = attn_masked.gather(1, topk_indices) != float('-inf')

        hotspot_info = {
            'indices': topk_indices,
            'scores': torch.softmax(topk_values.masked_fill(
                topk_values == float('-inf'), 0), dim=1),
            'valid_mask': valid_mask,
        }

        return hotspot_features, hotspot_info


class ResidueAttributor:
    """Compute per-residue contribution to aggregation score."""

    @staticmethod
    def compute_gradient_attribution(model, hidden_states, attn_weights, mask):
        if not hidden_states.requires_grad:
            hidden_states = hidden_states.detach().requires_grad_(True)

        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        features = model.hotspot_fusion(pooled)
        score = model.proj(features)

        score.sum().backward()

        if hidden_states.grad is not None:
            grad = hidden_states.grad
            attribution = (grad * hidden_states).sum(dim=-1)
            attribution = attribution.masked_fill(~mask.bool(), 0)
        else:
            attribution = torch.zeros_like(attn_weights)

        return attribution.detach()

    @staticmethod
    def compute_attention_attribution(attn_weights, mask):
        return attn_weights.masked_fill(~mask.bool(), 0)


@register_proagg_model("proagg_mlp_v33")
class ProAggMLPV33(nn.Module):
    """v33: Interpretable ProAgg with hotspot encoding and residue attribution."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_lora = False

        from transformers import EsmForMaskedLM
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

        self.attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        hotspot_cfg = cfg.model.get("hotspot", {})
        self.hotspot_encoder = HotspotEncoder(
            top_k=hotspot_cfg.get("top_k", 5),
            min_window=hotspot_cfg.get("min_window", 3)
        )

        self.hotspot_fusion = nn.Sequential(
            nn.Linear(2560, 1280),
            nn.LayerNorm(1280),
            nn.GELU(),
            nn.Dropout(0.5),
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
            nn.Linear(256, 1)
        )

    def _get_esm(self):
        if self.use_lora:
            return self.backbone.base_model.model.esm
        return self.backbone.esm

    def forward(self, batch_dict, return_interpretation=False):
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']

        esm = self._get_esm()
        outputs = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]

        B, L, D = hidden_states.shape

        mask = attention_mask.clone().bool()
        mask[:, 0] = False

        seq_lengths = attention_mask.sum(dim=1) - 1
        for b in range(B):
            eos_pos = seq_lengths[b].item()
            if eos_pos > 0 and eos_pos < L:
                mask[b, eos_pos] = False

        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        global_pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        hotspot_feat, hotspot_info = self.hotspot_encoder(
            hidden_states, attn_weights, mask
        )

        fused = torch.cat([global_pooled, hotspot_feat], dim=-1)
        fused = self.hotspot_fusion(fused)
        score = self.proj(fused)

        out = {
            'score': score,
            'attention_weights': attn_weights,
        }

        if return_interpretation:
            attr = ResidueAttributor.compute_attention_attribution(attn_weights, mask)

            sa_sequences = batch_dict.get('sa_sequence', None)
            interpretation = []

            for b in range(B):
                valid = hotspot_info['valid_mask'][b]
                indices = hotspot_info['indices'][b][valid].cpu().tolist()
                h_scores = hotspot_info['scores'][b][valid].cpu().tolist()

                attr_b = attribution[b] if 'attribution' in locals() else attr[b]
                top_attr_values, top_attr_indices = torch.topk(attr_b, k=min(5, mask[b].sum().item()))

                interp = {
                    'hotspot_indices': indices[:5],
                    'hotspot_scores': [round(s, 3) for s in h_scores[:5]],
                    'top_attributed_indices': top_attr_indices.cpu().tolist(),
                    'attribution_values': [round(v.item(), 3) for v in top_attr_values],
                    'score': round(score[b].item(), 3),
                }

                if sa_sequences is not None and len(sa_sequences) > b:
                    seq = sa_sequences[b] if isinstance(sa_sequences, list) else sa_sequences[b]
                    if isinstance(seq, str) and len(seq) > 0:
                        contexts = []
                        for idx in indices[:3]:
                            if isinstance(idx, int) and idx < len(seq):
                                start = max(0, idx - 2)
                                end = min(len(seq), idx + 3)
                                contexts.append(f"{idx}:{seq[start:end]}")
                        interp['hotspot_contexts'] = contexts

                interpretation.append(interp)

            out['interpretation'] = interpretation if B > 1 else interpretation[0]

        return out




ProAgg = ProAggMLPV1


if __name__ == "__main__":
    from src.config.utils import load_yaml_config
    cfg = load_yaml_config('/home/xy_th/Project_protein_aggregation/configs/default.yaml')
    model = ProAgg(cfg)
    print(model)
