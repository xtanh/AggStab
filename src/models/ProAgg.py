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


ProAgg = ProAggMLPV1


if __name__ == "__main__":
    from src.config.utils import load_yaml_config
    cfg = load_yaml_config('/home/xy_th/Project_protein_aggregation/configs/default.yaml')
    model = ProAgg(cfg)
    print(model)
