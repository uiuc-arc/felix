import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)


class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        out = x + self.pos_embedding
        return self.dropout(out)


class MlpBlock(nn.Module):
    """Transformer Feed-Forward Block"""

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearIn(nn.Module):
    def __init__(self, in_dim=768, feat_dim=(12, 64)):
        super().__init__()
        self.weight = Parameter(torch.randn(in_dim, *feat_dim))
        self.bias = Parameter(torch.zeros(*feat_dim))

    def forward(self, x):
        # (x) [n, m, 768] * (weight) [768, 12, 64] -> [n, m, 12, 64]
        d0, d1, d2 = x.shape
        sd0, sd1, sd2 = self.weight.shape
        # assert d2 == sd0
        out = x @ self.weight.flatten(1, 2)
        out = out.reshape(d0, d1, sd1, sd2) + self.bias
        return out.permute(0, 2, 1, 3)


class LinearOut(nn.Module):
    def __init__(self, feat_dim=(12, 64), in_dim=768):
        super().__init__()
        self.weight = Parameter(torch.randn(*feat_dim, in_dim))
        self.bias = Parameter(torch.zeros(in_dim))

    def forward(self, x):
        # (x) [n, m, 12, 64] * (weight) [12, 64, k] -> [n, m, k]
        # d0, d1, d2, d3 = x.shape
        # sd0, sd1, sd2 = self.weight.shape
        # assert (d2, d3) == (sd0, sd1)
        out = x.flatten(2, 3) @ self.weight.flatten(0, 1)
        return out + self.bias


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim**0.5

        self.query = LinearIn(in_dim, (self.heads, self.head_dim))
        self.key = LinearIn(in_dim, (self.heads, self.head_dim))
        self.value = LinearIn(in_dim, (self.heads, self.head_dim))
        self.out = LinearOut((self.heads, self.head_dim), in_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = self.out(out.permute(0, 2, 1, 3))
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = CustomLayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.norm2 = CustomLayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        num_patches,
        emb_dim,
        mlp_dim,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
    ):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = CustomLayerNorm(in_dim)

    def forward(self, x):
        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        image_size=(256, 256),
        patch_size=(16, 16),
        emb_dim=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        num_classes=1000,
        attn_dropout_rate=0.0,
        dropout_rate=0.1,
    ):
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        self.cls_token = Parameter(torch.zeros(1, 1, emb_dim))

        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )

        # classfier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb)

        # classifier
        logits = self.classifier(feat[:, 0])
        return logits


class CustomLayerNorm(nn.LayerNorm):
    def forward(self, x):
        mean = torch.mean(x, -1, keepdim=True)
        variance = torch.var(x, -1, unbiased=False, keepdim=True)
        denom = torch.sqrt(variance + self.eps)
        normalized = (x - mean) / denom
        return normalized * self.weight + self.bias


# ViT-B/16 configuration
b16_config = {
    "patch_size": 16,
    "emb_dim": 768,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "attn_dropout_rate": 0.0,
    "dropout_rate": 0,  # For inference.
}
# ViT-B/32 configuration
b32_config = {**b16_config, "patch_size": 32}
# ViT-L/16 configuration
l16_config = {
    "patch_size": 16,
    "emb_dim": 1024,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "attn_dropout_rate": 0.0,
    "dropout_rate": 0.1,
}
# ViT-L/32 configuration
l32_config = {**l16_config, "patch_size": 32}
# ViT-H/14 configuration"""
h14_config = {
    "patch_size": 14,
    "emb_dim": 1280,
    "mlp_dim": 5120,
    "num_heads": 16,
    "num_layers": 32,
    "attn_dropout_rate": 0.0,
    "dropout_rate": 0.1,
}
CONFIGS = {
    "b16": b16_config,
    "b32": b32_config,
    "l16": l16_config,
    "l32": l32_config,
    "h14": h14_config,
}


def get_eval_config(custom_kwargs):
    from argparse import Namespace

    return Namespace(**b32_config, num_classes=1000, image_size=224, **custom_kwargs)


def vit(**vit_kwargs):
    config = get_eval_config(vit_kwargs)
    print_config(config)
    vit = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate,
    )
    return vit, (3, config.image_size, config.image_size)


def print_config(config):
    message = "\n"
    message += "----------------- Config ---------------\n"
    for k, v in sorted(vars(config).items()):
        comment = ""
        message += "{:>35}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    logger.info(message)
