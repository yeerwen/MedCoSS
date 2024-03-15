import torch
from torch import nn
from model.Base_module import TransformerEncoderLayer, VideoBaseEmbedding
from functools import partial
from timm.models.vision_transformer import Block
import numpy as np
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, auc

class BertPredictionHeadTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.transform_act_fn = nn.GELU()

        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MLMHead(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform()
        self.decoder = nn.Linear(768, 49411, bias=False)
        self.bias = nn.Parameter(torch.zeros(49411))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        layers = []
        dpr = [0.1 for _ in range(12)]
        for layer_idx in range(12):

            layers.append(
                TransformerEncoderLayer(
                    d_model=768,
                    nhead=12,
                    dim_feedforward=3072,
                    dropout=0.,
                    drop_path_ratio=dpr[layer_idx],
                    activation="gelu",
                    layer_scale=True,
                    ls_init_values=1e-3,
                    batch_first=True,
                    norm_first=True,
                ))
        self.layers = nn.ModuleList(
            layers
        )


    def forward(self, data, mask=None):

        for l, layer_module in enumerate(self.layers):
            data = layer_module(src=data, src_mask=mask)
        return data

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Unified_Model(nn.Module):
    def __init__(self, now_2D_input_size, input_size_3D=(256, 256, 256), input_size_2D=(512, 512), input_size_1D=(1024, 1), patch_size=16, global_pooling=True, num_classes=3, pre_trained=False, pre_trained_weight=None, local_trained_weight=None):
        super(Unified_Model, self).__init__()
        self.num_head = 12
        self.fused_encoder = Encoder()
        self.video_embed = VideoBaseEmbedding(input_size_3D=input_size_3D, input_size_2D=input_size_2D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.patch_size = patch_size

        self.input_size_2D = now_2D_input_size
        self.global_pooling = global_pooling
        self.fc_norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, num_classes) if num_classes > 0 else nn.Identity()
        self.cal_acc = False
        self.initialize_weights()

        if pre_trained:
            print("load parameters from ", pre_trained_weight)
            model_dict = self.state_dict()
            pre_dict = torch.load(pre_trained_weight, map_location='cpu')["model"]
            pre_dict_update = {k: v for k, v in pre_dict.items() if k in model_dict}

            pre_dict_no_update = [k for k in pre_dict.keys() if k not in model_dict]
            print("no update: ", pre_dict_no_update)
            print("[pre_%d/mod_%d]: %d shared layers" % (len(pre_dict), len(model_dict), len(pre_dict_update)))
            model_dict.update(pre_dict_update)
            self.load_state_dict(model_dict)

    def init_weights_embedding(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _tokenize(self, data):
        # toknizer
        if data['modality'] in ['2D image'] or data['modality'] in ['3D image']:
            data['data'] = self.video_embed(data["data"])
        elif data['modality'] == 'text':
            data['data'] = self.token_embed(data["data"])
        else:
            raise NotImplementedError


    def forward(self, data):
        self._tokenize(data)
      
        if data["modality"] != "text":
            # append cls token
            x = data["data"]
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            x = self.fused_encoder(x, None)

            if self.global_pooling:
                x = x[:,1:, :].mean(dim=1)
                x = self.fc_norm(x)
                cls_logits = self.head(x)
                cls_loss = F.cross_entropy(cls_logits, data["labels"])

                if self.cal_acc:
                    cls_logits_argmax = torch.argmax(cls_logits, dim=1)
                    sklearn_accuracy = accuracy_score(data["labels"].cpu().numpy(),
                                                      cls_logits_argmax.cpu().numpy())
                    return sklearn_accuracy, F.softmax(cls_logits, dim=1)

                return cls_loss

            else:
                pass

        else:
           pass
