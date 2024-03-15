import torch
from torch import nn
from model.Base_module import TransformerEncoderLayer, TokenBaseEmbedding
import numpy as np
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

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
    def __init__(self, num_classes, input_size_1D=512, patch_size=16, embed_dim=768, pre_trained=False, pre_trained_weight=None, model_name=None):
        super(Unified_Model, self).__init__()
        self.num_head = 12
        self.fused_encoder = Encoder()
        self.token_embed = TokenBaseEmbedding(input_size_1D=input_size_1D)

        self.patch_size = patch_size
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

        self.cal_acc = False
        self.initialize_weights()

        if pre_trained:
            print("load parameters from ", pre_trained_weight, "model_name", model_name)
            model_dict = self.state_dict()
            pre_dict = torch.load(pre_trained_weight, map_location='cpu')[model_name]
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


    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]

            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)


            extended_attention_mask = causal_mask[:, :, :] * attention_mask[:, None, :]

        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, data):
        ori_imgs = data["data"]
        self._tokenize(data)

        mask_attention = data["mask_attention"]
        mask_attention = self.get_extended_attention_mask(data["mask_attention"], ori_imgs.shape, mask_attention.device).repeat(self.num_head, 1, 1)
        x = self.fused_encoder(data["data"], mask=mask_attention)

        x = x.mean(dim=1)  # global pooling without cls token
        cls_logits = self.cls_head(x)
        cls_loss = F.cross_entropy(cls_logits, data["text_labels"])

        if self.cal_acc:
            cls_logits_argmax = torch.argmax(cls_logits, dim=1)
            sklearn_accuracy = accuracy_score(data["text_labels"].cpu().numpy(),
                                              cls_logits_argmax.cpu().numpy())
            return sklearn_accuracy, F.softmax(cls_logits, dim=1)
        return cls_loss



