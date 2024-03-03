#The model is from https://github.com/fundamentalvision/Uni-Perceiver
import torch
from torch import nn
from model.Base_module import TransformerEncoderLayer, TokenBaseEmbedding, VideoBaseEmbedding
from functools import partial
from timm.models.vision_transformer import Block
import numpy as np
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
import torch.nn.functional as F


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

class MLP(nn.Module):
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
    

class Unified_Model(nn.Module):
    def __init__(self, now_1D_input_size, now_2D_input_size, now_3D_input_size, input_size_3D=(256, 256, 256), input_size_2D=(512, 512), in_chans=1, patch_size=16, embed_dim=768, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_pix_loss=None):
        super(Unified_Model, self).__init__()
        self.num_head = 12
        self.fused_encoder = Encoder()
        self.token_embed = TokenBaseEmbedding(input_size_1D=now_1D_input_size)
        self.video_embed = VideoBaseEmbedding(input_size_3D=input_size_3D, input_size_2D=input_size_2D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.norm_pix_loss = norm_pix_loss
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.patch_size = patch_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        num_patches_2D = (now_2D_input_size[0] // patch_size) *  (now_2D_input_size[1] // patch_size)
        num_patches_3D = (now_3D_input_size[0] // patch_size) *  (now_3D_input_size[1] // patch_size) * (now_3D_input_size[2] // patch_size)

        self.now_input_size_2D = now_2D_input_size
        self.now_input_size_3D = now_3D_input_size

        self.decoder_pos_embed_1D = nn.Embedding(2, embed_dim)
        self.decoder_pos_embed_1D.apply(self.init_weights_embedding)

        self.decoder_pos_embed_2D = nn.Parameter(torch.zeros(1, num_patches_2D + 1, decoder_embed_dim))  # fixed sin-cos embedding
        self.decoder_pos_embed_3D = nn.Parameter(torch.zeros(1, num_patches_3D + 1, decoder_embed_dim))  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_2D = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3, bias=True)  # decoder to patch
        self.decoder_pred_3D = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans, bias=True)  # decoder to patch
        self.mlm_head = MLMHead()


        self.initialize_weights()

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
        torch.nn.init.normal_(self.mask_token, std=.02)

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
            data["data"] = self.token_embed(data["data"])
        else:
            raise NotImplementedError

    def random_masking(self, x, mask_ratio=0.75, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if noise == None: #for continual learning (two paths can use the same noise, i.e., the same masked patches)
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, noise


    def forward_decoder(self, x, modality, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        if modality == "2D image":
            x += self.decoder_pos_embed_2D
        elif modality == "3D image":
            x += self.decoder_pos_embed_3D
        else:
            exit()

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        if modality == "2D image":
            x = self.decoder_pred_2D(x)
        elif modality == "3D image":
            x = self.decoder_pred_3D(x)
        else:
            exit()

        # remove cls token
        x = x[:, 1:, :]

        return x

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

    def forward(self, data, mask_ratio=0.75, feature=False, out_noise=False):

        ori_imgs = data["data"]
        self._tokenize(data)

        noise = None
        if data["modality"] != "text":
            if mask_ratio != 0:
                x, mask, ids_restore, noise = self.random_masking(data["data"], mask_ratio)
            else:
                x = data["data"]
            # append cls token
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            x = self.fused_encoder(x, None)

            if feature:
                return x, noise
            
          
            feature_embed = x

            pred = self.forward_decoder(x, data["modality"], ids_restore)
            loss, (mean, var) = self.forward_loss(ori_imgs, pred, mask, data["modality"])

            if out_noise:
                return (loss, feature_embed, noise), pred, mask, (mean, var)
            else:
                return (loss, feature_embed), pred, mask, (mean, var)
            
        elif data["modality"] == "text":

            mask_attention = data["mask_attention"]
            # print(mask_attention.size(),  ori_imgs.shape)
            mask_attention = self.get_extended_attention_mask(data["mask_attention"], ori_imgs.shape, mask_attention.device).repeat(self.num_head, 1, 1)
            # print(mask_attention.size())
            x = self.fused_encoder(data["data"], mask_attention)

            if feature:
                return x, None
            
          
            feature_embed = x
            x += self.decoder_pos_embed_1D(torch.zeros_like(data["mask_attention"]).long())
            mlm_logits = self.mlm_head(x)
            mlm_labels = data["text_labels"]
            loss = F.cross_entropy(
                mlm_logits.view(-1, 49411),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            if out_noise:
                return (loss, feature_embed, None), None, None, None
            else:
                return (loss, feature_embed), None, None, None


    def forward_loss(self, imgs, pred, mask, modality):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs, modality)
        # print("target", target.size())
        if self.norm_pix_loss:

            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        # print("pred", pred.size(), torch.min(pred), torch.max(pred))
        # print("target", target.size(), torch.min(target), torch.max(target))
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, (mean, var)


    def patchify(self, imgs, modality):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        if modality == "text":
            c = 1
            h = self.now_input_size_1D[0]
            # print(imgs.size(), h)
            x = imgs.reshape(shape=(imgs.shape[0], h, c))

        elif modality == "2D image":
            c = 3
            h, w = self.now_input_size_2D[0] // self.patch_size, self.now_input_size_2D[1] // self.patch_size
            # print(imgs.size(), h, w)
            x = imgs.reshape(shape=(imgs.shape[0], c, h, self.patch_size, w, self.patch_size))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, self.patch_size ** 2 * c))
        elif modality == "3D image":
            c = 1
            d, h, w = self.now_input_size_3D[0] // self.patch_size, self.now_input_size_3D[1] // self.patch_size, self.now_input_size_3D[2] // self.patch_size
            # print(imgs.size(), d, h, w)
            x = imgs.reshape(shape=(imgs.shape[0], c, d, self.patch_size, h, self.patch_size, w, self.patch_size))
            x = torch.einsum('ncdkhpwq->ndhwkpqc', x)
            x = x.reshape(shape=(imgs.shape[0], d * h * w, self.patch_size ** 3 * c))

        return x

    def unpatchify_2D(self, x):

        h, w = self.now_input_size_2D[0] // self.patch_size, self.now_input_size_2D[1] // self.patch_size

        x = x.reshape(shape=(x.shape[0], h, w, self.patch_size, self.patch_size, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * self.patch_size, h * self.patch_size))
        return imgs

    def unpatchify_3D(self, x):

        d, h, w =  self.now_input_size_3D[0] // self.patch_size,  self.now_input_size_3D[1] // self.patch_size,  self.now_input_size_3D[2] // self.patch_size

        x = x.reshape(shape=(x.shape[0], d, h, w, self.patch_size, self.patch_size, self.patch_size, 1))
        x = torch.einsum('ndhwpqkc->ncdphqwk', x)
        imgs = x.reshape(shape=(x.shape[0], 1, d * self.patch_size, h * self.patch_size, w * self.patch_size))
        return imgs


