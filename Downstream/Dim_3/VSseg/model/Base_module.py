import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from torch import Tensor
import math
from timm.models.layers import DropPath
try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)
except ImportError:
    has_fused_layernorm = False

class DeepPrompt(torch.nn.Module):
    # naive implementation
    def __init__(self, cfg):
        super().__init__()

        embedding_hidden_size = cfg.MODEL.BERT.HIDDEN_SIZE
        self.target_prompt = cfg.MODEL.PROMPT_EMBED.TARGET_DEEP_PROMPT and not cfg.MODEL.PROMPT_EMBED.SHARE_DEEP_PROMPT
        self.embedding = torch.nn.Embedding(cfg.MODEL.PROMPT_EMBED.INPUT_DEEP_PROMPT_LENGTH, embedding_hidden_size)
        if self.target_prompt:
            self.target_embedding = torch.nn.Embedding(cfg.MODEL.PROMPT_EMBED.TARGET_DEEP_PROMPT_LENGTH, embedding_hidden_size)


    def forward(self, x, batch_first=False, data_type=None, **kwargs):
        # x: length, bs, hidden_size

        if data_type == 'target' and self.target_prompt:
            embddings = self.target_embedding.weight
        else:
            embddings = self.embedding.weight

        if batch_first:
            bs = x.shape[0]
            embddings = embddings.unsqueeze(0).expand(bs, -1, -1)
        else:
            bs = x.shape[1]
            embddings = embddings.unsqueeze(1).expand(-1,bs, -1)
        return embddings

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    else:
        return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

_ACT_LAYER_DEFAULT = dict(
    relu=nn.ReLU,
    elu=nn.ELU,
    celu=nn.CELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
)

def get_act_layer(name='none'):
    if name in _ACT_LAYER_DEFAULT:
        return _ACT_LAYER_DEFAULT[name]
    else:
        return None

def swish(x):
    return x * torch.sigmoid(x)

def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

gelu = getattr(F, "gelu", _gelu_python)

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": F.tanh,
    "gelu_new": gelu_new,
    "mish": mish
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            "function {} not found in ACT2FN mapping {} or torch.nn.functional".format(
                activation_string, list(ACT2FN.keys())
            )
        )

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
    #     >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    #     >>> src = torch.rand(10, 32, 512)
    #     >>> out = encoder_layer(src)
    # Alternatively, when ``batch_first`` is ``True``:
    #     >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
    #     >>> src = torch.rand(32, 10, 512)
    #     >>> out = encoder_layer(src)
    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:
        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - norm_first is ``False`` (this restriction may be loosened in the future)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)
        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first'] # we inherit this variable from pytorch's code for jit

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, drop_path_ratio: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_scale: bool = False, ls_init_values: float = 1e-3,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, cfg: dict = None) -> None:
        #
        factory_kwargs = {}
        super(TransformerEncoderLayer, self).__init__()


        # The interface of nn.MultiheadAttention changed since torch 1.9.0.
        _torch_version_main = torch.__version__.split('.')[:2]
        if (int(_torch_version_main[0]) >= 1) and (int(_torch_version_main[1])) >= 9:
            self._torch_nn_new_interface = True
        else:
            self._torch_nn_new_interface = False

        if self._torch_nn_new_interface:
            factory_kwargs = {'device': device, 'dtype': dtype}
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        else:
            factory_kwargs = {}
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                **factory_kwargs)

        self.batch_first = batch_first

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        # if self.cfg.SOLVER.FUSED_LAYERNORM:
        #     self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        #     self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        # else:
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path1 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma_1 = nn.Parameter(ls_init_values * torch.ones((d_model)),requires_grad=True)
            self.gamma_2 = nn.Parameter(ls_init_values * torch.ones((d_model)),requires_grad=True)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = get_activation(activation)

        self.activation = activation
        self.deep_prompt = None


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if self.batch_first and not self._torch_nn_new_interface:
            x = src.transpose(0,1)
        else:
            x = src

        if self.norm_first:
            history_states_norm = None
            x = x + self._sa_block(self.norm1(x), src_mask, None, history_states=history_states_norm)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, None, history_states=None))
            x = self.norm2(x + self._ff_block(x))

        if self.batch_first and not self._torch_nn_new_interface:
            x = x.transpose(0, 1)

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], history_states: Optional[Tensor],
                  **kwargs) -> Tensor:

        if history_states is not None:
            kv = torch.cat(
                [history_states, x],
                dim=1 if (self.batch_first and self._torch_nn_new_interface) else 0
            )
            # TODO: changes for attn_mask and key_padding_mask
        else:
            kv = x

        if self.deep_prompt:

            deep_prompt_embedding = self.deep_prompt_embedding(x, batch_first=(self.batch_first and self._torch_nn_new_interface), **kwargs)
            if self.norm_first:
                deep_prompt_embedding = self.norm1(deep_prompt_embedding)
            kv = torch.cat([deep_prompt_embedding, kv], dim=1 if (self.batch_first and self._torch_nn_new_interface) else 0)
            if attn_mask is not None:
                L, S = attn_mask.shape
                pe_length = deep_prompt_embedding.shape[1 if
                                                        (self.batch_first and self._torch_nn_new_interface) else 0]  # length, bs, hidden_size
                attn_mask = torch.cat([torch.zeros((L, pe_length), dtype=attn_mask.dtype, device=attn_mask.device), attn_mask], dim=1)
            if key_padding_mask is not None:
                if self.batch_first and self._torch_nn_new_interface:
                    bs, pe_length = deep_prompt_embedding.shape[:2]
                else:
                    pe_length, bs = deep_prompt_embedding.shape[:2]
                key_padding_mask = torch.cat(
                    [torch.zeros((bs, pe_length), dtype=key_padding_mask.dtype, device=key_padding_mask.device), key_padding_mask], dim=1)

        # print(x.size(), kv.size(), kv.size(), attn_mask.size(), key_padding_mask,False)
        x = self.self_attn(x, kv, kv,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        x = self.drop_path1(self.dropout1(x))
        if self.layer_scale:
            x = self.gamma_1 * x
        return x


    # feed forward block
    def _ff_block(self, x: Tensor, **kwargs) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.drop_path2(self.dropout2(x))
        if self.layer_scale:
            x = self.gamma_2 * x
        return x


import torch
from torch import nn
import copy
import math
import torch
from torch import nn


def build_position_encoding(dim, max_len):
    return NNEmbeddingEncoding(dim, max_len)


class SinusoidEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(SinusoidEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() *
                             -(math.log(max_len * 2.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if isinstance(x, int):
            return self.pe[:, x]
        else:
            x_size = x.size(1)
            return self.pe[:, :x_size]


class NNEmbeddingEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(NNEmbeddingEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)

    def forward(self, x, start_time=0):
        if isinstance(x, int):
            position_embeddings = self.position_embeddings(torch.tensor([x], dtype=torch.long).cuda())
        elif isinstance(x, torch.Tensor) and x.dim()==1:
            position_embeddings = self.position_embeddings(x)
        else:
            x_size = x.size(1)
            position_ids = torch.arange(x_size, dtype=torch.long, device=x.device) + start_time
            position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

class TokenBaseEmbedding(nn.Module):

    def __init__(
        self,
        dim=768,
        vocab_size=49411, # include <BOS>/<EOS>
        **kwargs
    ):
        super(TokenBaseEmbedding, self).__init__()
        kwargs = {
            "dim": 768,
            "vocab_size": 49411
        }

        activation_name = ('none').lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        embeddings_norm = nn.LayerNorm(768)
        kwargs['embeddings_norm'] = embeddings_norm

        embeddings_pos = build_position_encoding(768, 512)
        kwargs['embeddings_pos'] = embeddings_pos

        embeddings_token_type = nn.Embedding(2, 768)
        kwargs['embeddings_token_type'] = embeddings_token_type

        self.embeddings = nn.Embedding(vocab_size, dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop("embeddings_pos", None)
        self.embeddings_token_type = kwargs.pop('embeddings_token_type', None)
        self.embeddings_token_seg = kwargs.pop('embeddings_token_seg', None)
        self.bw_own_embed = kwargs.pop('bw_own_embed', False)
        self.pos_before = kwargs.pop('pos_before', True)



        if self.bw_own_embed:
            # only for debugging
            self.bw_embeddings = copy.deepcopy(self.embeddings)
            self.bw_embeddings_norm = copy.deepcopy(self.embeddings_norm)
            self.bw_embeddings_pos = copy.deepcopy(self.embeddings_pos)
            self.bw_embeddings_token_type = copy.deepcopy(self.embeddings_token_type)
        self.s_token_bias = None



    def set_s_token_bias(self, s_token_bias):
        self.s_token_bias = s_token_bias

    def forward(self, input_ids):

        embeddings = self.embeddings(input_ids)


        if self.s_token_bias is not None:
            # learnable
            embeddings[input_ids == 49410] = embeddings[input_ids == 49410] + self.s_token_bias

        if self.embeddings_pos is not None:
            pos_inputs = input_ids
            position_embeddings = self.embeddings_pos(pos_inputs)
            embeddings = embeddings + position_embeddings.to(embeddings.dtype)

        if self.embeddings_token_type is not None:

            embeddings_token_type = self.embeddings_token_type.weight[0].unsqueeze(0).unsqueeze(1)
            embeddings = embeddings + embeddings_token_type.to(embeddings.dtype)

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_pos is not None and not self.pos_before:
            pos_inputs = input_ids
            position_embeddings = self.embeddings_pos(pos_inputs)
            embeddings = embeddings + position_embeddings.to(embeddings.dtype)
        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)
        return embeddings

import torch
from torch import nn
from einops import rearrange, repeat

class VideoBaseEmbedding(nn.Module):

    def __init__(
        self,
        in_dim=768,
        out_dim=768,
        patch_size=16,
        input_size_3D=None,
        input_size_2D=None,
        max_spatial_size = 196,

    ):
        super(VideoBaseEmbedding, self).__init__()
        kwargs = {
            "in_dim": 768,
            "out_dim": 768,
            "patch_size": 16,
            "time_span": 1,
            "max_time_len": 8,
        }
        max_spatial_size = (input_size_3D[0] // patch_size) *  (input_size_3D[1] // patch_size) * (input_size_3D[2] // patch_size)
        max_spatial_size_2D =(input_size_2D[0] // patch_size) *  (input_size_2D[1] // patch_size)
        kwargs['max_spatial_size'] = max_spatial_size
        # activation_name = ('none').lower()


        embeddings_norm = nn.LayerNorm(768)
        self.embeddings_norm = embeddings_norm


        kwargs['embeddings_pos'] = "divide_st_pos"

        self.embeddings = nn.Linear(in_dim, out_dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)

        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)

        self.random_temporal_pos = kwargs.pop("random_temporal_pos", True)
        self.patch_size = patch_size
        self.time_span =  kwargs.pop("time_span", None)
        self.pos_before = kwargs.pop('pos_before', True)

        self.embeddings_st_pos = None
        self.max_spatial_size = max_spatial_size

        self.embeddings_st_pos_2D = Divide_ST_POS(
            max_spatial_size_2D, 8, out_dim,
            self.random_temporal_pos)
        self.embeddings_st_pos_3D = Divide_ST_POS(
            max_spatial_size, 8, out_dim,
            self.random_temporal_pos)
        # self.embeddings_pos = None
        del self.embeddings

        self.embeddings = nn.Conv2d(3, out_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.embeddings_3D = nn.Conv3d(1, out_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # print("video embedding", self.pos_before)

    def forward(self, data):

        if data.dim() == 4:
            bs = data.size(0)
            # print(data.size())
            x = self.embeddings(data) # b*t, dim, 14, 14
            x = x.flatten(2) # .flatten(2)
            # print(x.size())
            embeddings = rearrange(x, '(b t s) c hw -> b t hw (s c)', b=bs,  s = self.time_span)
            # print(embeddings.size())
            embeddings_pos = self.embeddings_st_pos_2D(embeddings).unsqueeze(
                0).flatten(1, 2)
            embeddings = embeddings.flatten(1, 2)
            if self.pos_before:
                embeddings = embeddings + embeddings_pos.to(embeddings.dtype)

        elif data.dim() == 5:
            bs = data.size(0)
            # print(data.size())
            x = self.embeddings_3D(data)  # b*t, dim, 14, 14
            x = x.flatten(2)  # .flatten(2)
            # print(x.size())
            embeddings = rearrange(x, '(b t s) c dhw -> b t dhw (s c)', b=bs, s=self.time_span)
            # print(embeddings.size())
            embeddings_pos = self.embeddings_st_pos_3D(embeddings).unsqueeze(
                0).flatten(1, 2)
            embeddings = embeddings.flatten(1, 2)
            if self.pos_before:
                embeddings = embeddings + embeddings_pos.to(embeddings.dtype)

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if not self.pos_before:
            embeddings = embeddings + embeddings_pos

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        return embeddings



class Divide_ST_POS(nn.Module):
    def __init__(self, num_patches, max_time_len, out_dim,
                 random_temporal_pos):
        super(Divide_ST_POS, self).__init__()
        self.spatial_pos_embed = nn.Embedding(num_patches, out_dim)
        self.temporal_pos_embed = nn.Embedding(max_time_len, out_dim)
        self.spatial_pos_embed_index = 0 # sometimes image has cls_token
        self.max_frames = max_time_len
        self.random_temporal_pos = random_temporal_pos

    def forward(self, x):
        dtype = x.dtype
        temp_len, spatial_size = x.size(1), x.size(2)

        if self.training and self.random_temporal_pos:
            temporal_pos_ids = torch.arange(temp_len, dtype=torch.long, device=x.device) + \
                torch.randint(0, self.max_frames - temp_len + 1, size=(1,), dtype=torch.long, device=x.device)
        else:
            temporal_pos_ids = torch.arange(temp_len, dtype=torch.long, device=x.device)
        # print(temporal_pos_ids.size(), self.temporal_pos_embed, self.spatial_pos_embed)
        pos_embed = self.temporal_pos_embed(temporal_pos_ids).unsqueeze(1).to(dtype=dtype) + \
            self.spatial_pos_embed(torch.arange(start= self.spatial_pos_embed_index, end=spatial_size +  self.spatial_pos_embed_index , dtype=torch.long, device=x.device)).unsqueeze(0).to(dtype=dtype)
        return pos_embed