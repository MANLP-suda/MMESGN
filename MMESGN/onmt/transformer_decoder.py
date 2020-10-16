"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

import onmt
from onmt.sublayer import PositionwiseFeedForward
from onmt.gate import GateAttention

import copy
import pdb
MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerDecoderLayer, self).__init__()

    self.self_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)

    self.context_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
    
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    self.self_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.enc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    self.dropout = dropout
    self.drop = nn.Dropout(dropout)
    mask = self._get_attn_subsequent_mask(MAX_SIZE)
    # Register self.mask as a buffer in TransformerDecoderLayer, so
    # it gets TransformerDecoderLayer's cuda behavior automatically.
    self.register_buffer('mask', mask)

  def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
              layer_cache=None, step=None):
    """

    :param inputs:[b,l,e] l=36
    :param memory_bank:[b,l,e] l=32
    :param src_pad_mask:[b,1=1,l=len]
    :param tgt_pad_mask:[b,l=1,l=len=36]
    :param layer_cache:none
    :param step:none
    :param self.mask:1*5000*5000
    :return:
    """
    dec_mask = None

    if step is None:#gt if same False else True
      dec_mask = torch.gt(tgt_pad_mask +
                          self.mask[:, :tgt_pad_mask.size(-1),
                                    :tgt_pad_mask.size(-1)], 0)
        #get the dec_mask:[b,len,len] len =36

    # do self attention
    input_norm = self.self_att_layer_norm(inputs)
    query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=dec_mask,
                                 layer_cache=layer_cache,
                                 type="self")
    query = self.drop(query) + inputs

    # do encoding output attention
    query_norm = self.enc_att_layer_norm(query)
    mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                  mask=src_pad_mask,
                                  layer_cache=layer_cache,
                                  type="context")
    mid = self.drop(mid) + query
    
    # do ffn
    mid_norm = self.ffn_layer_norm(mid)
    output = self.feed_forward(mid_norm)
    output = self.drop(output) + mid

    return output, attn

  def _get_attn_subsequent_mask(self, size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')#its meas a tri angle k=1 meas the last 1 line has no 1
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class TransformerDecoder(nn.Module):
  def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
    """
    :param num_layers:6
    :param d_model:512
    :param heads:8
    :param d_ff:2048
    :param dropout:0.1
    :param embeddings: one class  a module
    """
    super(TransformerDecoder, self).__init__()

    # Basic attributes.
    self.decoder_type = 'transformer'
    self.num_layers = num_layers
    self.embeddings = embeddings

    # Decoder State
    self.state = {}

    # Build TransformerDecoder.
    self.transformer_layers_l = nn.ModuleList(
      [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])

    # Build TransformerDecoder.
    self.transformer_layers_v= nn.ModuleList(
      [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])

    # Build TransformerDecoder.
    self.transformer_layers_a = nn.ModuleList(
      [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])

    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    self.gate=GateAttention(512)

  def init_state(self, src, src_enc):
    """ Init decoder state """
    self.state["src"] = src
    self.state["src_enc"] = src_enc
    self.state["cache_l"] = None
    self.state["cache_v"] = None
    self.state["cache_a"] = None

  def map_state(self, fn):
    def _recursive_map(struct, batch_dim=0):
      for k, v in struct.items():
        if v is not None:
          if isinstance(v, dict):
            _recursive_map(v)
          else:
            struct[k] = fn(v, batch_dim)

    self.state["src"] = fn(self.state["src"], 1)  # 30*125*409
    self.state["src_enc"] = fn(self.state["src_enc"], 1)  # 25*30*512->25*150*512
    if self.state["cache_l"] is not None:
      _recursive_map(self.state["cache_l"])
    if self.state["cache_v"] is not None:
      _recursive_map(self.state["cache_v"])
    if self.state["cache_a"] is not None:
      _recursive_map(self.state["cache_a"])
      

  def detach_state(self):
    self.state["src"] = self.state["src"].detach()

  def forward(self, tgt, step=None):
    """

    :param tgt: l*b
    :param step:
    :return:
    """
    """
    See :obj:`onmt.modules.RNNDecoderBase.forward()`
    """
    if step == 0:
      self._init_cache(self.num_layers)

    src = self.state["src"]  # [l,b]
    memory_bank = self.state["src_enc"]  # [l,b,e]
    src_words = src.transpose(0, 1)  # [b,l]
    tgt_words = tgt.transpose(0, 1)  # [b,l]  l is not equal the l before

    # Initialize return variables.
    attns = {"std": []}

    # Run the forward pass of the TransformerDecoder.
    emb = self.embeddings(tgt, step=step)  # [l,b,e]
    assert emb.dim() == 3  # len x batch x embedding_dim

    output = emb.transpose(0, 1).contiguous()  # [b,l,e]l=36
    src_memory_bank = memory_bank.transpose(0, 1).contiguous()  # [b,l,e]l=32
    src_memory_bank_l=src_memory_bank[:,:,:512]
    src_memory_bank_v=src_memory_bank[:,:,512:1024]
    src_memory_bank_a=src_memory_bank[:,:,1024:1536]

    pad_idx = self.embeddings.word_padding_idx
    
    

    # src_pad_mask = src_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_src]
    src_pad_mask = None
    tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

    #you need to check the output is changed?
    output_l=output
    output_v=output
    output_a=output
    for i in range(self.num_layers):
      output_l, attn = self.transformer_layers_l[i](
        output_l,
        src_memory_bank_l,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=(
          self.state["cache_l"]["layer_{}".format(i)]
          if step is not None else None),
        step=step)

    output_l = self.layer_norm(output_l)

    # Process the result and update the attentions.
    dec_outs_l = output_l.transpose(0, 1).contiguous()
    
    attns["std"] = attn
    for i in range(self.num_layers):
      output_v, attn = self.transformer_layers_v[i](
        output_v,
        src_memory_bank_v,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=(
          self.state["cache_v"]["layer_{}".format(i)]
          if step is not None else None),
        step=step)

    output_v = self.layer_norm(output_v)

    # Process the result and update the attentions.
    dec_outs_v = output_v.transpose(0, 1).contiguous()

    for i in range(self.num_layers):
      output_a, attn = self.transformer_layers_a[i](
        output_a,
        src_memory_bank_a,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=(
          self.state["cache_a"]["layer_{}".format(i)]
          if step is not None else None),
        step=step)

    output_a = self.layer_norm(output_a)

    # Process the result and update the attentions.
    dec_outs_a = output_a.transpose(0, 1).contiguous()
    attn = attn.transpose(0, 1).contiguous()

    dec_outs_l=dec_outs_l.unsqueeze(2)
    dec_outs_v=dec_outs_v.unsqueeze(2)
    dec_outs_a=dec_outs_a.unsqueeze(2)

    dec_outs_all=torch.cat([dec_outs_l,dec_outs_v,dec_outs_a],2)  #[tgt_len,batch_size,seq_len,emb_dim]
    #gate
    tgt_len,batch_size,seq_len,emb_dim=dec_outs_all.size()
    dec_outs_all_shape=torch.reshape(dec_outs_all,(-1,seq_len,emb_dim))
    dec_outs,_=self.gate(dec_outs_all_shape.transpose(0,1))
    dec_outs=torch.reshape(dec_outs, (tgt_len,batch_size,-1))

    
    # TODO change the way attns is returned dict => list or tuple (onnx)
    return dec_outs, attns

  def _init_cache(self, num_layers):
    self.state["cache_l"] = {}
    self.state["cache_v"] = {}
    self.state["cache_a"] = {}

    for l in range(num_layers):
      layer_cache = {
        "memory_keys": None,
        "memory_values": None
      }
      layer_cache["self_keys"] = None
      layer_cache["self_values"] = None
      self.state["cache_l"]["layer_{}".format(l)] = copy.deepcopy(layer_cache)
      self.state["cache_v"]["layer_{}".format(l)] =  copy.deepcopy(layer_cache)
      self.state["cache_a"]["layer_{}".format(l)] =  copy.deepcopy(layer_cache)


