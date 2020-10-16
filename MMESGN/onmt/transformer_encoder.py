"""Base class for encoders and generic multi encoders."""
import torch
import torch.nn as nn
import onmt
from utils.misc import aeq
from onmt.sublayer import PositionwiseFeedForward
from onmt.fusion import ModalFusion

import pdb
class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerEncoderLayer, self).__init__()

    self.self_attn = onmt.sublayer.MultiHeadedAttention(
        heads, d_model, dropout=dropout)#its self_attention
    
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)#PFF d_model->dff->dmodel use the relu
    
    self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs, mask):

    input_norm = self.att_layer_norm(inputs)
    outputs, _ = self.self_attn(input_norm, input_norm, input_norm,
                                mask=mask)
    inputs = self.dropout(outputs) + inputs#add &norm
    
    input_norm = self.ffn_layer_norm(inputs)
    outputs = self.feed_forward(input_norm)
    inputs = outputs + inputs
    return inputs


class TransformerEncoder(nn.Module):

  def __init__(self, num_layers, d_model, heads, d_ff,
               dropout, embeddings):
    """

    :param num_layers: 6
    :param d_model: 512
    :param heads: 8
    :param d_ff: 2048
    :param dropout: 0.1
    :param embeddings: vocab_size->emb_size(512)
    """
    super(TransformerEncoder, self).__init__()

    self.num_layers = num_layers#6
    self.embeddings = embeddings#vocab_size->emb_size
    self.transformer_text = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])
    self.transformer_visual = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])
    self.transformer_audio = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])

    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    # self.linear_text=nn.Linear(300,512)
    # self.linear_visual=nn.Linear(35,512)
    # self.linear_audio=nn.Linear(74,512)
    self.linear_align = nn.Linear(1536, 512)
    self.fusion = ModalFusion()

  def _check_args(self, src, lengths=None):
    _, n_batch,_ = src.size()
    if lengths is not None:
      n_batch_, = lengths.size()
      aeq(n_batch, n_batch_)

  def forward(self, src, lengths=None):
    """

    :param src: seq_len*bath_size
    :param lengths:batch_size  saving the length of the sequence
    :return:
    """
    """ See :obj:`EncoderBase.forward()`"""
    #25*64*409
    x_l=src[:,:,:300]#300
    x_v=src[:,:,300:335]#35
    x_a=src[:,:,335:409]#74

    x_l = x_l.transpose(0, 1)  # 64*seq_len*emb_size
    x_v = x_v.transpose(0, 1)
    x_a = x_a.transpose(0, 1)

    # src_text=self.linear_text(src_text)
    # src_visual=self.linear_visual(src_visual)
    # src_audio=self.linear_audio(src_audio)
    fus_l,fus_v,fus_a = self.fusion(x_l, x_v, x_a)  # [seq_len,batch_size,emb_size)
    src_text=fus_l
    src_visual=fus_v
    src_audio=fus_a



    # src = self.linear_align(src)
    #(1) preprocess text
    self._check_args(src_text, lengths)
    # emb = self.embeddings(src)
    emb=src_text
    out_text = emb.transpose(0, 1).contiguous()#bath_size*seq_len*emb_size[113*32*512]
    # words = src.transpose(0, 1)#bath_size*seq_len
    # padding_idx = self.embeddings.word_padding_idx#the position of the <bank>
    # mask = torch.zeros(src.size(1),1,src.size(0))#words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]  #in here I dont want to mask
    # Run the forward pass of every layer of the tranformer.
    mask=None
    for i in range(self.num_layers):
      out_text = self.transformer_text[i](out_text, mask)
    out_text = self.layer_norm(out_text)

    #(2)preprocess visual
    self._check_args(src_visual, lengths)
    out_visual = src_visual.transpose(0, 1).contiguous()  # bath_size*seq_len*emb_size[113*32*512]
    mask = None
    for i in range(self.num_layers):
      out_visual = self.transformer_visual[i](out_visual, mask)
    out_visual = self.layer_norm(out_visual)

    #(3) preprocess audio
    self._check_args(src_audio, lengths)
    out_audio = src_audio.transpose(0, 1).contiguous()  # bath_size*seq_len*emb_size[113*32*512]
    mask = None
    for i in range(self.num_layers):
      out_audio = self.transformer_audio[i](out_audio, mask)
    out_audio = self.layer_norm(out_audio)



    # make the three modalities into one src_enc
    #3   64*25*512
    modal_all=torch.cat([out_text,out_visual,out_audio],2)#64*25*1536
    # out=self.linear_align(modal_all)
    #end
    emb=src_text
    return emb, modal_all.transpose(0, 1).contiguous(), lengths

