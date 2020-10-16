""" Embeddings module """
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
  def __init__(self, dropout, dim, max_len=5000):
    pe = torch.zeros(max_len, dim)#[max_len,dim]
    position = torch.arange(0, max_len).unsqueeze(1)#[max_len,1]
    dim_even=(int((dim+1)/2))*2
    if dim_even==dim:
      div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                           -(math.log(10000.0) / dim)))#[dim/2]=256
      pe[:, 0::2] = torch.sin(position.float() * div_term)
      pe[:, 1::2] = torch.cos(position.float() * div_term)
    else:
      div_term = torch.exp((torch.arange(0, dim_even, 2, dtype=torch.float) *
                            -(math.log(10000.0) / dim_even)))  # [dim/2]=256
      pe[:, 0::2] = torch.sin(position.float() * div_term)#5000*205
      pe[:, 1::2] = torch.cos(position.float() * div_term[:-1])#5000*204
    pe = pe.unsqueeze(1)#[max_len,1,dim]
    super(PositionalEncoding, self).__init__()
    self.register_buffer('pe', pe)
    self.dropout = nn.Dropout(p=dropout)
    self.dim = dim

  def forward(self, emb, step=None):
    emb = emb * math.sqrt(self.dim)
    if step is None:
      emb = emb + self.pe[:emb.size(0)]
    else:
      emb = emb + self.pe[step]
    emb = self.dropout(emb)
    return emb


class Embeddings(nn.Module):
  def __init__(self, word_vec_size,
               word_vocab_size,
               word_padding_idx,
               position_encoding=False,
               dropout=0,
               sparse=False):

    self.word_padding_idx = word_padding_idx

    self.word_vec_size = word_vec_size
    
    embedding = nn.Embedding(word_vocab_size, word_vec_size, padding_idx=word_padding_idx, sparse=sparse)

    self.embedding_size = word_vec_size
    
    super(Embeddings, self).__init__()# use the father class initialize the current class
    self.make_embedding = nn.Sequential()
    self.make_embedding.add_module('word', embedding)

    self.position_encoding = position_encoding

    if self.position_encoding:
      pe = PositionalEncoding(dropout, self.embedding_size)
      self.make_embedding.add_module('pe', pe)

  @property
  def word_lut(self):
    """ word look-up table """
    return self.make_embedding[0]

  def load_pretrained_vectors(self, emb_file, fixed):
    """Load in pretrained embeddings.

    Args:
      emb_file (str) : path to torch serialized embeddings
      fixed (bool) : if true, embeddings are not updated
    """
    if emb_file:
      pretrained = torch.load(emb_file)
      pretrained_vec_size = pretrained.size(1)
      if self.word_vec_size > pretrained_vec_size:
        self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
      elif self.word_vec_size < pretrained_vec_size:
        self.word_lut.weight.data \
            .copy_(pretrained[:, :self.word_vec_size])
      else:
        self.word_lut.weight.data.copy_(pretrained)
      if fixed:
        self.word_lut.weight.requires_grad = False

  def forward(self, source, step=None):
    """
    Computes the embeddings for words and features.

    Args:
        source (`LongTensor`): index tensor `[len x batch]`
    Return:
        `FloatTensor`: word embeddings `[len x batch x embedding_size]`
    """
    if self.position_encoding:
      for i, module in enumerate(self.make_embedding._modules.values()):
        if i == len(self.make_embedding._modules.values()) - 1:
          source = module(source, step=step)
        else:
          source = module(source)
    else:
      source = self.make_embedding(source)

    return source


class Embeddings_encoder(nn.Module):
  def __init__(self, word_vec_size,
               word_vocab_size,
               word_padding_idx,
               position_encoding=False,
               dropout=0,
               sparse=False):

    self.word_padding_idx = word_padding_idx

    self.word_vec_size = word_vec_size

    embedding = nn.Embedding(word_vocab_size, word_vec_size, padding_idx=word_padding_idx, sparse=sparse)

    self.embedding_size = word_vec_size

    super(Embeddings_encoder, self).__init__()  # use the father class initialize the current class
    self.make_embedding = nn.Sequential()
    self.make_embedding.add_module('word', embedding)

    self.position_encoding = position_encoding

    if self.position_encoding:
      pe = PositionalEncoding(dropout, self.embedding_size)
      self.make_embedding.add_module('pe', pe)

  @property
  def word_lut(self):
    """ word look-up table """
    return self.make_embedding[0]

  def load_pretrained_vectors(self, emb_file, fixed):
    """Load in pretrained embeddings.

    Args:
      emb_file (str) : path to torch serialized embeddings
      fixed (bool) : if true, embeddings are not updated
    """
    if emb_file:
      pretrained = torch.load(emb_file)
      pretrained_vec_size = pretrained.size(1)
      if self.word_vec_size > pretrained_vec_size:
        self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
      elif self.word_vec_size < pretrained_vec_size:
        self.word_lut.weight.data \
          .copy_(pretrained[:, :self.word_vec_size])
      else:
        self.word_lut.weight.data.copy_(pretrained)
      if fixed:
        self.word_lut.weight.requires_grad = False

  def forward(self, source, step=None):
    """
    Computes the embeddings for words and features.

    Args:
        source (`LongTensor`): index tensor `[len x batch]`
    Return:
        `FloatTensor`: word embeddings `[len x batch x embedding_size]`
    """
    if self.position_encoding:
      # for i, module in enumerate(self.make_embedding._modules.values()):
      #   if i == len(self.make_embedding._modules.values()) - 1:
      #     source = module(source, step=step)
      #   else:
      source = self.make_embedding[1](source,step=step)
    else:
      source = source#self.make_embedding(source)

    return source

