import torch
from torch import nn as nn
from onmt.FusionEncoder import TransformerEncoder as FusionEncoder

class ModalFusion(nn.Module):
    def __init__(self):
        super(ModalFusion,self).__init__()
        # self.proj_l = nn.Conv1d(300, 256, kernel_size=1, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(74, 256, kernel_size=1, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(35, 256, kernel_size=1, padding=0, bias=False)
        self.proj_l = nn.Linear(300,256)
        self.proj_a = nn.Linear(74,256)
        self.proj_v = nn.Linear(35,256)
        self.orig_d_l=300
        self.orig_d_v=74
        self.orig_d_a=35
        self.d_l=256
        self.d_v=256
        self.d_a=256
        self.vonly = True
        self.aonly = True
        self.lonly = True
        self.num_heads = 8
        self.layers = 5
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0.0
        self.attn_dropout_v =0.0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout =0.0
        self.embed_dropout = 0.25
        self.attn_mask = False

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)



    def forward(self,x_l,x_v,x_a):
        """

        :param x_l: batch_size*seq_len*emb_size
        :param x_v:
        :param x_a:
        :return:
        """

        # [batch_size, seq_len,256]
        # Project the textual/visual/audio features

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)  # dim->256
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_a = proj_x_a.transpose(0,1)
        proj_x_v = proj_x_v.transpose(0,1)  # [seq_len,batch_size,256]
        proj_x_l = proj_x_l.transpose(0,1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)

        return h_ls,h_vs,h_as


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return FusionEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

