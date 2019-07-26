from torch import nn
import torch
import torch.nn.functional as F
class Attention_one_step(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention_one_step, self).__init__()
        # attention
        self.W_c = nn.Linear(1, hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_feature = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, key, key_padding_mask):
        query = query.transpose(dim0 = 0,dim1 =1)
        key = key.transpose(dim0 = 0,dim1 =1)
        # query (B, 1, hidden_dim)
        # K = V
        # 需要注意的是trg_dim和enc_dim是相等的，所以这里不需要关注它俩的区别
        # t_k就是src_len, n是hidden_dim
        b, src_len, n = list(key.size())
        # B * q_len* src_len x hidden_dim
        key_feature = key.contiguous().view(-1, n)
        key_feature = self.key_feature(key_feature)
        query_fea = self.query_proj(query) # B x q_len x hidden_dim
        query_fea_expanded = query_fea.expand(b, src_len, n).contiguous() # B x t_k x hidden_dim
        query_fea_expanded = query_fea_expanded.view(-1, n)  # B * t_k x hidden_dim
        att_features = key_feature + query_fea_expanded # B * t_k x hidden_dim
        e = torch.tanh(att_features) # B * t_k x hidden_dim
        scores = self.v(e)  # B *t_k x 1
        scores = scores.view(-1, src_len)  # B x t_k
        # key_padding_mask = key_padding_mask.unsqueeze(1).expand(b, q_len, src_len).contigunous()

        attn_dist_ = F.softmax(scores, dim=1)
        # print("attn所需要的shape是：",attn_dist_1.shape)
        # attn_dist_ = attn_dist_1 * key_padding_mask # B x src_len
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor  # B x src_len

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x src_len
        c_t = torch.bmm(attn_dist, key)  # B x 1 x n
        return c_t, attn_dist