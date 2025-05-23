
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, CLLayer

import torch.nn as nn
import torch.fft as fft
from scipy.fftpack import next_fast_len


class FSM(nn.Module):
    def __init__(self, hidden_size ,layer_norm_eps, dropout,A):
        super().__init__()
        self.layer_norm_eps = layer_norm_eps
        self.hidden_size = hidden_size
        self._smoothing_weight = nn.Parameter(torch.randn(1))
        self.v0 = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.v1 = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.out_dropout = nn.Dropout(dropout)
        self.A=A
        torch.nn.init.trunc_normal_(self.v0, std=.02)
        torch.nn.init.trunc_normal_(self.v1, std=.02)

    def forward(self, x_in):
        B, N, C = x_in.shape
        x = x_in.to(torch.float32)
        powers = torch.arange(N, dtype=torch.float, device=self.v0.device)
        smoothed=torch.sigmoid(self._smoothing_weight)
        weight = (1 - smoothed) * (smoothed ** torch.flip(powers, dims=(0,)))
        init_weight = smoothed ** (powers + 1)
        init_weight=init_weight.view(1, N, 1)
        weight=weight.view(1, N, 1)
        N = x.size(1)
        M = weight.size(1)
        fast_len = next_fast_len(N + M - 1)
        F_f = fft.fft(x, fast_len, dim=1, norm='ortho')
        F_g = fft.fft(weight, fast_len, dim=1, norm='ortho')
        F_fg = F_f * F_g.conj()
        out = fft.ifft(F_fg, fast_len, dim=1, norm='ortho')
        out = out.real
        out = out.roll((-1,), dims=(1,))
        idx = torch.as_tensor(range(fast_len - N, fast_len)).to(out.device)
        output = out.index_select(1, idx)
        output = init_weight * self.v0 + output * self.v1
        hidden_states = self.A * output+(1-self.A)*x_in
        return hidden_states

class ASRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(ASRec, self).__init__(config, dataset)

        self.config = config
        # load parameters info
        self.device = config["device"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        # the dimensionality in feed-forward layer
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']


        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size)  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_seq_length=self.max_seq_length
        )
        self.LayerNorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        self.loss_fct = nn.CrossEntropyLoss()
        self.FSM = FSM(self.hidden_size, self.layer_norm_eps, self.hidden_dropout_prob, config['A'])

        # parameters initialization
        self.apply(self._init_weights)

    def forward(self, item_seq, item_seq_len, return_all=False):
        position_ids = torch.arange(item_seq.size(
            1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        input_emb = self.FSM(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        if return_all:
            return output
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, task_label=False):
        """Generate bidirectional attention mask for multi-head attention."""
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        pos_items = interaction[self.ITEM_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len, return_all=False)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits + 1e-8, pos_items)
        return loss

    def fast_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction["item_id_with_negs"]
        seq_output = self.forward(item_seq, item_seq_len)


        test_item_emb = self.item_embedding(test_item)  # [B, num, H]
        scores = torch.matmul(seq_output.unsqueeze(
            1), test_item_emb.transpose(1, 2)).squeeze()
        return scores


    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len)


        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)).squeeze()
        return scores#, test_items_emb

    def full_sort_predict_valid(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len)


        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)).squeeze()
        return scores, test_items_emb
