# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from model.EncoderVid import EncoderVid
from transformers.activations import gelu
import torch.nn as nn
import numpy as np
import torch
import math
from model.language_model import AModel
import copy
from transformers.modeling_outputs import BaseModelOutput
from transformers import BertConfig
from model.graph import Graph
from util import get_mask
import torch.nn.functional as F
from model.cmatt import CMAtten


def create_sinusoidal_embeddings(n_pos, dim, out):
    with torch.no_grad():
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.num_attention_heads #config.n_heads
        self.dim = config.hidden_size #config.dim
        dp_rate = config.attention_probs_dropout_prob #config.attention_dropout
        self.dropout = nn.Dropout(p=dp_rate)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        dropout, dim, hidden_dim = config.attention_probs_dropout_prob, config.hidden_size, config.intermediate_size
        activation = config.hidden_act

        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        assert activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(activation)
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # dim = config.dim
        dim = config.hidden_size
        # assert config.dim % config.n_heads == 0
        assert dim % config.num_attention_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output
        )  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.n_layers = config.n_layers
        self.n_layers = config.num_hidden_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.n_layers)]
        )

    def forward(
        self,
        x,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
    ):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            if head_mask is not None:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Embeddings(nn.Module):
    def __init__(
        self, d_model, language_len, vision_len, dropout, sinusoidal_pos_embds, d_pos=128
    ):
        super().__init__()
        max_position_embeddings = language_len + vision_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                # out=self.position_embeddings.weight,
                out=self.position_embeddings.weight,
            )
        self.modality_embedding = nn.Embedding(2, d_model)
        self.language_len = language_len
        self.vision_len = vision_len
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, embeddings):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)
        # if self.language_len != 0:
        #     modality_embeddings = self.modality_embedding(
        #         torch.tensor(
        #             [0] * (seq_length-self.vision_len) + [1] * self.vision_len, dtype=torch.long
        #         ).to(embeddings.device)
        #     )
        #     embeddings = (
        #         embeddings + position_embeddings + modality_embeddings
        #     )  # (bs, max_seq_length, dim)
        # else:
        embeddings = embeddings + position_embeddings # (bs, max_seq_length, dim)

        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        
        return embeddings


class POSEmbeddings(nn.Module):
    def __init__(
        self, d_model, max_seq_len, dropout, sinusoidal_pos_embds,d_pos=128
    ):
        super().__init__()
        max_position_embeddings = max_seq_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_pos)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_pos,
                out=self.position_embeddings.weight,
            )
        self.merge_pos = nn.Sequential(
            nn.Linear(d_model+d_pos, d_model),
            nn.ELU(inplace=True))

    def forward(self, embeddings, cid):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  # (max_seq_length)
        position_ids += cid*seq_length
        
        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)
        # print(position_embeddings.shape)
        embeddings = self.merge_pos(torch.cat([embeddings, position_embeddings],dim=-1)) # (bs, max_seq_length, dim)
        
        cpos_embed = position_embeddings.mean(dim=1) #(bs, dim)
        return embeddings, cpos_embed


class VGT(nn.Module):
    def __init__(
        self,
        bert_tokenizer,
        feature_dim=1024,
        word_dim=768,
        N=2,
        h=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        Q=20,
        T=20,
        vocab_size=30522,
        baseline="",
        n_negs=1,
        bnum=10,
        CM_PT=0,
        dataset=""
    ):
        """
        :param feature_dim: dimension of the input video features
        :param word_dim: dimension of the input question features
        :param N: number of transformer layers
        :param h: number of transformer heads
        :param d_model: dimension for the transformer and final embedding
        :param d_ff: hidden dimension in the transformer
        :param dropout: dropout rate in the transformer
        :param Q: maximum number of tokens in the question
        :param T: maximum number of video features
        :param vocab_size: size of the vocabulary for the masked language modeling head
        :param baseline: set as "qa" not to use the video
        :param n_negs: number of negatives sampled for cross-modal matching
        :param bnum: number of selected region proposals
        :param CM_PT: whether use cross-modal petrained weights
        """
        super(VGT, self).__init__()
        self.baseline = baseline
        self.Q = Q
        self.T = T
        self.n_negs = n_negs
        d_pos = 128
        self.CM_PT = CM_PT
        self.task = dataset.split['/'][-1]
        # video modules
        self.encode_vid = EncoderVid(feat_dim=feature_dim, 
                                    bbox_dim=5, 
                                    feat_hidden=d_model, 
                                    pos_hidden=d_pos)

        self.linear_video = nn.Linear(feature_dim, d_model)
        self.norm_video = nn.LayerNorm(d_model, eps=1e-12)

        #####################clip position###################
        # self.position_v = Embeddings(d_model, Q, T//4, dropout, True, d_pos)
        self.position_v = Embeddings(d_model, 20, 8, dropout, True, d_pos)
        #####################hie position###################

        self.config = BertConfig.from_pretrained(
            "bert-base-uncased",
            num_hidden_layers=N,
            hidden_size=d_model,
            attention_probs_dropout_prob=dropout,
            intermediate_size=d_ff,
            num_attention_heads=h,
        )
        self.mmt = Transformer(self.config)
        self.vqproj = nn.Sequential(
                                  nn.Dropout(dropout), 
                                  nn.Linear(d_model, d_model)
                                  )

        # masked language modeling head
        self.vocab_transform = nn.Linear(d_model, d_model)
        self.vocab_norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-12)
        self.vocab_projector = nn.Linear(d_model, vocab_size)
        self.mlm_loss_fct = nn.CrossEntropyLoss()

        # cross-modal matching head
        self.crossmodal_matching = nn.Linear(d_model, 1)
        self.cm_loss_fct = nn.BCELoss()

        # weight initialization
        self.apply(self._init_weights)
        self.answer_embeddings = None

        # answer modules
        self.amodel = AModel(bert_tokenizer, out_dim=d_model)

        self.merge_fr = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ELU(inplace=True))
        
        self.satt_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=-2))

        self.gnn = Graph(dim_in=d_model, dim_hidden=d_model//2,
                         dim_out=d_model, num_layers=2, dropout=dropout)

        config_gt = BertConfig.from_pretrained(
            "bert-base-uncased",
            num_hidden_layers=N,
            hidden_size=bnum*bnum,
            dropout=dropout,
            intermediate_size=d_ff,
            attention_probs_dropout_prob=dropout,
            num_attention_heads=5,
        )
        #The pretrained model on webvid use N=10 region proposals, to adapt to NExT-QA
        #which has different number of regions, please change the layer_name to a new one, e.g., graph_trans_n
        if self.CM_PT and bnum != 10:
            self.graph_trans_n = Transformer(config_gt)
        else:
            self.graph_trans = Transformer(config_gt) 
        
        self.ttrans = Transformer(self.config)
        # # # # ###############cross-mode interaction########### 
        self.bidirec_att = CMAtten()

        # self.final_proj = nn.Linear(d_model, 1) # classification in multi-choice QA
        
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _compute_answer_embedding(self, a2v):
        self.answer_embeddings = self.get_answer_embedding(a2v)

    def get_answer_embedding(self, answer):
        answer_g, answer = self.amodel(answer)
        return answer_g, answer 

    def get_question_embedding(self, question):
        question = self.linear_question(question)
        question = gelu(question)
        question = self.norm_question(question)
        return question
    
    def get_vqa_embedding_rfcpos(self, video, language, language_lens):
        video_o, video_f = video[0], video[1]
        video_f = self.linear_video(video_f)
        video_f = gelu(video_f)
        video_f = self.norm_video(video_f) #(bs, numc, numf, dmodel)
        

        bsize, numc, numf, numr, fdim = video_o.size()
        bsize_lan, len_lan, dim_lan = language.size()
        ans_n = bsize_lan // bsize
       
        X = self.encode_vid(video_o) #(bs, numc*numf, numr, dmodel)

        #######################NTrans ################################
        X = X.view(bsize, numc, numf, numr, -1).permute(0,1,3,2,4)
        short_mask = get_mask(torch.tensor([numf]*bsize*numc*numr, dtype=torch.long), numf).cuda()
        X = self.ttrans(X.reshape(bsize*numc*numr, numf, -1), short_mask)[0]
        X = X.reshape(bsize*numc, numr, numf, -1).permute(0,2,1,3)
        
        ####################################################################
        hd_dim = X.shape[-1]
        X = X.reshape(bsize*numc*numf, numr, hd_dim)       
        A = self.gnn.build_graph(X) #(bs*numc*numf, numr, numr)
        ####################################ETrans################
        A = A.view(bsize*numc, numf, numr*numr)
        graph_mask = get_mask(torch.tensor([numf]*bsize*numc, dtype=torch.long), numf).cuda()
        #The pretrained model on webvid use N=10 region proposals, to adapt to NExT-QA
        #which has different number of regions, please change the layer_name to a new one, e.g., graph_trans_n
        if self.CM_PT and numr != 10:
            A = self.graph_trans_n(x=A, attn_mask=graph_mask)[0]
        else:
            A = self.graph_trans(x=A, attn_mask=graph_mask)[0]
        A = A.view(bsize*numc*numf, numr, numr)
        ###################################################################
        A = F.softmax(A, dim=-1)
        X_o = self.gnn(X, A)
        X_o += X
        
        satt = self.satt_pool(X_o)
        X_o = torch.sum(X_o*satt, dim=-2)

        X_o = X_o.view(bsize, numc, numf, -1)
        
        video = self.merge_fr(torch.cat([video_f, X_o], dim=-1))
       
        #########frame-level cross-model interaction##############
        if self.task in ['action', 'transition']:
            xlen = numc*numf
            video = video.view(bsize, xlen, -1)
            video = self.cm_interaction(video, xlen, language, language_lens, ans_n)
            video = video.view(bsize, numc, numf, -1)
        #########cross-model interaction##############

        video = video.mean(dim=-2)

        #####clip-level cross-model attention#############        
        video = self.cm_interaction(video, numc, language, language_lens, ans_n)
        #####cross-model attention#############
        return video

    def cm_interaction(self, X, xlen, language, language_lens, ans_n=5):
        bsize = X.shape[0]
        X_len = torch.tensor([xlen] * bsize, dtype=torch.long).to(X.device)
        if ans_n == 1:
            Xatt, _ = self.bidirec_att(X, X_len, language, language_lens.view(-1))
        else:
            batch_repeat = np.reshape(np.tile(np.expand_dims(np.arange(bsize),
                                                            axis=1),[1, ans_n]), [-1])
            X = X[batch_repeat]
            X_len = X_len[batch_repeat]
            Xatt, _ = self.bidirec_att(X, X_len, language, language_lens.view(-1))
        Xatt += X
        #max pool over different candicates
        X = Xatt.view(bsize, ans_n, xlen, -1).max(dim=1)[0] 
        return X
    
    def forward(
        self,
        video,
        question=None,
        labels=None,
        answer=None,
        seq_len = None,
        video_mask=None,
        text_mask=None,
        max_seq_len=0,
        mode="vqa",
    ):
        """
        :param video: [bs, T, feature_dim]
        :param question: [bs, Q]
        :param labels: [bs, Q] used for masked language modeling
        :param answer: [batch_size, amax_words, 300] used for contrastive loss training, otherwise precomputed at the vocabulary level
        :param video_mask: [bs, T]
        :param text_mask: [bs, Q]
        """
        if mode == "vqa":
            answer_g, answer_w = (
                self.get_answer_embedding(answer)
                if answer is not None
                else self.answer_embeddings
            )
            if self.Q != 0:
                question_g, question_w = self.amodel(question)
                if question_w.shape[1] < self.Q:
                    question_w = torch.cat(
                        [
                            question_w,
                            torch.zeros(
                                question_w.shape[0],
                                self.Q - question_w.shape[1],
                                question_w.shape[2],
                            ).cuda(),
                        ],
                        1,
                    )
                    text_mask = torch.cat(
                        [
                            text_mask,
                            torch.zeros(
                                text_mask.shape[0], self.Q - text_mask.shape[1]
                            ).cuda(),
                        ],
                        1,
                    )

                video_proj = self.get_vqa_embedding_rfcpos(video, question_w, seq_len)
                video_proj = self.position_v(video_proj)
                attended_qv = self.mmt(x=video_proj, attn_mask=video_mask)[0]
                global_feat = attended_qv.mean(dim=1)
                fusion_proj = self.vqproj(global_feat)
            else:
                video_proj = self.get_vqa_embedding_rfcpos(video, answer_w, seq_len)

                video_proj = self.position_v(video_proj)
                attended_v = self.mmt(x=video_proj, attn_mask=video_mask)[0]
                global_feat = attended_v.mean(dim=1)
                fusion_proj = self.vqproj(global_feat)
                
            if fusion_proj is not None and answer_g.device != fusion_proj.device:
                answer_g = answer_g.to(fusion_proj.device)
            if answer is not None:
                return fusion_proj, answer_g
                # return self.final_proj(fusion_proj*answer_g), answer_g
            else:
                # pred = self.final_proj(fusion_proj*question_g)
                pred = (fusion_proj @ answer_g.t()) * (question_g @ answer_g.t())
                return pred

        elif mode == "mlm":
            if text_mask.shape[1] < max_seq_len:
                text_mask = torch.cat(
                    [
                        text_mask,
                        torch.zeros(
                            text_mask.shape[0], max_seq_len - text_mask.shape[1]
                        ).cuda(),
                    ],
                    1,
                )
                labels = torch.cat(
                    [
                        labels,
                        -100
                        * torch.ones(labels.shape[0], max_seq_len - labels.shape[1])
                        .long()
                        .cuda(),
                    ],
                    1,
                )
            text_g, text = self.amodel(question)
            if text.shape[1] < max_seq_len:
                text = torch.cat(
                    [
                        text,
                        torch.zeros(
                            text.shape[0], max_seq_len - text.shape[1], text.shape[2]
                        ).cuda(),
                    ],
                    1,
                )

            prediction_logits = self.vocab_transform(text)
            prediction_logits = gelu(prediction_logits)
            prediction_logits = self.vocab_norm(prediction_logits)
            prediction_logits = self.vocab_projector(prediction_logits)
            # print(prediction_logits.shape, labels.shape)
            mlm_loss = self.mlm_loss_fct(
                prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1)
            )
            return mlm_loss

