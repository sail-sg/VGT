import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.activations import gelu

class Bert(nn.Module):
    """ Finetuned *BERT module """

    def __init__(self, bert_tokenizer):
        super(Bert, self).__init__()
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        
        self.bert.resize_token_embeddings(len(bert_tokenizer))
        
        # You can uncomment this to freeze the language model for the 2nd-stage finetuning
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False
        
        
    def forward(self, tokens):
        attention_mask = (tokens > 0).float()
        outs = self.bert(tokens, attention_mask=attention_mask)
        embds = outs[0]
        
        return embds, outs[1][-2]


class Sentence_Maxpool(nn.Module):
    """ Utilitary for the answer module """

    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x_in):
        x = self.fc(x_in)
        x = torch.max(x, dim=1)[0]
        if self.relu:
            x = F.relu(x)
        return x


class FFN(nn.Module):
    def __init__(self, word_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        activation = "gelu"
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=word_dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
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

class AModel(nn.Module):
    """
    Answer embedding module
    """

    def __init__(self, bert_tokenizer, word_dim=768, out_dim=512):
        super(AModel, self).__init__()
        self.bert = Bert(bert_tokenizer)
        self.linear_text = nn.Linear(word_dim, out_dim)
        # self.linear_text = FFN(word_dim, out_dim, out_dim)

        
    def forward(self, answer):
        if len(answer.shape) == 3:
            #multi-choice
            bs, nans, lans = answer.shape
            answer = answer.view(bs * nans, lans)
            answer, hd_state = self.bert(answer)
            answer = self.linear_text(answer)
            answer_g = answer.mean(dim=1)
            # answer_g = answer[:, 0, :]
            answer_g = answer_g.view(bs, nans, -1)
        else:
            answer, hd_state = self.bert(answer)
            answer = self.linear_text(answer)
            answer_g = answer.mean(dim=1)
            # answer_g = answer[:, 0, :]
        
        return answer_g, answer
