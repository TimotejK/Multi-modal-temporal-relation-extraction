import random

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import optim
import torch_geometric.utils as pyg_utils
import torch.nn.functional as F

class TemporalRelationAggregation(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_features):
        super(TemporalRelationAggregation, self).__init__(aggr='add')  # "Add" aggregation.
        self.edge_features = edge_features
        self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=True) for i in range(edge_features)])
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, edge_attr):
        # Transform node feature matrix.
        self_x = self.lin_self(x)
        return self_x + self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        sum = None
        for i in range(self.edge_features):
            a = self.lins[i](x_j)
            a = a * edge_attr[:,i][:,None] # weigth according to feature vector
            if sum is None:
                sum = a
            else:
                sum += a

        return sum

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out


class MultiModalPrediction(nn.Module):
    def __init__(self, number_of_relations=3, combine_embeddings=True):
        super(MultiModalPrediction, self).__init__()
        self.combine_embeddings = combine_embeddings
        self.number_of_relations = number_of_relations
        self.graph_model = GNNRelationPrediction(number_of_relations=number_of_relations)
        self.text_model = EntityBERTRelationExtraction(number_of_relations=number_of_relations, pooling_strategy='both_events')
        self.pooling_strategy = self.text_model.pooling_strategy
        if self.combine_embeddings:
            self.linear = nn.Linear(50 * 2 + 768 * 2, self.number_of_relations).double()
        else:
            self.linear = nn.Linear(self.number_of_relations * 2, self.number_of_relations).double()

    def forward(self, data):
        graph_prediction = self.graph_model(data, return_embedding=self.combine_embeddings)
        text_prediction = self.text_model(data, return_embedding=self.combine_embeddings)
        concatenated = torch.cat((graph_prediction, text_prediction), 1)
        concatenated = concatenated.double()
        x = self.linear(concatenated)
        return x
    pass


class GNNRelationPrediction(nn.Module):
    def __init__(self, input_dim=50, number_of_relations=3, dropout=0.2, use_edge_features=False, num_edge_features=3):
        super(GNNRelationPrediction, self).__init__()
        self.num_edge_features = num_edge_features
        self.simulate_mislabeled_relations = False
        self.use_edge_features = use_edge_features
        self.pooling_strategy = "concat_entities"
        self.number_of_relations = number_of_relations
        hidden_dim = 50
        heads = 1
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim, heads))
        self.lns = nn.ModuleList()
        self.num_layers = 2
        for l in range(self.num_layers - 1):
            self.lns.append(nn.LayerNorm(hidden_dim * heads).float())
            self.convs.append(self.build_conv_model(hidden_dim * heads, hidden_dim, heads))
        self.linear = nn.Linear(hidden_dim * heads, hidden_dim).float()

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, number_of_relations))
        self.dropout = dropout

    def build_conv_model(self, input_dim, hidden_dim, heads):
        if self.use_edge_features:
            num_edge_features = self.num_edge_features
            # return pyg_nn.NNConv(input_dim, hidden_dim, nn=nn.Sequential(nn.Linear(num_edge_features, input_dim*hidden_dim)), aggr="add").float()
            return TemporalRelationAggregation(input_dim, hidden_dim, num_edge_features)
        else:
            return pyg_nn.RGCNConv(input_dim, hidden_dim, self.number_of_relations).float()

    def forward(self, data, return_embedding=False):
        # if not self.training:
        #     return nn.functional.one_hot(data.rule_based_prediction).double()
        if self.use_edge_features:
            x, edge_index, edge_type, batch, event1, event2, edge_attr = data.x, data.edge_index, data.edge_type, data.batch, data.event1_index, data.event2_index, data.edge_attr
        else:
            x, edge_index, edge_type, batch, event1, event2 = data.x, data.edge_index, data.edge_type, data.batch, data.event1_index, data.event2_index
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)
        if self.simulate_mislabeled_relations:
            edge_type_realistic = edge_type.clone()
            for i in range(len(edge_type_realistic)):
                if random.random() > 0.75:
                    r = edge_type_realistic[i]
                    while r == edge_type_realistic[i]:
                        r = random.randint(0, self.number_of_relations)
                    edge_type_realistic[i] = r
        else:
            edge_type_realistic = edge_type
        # edge_attr = nn.functional.one_hot(edge_type.long(), self.number_of_relations + 1).double()

        for i in range(self.num_layers):
            if self.use_edge_features:
                x = self.convs[i](x=x.float(), edge_index=edge_index, edge_attr=edge_attr.float())
            else:
                x = self.convs[i](x=x.float(), edge_index=edge_index, edge_type=edge_type_realistic.long())
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = self.linear(x)

        event1_emb = x[event1]
        event2_emb = x[event2]
        x = torch.cat((event1_emb, event2_emb), 1)

        if return_embedding:
            return x

        x = self.post_mp(x)

        # return F.log_softmax(x, dim=1)
        return x

# pooling_strategy: [both_events, cls, pool]
class BaselineBERTrelationExtraction(nn.Module):
    def __init__(self, plm_model='bert-base-cased', number_of_relations=3, dropout=0.2, deeper_network=False, pooling_strategy='both_events'):
        super(BaselineBERTrelationExtraction, self).__init__()
        self.EntityBert = AutoModel.from_pretrained(plm_model)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_model)
        self.pooling_strategy = pooling_strategy
        for param in self.EntityBert.parameters():
            param.requires_grad = False
        if pooling_strategy == 'both_events':
            input_size = 768 * 2
        else:
            input_size = 768
        self.post_layers = nn.Linear(input_size, number_of_relations)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        text, event1_start, event1_end, event2_start, event2_end = data.text, data.event1_start, data.event1_end, data.event2_start, data.event2_end
        text = list(text)
        tokens = self.tokenizer(text, return_tensors="pt", max_length=100, padding='max_length', truncation=True)
        tokens.to(self.EntityBert.device)
        x = self.EntityBert(**tokens)
        if self.pooling_strategy == 'cls':
            bert_output = x['last_hidden_state'][:, 0, :]
        elif self.pooling_strategy == 'pool':
            bert_output = x['pooler_output']
        else:
            bert_output = []
            for i, layer in enumerate(x['last_hidden_state']):
                event1_emb = x['last_hidden_state'][i][
                             tokens[i].char_to_token(event1_start[i]):tokens[i].char_to_token(event1_end[i])]
                event1_emb = torch.mean(event1_emb, 0)
                event2_emb = x['last_hidden_state'][i][
                             tokens[i].char_to_token(event2_start[i]):tokens[i].char_to_token(event2_end[i])]
                event2_emb = torch.mean(event2_emb, 0)
                output = torch.cat((event1_emb, event2_emb))
                bert_output.append(output)
            bert_output = torch.stack(bert_output)

        x = self.post_layers(bert_output)
        return x

class EntityBERTRelationExtraction(nn.Module):
    def __init__(self, number_of_relations=3, dropout=0.2, deeper_network=False, pooling_strategy='cls'):
        super(EntityBERTRelationExtraction, self).__init__()
        self.EntityBert = AutoModel.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT")
        self.pooling_strategy = pooling_strategy
        for param in self.EntityBert.parameters():
            param.requires_grad = False

        if pooling_strategy == 'both_events':
            input_size = 768 * 2
        else:
            input_size = 768

        if deeper_network:
            self.post_layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_size, 256),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, number_of_relations))
        else:
            self.post_layers = nn.Linear(input_size, number_of_relations)
        self.softmax = nn.Softmax(dim=1)

    # def forward(self, text, event1_start, event1_end, event2_start, event2_end):
    def forward(self, data, return_embedding=False):
        text, event1_start, event1_end, event2_start, event2_end = data.text, data.event1_start, data.event1_end, data.event2_start, data.event2_end
        text = list(text)
        tokens = self.tokenizer(text, return_tensors="pt", max_length=100, padding='max_length', truncation=True)
        tokens.to(self.EntityBert.device)
        x = self.EntityBert(**tokens)
        if self.pooling_strategy == 'cls':
            bert_output = x['last_hidden_state'][:, 0, :]
        elif self.pooling_strategy == 'pool':
            bert_output = x['pooler_output']
        else:
            bert_output = []
            for i, layer in enumerate(x['last_hidden_state']):
                event1_emb = x['last_hidden_state'][i][
                             tokens[i].char_to_token(event1_start[i]):tokens[i].char_to_token(event1_end[i])]
                event1_emb = torch.mean(event1_emb, 0)
                event2_emb = x['last_hidden_state'][i][
                             tokens[i].char_to_token(event2_start[i]):tokens[i].char_to_token(event2_end[i])]
                event2_emb = torch.mean(event2_emb, 0)
                output = torch.cat((event1_emb, event2_emb))
                bert_output.append(output)
            bert_output = torch.stack(bert_output)

        if return_embedding:
            return bert_output

        x = self.post_layers(bert_output)
        return x
        # return self.softmax(x)
        pass