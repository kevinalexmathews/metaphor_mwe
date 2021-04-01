import torch
import torch.nn as nn
from torch.nn.functional import pad
from transformers import AdamW


class Seqlab(nn.Module):

    def __init__(self, config, dropout, bert, layer_no, num_labels=2):
        super(Seqlab, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(config.hidden_size, hidden_size=100, num_layers=1, bidirectional=True)
        self.classifier = nn.Linear(100*2, num_labels)
        self.layer_no = layer_no

    def forward(self, input_ids, target_token_idx, attention_mask, batch, labels=None):
        outputs = self.bert(input_ids, output_hidden_states=True)
        out_bert = outputs.hidden_states[self.layer_no]

        out_recurrent, _ = self.bilstm(out_bert)
        logits = self.classifier(out_recurrent)

        if labels is not None: # train time
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels).cpu(), labels.view(-1).cpu())
            return loss
        else: # inference time
            return nn.functional.log_softmax(logits,dim=1)

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True

class Seqlabbase(nn.Module):

    def __init__(self, config, dropout, bert, window_size, num_labels=2):
        super(Seqlabbase, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)
        num_units = 256
        self.lstm = nn.LSTM(config.hidden_size, hidden_size=num_units, num_layers=1, bidirectional=False)
        self.linear = nn.Linear(num_units*2,num_units) # x2 for LR contexts
        self.classifier = nn.Linear(num_units, num_labels)

    def forward(self, input_ids, target_token_idx, lcontext_indices, rcontext_indices, attention_mask, batch, labels=None):
        bert_outputs = self.bert(input_ids)
        bert_lhs = bert_outputs.last_hidden_state

        # pad with window_size of words to account for negative indices
        bert_lhs_padded = pad(bert_lhs, (0, 0, self.window_size, self.window_size))

        lcontext_indices = lcontext_indices.add(self.window_size)
        lcontext_indices_for_gather = lcontext_indices.unsqueeze(2) # convert 2D tensor to 3D
        lcontext_indices_for_gather = lcontext_indices_for_gather.expand(-1,-1,bert_lhs.shape[-1])
        out_lcontexts = torch.gather(bert_lhs_padded,1,lcontext_indices_for_gather)

        rcontext_indices = rcontext_indices.add(self.window_size)
        rcontext_indices_for_gather = rcontext_indices.unsqueeze(2) # convert 2D tensor to 3D
        rcontext_indices_for_gather = rcontext_indices_for_gather.expand(-1,-1,bert_lhs.shape[-1])
        out_rcontexts = torch.gather(bert_lhs_padded,1,rcontext_indices_for_gather)

        out_recurrent_left, _ = self.lstm(out_lcontexts)
        out_recurrent_left = out_recurrent_left[:, -1, :] # keep only the last timestep
        out_recurrent_right, _ = self.lstm(out_rcontexts)
        out_recurrent_right = out_recurrent_right[:, -1, :] # keep only the last timestep
        out_recurrent = torch.cat((out_recurrent_right, out_recurrent_left), 1)

        linear_out = self.dropout(self.linear(out_recurrent))
        logits = self.classifier(linear_out)

        if labels is not None: # train time
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels).cpu(), labels.view(-1).cpu())
            return loss
        else: # inference time
            return nn.functional.log_softmax(logits,dim=1)

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True
