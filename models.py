import torch
import torch.nn as nn
from transformers import AdamW


class BertWithGCNAndMWE(nn.Module):

    def __init__(self, config, dropout, bert, num_labels=2):
        super(BertWithGCNAndMWE, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(config.hidden_size,256)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, target_token_idx, attention_mask, batch, labels=None):
        outputs = self.bert(input_ids)
        gcn = outputs.last_hidden_state

        target_token_idx_for_gather = target_token_idx.reshape(-1,1,1)
        target_token_idx_for_gather = target_token_idx_for_gather.expand(-1,1,gcn.shape[-1])
        gcn_pooled = torch.gather(gcn,1,target_token_idx_for_gather).view(batch,-1)

        output = self.dropout(self.linear(gcn_pooled))
        logits = self.classifier(output)

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
