import torch
from torch import nn
#from torch.utils.data import Dataset, DataLoader
#from torch.optim import Adam

import pandas as pd
import numpy as np
import os
from utils import parse_args
from utils import MODEL_CLASSES, MODEL_PATH_MAP #SPECIAL_TOKENS
from utils import jsonload

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = 100 #100 #20 

args=parse_args()

model_name = args.base_model #설정한 모델이름
config_class, model_class, _ = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
model_path = MODEL_PATH_MAP[model_name]
modelClass_config = config_class.from_pretrained(model_path)

## model ##
class model_ABSA(nn.Module):
    def __init__(self,args, num_label,len_tokenizer):
        super(model_ABSA, self).__init__()
        self.num_label = num_label
        self.model_PLM = model_class.from_pretrained(args.base_model)
        self.model_PLM.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(args,self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model_PLM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )
        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_label),
                                                labels.view(-1))
        return loss, logits
    
class SimpleClassifier(nn.Module):
    
    def __init__(self,args,num_label):
        super().__init__()
        self.FC_layer1 = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.FC_layer2 = nn.Linear(args.classifier_hidden_size, num_label)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.FC_layer1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.FC_layer2(x)
        return x

