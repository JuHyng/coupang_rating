import torch
from torch import nn
from torch.utils.data import Dataset #, DataLoader
#from torch.optim import Adam

import pandas as pd
import numpy as np
import os
from utils import parse_args
from utils import MODEL_CLASSES, MODEL_PATH_MAP 
from utils import TOKEN_MAX_LENGTH, getTokLength #SPECIAL_TOKENS
from utils import jsonlload
from torch.utils.data import TensorDataset

args=parse_args()

model_name = args.base_model #설정한 모델이름
_1, _2, model_tokenizer = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
model_path = MODEL_PATH_MAP[model_name]

data_path=os.getcwd()+'/../../data/'
max_tokenizer_length = 256#100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.
#max_tokenizer_length = TOKEN_MAX_LENGTH['WiC']

polarity_id_to_name = ['positive', 'negative']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

polarity_count = 0


## model dataloader ##

def get_dataset(data_path, tokenizer, max_len):
    raw_data = jsonlload(data_path)

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []

    for utterance in raw_data:
        polarity_data_dict = tokenize_and_align_labels(tokenizer, utterance['review_content'], utterance['label'], max_len)

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])

    print('polarity_data_count: ', polarity_count)

    return TensorDataset(torch.tensor(polarity_input_ids_list), torch.tensor(polarity_attention_mask_list),
                         torch.tensor(polarity_token_labels_list))                            

def tokenize_and_align_labels(tokenizer, sentence, label, max_len):
    
    global polarity_count

    polarity_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }

    tokenized_data = tokenizer(sentence, padding='max_length', max_length=max_len, truncation=True)

    polarity_count += 1

    polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
    polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    if label not in polarity_name_to_id:
        print(label)
        print(sentence)
    polarity_data_dict['label'].append(polarity_name_to_id[label])
    
    # input='다음 후기의 '+pair+'에 대해 어떻게 생각하나요?'+form
    # tokenized_polarity_data = tokenizer(input, padding='max_length', max_length=max_len, truncation=True)
    # polarity_data_dict['input_ids'].append(tokenized_polarity_data['input_ids'])
    # polarity_data_dict['attention_mask'].append(tokenized_polarity_data['attention_mask'])
    # polarity_data_dict['label'].append(polarity_name_to_id[polarity])


    return polarity_data_dict



