import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os
import json

from pprint import pprint
from datetime import datetime

from transformers import AdamW
from tqdm import trange
from utils import compute_metrics,  MCC, get_label, set_seed, jsonlload, parse_args,jsondump,jsonload
from utils import MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS #print_timeNow
from utils import MODEL_CLASSES, MODEL_PATH_MAP 
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import wandb
import copy


from datasets import get_dataset
from model import model_ABSA

from transformers import AutoTokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path=os.getcwd()+'/../../dataset/'
model_name = 'kykim/electra-kor-base' #'kobert', 'roberta-base', 'koelectra'
_1, _2, model_tokenizer = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
task_name = 'ABSA' 
taskDir_path, fname_train, fname_dev, fname_test,  = DATASET_PATHS[task_name]

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = TOKEN_MAX_LENGTH[task_name] #100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.
model_path = MODEL_PATH_MAP[model_name]
label_id_to_name = ['True', 'False']
polarity_id_to_name = ['positive', 'negative']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

def predict_from_korean_form(polarity_tokenizer, pc_model, data):
    
    idx=0
    for utterance in data:
        
        review=utterance['review_content']
        label=utterance['label']

        p_tokenized_data = polarity_tokenizer(review, padding='max_length', max_length=256, truncation=True)

        p_input_ids = torch.tensor([p_tokenized_data['input_ids']]).to(device)
        p_attention_mask = torch.tensor([p_tokenized_data['attention_mask']]).to(device)
        
        with torch.no_grad():
            _, pc_logits = pc_model(p_input_ids, p_attention_mask)

        pc_predictions = torch.argmax(pc_logits, dim=-1)
        pc_result = polarity_id_to_name[pc_predictions[0]]

        data[idx]['label']=pc_result
        idx+=1
    return data
    
def evaluation(testdata, pred_data, label_len):
    y_true=[]
    y_pred=[]
    for utterance in testdata:
        print(polarity_name_to_id[utterance['label']])
        y_true.append(polarity_name_to_id[utterance['label']])
        
        
    for utterance in pred_data:
        y_pred.append(polarity_name_to_id[utterance['label']])
        
    print(y_true)
    
    count_list = [0]*label_len
    hit_list = [0]*label_len
    for i in range(len(y_true)):
        count_list[y_true[i]] += 1
        if y_true[i] == y_pred[i]:
            hit_list[y_true[i]] += 1
    acc_list = []

    for i in range(label_len):
        acc_list.append(hit_list[i]/count_list[i])

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    
    # Polarity 모델 평가
    print(count_list)
    print(hit_list)
    print(acc_list)
    print('Polarity_accuracy: ', (sum(hit_list) / sum(count_list)))
    # wandb.log({"Polarity_accuracy": (sum(hit_list) / sum(count_list))})
    print('Polarity_macro_accuracy: ', sum(acc_list) / 3)
    # wandb.log({"Polarity_macro_accuracy": sum(acc_list) / 3})
    print('Polarity_f1_score: ', f1_score(y_true, y_pred, average=None))
    print('Polarity_f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))#micro는 데이터 불균형을 고려
    print('Polarity_f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))#macro는 데이터 뷸균형 고려 X

    # wandb.log({"Inference_Polarity_f1_score": f1_score(y_true, y_pred, average=None)})
    # wandb.log({"Inference_Polarity_f1_score_micro": f1_score(y_true, y_pred, average='micro')})
    # wandb.log({"Inference_Polarity_f1_score_macro": f1_score(y_true, y_pred, average='macro')})
    

def test_sentiment_analysis(args):
    
    polarity_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # test_data = jsonlload(args.test_data)
    
    data_path='/home/dl/TP_DL/AL2/model/dataset/test_data.jsonl'
    test_data = jsonlload(data_path)

    polarity_model = model_ABSA(args, len(polarity_id_to_name), len(polarity_tokenizer))
    polarity_model.load_state_dict(torch.load(args.polarity_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form(polarity_tokenizer, polarity_model, copy.deepcopy(test_data))


    jsondump(pred_data, '/home/dl/TP_DL/AL2/model/dataset/inference.json')
    pred_data = jsonload('/home/dl/TP_DL/AL2/model/dataset/inference.json')

    #print('F1 result: ', evaluation_f1(test_data, pred_data))

    # pred_list = []
    # label_list = []
    # print('polarity classification result')
    # for batch in polarity_test_dataloader:
    #     batch = tuple(t.to(device) for t in batch)
    #     b_input_ids, b_input_mask, b_labels = batch

    #     with torch.no_grad():
    #         loss, logits = polarity_model(b_input_ids, b_input_mask, b_labels)

    #     predictions = torch.argmax(logits, dim=-1)
    #     pred_list.extend(predictions)
    #     label_list.extend(b_labels)

    evaluation(test_data, pred_data, len(polarity_id_to_name))

    
if __name__ == "__main__":
    args = parse_args()

    # if args.do_train:
    #     train_sentiment_analysis(args)
    # # elif args.do_demo:bash t
    # #     demo_sentiment_analysis(args)
    if args.do_test:
        test_sentiment_analysis(args)
    