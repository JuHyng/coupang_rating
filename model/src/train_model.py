import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os
import argparse

from pprint import pprint
from datetime import datetime

from transformers import AdamW
from tqdm import trange
from utils import compute_metrics,  MCC, get_label, set_seed, parse_args
from utils import MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH 
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS #print_timeNow
from utils import MODEL_CLASSES, MODEL_PATH_MAP 
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import wandb
import config
import math


from datasets import get_dataset
from model import model_ABSA

from transformers import AutoTokenizer

data_path=os.getcwd()+'/../../dataset/'

args=parse_args()

model_name = args.base_model #설정한 모델이름

_1, _2, model_tokenizer = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
task_name = 'ABSA' 
taskDir_path, fname_train, fname_dev, fname_test,  = DATASET_PATHS[task_name]


model_path = MODEL_PATH_MAP[model_name]

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = TOKEN_MAX_LENGTH[task_name] 

label_id_to_name = ['True', 'False']
polarity_id_to_name = ['positive', 'negative']



#wandb 연결(학습내용 시각화)
wandb.init(project="DL_gradientAscent", entity="kimy", name=args.run_name)
# run_name = str(args.base_model)+" lr:"+str(args.learning_rate)+" eps:"+str(args.eps)
# wandb.init(name=run_name)
w_config = wandb.config

def evaluation(y_true, y_pred, label_len):
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
    wandb.log({"Polarity_accuracy": (sum(hit_list) / sum(count_list))})
    print('Polarity_macro_accuracy: ', sum(acc_list) / 3)
    wandb.log({"Polarity_macro_accuracy": sum(acc_list) / 3})
    print('Polarity_f1_score: ', f1_score(y_true, y_pred, average=None))
    print('Polarity_f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))#micro는 데이터 불균형을 고려
    print('Polarity_f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))#macro는 데이터 뷸균형 고려 X

    wandb.log({"Polarity_f1_score": f1_score(y_true, y_pred, average=None)})
    wandb.log({"Polarity_f1_score_micro": f1_score(y_true, y_pred, average='micro')})
    wandb.log({"Polarity_f1_score_macro": f1_score(y_true, y_pred, average='macro')})
        

if __name__ == "__main__":
    
    #모델 저장 경로 생성
    if not os.path.exists(args.polarity_model_path):
        os.makedirs(args.polarity_model_path)
        
    homePth = getParentPath(os.getcwd())
    datasetPth = homePth+'/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())
    #start_day_time=print_timeNow()
    #print('training start at (date, time): ',print_timeNow())

    tsvPth_train='/home/dl/TP_DL/AL2/model/dataset/train_data.jsonl' 
    tsvPth_dev = '/home/dl/TP_DL/AL2/model/dataset/dev_data.jsonl'
    tsvPth_test = '/home/dl/TP_DL/AL2/model/dataset/test_data.jsonl'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed 정수로 고정.

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    lf = nn.CrossEntropyLoss()

    train_data = get_dataset(tsvPth_train, tokenizer, args.max_len)
    dev_data = get_dataset(tsvPth_dev, tokenizer, args.max_len)
    
    train_dataloader = DataLoader(train_data, shuffle=True,
                                                  batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_data, shuffle=True,
                                                batch_size=args.batch_size)
    

    
    print('loading model')

    polarity_model = model_ABSA(args,len(polarity_id_to_name),len(tokenizer))
    polarity_model.to(device)

    print('end loading')
    
    FULL_FINETUNING = True

    # polarity_model_optimizer_setting
    if FULL_FINETUNING:
        polarity_param_optimizer = list(polarity_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        polarity_optimizer_grouped_parameters = [
            {'params': [p for n, p in polarity_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in polarity_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        polarity_param_optimizer = list(polarity_model.classifier.named_parameters())
        polarity_optimizer_grouped_parameters = [{"params": [p for n, p in polarity_param_optimizer]}]

    polarity_optimizer = AdamW(
        polarity_optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
        # lr = w_config.learning_rate,
        # eps=w_config.eps
    )
    # epochs = w_config.num_train_epochs
    epochs = args.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(train_dataloader)

    polarity_scheduler = get_linear_schedule_with_warmup(
        polarity_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    
    print('[Training Phase]')
    epoch_step = 0

    for _ in trange(epochs, desc="Epoch"):
        epoch_step += 1
    
        polarity_model.train()
        
        # polarity train
        polarity_total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            #batch = tuple(nn.DataParallel(t).to(device) for t in batch) #GPU parellel            
            b_input_ids, b_input_mask, b_labels = batch
        
            
            polarity_model.zero_grad()

            loss, _ = polarity_model(b_input_ids, b_input_mask, b_labels)

            loss.backward()

            polarity_total_loss += loss.item()
            # print('batch_loss: ', loss.item())

            torch.nn.utils.clip_grad_norm_(parameters=polarity_model.parameters(), max_norm=max_grad_norm)
            polarity_optimizer.step()
            polarity_scheduler.step()

        avg_train_loss = polarity_total_loss / len(train_dataloader)
        print("Entity_Property_Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        if epoch_step==27:
            model_saved_path = args.polarity_model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
            torch.save(polarity_model.state_dict(), model_saved_path)

        with torch.no_grad():
            polarity_model.eval()

            pred_list = []
            label_list = []

            for batch in dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                #batch = tuple(nn.DataParallel(t).to(device) for t in batch) #GPU parellel
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    loss, logits = polarity_model(b_input_ids, b_input_mask, b_labels)
                    wandb.log({"loss":loss})
                    

                predictions = torch.argmax(logits, dim=-1)
                pred_list.extend(predictions)
                label_list.extend(b_labels)

            evaluation(label_list, pred_list, len(polarity_id_to_name))

    print("training is done")

    print('end main')



