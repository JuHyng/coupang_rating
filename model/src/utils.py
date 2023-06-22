import os
import random
import logging
import math

from datetime import datetime

import glob
import ntpath
import argparse

import json

from sklearn.metrics import matthews_corrcoef

import torch
import numpy as np

from transformers.utils.dummy_pt_objects import BertForQuestionAnswering
from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    BertTokenizer,
    ElectraTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    ElectraForSequenceClassification,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertModel,
    RobertaModel,
    ElectraModel,
    AutoTokenizer,
    AutoModel,
    ElectraForQuestionAnswering
)
from tokenization_kobert import KoBertTokenizer

'''
##util-functions for belows##
- model config infos: MODEL_CLASSES, MODEL_PATH_MAP, SPECIAL_TOKENS, TOKEN_MAX_LENGTH
- metric functions: MCC, compute_metrics, ...
- path functions: getParentPath
- save/load model_path functions: save_model, load_model
- token functions: getTokLength
'''


MODEL_CLASSES = {
    # 'distilkobert': (DistilBertConfig, DistilBertForSequenceClassification, KoBertTokenizer),
    # 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # 'kobert-lm': (BertConfig, BertForSequenceClassification, KoBertTokenizer),
    # 'koelectra-base': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
    # 'koelectra-small': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
    # 'kobert': (BertConfig, BertModel, KoBertTokenizer),
    # 'kobert-QA': (BertConfig, BertForQuestionAnswering, KoBertTokenizer),
    # 'roberta-base': (AutoConfig, RobertaModel, AutoTokenizer),
    # 'koelectra': (ElectraConfig, ElectraModel, ElectraTokenizer),
    # 'koelectraQA': (ElectraConfig, ElectraForQuestionAnswering, ElectraTokenizer),
    # 'koelectra_tunib':(AutoConfig, AutoModel, AutoTokenizer),
    'klue/roberta-large':(AutoConfig, AutoModel, AutoTokenizer),
    
    #encoder models
    'beomi/KcBERT-v2023':(),
    'beomi/KcELECTRA-base':(AutoConfig, AutoModel, AutoTokenizer),
    'BM-K/KoMiniLM-68M':(AutoConfig, AutoModel, AutoTokenizer),
    'kykim/electra-kor-base':(ElectraConfig, ElectraModel, ElectraTokenizer),
    
    #decoder
    'beomi/KoRWKV-1.5B':(),
    'beomi/KoAlpaca-Polyglot-12.8B':(),
    

}
#'kobert-QA': (BertConfig, BertForQuestionAnswering, KoBertTokenizer)
#'klue_roberta-base': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
#'kobert_NSP': (BertConfig, BertForNextSentencePrediction, KoBertTokenizer),

MODEL_PATH_MAP = {
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
    'koelectra-base': 'monologg/koelectra-base-discriminator',
    'koelectra-small': 'monologg/koelectra-small-discriminator',
    'kobert': 'monologg/kobert',
    'koelectra': 'monologg/koelectra-base-v3-discriminator',
    'koelectraQA': 'monologg/koelectra-base-discriminator',
    'roberta-base': 'klue/roberta-base',
    'koelectra_tunib': 'tunib/electra-ko-base',
    'klue/roberta-large': 'klue/roberta-large',
    'beomi/KcELECTRA-base': 'beomi/KcELECTRA-base',
    'BM-K/KoMiniLM-68M': 'BM-K/KoMiniLM-68M',
    'kykim/electra-kor-base':'kykim/electra-kor-base'
}

DATASET_PATHS = {
    'ABSA' : ('task_ABSA/','nikluge-sa-2022-train.jsonl', 'nikluge-sa-2022-dev.jsonl', 'nikluge-sa-2022-test.jsonl')
} #train, dev ,test, test_labeled
##COLA test_label 바꿈. 

SPECIAL_TOKENS_NUM = {
    'koelectra': (2, 3, 1),
    'roberta-base': (0, 2, 1),
    'koelectra_tunib': (2, 3, 1),
    'kobert': (2, 3, 1),
    'koelectraQA': (2, 3, 1),
} #token_num(CLS,SEP,PAD): koelectra(CLS=2,SEP=3,PAD=1), roberta(CLS=0,SEP=2,PAD=1)
#kobert 확인하기.

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<sep>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
} #WiC, 

TOKEN_MAX_LENGTH = {
    'ABSA' : 256,

} 


def get_label(args):
    return [0, 1]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(seedNum, device):
    torch.manual_seed(seedNum)
    torch.cuda.manual_seed(seedNum)
    torch.cuda.manual_seed_all(seedNum) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seedNum)
    random.seed(seedNum)

'''
def set_seed(seedNum, device):
    random.seed(seedNum) #default int42
    np.random.seed(seedNum)
    torch.manual_seed(seedNum)
    if device is not torch.device('cpu') and torch.cuda.is_available():
        torch.cuda.manual_seed(seedNum)
        torch.cuda.manual_seed_all(seedNum)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
'''

'''
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
'''

def parse_args():
    parser = argparse.ArgumentParser(description="sentiment analysis")
    parser.add_argument(
        "--train_data", type=str, default="../data/input_data_v1/train.json",
        help="train file"
    )
    parser.add_argument(
        "--test_data", type=str, default="../data/input_data_v1/test.json",
        help="test file"
    )
    parser.add_argument(
        "--load_pretrain",action="store_true"
    )
    parser.add_argument(
        "--load_model_path",type=str
    )
    parser.add_argument(
        "--dev_data", type=str, default="../data/input_data_v1/dev.json",
        help="dev file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--do_test", action="store_true"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--base_model", type=str, default="kykim/electra-kor-base"
    )
    parser.add_argument(
        "--entity_property_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--polarity_model_path", type=str, default="../saved_model/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/default_path/"
    )
    parser.add_argument(
        "--do_demo", action="store_true"
    )
    parser.add_argument(
        "--max_len", type=int, default=120
    )
    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    parser.add_argument(
        "--run_name", type=str, default='Coupang'
    )
    
    args = parser.parse_args()
    return args


#individual Metric
def MCC(preds, labels):
    assert len(preds) == len(labels)
    return matthews_corrcoef(labels, preds)

#monologg/kobert Metric
def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_score(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }





##get Path functions
def path_fpath(path):
    fpath, fname = ntpath.split(path)
    return fpath #fpath or ntpath.basename(fname)
def path_leaf(path):
    fpath, fname = ntpath.split(path)
    return ntpath.basename(fname) #fpath or ntpath.basename(fname)
def getFName(fname):
    fname_split = fname.split('.') #name, extenstion
    new_fname=fname_split[0]#+'.jpg'
    return new_fname

##get parent/home directory path##
def getParentPath(pathStr):
    return os.path.abspath(pathStr+"../../")
#return parentPth/parentPth of pathStr -> hdd1/
def getHomePath(pathStr):
    return getParentPath(getParentPath(getParentPath(pathStr))) #ast/src/

def print_timeNow():
    cur_day_time = datetime.now().strftime("%m/%d, %H:%M:%S") #Date %m/%d %H:%M:%S
    return cur_day_time

##save/load model path##
def save_model(saveDirPth_str, fileName_str, modelObj, optimizerObj): # '../model_chkpt/'
        # Save model checkpoint (Overwrite)
        if not os.path.exists(saveDirPth_str):
            os.makedirs(saveDirPth_str)
        #model_to_save = modelObj.module if hasattr(modelObj, 'module') else modelObj
        #model_to_save.save_pretrained(saveDirPth_str+fileName_str)
        torch.save({'model_state_dict':modelObj.state_dict(), 'optimizer_state_dict':optimizerObj.state_dict()}, saveDirPth_str+fileName_str)

        # Save training arguments together with the trained model
        #torch.save(self.args, os.path.join(saveDirPth_str, 'training_args.bin'))
        #logger.info("Saving model checkpoint to %s", saveDirPth_str)

def load_model(saveDirPth_str, fileName_str, modelObj, device): # '../model_chkpt/',
        # Save model checkpoint (Overwrite)
        if not os.path.exists(saveDirPth_str):
            print(saveDirPth_str,' not exist.')
            assert os.path.exists(saveDirPth_str) #return error
        model_chkpoint = torch.load(saveDirPth_str+fileName_str, map_location=device)
        modelObj.load_state_dict(model_chkpoint['model_state_dict'])

        return modelObj #model_chkpoint

def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list

def save_json(saveDirPth_str, jsonFileName_str, taskName_str, modelOutput): #modelOutput(array)
    makeJson = False
    modelOutput_saveDirPth = saveDirPth_str+taskName_str+'/'
    if not os.path.exists(modelOutput_saveDirPth):
        os.makedirs(modelOutput_saveDirPth)
    
    if makeJson is True:
        json_data = json.load(modelOutput_saveDirPth)
    else:
        json_data = {}
    #print('json type: ',type(json_data), len(json_data))
    json_data[taskName_str] = []#.append({"boolq":[]})
    for idx, out in enumerate(modelOutput):
        if taskName_str == 'cola':
            #print('COLA')
            newData = {"idx":idx, "label":int(out)}
        elif taskName_str == 'copa':
            #print('COPA')
            newData = {"idx":idx+1, "label":int(out)+1}
        else: #'wic', 'copa', 'boolq'
            #print('WiC or BoolQ')
            newData = {"idx":idx+1, "label":int(out)}
        json_data[taskName_str].append(newData)
    
    #print('json type: ',type(json_data), len(json_data))
    #print(json_data)

    with open(modelOutput_saveDirPth+jsonFileName_str, 'w') as json_file:
        json_str = json.dump(json_data, json_file, indent=4)


##token functions##
def getTokLength(tokStr):
    if tokStr == '[CLS]': #'[ C L S ]':
        tokLen = 1#0
    elif tokStr in ['[SEP]', '[PAD]']: #['[ S E P ]', '[ P A D ]']: #tokStr == '[ S E P ]' or tokStr == '[ P A D ]':
        tokLen = 1
    elif tokStr.startswith('#'): #ex) '##에' -> '에'
        #print('## string: ',tokStr)
        tokLen = len(tokStr[2:]) #tokLen = getTokLength(tokStr[4:])+1 #len(tokStr)-4
    else:
        tokLen = len(tokStr)+1#math.ceil(len(tokStr)/2) #띄어쓰기 포함
    return tokLen
def tokPooling(emblist,strategy): # Pooling Strategy: bs,
    #strategy = 1
    #mean,max,concat 등 시도해보기.
    if strategy == 'meanPooling': #1) mean-pooling 
        embeddings = torch.stack([emblist[i] for i in range(len(emblist))]) #torch.tensor(embedding1)
        outEmbedding = torch.mean(embeddings, dim=0) #1) mean-pooling
    elif strategy == 'concat':#2) 2개씩
        #print(len(emblist), emblist[0].shape, emblist[-1].shape)
        if len(emblist)==1:
            emblist = emblist+emblist
        elif len(emblist)==2:
            emblist = emblist
        elif len(emblist)>2:
            #print('here ',emblist[0].shape,emblist[-1].shape)
            emblist = [emblist[0],emblist[-1]]
        else:
            pass
        embeddings = torch.stack([emblist[i] for i in range(len(emblist))]) # list(2*[768]) -> torch.Size([2, 768])
        outEmbedding =embeddings#emblist embeddings
    else:
        print('no strategy.')
        pass
    #embeddings = torch.stack([emblist[i] for i in range(len(emblist))]) # list(2*[768]) -> torch.Size([2, 768])
    #print(embeddings.shape)

    #torch.cat([embeddings[0],embeddings[1]],dim=1)
    #(len(emblist), embeddings.shape)
    #print(len(emblist))
    #outEmbedding =embeddings#emblist embeddings
    
    return outEmbedding