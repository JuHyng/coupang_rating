#!/bin/bash

#. conda/bin/activate
#conda activate [가상환경 이름:team#] #임시(lss)

echo run task: test polarity_model  !!
wandb login d06ca290bf76e5ed0d636a32474b9719e1666f39
python test.py \
  --base_model kykim/electra-kor-base\
  --do_test \
  --polarity_model_path /home/dl/TP_DL/AL2/model/saved_model/saved_model_epoch_6.pt \
  --max_len 256\
  --load_pretrain\
  --classifier_hidden_size 768
