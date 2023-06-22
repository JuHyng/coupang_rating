#!/bin/bash

#. conda/bin/activate
#conda activate [가상환경 이름:team#] #임시(lss)

echo run task: Train model!!
wandb login d06ca290bf76e5ed0d636a32474b9719e1666f39
python train_model.py \
  --base_model kykim/electra-kor-base \
  --do_train \
  --do_eval \
  --learning_rate 3.0774563978533025e-07 \
  --eps 2.6932577697450457e-08 \
  --num_train_epochs 30 \
  --polarity_model_path ../saved_model/ \
  --batch_size 64 \
  --max_len 120\
  --run_name 'kykim/electra-kor-base, lr=3.0774563978533025e-07, eps=2.6932577697450457e-08, bs=64'