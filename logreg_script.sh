#!/bin/bash
lrarray=(0.03 0.05 0.07)

#initial training

for lr in ${lrarray[@]}; do


python3 -u main.py --model logreg --dataset extracted_cifar10 --use_lora --lora_rank 4 --lr ${lr} --num_classes 10 --epochs 60  --batch_size 512


done

python3 -u main.py --model logreg --dataset extracted_cifar10 --use_lora --lora_rank 4 --lr_schedule adaptive --lr 1 --num_classes 10 --epochs 60  --batch_size 512
python3 -u main.py --model logreg --dataset extracted_cifar10 --use_lora --lora_rank 4 --lr_schedule adaptive_2 --lr 0.8 --num_classes 10 --epochs 60  --batch_size 512
python3 -u main.py --model logreg --dataset extracted_cifar10 --use_lora --lora_rank 4 --lr_schedule normalized --lr 0.06 --num_classes 10 --epochs 60  --batch_size 512


