#!/bin/bash

lrarray=( 0.03 0.04 0.05 )

#initial training

for lr in ${lrarray[@]}; do


python3 -u main.py --model resnet18 --dataset cifar10 --use_lora --lora_rank 20 --lr ${lr} --num_classes 10 --epochs 250 --batch_size 512 --device 2 

done

python3 -u main.py --model resnet18 --dataset cifar10 --use_lora --lora_rank 20 --lr_schedule normalized --lr 0.1  --num_classes 10 --epochs 250 --batch_size 512
python3 -u main.py --model resnet18 --dataset cifar10 --use_lora --lora_rank 20 --lr_schedule adaptive_2 --lr 15  --num_classes 10 --epochs 250 --batch_size 512


#sudo apt install bc