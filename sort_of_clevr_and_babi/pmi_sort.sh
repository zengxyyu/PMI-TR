#!/bin/bash
#source ~/.bashrc

embed_dim=$1
num_layers=$2
num_heads=$3
share_vanilla_parameters=$4
use_topk=$5
topk=$6
shared_memory_attention=$7
mem_slots=$8
null_attention=False
use_long_men=$9
long_mem_segs=${10}
long_mem_aggre=${11}
use_wm_inference=${12}
seed=${13}
set_transformer=${14}



save_dir=$embed_dim-$num_layers-$num_heads-$set_transformer-$share_vanilla_parameters-$use_topk-$topk-$shared_memory_attention-$mem_slots-$null_attention-$seed

mkdir $save_dir

python sort_main.py --model Transformer --epochs 200 --embed_dim $embed_dim --num_layers $num_layers \
         --num_heads $num_heads --functional $set_transformer --share_vanilla_parameters $share_vanilla_parameters \
			   --use_topk $use_topk --topk $topk --shared_memory_attention $shared_memory_attention \
			   --save_dir $save_dir --mem_slots $mem_slots --null_attention $null_attention \
			   --use_long_men $use_long_men --long_mem_segs $long_mem_segs \
			   --long_mem_aggre $long_mem_aggre --use_wm_inference $use_wm_inference --seed $seed



