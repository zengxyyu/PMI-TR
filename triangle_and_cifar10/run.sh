#!/bin/bash

#data='Triangle or  cifar10'
data=${1}
model=${2}
patch_size=${3}
version=0
num_layers=${4}
num_templates=1
dropout=0.1
epochs=200
num_heads=4
batch_size=100
lr=0.0001
h_dim=${5}
ffn_dim=${6}
num_gru_schemas=2
num_attention_schemas=2
schema_specific=True
num_eval_layers=6
share_vanilla_parameters=${7}
use_topk=${8}
topk=${9}
shared_memory_attention=${10}
mem_slots=${11}
null_attention=False
use_long_men=${12}
long_mem_segs=${13}
long_mem_aggre=${14}
use_wm_inference=${15}
seed=${16}



name=$data-$model"-version-"$version"-num_layers-"$num_layers"-num_templates-"$num_layers"-dropout-"$dropout"-epochs-"$epochs"-patch_size-"$patch_size"-num_heads-"$num_heads"-batch_size-"$batch_size"-lr-"$lr-$h_dim-$ffn_dim-$num_gru_schemas-$num_attention_schemas-$schema_specific-$num_eval_layers-$share_vanilla_parameters-$use_topk-$topk-$shared_memory_attention-$mem_slots-$null_attention-$seed

echo $name

python run_cnn.py --model $model --data $data --version $version --num_layers $num_layers \
        --num_templates $num_templates --dropout $dropout --epochs $epochs --patch_size $patch_size \
        --num_heads $num_heads --name $name --batch_size $batch_size --lr $lr \
				--h_dim $h_dim --ffn_dim $ffn_dim --num_gru_schemas $num_gru_schemas \
				--num_attention_schemas $num_attention_schemas --schema_specific $schema_specific \
				--num_eval_layers $num_eval_layers --share_vanilla_parameters $share_vanilla_parameters \
				--use_topk $use_topk --topk $topk --shared_memory_attention $shared_memory_attention \
				--mem_slots $mem_slots --null_attention $null_attention \
				--use_long_men $use_long_men --long_mem_segs $long_mem_segs --long_mem_aggre $long_mem_aggre \
				--use_wm_inference $use_wm_inference --seed $seed

#sh run_local.sh functional cifar10 1 12 3 0.1 200 4 4 128 0.0001
