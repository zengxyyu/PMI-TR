# A Framework for Inference Inspired by Human Memory Mechanisms

This repository contains the code to reproduce the `relational reasoning: sort_of_clever`,`text-based question-answering: bAbI`, `detecting equilateral triangles` and `cifar-10` tasks from our paper.  


## Install relevant libraries
```
pip install -r requirements.txt 
```
## Task1: Sort-of-CLEVR
You can find the source code for the Sort-of-CLEVR task in `sort_of_clevr_and_babi` folder.

Firstly, dataset generation:
```
python sort_of_clevr_generator.py
```
**Then, you can run `sort_main.py` directly.**

**Or execute the following commands to reproduce the experimental results of the Sort-of-CLEVR dataset in our paper.**
```
sh pmi_sort.sh h_dim num_layers num_heads share_vanilla_parameters use_topk topk shared_memory_attention mem_slots use_long_men long_mem_segs long_mem_aggre use_wm_inference seed set_transformer
```
**Explanation of Parameters:**

`h_dim`: Embedding dimensions

`num_layers`: Number of model layers

`num_heads`: Number of heads in multi-headed attention

`share_vanilla_parameters`: Whether share parameters across layers.

`use_topk`: Whether to use top-k competition

`topk`: Value of k in top-k competition

`shared_memory_attention`: Whether to use shared working memory and long-term memory. 
 If shared_memory_attention is false, then vanilla multi-head attention is used.

`mem_slots`: Number of slots in workspace

`use_long_men`: Whether to use long-term memory component. It must be True in our PMI-TR.

`long_mem_segs`: Number of long-term memory segments

`long_mem_aggre`: Whether cross-attention is performed on information retrieved from the working memory and long-term memory. If True, it will run PMI-TR $_{m}w/o2$.

`use_wm_inference`: Whether working memory come into play during the reasoning process

`seed`: Random seed

`functional`: ues Set Transformer (ISAB) or not. If True, it will run ISAB.

**Specifically, please execute the following commands to reproduce all experiments for the Sort-of-CLEVR task in the paper:**

```
PMI-TR $_{s}$
sh pmi_sort.sh 256 4 4 True True 5 True 8 True 5 True True 1 False

PMI-TR $_{m}$
sh pmi_sort.sh 256 8 8 True True 5 True 8 True 5 True True 1 False

PMI-TR $_{l}$
sh pmi_sort.sh 256 12 16 True True 5 True 8 True 5 True True 1 False

PMI-TR $_{m}w/o1$ (without memory sharing among its layers)
sh pmi_sort.sh 256 8 8 False True 5 True 8 True 5 True True 1 False

PMI-TR $_{m}w/o2$  (info retrieved from LTM is directly aggregated with data from WM via $\alpha$ without correction step.
That is,there is no Equation 11 in the paper, and in Equation 12, \(U_{wl}^t\) is changed to \(U_{l}^t\).)

sh pmi_sort.sh 256 8 8 True True 5 True 8 True 5 False True 1 False

PMI-TR $_{m}w/o3$ (without WM involvement during inference)
sh pmi_sort.sh 256 8 8 True True 5 True 8 True 5 True False 1 False

PMI-TR $_{m}$ +soft
sh pmi_sort.sh 256 8 8 True False 5 True 8 True 5 True True 1 False

TR + HSW
sh pmi_sort.sh 256 4 4 True True 5 True 8 False 5 False False 1 False

TR
sh pmi_sort.sh 256 4 4 True False 5 False 8 False 5 False False 1 False

TR + HC
sh pmi_sort.sh 256 4 4 False False 5 False 8 False 5 False False 1 False

ISAB
sh pmi_sort.sh 256 4 4 False False 5 False 8 False 5 False False 1 True

```

## Task2: bAbI
You can find the source code for the bAbI task in `sort_of_clevr_and_babi` folder.

**Specifically, please run `babi_main.py` directly.**

**Or execute the following commands to reproduce all experiments for the bAbI task in the paper:**
```
sh pmi_babi.sh h_dim num_layers num_heads share_vanilla_parameters use_topk topk shared_memory_attention mem_slots use_long_men long_mem_segs long_mem_aggre use_wm_inference seed set_transformer
```

```
PMI-TR $_{m}$
sh pmi_babi.sh 256 8 8 True True 5 True 8 True 5 True True 1 False

PMI-TR $_{m}w/o1$ 
sh pmi_babi.sh 256 8 8 False True 5 True 8 True 5 True True 1 False

PMI-TR $_{m}w/o2$ 
sh pmi_babi.sh 256 8 8 True True 5 True 8 True 5 False True 1 False

PMI-TR $_{m}w/o3$ 
sh pmi_babi.sh 256 8 8 True True 5 True 8 True 5 True False 1 False

PMI-TR $_{m}$ +soft
sh pmi_babi.sh 256 8 8 True False 5 True 8 True 5 True True 1 False

TR + HSW
sh pmi_babi.sh 256 4 4 True True 5 True 8 False 5 False False 1 False
```

## Task3: Detecting Equilateral Triangles 
You can find the source code for the Triangle task in `triangle_and_cifar10` folder.

**Specifically, please run `run.py` directly.**

**Or execute the following commands to reproduce all experiments for the Triangle task in the paper:**

```
sh run.sh dataset model patch_size num_layers h_dim ffn_dim share_vanilla_parameters use_topk topk
shared_memory_attention mem_slots use_long_men long_mem_segs long_mem_aggre use_wm_inference seed
```

```
PMI-TR
sh run.sh "Triangle" "default" 32 2 128 256 True True 5 True 8 True 5 True True 1

PMI-TR+S
sh run.sh "Triangle" "default" 32 2 128 256 True False 5 True 8 True 5 True True 1

TR + HSW
sh run.sh "Triangle" "default" 4 4 128 256 True True 5 True 8 False 5 False True 1

TR
sh run.sh "Triangle" "default" 4 4 128 256 True False 5 False 8 False 5 False True 1

STR
sh run.sh "Triangle" "default" 4 4 128 256 True True 5 False 8 False 5 False True 1

ISAB
sh run.sh "Triangle" "functional" 4 4 128 256 False False 5 False 8 False 5 False True 1
```

## Task4: Image Classification
You can find the source code for the Cifar-10 task in `triangle_and_cifar10` folder.

**Specifically, please run `run.py` directly.**

**Or execute the following commands to reproduce all experiments for the Cifar-10 task in the paper:**

```
sh run.sh dataset model patch_size num_layers h_dim ffn_dim share_vanilla_parameters use_topk topk
shared_memory_attention mem_slots use_long_men long_mem_segs long_mem_aggre use_wm_inference seed
```

**1.Trans.**
```
PMI-TR
sh run.sh "cifar10" "default" 4 4 256 256 True True 5 True 8 True 5 True True 1

TR + HSW
sh run.sh "cifar10" "default" 4 4 256 256 True True 5 True 8 False 5 False True 1

ViT
sh run.sh "cifar10" "default" 4 4 256 256 True False 5 False 8 False 5 False True 1

ISAB
sh run.sh "cifar10" "functional" 4 4 256 256 False False 5 False 8 False 5 False True 1
```
**2.Conv.**
```
CNN_MLP
sh run.sh "cifar10" "CNN_MLP" 4 4 256 256 True True 5 True 8 True 5 False True 1

CNN_PMI
sh run.sh "cifar10" "CNN_PMI" 4 4 256 256 True True 5 True 8 True 5 True True 1

CNN_PMI $w/o$ 
sh run.sh "cifar10" "CNN_PMI" 4 4 256 256 True True 5 True 8 True 5 False True 1
```