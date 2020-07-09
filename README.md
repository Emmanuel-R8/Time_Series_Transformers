## Introduction

This directory contains a Pytorch/Pytorch Lightning implementation of 
transformers applied to time series. We focus on Transformer-XL and Compressive Transformers. 

Transformer-XL is described in this paper 
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860)
by Zihang Dai\*, Zhilin Yang\*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov 
(*: equal contribution) Preprint 2018. 

Part of this code is from the authors at [https://github.com/kimiyoung/transformer-xl]().

## Prerequisite

See `requirements.txt`. All installed via `pip` instead of `conda`.

## Data Prepration

See `etf_data_prep.py` in `data/etf`.

## Files

 `modules.py` contains the description of all the components of the model. `model.py` 
 actually builds it.

Everything is built using the `pytorch-lightning` wrapper which simplifies 
generating batches, logging to various experiment tracking frameworks (e.g. `Neptune` or `WandB`).
  
`notebook_TXL.ipynb` is a notebook that can be loaded in 
[Google Colab](https://colab.research.google.com/) to run on TPUs. Note that 
the entire model and run parameters are specified in a class called `GlobalState`
defined in `utils.py` with a number of default values. 

