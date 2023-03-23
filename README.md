<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">ReCOGS: How Incidental Details of a Logical Form Overshadow an Evaluation of Semantic Interpretation</h3>
  <p align="center">
    Zhengxuan Wu, Christopher D. Manning, Christopher Potts
    <br />
    <a href="https://nlp.stanford.edu/~wuzhengx/"><strong>Read our preprint »</strong></a>
    <br />
    <br />
    <a href="https://github.com/frankaging/ReCOGS/issues">Report Bug</a>
    ·
    <a href="https://nlp.stanford.edu/~wuzhengx/">Contact Us</a>
  </p>
</div>

## Introduction

Compositional generalization benchmarks seek to assess whether models can accurately compute **meanings** for novel sentences, but operationalize this in terms of **logical form** (LF) prediction. This raises the concern that semantically irrelevant details of the chosen LFs could shape model performance. We argue that this concern is realized for [the COGS benchmark](https://aclanthology.org/2020.emnlp-main.731.pdf).

## Citation
If you use this repository, please consider to cite our relevant papers:
```stex
  @article{wu-etal-2023-recogs,
        title={{ReCOGS}: How Incidental Details of a Logical Form Overshadow an Evaluation of Semantic Interpretation}, 
        author={Wu, Zhengxuan and Manning, Christopher D. and Potts, Christopher},
        year={2023},
        eprint={xxxx.xxxxx},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
  }
```

## Variants of Reformatted COGS in the Paper

We produce a set of COGS dataset artifacts, each of which is a reformat / transformation of the original COGS dataset. The purpose of these variants is to study how semantic representation of COGS LFs affects model performance. If want to reproduce any of these artifacts, you can simply follow our notebook `second_looks.ipynb`.

#### COGS Token Removal
In this split, we remove redundant tokens in the logical form (e.g., `x`, `_`).

#### COGS Concat
In this split, we concatenate existing examples in the training data to study length generalization v.s. structural generalization.

#### COGS Preposing
In this split, we prepose modifier phrases to study the effects of positional indices in LFs on compositional generalization.

#### COGS Preposing + Sprinkles
In addition to the modification in the previous split, we add in interjections to allow tokens appear in different positions without affecting the semantics.

#### COGS Participle Verb
In this split, we add in an additional semantic parsing rule by augmenting the current training set. Specifically, we add in sentences with participle verbs.

#### COGS Participle Verb (easy)
In addition to the modification in the previous split, we lower the difficulty. Please refer to the paper for details. 

#### ReCOGS
For ReCOGS, we try to reduce undesired properties in COGS found in the paper, and enable COGS to measure compositional generalization more *truthfully*.

#### Variable-free COGS
This is a variant we prove in the paper to be *incorrectly* represent the original semantics. We do not recommand to use this variant. This form is proposed by [Qiu et. al., 2022](https://arxiv.org/abs/2112.07610). We use the code released by the original paper to get this form.

## Model Training

### Requirements
- Python 3.6 or 3.7 are supported.
- Pytorch Version: 1.11.0
- Transfermers Version: 4.21.1
- Datasets Version: Version: 2.3.2

### Training **Transformers**

We only have a single training script `run_cogs.py`. You can use it to reproduce our Transformers result. Here is one example,

```bash
python run_cogs.py \
--model_name ende_transformer \
--gpu 1 \
--train_batch_size 128 \
--eval_batch_size 128 \
--lr 0.0001 \
--data_path ./cogs \
--output_dir ./results_cogs \
--lfs cogs \
--do_train \
--do_test \
--do_gen \
--max_seq_len 512 \
--output_json \
--epochs 300 \
--seeds "42;66;77;88;99"
```

### Training **LSTMs**

We only have a single training script `run_cogs.py`. You can use it to reproduce our LSTMs result. Here is one example,

```bash
python run_cogs.py \
--model_name ende_lstm \
--gpu 1 \
--train_batch_size 512 \
--eval_batch_size 256 \
--lr 0.0008 \
--data_path ./cogs \
--output_dir ./results_cogs \
--lfs cogs \
--do_train \
--do_test \
--do_gen \
--max_seq_len 512 \
--output_json \
--epochs 300 \
--seeds "42;66;77;88;99"
```
