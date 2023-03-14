import random
import torch
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import sys
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import time
from utils.cogs_utils import *
from utils.second_looks_utils import *
import _pickle as cPickle
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertModel, BertConfig
from model.encoder_decoder_hf import EncoderDecoderConfig, EncoderDecoderModel
from model.encoder_decoder_lstm import EncoderDecoderLSTMModel
import pandas as pd  

torch.cuda.empty_cache()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_partition_name(name, lf):
    if lf == "cogs":
        return name
    else:
        return name+f"_{lf}"
    
class COGSTrainer(object):
    def __init__(
        self, model,
        is_master,
        src_tokenizer, 
        tgt_tokenizer, 
        device,
        logger,
        lr=5e-5,
        apex_enable=False,
        n_gpu=1,
        early_stopping=5,
        do_statistic=False,
        is_wandb=False,
        model_name="",
        eval_acc=True,
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.is_master = is_master
        self.logger = logger
        self.is_wandb = is_wandb
        self.model_name = model_name
        self.eval_acc = eval_acc
        
        self.device = device
        self.lr = lr
        self.n_gpu = n_gpu
    
        self.early_stopping = early_stopping
    
    def evaluate(
        self, eval_dataloader,
    ):
        logging.info("Evaluating ...")
        loss_sum = 0.0
        eval_step = 0
        correct_count = 0
        total_count = 0
        self.model.eval()
        for step, inputs in enumerate(eval_dataloader):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = inputs["labels"]
            outputs = self.model(**inputs)
            loss = outputs.loss.mean() if self.n_gpu > 1 else outputs.loss
            loss_sum += loss.item()
            eval_step += 1
        self.model.train()
        if total_count == 0:
            return loss_sum / eval_step, 0
        return loss_sum / eval_step, correct_count / total_count
    
    def train(
        self, train_dataloader, eval_dataloader,
        optimizer, scheduler, output_dir,
        log_step, valid_steps, epochs, 
        gradient_accumulation_steps,
        save_after_epoch
    ):
        self.model.train()
        train_iterator = trange(
            0, int(epochs), desc="Epoch"
        )
        total_step = 0
        total_log_step = 0
        patient = 0
        min_eval_loss = 100
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True)
            for step, inputs in enumerate(epoch_iterator):
                if patient == self.early_stopping:
                    logging.info("Early stopping the training ...")
                    break
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
                loss = outputs.loss.mean() if self.n_gpu > 1 else outputs.loss
                
                if total_step % log_step == 0 and self.is_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                        },
                        step=total_log_step
                    )
                    total_log_step += 1
                loss_str = round(loss.item(), 2)
                epoch_iterator.set_postfix({'loss': loss_str})
                
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                if total_step % gradient_accumulation_steps == 0:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    
                total_step += 1
                
                if valid_steps != -1 and total_step % valid_steps == 0:
                    eval_loss, eval_acc = self.evaluate(eval_dataloader)
                    logging.info(f"Eval Loss: {eval_loss}; Eval Acc: {eval_acc}")
                    if self.is_wandb:
                        wandb.log(
                            {
                                "eval/loss": eval_loss.item(),
                                "eval/acc": eval_acc,
                            },
                            step=total_step
                        )
                    if eval_loss < min_eval_loss:
                        if self.is_master:
                            if self.n_gpu > 1:
                                self.model.module.save_pretrained(os.path.join(output_dir, 'model-best'))
                            else:
                                self.model.save_pretrained(os.path.join(output_dir, 'model-best'))
                        min_eval_loss = eval_loss
                        patient = 0
                    else:
                        patient += 1
                        
            if self.is_master:
                if save_after_epoch is not None and epoch % save_after_epoch == 0:
                    dir_name = f"model-epoch-{epoch}"
                else:
                    dir_name = "model-last"
                if self.n_gpu > 1:
                    self.model.module.save_pretrained(os.path.join(output_dir, dir_name))
                else:
                    self.model.save_pretrained(os.path.join(output_dir, dir_name))
            if patient == self.early_stopping:
                break
        logging.info("Training is finished ...") 
        if self.is_master:
            if self.n_gpu > 1:
                self.model.module.save_pretrained(os.path.join(output_dir, 'model-last'))
            else:
                self.model.save_pretrained(os.path.join(output_dir, 'model-last'))

def check_equal(left_lf, right_lf):
    index_mapping = {}
    current_idx = 0
    for t in left_lf.split():
        if t.isnumeric():
            if int(t) not in index_mapping:
                index_mapping[int(t)] = current_idx
                current_idx += 1
    decoded_labels_ii = []
    for t in left_lf.split():
        if t.isnumeric():
            decoded_labels_ii += [str(index_mapping[int(t)])]
        else:
            decoded_labels_ii += [t]

    index_mapping = {}
    current_idx = 0
    for t in right_lf.split():
        if t.isnumeric():
            if int(t) not in index_mapping:
                index_mapping[int(t)] = current_idx
                current_idx += 1
    decoded_preds_ii = []
    for t in right_lf.split():
        if t.isnumeric():
            decoded_preds_ii += [str(index_mapping[int(t)])]
        else:
            decoded_preds_ii += [t]


    decoded_labels_ii_str = " ".join(decoded_labels_ii)
    decoded_preds_ii_str = " ".join(decoded_preds_ii)

    if decoded_preds_ii_str == decoded_labels_ii_str:
        return True
    return False

def check_set_equal(left_lf, right_lf):
    try:
        if translate_invariant_form(left_lf) == \
        translate_invariant_form(right_lf):
            return True
        else:
            return False
    except:
        return False
    
recogs_np_re = re.compile(r"""
    ^
    \s*(\*)?
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

recogs_pred_re = re.compile(r"""
    ^
    \s*(\w+?)\s*
    \.
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    ,
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

recogs_mod_re = re.compile(r"""
    ^
    \s*(\w+?)\s*
    \.
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    ,
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

def translate_invariant_form(lf):
    nouns = lf.split(" AND ")[0].split(" ; ")[:-1]
    complements = set(lf.split(" ; ")[-1].split())
    nouns_map = {}
    new_var = 0
    for noun in nouns:
        # check format.
        if not recogs_np_re.search(noun):
            return {} # this is format error, we cascade the error.
        _, _, original_var = recogs_np_re.search(noun).groups()
        if original_var not in complements:
            return {} # var must be used, we cascade the error.
        new_noun = noun.replace(str(original_var), str(new_var))
        nouns_map[original_var] = new_noun
        new_var += 1
    
    nmod_conjs_set = set([])
    conjs = lf.split(" ; ")[-1].split(" AND ")
    vp_conjs_map = {}
    nested_conjs = []
    childen_count_map = {}
    for conj in conjs:
        if "nmod" in conj:
            if not recogs_mod_re.search(conj):
                return {} # this is format error, we cascade the error.
            role, pred, first_arg, second_arg = recogs_mod_re.search(conj).groups()
            new_conj = f"{role} . {pred} ( {nouns_map[first_arg]} , {nouns_map[second_arg]} )"
            nmod_conjs_set.add(new_conj)
        else:
            if not recogs_pred_re.search(conj):
                return {} # this is format error, we cascade the error.
            
            role, pred, first_arg, second_arg = recogs_pred_re.search(conj).groups()
            if first_arg == second_arg or first_arg in nouns_map or not first_arg.isnumeric():
                return {} # this is index collision, we cascade the error.
            if second_arg.isnumeric() and second_arg in nouns_map:
                second_arg = nouns_map[second_arg]
                new_conj = f"{role} . {pred} ( {second_arg} )"
                if first_arg in vp_conjs_map:
                    vp_conjs_map[first_arg].append(new_conj)
                else:
                    vp_conjs_map[first_arg] = [new_conj]
            elif second_arg.isnumeric():
                if first_arg not in childen_count_map:
                    childen_count_map[first_arg] = 1
                else:
                    childen_count_map[first_arg] += 1
                nested_conjs.append({
                    "pred": pred,
                    "role": role,
                    "first_arg": first_arg,
                    "second_arg": second_arg,
                })
            else:
                new_conj = f"{role} . {pred} ( {second_arg} )"
                if first_arg in vp_conjs_map:
                    vp_conjs_map[first_arg].append(new_conj)
                else:
                    vp_conjs_map[first_arg] = [new_conj]
    
    while_loop_count = 0
    while len(nested_conjs) > 0:
        while_loop_count += 1
        if while_loop_count > 100:
            return {}
        conj = nested_conjs.pop(0)
        if conj['second_arg'] not in childen_count_map or childen_count_map[conj['second_arg']] == 0:
            core = " AND ".join(vp_conjs_map[conj['second_arg']])
            vp_conjs_map[conj['first_arg']].append(f"{conj['role']} . {conj['pred']} ( {core} )")
            childen_count_map[conj['first_arg']] -= 1
        else:
            # if the conj is corrupted, then we abandon just let it go and fail to compare.
            if conj['first_arg'] == conj['second_arg']:
                return {}
            nested_conjs.append(conj)
    
    filtered_conjs_set = set([])
    for k, v in vp_conjs_map.items():
        vp_conjs_map[k].sort()
    for k, v in vp_conjs_map.items():
        vp_expression = " AND ".join(v)
        if vp_expression in filtered_conjs_set:
            return {} # this is not allowed. exact same VP expression is not allowed this time.
        filtered_conjs_set.add(vp_expression)
    for conj in nmod_conjs_set:
        if conj in filtered_conjs_set:
            return {} # this is not allowed. exact same VP expression is not allowed this time.
        filtered_conjs_set.add(conj)
    return filtered_conjs_set