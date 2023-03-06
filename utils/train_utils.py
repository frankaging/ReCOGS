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