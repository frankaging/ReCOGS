'''
Reference:
https://github.com/marumalo/pytorch-seq2seq/blob/master/model.py
'''

# -*- coding: utf-8 -*-

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel

class EncRNN(nn.Module):
    def __init__(self, vsz, embed_dim, hidden_dim, n_layers, use_birnn, dout, pad_token_id):
        super(EncRNN, self).__init__()
        self.embed = nn.Embedding(vsz, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           bidirectional=use_birnn)
        self.dropout = nn.Dropout(dout)
        self.pad_token_id = pad_token_id
    def forward(self, inputs, attention_mask=None):
        padded_sequence = self.dropout(self.embed(inputs))
        total_length = padded_sequence.shape[0]
        if attention_mask is not None:
            lens = attention_mask.sum(dim=0).tolist()
            packed_sequence = nn.utils.rnn.pack_padded_sequence(padded_sequence, lens, enforce_sorted=False)
            packed_output, hidden = self.rnn(packed_sequence)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output,
                                                               total_length=total_length,
                                                               padding_value=self.pad_token_id)
        else:
            output, hidden = self.rnn(packed_sequence)
        return self.dropout(output), (hidden[0], hidden[1])


class Attention(nn.Module):
    def __init__(self, hidden_dim, method):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim*2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, dec_out, enc_outs, attention_mask=None):
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(dec_out, enc_outs)
        if attention_mask is not None:
            attn_energies = attn_energies.masked_fill(attention_mask == 0, -10000)
        return F.softmax(attn_energies, dim=0)

    def dot(self, dec_out, enc_outs):
        return torch.sum(dec_out*enc_outs, dim=2)

    def general(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out*energy, dim=2)

    def concat(self, dec_out, enc_outs):
        dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((dec_out, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)


class DecRNN(nn.Module):
    def __init__(self, vsz, embed_dim, hidden_dim, n_layers, use_birnn, 
                 dout, attn='dot', tied=True):
        super(DecRNN, self).__init__()
        hidden_dim = hidden_dim*2 if use_birnn else hidden_dim
        
        self.embed = nn.Embedding(vsz, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim , n_layers)

        self.w = nn.Linear(hidden_dim*2, hidden_dim)
        self.attn = Attention(hidden_dim, attn)

        self.out_projection = nn.Linear(hidden_dim, vsz)
        if tied: 
            if embed_dim != hidden_dim:
                raise ValueError(
                    f"when using the tied flag, embed-dim:{embed_dim} \
                    must be equal to hidden-dim:{hidden_dim}")
            self.out_projection.weight = self.embed.weight
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs, hidden, enc_outs, attention_mask=None):
        inputs = inputs.unsqueeze(0)
        embs = self.dropout(self.embed(inputs))
        dec_out, hidden = self.rnn(embs, hidden)
        attn_weights = self.attn(dec_out, enc_outs, attention_mask).transpose(1, 0)
        enc_outs = enc_outs.transpose(1, 0)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outs)
        cats = self.w(torch.cat((dec_out, context.transpose(1, 0)), dim=2))
        pred = self.out_projection(cats.tanh().squeeze(0))
        return pred, hidden


class EncoderDecoderLSTMModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.decoder_start_token_id = config.decoder_start_token_id
        self.pad_token_id = config.pad_token_id
        self.src_vsz = config.encoder.vocab_size
        self.tgt_vsz = config.decoder.vocab_size
        self.use_birnn = False
        
        self.encoder = EncRNN(self.src_vsz, config.encoder.hidden_size, 
                              config.encoder.hidden_size, 
                              config.encoder.num_hidden_layers, self.use_birnn, 0.3,
                              pad_token_id=self.pad_token_id)
        self.decoder = DecRNN(self.tgt_vsz, config.decoder.hidden_size, config.decoder.hidden_size, 
                              config.decoder.num_hidden_layers, self.use_birnn, 0.3)
        self.n_layers = config.encoder.num_hidden_layers
        self.hidden_dim = config.encoder.hidden_size
        


    def forward(self, input_ids, labels=None, attention_mask=None, maxlen=512, tf_ratio=1.0):
        srcs = input_ids.transpose(0,1)
        tgts = labels.transpose(0,1) if labels is not None else None
        attention_mask = attention_mask.transpose(0,1) if attention_mask is not None else None
        
        slen, bsz = srcs.size()
        tlen = tgts.size(0) if isinstance(tgts, torch.Tensor) else maxlen
        tf_ratio = tf_ratio if isinstance(tgts, torch.Tensor) else 0.0
       
        enc_outs, hidden = self.encoder(srcs, attention_mask=attention_mask)

        dec_inputs = torch.ones_like(srcs[0]) * self.decoder_start_token_id # <eos> is mapped to id=2
        outs = []

        if self.use_birnn:
            def trans_hidden(hs):
                hs = hs.view(self.n_layers, 2, bsz, self.hidden_dim)
                hs = torch.stack([torch.cat((h[0], h[1]), 1) for h in hs])
                return hs
            hidden = tuple(trans_hidden(hs) for hs in hidden)
            
            
        for i in range(tlen):
            preds, hidden = self.decoder(dec_inputs, hidden, enc_outs, attention_mask=attention_mask)
            outs.append(preds)
            use_tf = random.random() < tf_ratio
            dec_inputs = tgts[i] if use_tf else preds.max(1)[1]
        outs = torch.stack(outs)
        loss = None
        if tgts != None:
            loss_fct = CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(
                outs.reshape(-1, self.tgt_vsz), 
                tgts.reshape(-1), 
            )

            return Seq2SeqLMOutput(
                loss=loss,
                logits=outs,
            )
        
        return outs

    def generate(self, input_ids, labels=None, attention_mask=None, maxlen=512, tf_ratio=1.0):
        srcs = input_ids.transpose(0,1)
        tgts = labels.transpose(0,1) if labels is not None else None
        attention_mask = attention_mask.transpose(0,1) if attention_mask is not None else None
        
        slen, bsz = srcs.size()
        tlen = tgts.size(0) if isinstance(tgts, torch.Tensor) else maxlen
        tf_ratio = tf_ratio if isinstance(tgts, torch.Tensor) else 0.0
       
        enc_outs, hidden = self.encoder(srcs, attention_mask=attention_mask)

        dec_inputs = torch.ones_like(srcs[0]) * self.decoder_start_token_id # <eos> is mapped to id=2
        outs = []

        if self.use_birnn:
            def trans_hidden(hs):
                hs = hs.view(self.n_layers, 2, bsz, self.hidden_dim)
                hs = torch.stack([torch.cat((h[0], h[1]), 1) for h in hs])
                return hs
            hidden = tuple(trans_hidden(hs) for hs in hidden)
            
            
        for i in range(tlen):
            preds, hidden = self.decoder(dec_inputs, hidden, enc_outs, attention_mask=attention_mask)
            outs.append(preds)
            use_tf = random.random() < tf_ratio
            dec_inputs = tgts[i] if use_tf else preds.max(1)[1]
        outs = torch.stack(outs)
        outs = outs.argmax(dim=-1)
        
        return outs.transpose(0,1)
    