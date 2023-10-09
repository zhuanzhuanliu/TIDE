from transformers import BertTokenizer, BertForMaskedLM


import mindspore

import random
import os
import argparse
import time
import logging
import json
from tqdm import tqdm
import numpy as np

from utils.pretrain_bert_utils import BERT_dataset,train_collate_fn

import setproctitle
setproctitle.setproctitle('pretrain_BERT_rep')

class BERT_model(object):
    def __init__(self, config):
        self.cfg = config
        self.device=self.cfg.device
        if self.cfg.mode == 'eval':
            self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model_path)
            self.model=BertForMaskedLM.from_pretrained(self.cfg.model_path)
            print("Loading finetuned bert paras!")
        elif self.cfg.mode == 'train':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            self.model=BertForMaskedLM.from_pretrained("bert-base-cased")
            print("Loading pretrained bert paras!")
        self.model.to(self.device)

    def forward_fn(model, data, label):
        output = model(data, label)
        loss = output['loss']
        return loss, output
    
    def train(self, extra_name=''):
        train_data=BERT_dataset(self.cfg.data_path, self.tokenizer, extra_name)
        dev_data=BERT_dataset(self.cfg.data_path, self.tokenizer, extra_name)
        train_dataloader=mindspore.dataset.GeneratorDataset(train_data, shuffle=True)
        train_dataloader = train_dataloader.batch(self.cfg.batch_size, per_batch_map=train_collate_fn)
        dev_dataloader=mindspore.dataset.GeneratorDataset(dev_data)
        dev_dataloader = dev_dataloader.batch(self.cfg.eval_batch_size, per_batch_map=train_collate_fn)

        global_step = 0
        min_loss=10000
        last_loss=10000

        optimizer = mindspore.nn.optim_ex.AdamW(self.model.trainable_params(), lr=self.cfg.learning_rate)

        grad_fn = mindspore.value_and_grad(self.forward_fn, None, optimizer.parameters, has_aux=True)

        for epoch in range(self.cfg.epoch_num):
            tr_loss = 0.0
            step_loss=0
            btm = time.time()
            oom_time = 0
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
            print("Epoch:{}".format(epoch))  
            for batch_idx, batch in pbar:
                try:  # avoid OOM
                    self.model.set_train()
                    inputs=batch.to(self.device) #B, T
                    labels=inputs
                    (loss, outputs), grads = grad_fn(self.model, inputs, labels)
                    loss = outputs['loss']
                    loss=loss/self.cfg.gradient_accumulation_steps
                    optimizer(grads)

                    tr_loss += loss.item()
                    step_loss+=loss.item()
                    mindspore.ops.clip_by_global_norm(self.model.parameters(), 5.0)
                    if (batch_idx+1) % self.cfg.gradient_accumulation_steps == 0 or batch_idx+1==len(train_dataloader):
                        global_step += 1

                        step_loss=0

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        print("WARNING: ran out of memory,times: {}".format(oom_time))
                    else:
                        print(str(exception))
                        raise exception
                     
            print('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss))
            current_loss = tr_loss
            if last_loss > current_loss:
                self.save_model(name='reddit_best_model.ckpt')
                last_loss = current_loss

    
    def eval(self, dataset):
        self.model.set_train(False)
        total_loss=0
        for batch in dataset:
            inputs=batch.to(self.device)
            labels=inputs
            outputs = self.model(inputs, labels)
            loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
            total_loss+=loss.item()
        return total_loss/len(dataset)
        


    def save_model(self, name):
        save_path = os.path.join(self.cfg.exp_path, name)
        mindspore.save_checkpoint([
                        {'name': 'model',
                         'data' : self.model},
                        {'name': 'tokenizer',
                         'data' : self.tokenizer}
                    ], save_path)
        print(f"model has been saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1895)
    parser.add_argument('--pad_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    # TODO
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data_path', default='/sdb/nlp21/Project/physics/depression-main/data/total_reddit_data.json')
    parser.add_argument('--exp_path', default='/sdb/nlp21/Project/physics/depression-main/checkpoints/pretrain_bert')
    parser.add_argument('--model_path', default=None)


    cfg = parser.parse_args()

    mindspore.dataset.config.set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = BERT_model(cfg)
    model.train(extra_name='reddit')

