import argparse
import copy
import json
import os
import pickle
from datetime import datetime

import random
import numpy as np
from sklearn.metrics import classification_report, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from model import HistoricCurrent, Historic, Current
from utils.main_utils import pad_ts_collate, DepressDataset

import mindspore

import sys
sys.path.append('/sdb/nlp21/Project/physics/depression-main/')
device = 'GPU'

mindspore.context.set_context(device_target="GPU")

def true_metric_loss(true, no_of_classes, scale=1):
    batch_size = true.size
    true = true.view(batch_size,1)
    true_labels = mindspore.Tensor(true).long()
    true_labels = mindspore.ops.tile(true_labels,(1, no_of_classes)).float()
    class_labels = mindspore.ops.arange(no_of_classes)
    class_labels = class_labels.float()
    phi = (scale * mindspore.ops.abs(class_labels - true_labels))
    softmax = mindspore.nn.Softmax(axis=1)
    y = softmax(-phi)
    return y

def loss_function(output, labels, expt_type, scale):
    targets = true_metric_loss(labels, expt_type, scale)
  
    return 	mindspore.ops.sum(- targets * mindspore.ops.log_softmax(output, -1), -1).mean()

def ordinal_focal_loss(labels, one_hot, logits, alpha, gamma, scale, no_of_classes):
    targets = true_metric_loss(labels, no_of_classes, scale)
    BCLoss = mindspore.ops.binary_cross_entropy_with_logits(logits=logits, 
                                                            label=targets, 
                                                            reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = mindspore.ops.exp(-gamma * one_hot * logits - 
                                      gamma * mindspore.ops.log(1 + mindspore.ops.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = mindspore.ops.sum(weighted_loss)

    focal_loss /= mindspore.ops.sum(one_hot)
    return focal_loss

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = mindspore.ops.binary_cross_entropy_with_logits(logits=logits, 
                                                            label=labels, 
                                                            reduction="none",
                                                            weight=mindspore.Tensor(np.ones(labels.size), mindspore.float32),
                                                            pos_weight=mindspore.Tensor(np.ones(labels.size), mindspore.float32))

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = mindspore.ops.exp(-gamma * labels * logits - 
                                      gamma * mindspore.ops.log(1 + mindspore.ops.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = mindspore.ops.sum(weighted_loss)

    focal_loss /= mindspore.ops.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    weight_cls = [1e-2]*no_of_classes
    
    for i in range(no_of_classes):
        if labels[labels==i].size:
            weight_cls[i]=samples_per_cls.pop(0)
        else:
            continue


    effective_num = 1.0 - np.power(beta, weight_cls)
    weights = (1.0 - beta) / (np.array(effective_num)+1e-4)
    weights = weights / np.sum(weights) * no_of_classes

    depth, on_value, off_value = no_of_classes, mindspore.Tensor(1.0, mindspore.float32), mindspore.Tensor(0.0, mindspore.float32)
    labels_one_hot = mindspore.ops.one_hot(labels, depth, on_value, off_value).float()

    weights = mindspore.Tensor(weights, mindspore.float32)
    weights = weights.unsqueeze(0)
    weights = mindspore.ops.tile(weights, (labels_one_hot.shape[0], 1))
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = mindspore.ops.tile(weights, (1, no_of_classes))

    if loss_type == 0: #"focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = mindspore.ops.binary_cross_entropy_with_logits(logits=logits, 
                                                                 label=labels_one_hot, 
                                                                 weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = mindspore.ops.binary_cross_entropy(logits=pred, labels=labels_one_hot, weight=weights)
    elif loss_type == 1:
        no_of_class = logits.shape[-1]
        softmax = mindspore.nn.Softmax(axis=-1)
        logits = softmax(logits)
        scale = 2
        cb_loss = loss_function(logits, labels, no_of_class, scale)
    elif loss_type == 'focal_ordinary':
        no_of_class = logits.size()[-1]
        scale = 2
        cb_loss = ordinal_focal_loss(labels, labels_one_hot, logits, weights, gamma, scale, no_of_class)
    return cb_loss


def train_loop(model, dataloader, optimizer, device, dataset_len, class_num, loss_type):
    def forward_fn(labels, tweet_features, temporal_features, dataset_len, timestamp, class_num, loss_type):
        output = model(tweet_features, temporal_features, dataset_len, timestamp)
        _, inverse_indices = mindspore.ops.unique(labels) # SEE SEE DOC
        loss = loss_fn(output, labels, inverse_indices.asnumpy().tolist(), class_num, loss_type)
        return loss, output
    model.set_train()
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    running_loss = 0.0
    running_corrects = 0

    for batch, (labels, tweet_features, temporal_features, timestamp) in enumerate(dataloader.create_tuple_iterator()):
        (loss, output), grads = grad_fn(labels, tweet_features, temporal_features, dataset_len, timestamp, class_num, loss_type)
        _, preds = mindspore.ops.max(output, 1)
        optimizer(grads)


        running_loss += loss.item()
        running_corrects += (preds==labels).sum()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.asnumpy() / dataset_len

    return epoch_loss, epoch_acc


def eval_loop(model, dataloader, device, dataset_len, class_num, loss_type, optimizer):
    model.set_train(False)
    def forward_fn(labels, tweet_features, temporal_features, dataset_len, timestamp, class_num, loss_type):
        output = model(tweet_features, temporal_features, dataset_len, timestamp)
        _, inverse_indices = mindspore.ops.unique(labels)
        loss = loss_fn(output, labels, inverse_indices.asnumpy().tolist(), class_num, loss_type)
        return loss, output
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    running_loss = 0.0
    running_corrects = 0

    fin_targets = []
    fin_outputs = []

    for batch, (labels, tweet_features, temporal_features, timestamp) in enumerate(dataloader.create_tuple_iterator()):

        (loss, output), grads = grad_fn(labels, tweet_features, temporal_features, dataset_len, timestamp, class_num, loss_type)
        _, preds = 	mindspore.ops.max(output, 1)
        running_loss += loss.item()

        running_corrects += (preds==labels).sum()

        fin_targets.append(labels.asnumpy())
        fin_outputs.append(preds.asnumpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.asnumpy() / dataset_len

    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets)


def loss_fn(output, targets, samples_per_cls, class_num, loss_type):
    beta = 0.9999
    gamma = 2.0

    no_of_classes = class_num

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)

def grade_f1_score(confusion_mat):
    G_TP=0
    G_FN=0
    G_FP=0
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat)):
            if i==j:
                G_TP+=confusion_mat[i][j]
            elif i<j:
                G_FN+= confusion_mat[i][j]
            elif i>j:
                G_FP+= confusion_mat[i][j]
    G_precision = G_TP/(G_TP+G_FP)
    G_recall = G_TP/(G_TP+G_FN)
    G_f1 = 2*G_precision*G_recall/(G_precision+G_recall)
    print(f"***Grade Eval: GP:{G_precision}, GR:{G_recall}, GF:{G_f1}")
    return {'GP':G_precision, 'GR':G_recall, 'GF':G_f1}



def main(config):
    EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size

    HIDDEN_DIM = config.hidden_dim
    EMBEDDING_DIM = config.embedding_dim

    NUM_LAYERS = config.num_layer
    DROPOUT = config.dropout
    CURRENT = config.current
    LOSS_FUNC = config.loss
    RANDOM = config.random

    DATA_DIR = config.data_dir
    DATA_NAME = config.dataset
    CLASS_NUM = config.class_num
    
    SEED = config.seed

    mindspore.dataset.config.set_seed(SEED)
    mindspore.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    if config.base_model == "historic":
        model = Historic(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, CLASS_NUM)
    elif config.base_model == "current":
        model = Current(HIDDEN_DIM, DROPOUT, CLASS_NUM)
    elif config.base_model == "historic-current":
        model = HistoricCurrent(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, config.model, CLASS_NUM, device)
    else:
        assert False

    with open(os.path.join(DATA_DIR, f'data/processed_{DATA_NAME}_data.pkl'), "rb") as f:
        df_total = pickle.load(f)


    
    total_dataset = DepressDataset(df_total.label.values, df_total.curr_enc.values, df_total.enc.values,
                                    df_total.hist_dates, CURRENT, RANDOM)

    train_size = int(len(total_dataset)*0.7)
    val_size = int(len(total_dataset)*0.1)
    test_size = len(total_dataset) - train_size - val_size
    total_dataset = mindspore.dataset.GeneratorDataset(total_dataset, column_names=["labels", "tweet_features", "temporal_tweet_features", "timestamp"], num_parallel_workers=1)
    train_dataset,val_dataset, test_dataset = total_dataset.split([train_size, val_size, test_size],
                                                                  randomize = True)
    print(f'train_size:{train_size},val_size:{val_size},test_size:{test_size}')
    train_dataloader = mindspore.dataset.GeneratorDataset(train_dataset, column_names=["labels", "tweet_features", "temporal_tweet_features", "timestamp"], shuffle=True, num_parallel_workers=1)
    print(len(train_dataloader))
    train_dataloader = train_dataloader.batch(BATCH_SIZE, per_batch_map=pad_ts_collate)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("BATCH_NUM:", len(train_dataloader))
    val_dataloader = mindspore.dataset.GeneratorDataset(val_dataset, column_names=["labels", "tweet_features", "temporal_tweet_features", "timestamp"],shuffle=True, num_parallel_workers=1)
    val_dataloader = val_dataloader.batch(BATCH_SIZE, per_batch_map=pad_ts_collate)
    test_dataloader = mindspore.dataset.GeneratorDataset(test_dataset, column_names=["labels", "tweet_features", "temporal_tweet_features", "timestamp"], num_parallel_workers=1)
    test_dataloader = test_dataloader.batch(BATCH_SIZE, per_batch_map=pad_ts_collate)


    LEARNING_RATE = config.learning_rate

    optimizer = mindspore.nn.optim_ex.AdamW(model.trainable_params(), lr=LEARNING_RATE)
    model_name = f'{int(datetime.timestamp(datetime.now()))}_{config.base_model}_{config.model}_{config.hidden_dim}_{config.num_layer}_{config.learning_rate}'

    best_metric = 0.0

    for epoch in range(EPOCHS):
        loss, accuracy = train_loop(model, train_dataloader, optimizer, device, len(train_dataset), CLASS_NUM, LOSS_FUNC)
        eval_loss, eval_accuracy, __, _ = eval_loop(model, val_dataloader, device, len(val_dataset), CLASS_NUM, LOSS_FUNC, optimizer)

        metric = f1_score(_, __, average="macro")
        recall = recall_score(_, __, average="macro")
        confusion = confusion_matrix(_, __, labels=[0, 1, 2, 3])
        grade_rep = grade_f1_score(confusion)

        print(
            f'epoch {epoch + 1}:: train: loss: {loss}, accuracy: {accuracy:.4f} | valid: loss: {eval_loss}, accuracy: {eval_accuracy:.4f}, f1: {metric}, recall: {recall}')
        if metric > best_metric:
            best_metric = metric


    if not os.path.exists('saved_model'):
        os.mkdir("saved_model")

    _, _, y_pred, y_true = eval_loop(model, val_dataloader, device, len(val_dataset), CLASS_NUM, LOSS_FUNC, optimizer)

    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True)

    result = {'best_f1': best_metric.item(),
              'lr': LEARNING_RATE,
              'model': str(model),
              'optimizer': str(optimizer),
              'base-model': config.base_model,
              'model-name': config.model,
              'epochs': EPOCHS,
              'embedding_dim': EMBEDDING_DIM,
              'hidden_dim': HIDDEN_DIM,
              'num_layers': NUM_LAYERS,
              'dropout': DROPOUT,
              'current': CURRENT,
              'loss':LOSS_FUNC,
              'val_report': report}

    if config.test:
        _, _, y_pred, y_true = eval_loop(model, test_dataloader, device, len(test_dataset), CLASS_NUM, LOSS_FUNC, optimizer)
        confusion = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        print(confusion)

        report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True)
        print(report)
        grade_rep = grade_f1_score(confusion)
        result['test_report'] = report
        result['grade_report'] = grade_rep

        with open(os.path.join(DATA_DIR, f'checkpoints/saved_model/TEST_{model_name}.json'), 'w') as f:
            json.dump(result, f, indent=2)




if __name__ == '__main__':
    mindspore.context.set_context(device_target="CPU")
    base_model_set = {"historic", "historic-current", "current"}
    model_set = {"tlstm", "bilstm", "bilstm-attention"}
    loss_set = {"focal","softmax","sigmoid","ordinary"}
    dataset_set = {"reddit","depression","5c"}

    parser = argparse.ArgumentParser(description="Temporal Suicidal Modelling")
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-bs", "--batch_size", default=256, type=int)
    parser.add_argument("-e", "--epochs", default=200, type=int)
    parser.add_argument("-hd", "--hidden_dim", default=1024, type=int)
    parser.add_argument("-ed", "--embedding-dim", default=768, type=int)
    parser.add_argument("-n", "--num_layer", default=2, type=int)
    parser.add_argument("-cn", "--class_num", default=4, type=int)
    parser.add_argument("-d", "--dropout", default=0.2, type=float)
    parser.add_argument("--base_model", type=str, choices=base_model_set, default="current")
    parser.add_argument("--loss", type=str, choices=base_model_set, default=1)
    parser.add_argument("--dataset", type=str, choices=base_model_set, default="reddit") # de2
    parser.add_argument("--model", type=str, choices=model_set, default="tlstm") # bilstm
    parser.add_argument("-t", "--test", action="store_true", default=True)
    parser.add_argument("--current", action="store_false")
    parser.add_argument("--data_dir", type=str, default="/mnt/LZZ")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", default=1895, type=int)
    config = parser.parse_args()

    main(config)

