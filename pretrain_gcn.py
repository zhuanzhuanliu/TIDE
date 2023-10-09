import mindspore

import os
import networkx as nx
import numpy as np
from argparse import ArgumentParser
import logging
import json

from utils.pretrain_gcn_utils import load_pickle, save_as_pickle, generate_text_graph

import sys
sys.path.append('/sdb/nlp21/Project/physics/depression-main/')

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

class gcn(mindspore.nn.Cell):
    def __init__(self, X_size, A_hat, args, bias=True): # X_size = num features
        super(gcn, self).__init__()
        self.A_hat = mindspore.Tensor(A_hat, mindspore.float32)
        self.weight = mindspore.Parameter(mindspore.Tensor((X_size, args.hidden_size_1), mindspore.float32))
        var = 2./(self.weight.size(1)+self.weight.size(0))
        self.weight.data.normal_(0,var)
        self.weight2 = mindspore.Parameter(mindspore.Tensor((args.hidden_size_1, args.hidden_size_2), mindspore.float32))
        var2 = 2./(self.weight2.size(1)+self.weight2.size(0))
        self.weight2.data.normal_(0,var2)
        if bias:
            self.bias = mindspore.Parameter(mindspore.Tensor(args.hidden_size_1, mindspore.float32))
            self.bias.data.normal_(0,var)
            self.bias2 = mindspore.Parameter(mindspore.Tensor(args.hidden_size_2, mindspore.float32))
            self.bias2.data.normal_(0,var2)
        else:
            self.register_parameter("bias", None)
        self.fc1 = mindspore.nn.Dense(args.hidden_size_2, args.num_classes)
        self.dropout = mindspore.nn.Dropout(p=args.dropout)
        
    def forward(self, X): ### 2-layer GCN architecture
        X = mindspore.ops.mm(X, self.weight)
        if self.bias is not None:
            X = (X + self.bias)
        X = self.dropout(X)
        X = mindspore.ops.relu(mindspore.ops.mm(self.A_hat, X))
        X = mindspore.ops.mm(X, self.weight2)
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = self.dropout(X)
        X = mindspore.ops.relu(mindspore.ops.mm(self.A_hat, X))
        return self.fc1(X)

    def get_state(self, X):
        X = mindspore.ops.mm(X, self.weight)
        if self.bias is not None:
            X = (X + self.bias)
        X = mindspore.ops.relu(mindspore.ops.mm(self.A_hat, X))
        X = mindspore.ops.mm(X, self.weight2)
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = mindspore.ops.relu(mindspore.ops.mm(self.A_hat, X))

        return X

def load_datasets(args, extra_name):
    """Loads dataset and graph if exists, else create and process them from raw data
    Returns --->
    f: torch tensor input of GCN (Identity matrix)
    X: input of GCN (Identity matrix)
    A_hat: transformed adjacency matrix A
    selected: indexes of selected labelled nodes for training
    test_idxs: indexes of not-selected nodes for inference/testing
    labels_selected: labels of selected labelled nodes for training
    labels_not_selected: labels of not-selected labelled nodes for inference/testing
    """
    logger.info("Loading data...")
    df_data_path = f"/sdb/nlp21/Project/physics/depression-main/data/{extra_name}_df_data.pkl"
    graph_path = f"/sdb/nlp21/Project/physics/depression-main/data/{extra_name}_text_graph.pkl"
    if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):
        logger.info("Building datasets and graph from raw data... Note this will take quite a while...")
        generate_text_graph(args.path)
    df_data = load_pickle(df_data_path)
    G = load_pickle(graph_path)
    
    logger.info("Building adjacency and degree matrices...")
    A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes()) # Features are just identity matrix
    A_hat = degrees@A@degrees
    f = X # (n X n) X (n X n) x (n X n) X (n X n) input of net
    
    logger.info("Splitting labels for training and inferring...")
    ### stratified test samples
    test_idxs = []
    for b_id in df_data["l"].unique():
        dum = df_data[df_data["l"] == b_id]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(args.test_ratio*len(dum)), replace=False)))
    selected = []
    for i in range(len(df_data)):
        if i not in test_idxs:
            selected.append(i)
    
    f_selected = f[selected]; f_selected = mindspore.Tensor.from_numpy(f_selected).float()
    labels_selected = [l for idx, l in enumerate(df_data["l"]) if idx in selected]
    f_not_selected = f[test_idxs]; f_not_selected = mindspore.Tensor.from_numpy(f_not_selected).float()
    labels_not_selected = [l for idx, l in enumerate(df_data["l"]) if idx not in selected]
    f = mindspore.Tensor.from_numpy(f).float()
    logger.info("Split into %d train and %d test lebels." % (len(labels_selected), len(labels_not_selected)))
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs
    
def load_state(net, model_no=0, load_best=False):
    """ Loads saved model and optimizer states if exists """
    logger.info("Initializing model and optimizer states...")
    base_path = "/sdb/nlp21/Project/physics/depression-main/checkpoints/pretrain_gcn"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.ckpt" % model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.ckpt" % model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = mindspore.load_checkpoint(best_path)
        mindspore.load_param_into_net(net, checkpoint[1]['state_dict'])
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = mindspore.load_checkpoint(checkpoint_path)
        mindspore.load_param_into_net(net, checkpoint[1]['state_dict'])
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint[0]['epoch']
        best_pred = checkpoint[2]['best_acc']
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "/sdb/nlp21/Project/physics/depression-main/data/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "/sdb/nlp21/Project/physics/depression-main/data/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch

def evaluate(output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum([e for e in labels_e] == labels)/len(labels)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hidden_size_1", type=int, default=600, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=300, help="Size of second GCN hidden weights")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of prediction classes")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test to training nodes")
    parser.add_argument("--path", default='/sdb/nlp21/Project/physics/depression-main/data/encoded_GCN_partial_data.csv', help="Path of Graph")
    parser.add_argument("--num_epochs", type=int, default=300, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--model_no", type=int, default=5, help="Model ID")
    args = parser.parse_args()
    
    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets(args, extra_name='depression')
    net = gcn(X.shape[1], A_hat, args)
    criterion = mindspore.nn.CrossEntropyLoss()
    scheduler = mindspore.nn.piecewise_constant_lr(milestones=[1000,2000,3000,4000,5000,6000], lr=args.lr)

    def forward_fn(data, label):
        output = net(data)
        loss = criterion(output[selected], label)
        return loss, output

    grad_fn = mindspore.value_and_grad(forward_fn, None, scheduler.parameters, has_aux=True)

    def train_step(data, label):
        (loss, output), grads = grad_fn(data, label)
        scheduler(grads)
        return loss, output

    start_epoch, best_pred = load_state(net, model_no=args.model_no, load_best=False)
    losses_per_epoch, evaluation_untrained = load_results(model_no=args.model_no)
    evaluation_trained = []
    for epoch in range(args.num_epochs):
        print(f"Epoch gcn {epoch+1}\n-------------------------------")
        net.set_train()
        for batch, (f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs) in enumerate(f.create_tuple_iterator()):
            loss, output = train_step(f, mindspore.Tensor(labels_selected).long())
            losses_per_epoch.append(loss.item())
            if epoch % 10 == 0:
                net.set_train(False)
                pred_labels = net(f)
                trained_accuracy = evaluate(output[selected], labels_selected)
                untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
                evaluation_trained.append((epoch, trained_accuracy))
                evaluation_untrained.append((epoch, untrained_accuracy))
                print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (epoch, trained_accuracy))
                print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (epoch, untrained_accuracy))
                print("Labels of trained nodes: \n", output[selected].max(1)[1])
                net.set_train(True)
                if trained_accuracy > best_pred:
                    best_pred = trained_accuracy
                    mindspore.save_checkpoint([
                        {'name': 'epoch',
                         'data' : epoch + 1},
                        {'name': 'state_dict',
                         'data' : net},
                        {'name': 'best_acc',
                         'data' : trained_accuracy}
                    ], os.path.join("/sdb/nlp21/Project/physics/depression-main/checkpoints/pretrain_gcn",
                                    "test_model_best_%d.ckpt" % args.model_no))
            if (e % 10) == 0:
                save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
                save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, evaluation_untrained)
                mindspore.save_checkpoint([
                        {'name': 'epoch',
                         'data' : epoch + 1},
                        {'name': 'state_dict',
                         'data' : net},
                        {'name': 'best_acc',
                         'data' : trained_accuracy}
                    ], os.path.join("/sdb/nlp21/Project/physics/depression-main/checkpoints/pretrain_gcn",
                                    "test_checkpoint_%d.ckpt" % args.model_no))
    evaluation_trained = np.array(evaluation_trained)
    evaluation_untrained = np.array(evaluation_untrained)
     