import os, sys

# Main entrance from https://github.com/MysteryVaibhav/RWR-GAE
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim
import torch.nn.functional as F
# from model import GCNModelVAE, GCNModelAE
from optimizer import loss_function
from utils import mask_test_edges, get_roc_score
from sklearn.cluster import KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm
from preprocessing import load_data, sparse_to_tuple, preprocess_graph
from model import ReGMM_VGAE


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--dw', type=int, default=1, help="whether to use deepWalk regularization, 0/1")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--walk-length', default=5, type=int, help='Length of the random walk started at each node')
parser.add_argument('--window-size', default=3, type=int, help='Window size of skipgram model.')
parser.add_argument('--number-walks', default=5, type=int, help='Number of random walks to start at each node')
parser.add_argument('--full-number-walks', default=0, type=int, help='Number of random walks from each node')
parser.add_argument('--lr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
parser.add_argument('--context', type=int, default=0, help="whether to use context nodes for skipgram")
parser.add_argument('--ns', type=int, default=1, help="whether to use negative samples for skipgram")
parser.add_argument('--n-clusters', default=7, type=int, help='number of clusters, 7 for cora, 6 for citeseer')
parser.add_argument('--plot', type=int, default=0, help="whether to plot the clusters using tsne")
parser.add_argument('--precisionModel', type=str, default='Float', 
                    help='Single Precision/Double precision: Float/Double (default:Float)')

# parameters for R-GMM-VGAE
parser.add_argument('--num_neurons', type=int, default=32, help="number of neurons in the hidden layer")
parser.add_argument('--embedding_size', type=int, default=16, help="dimension of the embedding")
parser.add_argument('--beta1', type=float, default=0.3)
parser.add_argument('--beta2', type=float, default=0.15)
args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(args.dataset_str)
    num_nodes, num_features = features.shape
    nClusters = args.n_clusters

    # Data processing 
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm = preprocess_graph(adj)
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2]))
    weight_mask_orig = adj_label.to_dense().view(-1) == 1
    weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
    weight_tensor_orig[weight_mask_orig] = pos_weight_orig

    # Store original adjacency matrix (without diagonal entries) for later
    # adj_orig = adj
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()

    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # adj = adj_train

    # Some preprocessing
    # adj_norm = preprocess_graph(adj)
    # adj_label = adj_train + sp.eye(adj_train.shape[0])
    # # adj_label = sparse_to_tuple(adj_label)
    # # adj_label = torch.DoubleTensor(adj_label.toarray())
    # adj_label = torch.FloatTensor(adj_label.toarray())

    # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # if args.model == 'gcn_vae':
    #     model = GCNModelVAE(num_features, args.hidden1, args.hidden2, args.dropout)
    # else:
    #     model = GCNModelAE(num_features, args.hidden1, args.hidden2, args.dropout)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    network = ReGMM_VGAE(adj = adj_norm , num_neurons=args.num_neurons, num_features=num_features, embedding_size=args.embedding_size, nClusters=nClusters, activation="ReLU")
    #network.pretrain(adj_norm, features, adj_label, labels, weight_tensor_orig, norm , epochs=epochs_pretrain, lr=lr_pretrain, save_path=save_path, dataset=dataset)
    network.train(adj_norm, adj,  features, true_labels , norm, epochs=args.epochs, lr=args.lr, beta1=args.beta1, beta2=args.beta2, save_path=save_path, dataset=args.dataset_str)

    hidden_emb = None
    for epoch in tqdm(range(args.epochs)):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        z, mu, logvar = model(features, adj_norm)

        loss = loss_function(preds=model.dc(z), labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=num_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        
        tqdm.write("Epoch: {}, train_loss_gae={:.5f}, val_ap={:.5f}, time={:.5f}".format(
            epoch + 1, cur_loss,
            ap_curr, time.time() - t))

        if (epoch + 1) % 10 == 0:
            tqdm.write("Evaluating intermediate results...")
            kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
            predict_labels = kmeans.predict(hidden_emb)
            cm = clustering_metrics(true_labels, predict_labels)
            cm.evaluationClusterModelFromLabel(tqdm)
            roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            tqdm.write('ROC: {}, AP: {}'.format(roc_score, ap_score))
            np.save('logs/emb_epoch_{}.npy'.format(epoch + 1), hidden_emb)

    tqdm.write("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    tqdm.write('Test ROC score: ' + str(roc_score))
    tqdm.write('Test AP score: ' + str(ap_score))
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
    predict_labels = kmeans.predict(hidden_emb)
    cm = clustering_metrics(true_labels, predict_labels)
    cm.evaluationClusterModelFromLabel(tqdm)

    if args.plot == 1:
        cm.plotClusters(tqdm, hidden_emb, true_labels)


if __name__ == '__main__':
    gae_for(args)