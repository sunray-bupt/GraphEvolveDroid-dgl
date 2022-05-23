import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import datetime
import sklearn.metrics as skm
import os

from model import GED
from dataset import EvolutionaryNet

def train_val_split(g):
    """Split the graph into training graph, validation graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['val_mask'])
    return train_g, val_g

def compute_metrics(pred, labels, detailed=False):
    """
    Compute the metrics of prediction given the labels.
    """
    acc = skm.accuracy_score(labels, pred)
    f1 = skm.f1_score(labels, pred)
    precision = skm.precision_score(labels, pred)
    recall = skm.recall_score(labels, pred)
    if detailed:
        print("Confusion_Matrix:\n{}".format(skm.confusion_matrix(labels, pred)))
    return acc, f1, precision, recall

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device, detailed=False):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    pred = pred[val_nid].cpu().data.numpy().argmax(axis=1)
    labels = labels[val_nid].data.numpy()
    return compute_metrics(pred, labels, detailed)

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = [nfeat[node].to_dense() for node in input_nodes]
    batch_inputs = th.stack(batch_inputs)
    batch_inputs = batch_inputs.to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
    # Unpack data
    n_classes, g, train_g, val_g, train_nfeat, train_labels, \
    val_nfeat, val_labels = data
    in_feats = train_nfeat.shape[1]
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

    start = datetime.datetime.now()
    dataloader_device = th.device('cpu')
    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=dataloader_device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    end = datetime.datetime.now()
    print("Total sampling time for training: %s (with %02d num_workers)" % (end - start, args.num_workers))

    # Define model and optimizer
    model = GED(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    # Hyperparameter weight (negative(0) : positive(1) = 1 : 5)
    loss_fcn = nn.CrossEntropyLoss(weight=th.Tensor([1, 5]).to(device))
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    avg = 0
    iter_tput = []
    hist_loss = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        batch_loss = []
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels.type(th.int64))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()
            batch_loss.append(loss.item())

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        hist_loss.append(sum(batch_loss) / len(batch_loss))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, eval_f1, eval_p, eval_r = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device, args.detailed)
            print('Eval Acc {:.4f} | Eval F1: {:.4f} | Eval Precision: {:.4f} | Eval Recall: {:.4f}'.format(eval_acc, eval_f1, eval_p, eval_r))
    
    if epoch >= 5:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    # Testing
    start = datetime.datetime.now()
    metrics = []
    for period, test_g, test_nid in test_split(g):
        test_nfeat = test_g.ndata.pop('features')
        test_labels = test_g.ndata.pop('labels')
        test_acc, test_f1, test_p, test_r = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device, args.detailed)
        print('Test period: {} | Test Acc: {:.4f} | Test F1: {:.4f} | Test Precision: {:.4f} | Test Recall: {:.4f}'.format(period, test_acc, test_f1, test_p, test_r))
        metrics.append([period, test_acc, test_f1, test_p, test_r])
    end = datetime.datetime.now()
    print("Total testing time (split testing dataset & inference): %s" % (end - start))

    # Save model
    start = datetime.datetime.now()
    save_path = "./checkpoints"
    checkpoint = save_model(args, save_path, model, hist_loss, metrics)
    end = datetime.datetime.now()
    print("Saving model in %s (consume time: %s)" % (checkpoint, end - start))

def compute_aut_metrics(metrics):
    """
    Compute AUT version of each metrics
    """
    # Delete 'period' field for each row
    metrics = [metric[1:] for metric in metrics]
    metrics = np.array(metrics, dtype=np.float32)
    norm = metrics.shape[0] * 2 - 2
    aut_metrics = metrics[1:-1].sum(axis=0) + metrics.sum(axis=0)
    return aut_metrics / norm

def save_model(args, save_path, model, loss, metrics):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name = model.__class__.__name__
    checkpoint_name = timestamp + "_" + model_name
    checkpoint_path = os.path.join(save_path, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Save model state_dict
    if args.save_model:
        th.save(model.state_dict(), os.path.join(checkpoint_path, model_name + "-model.pt"))

    # Save model loss & metrics
    hist_records = dict()
    hist_records['loss'] = loss
    hist_records['metrics'] = metrics
    th.save(hist_records, os.path.join(checkpoint_path, "historyRecords.pt"))

    # Append reports
    report_file = os.path.join(save_path, "reports.csv")
    aut_acc, aut_f1, aut_p, aut_r = compute_aut_metrics(metrics)
    with open(report_file, 'a') as f:
        f.write("{}\n".format(args))
        f.write("{}\tAUT_acc {:.4f}\tAUT_F1 {:.4f}\tAUT_P {:.4f}\tAUT_R {:.4f}\n".format(checkpoint_name, aut_acc, aut_f1, aut_p, aut_r))

    return checkpoint_name

def test_split(g):
    """
    Create a generator to yield testing graph on each month from 2015-2016 year by splitting the graph according to test masks
    """
    # for year in ['2013']:
    #     for month in range(1, 13):
    # for year in ['2015']:
    #     for month in range(1, 4):
    for year in ['2015', '2016']:
        for month in range(1, 13):
            period = year + "-" + str(month)
            test_mask = "test_mask_" + period
            test_g = g.subgraph(g.ndata[test_mask])
            test_nid = th.nonzero(test_g.ndata[test_mask], as_tuple=True)[0]
            yield period, test_g, test_nid

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    # argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument('--num-hidden', type=int, default=200)
    argparser.add_argument('--num-layers', type=int, default=2) # GraphSAGE layers
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--weight-decay', type=float, default=1e-5)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    # argparser.add_argument('--inductive', action='store_true',
    #                        help="Inductive learning setting")
    argparser.add_argument('--detailed', action='store_true', 
                            help="Print confusion matrix during inference process.")
    argparser.add_argument('--save-model', action='store_true', 
                            help="By default the script does not save model state_dict "
                                 "for disk space consumption. This may be undesired if you "
                                 "want to recover the model for inference. "
                                 "This flag disables that.")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()
    print()
    first_start = datetime.datetime.now()
    print(first_start)
    print(args)

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    start = datetime.datetime.now()
    # Dataset directory. For a more detailed explanation see README.md
    # data_dir = "/home/sunrui/data/apigraph/vocabulary-2012"
    data_dir = "/home/sunrui/data/GraphEvolveDroid"
    feat_mtx = "drebin_feat_mtx.npz"
    adj_mtx = "drebin_knn_5.npz"
    dataset = EvolutionaryNet(data_dir, feat_mtx, adj_mtx)
    g = dataset[0]
    end = datetime.datetime.now()
    print("Loading dataset time: %s" % (end - start))
    n_classes = 2

    start = datetime.datetime.now()
    train_g, val_g = train_val_split(g)
    train_nfeat = train_g.ndata.pop('features')
    val_nfeat = val_g.ndata.pop('features')
    train_labels = train_g.ndata.pop('labels')
    val_labels = val_g.ndata.pop('labels')
    end = datetime.datetime.now()
    print("Splitting training and validation dataset time: %s" % (end - start))

    print("train_nfeat info: ", type(train_nfeat), train_nfeat.dtype, train_nfeat.shape, train_nfeat.device)
    start = datetime.datetime.now()
    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)
    end = datetime.datetime.now()
    print("Moving train_nfeat & train_labels to device (%s) time: %s" % (device, end - start))

    # Pack data
    data = n_classes, g, train_g, val_g, train_nfeat, train_labels, \
           val_nfeat, val_labels

    run(args, device, data)
    last_end = datetime.datetime.now()
    print("Program total execution time: %s" % (last_end - first_start))