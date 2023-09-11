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
import random

from model import GED, SAGE
from dataset import EvolutionaryNet
from utils import EarlyStopping

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

def evaluate(model, g, nfeat, labels, val_nid, device):
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
    return compute_metrics(pred, labels, args.detailed)

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = [nfeat[node].to_dense() for node in input_nodes]
    batch_inputs = th.stack(batch_inputs)
    batch_inputs = batch_inputs.to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def collect_feature(args, g, nfeat, feature_extractor, device):
    feature_extractor.eval()
    with th.no_grad():
        f = feature_extractor.inference(g, nfeat, device, args.batch_size, args.num_workers)
    print("f: ", type(f), f.shape)
    return f

#### Entry point
def run(args, device, data):
    # Unpack data
    n_classes, train_g, val_g, test_data, train_nfeat, train_labels, \
    val_nfeat, val_labels, data_file_path = data
    in_feats = train_nfeat.shape[1]
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

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

    # Define model and optimizer
    if args.model == 'ged':
        model = GED(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    elif args.model == 'sage':
        model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)

    if args.phase == 'analysis':

        start = datetime.datetime.now()
        checkpoint_path = "./checkpoints/{}".format(args.timestamp)
        model.load_state_dict(th.load(os.path.join(checkpoint_path, "latest.pt")))

        feature_extractor = model.graphsage.to(device)
        
        g, nfeat, _ = test_data
        all_features = collect_feature(args, g, nfeat, feature_extractor, device)
        th.save(all_features, "./visual/features_drebin.pt")
        return

    # Hyperparameter weight (negative(0) : positive(1) = 1 : 5)
    loss_fcn = nn.CrossEntropyLoss(weight=th.Tensor([1, 5]).to(device))
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create checkpoint directory path
    save_path = "./checkpoints"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    model_name = model.__class__.__name__
    checkpoint_name = timestamp + "_" + model_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    checkpoint_path = os.path.join(save_path, checkpoint_name)
    if os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(save_path, checkpoint_name + str(random.randint(1, 10000)))
    os.mkdir(checkpoint_path)
    if args.early_stop:
        stopper = EarlyStopping(checkpoint_path, patience=10)

    # Training loop
    avg = 0
    iter_tput = []
    hist_loss = []
    eval_f1 = 0
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
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'\
                .format(epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()
            batch_loss.append(loss.item())

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        hist_loss.append(sum(batch_loss) / len(batch_loss))
        if epoch >= 5:
            avg += toc - tic
        
        if args.early_stop:
            if epoch % args.eval_every == 0 and epoch >= 10:
                eval_start = datetime.datetime.now()
                eval_acc, eval_f1, eval_p, eval_r = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
                eval_end = datetime.datetime.now()
                print('Epoch: {} | Time: {} | Eval Acc: {:.4f} | Eval F1: {:.4f} | Eval Precision: {:.4f} | Eval Recall: {:.4f}'\
                .format(epoch, eval_start-eval_end, eval_acc, eval_f1, eval_p, eval_r))

            if stopper.step(eval_f1, model):
                break
        
    if epoch >= 5:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    # Testing
    start = datetime.datetime.now()
    if args.early_stop:
        model.load_state_dict(stopper.load_checkpiont())
    metrics = testing_monthly(args, model, device, test_data)
    end = datetime.datetime.now()
    print("Total testing time (load state dict if early stopped & split testing subgraph if inductive learning & inference on graph): %s" % (end - start))

    # Save model
    start = datetime.datetime.now()
    save_model(args, data_file_path, save_path, checkpoint_path, model, hist_loss, metrics)
    end = datetime.datetime.now()
    print("Saving model in %s (consume time: %s)" % (checkpoint_path, end - start))

def compute_aut_metrics(metrics):
    """
        Compute AUT version of each metrics
        Refer to paper: TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time
    """
    # Delete 'period' field for each row
    metrics = [metric[1:] for metric in metrics]
    metrics = np.array(metrics, dtype=np.float32)
    norm = metrics.shape[0] * 2 - 2
    aut_metrics = metrics[1:-1].sum(axis=0) + metrics.sum(axis=0)
    return aut_metrics / norm

def save_model(args, data_path, save_path, checkpoint_path, model, loss, metrics):
    # Save model state_dict
    if args.save_model:
        th.save(model.state_dict(), os.path.join(checkpoint_path, "latest.pt"))

    # Save model loss & metrics
    # hist_records = dict()
    # hist_records['loss'] = loss
    # hist_records['metrics'] = metrics
    # th.save(hist_records, os.path.join(checkpoint_path, "historyRecords.pt"))

    # Append reports
    report_file = os.path.join(save_path, "reports.csv")
    aut_acc, aut_f1, aut_p, aut_r = compute_aut_metrics(metrics)
    with open(report_file, 'a') as f:
        f.write("\n")
        f.write("{}\n".format(args))
        f.write("{}\n".format(data_path))
        f.write("Records saved in {}\n".format(checkpoint_path))
        f.write("AUT_acc {:.4f}\tAUT_F1 {:.4f}\tAUT_P {:.4f}\tAUT_R {:.4f}\n".format(aut_acc, aut_f1, aut_p, aut_r))

def inductive_split(g):
    """Split the graph into training graph, validation graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['val_mask'] | g.ndata['train_mask'])
    return train_g, val_g

def testing_monthly(args, model, device, test_data):
    """
    Model is tested on dataset month by month.
    Save time by handling the inference process separately for different experimental settings.
    """
    metrics = list()
    g, nfeat, labels = test_data
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    for period, test_mask in getMonths():
        test_nid = th.nonzero(g.ndata[test_mask], as_tuple=True)[0]
        month_pred = pred[test_nid].cpu().data.numpy().argmax(axis=1)
        month_labels = labels[test_nid].data.numpy()
        test_acc, test_f1, test_p, test_r = compute_metrics(month_pred, month_labels, args.detailed)
        print('Test period: {} | Test Acc: {:.4f} | Test F1: {:.4f} | Test Precision: {:.4f} | Test Recall: {:.4f}'.format(period, test_acc, test_f1, test_p, test_r))
        metrics.append([period, test_acc, test_f1, test_p, test_r])
    return metrics

def getMonths():
    # for year in [2013]:
    #     for month in range(1, 13):
    # for year in [2015]:
    #     for month in range(1, 4):
    # for year in [2015, 2016]:
    #     for month in range(1, 13):
    for year in [2012]:
        for month in range(1, 11):
            period = "{}-{}".format(year, month)
            test_mask = "test_mask_" + period
            yield period, test_mask

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='ged',
                            help="'ged' and 'sage' are available. 'sage' stands for GraphSAGE. 'ged' stands for GraphEvolveDroid."
                                 "By default the script use GraphEvolveDroid architecture.")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=2,
                            help="Number of gnn layers.")
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
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--early-stop', action='store_true',
                           help="Indicates whether to use early stop")
    argparser.add_argument('--detailed', action='store_true', 
                            help="Print confusion matrix during inference process.")
    argparser.add_argument('--save-model', action='store_true', 
                            help="By default the script does not save model state_dict "
                                 "considering disk space consumption. This may be undesired"
                                 "if you want to recover the model for inference. "
                                 "This flag disables that.")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    # model ananlysis
    argparser.add_argument('--timestamp', type=str, default='',
                            help="Specify the name of trained model for testing/validation/analysis")
    argparser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'val', 'analysis'],
                        help="When phase is 'test', only test the model.")
    args = argparser.parse_args()
    assert args.model in ['ged', 'sage'], "Only ged(GraphEvolveDroid) and sage(GraphSAGE) are available."
    assert len(args.fan_out.split(',')) == args.num_layers, "Specify number of sampled neighbors for each layer."
    print()
    first_start = datetime.datetime.now()
    print(first_start)
    print(args)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    start = datetime.datetime.now()
    # Dataset directory. More detailed explanation see README.md
    # data_dir = "/home/sunrui/data/apigraph/vocabulary-2012"
    # data_dir = "/home/sunrui/data/GraphEvolveDroid"
    # feat_mtx = "drebin_feat_mtx.npz"
    data_dir = "/home/sunrui/data/drebin"
    feat_mtx = "drebin_feat.npz"
    adj_mtx = "drebin_knn_5.npz"
    # adj_mtx = "drebin_rev_tf_knn_5.npz"
    file_path = (data_dir, feat_mtx, adj_mtx)
    print(file_path)

    dataset = EvolutionaryNet(data_dir, feat_mtx, adj_mtx)
    g = dataset[0]
    end = datetime.datetime.now()
    print("Loading dataset time: %s" % (end - start))
    n_classes = 2

    start = datetime.datetime.now()
    if args.inductive:
        train_g, val_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_data = g, g.ndata.pop('features'), g.ndata.pop('labels')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        train_labels = val_labels = test_labels = g.ndata.pop('labels')
        test_data = test_g, test_nfeat, test_labels
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
    data = n_classes, train_g, val_g, test_data, train_nfeat, train_labels, \
           val_nfeat, val_labels, file_path

    run(args, device, data)
    last_end = datetime.datetime.now()
    print(file_path)
    print("Program total execution time: %s" % (last_end - first_start))