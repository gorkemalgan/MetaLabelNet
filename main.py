import os, sys
import time
import numpy as np
import math, random
import datetime
from collections import OrderedDict
import itertools
import gc

from dataset import get_data, get_dataloader, get_synthetic_idx, DATASETS_BIG, DATASETS_SMALL
from model_getter import get_model
from utils import *

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PARAMS_META = {'cifar10'           :{'beta':1e-2, 'stage1':44,'stage2':120, 'num_metadata': 5000},
               'clothing1M'        :{'beta':1e-3, 'stage1':1, 'stage2':10 , 'num_metadata': 5000},
               'clothing1M50k'     :{'beta':1e-3, 'stage1':1, 'stage2':10 , 'num_metadata': 5000},
               'clothing1Mbalanced':{'beta':1e-3, 'stage1':1, 'stage2':10 , 'num_metadata': 5000},
               'food101N'          :{'beta':1e-3, 'stage1':6, 'stage2':20 , 'num_metadata': 10000},
               'WebVision'         :{'beta':1e-3, 'stage1':14,'stage2':40 , 'num_metadata': 20000}}

def meta_noisy_train(beta, stage1, stage2):
    def warmup_training(model_s1_path):
        if not os.path.exists(model_s1_path):
            for epoch in range(stage1): 
                start_epoch = time.time()
                train_accuracy = AverageMeter()
                train_loss = AverageMeter()

                lr = lr_scheduler(epoch)
                set_learningrate(optimizer, lr)
                net.train()

                for batch_idx, (images, labels) in enumerate(t_dataloader):
                    start = time.time()
                    
                    # training images and labels
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)

                    # compute output
                    output, _feats = net(images,get_feat=True)
                    _, predicted = torch.max(output.data, 1)

                    # training
                    loss = criterion_cce(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), labels.size(0)) 
                    train_loss.update(loss.item())
                    
                    if VERBOSE == 2:
                        template = "Progress: {:6.5f}, Accuracy: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                        sys.stdout.write(template.format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_loss.avg, time.time()-start))
                if VERBOSE == 2:
                    sys.stdout.flush()  
                    
                # evaluate on validation and test data
                val_accuracy, val_loss, topk_accuracy = evaluate(net, m_dataloader, criterion_cce)
                test_accuracy, test_loss, topk_accuracy = evaluate(net, test_dataloader, criterion_cce)

                if SAVE_LOGS == 1:
                    summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
                    summary_writer.add_scalar('test_loss', test_loss, epoch)
                    summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
                    summary_writer.add_scalar('val_loss', val_loss, epoch)
                    summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
                    summary_writer.add_scalar('topk_accuracy', topk_accuracy, epoch)
                    summary_writer.add_figure('confusion_matrix', plot_confusion_matrix(net, test_dataloader), epoch)

                if VERBOSE > 0:
                    template = 'Epoch {}, Accuracy(train,top5,val,test): {:3.1f}/{:3.1f}/{:3.1f}/{:3.1f}, Loss(train,val,test): {:4.3f}/{:4.3f}/{:4.3f},Learning rate: {}, Time: {:3.1f}({:3.2f})'
                    print(template.format(epoch + 1, 
                                        train_accuracy.percentage, topk_accuracy, val_accuracy, test_accuracy,
                                        train_loss.avg, val_loss, test_loss,  
                                        lr, time.time()-start_epoch, (time.time()-start_epoch)/3600))   
            torch.save(net.state_dict(), model_s1_path)
            net.to(DEVICE) 
        else:
            net.load_state_dict(torch.load(model_s1_path, map_location=DEVICE))  
            val_accuracy, val_loss, topk_accuracy = evaluate(net, m_dataloader, criterion_cce)
            test_accuracy, test_loss, topk_accuracy = evaluate(net, test_dataloader, criterion_cce)
            if VERBOSE > 0:
                print('Pretrained model, Accuracy(topk,val,test): {:3.1f}/{:3.1f}/{:3.1f}, Loss(val,test): {:4.3f}/{:4.3f}'.format(topk_accuracy, val_accuracy, test_accuracy,val_loss, test_loss))
            if SAVE_LOGS == 1:
                summary_writer.add_scalar('test_loss', test_loss, stage1-1)
                summary_writer.add_scalar('test_accuracy', test_accuracy, stage1-1)
                summary_writer.add_scalar('val_loss', val_loss, stage1-1)
                summary_writer.add_scalar('val_accuracy', val_accuracy, stage1-1)
                summary_writer.add_scalar('topk_accuracy', topk_accuracy, stage1-1)

    def meta_loss(output, yy, images_meta, labels_meta):
        # meta training for predicted labels
        lc = criterion_meta(output, yy)                                 # classification loss
        # train for classification loss with meta-learning
        net.zero_grad()
        grads = torch.autograd.grad(lc, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        grads = [grad for grad in grads if grad is not None]
        for grad in grads:
            grad.detach()
        fast_weights = OrderedDict((name, param - grad) for ((name, param), grad) in zip(net.named_parameters(), grads))  
        fast_out = net.forward(images_meta,fast_weights)   
        return criterion_cce(fast_out, labels_meta)

    def conventional_train(output, _yy):
        # training base network
        lc = criterion_meta(output, _yy)                                # classification loss
        le = -torch.mean(torch.mul(softmax(output), logsoftmax(output)))# entropy loss
        loss = lc + le                                                  # overall loss
        optimizer.zero_grad()
        net.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def train():
        # initialize parameters and buffers
        t_meta_loader_iter = iter(m_dataloader) 
        labels_yy = np.zeros(NUM_TRAINDATA)
        test_acc_best = 0
        val_acc_best = 0
        epoch_best = 0
        topk_acc_best = 0
        for epoch in range(stage1,stage2): 
            start_epoch = time.time()
            train_accuracy = AverageMeter()
            train_loss = AverageMeter()
            train_accuracy_meta = AverageMeter()
            label_similarity = AverageMeter()
            meta_epoch = epoch - stage1

            lr = lr_scheduler(epoch)
            set_learningrate(optimizer, lr)
            net.train()
            meta_net.train()
            grads_dict = OrderedDict((name, 0) for (name, param) in meta_net.named_parameters()) 

            dl_dict = {'labeled':t_dataloader, 'unlabeled':u_dataloader}
            for data_type in dl_dict:
                dl = dl_dict[data_type]
                if not (dl is None):
                    for batch_idx, (images, labels) in enumerate(dl):
                        start = time.time()
                        index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
                        
                        # training images and labels
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)
                        size_of_batch = labels.size(0)

                        # compute output
                        output, _ = net(images,get_feat=True)
                        _, predicted = torch.max(output.data, 1)
                        train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), size_of_batch) 

                        # predict initial soft labels labels
                        meta_net.eval()
                        _, feats = feature_encoder(images,get_feat=True)
                        yy = meta_net(feats)
                        if data_type == 'labeled':
                            _, labels_yy[index] = torch.max(yy.cpu(), 1)
                            train_accuracy_meta.update(predicted.eq(torch.tensor(labels_yy[index]).to(DEVICE)).cpu().sum().item(), predicted.size(0)) 
                            label_similarity.update(labels.eq(torch.tensor(labels_yy[index]).to(DEVICE)).cpu().sum().item(), size_of_batch)

                        # meta-network training
                        #with torch.autograd.detect_anomaly():
                        meta_net.train()
                        meta_net.zero_grad()
                        optimizer_meta_net.zero_grad()
                        images_meta, labels_meta, t_meta_loader_iter = get_batch(m_dataloader,t_meta_loader_iter,size_of_batch)
                        loss_meta = meta_loss(output, yy, images_meta, labels_meta)
                        loss_meta.backward(retain_graph=True)
                        if DATASET in DATASETS_BIG:# == 'WebVision':
                            del images_meta,labels_meta, loss_meta
                            gc.collect()
                        optimizer_meta_net.step()
                        # keep log of gradients
                        for tag, parm in meta_net.named_parameters():
                            if parm.grad != None:
                                grads_dict[tag] += parm.grad.data.cpu().numpy()

                        # conventional training
                        meta_net.eval()
                        _yy = meta_net(feats).detach()
                        loss = conventional_train(output, _yy)
                        train_loss.update(loss.item())
                        if data_type == 'labeled':
                            new_y[meta_epoch+1,index,:] = _yy.cpu().numpy()
                        del feats, yy

                        if VERBOSE == 2:
                            template = "Progress: {:6.5f}, Accuracy: {:5.4f}, Accuracy Meta: {:5.4f}, Process time:{:5.4f}   \r"
                            sys.stdout.write(template.format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_accuracy_meta.percentage, time.time()-start))
            if VERBOSE == 2:
                sys.stdout.flush()  
             
            # evaluate on validation and test data
            val_accuracy, val_loss, topk_accuracy = evaluate(net, m_dataloader, criterion_cce)
            test_accuracy, test_loss, topk_accuracy = evaluate(net, test_dataloader, criterion_cce)
            if val_accuracy > val_acc_best: 
                val_acc_best = val_accuracy
                test_acc_best = test_accuracy
                epoch_best = epoch
                topk_acc_best = topk_accuracy

            if SAVE_LOGS == 1:
                np.save(log_dir + "y.npy", new_y)
                summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
                summary_writer.add_scalar('test_loss', test_loss, epoch)
                summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch)
                summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
                summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
                summary_writer.add_scalar('val_loss', val_loss, epoch)
                summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
                summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)
                summary_writer.add_scalar('topk_accuracy', topk_accuracy, epoch)
                summary_writer.add_scalar('topk_accuracy_best', topk_acc_best, epoch)
                summary_writer.add_scalar('label_similarity', label_similarity.percentage, epoch)
                summary_writer.add_scalar('label_diff_mean', abs(new_y[meta_epoch+1]-new_y[meta_epoch]).mean(), epoch)
                summary_writer.add_scalar('label_diff_var', abs(new_y[meta_epoch+1]-new_y[meta_epoch]).var(), epoch)
                summary_writer.add_figure('confusion_matrix', plot_confusion_matrix(net, test_dataloader), epoch)
                for tag, parm in meta_net.named_parameters():
                    summary_writer.add_histogram('grads_'+tag, grads_dict[tag], epoch)
                if not (n_idx is None) and epoch >= stage1:
                    hard_labels = np.argmax(new_y[meta_epoch+1], axis=1)
                    num_true_pred = np.sum(hard_labels == c_labels)
                    pred_similarity = (num_true_pred / c_labels.shape[0])*100
                    summary_writer.add_scalar('label_similarity_true', pred_similarity, epoch)

            if VERBOSE > 0:
                template = 'Epoch {}, Accuracy(train,meta_train,topk,val,test): {:3.1f}/{:3.1f}/{:3.1f}/{:3.1f}/{:3.1f}, Loss(train,val,test): {:4.3f}/{:4.3f}/{:4.3f}, Label similarity: {:6.3f}, Num-data(meta,meta-true,unlabeled): {}/{}/{}, Hyper-params(beta,s1,s2,mp,seed): {:5.4f}/{}/{}/{}/{}, Time: {:3.1f}({:3.2f})'
                print(template.format(epoch + 1, 
                                    train_accuracy.percentage, train_accuracy_meta.percentage, topk_accuracy, val_accuracy, test_accuracy,
                                    train_loss.avg, val_loss, test_loss,  
                                    label_similarity.percentage, NUM_METADATA, meta_true, NUM_UNLABELED, beta, stage1, stage2, MAGIC_PARAM,RANDOM_SEED,
                                    time.time()-start_epoch, (time.time()-start_epoch)/3600))

        print('{}({}): Train acc: {:3.1f}, Topk acc: {:3.1f}-{:3.1f} Validation acc: {:3.1f}-{:3.1f}, Test acc: {:3.1f}-{:3.1f}, Best epoch: {}, Num-data(meta,meta-true,unlabeled): {}/{}/{}, Hyper-params(beta,s1,s2,mp,seed): {:5.4f}/{}/{}/{}/{}, Time: {:3.2f}h'.format(
            NOISE_TYPE, NOISE_RATIO, train_accuracy.percentage, topk_accuracy, topk_acc_best, val_accuracy, val_acc_best, test_accuracy, test_acc_best, epoch_best, NUM_METADATA, meta_true, NUM_UNLABELED, beta, stage1, stage2, MAGIC_PARAM, RANDOM_SEED, (time.time()-TRAIN_START_TIME)/3600))
        if SAVE_LOGS == 1:
            summary_writer.close()
            # write log for hyperparameters
            hp_writer.add_hparams({'beta': beta, 'stage1':stage1, 'stage2':stage2, 'magicp':MAGIC_PARAM, 'num_meta':NUM_METADATA, 'num_train': NUM_TRAINDATA, 'num_unlabeled': NUM_UNLABELED}, 
                                  {'val_accuracy': val_acc_best, 'test_accuracy': test_acc_best, 'topk_accuracy_best':topk_acc_best, 'epoch_best':epoch_best})
            hp_writer.close()
            torch.save(net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))

    def init_labels(y_init_path):
        new_y = np.zeros([NUM_META_EPOCHS+1,NUM_TRAINDATA,NUM_CLASSES])
        if not os.path.exists(y_init_path) or (USE_SAVED == 0):
            y_init = np.zeros([NUM_TRAINDATA,NUM_CLASSES])
            for batch_idx, (_, labels) in enumerate(t_dataloader):
                index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
                onehot = torch.zeros(labels.size(0), NUM_CLASSES).scatter_(1, labels.view(-1, 1), 1).cpu().numpy()
                y_init[index, :] = onehot
            if not os.path.exists(y_init_path) or (USE_SAVED == 0):
                np.save(y_init_path,y_init)
        new_y[0] = np.load(y_init_path)
        return new_y

    class MetaNet(nn.Module):
        def __init__(self, input, output):
            super(MetaNet, self).__init__()
            self.linear = nn.Linear(input, output)
        def forward(self, x):
            out = self.linear(x)
            return softmax(out)

    meta_net = MetaNet(NUM_FEATURES, NUM_CLASSES).to(DEVICE)
    optimizer_meta_net = torch.optim.Adam(meta_net.parameters(), beta, weight_decay=1e-4)#optim.SGD(meta_net.parameters(), lr=beta, momentum=0.9, weight_decay=1e-4)
    meta_net.train()

    # get datasets
    t_dataset, m_dataset, t_dataloader, m_dataloader, u_dataset, u_dataloader = train_dataset, meta_dataset, train_dataloader, meta_dataloader, unlabeled_dataset, unlabeled_dataloader
    n_idx, c_labels = noisy_idx, clean_labels
    meta_true = None
    NUM_TRAINDATA = len(t_dataset)

    # loss functions
    criterion_cce = nn.CrossEntropyLoss()
    criterion_meta = lambda output, labels: torch.mean(softmax(output)*(logsoftmax(output+1e-10)-torch.log(labels+1e-10)))

    # paths for save and load
    path_ext = '{}_{}_{}_{}'.format(NUM_TRAINDATA,NUM_UNLABELED,RANDOM_SEED,stage1)
    if DATASET in DATASETS_SMALL:
        path_ext = '{}_{}_{}'.format(path_ext,NOISE_TYPE,NOISE_RATIO)
    model_s1_path = '{}/model_s1_{}.pt'.format(DATASET,path_ext)
    y_init_path = '{}/yinit_{}.npy'.format(DATASET,path_ext)

    # if not done beforehand, perform warmup-training
    warmup_training(model_s1_path)
    # set feature encoder as model trained after warm-up
    feature_encoder.load_state_dict(torch.load(model_s1_path, map_location=DEVICE))  
    feature_encoder.eval()

    # initialize predicted labels with given labels
    new_y = init_labels(y_init_path)
    # meta training
    train()

def image_grid(idx, train_dataset, grads):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(15,15))
    for i in range(25):
        index = idx[i]
        img,l = train_dataset.__getitem__(index)
        # if clean labels are known, print them as well
        if clean_labels is None:
            title = '{}/{}\n{:6.3f}'.format(class_names[l], class_names[np.argmax(grads[index])], grads[index].max())
            color = 'black'
        else:
            title = '{}/{}/{}\n{:6.3f}'.format(class_names[clean_labels[index]], class_names[l], class_names[np.argmax(grads[index])], grads[index].max())
            # if gradient direction is correct
            if clean_labels[index] == np.argmax(grads[index]):
                color = 'g'
            # if gradient direction is wrong
            else:
                color = 'r'
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=title)
        ax = plt.gca()
        ax.set_title(title, color=color)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        if DATASET == 'mnist_fashion':
            plt.imshow(np.squeeze(img), cmap=plt.cm.binary)
        else:
            img = np.moveaxis(img.numpy(),0,-1)
            plt.imshow(np.clip(img,0,1))
    return figure

def set_learningrate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_batch(dataloader,dataloader_iter,batch_size):
    try:
        images, labels = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        images, labels = next(dataloader_iter)
    images, labels = images[:batch_size], labels[:batch_size]
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)
    return images, labels, dataloader_iter

def evaluate(net, dataloader, criterion):
    eval_accuracy = AverageMeter()
    eval_loss = AverageMeter()
    topk_accuracy = AverageMeter()

    if dataloader:  
        net.eval()
        with torch.no_grad():
            for (inputs, targets) in dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = net(inputs) 
                loss = criterion(outputs, targets) 
                _, predicted = torch.max(outputs.data, 1) 
                _, topks = torch.topk(outputs.data, 5, 1) 
                eval_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
                eval_loss.update(loss.item())

                topks = topks.cpu().numpy()
                targets_tmp = targets.cpu().numpy()
                topk_num = 0
                for i in range(topks.shape[0]):
                    if targets_tmp[i] in topks[i]:
                        topk_num += 1
                topk_accuracy.update(topk_num, targets.size(0))
    return eval_accuracy.percentage, eval_loss.avg, topk_accuracy.percentage

def plot_confusion_matrix(net, dataloader):
    net.eval()
    labels, preds = np.zeros(len(dataloader)*BATCH_SIZE), np.zeros(len(dataloader)*BATCH_SIZE)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+targets.size(0))
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs) 
            _, predicted = torch.max(outputs.data, 1) 

            labels[index] = targets.cpu().numpy()
            preds[index] = predicted.cpu().numpy()

    cm = confusion_matrix(labels, preds)
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=False, type=str, default='cifar10',
        help="Dataset to use; either 'mnist_fashion', 'cifar10', 'cifar100', 'food101N', 'clothing1M'")
    parser.add_argument('-n', '--noise_type', required=False, type=str, default='feature-dependent',
        help="Noise type for cifar10: 'feature-dependent', 'symmetric'")
    parser.add_argument('-r', '--noise_ratio', required=False, type=int, default=40,
        help="Synthetic noise ratio in percentage between 0-100")
    parser.add_argument('-s', '--batch_size', required=False, type=int,
        help="Number of gpus to be used")
    parser.add_argument('-i', '--gpu_ids', required=False, type=int, nargs='+', action='append',
        help="GPU ids to be used")
    parser.add_argument('-f', '--folder_log', required=False, type=str,
        help="Folder name for logs")
    parser.add_argument('-v', '--verbose', required=False, type=int, default=0,
        help="Details of prints: 0(silent), 1(not silent)")
    parser.add_argument('-w', '--num_workers', required=False, type=int,
        help="Number of parallel workers to parse dataset")
    parser.add_argument('--save_logs', required=False, type=int, default=1,
        help="Either to save log files (1) or not (0)")
    parser.add_argument('--use_saved', required=False, type=int, default=1,
        help="Either to use presaved files (1) or not (0)")
    parser.add_argument('--seed', required=False, type=int, default=42,
        help="Random seed to be used in simulation")
    
    parser.add_argument('-b', '--beta', required=False, type=float,
        help="Beta paramter")
    parser.add_argument('-s1', '--stage1', required=False, type=int,
        help="Epoch num to end stage1 (straight training)")
    parser.add_argument('-s2', '--stage2', required=False, type=int,
        help="Epoch num to end stage2 (meta training)")

    parser.add_argument('-m', '--metadata_num', required=False, type=int,
        help="Number of samples to be used as meta-data")
    parser.add_argument('-u', '--unlabeleddata_num', required=False, type=int, default=0,
        help="Number of samples to be used as unlabeled-data")
    parser.add_argument('--magicparam', required=False, type=float, default=0,
        help="Free variable for using different tryouts through the code")
    
    args = parser.parse_args()
    #set default variables if they are not given from the command line
    if args.beta == None: args.beta = PARAMS_META[args.dataset]['beta']
    if args.stage1 == None: args.stage1 = PARAMS_META[args.dataset]['stage1']
    if args.stage2 == None: args.stage2 = PARAMS_META[args.dataset]['stage2']
    # configuration variables
    FRAMEWORK = 'pytorch'
    DATASET = args.dataset
    MODEL_NAME = 'MLNC'
    NOISE_TYPE = args.noise_type
    NOISE_RATIO = args.noise_ratio/100
    BATCH_SIZE = args.batch_size if args.batch_size != None else PARAMS[DATASET]['batch_size']
    NUM_CLASSES = PARAMS[DATASET]['num_classes']
    NUM_FEATURES = PARAMS[DATASET]['num_features']
    NUM_META_EPOCHS = args.stage2 - args.stage1
    NUM_METADATA = args.metadata_num if args.metadata_num != None else PARAMS_META[DATASET]['num_metadata']
    SAVE_LOGS = args.save_logs
    USE_SAVED = args.use_saved
    RANDOM_SEED = args.seed
    VERBOSE = args.verbose
    MAGIC_PARAM = args.magicparam
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_ids is None:
        gpu_ids = 0
        ngpu = 1
    else:
        gpu_ids = args.gpu_ids[0]
        ngpu = len(gpu_ids)
        if ngpu == 1: 
            DEVICE = torch.device("cuda:{}".format(gpu_ids[0]))
        
    if args.num_workers is None:
        num_workers = 2 if ngpu < 2 else ngpu*2
    else:
        num_workers = args.num_workers
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    # create necessary folders
    create_folder('{}/dataset'.format(DATASET))
    # datasets
    train_dataset, meta_dataset, test_dataset, unlabeled_dataset, class_names = get_data(DATASET,FRAMEWORK,NOISE_TYPE,NOISE_RATIO,RANDOM_SEED,NUM_METADATA,args.unlabeleddata_num,verbose=VERBOSE)
    noisy_idx, clean_idx, clean_labels = get_synthetic_idx(DATASET,RANDOM_SEED,NUM_METADATA,args.unlabeleddata_num,NOISE_TYPE,NOISE_RATIO)
    meta_dataloader = torch.utils.data.DataLoader(meta_dataset,batch_size=BATCH_SIZE,shuffle=False, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    if unlabeled_dataset is None:
        unlabeled_dataloader = None
        NUM_UNLABELED = 0
    else:
        unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset,batch_size=BATCH_SIZE,shuffle=True)
        NUM_UNLABELED = len(unlabeled_dataset)
    
    net = get_model(DATASET,FRAMEWORK,DEVICE).to(DEVICE)
    feature_encoder = get_model(DATASET,FRAMEWORK).to(DEVICE)
    if (DEVICE.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
        feature_encoder = nn.DataParallel(feature_encoder, list(range(ngpu)))
    lr_scheduler = get_lr_scheduler(DATASET)
    optimizer = optim.SGD(net.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=1e-4)
    logsoftmax = nn.LogSoftmax(dim=1).to(DEVICE)
    softmax = nn.Softmax(dim=1).to(DEVICE)
  
    if DATASET in DATASETS_SMALL:
        print("Dataset: {}, Noise type: {}, Noise ratio: {}, Device: {}, Batch size: {}, #GPUS to run: {}, beta:{}, stage1:{}, stage2:{}, mp:{}, Num-data(meta,unlabeled): {}/{}, Seed: {}".format(DATASET, NOISE_TYPE, NOISE_RATIO, DEVICE, BATCH_SIZE, ngpu, args.beta, args.stage1, args.stage2, MAGIC_PARAM, NUM_METADATA, NUM_UNLABELED, RANDOM_SEED))
    else:
        print("Dataset: {}, Model: {}, Device: {}, Batch size: {}, #GPUS to run: {}, beta:{}, stage1:{}, stage2:{}, mp:{}, Num-data(meta,unlabeled): {}/{}, Seed: {}".format(DATASET, MODEL_NAME, DEVICE, BATCH_SIZE, ngpu, args.beta, args.stage1, args.stage2, MAGIC_PARAM, NUM_METADATA, NUM_UNLABELED, RANDOM_SEED))

    # if logging
    if SAVE_LOGS == 1:
        base_folder = MODEL_NAME if DATASET in DATASETS_BIG else NOISE_TYPE + '/' + str(args.noise_ratio) + '/' + MODEL_NAME
        log_folder = '{}_{}'.format(args.folder_log,current_time) if args.folder_log else 'b{}_s{}_m{}_u{}_sd{}_{}'.format(args.beta, args.stage1, NUM_METADATA, NUM_UNLABELED, RANDOM_SEED, current_time)
        log_base = '{}/logs/{}/'.format(DATASET, base_folder)
        log_dir = log_base + log_folder + '/'
        log_dir_hp = '{}/logs_hp/{}/'.format(DATASET, base_folder)
        if args.folder_log:
            log_dir_hp = '{}{}/'.format(log_dir_hp, args.folder_log)
        #clean_emptylogs()
        create_folder(log_dir)
        summary_writer = SummaryWriter(log_dir)
        create_folder(log_dir_hp)
        hp_writer = SummaryWriter(log_dir_hp)
    
    TRAIN_START_TIME = time.time()
    meta_noisy_train(args.beta, args.stage1, args.stage2)