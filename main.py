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

PARAMS_META = {'mnist_fashion'     :{'alpha':0.5, 'beta':4000, 'gamma':1, 'stage1':1, 'stage2':20, 'k':1},
               'cifar10'           :{'alpha':0.5, 'beta':4000, 'gamma':1, 'stage1':44,'stage2':120,'k':1},
               'cifar100'          :{'alpha':0.5, 'beta':4000, 'gamma':1, 'stage1':21,'stage2':60, 'k':1},
               'clothing1M'        :{'alpha':0.5, 'beta':1500, 'gamma':1, 'stage1':1, 'stage2':10, 'k':1},
               'clothing1M50k'     :{'alpha':0.5, 'beta':1500, 'gamma':1, 'stage1':1, 'stage2':10, 'k':1},
               'clothing1Mbalanced':{'alpha':0.5, 'beta':1500, 'gamma':1, 'stage1':1, 'stage2':10, 'k':1},
               'food101N'          :{'alpha':0.5, 'beta':1500, 'gamma':1, 'stage1':1, 'stage2':10, 'k':1}}

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(4, 2)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def metapencil(alpha, beta, gamma, stage1, stage2, k):
    def warmup_training():
        model_s1_path = '{}/{}_{}_{}_{}_{}.pt'.format(dataset,dataset,noise_type,noise_ratio,NUM_TRAINDATA,stage1)
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
                    images, labels = images.to(device), labels.to(device)
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
                    
                    if verbose == 2:
                        template = "Progress: {:6.5f}, Accuracy: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                        sys.stdout.write(template.format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_loss.avg, time.time()-start))
                if verbose == 2:
                    sys.stdout.flush()  
                    
                # evaluate on validation and test data
                val_accuracy, val_loss = evaluate(net, m_dataloader, criterion_cce)
                test_accuracy, test_loss = evaluate(net, test_dataloader, criterion_cce)

                if SAVE_LOGS == 1:
                    summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
                    summary_writer.add_scalar('test_loss', test_loss, epoch)
                    summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
                    summary_writer.add_scalar('val_loss', val_loss, epoch)
                    summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
                    summary_writer.add_figure('confusion_matrix', plot_confusion_matrix(net, test_dataloader), epoch)

                if verbose > 0:
                    template = 'Epoch {}, Accuracy(train,val,test): {:3.1f}/{:3.1f}/{:3.1f}, Loss(train,val,test): {:4.3f}/{:4.3f}/{:4.3f},Learning rate: {}, Time: {:3.1f}({:3.2f})'
                    print(template.format(epoch + 1, 
                                        train_accuracy.percentage, val_accuracy, test_accuracy,
                                        train_loss.avg, val_loss, test_loss,  
                                        lr, time.time()-start_epoch, (time.time()-start_epoch)/3600))   
                torch.save(net.cpu().state_dict(), model_s1_path)
                net.to(device) 
        else:
            net.load_state_dict(torch.load(model_s1_path, map_location=device))  
            val_accuracy, val_loss = evaluate(net, m_dataloader, criterion_cce)
            test_accuracy, test_loss = evaluate(net, test_dataloader, criterion_cce)
            if verbose > 0:
                print('Pretrained model, Accuracy(val,test): {:3.1f}/{:3.1f}, Loss(val,test): {:4.3f}/{:4.3f}'.format(val_accuracy, test_accuracy,val_loss, test_loss))
            if SAVE_LOGS == 1:
                summary_writer.add_scalar('test_loss', test_loss, stage1-1)
                summary_writer.add_scalar('test_accuracy', test_accuracy, stage1-1)
                summary_writer.add_scalar('val_loss', val_loss, stage1-1)
                summary_writer.add_scalar('val_accuracy', val_accuracy, stage1-1)

    def meta_training():
        vnet.train()
        # meta training for predicted labels
        lc = criterion_meta(output, yy)                                            # classification loss
        # train for classification loss with meta-learning
        net.zero_grad()
        grads = torch.autograd.grad(lc, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        for grad in grads:
            grad.detach()
        fast_weights = OrderedDict((name, param - alpha*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))  
        fast_out = net.forward(images_meta,fast_weights)   

        if MIXUP_META == 1:
            loss_meta = mixup_criterion(criterion_cce, fast_out, targets_a_meta, targets_b_meta, lam_meta)
        else:
            loss_meta = criterion_cce(fast_out, labels_meta)
        if MIXUP_TRAIN == 1:
            loss_compatibility = mixup_criterion(criterion_cce, yy, targets_a, targets_b, lam)
        else:
            loss_compatibility = criterion_cce(yy, labels)
        loss_all = loss_meta + gamma*loss_compatibility

        optimizer_vnet.zero_grad()
        vnet.zero_grad()
        loss_all.backward(retain_graph=True)
        optimizer_vnet.step()

        # update labels
        vnet.eval()
        _yy = vnet(feats).detach()
        new_y[meta_epoch+1,index,:] = _yy.cpu().numpy()
        del grads

        # training base network
        lc = criterion_meta(output, _yy)                                # classification loss
        le = -torch.mean(torch.mul(softmax(output), logsoftmax(output)))# entropy loss
        loss = lc + k*le                                                # overall loss
        optimizer.zero_grad()
        net.zero_grad()
        loss.backward()
        optimizer.step()
        vnet.train()
        
        return loss

    def extract_features():
        net.eval()
        features = np.zeros((NUM_TRAINDATA,NUM_FEATURES))
        for batch_idx, (images, labels) in enumerate(t_dataloader):
            images, labels = images.to(device), labels.to(device)
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
            output, feats = net(images,get_feat=True)
            features[index] = feats.cpu().detach().numpy()
        net.train()
        return features

    print('use_clean:{}, mixup_train: {}, mixup_meta: {}'.format(use_clean_data, MIXUP_TRAIN, MIXUP_META))
    print('alpha:{}, beta:{}, gamma:{}, k:{}, stage1:{}, stage2:{}'.format(alpha, beta, gamma, k, stage1, stage2))

    class VNet(nn.Module):
        def __init__(self, input, output):
            super(VNet, self).__init__()
            layer1_size = input
            layer2_size = int(input/2)
            self.linear1 = nn.Linear(layer1_size, layer1_size)
            self.linear2 = nn.Linear(layer1_size, layer2_size)
            self.linear3 = nn.Linear(layer2_size, output)
            self.bn1 = nn.BatchNorm1d(layer1_size)
            self.bn2 = nn.BatchNorm1d(layer2_size)
        def forward(self, x):
            x = F.relu(self.bn1(self.linear1(x)))
            x = F.relu(self.bn2(self.linear2(x)))
            out = self.linear3(x)
            return softmax(out)
    vnet = VNet(NUM_FEATURES, NUM_CLASSES).to(device)
    optimizer_vnet = torch.optim.Adam(vnet.parameters(), beta, weight_decay=1e-4)
    vnet.train()

    # get datasets
    t_dataset, m_dataset, t_dataloader, m_dataloader = train_dataset, meta_dataset, train_dataloader, meta_dataloader
    NUM_TRAINDATA = len(t_dataset)
    # loss functions
    criterion_cce = nn.CrossEntropyLoss()
    criterion_meta = lambda output, labels: torch.mean(softmax(output)*(logsoftmax(output+1e-10)-torch.log(labels+1e-10)))

    # if not done beforehand, perform warmup-training
    warmup_training()

    # if no use clean data, extract reliable data for meta subset
    if use_clean_data == 0:
        t_dataset, m_dataset, t_dataloader, m_dataloader = get_dataloaders_meta()
    t_meta_loader_iter = iter(m_dataloader)  

    # initialize parameters and buffers
    NUM_TRAINDATA = len(t_dataset)
    labels_yy = np.zeros(NUM_TRAINDATA)
    new_y = np.zeros([NUM_META_EPOCHS+1,NUM_TRAINDATA,NUM_CLASSES])
    test_acc_best = 0
    val_acc_best = 0
    epoch_best = 0

    # initialize predicted labels with given labels
    y_init_path = '{}/y_{}_{}_{}_{}.npy'.format(dataset,noise_type,noise_ratio,NUM_TRAINDATA,stage1)
    labels_yy = np.zeros(NUM_TRAINDATA)
    if not os.path.exists(y_init_path):
        y_init = np.zeros([NUM_TRAINDATA,NUM_CLASSES])
        for batch_idx, (images, labels) in enumerate(t_dataloader):
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
            onehot = torch.zeros(labels.size(0), NUM_CLASSES).scatter_(1, labels.view(-1, 1), 1).cpu().numpy()
            y_init[index, :] = onehot
        if not os.path.exists(y_init_path):
            np.save(y_init_path,y_init)
    new_y[0] = np.load(y_init_path)
    # extract features for all training data
    features = extract_features()          

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
        vnet.train()
        grads_dict = OrderedDict((name, 0) for (name, param) in vnet.named_parameters()) 

        for batch_idx, (images, labels) in enumerate(t_dataloader):
            start = time.time()
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
            
            # training images and labels
            images, labels = images.to(device), labels.to(device)
            images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)
            if MIXUP_TRAIN == 1:
                images, targets_a, targets_b, lam = mixup_data(images, labels)

            # compute output
            output, _feats = net(images,get_feat=True)
            _, predicted = torch.max(output.data, 1)

            # predict labels
            feats = torch.tensor(features[index], dtype=torch.float, device=device)
            yy = vnet(feats)
            _, labels_yy[index] = torch.max(yy.cpu(), 1)
            # meta training images and labels
            try:
                images_meta, labels_meta = next(t_meta_loader_iter)
            except StopIteration:
                t_meta_loader_iter = iter(m_dataloader)
                images_meta, labels_meta = next(t_meta_loader_iter)
                images_meta, labels_meta = images_meta[:labels.size(0)], labels_meta[:labels.size(0)]
            images_meta, labels_meta = images_meta.to(device), labels_meta.to(device)
            images_meta, labels_meta = torch.autograd.Variable(images_meta), torch.autograd.Variable(labels_meta)
            if MIXUP_META == 1:
                images_meta, targets_a_meta, targets_b_meta, lam_meta = mixup_data(images_meta, labels_meta)
            
            #with torch.autograd.detect_anomaly():
            loss = meta_training()

            train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), labels.size(0)) 
            train_loss.update(loss.item())
            train_accuracy_meta.update(predicted.eq(torch.tensor(labels_yy[index]).to(device)).cpu().sum().item(), predicted.size(0)) 
            label_similarity.update(labels.eq(torch.tensor(labels_yy[index]).to(device)).cpu().sum().item(), labels.size(0))

            # keep log of gradients
            for tag, parm in vnet.named_parameters():
                if parm.grad != None:
                    grads_dict[tag] += parm.grad.data.cpu().numpy()

            if verbose == 2:
                template = "Progress: {:6.5f}, Accuracy: {:5.4f}, Accuracy Meta: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                sys.stdout.write(template.format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_accuracy_meta.percentage, train_loss.avg, time.time()-start))
        if verbose == 2:
            sys.stdout.flush()           

        if SAVE_LOGS == 1:
            np.save(log_dir + "y.npy", new_y)
        # evaluate on validation and test data
        val_accuracy, val_loss = evaluate(net, m_dataloader, criterion_cce)
        test_accuracy, test_loss = evaluate(net, test_dataloader, criterion_cce)
        if val_accuracy > val_acc_best: 
            val_acc_best = val_accuracy
            test_acc_best = test_accuracy
            epoch_best = epoch

        if SAVE_LOGS == 1:
            summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
            summary_writer.add_scalar('test_loss', test_loss, epoch)
            summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch)
            summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
            summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
            summary_writer.add_scalar('val_loss', val_loss, epoch)
            summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
            summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)
            summary_writer.add_scalar('label_similarity', label_similarity.percentage, epoch)
            summary_writer.add_figure('confusion_matrix', plot_confusion_matrix(net, test_dataloader), epoch)
            for tag, parm in vnet.named_parameters():
                summary_writer.add_histogram('grads_'+tag, grads_dict[tag], epoch)
            if not (noisy_idx is None) and epoch >= stage1:
                hard_labels = np.argmax(new_y[meta_epoch+1], axis=1)
                num_true_pred = np.sum(hard_labels == clean_labels)
                pred_similarity = (num_true_pred / clean_labels.shape[0])*100
                summary_writer.add_scalar('label_similarity_true', pred_similarity, epoch)

        if verbose > 0:
            template = 'Epoch {}, Accuracy(train,meta_train,val,test): {:3.1f}/{:3.1f}/{:3.1f}/{:3.1f}, Loss(train,val,test): {:4.3f}/{:4.3f}/{:4.3f}, Label similarity: {:6.3f}, Learning rate(lr,yy): {}/{}, Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1, 
                                train_accuracy.percentage, train_accuracy_meta.percentage, val_accuracy, test_accuracy,
                                train_loss.avg, val_loss, test_loss,  
                                label_similarity.percentage, lr, int(beta),
                                time.time()-start_epoch, (time.time()-start_epoch)/3600))

    print('{}({}): Train acc: {:3.1f}, Validation acc: {:3.1f}-{:3.1f}, Test acc: {:3.1f}-{:3.1f}, Best epoch: {}, Num meta-data: {}'.format(
        noise_type, noise_ratio, train_accuracy.percentage, val_accuracy, val_acc_best, test_accuracy, test_acc_best, epoch_best, NUM_METADATA))
    if SAVE_LOGS == 1:
        summary_writer.close()
        # write log for hyperparameters
        hp_writer.add_hparams({'alpha':alpha, 'beta': beta, 'gamma':gamma, 'k':k, 'stage1':stage1, 'use_clean':use_clean_data, 'num_meta':NUM_METADATA, 'mixup_train': MIXUP_TRAIN, 'mixup_meta':MIXUP_META}, 
                              {'val_accuracy': val_acc_best, 'test_accuracy': test_acc_best, 'epoch_best':epoch_best})
        hp_writer.close()
        torch.save(net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))

def get_dataloaders_meta():
    NUM_TRAINDATA = len(train_dataset)
    num_meta_data_per_class = int(NUM_METADATA/NUM_CLASSES)
    idx_meta = None
    
    loss_values = np.zeros(NUM_TRAINDATA)
    label_values = np.zeros(NUM_TRAINDATA)
    
    c = nn.CrossEntropyLoss(reduction='none').to(device)
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
        output = net(images)
        loss = c(output, labels)
        loss_values[index] = loss.detach().cpu().numpy()
        label_values[index] = labels.cpu().numpy()
    for i in range(NUM_CLASSES):
        idx_i = label_values == i
        idx_i = np.where(idx_i == True)
        loss_values_i = loss_values[idx_i]
        sorted_idx = np.argsort(loss_values_i)
        #inv_arr = np.exp(-1*loss_values_i)
        #prob_arr = np.array(np.exp(inv_arr)/np.sum(np.exp(inv_arr)))#
        #anchor_idx_i = np.random.choice(idx_i[0], num_meta_data_per_class, p=np.array(prob_arr/np.sum(prob_arr)))
        #anchor_idx_i = np.random.choice(idx_i[0], num_meta_data_per_class, p=prob_arr)
        selected_idx = np.random.choice(sorted_idx[:3*num_meta_data_per_class], num_meta_data_per_class, replace=False)  
        anchor_idx_i = np.take(idx_i, selected_idx)
        if idx_meta is None:
            idx_meta = anchor_idx_i
        else:
            idx_meta = np.concatenate((idx_meta,anchor_idx_i))
    random.Random(RANDOM_SEED).shuffle(idx_meta)
    idx_train = np.setdiff1d(np.arange(NUM_TRAINDATA),np.array(idx_meta))

    t_dataset = torch.utils.data.Subset(train_dataset, idx_train)
    m_dataset = torch.utils.data.Subset(train_dataset, idx_meta)
    t_dataloader = torch.utils.data.DataLoader(t_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers)
    m_dataloader = torch.utils.data.DataLoader(m_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers, drop_last=True)
    return t_dataset, m_dataset, t_dataloader, m_dataloader

def get_topk(arr, percent=0.01):
    arr_flat = arr.flatten()
    arr_len = int(len(arr_flat)*percent)
    idx = np.argsort(np.absolute(arr_flat))[-arr_len:]
    return arr_flat[idx]

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

        if dataset == 'mnist_fashion':
            plt.imshow(np.squeeze(img), cmap=plt.cm.binary)
        else:
            img = np.moveaxis(img.numpy(),0,-1)
            plt.imshow(np.clip(img,0,1))
    return figure

def set_learningrate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(net, dataloader, criterion):
    if dataloader:
        eval_accuracy = AverageMeter()
        eval_loss = AverageMeter()

        net.eval()
        with torch.no_grad():
            for (inputs, targets) in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs) 
                loss = criterion(outputs, targets) 
                _, predicted = torch.max(outputs.data, 1) 
                eval_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
                eval_loss.update(loss.item())
        return eval_accuracy.percentage, eval_loss.avg
    else:
        return 0, 0

def plot_confusion_matrix(net, dataloader):
    net.eval()
    labels, preds = np.zeros(len(dataloader)*BATCH_SIZE), np.zeros(len(dataloader)*BATCH_SIZE)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+targets.size(0))
            inputs, targets = inputs.to(device), targets.to(device)
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
    parser.add_argument('--seed', required=False, type=int, default=42,
        help="Random seed to be used in simulation")
    
    parser.add_argument('-c', '--clean_data', required=False, type=int, default=1,
        help="Either to use available clean data (1) or not (0)")
    parser.add_argument('-m', '--metadata_num', required=False, type=int, default=4000,
        help="Number of samples to be used as meta-data")

    parser.add_argument('-mt', '--mixup_train', required=False, type=int, default=0,
        help="")
    parser.add_argument('-mm', '--mixup_meta', required=False, type=int, default=0,
        help="")

    parser.add_argument('-a', '--alpha', required=False, type=float,
        help="Learning rate for meta iteration")
    parser.add_argument('-b', '--beta', required=False, type=float,
        help="Beta paramter")
    parser.add_argument('-g', '--gamma', required=False, type=float,
        help="Gamma paramter")
    parser.add_argument('-s1', '--stage1', required=False, type=int,
        help="Epoch num to end stage1 (straight training)")
    parser.add_argument('-s2', '--stage2', required=False, type=int,
        help="Epoch num to end stage2 (meta training)")
    parser.add_argument('-k', required=False, type=int, default=10,
        help="")

    args = parser.parse_args()
    #set default variables if they are not given from the command line
    if args.alpha == None: args.alpha = PARAMS_META[args.dataset]['alpha']
    if args.beta == None: args.beta = PARAMS_META[args.dataset]['beta']
    if args.gamma == None: args.gamma = PARAMS_META[args.dataset]['gamma']
    if args.stage1 == None: args.stage1 = PARAMS_META[args.dataset]['stage1']
    if args.stage2 == None: args.stage2 = PARAMS_META[args.dataset]['stage2']
    if args.k == None: args.k = PARAMS_META[args.dataset]['k']
    # configuration variables
    framework = 'pytorch'
    dataset = args.dataset
    model_name = 'MLNC'
    noise_type = args.noise_type
    noise_ratio = args.noise_ratio/100
    BATCH_SIZE = args.batch_size if args.batch_size != None else PARAMS[dataset]['batch_size']
    NUM_CLASSES = PARAMS[dataset]['num_classes']
    NUM_FEATURES = PARAMS[dataset]['num_features']
    NUM_META_EPOCHS = args.stage2 - args.stage1
    SAVE_LOGS = args.save_logs
    RANDOM_SEED = args.seed
    MIXUP_TRAIN = args.mixup_train
    MIXUP_META = args.mixup_meta
    use_clean_data = args.clean_data
    verbose = args.verbose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_ids is None:
        gpu_ids = 0
        ngpu = 1
    else:
        gpu_ids = args.gpu_ids[0]
        ngpu = len(gpu_ids)
        if ngpu == 1: 
            device = torch.device("cuda:{}".format(gpu_ids[0]))
        
    if args.num_workers is None:
        num_workers = 2 if ngpu < 2 else ngpu*2
    else:
        num_workers = args.num_workers
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    # create necessary folders
    create_folder('{}/dataset'.format(dataset))
    # global variables
    if use_clean_data == 0:
        train_dataset, val_dataset, test_dataset, meta_dataset, class_names = get_data(dataset,framework,noise_type,noise_ratio,RANDOM_SEED,0,0)
        noisy_idx, clean_labels = get_synthetic_idx(dataset,RANDOM_SEED,0,0,noise_type,noise_ratio)
        meta_dataloader = None
    else:
        train_dataset, val_dataset, test_dataset, meta_dataset, class_names = get_data(dataset,framework,noise_type,noise_ratio,RANDOM_SEED,args.metadata_num,0)
        noisy_idx, clean_labels = get_synthetic_idx(dataset,RANDOM_SEED,args.metadata_num,0,noise_type,noise_ratio)
        meta_dataloader = torch.utils.data.DataLoader(meta_dataset,batch_size=BATCH_SIZE,shuffle=False, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    
    NUM_METADATA = args.metadata_num
    net = get_model(dataset,framework).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    lr_scheduler = get_lr_scheduler(dataset)
    optimizer = optim.SGD(net.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=1e-4)
    logsoftmax = nn.LogSoftmax(dim=1).to(device)
    softmax = nn.Softmax(dim=1).to(device)
  
    print("Dataset: {}, Model: {}, Device: {}, Batch size: {}, #GPUS to run: {}".format(dataset, model_name, device, BATCH_SIZE, ngpu))
    if dataset in DATASETS_SMALL:
        print("Noise type: {}, Noise ratio: {}".format(noise_type, noise_ratio))

    # if logging
    if SAVE_LOGS == 1:
        base_folder = model_name if dataset in DATASETS_BIG else noise_type + '/' + str(args.noise_ratio) + '/' + model_name
        log_folder = args.folder_log if args.folder_log else 'c{}_a{}_b{}_g{}_s{}_m{}_{}'.format(use_clean_data, args.alpha, args.beta, args.gamma, args.stage1, NUM_METADATA, current_time)
        log_base = '{}/logs/{}/'.format(dataset, base_folder)
        log_dir = log_base + log_folder + '/'
        log_dir_hp = '{}/logs_hp/{}/'.format(dataset, base_folder)
        create_folder(log_dir)
        summary_writer = SummaryWriter(log_dir)
        create_folder(log_dir_hp)
        hp_writer = SummaryWriter(log_dir_hp)
    
    start_train = time.time()
    metapencil(args.alpha, args.beta, args.gamma, args.stage1, args.stage2, args.k)
    print('Total training duration: {:3.2f}h'.format((time.time()-start_train)/3600))