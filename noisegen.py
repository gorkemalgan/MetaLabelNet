import numpy as np
import torch
from model_getter import get_model
from dataset import get_data, get_dataloader, get_synthetic_idx, DATASETS_BIG, DATASETS_SMALL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(DEVICE)

train_dataset, meta_dataset, test_dataset, class_names = get_data('cifar10','pytorch','symmetric',0,42,None)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=False)
net = get_model('cifar10').to(DEVICE)
net.load_state_dict(torch.load('model_e35.pt', map_location=DEVICE)) 

NUM_TRAINDATA = len(train_dataset)
NUM_CLASSES = 10

y_init = np.zeros([NUM_TRAINDATA,NUM_CLASSES])
for batch_idx, (images, labels) in enumerate(train_dataloader):
    index = np.arange(batch_idx*32, (batch_idx)*32+labels.size(0))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    output = net(images)
    probs = softmax(output)
    y_init[index, :] = probs.cpu().detach().numpy()

np.save('t.npy',y_init)