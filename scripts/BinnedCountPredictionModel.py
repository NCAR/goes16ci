import random
import pickle
import numpy as np

from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from goes16ci.models import torch_accuracy, BinResNet, SmoothCrossEntropyLoss, bintrainloop
import torchvision.models as models


from tqdm import tqdm as tqdm_base
#pull arguments

from tqdm import tqdm as tqdm_base
#get cuda instance
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cuda")

if is_cuda:
    torch.backends.cudnn.benchmark = True

print(f'Preparing to use device {device}')
#Download Data
with open('/glade/u/home/gwallach/goes16ci/train_data_scaled.pkl','rb') as f:
    x = pickle.load(f)
    
print("Saved shape:", x.shape)
x = x.transpose(0,3,1,2)
print("Reshaped to:", x.shape)

with open('/glade/u/home/gwallach/goes16ci/train_counts.pkl','rb') as f:
    y = pickle.load(f)
    

print("Saved shape:", y.shape)
y = y.reshape(y.shape[0], 1)
print("Reshaped to:", y.shape)
#split into test and train
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state = 5000
)

y_scaler = StandardScaler()
#split data into bins
bins = [5,10,15,25,50,75,100]

y_train = np.where(Y_train[:] > 0.0, 1, 0)
y_test = np.where(Y_test[:] > 0.0, 1, 0)
for p,q in enumerate(bins):
    y_train = np.where(Y_train[:] > q, p+2, y_train)
    y_test = np.where(Y_test[:] > q, p+2, y_test)
    
print(y_train.shape)
print(y_test.shape)
print(X_train.shape)
print(X_test.shape)
##create weights based on the counts
from collections import Counter
counts = Counter()
for val in y_train:
    counts[val[0]] += 1
counts = dict(counts)

#weights = [1 - (counts[x] / sum(counts.values())) for x in sorted(counts.keys())]
weights = [np.log1p(max(counts.values()) / counts[x]) for x in sorted(counts.keys())]
weights = [x / max(weights) for x in weights]
weights = torch.FloatTensor(weights).to(device)

print(weights)
##Load a Model

output_size = len(weights) # len(range(int(max(Y_train.squeeze(-1))))) + 1
fcl_layers = []
dropout = 0.5

model = BinResNet(fcl_layers, dr = dropout, output_size = output_size, resnet_model=18, pretrained = False).to(device)
#test model to ensure consistency
X = torch.from_numpy(X_train[:2]).float().to(device)
print(X.shape)
g = model(X).exp()
print(torch.max(g,1)) # exp to turn the logits into probabilities, since we used LogSoftmax
print(g)
print(torch.argmax(g,1))
#Load an optimizer
learning_rate = 1e-05
weight_decay = 1e-04
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#load a loss function
train_criterion = SmoothCrossEntropyLoss(weight = weights) #weight = weights, smoothing = 0.1) 
test_criterion = torch.nn.CrossEntropyLoss()
#set scheduler
lr_scheduler = ReduceLROnPlateau(
    optimizer, 
    patience = 1, 
    verbose = False
)
epochs = 1000 
train_batch_size = 32
valid_batch_size = 128
batches_per_epoch = 1000

topk = (1, 2)
patience = 5 # this is how many epochs we will keep training since we last saw a "best" model -- "early stopping"

#train the model
bintrainloop(epochs, X_train, train_batch_size, batches_per_epoch, valid_batch_size, topk, x, y, model, train_criterion, test_criterion, patience, optimizer, lr_scheduler)        
    #-l gpu_type = gpu100, v100
#Load best model
checkpoint = torch.load(
    "best.pt",
    map_location=lambda storage, loc: storage
)
best_epoch = checkpoint["epoch"]
#model = Net(filter_sizes, fcl_layers).to(device)
model = BinResNet(fcl_layers, dr = dropout, output_size = output_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
#predict on test dataset
topk = (1,2)


model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    batch_loss = []
    accuracy = {k: [] for k in topk}
    # split test data into batches

    valid_batches_per_epoch = int(X_test.shape[0] / valid_batch_size) 
    batch_group_generator = tqdm(
        test_loader, 
        total=valid_batches_per_epoch, 
        leave=True
    )
    
    for (x, y) in batch_group_generator:
        # Converting to torch tensors and moving to GPU
        inputs = x.float().to(device)
        lightning_counts = y.long().to(device)
        # get output from the model, given the inputs
        pred_lightning_counts = model(inputs)
        # get loss for the predicted output
        loss = test_criterion(pred_lightning_counts, lightning_counts.squeeze(-1))
        batch_loss.append(loss.item())
        # compute the accuracy
        acc = torch_accuracy(pred_lightning_counts, lightning_counts, topk = topk)
        for i,k in enumerate(topk):
            accuracy[k] += [acc[i]]
        
        y_true.append(lightning_counts.squeeze(-1))
        # Taking the top-1 answer here, but here is where we could compute the average predicted rather than take top-1
        y_pred.append(torch.argmax(pred_lightning_counts, 1))

y_true = torch.cat(y_true, axis = 0)
y_pred = torch.cat(y_pred, axis = 0)
#save out to csv for analysis
#save out batch loss and accuracy
print("batch_loss",np.mean(batch_loss),"accuracy",np.mean(accuracy))
y_true = y_true.cpu().numpy()
y_pred = y_pred.cpu().numpy()
np.savetxt("y_true.csv", y_true, delimiter=",")
np.savetxt("y_pred.csv", y_pred, delimiter=",")
print("y_true",y_true)
print("y_pred",y_pred)
print("val_loss", np.mean(batch_loss))
for k in topk:
    print(f"top-{k} {np.mean(accuracy[k])}")
for label in list(set(y_true)):
    c = (y_true == label)
    print(label, (y_true[c] == y_pred[c]).mean())
list(set(y_pred))
import seaborn as sn
import pandas as pd
cm = metrics.confusion_matrix(y_true, y_pred, normalize = 'true')
df_cm = pd.DataFrame(cm, index = sorted(list(set(y_true))), columns = sorted(list(set(y_true))))
#save out confusion matrix to csv
df_cm.to_csv('Confusion_Matrix.csv')
#save out confusion matrix as plot
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.xlabel("Actual lighting class")
plt.ylabel("Predicted lighting class")
plt.savefig('BinConfusion.png')



