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
bins = [2,3,4,5,6,7,8,9,10,50,100]


#y_train = np.where(Y_train[:] > 0.0, 1, 0)
#y_test = np.where(Y_test[:] > 0.0, 1, 0)


print(Y_train.shape)
print(Y_test.shape)
print(X_train.shape)
print(X_test.shape)

y_loc_train = np.where(Y_train[:] >= 1.0)
y_loc_test = np.where(Y_test[:] >= 1.0)

y_train = Y_train[:][Y_train >= 1.0]
y_test = Y_test[:][Y_test >= 1.0]



if 0.0 in y_train:
    print("Found1")
    
if 0.0 in y_test:
    print("Found2")
    
X_train = X_train[:][y_loc_train]
X_test = X_test[:][y_loc_test]

X_train = np.resize(X_train,(41665, 4, 32, 32))
X_test = np.resize(X_test,(30250, 4, 32, 32))


#X_train = np.expand_dims(X_train, )

for p,q in enumerate(bins):
    y_train = np.where(Y_train[:] >= q, p+1, y_train)
    y_test = np.where(Y_test[:] >= q, p+1, y_test)

    
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
class ResNet(nn.Module):
    def __init__(self, fcl_layers = [], dr = 0.0, output_size = 1, resnet_model = 18, pretrained = True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.resnet_model = resnet_model 
        if self.resnet_model == 18:
            resnet = models.resnet18(pretrained=self.pretrained)
        elif self.resnet_model == 34:
            resnet = models.resnet34(pretrained=self.pretrained)
        elif self.resnet_model == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.resnet_model == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        elif self.resnet_model == 152:
            resnet = models.resnet152(pretrained=self.pretrained)
        resnet.conv1 = torch.nn.Conv1d(4, 64, (7, 7), (2, 2), (3, 3), bias=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet_output_dim = resnet.fc.in_features
        self.resnet = nn.Sequential(*modules)
        self.fcn = self.make_fcn(self.resnet_output_dim, output_size, fcl_layers, dr)
        
    def make_fcn(self, input_size, output_size, fcl_layers, dr):
        if len(fcl_layers) > 0:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, fcl_layers[0]),
                nn.BatchNorm1d(fcl_layers[0]),
                torch.nn.LeakyReLU()
            ]
            if len(fcl_layers) == 1:
                fcn.append(nn.Linear(fcl_layers[0], output_size))
            else:
                for i in range(len(fcl_layers)-1):
                    fcn += [
                        nn.Linear(fcl_layers[i], fcl_layers[i+1]),
                        nn.BatchNorm1d(fcl_layers[i+1]),
                        torch.nn.LeakyReLU(),
                        nn.Dropout(dr)
                    ]
                fcn.append(nn.Linear(fcl_layers[i+1], output_size))
        else:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, output_size)
            ]
        if output_size > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))
        return nn.Sequential(*fcn)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x

output_size = len(weights) # len(range(int(max(Y_train.squeeze(-1))))) + 1
fcl_layers = []
dropout = 0.5

model = ResNet(fcl_layers, dr = dropout, output_size = output_size, resnet_model=18, pretrained = False).to(device)
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
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))

train_criterion = SmoothCrossEntropyLoss(weight = weights) #weight = weights, smoothing = 0.1) 
test_criterion = torch.nn.CrossEntropyLoss()
#set scheduler
def torch_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc.item())
        return list_topk_accs 
##Train the model
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
##use pytorch data iterators now instead of lazy sampler
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=train_batch_size, 
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)),
    batch_size=valid_batch_size,
    shuffle=False)
#run training
epoch_test_losses = []

for epoch in range(epochs):

    ### Train the model 
    model.train()

    # Shuffle the data first
    batch_loss = []
    accuracy = {k: [] for k in topk}
    indices = list(range(X_train.shape[0]))
    random.shuffle(indices)
    
    # Now split into batches
    train_batches_per_epoch = int(X_train.shape[0] / train_batch_size) 
    train_batches_per_epoch = min(batches_per_epoch, train_batches_per_epoch)
    
    # custom tqdm so we can see the progress
    batch_group_generator = tqdm(
        enumerate(train_loader), 
        total=train_batches_per_epoch, 
        leave=True
    )

    for k, (x, y) in batch_group_generator:

        # Converting to torch tensors and moving to GPU
        inputs = x.float().to(device)
        lightning_counts = y.long().to(device)

        # Clear gradient
        optimizer.zero_grad()

        # get output from the model, given the inputs
        pred_lightning_counts = model(inputs)

        # get loss for the predicted output
        loss = train_criterion(pred_lightning_counts, lightning_counts.squeeze(-1))
        
        # compute the top-k accuracy
        acc = torch_accuracy(pred_lightning_counts.cpu(), lightning_counts.cpu(), topk = topk)
        for i,l in enumerate(topk):
            accuracy[l] += [acc[i]]

        # get gradients w.r.t to parameters
        loss.backward()
        batch_loss.append(loss.item())

        # update parameters
        optimizer.step()

        # update tqdm
        to_print = "Epoch {} train_loss: {:.4f}".format(epoch, np.mean(batch_loss))
        for l in sorted(accuracy.keys()):
            to_print += " top-{}_acc: {:.4f}".format(l,np.mean(accuracy[l]))
        #to_print += " top-2_acc: {:.4f}".format(np.mean(accuracy[2])
        #to_print += " top-3_acc: {:.4f}".format(np.mean(accuracy[3]))
        to_print += " lr: {:.12f}".format(optimizer.param_groups[0]['lr'])
        batch_group_generator.set_description(to_print)
        batch_group_generator.update()
                                  
        if k >= train_batches_per_epoch and k > 0:
            break
        
    torch.cuda.empty_cache()

    ### Test the model 
    model.eval()
    with torch.no_grad():

        batch_loss = []
        accuracy = {k: [] for k in topk}
        
        # custom tqdm so we can see the progress
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
            # update tqdm
            to_print = "Epoch {} test_loss: {:.4f}".format(epoch, np.mean(batch_loss))
            for k in sorted(accuracy.keys()):
                to_print += " top-{}_acc: {:.4f}".format(k,np.mean(accuracy[k]))
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

    test_loss = 1 - np.mean(accuracy[1])
    epoch_test_losses.append(test_loss)
    
    # Lower the learning rate if we are not improving
    lr_scheduler.step(test_loss)

    # Save the model if its the best so far.
    if test_loss == min(epoch_test_losses):
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss
        }
        torch.save(state_dict, "bestNo0.pt")
        
    # Stop training if we have not improved after X epochs
    best_epoch = [i for i,j in enumerate(epoch_test_losses) if j == min(epoch_test_losses)][-1]
    offset = epoch - best_epoch
    if offset >= patience:
        break
        
    #-l gpu_type = gpu100, v100
#Load best model
checkpoint = torch.load(
    "best.pt",
    map_location=lambda storage, loc: storage
)
best_epoch = checkpoint["epoch"]
#model = Net(filter_sizes, fcl_layers).to(device)
model = ResNet(fcl_layers, dr = dropout, output_size = output_size).to(device)
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
df_cm.to_csv('Confusion_Matrix.csv')



