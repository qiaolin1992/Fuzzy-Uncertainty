import os
import torch
from datauncertainty.crossvaliation.convnet import UNet
from torch import optim
from datauncertainty.crossvaliation.crossdata import BasicDataset
from torch.utils.data import DataLoader
from datauncertainty.crossvaliation.crosspytorchtools import EarlyStopping
import numpy as np
import time

# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
img_path='C:/lq/code/uncertainty/datauncertainty/data/lung/datasets/img/'
label_path='C:/lq/code/uncertainty/datauncertainty/data/lung/datasets/label/'

def accu_iou(output,label):
    n=output.shape[0]
    matrix_iou_stack = 0
    for i in range(n):
        pred_y=output[i,:,:,:]
        y=label[i,:,:,:]
        # B is the mask pred, A is the malanoma
        y_pred = (pred_y > 0.5) * 1.0
        y_true = (y > 0.5) * 1.0
        pred_flat = y_pred.view(y_pred.numel())#
        true_flat = y_true.view(y_true.numel())

        intersection = float(torch.sum(pred_flat * true_flat)) + 1e-7
        denominator = float(torch.sum(pred_flat + true_flat)) - intersection + 2e-7

        matrix_iou = intersection / denominator
        matrix_iou_stack += matrix_iou
    return matrix_iou_stack/n
def accu_dice(output,label):
    n = output.shape[0]
    matrix_iou_stack = 0
    for i in range(n):
        pred_y = output[i, :, :, :]
        y = label[i, :, :, :]
        # B is the mask pred, A is the malanoma
        y_pred = (pred_y > 0.5) * 1.0
        y_true = (y > 0.5) * 1.0
        pred_flat = y_pred.view(y_pred.numel())  #
        true_flat = y_true.view(y_true.numel())

        intersection = float(torch.sum(pred_flat * true_flat)) + 1e-7
        denominator = float(torch.sum(pred_flat + true_flat)) + 2e-7

        matrix_iou =2* intersection / denominator
        matrix_iou_stack += matrix_iou
    return matrix_iou_stack / n
def sesp(output,label):
    n = output.shape[0]
    max_se=0
    max_sp=0

    for i in range(n):
        pred_y = output[i, :, :, :]
        y = label[i, :, :, :]
        # B is the mask pred, A is the malanoma
        y_pred = (pred_y > 0.5) * 1.0
        y_true = (y > 0.5) * 1.0
        precited = y_pred.view(y_pred.numel()).cpu.numpy()  #
        expected = y_true.view(y_true.numel()).cpu.numpy()
        res = (precited ^ expected)  # 亦或使得判断正确的为0,判断错误的为1
        r = np.bincount(res)
        tp_list = ((precited) & (expected))
        fp_list = (precited & (~expected))
        tp_list = tp_list.tolist()
        fp_list = fp_list.tolist()
        tp = tp_list.count(1)
        fp = fp_list.count(1)
        tn = r[0] - tp
        fn = r[1] - fp
        max_se=max_se+tp/(tp+fn)
        max_sp=max_sp+tn/(tn+fp)
    return max_se/n, max_sp/n





def train_model(model, criterion, optimizer,train_loader,test_loader,numfold, num_epochs=100):

    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    avg_IOU=[]
    avg_DICE=[]


    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=30, verbose=True,fold=numfold)
    begin = time.time()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the IOU
        IOU_accuracy = []
        # to track the dice
        DICE_accuracy = []
        se=[]
        sp=[]


        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        print('begin to train model')

        #i=0
        for data, target in train_loader:

            #print('data.size',data.shape)

            inputs=data.to(device)
            labels=target.to(device)
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the loss
            loss = criterion(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        print('begin to test model')
        with torch.no_grad():

            for data, target in test_loader:
                #print('test data')
                # forward pass: compute predicted outputs by passing inputs to the model
                inputs = data.to(device)
                labels = target.to(device)
                output = model(inputs)
                #print(' the shape of output:',output.shape)
                #print('the shape of label:',labels.shape)
                # calculate the loss
                loss = criterion(output, labels)

                iou=accu_iou(output,labels)
                dice=accu_dice(output,labels)
                #se,sp=sesp(output,labels)
                # record validation loss
                valid_losses.append(loss.item())
                IOU_accuracy.append(iou)
                DICE_accuracy.append(dice)
                #se.append(se)
                #sp.append(sp)

        IOU=1-np.average(IOU_accuracy)

        print('test_IOU:',1-IOU)
        print('test_DICE:',np.average(DICE_accuracy))
        # print('test_sensitivity:',np.average(se))
        #print('test_specificity',np.average(sp))
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_IOU.append(1-IOU)
        avg_DICE.append(DICE_accuracy)
        epoch_len = len(str(num_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        #train_losses = []
        #valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping((1-np.average(DICE_accuracy)), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    #model.load_state_dict(torch.load('checkpoint.pt'))
    end = time.time()
    print('train time:', end - begin)
    '''
    train_curve = np.array(avg_train_losses)
    valid_curve = np.array(avg_valid_losses)
    train_IOU=np.array(avg_IOU)
    train_DICE=np.array(avg_DICE)
    
    np.save('./skin/model/unet_train_curve.npy', train_curve)
    np.save('./skin/model/unet_valid_curve.npy', valid_curve)
    np.save('./skin/model/unet_train_IOU.npy', train_IOU)
    np.save('./skin/model/unet_train_DICE.npy', train_DICE)
'''

    return model, avg_train_losses, avg_valid_losses

#data analysis
#obtain all image paths and label paths
img=[]
#label=[]
img1=[]
img2=[]
img3=[]
img4=[]
img5=[]
for filename1 in os.listdir(img_path):
    path=os.path.join(img_path,filename1)
    img.append(path)

for i in range(0,len(img)-5,5):
    #print('i:',i)
    img1.append(img[i])
    img2.append(img[i+1])
    img3.append(img[i+2])
    img4.append(img[i+3])
    img5.append(img[i+4])
#obtain each fold image path
train_img1=img1+img2+img3+img4
test_img1=img5
train_img2=img1+img2+img3+img5
test_img2=img4
train_img3=img1+img2+img4+img5
test_img3=img3
train_img4=img1+img3+img4+img5
test_img4=img2
train_img5=img2+img3+img4+img5
test_img5=img1
for fold in range(1,6):

# 训练模型
    print('step1:model')
    model = UNet(3, 1).to(device)
    batch_size = 8
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # model.parameters():Returns an iterator over module parameters
    if fold==1:
        train_dataset = BasicDataset(train_img1,label_path,1)
        test_dataset=BasicDataset(test_img1,label_path,1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        final_model,train_process,valid_process=train_model(model, criterion, optimizer, train_loader,test_loader,fold,100)
    elif fold==2:
        train_dataset = BasicDataset(train_img2,label_path,1)
        test_dataset=BasicDataset(test_img2,label_path,1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        final_model,train_process,valid_process=train_model(model, criterion, optimizer, train_loader,test_loader,fold,100)
    elif fold==3:
        train_dataset = BasicDataset(train_img3,label_path,1)
        test_dataset=BasicDataset(test_img3,label_path,1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        final_model,train_process,valid_process=train_model(model, criterion, optimizer, train_loader,test_loader,fold,100)
    elif fold==4:
        train_dataset = BasicDataset(train_img4,label_path,1)
        test_dataset=BasicDataset(test_img4,label_path,1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        final_model,train_process,valid_process=train_model(model, criterion, optimizer, train_loader,test_loader,fold,100)
    else:
        train_dataset = BasicDataset(train_img5,label_path,1)
        test_dataset=BasicDataset(test_img5,label_path,1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        final_model,train_process,valid_process=train_model(model, criterion, optimizer, train_loader,test_loader,fold,100)














