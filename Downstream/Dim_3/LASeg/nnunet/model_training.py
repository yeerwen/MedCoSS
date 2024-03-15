
import torch.nn.functional as functional


import os


import copy
import numpy as np
import random
from tensorboardX import SummaryWriter


from unet import *
from dataset import *
from visualization import *
from dice_loss import *
import torch

# TODO: when training, turn this false
debug = False

def train(model,scheduler,optimizer,dice_loss,train_generator,train_dataset,writer,n_itr):
    # function to train  model for segmentation task
    # params:
        # model
        # scheduler
        # optimizer
        # dice_loss: dice loss object
        # train_generator: data generator for training set
        # train_dataset: traning dataset
        # writer: summary writer for tensorboard
        # n_iter: current iteration number, for loss plot
    scheduler.step()
    model.train()  # Set model to training mode           

    running_loss = 0.0
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    
    for i_batch, sample_batch in enumerate(train_generator):
        img = sample_batch['img'].permute(1,0,2,3) # change the order so that we have num of image at the first place
        label = sample_batch['label'].permute(1,0,2,3) # change the order so that we have num of image at the first place
        
        if debug:      
            print(img.shape)
            print(label.shape)

            imshow(img[0,:,:,:].permute(1,2,0),denormalize=False)
            imshow(label[0,:,:,:].permute(1,2,0),denormalize=False)
            imshow(img[-1,:,:,:].permute(1,2,0),denormalize=False)
            imshow(label[-1,:,:,:].permute(1,2,0),denormalize=False)       

        # transfer to GPU
        img, label = img.cuda(), label.cuda()
    
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backprop + optimize
        outputs = model(img)
        loss,probas,true_1_hot = dice_loss.forward(outputs, label.long())
        loss.backward()
        optimizer.step()
        loss.detach()
        
        # statistics
        running_loss += loss.item() * img.size(0)    
        writer.add_scalar('data/training_loss',loss.item(),n_itr)
        n_itr = n_itr + 1
        
        curr_tp, curr_fp, curr_fn = label_accuracy(probas.cpu(),true_1_hot.cpu())
        tp += curr_tp
        fp += curr_fp
        fn += curr_fn
        
        if debug:
            break
                
    train_loss = running_loss / len(train_dataset)
    print('Training Loss: {:.4f}'.format(train_loss))
    for i_class, (tp_val, fp_val, fn_val) in enumerate(zip(tp, fp, fn)):
        print ('{} Class, True Pos {}, False Pos {}, False Neg {}, Dice score {:1.2f}'.format(i_class, tp_val,fp_val,fn_val,(2*tp_val + 1e-7)/ (2*tp_val+fp_val+fn_val+1e-7)))
    print('-' * 10)
    
    return train_loss, n_itr


def validate(model,dice_loss,num_class,validation_generator,validation_dataset,writer,n_itr):
    ########################### Validation #####################################
    model.eval()  # Set model to validation mode   
    validation_loss = 0.0
    tp = torch.zeros(num_class)
    fp = torch.zeros(num_class)
    fn = torch.zeros(num_class)
    worst_dice = 1.0
    best_dice = 0.0
    worst_indx = 0
    best_indx = 0
    
    for i_batch, sample_batch in enumerate(validation_generator):
        img = sample_batch['img'].permute(1,0,2,3) # change the order so that we have num of image at the first place
        label = sample_batch['label'].permute(1,0,2,3) # change the order so that we have num of image at the first place
        indx = sample_batch['indx']
        
        if debug:      
            print(img.shape)
            print(label.shape)

            imshow(img[0,:,:,:].permute(1,2,0),denormalize=False)
            imshow(label[0,:,:,:].permute(1,2,0),denormalize=False)
            imshow(img[-1,:,:,:].permute(1,2,0),denormalize=False)
            imshow(label[-1,:,:,:].permute(1,2,0),denormalize=False)
        
        # transfer to GPU
        img, label = img.cuda(), label.cuda()
   
        # forward 
        outputs = model(img)
        loss, probas, true_1_hot = dice_loss.forward(outputs, label.long())
        
        # statistics
        validation_loss += loss.item() * img.size(0)
        writer.add_scalar('data/validation_loss',loss.item(),n_itr)
        n_itr = n_itr + 1
        
        curr_tp, curr_fp, curr_fn = label_accuracy(probas.cpu(),true_1_hot.cpu())
        tp += curr_tp
        fp += curr_fp
        fn += curr_fn
        curr_dice = ((2*curr_tp + 1e-7)/ (2*curr_tp+curr_fp+curr_fn+1e-7)).mean()
        
        # find best and worst
        if worst_dice > curr_dice:
            worst_indx = indx
            worst_dice = curr_dice
        if best_dice < curr_dice:
            best_indx = indx
            best_dice = curr_dice
        
        if debug:
            break
            
    validation_loss = validation_loss / len(validation_dataset)
    print('Vaildation Loss: {:.4f}'.format(validation_loss))
    for i_class, (tp_val, fp_val, fn_val) in enumerate(zip(tp, fp, fn)):
        print ('{} Class, True Pos {}, False Pos {}, False Neg {}, Dice score {:1.2f}'.format(i_class, tp_val,fp_val,fn_val,(2*tp_val + 1e-7)/ (2*tp_val+fp_val+fn_val+1e-7)))
    print('-' * 10)
    
    return validation_loss, tp, fp, fn, n_itr,[worst_indx,worst_dice],[best_indx,best_dice]

def run_training(model,num_class,scheduler,optimizer,dice_loss,num_epochs,train_generator,train_dataset,validation_generator,validation_dataset,writer):
    print("Training Started!")

    # initialize best_acc for comparison
    best_acc = 0.0
    train_iter = 0
    val_iter = 0

    for epoch in range(num_epochs):
        print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")

        # train
        train_loss, train_iter = train(model,scheduler,optimizer,dice_loss,train_generator,train_dataset,writer,train_iter)

        # validate
        with torch.no_grad():
            validation_loss, tp, fp, fn, val_iter, worst,best = validate(model,dice_loss,num_class,validation_generator,validation_dataset,writer,val_iter)
            epoch_acc = (2*tp + 1e-7)/ (2*tp+fp+fn+1e-7)
            epoch_acc = epoch_acc.mean()
    
            # loss
            writer.add_scalar('data/Training Loss (per epoch)',train_loss,epoch)
            writer.add_scalar('data/Validation Loss (per epoch)',validation_loss,epoch)
            
            # show best and worst 
            print("worst performance: dice {:.2f}, idx {}".format(worst[1],worst[0].item()))
            sample = validation_dataset.__getitem__(worst[0])
            img = sample['img'][0,:,:]
            label = sample['label'][0,:,:]
            tmp_img = img.reshape(1,1,256,256)
            pred = torch.softmax(model(tmp_img.cuda()), dim=1)
            pred_label = torch.max(pred,dim=1)[1]
            pred_label = pred_label.type(label.type())
            # to plot
            tp_img = np.array(img)
            tp_label = np.array(label)
            tp_pred = np.array(pred_label.cpu())
            imshow(tp_img)
            imshow(tp_label)
            imshow(tp_pred)

            print("best performance: dice {:.2f}, idx {}".format(best[1],best[0].item()))
            sample = validation_dataset.__getitem__(best[0])
            img = sample['img'][0,:,:]
            label = sample['label'][0,:,:]
            tmp_img = img.reshape(1,1,256,256)
            pred = torch.softmax(model(tmp_img.cuda()), dim=1)
            pred_label = torch.max(pred,dim=1)[1]
            pred_label = pred_label.type(label.type())
            # to plot
            tp_img = np.array(img)
            tp_label = np.array(label)
            tp_pred = np.array(pred_label.cpu())
            imshow(tp_img)
            imshow(tp_label)
            imshow(tp_pred)

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Dice Score: {:.4f}'.format(best_acc.item()))
                
            print('-' * 10)
            
    return best_model_wts, best_acc


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_batch_size = 2
    validation_batch_size = 1
    learning_rate = 0.001
    num_epochs = 70
    num_class = 4
    writer = SummaryWriter()

    model = unet(useBN=True)
    model.cuda()
    
    dice_loss = DICELoss(np.ones((num_class,1))) 
    dice_loss.cuda()
    
    # intialize optimizer and lr decay
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_both_aug = Compose([
        PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0,p=1),
        RandomCrop(height=256, width=256, p=1),
        Cutout(p=0.5),
        OneOf([
            ShiftScaleRotate(p=0.7),
            HorizontalFlip(p=0.8),
            VerticalFlip(p=0.8)
        ])
    ])
    train_img_aug = Compose([
#         Normalize(p=1,mean=np.array([0.5,]),std=np.array([0.5,])),
        OneOf([
            RandomBrightnessContrast(brightness_limit=1, contrast_limit=1,p=0.5),
            RandomGamma(p=0.5)]),
    ])
    
    
    val_both_aug = Compose([
        PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0,p=1),
        RandomCrop(height=256, width=256, p=1)
    ])

    train_dataset=ACDCDataset(data_type="train",transform_both=train_both_aug,transform_image=None)
    validation_dataset=ACDCDataset(data_type="validation",transform_both=val_both_aug,transform_image=None)
    
    # intialize the dataloader
    train_generator = DataLoader(train_dataset,shuffle=True,batch_size=train_batch_size,num_workers=8)
    validation_generator = DataLoader(validation_dataset,shuffle=True,batch_size=validation_batch_size,num_workers=8)
    
    run_training(model,num_class,scheduler,optimizer,dice_loss,num_epochs,train_generator,train_dataset,validation_generator,validation_dataset,writer)