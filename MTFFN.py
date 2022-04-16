# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import time
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##set file name
dataset='IN'
train_ratio=0.03
method='IN_ALL_specattention_3D7x7'
Epoch1=150
Epoch2=50
# load data
mat_data = sio.loadmat('E:\data\Indian_pines_corrected.mat')
mat_gt = sio.loadmat('E:\data\Indian_pines_gt.mat')
data_ = mat_data['indian_pines_corrected']
gt_IN = mat_gt['indian_pines_gt']

print (data_.shape)
## Data preprocessing
H,W,band_num=data_.shape
num_class = np.max(gt_IN)

data_ = data_.reshape(np.prod(data_.shape[:2]),np.prod(data_.shape[2:]))
data_ = preprocessing.scale(data_)
data_ = data_.reshape(H, W,band_num)
data_ = np.float64(data_.transpose((2,0,1)))

res_seed_list=[]
Htrain_loss=[]
Hval_loss=[]
Htrain_acc=[]
Hval_acc=[]

res_seed_list1=[]
res_seed_list2=[]
res_seed_list3=[]
train1stage_time=[]
test1stage_time=[]
train2stage_time=[]
test2stage_time=[]

from operator import truediv          
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
def reports(y_pred, y_test):
 	classification = classification_report(y_test, y_pred)
 	oa = accuracy_score(y_test, y_pred)
 	confusion = confusion_matrix(y_test, y_pred)
 	each_acc, aa = AA_andEachClassAccuracy(confusion)
 	kappa = cohen_kappa_score(y_test, y_pred)
 	return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))

deeplabv3=models.segmentation.deeplabv3_resnet50(pretrained=True)
deeplabv3.classifier[4]=nn.Conv2d(256,num_class,kernel_size=(1, 1), stride=(1, 1))

#rgb_fcn_resnet50=models.segmentation.fcn_resnet50(pretrained=True)
#rgb_fcn_resnet50.classifier[4]=nn.Conv2d(512,16,kernel_size=(1, 1), stride=(1, 1))

class SPE_Attention_3D(nn.Module):
    def __init__(self,in_ch,band_num,spa_pool_size):
        super(SPE_Attention_3D, self).__init__()     
        self.CA_local = nn.Sequential(           
           nn.AvgPool2d(kernel_size=(spa_pool_size,spa_pool_size),stride=(1,1),padding=(spa_pool_size//2,spa_pool_size//2)),##空间局部pooling,poolsize:行列最小值的比例：1/2,1/4,1/8,1/16,grid search
           nn.Conv2d(in_ch*band_num,in_ch*band_num,kernel_size=1,stride=1,padding=0,groups=in_ch),
           nn.Sigmoid(),
         )        
        self.BN=nn.Sequential(
            nn.BatchNorm3d(in_ch),                           
            nn.ReLU(),)
    def forward(self, inputx):
        x1=inputx
        x1=x1.reshape(x1.shape[0],x1.shape[1]*x1.shape[2],x1.shape[3],x1.shape[4])
        w1=self.CA_local(x1)
        w1=w1.reshape(inputx.shape)
        output=self.BN(w1*inputx)     
        return output

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.spe1=nn.Sequential(nn.Conv3d(1, 4, (7, 1, 1), stride=(2, 1, 1)),
                                nn.BatchNorm3d(4, affine=False),
                                nn.ReLU(),)
        
        self.spe2=nn.Sequential(nn.Conv3d(4, 16, (7, 1, 1), padding=(3, 0,0)),
                                nn.BatchNorm3d(16, affine=False),
                                nn.ReLU(),
                                nn.Conv3d(16, 4, (7, 1, 1), padding=(3, 0,0)),
                                nn.BatchNorm3d(4, affine=False),
                                nn.ReLU(),)
        self.spe3=nn.Sequential(nn.Conv3d(4, 16, (7, 1, 1), stride=(2,1,1)),
                                nn.BatchNorm3d(16, affine=False),
                                nn.ReLU(),)
        self.spe4=nn.Sequential(nn.Conv3d(16, 8, (3, 1, 1), padding=(1, 0,0)),
                                nn.BatchNorm3d(8, affine=False),
                                nn.ReLU(),
                                nn.Conv3d(8, 16, (3, 1, 1), padding=(1, 0,0)),
                                nn.BatchNorm3d(16, affine=False),
                                nn.ReLU(),) 
        self.spehe=nn.Sequential(nn.Conv3d(16, 16, (B2, 1, 1)),
                                nn.BatchNorm3d(16, affine=False),
                                nn.ReLU(),)
        self.spepre=nn.Sequential(nn.AvgPool3d((1, 5, 5), padding=(0, 2,2),stride=(1, 1, 1)),
                                nn.Conv3d(16, 32, (1, 1, 1)), 
                                nn.BatchNorm3d(32, affine=False),
                                nn.ReLU(),
                                nn.Dropout(p=0.1, inplace=False),
                                nn.Conv3d(32, num_class,  (1, 1, 1)),) 
        self.clalast = nn.Sequential(
            nn.Conv2d(num_class*2, num_class*10,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_class*10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(num_class*10, num_class, kernel_size=(1, 1), stride=(1, 1)),
        )        
        self.cnnrgb=deeplabv3#rgb_fcn_resnet50#nn.Sequential(*list(rgb_fcn_resnet50.children()))
        self.encoder1=nn.Sequential(
            nn.Conv2d(band_num, band_num//2,kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(band_num//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),)
        self.encoder2=nn.Sequential(            
            nn.Conv2d(band_num//2,band_num//4,kernel_size=(1,1),bias=False),
            nn.BatchNorm2d(band_num//4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),)
        self.encoder3=nn.Sequential(  
            nn.Conv2d(band_num//4,3,kernel_size=(1,1)),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),)           
        self.SAM1=SPE_Attention_3D(4,B1,7)
        self.SAM2=SPE_Attention_3D(16,B2,7)
    def forward(self, inputx,inputrgb):
        x1=inputx.unsqueeze(0)
        x1=self.spe1(x1)
        x21=self.spe2(x1)+x1
        x21=self.SAM1(x21)+x21
        x31=self.spe3(x21)
        x41=self.spe4(x31)+x31
        x41=self.SAM2(x41)+x41
        xhe=self.spehe(x41)
        y1=self.spepre(xhe)        
        y1=y1.squeeze(0)

        output_spe=torch.transpose(y1,1,0)
        xrgb=self.encoder1(inputrgb)
        xrgb=self.encoder2(xrgb)
        xrgb=self.encoder3(xrgb)        
        xrgb=torch.cat((xrgb,xrgb),dim=0)
        yout=self.cnnrgb(xrgb)
        xrgbout=yout['out'][0].unsqueeze(0)
        feature_fusion=torch.cat((output_spe,xrgbout),1)
        outputlast=self.clalast(feature_fusion)
        return output_spe,xrgbout,xrgbout,outputlast

def DrawResult_img(Pred_result):
   False_color_resultimag=np.zeros((Pred_result.shape[0],Pred_result.shape[1],3),dtype=np.uint8)      
   for i in range(Pred_result.shape[0]):
      for j in range(Pred_result.shape[1]):
            item=Pred_result[i,j]                       
            if item == 1:
               False_color_resultimag[i,j,:] = np.array([255, 0, 0]) 
            if item == 2:
               False_color_resultimag[i,j,:]  = np.array([0, 255, 0])
            if item == 3:
               False_color_resultimag[i,j,:] = np.array([0, 0, 255]) 
            if item == 4:
               False_color_resultimag[i,j,:]  = np.array([255, 255, 0]) 
            if item == 5:
               False_color_resultimag[i,j,:]  = np.array([0, 255, 255]) 
            if item == 6:
               False_color_resultimag[i,j,:]  = np.array([255, 0, 255])
            if item == 7:
               False_color_resultimag[i,j,:] = np.array([192, 192, 192])
            if item == 8:
               False_color_resultimag[i,j,:] = np.array([128, 128, 128]) 
            if item == 9:
               False_color_resultimag[i,j,:]  = np.array([128, 0, 0]) 
            if item == 10:
               False_color_resultimag[i,j,:]  = np.array([128, 128, 0]) 
            if item == 11:
               False_color_resultimag[i,j,:]  = np.array([0, 128, 0]) 
            if item == 12:
               False_color_resultimag[i,j,:]  = np.array([128, 0, 128]) 
            if item == 13:
               False_color_resultimag[i,j,:] = np.array([0, 128, 128]) 
            if item == 14:
               False_color_resultimag[i,j,:]  = np.array([0, 0, 128]) 
            if item == 15:
               False_color_resultimag[i,j,:]  = np.array([255, 165, 0]) 
            if item == 16:
               False_color_resultimag[i,j,:]  = np.array([255, 215, 0]) 
   return False_color_resultimag
sumtrain1time=0
sumtest1time=0
sumtrain2time=0
sumtest2time=0
runtime=0
   
for curr_seed in [1231,1232,1233,1234,1235,1236,1237,1238,1239,1240]:
   torch.cuda.empty_cache()
   runtime+=1
   SEED=curr_seed
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   torch.cuda.manual_seed_all(SEED)
   torch.backends.cudnn.deterministic=True
   torch.backends.cudnn.benchmark = False

   Y=gt_IN
   num_class = np.max(Y)
   Y_train = np.zeros(Y.shape).astype('int')
   Y_test = np.zeros(Y.shape).astype('int')
   n_sample_train = 0
   n_sample_test = 0
   #
   Train_Number_perClass=np.zeros(num_class,dtype=int)
   for i in range(1,num_class+1):
       index = np.where(Y==i)  
       n_sample = len(index[0])  
       array = np.random.permutation(n_sample)  
       n_per =math.ceil(n_sample*train_ratio) 
       Train_Number_perClass[i-1]=n_per
       print(n_sample,n_per)
       if i==1:
           array1_train = index[0][array[:n_per]]
           array2_train = index[1][array[:n_per]]
           array1_test = index[0][array[n_per:]]
           array2_test = index[1][array[n_per:]]
       else:
           array1_train = np.concatenate((array1_train,index[0][array[:n_per]]))
           array2_train = np.concatenate((array2_train,index[1][array[:n_per]]))
           array1_test = np.concatenate((array1_test,index[0][array[n_per:]]))
           array2_test = np.concatenate((array2_test,index[1][array[n_per:]]))
       Y_train[index[0][array[:n_per]],index[1][array[:n_per]]] = i
       Y_test[index[0][array][n_per:],index[1][array[n_per:]]]=i
       n_sample_train += n_per
       n_sample_test += n_sample-n_per
       
   row,col,n_band = data_.shape
   y_train = Y_train-1 
   y_test = Y_test-1
   
   X = data_  
   X_train =torch.tensor(X, dtype=torch.float,device=device)
   X_train=X_train.unsqueeze(0).to(device)

   y_train0=y_train
   y_train =torch.tensor(y_train, dtype=torch.float,device=device)
   y_test0=y_test
   y_test =torch.tensor(y_test, dtype=torch.float,device=device)
   
   we2 = torch.tensor(Train_Number_perClass+1, dtype=torch.float32,device=device)
   we2 = 1. / torch.sqrt(we2)
   max_acc=0
   LR = 0.01     
   B1=math.ceil((band_num-6)/2)
   B2=math.ceil((B1-6)/2)
   cnn = CNN().to(device)

   Transfer_hsi_model= torch.load('FCN-HSI-SN-3D-Spe0.03_INtrain_seed1237_model_.pt')  
   cnn.spe1=Transfer_hsi_model.spe1
   cnn.spe2=Transfer_hsi_model.spe2
   cnn.spe3=Transfer_hsi_model.spe3
   cnn.spe4=Transfer_hsi_model.spe4

   loss_func =torch.nn.CrossEntropyLoss(we2)   
   loss_KL=torch.nn.KLDivLoss()

   maxrgb_acc=0
   maxhsi_acc=0
   maxlast_acc=0
   losslist_rgb=[]
   losslist_hsi=[]
   losslist_last=[]
    
   def train_model1(X_train,Xrgb_train,fun_ytrain,fun_ytest):     
      optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma = 0.9)
      loss_func =torch.nn.CrossEntropyLoss(we2)  
      max_acc=0
      for epoch in range(Epoch1):
          batch_x=Variable(X_train).to(device)
          batch_y=Variable(y_train).to(device).long() 
          cla_hsi,rgb_out,rgb_rec,clalast= cnn(batch_x,batch_x)            # cnn output   
 
          preds_last=clalast[:,0:num_class,array1_train,array2_train]
          preds_last=preds_last.squeeze(0)
          preds_last=preds_last.transpose(1,0)
                             
          preds_hsi=cla_hsi[:,0:num_class,array1_train,array2_train]
          preds_hsi=preds_hsi.squeeze(0)
          preds_hsi=preds_hsi.transpose(1,0)
                   
          preds_rgbout=rgb_out[:,0:num_class,array1_train,array2_train]
          preds_rgbout=preds_rgbout.squeeze(0)
          preds_rgbout=preds_rgbout.transpose(1,0)

          truth_train=batch_y[array1_train,array2_train]
          truth_train=truth_train.squeeze(0)
      
          correct=0
          correct2=0
          total_train=array1_train.shape[0]
          total_test=array1_test.shape[0]
      
          print(epoch)
          preds=clalast
          predicted=torch.argmax(preds,1).squeeze(0).squeeze(0)
          y_true=fun_ytrain
          correct+=(predicted[array1_train,array2_train]==y_true[array1_train,array2_train]).sum().item()          
          accuracy=correct/total_train
          print('train acc:{:.2%}'.format(accuracy))
          Htrain_acc.append(accuracy)
         
          y_true2=fun_ytest
          correct2+=(predicted[array1_test,array2_test]==y_true2[array1_test,array2_test]).sum().item()          
          accuracy2=correct2/total_test
          print('test acc:{:.2%}'.format(accuracy2))
          Hval_acc.append(accuracy2)
                   
          if accuracy2 > max_acc:
              max_acc = accuracy2
              print("save model")
              torch.save(cnn,method+str(train_ratio)+'_'+dataset+'train_seed'+str(curr_seed)+'_model_'+'.pt') 
 
          loss_last=loss_func(preds_last,truth_train)
          loss_hsi=loss_func(preds_hsi,truth_train)
          loss_rgbout=loss_func(preds_rgbout,truth_train)
          loss1=loss_rgbout+loss_hsi                                    
          if epoch % 10<5:
             optimizer.zero_grad()         
             loss1.backward()          
             optimizer.step() 
             scheduler.step()
             optimizer.param_groups[0]['lr']
             print("spe_spa_loss",loss1)
          else:                      
             optimizer.zero_grad()         
             loss_last.backward()          
             optimizer.step() 
             scheduler.step()
             optimizer.param_groups[0]['lr']
             print("last_loss",loss_last)
          
          print("1stage",loss_last)
          print('max_acc=',max_acc)
      return max_acc       
         
   tic1_train=time.time()       
   max_acc1stage=train_model1(X_train,X_train,y_train,y_test)
   toc1_train=time.time()   
   load_bestmodel= torch.load(method+str(train_ratio)+'_'+dataset+'train_seed'+str(curr_seed)+'_model_'+'.pt')  
   batch_x=Variable(X_train).to(device)
   tic1_test=time.time()
   cla_hsi,rgb_out,rgb_aux,clalast=load_bestmodel(batch_x,batch_x)
   #results 
   preds=clalast
   predicted=torch.argmax(preds,1).squeeze(0)
   tttp=(predicted.cpu().numpy()).astype(int)+1
   toc1_test=time.time()
   
   ttty=(y_test.cpu().numpy()).astype(int)+1
   ttttest=[]
   tttpred=[]
   for i in range (ttty.shape[0]):
       for j in range (ttty.shape[1]):
         if ttty[i,j]!=0:
             ttttest.append(ttty[i,j])
             tttpred.append(tttp[i,j])
   
   clares_1,confu_1,reslist_1=reports(tttpred,ttttest)
   res_seed_list1.append(reslist_1)

   label=np.reshape(tttp, (gt_IN.shape[0], gt_IN.shape[1]))
   False_color_resultimag=DrawResult_img(label)
   plt.imshow(False_color_resultimag) 
   plt.axis('off') 
   plt.show()

   OA=reslist_1[0]
   plt.imsave(method+str(train_ratio)+'_'+dataset+'train_seed'+str(curr_seed)+'_OA_'+repr(int(OA*10000))+'.png',False_color_resultimag)
   Y=gt_IN
   img_gt=False_color_resultimag.copy()
   for i in range(Y.shape[0]):
      for j in range(Y.shape[1]):
         if Y[i,j]==0:
            img_gt[i,j,:]=0
   plt.imshow(img_gt) 
   plt.axis('off') 
   plt.show()
   plt.imsave('GT_'+method+'_1STAGE_train'+str(train_ratio)+'_'+dataset+'train_seed'+str(curr_seed)+'_OA_'+repr(int(OA*10000))+'.png',img_gt)
   sio.savemat('GT_'+method+'_1STAGE_train'+str(train_ratio)+'_'+dataset+'train_seed'+str(curr_seed)+'_OA_'+repr(int(OA*10000))+'_ALLdata_predict.mat',{'data': label})
   
   train1_time=toc1_train-tic1_train
   test1_time=toc1_test-tic1_test
   train1stage_time.append(train1_time)
   test1stage_time.append(test1_time)
        
   file= open(method+str(train_ratio)+'_'+dataset+'_results.txt', 'a+') 
   str_train1_time='1_STAGE_RESULTS:train time:'+str(train1_time)
   str_test1_time='1_STAGE_RESULTS:test time:'+str(test1_time)

   file.write('class_train_num')
   file.write('\n')
   for fp in Train_Number_perClass:
           file.write(str(fp))
           file.write('\n')    

   file.write(str_train1_time)
   file.write('\n')
   for fp in train1stage_time:
           file.write(str(fp))
           file.write('\n')   
   file.write(str_test1_time)
   file.write('\n')           
   for fp in test1stage_time:
           file.write(str(fp))
           file.write('\n')
   file.write('1_stage:OA,AA,KAPPA,C_A')
   file.write('\n')
   for fp in res_seed_list1:
           file.write(str(fp))
           file.write('\n')
   file.close()
