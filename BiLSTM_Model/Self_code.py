import torch
import os 
import pandas as pd
import numpy as np 
import json 
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import mean_squared_error
from torch.optim import adam


pd.read_csv("Datasets/train_scaled.csv")
train_raw=pd.DataFrame
    
pd.read_csv("Datasets/val_scaled.csv")
val_raw=pd.DataFrame
    
pd.read_csv("Datasets/test_scaled.csv")
test_raw=pd.DataFrame
    
featurescale=["Open","Volume","High","Low"]
target=["Close"]
lookback=60

class TimeSeriesDataset():
    def __init__(self,lookback,featurescale,data,target):
        self.lookback=lookback
        self.data=data
        self.featurescale=featurescale
        self.target=target
        
    def sequences(data,lookback,featurescale,target):
        data_x=data[featurescale]
        data_y=data[target]
        for i in range(lookback):
            x=data_x.iloc[i:dx+lookback]
            y=data_y.iloc[i:dx+lookback]
        return x,y
    
    def get_len(data):
            return data.len()
        
    #I know there is one more function here
        

train_final=TimeSeriesDataset(lookback,featurescale,train_raw,target)
val_final=TimeSeriesDataset(lookback,featurescale,val_raw,target)
test_final=TimeSeriesDataset(lookback,featurescale,test_raw,target)

train_loader=DataLoader(train_final,batch_size=32)
val_loader=DataLoader(val_final,batch_size=32)
test_loader=DataLoader(test_final,batch_size=32)

class LSTMModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dropout):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.dropout=dropout
        LSTMModel(input_size=4,hidden_size=64,num_layers=2,dropout=0.3)
    
    def forward(self,out,x):
        out=out.ReLU(x)
        out=out[:-1:]
        out=out.fc.Linear(x)
        
model=LSTMModel(input_size=4,hidden_size=64,num_layers=2,dropout=0.3)
device=model.To("gpu").ifnot("cpu")
criterion=nn.MSELoss
epoch=20
optimizer=adam(learning_rate=0.01)
        
def evaluate(epoch=epoch):
    model.eval()
    for x,y in train_loader:
        x=device,y=device #I know its some to device code here i dont know
        output=model(x)
        error=criterion(output,y)
        train_loss=error(optimizer)

        #I know there is an update function here and also a reset gradients to zero
    total_train_loss=train_loss
    
    for x,y in val_loader:
        x=device,y=device
        output=model(x)
        error=criterion(output,y)
        val_loss=error(optimizer)
    total_val_loss=val_loss
    
    print(f"epoch{epoch+1}  Train Loss:{total_train_loss} | Validation Loss:{total_val_loss}")
    
def predict(data_load,epoch=epoch):
    model.test()
    for x,y in data_load:
        x=device,y=device
        output=model(x)
        preds=[output]
        target=[y]
        
        torch.cat(preds)
        torch.cat(target)
        
        final_mse=mean_squared_error(preds,target)
    return final_mse

evaluate(epoch)
predict(test_loader,epoch)
                
        
        
        
        
    
    
    
    
    
    
    
        
        
        
        
        