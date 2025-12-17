from torch.utils.data import DataLoader
import model
import data
import torch
import torch.nn as nn
from torch import optim
import config

def resTrain(split=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Dataset=data.dataset()
    if split:
        datasetTrain,datasetTest=data.split_within_year_month(Dataset)
    else:
        datasetTrain=Dataset

    trainDataloader=DataLoader(datasetTrain,
                               batch_size=config.batchSize,
                               shuffle=True)

    Model=model.imgResTrain().to(device)

    criterion=nn.MSELoss()
    
    optimizer=optim.Adam(Model.parameters(),lr=config.lr)

    for epoch in range(config.epochs):
        Model.train()
        running_loss=0.00
        for inputs,labels in trainDataloader:
            inputs,labels=inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            outputs=Model(inputs)

            loss=criterion(outputs,labels)

            running_loss += loss.item()

            loss.backward()

            optimizer.step()

            print(f"当前误差: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {running_loss/len(trainDataloader):.4f}")

    print("训练完成")
    torch.save(Model.state_dict(),"model_res_img.pth")
