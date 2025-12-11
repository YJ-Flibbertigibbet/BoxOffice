import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
import model
import data

def test_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Model = model.imgResTrain().to(device)

    Model.load_state_dict(torch.load(model_path))
    Model.eval()
    
    dataset = data.dataset()
    _, testDataset = data.split_within_year_month(dataset)
    testDataloader = DataLoader(testDataset,
                              batch_size=config.batchSize,
                              shuffle=False)  
    result = pd.DataFrame(columns=["real_box", "predict_box", "mse"])
    

    with torch.no_grad():  
        for batch_idx, (input_batch, label_batch) in enumerate(testDataloader):

            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)
            
            predict_batch = Model(input_batch)
            
            for i in range(len(label_batch)):
                loss = nn.MSELoss()(predict_batch[i], label_batch[i]).item()
                
                row = pd.DataFrame({
                    "real_box": [label_batch[i].cpu().numpy().tolist()],
                    "predict_box": [predict_batch[i].cpu().numpy().tolist()],
                    "mse": [loss]
                })
                
                result = pd.concat([result, row], ignore_index=True)
            
            print(f"Processed batch {batch_idx+1}/{len(testDataloader)}")
    
    return result

if __name__ == "__main__":
    result_df = test_model("/home/juan/桌面/NKU/数据分析/box_office/res_fc/trainpth/model_res_img_1.pth")

    result_df.to_csv("test_results.csv", index=False)
 
    avg_mse = result_df["mse"].mean()
    print(f"Average MSE: {avg_mse}")

    print("\nFirst 5 rows:")
    print(result_df.head())