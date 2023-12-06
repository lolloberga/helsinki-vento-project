
import torch
import pandas as pd 
import streamlit as st
from model.smoke_ann import SmokeNeuralNetwork
from model.stroke_ann import StrokeNeuralNetwork

SEED = 3007 
torch.manual_seed(SEED) 
torch.cuda.manual_seed(SEED)

# @st.cache
def smoke_model_load():
    model = SmokeNeuralNetwork(22, 1, 100, 50)
    model.load_state_dict(torch.load('model/checkpoints/Smoke-ANN/Smoke-ANN_2023-12-06_09-21.pt'))
    model.eval()
    return model 

def stroke_model_load():
    model = StrokeNeuralNetwork(22, 1, 100, 50)
    model.load_state_dict(torch.load('model/checkpoints/Smoke-ANN/Smoke-ANN_2023-12-06_09-21.pt'))
    model.eval()
    return model 

# @st.cache
def inference(input, model):
    tensor = torch.tensor(input, dtype=torch.float)
    output = model(tensor)
    return output

df = pd.read_csv('./resources/dataset/smoking_train_dataset.csv').drop('smoking', axis=1)
print(df.iloc[0])
smoke_model = smoke_model_load()

out = inference(df.iloc[0], smoke_model)
print("PREDICTION",out.round().item())
