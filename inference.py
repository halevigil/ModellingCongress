import torch
from prediction_model_analysis import test_dataset
state_dict = torch.load("outputs/models/08-04_withextras_bce_lr1e-5_beta1e-06/epoch200.pt")
weights = state_dict["model"]["weight"]
model=torch.nn.Linear(weights.shape[1],weights.shape[0])
def predict()
