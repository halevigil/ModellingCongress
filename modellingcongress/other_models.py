from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
import json
import os
import pickle as pkl
from prepare_inputs_outputs import CreateInputsOutputs

input_vecs = np.load("outputs/preprocess5/input_vectors.npy")
with open("outputs/preprocess5/output_generics.json","r") as file:
  output_generics = json.load(file)
print(input_vecs.shape)
print(len(output_generics))
creator = CreateInputsOutputs("outputs/preprocess5/inference")
gradboost = HistGradientBoostingClassifier(verbose=1)
gradboost.fit(input_vecs,output_generics)
os.makedirs("outputs/preprocess5/models",exist_ok=True)
with open("outputs/preprocess5/models/histgradboost.pkl","wb") as file:
  pkl.dump(gradboost,file)
with open("outputs/preprocess5/models/histgradboost.pkl","rb") as file:
  gradboost=pkl.load(file)
generics = creator.get_generics()
transformed=gradboost.predict_proba(input_vecs[:1000])
uncertain_transformed = [x for x in transformed if np.max(x)<1-1e-5]
uncertain_probs = [creator.vector_to_probabilities(v,gradboost.classes_) for v in uncertain_transformed]
display(uncertain_probs)
