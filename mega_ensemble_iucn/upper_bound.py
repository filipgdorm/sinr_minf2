import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

sys.path.append('../')
import datasets
import models
import utils
import setup
import logging

RESULT_DIR = "upper_bound_env"
# Directory containing model files
MODEL_DIR = '../five_models_env/an_full_env/'
DEVICE = torch.device('cpu')

# Set up logging to file
log_file_path = RESULT_DIR + f"/log.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Function to collect all model paths in the nested directory
def collect_model_paths(model_dir):
    model_paths = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pt'):
                model_paths.append(os.path.join(root, file))
    return model_paths

MODEL_PATHS = collect_model_paths(MODEL_DIR)

models_list = []
for model_path in MODEL_PATHS:
    train_params = torch.load(model_path, map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(DEVICE)
    model.eval()
    models_list.append(model)

#load reference from iucn
with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
species_ids = list(int(key) for key in (data['taxa_presence'].keys()))

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

obs_locs = np.array(data['locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(tt)])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

# Load models and store loc_emb and wt
loc_emb_list = []
wt_list = []

for model_path in MODEL_PATHS:
    train_params = torch.load(model_path, map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        loc_emb = model(loc_feat, return_feats=True)
        wt = model.class_emb.weight[classes_of_interest, :]
        loc_emb_list.append(loc_emb)
        wt_list.append(wt)
        
def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
    y_thresh = y_pred > thresh
    return f1_score(y_true, y_thresh, average=type)

per_species_f1 = np.zeros((len(species_ids)))
for tt_id, taxa in tqdm(enumerate(species_ids), total=len(species_ids)):
    predictions = []
    for loc_emb, wt in zip(loc_emb_list, wt_list):
        with torch.no_grad():
            wt_1 = wt[tt_id, :]
            preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
            predictions.append(preds)

    stacked_tensors = np.stack(predictions)
    preds = np.mean(stacked_tensors, axis=0)    
    species_locs = data['taxa_presence'].get(str(taxa))

    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1

    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    per_species_f1[tt_id] = fscore.max()

mean_f1 = np.mean(per_species_f1)
logging.info(f"Mean f1 score: {mean_f1}")
