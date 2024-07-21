import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score
import torch
from sklearn.metrics import precision_recall_curve
import argparse
from tqdm import tqdm

sys.path.append('../')
import datasets
import models
import utils
import setup
import logging

RESULT_DIR = "lpt_an_full"
# Directory containing model files
#MODEL_DIR = '../five_models/'
MODEL_DIR = '../five_models/an_full_1000/'

threshs = pd.read_csv(RESULT_DIR + f"/thresholds.csv")


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

DEVICE = torch.device('cpu')
models_list = []
for model_path in MODEL_PATHS:
    train_params = torch.load(model_path, map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(DEVICE)
    model.eval()
    models_list.append(model)

#load reference from snt
data2 = np.load(os.path.join('../data/eval/snt/', 'snt_res_5.npy'), allow_pickle=True)
data2 = data2.item()
species_ids = data2['taxa']
loc_indices_per_species = data2['loc_indices_per_species']
labels_per_species = data2['labels_per_species']

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

obs_locs = np.array(data2['obs_locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(threshs.taxon_id), dtype=torch.int64)
for tt_id, tt in enumerate(threshs.taxon_id):
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

per_species_f1 = np.zeros((len(threshs.taxon_id)))
for tt_id, taxa in tqdm(enumerate(threshs.taxon_id), total=len(threshs.taxon_id)):
    predictions = []
    for loc_emb, wt in zip(loc_emb_list, wt_list):
        with torch.no_grad():
            wt_1 = wt[tt_id, :]
            preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
            predictions.append(preds)

    stacked_tensors = np.stack(predictions)
    preds = np.mean(stacked_tensors, axis=0)

    # generate ground truth labels for current taxa
    cur_loc_indices = np.array(loc_indices_per_species[tt_id])
    cur_labels = np.array(labels_per_species[tt_id])

    pred = preds[cur_loc_indices]

    thresh = threshs['thres'][tt_id]
    per_species_f1[tt_id] = f1_at_thresh(cur_labels, pred, thresh, type='binary')

mean_f1 = np.mean(per_species_f1)
logging.info(f"Mean f1 score: {mean_f1}")
