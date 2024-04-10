
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score
import h3pandas
import torch
import h3
from sklearn.metrics import precision_recall_curve
import argparse
from tqdm import tqdm
import uncertainty_metrics.numpy as um
from sklearn.metrics import brier_score_loss
import logging

sys.path.append('../')
import datasets
import models
import utils
import setup

RESULT_DIR = './calibration_metrics_results/ensemble/an_full/'
MODEL_1_PATH = '../pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt'
MODEL_2_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
MODEL_3_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_-1/model.pt'

# RESULT_DIR = './calibration_metrics_results/ensemble/loss/'
# MODEL_1_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
# MODEL_2_PATH = '../pretrained_models/1000_cap_models/final_loss_an_slds_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
# MODEL_3_PATH = '../pretrained_models/1000_cap_models/final_loss_an_ssdl_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'

if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

# Set up logging to file
log_file_path = RESULT_DIR+"/log_v2.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

### MODEL 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_params = torch.load(MODEL_1_PATH, map_location='cpu')
model1 = models.get_model(train_params['params'])
model1.load_state_dict(train_params['state_dict'], strict=True)
model1 = model1.to(DEVICE)
model1.eval()

### MODEL 2
train_params = torch.load(MODEL_2_PATH, map_location='cpu')
model2 = models.get_model(train_params['params'])
model2.load_state_dict(train_params['state_dict'], strict=True)
model2 = model2.to(DEVICE)
model2.eval()

### MODEL 3 uncapped version
train_params = torch.load(MODEL_3_PATH, map_location='cpu')
model3 = models.get_model(train_params['params'])
model3.load_state_dict(train_params['state_dict'], strict=True)
model3 = model3.to(DEVICE)
model3.eval()

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

#load reference from iucn
with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
species_ids = list((data['taxa_presence'].keys()))

def generate_h3_cells_atRes(resolution=4):
    h3_cells = list(h3.get_res0_indexes())
    h3_atRes_cells = set()
    for cell in h3_cells:
        h3_atRes_cells = h3_atRes_cells.union(h3.h3_to_children(cell, resolution))
    return list(h3_atRes_cells)

obs_locs = np.array(data['locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

###Model 1
with torch.no_grad():
    loc_emb_1 = model1(loc_feat, return_feats=True)
    wt_1 = model1.class_emb.weight[classes_of_interest, :]

###Model 2
with torch.no_grad():
    loc_emb_2 = model2(loc_feat, return_feats=True)
    wt_2 = model2.class_emb.weight[classes_of_interest, :]

###Model 3
with torch.no_grad():
    loc_emb_3 = model3(loc_feat, return_feats=True)
    wt_3 = model3.class_emb.weight[classes_of_interest, :]

def fscore_and_thres(y_test, preds):
    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thres = thresholds[index]
    max_fscore = fscore[index]
    return max_fscore, thres

def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
        y_thresh = y_pred > thresh
        return f1_score(y_true, y_thresh, average=type)

output = []
for class_index, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    wt_column_1 = wt_1[class_index,:]
    preds1 = torch.sigmoid(torch.matmul(loc_emb_1, wt_column_1)).cpu().numpy()

    wt_column_2 = wt_2[class_index,:]
    preds2 = torch.sigmoid(torch.matmul(loc_emb_2, wt_column_2)).cpu().numpy()

    wt_column_3 = wt_3[class_index,:]
    preds3 = torch.sigmoid(torch.matmul(loc_emb_3, wt_column_3)).cpu().numpy()
    
    # Stack the tensors along a new axis
    stacked_tensors = np.stack([preds1, preds2, preds3])

    # Take the mean across the new axis
    preds = np.mean(stacked_tensors, axis=0)
    
    taxa = species_ids[class_index]
    species_locs = data['taxa_presence'].get(str(taxa))

    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1

    _, optimal_thres = fscore_and_thres(y_test, preds)
    fscore50 = f1_at_thresh(y_test,preds,0.5)

    row = {
        "taxon_id": taxa,
        "optimal_thres": optimal_thres,
        "fscore50": fscore50
    }
    row_dict = dict(row)
 
    output.append(row_dict)

output_pd = pd.DataFrame(output)
    
output_pd.to_csv(RESULT_DIR+"/scores_v2.csv")
logging.info(f"Mean fscore50: {output_pd.fscore50.mean()}")
logging.info(f"Std optimal_thres: {output_pd.optimal_thres.std()}")



