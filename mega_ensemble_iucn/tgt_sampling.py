
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
import csv

sys.path.append('../')
import datasets
import models
import utils
import setup

RESULT_DIR = "tgt_sampling_iucn"

#load relevant data
train_df_h3 = pd.read_csv("../pseudo_absence_generation_data/train_df_h3.csv", index_col=0)
gdfk = pd.read_csv("../pseudo_absence_generation_data/gdfk_res3.csv", index_col=0)
all_spatial_grid_counts = train_df_h3.index.value_counts()
presence_absence = pd.DataFrame({
    "background": all_spatial_grid_counts,
})
presence_absence = presence_absence.fillna(0)

# Directory containing model files
MODEL_DIR = '../five_models/'

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

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

obs_locs = np.array(gdfk[['lng', 'lat']].values, dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

#load reference from iucn
with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
species_ids = list((data['taxa_presence'].keys()))

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
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

with open(RESULT_DIR+f"/thresholds.csv", mode='w', newline='') as file:
    fieldnames = ["taxon_id", "thres"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header

    for class_index, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
        predictions = []

        for loc_emb, wt in zip(loc_emb_list, wt_list):
            with torch.no_grad():
                wt_1 = wt[class_index, :]
                preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
                predictions.append(preds)

        stacked_tensors = np.stack(predictions)
        preds = np.mean(stacked_tensors, axis=0)

        gdfk["pred"] = preds

        target_spatial_grid_counts = train_df_h3[train_df_h3.label==class_id.item()].index.value_counts()

        presence_absence["forground"] = target_spatial_grid_counts
        presence_absence["predictions"] = gdfk["pred"]
        presence_absence.forground = presence_absence.forground.fillna(0)
        yield_cutoff = np.percentile((presence_absence["background"]/presence_absence["forground"])[presence_absence["forground"]>0], 95)
        absences = presence_absence[(presence_absence["forground"]==0) & (presence_absence["background"] > yield_cutoff)]["predictions"]
        presences = presence_absence[(presence_absence["forground"]>0)]["predictions"]
        df_x = pd.DataFrame({'predictions': presences, 'test': 1})
        df_y = pd.DataFrame({'predictions': absences, 'test': 0})
        for_thres = pd.concat([df_x, df_y], ignore_index=False)
        precision, recall, thresholds = precision_recall_curve(for_thres.test, for_thres.predictions)
        p1 = (2 * precision * recall)
        p2 = (precision + recall)
        out = np.zeros( (len(p1)) )
        fscore = np.divide(p1,p2, out=out, where=p2!=0)
        index = np.argmax(fscore)
        thres = thresholds[index]
        max_fscore = fscore[index]
        
        row = {
            "taxon_id": species_ids[class_index],
            "thres": thres,
        }
        writer.writerow(row)


