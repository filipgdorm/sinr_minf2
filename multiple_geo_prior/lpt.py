
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

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--result_dir", type=str, default='test', help="Experiment name")
parser.add_argument("--counter", type=int, default='test', help="Experiment name")

args = parser.parse_args()

train_df_h3 = pd.read_csv("../pseudo_absence_generation_data/train_df_h3.csv", index_col=0)
gdfk = pd.read_csv("../pseudo_absence_generation_data/gdfk_res3.csv", index_col=0)
all_spatial_grid_counts = train_df_h3.index.value_counts()
presence_absence = pd.DataFrame({
    "background": all_spatial_grid_counts,
})
presence_absence = presence_absence.fillna(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_params = torch.load(args.model_path, map_location='cpu')
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to(DEVICE)
model.eval()

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

obs_locs = np.array(gdfk[['lng', 'lat']].values, dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight.detach().clone()
    wt.requires_grad = False

with open(args.result_dir+f"/thresholds_{args.counter}.csv", mode='w', newline='') as file:
    fieldnames = ["thres"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header

    for class_index, class_id in tqdm(enumerate(range(wt.shape[0])), total=wt.shape[0]):
        wt_1 = wt[class_index,:]
        preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
        gdfk["pred"] = preds

        target_spatial_grid_counts = train_df_h3[train_df_h3.label==class_id].index.value_counts()
            
        presence_absence["forground"] = target_spatial_grid_counts
        presence_absence["predictions"] = gdfk["pred"]
        presence_absence.forground = presence_absence.forground.fillna(0)

        presences = presence_absence[(presence_absence["forground"]>0)]["predictions"]

        thres = np.min(presences)
        
        row = {
            "thres": thres,
        }
        writer.writerow(row)
        