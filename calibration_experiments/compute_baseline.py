
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
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import uncertainty_metrics.numpy as um
from tqdm import tqdm
import pickle

sys.path.append('../')
import datasets
import models
import utils
import setup

from sklearn.linear_model import LogisticRegression
import argparse
import logging

MODEL_PATH = "../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_-1/model.pt"
RESULT_DIR = "./uncalibrated_baseline/"

# Set up logging to file
log_file_path = RESULT_DIR+"log.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info(f"Model used for experiment: {MODEL_PATH}")

train_params = {}

train_params['experiment_name'] = 'demo' # This will be the name of the directory where results for this run are saved.
train_params['species_set'] = 'all'
train_params['hard_cap_num_per_class'] = 1000
train_params['num_aux_species'] = 0
train_params['input_enc'] = 'sin_cos'
train_params['loss'] = 'an_full'

params = setup.get_default_params_train(train_params)

train_dataset = datasets.get_train_data(params)

train_df = pd.DataFrame(train_dataset.locs, columns=['lng','lat'])
train_df['lng'] = train_df['lng']*180
train_df['lat'] = train_df['lat']*90
train_df['label'] = train_dataset.labels

h3_resolution = 5
train_df_h3 = train_df.h3.geo_to_h3(h3_resolution)
all_spatial_grid_counts = train_df_h3.index.value_counts()
presence_absence = pd.DataFrame({
    "background": all_spatial_grid_counts,
})
presence_absence = presence_absence.fillna(0)

resolution = h3_resolution
area = h3.hex_area(resolution)

# load model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_params = torch.load(MODEL_PATH, map_location='cpu')
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to(DEVICE)
model.eval()

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

#load reference from iucn
with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
species_ids = list((data['taxa_presence'].keys()))

h3_atRes_cells = [h3.geo_to_h3(coord[1], coord[0], resolution=5) for coord in data['locs']]
gdfk = pd.DataFrame(index=h3_atRes_cells)
gdfk["lng"] = np.array(data['locs'])[:,0]
gdfk["lat"] = np.array(data['locs'])[:,1]
gdfk = gdfk.rename_axis('h3index')

obs_locs = np.array(gdfk[['lng', 'lat']].values, dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
taxa_ids = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)
    

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

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
    wt_1 = wt[class_index,:]
    raw_preds = torch.matmul(loc_emb, wt_1)
    preds = torch.sigmoid(raw_preds).cpu().numpy()

    gdfk["pred"] = preds
    gdfk["raw_preds"] = raw_preds
    ###eval
    taxa = train_dataset.class_to_taxa[class_id.item()]
    species_locs = data['taxa_presence'].get(str(taxa))
    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1

    target_spatial_grid_counts = train_df_h3[train_df_h3.label==class_id.item()].index.value_counts()

    presence_absence["forground"] = target_spatial_grid_counts
    presence_absence["predictions"] = gdfk["pred"]
    presence_absence["raw_preds"] = gdfk["raw_preds"]
    presence_absence.forground = presence_absence.forground.fillna(0)
    yield_cutoff = np.percentile((presence_absence["background"]/presence_absence["forground"])[presence_absence["forground"]>0], 95)
    absences = presence_absence[(presence_absence["forground"]==0) & (presence_absence["background"] > yield_cutoff)][["predictions", "raw_preds"]]
    presences = presence_absence[(presence_absence["forground"]>0)][["predictions", "raw_preds"]]
    df_x = pd.DataFrame({'predictions': presences['predictions'],'raw_preds': presences['raw_preds'], 'test': 1})
    df_y = pd.DataFrame({'predictions': absences['predictions'],'raw_preds': absences['raw_preds'], 'test': 0})
    for_thres = pd.concat([df_x, df_y], ignore_index=False)
    precision, recall, thresholds = precision_recall_curve(for_thres.test, for_thres.predictions)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    tgt_thres = thresholds[index]
    tgt_fscore = f1_at_thresh(y_test, preds, tgt_thres, type='binary')
    
    row = {
        "taxon_id": taxa,
        "tgt_thres": tgt_thres,
        "tgt_fscore": tgt_fscore,
    }
    row_dict = dict(row)
    output.append(row_dict)
    

output_pd = pd.DataFrame(output)
output_pd.to_csv(RESULT_DIR+"scores.csv")

logging.info(f"Mean tgt_thres: {output_pd.tgt_thres.mean()}")
logging.info(f"Mean tgt_fscore: {output_pd.tgt_fscore.mean()}")
