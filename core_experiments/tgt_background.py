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

sys.path.append('../')
import datasets
import models
import utils
import setup

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--exp_name", type=str, default='test', help="Experiment name")

args = parser.parse_args()

MODEL_PATH = '../pretrained_models/' + args.model_path
RESULT_DIR = './tgt_background_results/'

if not os.path.exists(RESULT_DIR+args.exp_name):
        os.mkdir(RESULT_DIR+args.exp_name)

train_params = {}

train_params['species_set'] = 'all'
train_params['hard_cap_num_per_class'] = 1000
train_params['num_aux_species'] = 0
train_params['input_enc'] = 'sin_cos'

params = setup.get_default_params_train(train_params)

train_dataset = datasets.get_train_data(params)

train_df = pd.DataFrame(train_dataset.locs, columns=['lng','lat'])
train_df['lng'] = train_df['lng']*180
train_df['lat'] = train_df['lat']*90
train_df['label'] = train_dataset.labels

h3_resolution = 4
train_df_h3 = train_df.h3.geo_to_h3(h3_resolution)
all_spatial_grid_counts = train_df_h3.index.value_counts()
presence_absence = pd.DataFrame({
    "background": all_spatial_grid_counts,
})
presence_absence = presence_absence.fillna(0)

resolution = h3_resolution
area = h3.hex_area(resolution)

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

def generate_h3_cells_atRes(resolution=4):
    h3_cells = list(h3.get_res0_indexes())
    h3_atRes_cells = set()
    for cell in h3_cells:
        h3_atRes_cells = h3_atRes_cells.union(h3.h3_to_children(cell, resolution))
    return list(h3_atRes_cells)

#generate gdfk table
h3_atRes_cells = generate_h3_cells_atRes(h3_resolution)
gdfk = pd.DataFrame(index=h3_atRes_cells).h3.h3_to_geo()
gdfk["lng"] = gdfk["geometry"].x
gdfk["lat"] = gdfk["geometry"].y
_ = gdfk.pop("geometry")
gdfk = gdfk.rename_axis('h3index')


obs_locs = np.array(gdfk[['lng', 'lat']].values, dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]


output = []
for class_index, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    wt_1 = wt[class_index,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
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
        "taxon_id": train_dataset.class_to_taxa[class_id.item()],
        "thres": thres,
        "area": len(gdfk[gdfk.pred >= thres])*area,
        "pseudo_fscore": max_fscore
    }
    row_dict = dict(row)
    output.append(row_dict)

output_pd = pd.DataFrame(output)
    
output_pd.to_csv(RESULT_DIR+args.exp_name+f"/thresholds.csv")





