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

RESULT_DIR = './tgt_background_ensemble_results/loss/'

# MODEL_1_PATH = '../pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt'
# MODEL_2_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
# MODEL_3_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_-1/model.pt'

MODEL_1_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
MODEL_2_PATH = '../pretrained_models/1000_cap_models/final_loss_an_slds_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
MODEL_3_PATH = '../pretrained_models/1000_cap_models/final_loss_an_ssdl_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'


if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

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
    
output_pd.to_csv(RESULT_DIR+f"/thresholds.csv")





