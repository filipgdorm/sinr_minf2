
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
from tqdm import tqdm
import uncertainty_metrics.numpy as um
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import pickle

sys.path.append('../')
import datasets
import models
import utils
import setup

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_params = torch.load('../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_-1/model.pt', map_location='cpu')
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to(DEVICE)
model.eval()

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

with open('paths.json', 'r') as f:
            paths = json.load(f)
# load vision model predictions:
data = np.load(os.path.join(paths['geo_prior'], 'geo_prior_model_preds.npz'))

def find_mapping_between_models(vision_taxa, geo_taxa):
    # this will output an array of size N_overlap X 2
    # the first column will be the indices of the vision model, and the second is their
    # corresponding index in the geo model
    taxon_map = np.ones((vision_taxa.shape[0], 2), dtype=np.int32)*-1
    taxon_map[:, 0] = np.arange(vision_taxa.shape[0])
    geo_taxa_arr = np.array(geo_taxa)
    for tt_id, tt in enumerate(vision_taxa):
        ind = np.where(geo_taxa_arr==tt)[0]
        if len(ind) > 0:
            taxon_map[tt_id, 1] = ind[0]
    inds = np.where(taxon_map[:, 1]>-1)[0]
    taxon_map = taxon_map[inds, :]
    return taxon_map

taxon_map = find_mapping_between_models(data['model_to_taxa'], train_params['params']['class_to_taxa'])

#find unique species we need to train calibrators for:
uniq_vision_ids = np.unique(data['labels'])

# Find indices where values in second_array match values in the first column of first_array
indices = np.where(np.isin(taxon_map[:,0], uniq_vision_ids))
vision_ids = taxon_map[indices][:, 0]
geo_ids = taxon_map[indices][:, 1]

train_params = {}

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

classes_of_interest = torch.tensor(geo_ids, dtype=torch.int64)

with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
h3_atRes_cells = [h3.geo_to_h3(coord[1], coord[0], resolution=5) for coord in data['locs']]
gdfk = pd.DataFrame(index=h3_atRes_cells)
gdfk["lng"] = np.array(data['locs'])[:,0]
gdfk["lat"] = np.array(data['locs'])[:,1]
gdfk = gdfk.rename_axis('h3index')

obs_locs = np.array(gdfk[['lng', 'lat']].values, dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]


output = []
for class_index, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    wt_1 = wt[class_index,:]
    raw_preds = torch.matmul(loc_emb, wt_1)
    preds = torch.sigmoid(raw_preds).cpu().numpy()

    ###generate pseudo absences for calibration
    gdfk["pred"] = preds
    gdfk["raw_preds"] = raw_preds
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

    ###calibration
    logistic_regression = LogisticRegression(random_state=42)
    logistic_regression.fit(for_thres.raw_preds.values.reshape(-1, 1), for_thres.test.values)  # true_labels are the true labels corresponding to raw_predictions
    model_filename = f'./calibrators/platt/{vision_ids[class_index]}_calibrator.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(logistic_regression, file)

    ###calibration
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(for_thres.predictions.values.reshape(-1, 1), for_thres.test.values)
    model_filename = f'./calibrators/isotonic/{vision_ids[class_index]}_calibrator.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(ir, file)
