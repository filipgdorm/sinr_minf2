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

sys.path.append('../')
import datasets
import models
import utils
import setup

import argparse
import logging

# MODEL_1_PATH = '../pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt'
# MODEL_2_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
# MODEL_3_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_-1/model.pt'
# RESULT_DIR = './ensemble_results/an_full/'

MODEL_1_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
MODEL_2_PATH = '../pretrained_models/1000_cap_models/final_loss_an_slds_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
#MODEL_3_PATH = '../pretrained_models/1000_cap_models/final_loss_an_ssdl_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
RESULT_DIR = './ensemble_results/loss_2_models/'

if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

# Set up logging to file
log_file_path = RESULT_DIR+"/log.out"
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

# ### MODEL 3 uncapped version
# train_params = torch.load(MODEL_3_PATH, map_location='cpu')
# model3 = models.get_model(train_params['params'])
# model3.load_state_dict(train_params['state_dict'], strict=True)
# model3 = model3.to(DEVICE)
# model3.eval()

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

def convert_to_inat_vision_order(geo_pred_ip, vision_top_k_prob, vision_top_k_inds, vision_taxa, taxon_map):
        # this is slow as we turn the sparse input back into the same size as the dense one
        vision_pred = np.zeros((geo_pred_ip.shape[0], len(vision_taxa)), dtype=np.float32)
        geo_pred = np.ones((geo_pred_ip.shape[0], len(vision_taxa)), dtype=np.float32)
        vision_pred[np.arange(vision_pred.shape[0])[..., np.newaxis], vision_top_k_inds] = vision_top_k_prob

        geo_pred[:, taxon_map[:, 0]] = geo_pred_ip[:, taxon_map[:, 1]]

        return geo_pred, vision_pred


with open('paths.json', 'r') as f:
            paths = json.load(f)
# load vision model predictions:
data = np.load(os.path.join(paths['geo_prior'], 'geo_prior_model_preds.npz'))
print(data['probs'].shape[0], 'total test observations')
# load locations:
meta = pd.read_csv(os.path.join(paths['geo_prior'], 'geo_prior_model_meta.csv'))
obs_locs  = np.vstack((meta['longitude'].values, meta['latitude'].values)).T.astype(np.float32)
# taxonomic mapping:
taxon_map = find_mapping_between_models(data['model_to_taxa'], train_params['params']['class_to_taxa'])
print(taxon_map.shape[0], 'out of', len(data['model_to_taxa']), 'taxa in both vision and geo models')

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

#def run_evaluation(model1, model2, model3, enc):
def run_evaluation(model1, model2, enc):
        results = {}

        # loop over in batches
        batch_start = np.hstack((np.arange(0, data['probs'].shape[0], 2048), data['probs'].shape[0]))
        correct_pred = np.zeros(data['probs'].shape[0])

        for bb_id, bb in tqdm(enumerate(range(len(batch_start)-1)), total=len(batch_start)-1):
            batch_inds = np.arange(batch_start[bb], batch_start[bb+1])

            vision_probs = data['probs'][batch_inds, :]
            vision_inds = data['inds'][batch_inds, :]
            gt = data['labels'][batch_inds]

            obs_locs_batch = torch.from_numpy(obs_locs[batch_inds, :]).to('cpu')
            loc_feat = enc.encode(obs_locs_batch)

            with torch.no_grad():
                geo_pred1 = model1(loc_feat).cpu().numpy()
                geo_pred2 = model2(loc_feat).cpu().numpy()
                #geo_pred3 = model3(loc_feat).cpu().numpy()
            # Stack the tensors along a new axis
            #stacked_tensors = np.stack([geo_pred1, geo_pred2, geo_pred3])
            stacked_tensors = np.stack([geo_pred1, geo_pred2])

            # Take the mean across the new axis
            geo_pred = np.mean(stacked_tensors, axis=0)

            geo_pred, vision_pred = convert_to_inat_vision_order(geo_pred, vision_probs, vision_inds,
                                                                 data['model_to_taxa'], taxon_map)

            comb_pred = np.argmax(vision_pred*geo_pred, 1)
            comb_pred = (comb_pred==gt)
            correct_pred[batch_inds] = comb_pred

        results['vision_only_top_1'] = float((data['inds'][:, -1] == data['labels']).mean())
        results['vision_geo_top_1'] = float(correct_pred.mean())
        return results

#results = run_evaluation(model1, model2, model3, enc)
results = run_evaluation(model1, model2, enc)


def report(results):
        print('Overall accuracy vision only model', round(results['vision_only_top_1'], 3))
        print('Overall accuracy of geo model     ', round(results['vision_geo_top_1'], 3))
        print('Gain                              ', round(results['vision_geo_top_1'] - results['vision_only_top_1'], 3))
        res1 = round(results['vision_only_top_1'], 3)
        res2 = round(results['vision_geo_top_1'], 3)
        res3 = round(results['vision_geo_top_1'] - results['vision_only_top_1'], 3)
        logging.info(f'Overall accuracy vision only model {res1}')
        logging.info(f'Overall accuracy of geo model      {res2}')
        logging.info(f'Gain                               {res3}')

report(results)

