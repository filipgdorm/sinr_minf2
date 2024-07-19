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

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--result_dir", type=str, default='test', help="Experiment name")
parser.add_argument("--counter", type=int, default='test', help="Experiment name")

args = parser.parse_args()

print(args.counter, args.result_dir, args.model_path)

threshs = pd.read_csv(args.result_dir + f"/thresholds/thresholds_{args.counter}.csv")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up logging to file
log_file_path = args.result_dir + f"/results/log_{args.counter}.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info(f"Model used for experiment: {args.model_path}")

# load model
train_params = torch.load(args.model_path, map_location='cpu')
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to(DEVICE)
model.eval()

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

def run_evaluation(model, enc):
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
            loc_emb = model(loc_feat, return_feats=True)
            wt = model.class_emb.weight.detach().clone()
            wt.requires_grad = False
        raw_preds = torch.matmul(loc_emb, wt.T)
        preds = torch.sigmoid(raw_preds)
        #geo_pred = np.where(preds.numpy() > threshs['thres'].values, 1.0, 0.01)
        geo_pred = preds.numpy() > threshs['thres'].values

        geo_pred, vision_pred = convert_to_inat_vision_order(geo_pred, vision_probs, vision_inds,
                                                                data['model_to_taxa'], taxon_map)
        comb_pred = np.argmax(vision_pred*geo_pred, 1)
        comb_pred = (comb_pred==gt)
        correct_pred[batch_inds] = comb_pred

    results['vision_only_top_1'] = float((data['inds'][:, -1] == data['labels']).mean())
    results['vision_geo_top_1'] = float(correct_pred.mean())

    return results
results = run_evaluation(model, enc)

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

        # Append the mean F1 score to a CSV file
        results_file = args.result_dir + '/mean_f1_scores.csv'
        results_data = pd.DataFrame({'counter': [args.counter], 'vision_only_top_1': [res1], 'vision_geo_top_1': [res2], 'gain': [res3] })

        if os.path.isfile(results_file):
            results_data.to_csv(results_file, mode='a', header=False, index=False)
        else:
            results_data.to_csv(results_file, mode='w', header=True, index=False)

report(results)
