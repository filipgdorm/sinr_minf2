
import torch
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging

sys.path.append('../')
import datasets
import models
import utils

import argparse

parser = argparse.ArgumentParser(description='A script with flags')

# Define your flags
parser.add_argument('--log', action='store_true', help='Use logarithmic spacing', default = False)
parser.add_argument('--num_points', type=int, help='Number of points in thresh interval', default=19)
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--exp_name", type=str, default='test', help="Experiment name")
parser.add_argument("--result_dir", type=str, default='test', help="Result dir.")


args = parser.parse_args()

MODEL_PATH = '../pretrained_models/' + args.model_path
RESULT_DIR = f'./upper_b_1_thres_results/log_{str(args.log)}/'

if not os.path.exists(RESULT_DIR+args.exp_name):
        os.mkdir(RESULT_DIR+args.exp_name)

# Set up logging to file
log_file_path = RESULT_DIR+args.exp_name+"/log.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info(f"Model used for experiment: {args.model_path}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
train_params = torch.load(MODEL_PATH, map_location='cpu')
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to(DEVICE)
model.eval()

#load reference from iucn
with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
species_ids = list((data['taxa_presence'].keys()))


if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

obs_locs = np.array(data['locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
    y_thresh = y_pred > thresh
    return f1_score(y_true, y_thresh, average=type)


num_points = args.num_points

# Generate logarithmically spaced numbers in the range [0, 1]
geomspace_values = np.geomspace(0.01, 1, num=num_points, endpoint=False)

linspace_values = np.linspace(0.05, 1, num=num_points, endpoint=False)

if (args.log):
    threshs = geomspace_values
else:
    threshs = linspace_values

per_species_f1 = np.zeros((len(species_ids),len(threshs)))
for tt_id, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    taxa = species_ids[tt_id]
    wt_1 = wt[tt_id,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
    species_locs = data['taxa_presence'].get(str(taxa))
    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1

    for i, thresh in enumerate(threshs):
        per_species_f1[tt_id][i] = f1_at_thresh(y_test, preds, thresh, type='binary')

f1score = per_species_f1.mean(axis=0).max()
if args.log: best_thres = geomspace_values[per_species_f1.mean(axis=0).argmax()]
else: best_thres = linspace_values[per_species_f1.mean(axis=0).argmax()]

logging.info(f"Best unified threshold: {best_thres}")
logging.info(f"Mean f1 score: {f1score}")


np.save(RESULT_DIR+args.exp_name+f'/f1_scores.npy', per_species_f1)