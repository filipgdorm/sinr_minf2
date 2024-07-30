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

DEVICE = torch.device('cpu')

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

#load reference from snt
data2 = np.load(os.path.join('../data/eval/snt/', 'snt_res_5.npy'), allow_pickle=True)
data2 = data2.item()
species_ids2 = data2['taxa']
loc_indices_per_species = data2['loc_indices_per_species']
labels_per_species = data2['labels_per_species']

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

obs_locs = np.array(data2['obs_locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids2), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids2):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(tt)])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
    y_thresh = y_pred > thresh
    return f1_score(y_true, y_thresh, average=type)

per_species_f1 = np.zeros((len(classes_of_interest)))
for tt_id, taxa in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    wt_1 = wt[tt_id,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()

    # generate ground truth labels for current taxa
    cur_loc_indices = np.array(loc_indices_per_species[tt_id])
    cur_labels = np.array(labels_per_species[tt_id])

    pred = preds[cur_loc_indices]

    precision, recall, thresholds = precision_recall_curve(cur_labels, pred)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    per_species_f1[tt_id] = fscore.max()

mean_f1 = np.mean(per_species_f1)
logging.info(f"Mean f1 score: {mean_f1}")
np.save(args.result_dir+f'/results/f1_scores_{args.counter}.npy', per_species_f1)

# Append the mean F1 score to a CSV file
results_file = args.result_dir + '/mean_f1_scores.csv'
results_data = pd.DataFrame({'counter': [args.counter], 'mean_f1': [mean_f1]})

if os.path.isfile(results_file):
    results_data.to_csv(results_file, mode='a', header=False, index=False)
else:
    results_data.to_csv(results_file, mode='w', header=True, index=False)