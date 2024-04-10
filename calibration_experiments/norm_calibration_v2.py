
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

sys.path.append('../')
import datasets
import models
import utils
import setup

import argparse
import logging

parser = argparse.ArgumentParser(description='A script with flags')

# Define your flags
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--exp_name", type=str, default='test', help="Experiment name")
parser.add_argument("--gamma", type=float, default=1, help="Gamma parameter")

args = parser.parse_args()

MODEL_PATH = '../pretrained_models/' + args.model_path
RESULT_DIR = f'./norm_calibration_results/'

if not os.path.exists(RESULT_DIR+args.exp_name+"/"+str(args.gamma)):
        os.mkdir(RESULT_DIR+args.exp_name+"/"+str(args.gamma))

# Set up logging to file
log_file_path = RESULT_DIR+args.exp_name+"/"+str(args.gamma)+"/log_v2.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info(f"Model used for experiment: {args.model_path}")

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

obs_locs = np.array(data['locs'], dtype=np.float32)
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

freq_df = pd.read_csv('../data/train/geo_prior_train.csv')

counts = freq_df.taxon_id.value_counts()

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

def norm_cal_activation(preds,N,gamma):
    ac = N**gamma
    return (np.exp(preds)/ ac) / ((np.exp(preds) / ac) + 1)

def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
        y_thresh = y_pred > thresh
        return f1_score(y_true, y_thresh, average=type)

output = []
for class_index, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    wt_1 = wt[class_index,:]
    raw_preds = torch.matmul(loc_emb, wt_1)
    taxa = species_ids[class_index]
    N = counts[int(taxa)]

    species_locs = data['taxa_presence'].get(taxa)
    y_test = np.zeros(raw_preds.shape, int)
    y_test[species_locs] = 1

    preds = norm_cal_activation(raw_preds.numpy(), N, gamma=args.gamma)

    fscore50 = f1_at_thresh(y_test,preds,0.5)

    ##optimal scores
    _, optimal_thres = fscore_and_thres(y_test, preds)
    
    row = {
        "taxon_id": taxa,
        "optimal_thres": optimal_thres,
        "fscore50": fscore50
    }
    row_dict = dict(row)
    output.append(row_dict)

output_pd = pd.DataFrame(output)
output_pd.to_csv(RESULT_DIR+args.exp_name+"/"+str(args.gamma)+"/scores_v2.csv")
logging.info(f"Mean fscore50: {output_pd.fscore50.mean()}")
logging.info(f"Std optimal_thres: {output_pd.optimal_thres.std()}")

    


