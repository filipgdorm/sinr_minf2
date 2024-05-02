
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
from sklearn.metrics import brier_score_loss
from tqdm import tqdm
import uncertainty_metrics.numpy as um
import argparse
import logging

sys.path.append('../')
import datasets
import models
import utils
import setup

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--exp_name", type=str, default='test', help="Experiment name")
parser.add_argument("--scaling_factor", type=float, default=1.0, help="Scaling factor")

args = parser.parse_args()

MODEL_PATH = '../pretrained_models/' + args.model_path
RESULT_DIR = './upscale_baseline_results/'

if not os.path.exists(RESULT_DIR+args.exp_name+"/"+str(args.scaling_factor)):
        os.mkdir(RESULT_DIR+args.exp_name+"/"+str(args.scaling_factor))

# Set up logging to file
log_file_path = RESULT_DIR+args.exp_name+"/"+str(args.scaling_factor)+"/log.out"
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


def scale_and_clip_sigmoid(raw_preds, s):
    scaled_values = s * torch.sigmoid(raw_preds)  # Scale the sigmoid values
    clipped_values = np.clip(scaled_values, 0, 1)  # Clip the values to [0, 1]
    return clipped_values

output = []
for class_index, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    wt_1 = wt[class_index,:]
    raw_preds = torch.matmul(loc_emb, wt_1)

    preds = scale_and_clip_sigmoid(raw_preds,args.scaling_factor)
    
    taxa = species_ids[class_index]
    species_locs = data['taxa_presence'].get(str(taxa))

    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1

    #generate calibration curve data for clustering
    y_true = y_test
    y_prob = preds
   
    ece = um.ece(y_true,y_prob,num_bins=20)

    tace = um.tace(y_true,y_prob,num_bins=20)

    fscore50 = f1_at_thresh(y_test,preds,0.5)

    ##optimal scores
    _, optimal_thres = fscore_and_thres(y_test, preds)

    row = {
        "taxon_id": taxa,
        "ece": ece,
        "tace": tace,
        "optimal_thres": optimal_thres,
        "fscore50": fscore50
    }
    row_dict = dict(row)
    output.append(row_dict)

output_pd = pd.DataFrame(output)
output_pd.to_csv(RESULT_DIR+args.exp_name+"/"+str(args.scaling_factor)+f"/scores.csv")
logging.info(f"Mean ECE: {output_pd.ece.mean()}")
logging.info(f"Mean TACE: {output_pd.tace.mean()}")
logging.info(f"Mean fscore50: {output_pd.fscore50.mean()}")
logging.info(f"Std optimal_thres: {output_pd.optimal_thres.std()}")