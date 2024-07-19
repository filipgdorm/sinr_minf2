
import torch
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import argparse
import logging
from tqdm import tqdm

sys.path.append('../')
import datasets
import models
import utils

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--prop_absence", type=float, default='test', help="Experiment name")

args = parser.parse_args()

MODEL_PATH = '../pretrained_models/' + 'model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt'
RESULT_DIR = f'./result_prop_{args.prop_absence}/'

# Set up logging to file
log_file_path = RESULT_DIR+"/log.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


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
taxa_ids = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)
    taxa_ids[tt_id] = int(tt)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
    y_thresh = y_pred > thresh
    return f1_score(y_true, y_thresh, average=type)

output = list()
for tt_id, taxa in tqdm(enumerate(taxa_ids), total=len(taxa_ids)):
    wt_1 = wt[tt_id,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
    taxa = taxa.item()
    species_locs = data['taxa_presence'].get(str(taxa))
    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1

    #split into 10% used for setting thresholds 90% for final eval
    eval_preds, thresh_preds, eval_y_test, thresh_y_test = train_test_split(
        preds, y_test, test_size=args.prop_absence, random_state=42
    )

    #calculate thresholds 
    precision, recall, thresholds = precision_recall_curve(thresh_y_test, thresh_preds)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thres = thresholds[index]

    #evaluate performance
    f1 = f1_at_thresh(eval_y_test, eval_preds, thres)
    
    row = {
        "taxon_id": taxa,
        "thres": thres,
        "fscore": f1
    }
    row_dict = dict(row)
    output.append(row_dict)
output_pd = pd.DataFrame(output)
logging.info(f"Mean F1 score {output_pd.fscore.mean()}")
output_pd.to_csv(RESULT_DIR+"/thresholds_w_f1scores.csv")


