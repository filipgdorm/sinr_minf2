# IMPORTS
import torch
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score
import argparse
import logging
from tqdm import tqdm

import datasets
import models
import utils

from threshold_methods.methods import *

RESULT_DIR = './new_results/'
np.random.seed(42)
thresh_methods = [
     "upper_bound",
     "tgt_sampling",
     "rdm_sampling",
]

def data_split(species_ids, p=1):
    num_samples = len(species_ids)
    random_indices = np.random.choice(num_samples, size=int(num_samples * p), replace=False)

    # Creating a boolean mask
    mask = np.full(num_samples, False)
    mask[random_indices] = True
    return np.array(species_ids)[mask], np.array(species_ids)[~mask]


#PARSE ARGUMENTS
def main(args):
    # load model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_params = torch.load(f'./pretrained_models/{args.model_path}', map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(DEVICE)
    model.eval()

    #load reference from iucn
    with open(os.path.join('./data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
                data = json.load(f)
    species_ids = list((data['taxa_presence'].keys()))
    
    species_ids, calibration_species = data_split(species_ids)
    logging.info(f"Loaded {len(species_ids)} eval species and {len(calibration_species)} calibration species.")

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
    logging.info(f"Loaded embeddings and weights.")

    output = list()
    for tt_id, taxa in tqdm(enumerate(taxa_ids), total=len(taxa_ids)):
        taxa = str(taxa.item())
        wt_1 = wt[tt_id,:]
        preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
        species_locs = data['taxa_presence'].get(taxa)

        if args.method == "upper_bound": thres, fscore = upper_b_threshold(preds,species_locs)
        elif args.method == "tgt_sampling": thres, fscore = target_sampling(preds,species_locs)

        row = {
            "taxon_id": taxa,
            "thres": thres,
            "fscore": fscore
        }
        row_dict = dict(row)
        output.append(row_dict)

    output_pd = pd.DataFrame(output)
    output_pd.to_csv(f"./new_results/{args.exp_name}/scores.csv")
    logging.info(f"Mean F1 score {output_pd.fscore.mean()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
    parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
    parser.add_argument("--exp_name", type=str, default='test', help="Experiment name")
    parser.add_argument("--method", choices=thresh_methods, help="Method for thresholding")

    args = parser.parse_args()

    #create folder
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

    main(args)   