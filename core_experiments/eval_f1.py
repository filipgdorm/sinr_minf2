
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

sys.path.append('../')
import datasets
import models
import utils

def check_containment(list1, list2):
    if set(list1).issubset(set(list2)):
        logging.info("All species in thresholds have expert range maps.")
    else:
        not_contained = set(list1) - set(list2)
        raise ValueError(f"Error: Taxa {not_contained} in thresholds have expert range maps.")
    
RESULT_DIR = './tgt_background_results/'

def main(args):

    threshs = pd.read_csv(RESULT_DIR+args.exp_name+f"/thresholds.csv")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = '../pretrained_models/' + args.model_path

    # load model
    train_params = torch.load(MODEL_PATH, map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(DEVICE)
    model.eval()

    #load reference from iucn
    with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
                data = json.load(f)
    species_ids = list(int(key) for key in (data['taxa_presence'].keys()))

    check_containment(threshs.taxon_id.values, species_ids)

    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env()
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

    obs_locs = np.array(data['locs'], dtype=np.float32)
    obs_locs = torch.from_numpy(obs_locs).to('cpu')
    loc_feat = enc.encode(obs_locs)

    classes_of_interest = torch.zeros(len(threshs.taxon_id), dtype=torch.int64)
    for tt_id, tt in enumerate(threshs.taxon_id):
        class_of_interest = np.array([train_params['params']['class_to_taxa'].index(tt)])
        classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

    with torch.no_grad():
        loc_emb = model(loc_feat, return_feats=True)
        wt = model.class_emb.weight[classes_of_interest, :]

    def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
        y_thresh = y_pred > thresh
        return f1_score(y_true, y_thresh, average=type)

    per_species_f1 = np.zeros((len(threshs.taxon_id)))
    for tt_id, taxa in tqdm(enumerate(threshs.taxon_id), total=len(threshs.taxon_id)):
        wt_1 = wt[tt_id,:]
        preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
        species_locs = data['taxa_presence'].get(str(taxa))

        y_test = np.zeros(preds.shape, int)
        y_test[species_locs] = 1

        thresh = threshs['thres'][tt_id]
        per_species_f1[tt_id] = f1_at_thresh(y_test, preds, thresh, type='binary')

    logging.info(f"Mean f1 score: {np.mean(per_species_f1)}")
    np.save(RESULT_DIR+args.exp_name+'/f1_scores.npy', per_species_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
    parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
    parser.add_argument("--exp_name", type=str, default='test', help="Experiment name")
    args = parser.parse_args()

    # Set up logging to file
    log_file_path = RESULT_DIR+args.exp_name+"/log.out"
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info(f"Model used for experiment: {args.model_path}")

    main(args)

