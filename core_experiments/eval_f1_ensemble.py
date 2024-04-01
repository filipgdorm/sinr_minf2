
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

#RESULT_DIR = './masking_ensemble_results/loss/'
RESULT_DIR = './tgt_background_ensemble_results/loss/'


def main():

    threshs = pd.read_csv(RESULT_DIR+"/thresholds.csv")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # MODEL_1_PATH = '../pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt'
    # MODEL_2_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
    # MODEL_3_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_-1/model.pt'
    MODEL_1_PATH = '../pretrained_models/1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
    MODEL_2_PATH = '../pretrained_models/1000_cap_models/final_loss_an_slds_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'
    MODEL_3_PATH = '../pretrained_models/1000_cap_models/final_loss_an_ssdl_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt'

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

    ### MODEL 3 uncapped version
    train_params = torch.load(MODEL_3_PATH, map_location='cpu')
    model3 = models.get_model(train_params['params'])
    model3.load_state_dict(train_params['state_dict'], strict=True)
    model3 = model3.to(DEVICE)
    model3.eval()

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

    ###Model 1
    with torch.no_grad():
        loc_emb_1 = model1(loc_feat, return_feats=True)
        wt_1 = model1.class_emb.weight[classes_of_interest, :]

    ###Model 2
    with torch.no_grad():
        loc_emb_2 = model2(loc_feat, return_feats=True)
        wt_2 = model2.class_emb.weight[classes_of_interest, :]

    ###Model 3
    with torch.no_grad():
        loc_emb_3 = model3(loc_feat, return_feats=True)
        wt_3 = model3.class_emb.weight[classes_of_interest, :]

    def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
        y_thresh = y_pred > thresh
        return f1_score(y_true, y_thresh, average=type)

    per_species_f1 = np.zeros((len(threshs.taxon_id)))
    for tt_id, taxa in tqdm(enumerate(threshs.taxon_id), total=len(threshs.taxon_id)):
        wt_column_1 = wt_1[tt_id,:]
        preds1 = torch.sigmoid(torch.matmul(loc_emb_1, wt_column_1)).cpu().numpy()

        wt_column_2 = wt_2[tt_id,:]
        preds2 = torch.sigmoid(torch.matmul(loc_emb_2, wt_column_2)).cpu().numpy()

        wt_column_3 = wt_3[tt_id,:]
        preds3 = torch.sigmoid(torch.matmul(loc_emb_3, wt_column_3)).cpu().numpy()
        
        # Stack the tensors along a new axis
        stacked_tensors = np.stack([preds1, preds2, preds3])

        # Take the mean across the new axis
        preds = np.mean(stacked_tensors, axis=0)
        species_locs = data['taxa_presence'].get(str(taxa))

        y_test = np.zeros(preds.shape, int)
        y_test[species_locs] = 1

        thresh = threshs['thres'][tt_id]
        per_species_f1[tt_id] = f1_at_thresh(y_test, preds, thresh, type='binary')

    logging.info(f"Mean f1 score: {np.mean(per_species_f1)}")
    np.save(RESULT_DIR+'/f1_scores.npy', per_species_f1)


if __name__ == "__main__":


    # Set up logging to file
    log_file_path = RESULT_DIR + "log.out"
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


    main()


