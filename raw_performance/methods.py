import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
import h3
import h3pandas
import datasets
import pandas as pd
import torch
import setup

def upper_b_threshold(preds, species_locs):
    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1 

    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thres = thresholds[index]
    max_fscore = fscore[index]

    return thres, max_fscore

def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
        y_thresh = y_pred > thresh
        return f1_score(y_true, y_thresh, average=type)

def generate_h3_cells_atRes(resolution=4):
    h3_cells = list(h3.get_res0_indexes())
    h3_atRes_cells = set()
    for cell in h3_cells:
        h3_atRes_cells = h3_atRes_cells.union(h3.h3_to_children(cell, resolution))
    return list(h3_atRes_cells)

def load_tgt_background_data(enc, h3_resolution=4, load=True):
    if load==False:
        train_params = {}
        train_params['species_set'] = 'all'
        train_params['hard_cap_num_per_class'] = 1000
        train_params['num_aux_species'] = 0
        train_params['input_enc'] = 'sin_cos'

        params = setup.get_default_params_train(train_params)
        train_dataset = datasets.get_train_data(params)
        train_df = pd.DataFrame(train_dataset.locs, columns=['lng','lat'])
        train_df['lng'] = train_df['lng']*180
        train_df['lat'] = train_df['lat']*90
        train_df['label'] = train_dataset.labels

        train_df_h3 = train_df.h3.geo_to_h3(h3_resolution)
        all_spatial_grid_counts = train_df_h3.index.value_counts()
        presence_absence = pd.DataFrame({
            "background": all_spatial_grid_counts,
        })
        presence_absence = presence_absence.fillna(0)        
    else:
         presence_absence = pd.read_csv('background_data_dir/presence_absence_4.csv')
         train_df_h3 = pd.read_csv('background_data_dir/train_df_h3_4.csv')
    
    h3_atRes_cells = generate_h3_cells_atRes(h3_resolution)
    gdfk = pd.DataFrame(index=h3_atRes_cells).h3.h3_to_geo()
    gdfk["lng"] = gdfk["geometry"].x
    gdfk["lat"] = gdfk["geometry"].y
    _ = gdfk.pop("geometry")
    gdfk = gdfk.rename_axis('h3index')

    obs_locs = np.array(gdfk[['lng', 'lat']].values, dtype=np.float32)
    obs_locs = torch.from_numpy(obs_locs).to('cpu')
    loc_feat = enc.encode(obs_locs)
    return presence_absence, loc_feat, train_df_h3

def target_sampling_threshold(loc_emb_thresh, wt_1, preds, species_locs, presence_absence, train_df_h3, class_id):
    #do all processing for generating thresh
    preds_thres = torch.sigmoid(torch.matmul(loc_emb_thresh, wt_1)).cpu().numpy()
    target_spatial_grid_counts = train_df_h3[train_df_h3.label==class_id.item()].index.value_counts()
    presence_absence["forground"] = target_spatial_grid_counts
    presence_absence["predictions"] = preds_thres
    presence_absence.forground = presence_absence.forground.fillna(0)
    yield_cutoff = np.percentile((presence_absence["background"]/presence_absence["forground"])[presence_absence["forground"]>0], 95)
    absences = presence_absence[(presence_absence["forground"]==0) & (presence_absence["background"] > yield_cutoff)]["predictions"]
    presences = presence_absence[(presence_absence["forground"]>0)]["predictions"]
    df_x = pd.DataFrame({'predictions': presences, 'test': 1})
    df_y = pd.DataFrame({'predictions': absences, 'test': 0})
    for_thres = pd.concat([df_x, df_y], ignore_index=False)
    precision, recall, thresholds = precision_recall_curve(for_thres.test, for_thres.predictions)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thres = thresholds[index]

    #check for species what f1 score is
    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1

    fscore = f1_at_thresh(y_test, preds, thres, type='binary')

    return thres, fscore