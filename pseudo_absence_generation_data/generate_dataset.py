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
import argparse
from tqdm import tqdm

sys.path.append('../')
import datasets
import models
import utils
import setup

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

h3_resolution = 4
train_df_h3 = train_df.h3.geo_to_h3(h3_resolution)


def generate_h3_cells_atRes(resolution=4):
    h3_cells = list(h3.get_res0_indexes())
    h3_atRes_cells = set()
    for cell in h3_cells:
        h3_atRes_cells = h3_atRes_cells.union(h3.h3_to_children(cell, resolution))
    return list(h3_atRes_cells)

#generate gdfk table
h3_atRes_cells = generate_h3_cells_atRes(h3_resolution)
gdfk = pd.DataFrame(index=h3_atRes_cells).h3.h3_to_geo()
gdfk["lng"] = gdfk["geometry"].x
gdfk["lat"] = gdfk["geometry"].y
_ = gdfk.pop("geometry")
gdfk = gdfk.rename_axis('h3index')

#save two files
train_df_h3 = pd.to_csv("./pseudo_absence_generation_data/test_train_df_h3.csv", index_col=0)
gdfk = pd.to_csv("./pseudo_absence_generation_data/test_gdfk_res3.csv", index_col=0)