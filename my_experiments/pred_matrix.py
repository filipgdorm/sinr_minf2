# %% [markdown]
# attempt of making a calibration plot

# %%
import pandas as pd
import numpy as np
import json
import os
import datasets
import utils
import torch
import models
import ml_insights as mli
import matplotlib.pyplot as plt

# %%
HIGH_RES = True
THRESHOLD = 0.5
DISABLE_OCEAN_MASK = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#TAXA = 13270
SET_MAX_CMAP_TO_1 = False

# %%
# load model
train_params = torch.load('./pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', map_location='cpu')
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to(DEVICE)
model.eval()

# %%
#load reference from iucn
with open(os.path.join('./data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)

# %%
species_ids = (data['taxa_presence'].keys())

# %%
if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

# %%
obs_locs = np.array(data['locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

# %%
classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
taxa_ids = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)
    taxa_ids[tt_id] = int(tt)

# %%
taxa_ids

# %%
with torch.no_grad():
            # generate model predictions for classes of interest at eval locations
            loc_emb = model(loc_feat, return_feats=False)
            wt = model.class_emb.weight[classes_of_interest, :]
            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1)).cpu().numpy()

# %%
np.save('pred_mtx2.npy', pred_mtx)

# %%



