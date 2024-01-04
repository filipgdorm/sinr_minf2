import torch
import numpy as np
import os
import sys
import eval
import models

# load model
train_params = torch.load('./pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', map_location='cpu')
train_params['experiment_name'] = "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt"
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to('cpu')
model.eval()

for eval_type in ['iucn']:
    eval_params = {}
    eval_params['exp_base'] = './pretrained_models'
    eval_params['experiment_name'] = train_params['experiment_name']
    eval_params['eval_type'] = eval_type
    if eval_type == 'iucn':
        eval_params['device'] = torch.device('cpu') # for memory reasons
    cur_results = eval.launch_eval_run(eval_params)
    np.save(os.path.join('./results', f'results_{eval_type}.npy'), cur_results)


