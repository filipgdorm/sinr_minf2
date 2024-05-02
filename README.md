# Generating Species Range Maps Using Neural Networks 2024

Code for replicating results of my dissertation. The setup is based on code developed by Cole et al. https://github.com/elijahcole/sinr

Please refer to their documentation to download data, code and pretrained models in order to replicate my experiments. An overview of the submission is here:

```bash
.
├── calibration_experiments
├── core_experiments
├── data
├── geo_prior_experiments
├── pretrained_models
└── raw_performance
```


## 🔍 Getting Started 

#### Installing Required Packages


Create a new environment and activate it:
```bash
conda env create -f environment.yml
```



#### Data Download and Preparation
Instructions for downloading the data in `data/README.md`, alternatively follow instructions at https://github.com/elijahcole/sinr


## 🚅 Evaluating Models
My experiments are mainly collected in the following directories:
1. ```core_experiments```

The different files in this directory correspond to the different binarisation experiments from Section 5.1 of the report. The files are named after what experiments they represent.
Refer to ```core_experiments/run_experiments.sh``` for an example of how these experiments can be run.

2. ```calibration_experiments```

The files in this folder correspond to the calibration experiments in Section 5.3 of the report. The name reflects what experiments they correspond to. An example of how these experiments can be run is in ```calibration_experiments/run_experiments.sh```.

3. ```geo_prior_experiments```

The files in the folder contain the experiments regarding the Geo Prior task for calibration in Section 5.3 of the report. An experiment of how this is run is in ```geo_prior_experiments/upscale_bash.sh```

##  🙏 Acknowledgements
The code that constitutes the backbone of my experiments is based on the following project found through this link:
https://github.com/elijahcole/sinr

