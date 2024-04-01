# Global Gallery: The Fine Art of Painting Culture Portraits through Multilingual Instruction Tuning

This repository contains code for our paper accepted to NAACL 2024 (Main conference).

## Requirements - External libraries

Clone the repository and create a virtual environment with the following
libraries from pypi and a python version >= 3.8 to execute all the files with full functionality.

```bash
pip install "transformers==4.31.0" "datasets==2.13.0" "peft==0.4.0" "accelerate==0.21.0" "bitsandbytes==0.40.2" "trl==0.4.7" "safetensors>=0.3.1" "cappr==0.8.7" --upgrade
```

## Reproduction steps

The code is contained in the ```src``` directory.

- The ``plots`` folder contains files for plotting the figures in the paper.
- ``llama_patch.py`` is a helper file from huggingface for getting around
  different versions and still being able to use Flash Attention.
- ``ft.py`` contains code for instruction tuning using the translated Alpaca datasets in the paper.
- ``geomlama_configs.json`` contains all the configurations tested for the geomlama dataset.
- ``no_country_prompts.py`` contains the prompts for the no_country setting of GeoMLaMA.
- ``geomlama_cappr.py`` solves the MCQ setting of GeoMLaMA benchmark.
- ``geomlama_eval.py`` is used to evaluate the results of the GeoMLaMA benchmark.
- ``camel_prep.py`` is used to prepare the CAMeL dataset for instruction tuning for each of the 5 settings.
- ``camel_cappr.py`` solves the different MCQ settings of the CAMeL dataset.
- ``camel_eval.py`` is used to evaluate the results of the CAMeL dataset.
- ``process_text.py`` contains utility functions for determining whether a text is translatable (does not contain code and is not an empty string.)
- ``translations.py`` contains code for translating the Alpaca dataset to different languages using NLLB.
- ``quality_estimation.py`` contains code for reference-free estimation of the
  quality of translations using CometKiwi. First execute ``qe_data_prep.py`` to
  prepare the ``qe.json`` file needed for measuring the quality of translations.

## Results

Results for all experiments referred to in the paper are given in the
```outputs``` folder. It includes csv files organized into subfolders by
dataset name. Figures for these results are given in the ```figs``` folder.

The main structure of the repository is as follows :

```bash
.
.
├── README.md
├── data
│   ├── qe.json
│   ├── camel
│   │   ├── prompts.json
│   │   └── targets.json
│   └── geomlama
│       ├── el.csv
│       ├── en.csv
│       ├── fa.csv
│       ├── hi.csv
│       ├── sw.csv
│       └── zh.csv
├── figs
│   ├── rq3_1_70.pdf
│   └── rq3_2_70.pdf
├── outputs
│   ├── camel
│   │   ├── evals
│   │   │   ├── english_results_alpaca-2_type_1.csv
│   │   │   ├── ...
│   │   │   └── non_granular
│   │   │       └── english_results_alpaca-2_type_2_non_granular.csv
│   │   ├── type1
│   │   │   ├── english_results_alpaca-2_13b_type_1.csv
│   │   │   ├── ...
│   │   ├── type2
│   │   │   ├── english_results_alpaca-2_13b_type_2.csv
│   │   │   ├── ...
│   │   ├── type3
│   │   │   ├── english_results_alpaca-2_13b_type_3.csv
│   │   │   ├── ...
│   │   ├── type4
│   │   │   ├── english_results_alpaca-2_13b_type_4.csv
│   │   │   ├── ...
│   │   └── type5
│   │       ├── english_results_alpaca-2_13b_type_5.csv
│   │       ├── ...
│   └── geomlama
│       ├── country
│       │   ├── chinese_results_alpaca-2_13b.csv
│       │   ├── ...
│       ├── evals
│       │   ├── all_results.csv
│       │   ├── ...
│       └── no_country
│           ├── chinese_results_alpaca-2_13b.csv
│           ├── ...
└── src
    ├── camel_cappr.py
    ├── camel_eval.py
    ├── camel_prep.py
    ├── ft.py
    ├── geomlama_cappr.py
    ├── geomlama_configs.json
    ├── geomlama_eval.py
    ├── llama_patch.py
    ├── no_country_prompts.py
    ├── process_text.py
    ├── qe_data_prep.py
    ├── quality_estimation.py
    ├── translations.py
    └── plots
        ├── rq3_1.py
        └── rq3_2.py
```
