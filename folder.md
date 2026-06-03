pokemon-generative-models/
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   └── parameters.yml
│   └── local/
│       └── catalog.yml
│
├── data/
│   ├── 03_primary/
│   │   ├── cleaned_manifest.csv
│   │   └── cleaned_images/
│   ├── 05_model_input/
│   │   ├── train_manifest.csv
│   │   ├── val_manifest.csv
│   │   └── test_manifest.csv
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
│
├── src/
│   └── pokemon_generative/
│       ├── pipelines/
│       │   ├── prepare_data/
│       │   ├── ae/
│       │   ├── vae/
│       │   ├── cvae/
│       │   ├── ldm/
│       │   ├── conditional_ldm/
│       │   └── evaluation/
│       └── utils/
│           ├── image_utils.py
│           ├── dataset_utils.py
│           └── visualization.py
│
└── notebooks/
    ├── 00_check_data.ipynb
    ├── 01_debug_training.ipynb
    └── 02_compare_results.ipynb