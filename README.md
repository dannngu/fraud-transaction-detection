# fraud-transaction-detection


A project fraturing ML fraud transaction detection.

```
fraud_detection_mlops_project/
├── .github/
│   └── workflows/
│       └── main.yaml # (For GitHub actions CI/CD)
├── config/
│   └── config.yaml   # General project configuration
├── data/
│   ├── external/     # Data from third party sources.
│   ├── interim/      # Intermediate data that has been transformed.
│   ├── processed/    # The final, canonical data sets for modeling.
│   └── raw/          # The original, immutable data dump.
├── notebooks/        # For EDA & inital experiments (Jupyter notebooks)
├── src/
│   └── fraud_detector/     # Our principal Python package ({{cookiecutter.project_slug}})
│       ├── __init__.py     # Initialization package file
│       ├── components/     # Clasess for each strep of the ML pipeline
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py # Preprocessing and Feature Engineering
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── config/         # Modules for config management 
│       │   └── configuration.py
│       ├── entity/         # Class definitions for configurations (POO)
│       │   └── config_entity.py
│       ├── pipeline/       # Classes to orchestrate components
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_transformation.py
│       │   ├── stage_03_model_training.py
│       │   └── stage_04_model_evaluation.py
│       └── utils/          # Reusable utility functions
│           └── common.py
├── app.py            # Streamlit app for user interface 
├── main.py           # Main entry point to execute the pipeline
├── requirements.txt  # Dependencies list of our project 
├── setup.py          # To install our package 'fraud_detector'
├── LICENSE           # Open-source license if one is chosen
├── dvc.yaml          # For Data Version Control (DVC) pipeline
├── params.yaml       # Parameters of the model and hyperparameters 
├── Dockerfile        # Instruction to build our para Docker image
├── README.md         # Project documentation

```
