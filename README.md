# Student Score Predictor 🎓

An end-to-end machine learning project that predicts student **math scores** based on demographic and academic factors. Built with a modular, production-ready pipeline architecture.

## Overview

This project investigates how factors like gender, ethnicity, parental education level, lunch program type, test preparation, and reading/writing scores influence a student's math performance. It compares multiple regression algorithms and deploys the best one behind a Flask web interface.

## Project Structure

```
student-score-predictor/
├── data/                        # Raw dataset
│   └── student_scores.csv
├── notebooks/                   # EDA and experimentation notebooks
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_experiments.ipynb
├── score_predictor/             # Main Python package
│   ├── components/              # Pipeline stages
│   │   ├── data_ingestion.py    # Load & split data
│   │   ├── feature_engineering.py  # Preprocessing pipelines
│   │   └── model_selection.py   # Train, tune, & pick best model
│   ├── pipeline/
│   │   └── inference.py         # Prediction service
│   ├── exception.py             # Custom error handling
│   ├── logger.py                # Logging config
│   └── utils.py                 # Serialization & evaluation helpers
├── views/                       # Jinja2 HTML templates
├── output/                      # Generated artifacts (model, preprocessor, etc.)
├── main.py                      # Flask application entry point
├── train.py                     # One-command training script
├── setup.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This runs the full pipeline: data ingestion → feature engineering → model selection.  
The best model and preprocessor are saved under `output/`.

### 3. Launch the web app

```bash
python main.py
```

Visit `http://localhost:5000` and fill in the student details to get a predicted math score.

## Models Evaluated

| Algorithm | Tuned via GridSearchCV |
|---|---|
| Linear Regression | ✗ |
| Decision Tree | ✓ |
| Random Forest | ✓ |
| Gradient Boosting | ✓ |
| XGBoost | ✓ |
| CatBoost | ✓ |
| AdaBoost | ✓ |

The model with the highest R² on the test set is automatically selected as the champion.

## Dataset

The dataset contains 1000 student records with these features:

- **gender** — male / female  
- **race_ethnicity** — group A through E  
- **parental_level_of_education** — from some high school to master's degree  
- **lunch** — standard or free/reduced  
- **test_preparation_course** — completed or none  
- **reading_score** — score out of 100  
- **writing_score** — score out of 100  
- **math_score** — *(target)* score out of 100  

## Tech Stack

- Python 3.8+
- scikit-learn, XGBoost, CatBoost
- Flask
- pandas, numpy, seaborn, matplotlib

## License

This project is for educational purposes.
