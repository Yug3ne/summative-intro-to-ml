# Who Gets Heard in Kenya's Counties?

Predicting which Kenyans contact their Member of County Assembly (MCA), using survey data joined to county budget records. Machine Learning summative: scikit-learn models compared with Keras Sequential and Functional networks.



## The question

Kenya's constitution says county assemblies have to involve the public when they pass budgets and laws. In practice a lot of that participation is box-ticking: hearings far from where people live, documents in legal English, and data bundles that cost money. The MCA is the elected representative closest to ordinary people, so I used contact with the MCA as a simple test of who is being heard.

The task is binary classification. Given a Kenyan's circumstances and their county's budget record, did they contact their MCA in the past year? About 1 in 4 Kenyans (25.5%) did.

## Data

Everything is real and public. Nothing is synthetic.

| Source | Level | What I took from it |
|---|---|---|
| [IBP Kenya, County Budget Transparency Survey 2023](https://internationalbudget.org/publications/kenyas-county-budget-transparency-survey-2023/) | County (47) | Transparency score, number of key budget documents published, public-participation reports, Citizens Budget |
| [Controller of Budget, County Budget Implementation Review Report FY 2020/21](https://cob.go.ke/reports/consolidated-county-budget-implementation-review-reports/) | County (47) | Development budget absorption, own-source revenue against target, MCA sitting allowances, pending bills |
| [Afrobarometer Round 9](https://www.afrobarometer.org/data/merged-data/) | Person | The target (Q11B, contact with a local councillor, i.e. the MCA) plus settlement, services nearby, phone and internet use, poverty, media, language, age, gender and education |
| [IBP Open Budget Survey 2023](https://internationalbudget.org/open-budget-survey) | Country | National transparency, participation and oversight scores |
| [World Bank WDI](https://data.worldbank.org) | Country | Internet users, mobile subscriptions, rural population, GDP per capita |

`get_data.py` downloads all five and merges them. Each Kenyan respondent is matched to their county by name, and all 47 counties match. The Controller of Budget year (July 2020 to June 2021) covers the 12 months before Afrobarometer interviewed Kenyans in November 2021.

**Why other countries are in a Kenya project.** Afrobarometer only interviewed 2,399 usable Kenyans. That's below the 20,000-row minimum and small for a neural network. The other 29 countries asked the same question, so their 40,131 respondents are used as extra training data. They are never used to pick or score a model. Every number reported here is on Kenyans. Experiment 1 checks whether this borrowing helps (it does, a little: Gradient Boosting improved in 11 of 15 Kenyan folds).

The raw data is not in this repo. Run `get_data.py`, or download the zip from the Drive link above.

## Repo layout

```
citizen_voice_classification.ipynb   main notebook (data → EDA → features → ML → DL → evaluation)
get_data.py                          downloads and merges the five sources
requirements.txt
figures/                             every figure in the report
results/                             experiment tables as CSV
models/model_card.json               features, thresholds and test scores (weights are on Drive)
```

## Running it

Python 3.10 to 3.13. TensorFlow doesn't have 3.14 wheels yet.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python get_data.py          # about 85 MB of downloads, then the merge
jupyter lab citizen_voice_classification.ipynb
```

On Colab:

```python
!git clone https://github.com/Yug3ne/summative-intro-to-ml && %cd summative-intro-to-ml
!pip install -q pyreadstat pdfplumber openpyxl
!python get_data.py
```

Then Run All. A full run takes 15 to 20 minutes on a laptop CPU.

## What's in the notebook

- **Cleaning.** Survey codes like 8, 9, 98 and 998 ("don't know", "refused") are set to missing column by column. "Not applicable" is recoded where it means something (no phone means no internet on a phone).
- **Features.** Indices for each barrier: distance to services, digital access, language of interview, Lived Poverty Index, assets and media. County indicators are standardised across the 47 counties.
- **Evaluation.** 20% of Kenyans for validation and 20% for test. Custom cross-validation folds train on Africa plus Kenya but score only held-out Kenyans. Fold-by-fold paired comparisons and bootstrap confidence intervals.
- **Experiments.**
  1. Training source: Kenya only vs Africa only vs both.
  2. Feature sets, added one group at a time.
  3. Four scikit-learn algorithms.
  4. Randomized and grid search, plus validation and learning curves.
  5. Seven Sequential networks and a three-input Functional network (person, county, country) fed by `tf.data`.
  6. Predicting counties the model never saw (`GroupKFold`).
- **After that.** ROC and PR curves, confusion matrices, permutation importance, partial dependence, and a fairness check by gender, age, rural/urban and language.

## Results on 480 held-out Kenyans

| Model | ROC-AUC (95% CI) | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Random guess | 0.500 | 0.256 | | |
| Logistic Regression | 0.652 (0.593–0.703) | 0.359 | 0.772 | 0.323 |
| SVM (RBF, Nyström) | 0.648 (0.591–0.699) | 0.375 | 0.602 | 0.381 |
| Random Forest | 0.665 (0.608–0.717) | 0.386 | 0.748 | 0.347 |
| **Gradient Boosting** | 0.654 (0.595–0.707) | **0.430** | 0.707 | 0.328 |
| Functional network (Keras) | 0.659 (0.600–0.712) | 0.394 | 0.642 | 0.371 |

The confidence intervals overlap, so no model clearly wins on ROC-AUC. I kept Gradient Boosting as the main model because it has the best PR-AUC.

**Main findings:**
- **Age and gender matter most.** 16% of 18–24 year olds contacted their MCA, against 33% of 35–49 year olds. For women the rate was 19%, against 32% for men.
- **Rural people contact their MCA more than people in towns.**
- **County budget transparency doesn't predict contact.** It is weakly negative across counties. Makueni scores 71/100 on transparency but only 16% of its respondents had contacted their MCA.
- **The model misses engaged women and young people.** Recall is 0.46 for women vs 0.87 for men, and 0.47 for youth vs 0.84 for over-35s.

## Drive folder contents

```bash
zip -r raw_dataset.zip data/raw     # Afrobarometer .sav, IBP county .xlsx, CoB PDF, OBS PDF, World Bank CSV
zip -r best_models.zip models       # .joblib pipeline, .keras and .h5 network, preprocessor, model card
```

Loading the saved models:

```python
import joblib, json, keras
card = json.load(open("models/model_card.json"))
gb = joblib.load("models/best_traditional_model.joblib")      # whole pipeline, takes the raw feature columns
p = gb.predict_proba(df[card["features"]])[:, 1]

net = keras.models.load_model("models/best_dl_model.keras")   # three named inputs: individual, county, country
prep = joblib.load("models/dl_preprocessor.joblib")
X = prep.transform(df[card["features"]]).astype("float32")
names = list(prep.get_feature_names_out())
p_dl = net.predict({b: X[:, [names.index(f) for f in fs]] for b, fs in card["dl_inputs"].items()}).ravel()
```

`df` needs the engineered columns from Sections 4 to 6 of the notebook. The decision thresholds are in `model_card.json`.

## Data citations

- International Budget Partnership Kenya (2023). *County Budget Transparency Survey 2023*. Nairobi.
- Office of the Controller of Budget (2021). *County Governments Budget Implementation Review Report for FY 2020/21*. Nairobi.
- Afrobarometer Data, Round 9 merged dataset (39 countries), 2021–2023. http://www.afrobarometer.org
- International Budget Partnership (2024). *Open Budget Survey 2023*. Washington, DC.
- World Bank. *World Development Indicators*. Washington, DC.
