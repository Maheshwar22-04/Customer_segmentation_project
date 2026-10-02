# Customer Segmentation & Purchase Prediction

Minor Project — E-commerce / Retail — End-to-End Machine Learning Project

## 1. Project Overview

This project analyzes online shopping session data to (1) segment sessions
into behaviorally distinct customer groups using clustering, and (2) predict
whether a given session will end in a purchase using classification. The
goal is to turn raw session logs into concrete marketing and conversion
recommendations for an e-commerce business.

## 2. Dataset Source

**Dataset:** Online Shoppers Purchasing Intention Dataset
**Source:** UCI Machine Learning Repository
**Link:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention
**Citation:** Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing
Intention Dataset [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5F88Q
**Target variable:** `Revenue` (True = session ended in a purchase)
**Size:** 12,330 sessions x 18 columns (10 numeric, 8 categorical, incl. target)

> **Important — data provenance in this delivery:** the notebook in this
> package was built and run inside a sandboxed environment with **no network
> access**, so it could not download the real UCI file directly. In its
> place, `data/online_shoppers_intention.csv` is a **synthetically generated
> dataset** (see `generate_dataset.py`) built to exactly match the real
> dataset's column names, dtypes, size (12,330 rows), class balance (~15.5%
> positive `Revenue`), and known feature relationships (PageValues as the
> strongest predictor of purchase, May/November traffic peaks, ~85% returning
> visitors, right-skewed durations, etc.), based on the dataset's published
> documentation and academic literature.
>
> **To reproduce these results on the genuine data:** download
> `online_shoppers_intention.csv` from the UCI link above and replace the
> file at `data/online_shoppers_intention.csv`. No other code changes are
> required — column names and dtypes match exactly — then simply re-run the
> notebook top to bottom.

## 3. Technologies Used

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- SciPy (statistical tests)
- Scikit-learn (preprocessing, clustering, classification, evaluation)
- Jupyter Notebook

## 4. Installation & Setup

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

## 5. How to Execute

1. Ensure `data/online_shoppers_intention.csv` is present (swap in the real
   UCI file for production-accurate results — see the note above).
2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Customer_Segmentation_Purchase_Prediction.ipynb
   ```
3. Run all cells top to bottom (`Kernel -> Restart & Run All`). The notebook
   is fully self-contained: it loads the data, computes statistics, produces
   all EDA plots, runs clustering, trains and evaluates all five
   classification models, and prints the business insights.

## 6. Project Structure

```
.
├── Customer_Segmentation_Purchase_Prediction.ipynb   # main deliverable notebook (pre-run, with outputs)
├── analysis.py                                        # same analysis as plain, cell-annotated .py (jupytext "percent" format)
├── generate_dataset.py                                 # synthetic dataset generator (see data note above)
├── data/
│   └── online_shoppers_intention.csv                  # dataset used by the notebook
├── outputs/                                             # exported PNG copies of every chart in the notebook
├── Project_Report.docx                                  # written project report
├── Presentation.pptx                                    # summary slide deck
└── README.md                                            # this file
```

## 7. Summary of Results

- **Data quality:** no missing values, no duplicate rows; target is
  imbalanced (~85% no purchase / ~15% purchase).
- **Strongest predictor:** `PageValues`, followed by `ExitRates`,
  `BounceRates`, and product-page engagement — confirmed by correlation
  analysis, a Mann-Whitney significance test, and Random Forest feature
  importance (three independent methods agree).
- **Customer segments (KMeans, k=4, cross-checked with Hierarchical
  clustering):**
  1. High-Value / High-Intent Shoppers — highest PageValues & purchase rate
  2. Engaged Browsers (Warming Up) — solid engagement, room to convert
  3. Casual Window Shoppers — moderate browsing, low urgency
  4. Low-Engagement / Bounce-Prone Visitors — high bounce/exit, rarely convert
- **Purchase prediction:** five models compared (Logistic Regression,
  Decision Tree, Random Forest, AdaBoost, KNN) on Accuracy, Precision,
  Recall, F1, and ROC-AUC, with the minority class balanced via training-set
  oversampling. Model selection is driven by **F1/Recall/AUC, not raw
  accuracy**, because of the class imbalance and the asymmetric business cost
  of missing a real buyer vs. over-targeting a non-buyer.
- **Validation:** 5-fold cross-validation, train/test gap analysis (overfitting
  check), feature importance, and misclassification analysis are all included
  and cross-confirm each other.

## 8. Limitations & Future Improvements

See Section 11 of the notebook and the "Limitations & Future Improvements"
section of the project report for the full discussion, including the
synthetic-data caveat above, single-year/single-site scope, and suggested
next steps (real-data validation, gradient boosting, threshold tuning,
SMOTE-based balancing).
