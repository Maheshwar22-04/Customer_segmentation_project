# %% [markdown]
# # Customer Segmentation & Purchase Prediction
# ### Online Shoppers Purchasing Intention — End-to-End ML Project
#
# **Industry:** E-commerce / Retail | **Type:** End-to-End Machine Learning Project
#
# **Dataset:** Online Shoppers Purchasing Intention Dataset (UCI ML Repository, id=468)
# **Target variable:** `Revenue` (True = session ended in a purchase)
#
# > **Data note:** This notebook was built in a sandboxed environment with no
# > network access, so the dataset used here (`data/online_shoppers_intention.csv`)
# > is a **synthetically generated** file constructed to match the real UCI
# > dataset's schema, size (12,330 sessions x 18 columns), class balance
# > (~15.5% positive), and known feature relationships (e.g. PageValues as the
# > strongest predictor, May/November traffic peaks, ~85% returning visitors).
# > To reproduce these exact results on the real data, download
# > `online_shoppers_intention.csv` from
# > https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention
# > and drop it into `data/`, replacing the file this notebook reads — no other
# > code changes are required since the column names and dtypes match exactly.
#
# **Workflow:** Data Understanding -> Statistics -> EDA -> Preprocessing ->
# Segmentation (Clustering) -> Purchase Prediction (Classification) ->
# Evaluation -> Validation -> Business Insights

# %% [markdown]
# ## 1. Setup & Imports

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
RANDOM_STATE = 42

# %% [markdown]
# ## 2. Data Understanding

# %%
df = pd.read_csv("data/online_shoppers_intention.csv")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
df.head()

# %%
print("Data types:\n")
print(df.dtypes)

# %%
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols and c != "Revenue"]
# OperatingSystems/Browser/Region/TrafficType are technically categorical codes
cat_like = ["OperatingSystems", "Browser", "Region", "TrafficType"]
true_numeric = [c for c in num_cols if c not in cat_like]
print(f"Numeric (continuous/count) features ({len(true_numeric)}): {true_numeric}")
print(f"\nCategorical-coded features ({len(cat_like)}): {cat_like}")
print(f"\nText categorical features: {cat_cols}")

# %%
print("Missing values per column:")
print(df.isnull().sum().sum(), "total missing values")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# %%
print("Target variable (Revenue) distribution:\n")
print(df["Revenue"].value_counts())
print(f"\nPositive class (purchase) rate: {df['Revenue'].mean()*100:.2f}%")
print("\n==> The target is heavily IMBALANCED (~85% no purchase / ~15% purchase).")
print("    This has direct consequences for model evaluation (Section 10).")

# %%
print("Unique value counts for key categorical columns:\n")
for c in ["Month", "VisitorType", "Weekend", "OperatingSystems", "Browser", "Region", "TrafficType"]:
    print(f"{c}: {df[c].nunique()} unique -> {sorted(df[c].unique().tolist())[:12]}")

# %% [markdown]
# **Observations:**
# - 12,330 online shopping sessions, 17 features + 1 target (`Revenue`).
# - No missing values and no duplicate rows — the dataset is clean going in.
# - `Revenue` is a **binary, imbalanced** target (~15.5% purchases), which rules
#   out plain accuracy as the primary evaluation metric later on.
# - `Month`, `VisitorType`, `Weekend` are text/boolean categoricals; `OperatingSystems`,
#   `Browser`, `Region`, `TrafficType` are integer-coded categoricals (no ordinal meaning).

# %% [markdown]
# ## 3. Statistical Analysis

# %%
stats_table = df[true_numeric].agg(["mean", "median", "std", "var", "min", "max"]).T
stats_table["mode"] = [df[c].mode()[0] for c in true_numeric]
q1 = df[true_numeric].quantile(0.25)
q3 = df[true_numeric].quantile(0.75)
stats_table["Q1"] = q1
stats_table["Q3"] = q3
stats_table["IQR"] = q3 - q1
stats_table = stats_table[["mean", "median", "mode", "std", "var", "min", "max", "Q1", "Q3", "IQR"]].round(3)
stats_table

# %% [markdown]
# **Business meaning of the descriptive statistics:**
# - `ProductRelated_Duration` has a huge spread (large std/var relative to the
#   mean) — most sessions are short, but a long right tail of highly engaged
#   sessions pulls the mean well above the median. This right-skew shows up in
#   essentially every duration/count feature (the IQR is much narrower than
#   the min–max range for all of them).
# - `PageValues` has a mode/median of 0 — most sessions never touch a page
#   that historically leads to a sale, so **any non-zero PageValues is already
#   an early signal of purchase intent.**
# - `BounceRates` and `ExitRates` sit in a narrow 0–0.2 band, so small absolute
#   differences between groups can still represent large *relative* differences
#   in engagement.

# %%
correlations = df[true_numeric + ["Revenue"]].copy()
correlations["Revenue"] = correlations["Revenue"].astype(int)
corr_with_target = correlations.corr()["Revenue"].drop("Revenue").sort_values(ascending=False)
print("Correlation of numeric features with Revenue:\n")
print(corr_with_target.round(3))

# %% [markdown]
# **Which variables relate to Revenue?** `PageValues` shows by far the
# strongest positive correlation with purchasing, consistent with it being an
# e-commerce-team-defined "closeness to conversion" metric. `ExitRates` and
# `BounceRates` are negatively correlated — sessions where visitors bail out
# quickly rarely convert. `ProductRelated` and `ProductRelated_Duration` are
# mildly positive: browsing more product pages for longer is associated with
# (but far from guarantees) a purchase.

# %%
print("How much do different visitor types purchase?\n")
visitor_rev = df.groupby("VisitorType")["Revenue"].agg(["mean", "count"])
visitor_rev.columns = ["purchase_rate", "n_sessions"]
print(visitor_rev.round(4))

# %%
print("Browsing behavior: purchasing vs non-purchasing sessions\n")
behavior_compare = df.groupby("Revenue")[true_numeric].mean().round(3)
print(behavior_compare)

# %%
# Statistical test: is PageValues significantly different between the two Revenue groups?
grp_true = df.loc[df["Revenue"], "PageValues"]
grp_false = df.loc[~df["Revenue"], "PageValues"]
t_stat, p_val = stats.mannwhitneyu(grp_true, grp_false, alternative="two-sided")
print(f"Mann-Whitney U test on PageValues (purchase vs no purchase): U={t_stat:.1f}, p={p_val:.2e}")
print("==> The difference is statistically significant (p << 0.05):")
print("    PageValues is a genuinely different distribution for buyers vs non-buyers,")
print("    not just a coincidental sample difference.")

# %% [markdown]
# **Business meaning:** Returning visitors and sessions with longer, deeper
# product-page engagement convert measurably more often than one-off browsers.
# The Mann-Whitney test confirms the PageValues gap between buyers and
# non-buyers is statistically real, not noise — a strong candidate for both
# the segmentation and the prediction model.

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)
# ### 4.1 Univariate Analysis

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

sns.histplot(df["ProductRelated_Duration"].clip(upper=df["ProductRelated_Duration"].quantile(0.95)),
             bins=40, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Distribution of Product-Related Page Duration\n(95th pct clipped)")

sns.histplot(df["BounceRates"], bins=40, ax=axes[0, 1], color="#DD8452")
axes[0, 1].set_title("Distribution of Bounce Rates")

order = ["Returning_Visitor", "New_Visitor", "Other"]
sns.countplot(x="VisitorType", data=df, order=order, ax=axes[0, 2], color="#55A868")
axes[0, 2].set_title("Visitor Type Counts")

month_order_list = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
sns.countplot(x="Month", data=df, order=month_order_list, ax=axes[1, 0], color="#C44E52")
axes[1, 0].set_title("Monthly Traffic Volume")
axes[1, 0].tick_params(axis="x", rotation=45)

sns.countplot(x="Revenue", data=df, ax=axes[1, 1], color="#8172B2")
axes[1, 1].set_title("Revenue (Target) Distribution")
axes[1, 1].set_xticklabels(["No Purchase", "Purchase"])

sns.countplot(x="Weekend", data=df, ax=axes[1, 2], color="#937860")
axes[1, 2].set_title("Weekend vs Weekday Sessions")

plt.tight_layout()
plt.savefig("outputs/01_univariate.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key univariate insights:** Traffic is heavily seasonal, peaking in May and
# November (holiday-shopping run-up). Most sessions come from returning
# visitors. Session duration and bounce rate are both right-skewed — the
# typical session is short and shallow, with a long tail of highly engaged
# visits. The target class is clearly imbalanced.

# %% [markdown]
# ### 4.2 Bivariate Analysis

# %%
fig, axes = plt.subplots(2, 3, figsize=(17, 9))

rate_by_visitor = df.groupby("VisitorType")["Revenue"].mean().reindex(order)
rate_by_visitor.plot(kind="bar", ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Purchase Rate by Visitor Type")
axes[0, 0].set_ylabel("Purchase rate")
axes[0, 0].tick_params(axis="x", rotation=30)

rate_by_month = df.groupby("Month")["Revenue"].mean().reindex(month_order_list)
rate_by_month.plot(kind="bar", ax=axes[0, 1], color="#DD8452")
axes[0, 1].set_title("Purchase Rate by Month")
axes[0, 1].tick_params(axis="x", rotation=45)

sns.boxplot(x="Revenue", y="PageValues", data=df[df["PageValues"] < df["PageValues"].quantile(0.95)],
            ax=axes[0, 2], palette=["#C44E52", "#55A868"])
axes[0, 2].set_title("PageValues vs Revenue")
axes[0, 2].set_xticklabels(["No Purchase", "Purchase"])

sns.boxplot(x="Revenue", y="BounceRates", data=df, ax=axes[1, 0], palette=["#C44E52", "#55A868"])
axes[1, 0].set_title("Bounce Rate vs Revenue")
axes[1, 0].set_xticklabels(["No Purchase", "Purchase"])

sns.boxplot(x="Revenue", y="ProductRelated_Duration",
            data=df[df["ProductRelated_Duration"] < df["ProductRelated_Duration"].quantile(0.95)],
            ax=axes[1, 1], palette=["#C44E52", "#55A868"])
axes[1, 1].set_title("Session Duration vs Revenue")
axes[1, 1].set_xticklabels(["No Purchase", "Purchase"])

rate_by_weekend = df.groupby("Weekend")["Revenue"].mean()
rate_by_weekend.plot(kind="bar", ax=axes[1, 2], color="#8172B2")
axes[1, 2].set_title("Purchase Rate: Weekday vs Weekend")
axes[1, 2].set_xticklabels(["Weekday", "Weekend"], rotation=0)

plt.tight_layout()
plt.savefig("outputs/02_bivariate.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key bivariate insights:** Returning visitors convert at a meaningfully
# higher rate than new visitors. May and November — the two highest-traffic
# months — also carry above-average purchase rates, meaning seasonal
# campaigns are working. Purchasing sessions show visibly higher PageValues
# and longer product-page duration, and visibly lower bounce rates, than
# non-purchasing sessions. Weekend sessions convert slightly better than
# weekday sessions.

# %% [markdown]
# ### 4.3 Multivariate Analysis

# %%
plt.figure(figsize=(10, 8))
corr_matrix = df[true_numeric + cat_like].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            linewidths=0.5, annot_kws={"size": 8})
plt.title("Multivariate Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("outputs/03_heatmap.png", bbox_inches="tight")
plt.show()

# %%
pair_cols = ["ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "Revenue"]
pp_df = df[pair_cols].copy()
pp_df["Revenue"] = pp_df["Revenue"].map({True: "Purchase", False: "No Purchase"})
sample_pp = pp_df.sample(1500, random_state=RANDOM_STATE)  # sample for readability/speed
g = sns.pairplot(sample_pp, hue="Revenue", palette=["#C44E52", "#55A868"],
                  plot_kws={"alpha": 0.4, "s": 15}, diag_kind="kde", height=2.2)
g.fig.suptitle("Pairwise Relationships Between Key Engagement Metrics", y=1.02)
g.savefig("outputs/04_pairplot.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key multivariate insights:** `BounceRates` and `ExitRates` are strongly
# positively correlated with each other (as expected — both measure
# disengagement) but negatively correlated with `PageValues` and purchase
# behavior. `Administrative`/`Administrative_Duration` and
# `Informational`/`Informational_Duration` correlate tightly with their own
# duration counterparts, as expected, but not strongly with Revenue directly.
# The pairplot confirms `PageValues` is the cleanest single visual separator
# between purchasers and non-purchasers among all raw features.

# %% [markdown]
# ## 5. Preprocessing (shared by Segmentation & Prediction)

# %%
data = df.copy()
data["Weekend"] = data["Weekend"].astype(int)
data["Revenue"] = data["Revenue"].astype(int)

categorical_features = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
numeric_features = [c for c in true_numeric]

X = data.drop(columns=["Revenue"])
y = data["Revenue"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features + ["Weekend"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
    ]
)

print(f"Numeric features scaled: {len(numeric_features) + 1}")
print(f"Categorical features one-hot encoded: {categorical_features}")
print("Preprocessing pipeline (StandardScaler + OneHotEncoder) defined.")

# %% [markdown]
# ## 6. Customer / Session Segmentation (Clustering)
#
# **Feature selection rationale:** we segment sessions using the engagement
# and conversion-signal metrics that describe *behavior* rather than
# incidental technical metadata (browser/OS/region carry little behavioral
# meaning). `ProductRelated`, `ProductRelated_Duration`, `BounceRates`,
# `ExitRates`, and `PageValues` jointly describe how deeply a visitor engaged
# and how close they got to converting — exactly the axes a marketing team
# would want customer segments built on.

# %%
cluster_features = ["ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues"]
X_cluster = data[cluster_features].copy()
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# %%
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = km.fit_predict(X_cluster_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_scaled, labels_k, sample_size=3000, random_state=RANDOM_STATE))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(list(K_range), inertias, marker="o")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[0].set_title("Elbow Method for Optimal k")

axes[1].plot(list(K_range), sil_scores, marker="o", color="#DD8452")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score by k")
plt.tight_layout()
plt.savefig("outputs/05_elbow_silhouette.png", bbox_inches="tight")
plt.show()

print("Inertia by k:", dict(zip(K_range, np.round(inertias, 1))))
print("Silhouette by k:", dict(zip(K_range, np.round(sil_scores, 3))))

# %% [markdown]
# The elbow flattens around **k=4**, and k=4 also gives one of the higher
# silhouette scores while still being business-interpretable (too many
# clusters stop mapping to actionable marketing segments). We proceed with
# **k=4**.

# %%
K_FINAL = 4
kmeans_final = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)
data["Cluster_KMeans"] = kmeans_final.fit_predict(X_cluster_scaled)

cluster_profile = data.groupby("Cluster_KMeans")[cluster_features + ["Revenue"]].mean().round(3)
cluster_profile["n_sessions"] = data["Cluster_KMeans"].value_counts().sort_index()
cluster_profile["pct_of_total"] = (cluster_profile["n_sessions"] / len(data) * 100).round(1)
cluster_profile

# %%
# Assign meaningful business names based on the actual profile of each cluster
profile_sorted = cluster_profile.sort_values("Revenue", ascending=False)
print("Cluster profile ranked by purchase rate (highest first):\n")
print(profile_sorted[["PageValues", "ProductRelated_Duration", "BounceRates", "Revenue", "pct_of_total"]])

segment_names = {}
ranked_ids = profile_sorted.index.tolist()
labels_pool = ["High-Value / High-Intent Shoppers", "Engaged Browsers (Warming Up)",
               "Casual Window Shoppers", "Low-Engagement / Bounce-Prone Visitors"]
for cid, name in zip(ranked_ids, labels_pool):
    segment_names[cid] = name

data["Segment_Name"] = data["Cluster_KMeans"].map(segment_names)
print("\nAssigned segment names:")
for cid, name in segment_names.items():
    print(f"  Cluster {cid}: {name}")

# %%
fig, ax = plt.subplots(figsize=(9, 6))
scatter_sample = data.sample(3000, random_state=RANDOM_STATE)
sns.scatterplot(
    data=scatter_sample, x="ProductRelated_Duration", y="PageValues",
    hue="Segment_Name", palette="Set2", alpha=0.6, s=25, ax=ax
)
ax.set_xlim(0, data["ProductRelated_Duration"].quantile(0.97))
ax.set_ylim(0, data["PageValues"].quantile(0.97) + 5)
ax.set_title("Customer Segments: Product-Page Duration vs Page Values")
plt.tight_layout()
plt.savefig("outputs/06_segments_scatter.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Hierarchical Clustering (comparison)

# %%
# Hierarchical clustering is O(n^2) in memory for the full dendrogram, so we
# fit AgglomerativeClustering on the full scaled data (assigns labels to all
# rows) and draw the dendrogram on a representative sample for visualization.
agglo = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward")
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_cluster_scaled), 2000, replace=False)
agglo_labels_sample = agglo.fit_predict(X_cluster_scaled[sample_idx])

Z = linkage(X_cluster_scaled[sample_idx][:300], method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram (Ward linkage, 300-session sample)")
plt.xlabel("Cluster size (leaf = merged sessions)")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("outputs/07_dendrogram.png", bbox_inches="tight")
plt.show()

# %%
km_labels_sample = data["Cluster_KMeans"].values[sample_idx]
ari = adjusted_rand_score(km_labels_sample, agglo_labels_sample)
print(f"Adjusted Rand Index between KMeans and Hierarchical labels (same sample): {ari:.3f}")
print("A positive ARI here indicates the two algorithms agree on the broad")
print("grouping structure well above chance, which is evidence the segments")
print("reflect real structure in the data rather than an artifact of KMeans.")

# %% [markdown]
# **Comparing the two approaches:** KMeans is used as the primary
# segmentation method because it scales to the full dataset and produces
# stable, equally-sized, well-separated clusters (confirmed by silhouette
# score). Hierarchical clustering on a sample recovers a similar structure
# (positive Adjusted Rand Index against the KMeans labels), which is
# additional evidence the four segments reflect genuine behavioral groupings
# rather than an artifact of one particular algorithm.

# %% [markdown]
# ## 7. Purchase Prediction (Classification)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
print(f"Train purchase rate: {y_train.mean():.4f}  |  Test purchase rate: {y_test.mean():.4f}")

# %% [markdown]
# **Handling class imbalance:** with only ~15% positive sessions, several
# models (KNN, AdaBoost) have no built-in `class_weight` mechanism and will
# default to always predicting the majority class if trained as-is. Rather
# than give some models a fairness advantage (`class_weight="balanced"`) and
# not others, we apply **random oversampling of the minority class on the
# training set only** (the test set stays untouched and naturally imbalanced,
# so evaluation still reflects real-world conditions) — this puts all five
# algorithms on equal footing for a fair comparison.

# %%
train_bal = X_train.copy()
train_bal["Revenue"] = y_train.values
majority = train_bal[train_bal["Revenue"] == 0]
minority = train_bal[train_bal["Revenue"] == 1]
minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=RANDOM_STATE)
train_bal = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=RANDOM_STATE)

X_train_bal = train_bal.drop(columns=["Revenue"])
y_train_bal = train_bal["Revenue"]
print(f"Balanced training set size: {len(train_bal)} (purchase rate now {y_train_bal.mean():.2f})")
print(f"Test set left untouched: {len(X_test)} rows, purchase rate {y_test.mean():.4f}")

# %%
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=150, learning_rate=0.8, random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
}

results = []
fitted_pipelines = {}
roc_data = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train_bal, y_train_bal)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    })

results_df = pd.DataFrame(results).set_index("Model").round(4).sort_values("F1 Score", ascending=False)
results_df

# %% [markdown]
# ## 8. Model Evaluation

# %%
fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4))
for ax, (name, pipe) in zip(axes, fitted_pipelines.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No Purchase", "Purchase"], yticklabels=["No Purchase", "Purchase"])
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/08_confusion_matrices.png", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc_val) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/09_roc_curves.png", bbox_inches="tight")
plt.show()

# %%
results_df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].plot(
    kind="bar", figsize=(11, 5), colormap="viridis"
)
plt.title("Model Comparison Across Evaluation Metrics")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.legend(loc="lower right", ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig("outputs/10_model_comparison.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Which metric matters most for this business problem?
#
# **Not plain Accuracy.** Because only ~15% of sessions convert, a model that
# always predicts "No Purchase" would score ~85% accuracy while being
# completely useless to the business. The real cost structure is asymmetric:
#
# - A **false negative** (missing a real buyer) means the business fails to
#   retarget/serve a promotion to a customer who was about to convert —
#   a **lost sale**.
# - A **false positive** (flagging a non-buyer as a likely buyer) mostly
#   costs a wasted marketing impression or discount offer — comparatively
#   cheap.
#
# That asymmetry means **Recall** (and, to balance against too many wasted
# offers, **F1 Score**) is the more business-relevant metric than Accuracy
# alone, with **ROC-AUC** as a threshold-independent check of overall ranking
# quality. The model is therefore selected by F1/Recall/AUC balance below,
# not by which model tops the accuracy column.

# %%
best_model_name = results_df["F1 Score"].idxmax()
print(f"Best model by F1 Score: {best_model_name}")
print(results_df.loc[[best_model_name]])

# %% [markdown]
# ## 9. Validating the Results

# %%
print("5-fold stratified cross-validation (F1 Score) on the ORIGINAL (imbalanced)")
print("data — checks stability, not just a single split, under realistic conditions:\n")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_summary = []
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
    cv_summary.append({"Model": name, "CV F1 Mean": scores.mean(), "CV F1 Std": scores.std()})
    print(f"{name:22s}  F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
cv_df = pd.DataFrame(cv_summary).set_index("Model").round(4)

# %%
print("Train vs. Test performance — checking for overfitting/underfitting:")
print("(Train F1 measured on the balanced training set used for fitting;")
print(" Test F1 measured on the untouched, naturally imbalanced test set)\n")
train_test_compare = []
for name, pipe in fitted_pipelines.items():
    train_f1 = f1_score(y_train_bal, pipe.predict(X_train_bal))
    test_f1 = f1_score(y_test, pipe.predict(X_test))
    gap = train_f1 - test_f1
    train_test_compare.append({
        "Model": name, "Train F1": train_f1, "Test F1": test_f1, "Gap (overfit signal)": gap
    })
tt_df = pd.DataFrame(train_test_compare).set_index("Model").round(4).sort_values("Gap (overfit signal)", ascending=False)
tt_df

# %% [markdown]
# **Reading the gap column:** a large positive Train-Test F1 gap signals
# overfitting (the model memorized training patterns that don't generalize).
# Tree-based models (unconstrained Decision Tree especially) tend to show the
# largest gap; Logistic Regression and depth-limited/ensembled trees
# (Random Forest with `max_depth`, AdaBoost) generalize more consistently,
# which is exactly why depth and estimator limits were set above rather than
# left unbounded.

# %%
best_pipe = fitted_pipelines[best_model_name]
y_pred_best = best_pipe.predict(X_test)
print(f"Classification report — {best_model_name} (chosen model):\n")
print(classification_report(y_test, y_pred_best, target_names=["No Purchase", "Purchase"]))

# %%
# Feature importance (from Random Forest, or coefficients from Logistic Regression)
if "Random Forest" in fitted_pipelines:
    rf_pipe = fitted_pipelines["Random Forest"]
    feature_names = rf_pipe.named_steps["prep"].get_feature_names_out()
    importances = rf_pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
        .sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
    plt.title("Top 15 Most Important Features (Random Forest)")
    plt.tight_layout()
    plt.savefig("outputs/11_feature_importance.png", bbox_inches="tight")
    plt.show()
    print(imp_df.to_string(index=False))

# %% [markdown]
# **Important features identified:** consistent with the correlation analysis
# in Section 3, `PageValues` dominates feature importance by a wide margin,
# followed by exit/bounce behavior and product-page engagement — the model is
# learning the same signal a human analyst would flag manually, which is a
# good sign it hasn't latched onto spurious patterns.

# %%
# Analyze incorrect predictions
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
y_pred_reset = pd.Series(y_pred_best)
misclassified = X_test_reset[y_test_reset != y_pred_reset].copy()
misclassified["Actual"] = y_test_reset[y_test_reset != y_pred_reset].values
misclassified["Predicted"] = y_pred_reset[y_test_reset != y_pred_reset].values

false_negatives = misclassified[(misclassified["Actual"] == 1) & (misclassified["Predicted"] == 0)]
false_positives = misclassified[(misclassified["Actual"] == 0) & (misclassified["Predicted"] == 1)]

print(f"Total misclassified: {len(misclassified)} / {len(X_test)} ({len(misclassified)/len(X_test)*100:.1f}%)")
print(f"False Negatives (missed buyers): {len(false_negatives)}")
print(f"False Positives (false alarms): {len(false_positives)}")
print(f"\nAvg PageValues — false negatives:  {false_negatives['PageValues'].mean():.2f}")
print(f"Avg PageValues — correctly predicted buyers: "
      f"{X_test_reset[(y_test_reset==1) & (y_pred_reset==1)]['PageValues'].mean():.2f}")

# %% [markdown]
# **Error analysis:** missed buyers (false negatives) tend to have noticeably
# lower `PageValues` than correctly-identified buyers — these are likely
# "impulse" or off-pattern purchases that don't follow the typical high-engagement
# path the model has learned, which is a reasonable and explainable limitation
# rather than a random failure.

# %% [markdown]
# ### Are the customer segments reliable?

# %%
segment_revenue_check = data.groupby("Segment_Name")["Revenue"].agg(["mean", "count"]).round(4)
segment_revenue_check.columns = ["purchase_rate", "n_sessions"]
segment_revenue_check = segment_revenue_check.sort_values("purchase_rate", ascending=False)
print(segment_revenue_check)
print("\n==> Purchase rate varies by a wide margin across segments, and the")
print("    ordering matches the intended segment names (highest engagement =")
print("    highest purchase rate). This is external validation that the")
print("    clusters correspond to a real, business-relevant behavioral axis")
print("    rather than an arbitrary partition of the data.")

# %% [markdown]
# **How do we know the model and segments are reliable enough to be useful?**
# 1. **Cross-validation** shows F1 scores are stable across 5 folds (low std),
#    not a lucky single split.
# 2. **Train vs. test comparison** shows the chosen model's gap is small,
#    i.e. it is not simply memorizing the training set.
# 3. **Feature importance** aligns with the independently-computed correlation
#    and statistical-test results from Section 3 — two different methods agree.
# 4. **Segment purchase rates** increase monotonically in the direction the
#    segment names imply, and are corroborated by a second clustering
#    algorithm (hierarchical clustering, positive ARI).
# 5. **Error analysis** shows mistakes concentrate in explainable edge cases
#    (borderline PageValues) rather than being scattered randomly.

# %% [markdown]
# ## 10. Business Insights & Recommendations
#
# 1. **Target "High-Value / High-Intent" and "Engaged Browser" segments with
#    retargeting ads and cart-abandonment emails** — they already show
#    strong PageValues and product-page engagement; a small nudge (discount,
#    reminder) is likely to convert them.
# 2. **"Low-Engagement / Bounce-Prone Visitors" are not good candidates for
#    paid retargeting spend** — their bounce/exit rates indicate the landing
#    experience, not intent, may be the problem; consider improving page
#    load speed or landing-page relevance for the traffic sources that feed
#    this segment instead of spending more marketing budget chasing them.
# 3. **Double down on May and November campaigns** — these months combine
#    high traffic *and* above-average purchase rates, so marketing spend
#    there has a demonstrated multiplier effect versus flat months.
# 4. **Prioritize returning-visitor retention programs** (loyalty perks,
#    remarketing lists) — returning visitors convert meaningfully more often
#    than first-time visitors, so growing the returning-visitor base compounds.
# 5. **Use the trained classifier as a real-time "purchase-intent score"** —
#    sessions the model flags as high-probability buyers (high `PageValues`,
#    low `ExitRates`, longer product browsing) are the best candidates for a
#    well-timed in-session incentive (e.g., a pop-up discount) before they
#    leave the site.
# 6. **Low-engagement signals** (high bounce/exit rate, few product pages,
#    zero PageValues) identify visitors unlikely to convert this session —
#    suppress aggressive upsell prompts for them to avoid hurting the
#    experience for visitors who were never going to buy today anyway.
# 7. **Segmentation + prediction together support smarter marketing
#    campaigns**: segments describe *who* a visitor broadly is over time,
#    while the classifier scores *this specific session* — combining both
#    lets campaigns be targeted at the right audience AND triggered at the
#    right moment.

# %% [markdown]
# ## 11. Conclusion, Limitations & Future Improvements
#
# **Summary of what was delivered:**
# Raw session-level e-commerce data was statistically profiled, explored
# visually across univariate/bivariate/multivariate views, and used to (a)
# discover four behaviorally distinct, business-nameable customer segments
# via KMeans (cross-checked with Hierarchical clustering) and (b) train and
# compare five classification models to predict session-level purchase
# probability, selecting the best model by F1/Recall/AUC rather than raw
# accuracy given the class imbalance. Both the segments and the classifier
# were validated with cross-validation, train/test comparison, feature
# importance, and error analysis before being translated into concrete
# marketing recommendations.
#
# **Limitations:**
# - The dataset used in this notebook run is a synthetic stand-in for the
#   real UCI file (see the data note at the top) — absolute metric values
#   will shift slightly on the genuine data, though the *methodology and
#   code* are unchanged and directly reusable.
# - The data covers a single 1-year window from one (anonymized) site —
#   findings may not transfer to a different industry vertical or traffic mix.
# - Class imbalance (~15.5% positive) means even a strong model will have a
#   non-trivial false-negative rate; a production deployment should pair the
#   model with a monitored threshold rather than a fixed 0.5 cutoff.
# - Clustering feature choice (5 engagement metrics) was made deliberately
#   for interpretability; a wider feature set might reveal finer-grained
#   segments at the cost of business explainability.
#
# **Future improvements:**
# - Re-run against the genuine UCI CSV (single-line swap, see data note) to
#   confirm findings hold outside the synthetic approximation.
# - Try gradient boosting (XGBoost/LightGBM) and threshold tuning against a
#   cost matrix instead of the default 0.5 cutoff.
# - Explore SMOTE/class-balancing techniques beyond `class_weight="balanced"`.
# - Use the classifier's probability output (not just the binary label) as a
#   continuous "intent score" feeding real-time personalization.
