"""
Netflix Movie Rating Prediction
================================
Advanced ML project using regression techniques to predict movie ratings.

Requirements (install with pip):
    pip install pandas numpy matplotlib scikit-learn
"""

# =============================================================
# 0. IMPORTS
# =============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================
# 1. LOAD DATA
# =============================================================
print("=" * 60)
print("STEP 1 — LOADING DATA")
print("=" * 60)

MOVIE_PATH  = "Netflix_Dataset_Movie.csv"
RATING_PATH = "Netflix_Dataset_Rating.csv"

movies = pd.read_csv(MOVIE_PATH)
print(f"Movies dataset : {movies.shape[0]:,} rows x {movies.shape[1]} cols")
print(movies.head())

if os.path.exists(RATING_PATH):
    ratings = pd.read_csv(RATING_PATH,
                          names=["Movie_ID", "Cust_ID", "Rating", "Date"],
                          skiprows=1)
    print(f"\nRatings dataset: {ratings.shape[0]:,} rows x {ratings.shape[1]} cols")
    print(ratings.head())
else:
    print("\n[INFO] Netflix_Dataset_Rating.csv not found — generating synthetic ratings.")
    n_users   = 2_000
    n_ratings = 100_000
    rng = np.random.default_rng(42)

    movie_ids  = movies["Movie_ID"].values
    movie_bias = (np.where(movies["Year"] < 1990, -0.3, 0.0) +
                  np.where(movies["Year"] > 2000,  0.2, 0.0))
    bias_dict  = dict(zip(movie_ids, movie_bias))

    sampled = rng.choice(movie_ids, size=n_ratings)
    biases  = np.array([bias_dict[m] for m in sampled])
    raw     = np.clip(rng.normal(3.5 + biases, 1.0), 1, 5).round().astype(int)

    ratings = pd.DataFrame({
        "Movie_ID": sampled,
        "Cust_ID" : rng.integers(1, n_users + 1, size=n_ratings),
        "Rating"  : raw,
        "Date"    : pd.date_range("2000-01-01", periods=n_ratings, freq="1h"),
    })
    print(f"Synthetic ratings: {ratings.shape[0]:,} rows")
    print(ratings.head())

# =============================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================
print("\n" + "=" * 60)
print("STEP 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("Rating distribution:")
print(ratings["Rating"].value_counts().sort_index())
print(f"\nMean rating: {ratings['Rating'].mean():.3f}")
print(f"Std  rating: {ratings['Rating'].std():.3f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Netflix Dataset — EDA", fontsize=16, fontweight="bold")

rc = ratings["Rating"].value_counts().sort_index()
axes[0, 0].bar(rc.index, rc.values,
               color=["#e50914","#ff6b6b","#ffd93d","#6bcb77","#4d96ff"])
axes[0, 0].set_title("Rating Distribution")
axes[0, 0].set_xlabel("Rating (1-5)")
axes[0, 0].set_ylabel("Count")

movies["Decade"] = (movies["Year"] // 10) * 10
dc = movies["Decade"].value_counts().sort_index()
axes[0, 1].bar(dc.index.astype(str), dc.values, color="#e50914")
axes[0, 1].set_title("Movies per Decade")
axes[0, 1].set_xlabel("Decade")
axes[0, 1].set_ylabel("Count")
axes[0, 1].tick_params(axis="x", rotation=45)

merged_eda = ratings.merge(movies[["Movie_ID","Year","Decade"]], on="Movie_ID")
abd = merged_eda.groupby("Decade")["Rating"].mean()
axes[0, 2].plot(abd.index, abd.values, marker="o", color="#e50914", linewidth=2)
axes[0, 2].set_title("Avg Rating by Decade")
axes[0, 2].set_xlabel("Decade"); axes[0, 2].set_ylabel("Avg Rating")
axes[0, 2].set_ylim(1, 5)

top_rated = (ratings.groupby("Movie_ID")["Rating"].count()
             .nlargest(20).reset_index()
             .merge(movies[["Movie_ID","Name"]], on="Movie_ID"))
axes[1, 0].barh(top_rated["Name"].str[:30], top_rated["Rating"], color="#e50914")
axes[1, 0].set_title("Top 20 Most-Rated Movies")
axes[1, 0].set_xlabel("Number of Ratings"); axes[1, 0].invert_yaxis()

top_avg = (ratings.groupby("Movie_ID")["Rating"]
           .agg(["mean","count"]).query("count >= 50")
           .nlargest(20,"mean").reset_index()
           .merge(movies[["Movie_ID","Name"]], on="Movie_ID"))
axes[1, 1].barh(top_avg["Name"].str[:30], top_avg["mean"], color="#4d96ff")
axes[1, 1].set_title("Top 20 Highest-Avg-Rated\n(>=50 ratings)")
axes[1, 1].set_xlabel("Avg Rating"); axes[1, 1].invert_yaxis()
axes[1, 1].set_xlim(0, 5)

yr = merged_eda.groupby("Year")["Rating"].mean().reset_index()
axes[1, 2].scatter(yr["Year"], yr["Rating"], alpha=0.6, color="#e50914",
                   edgecolors="black", linewidths=0.3)
axes[1, 2].set_title("Year vs Avg Rating")
axes[1, 2].set_xlabel("Release Year"); axes[1, 2].set_ylabel("Avg Rating")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] eda_plots.png")

# =============================================================
# 3. FEATURE ENGINEERING
# =============================================================
print("\n" + "=" * 60)
print("STEP 3 — FEATURE ENGINEERING")
print("=" * 60)

movie_stats = (ratings.groupby("Movie_ID")["Rating"]
               .agg(avg_rating="mean", rating_count="count",
                    rating_std="std", rating_min="min", rating_max="max")
               .reset_index())
movie_stats["rating_std"] = movie_stats["rating_std"].fillna(0)

df = movies.merge(movie_stats, on="Movie_ID", how="inner")
print(f"Merged dataset: {df.shape}")

df["movie_age"]        = 2024 - df["Year"]
df["log_rating_count"] = np.log1p(df["rating_count"])
df["name_length"]      = df["Name"].str.len()
df["name_word_count"]  = df["Name"].str.split().str.len()
df["has_number"]       = df["Name"].str.contains(r"\d", regex=True).astype(int)
df["has_sequel_word"]  = df["Name"].str.lower().str.contains(
    r"\b(2|ii|iii|iv|v|part|sequel|returns|again)\b", regex=True).astype(int)
df["popularity_tier"]  = pd.qcut(df["rating_count"], q=5,
                                  labels=[1,2,3,4,5]).astype(int)
df["Decade"] = (df["Year"] // 10) * 10
df = pd.get_dummies(df, columns=["Decade"], prefix="decade")

def get_era(year):
    if year < 1940:   return "silent_classic"
    elif year < 1960: return "golden_age"
    elif year < 1980: return "new_hollywood"
    elif year < 2000: return "blockbuster_era"
    else:             return "streaming_era"

df["era"] = df["Year"].apply(get_era)
df = pd.get_dummies(df, columns=["era"], prefix="era")
print("Feature engineering done.")

# =============================================================
# 4. PREPARE ML DATA
# =============================================================
print("\n" + "=" * 60)
print("STEP 4 — PREPARING ML DATA")
print("=" * 60)

TARGET    = "avg_rating"
DROP_COLS = ["Movie_ID", "Name", "Year", TARGET]

X = df[[c for c in df.columns if c not in DROP_COLS]].select_dtypes(include=[np.number])
y = df[TARGET]

print(f"Features : {X.shape[1]}  |  Samples: {X.shape[0]:,}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# =============================================================
# 5. TRAIN & EVALUATE ALL MODELS
# =============================================================
print("\n" + "=" * 60)
print("STEP 5 — TRAINING MODELS")
print("=" * 60)

models = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=1.0),
    "Lasso Regression"  : Lasso(alpha=0.01),
    "ElasticNet"        : ElasticNet(alpha=0.01, l1_ratio=0.5),
    "Decision Tree"     : DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
}
LINEAR = {"Linear Regression","Ridge Regression","Lasso Regression","ElasticNet"}

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
kf        = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    Xtr = X_train_s if name in LINEAR else X_train.values
    Xte = X_test_s  if name in LINEAR else X_test.values
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    cv    = -cross_val_score(model, Xtr, y_train, cv=kf,
                              scoring="neg_mean_absolute_error").mean()
    results[name] = {"MAE":round(mae,4),"RMSE":round(rmse,4),
                     "R2":round(r2,4),"CV_MAE":round(cv,4)}
    print(f"  {name:<28}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  CV_MAE={cv:.4f}")

results_df = pd.DataFrame(results).T.sort_values("RMSE")
print("\n── Leaderboard (sorted by RMSE) ──")
print(results_df.to_string())

# =============================================================
# 6. MODEL COMPARISON PLOT
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")
best_name = results_df.index[0]
colors = ["#e50914" if n == best_name else "#4d96ff" for n in results_df.index]

for i, metric in enumerate(["MAE","RMSE","R2"]):
    axes[i].barh(results_df.index, results_df[metric], color=colors)
    axes[i].set_title(metric); axes[i].set_xlabel(metric)
    axes[i].invert_yaxis()
    for j, val in enumerate(results_df[metric]):
        axes[i].text(val+0.001, j, f"{val:.4f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] model_comparison.png")

# =============================================================
# 7. BEST MODEL — ACTUAL vs PREDICTED + FEATURE IMPORTANCE
# =============================================================
print("\n" + "=" * 60)
print("STEP 6 — BEST MODEL DEEP DIVE")
print("=" * 60)

best_model = models[best_name]
print(f"Best model: {best_name}")
Xte_best = X_test_s if best_name in LINEAR else X_test.values
y_pred   = best_model.predict(Xte_best)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Best Model: {best_name}", fontsize=13, fontweight="bold")

axes[0].scatter(y_test, y_pred, alpha=0.4, color="#e50914", edgecolors="none", s=20)
mn, mx = float(y_test.min()), float(y_test.max())
axes[0].plot([mn,mx],[mn,mx],"k--",linewidth=1.5,label="Perfect fit")
axes[0].set_xlabel("Actual Rating"); axes[0].set_ylabel("Predicted Rating")
axes[0].set_title("Actual vs Predicted"); axes[0].legend()

if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=X.columns).nlargest(15)
    fi.plot(kind="barh", ax=axes[1], color="#e50914")
    axes[1].set_title("Top 15 Feature Importances"); axes[1].invert_yaxis()
elif hasattr(best_model, "coef_"):
    pd.Series(best_model.coef_, index=X.columns).abs().nlargest(15)\
      .plot(kind="barh", ax=axes[1], color="#4d96ff")
    axes[1].set_title("Top 15 |Coefficients|"); axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("best_model_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] best_model_analysis.png")

# =============================================================
# 8. RESIDUAL ANALYSIS
# =============================================================
residuals = y_test.values - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Residual Analysis", fontsize=13, fontweight="bold")

axes[0].scatter(y_pred, residuals, alpha=0.3, color="#e50914", s=15)
axes[0].axhline(0, color="black", linewidth=1.5, linestyle="--")
axes[0].set_xlabel("Predicted Rating"); axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")

axes[1].hist(residuals, bins=40, color="#e50914", edgecolor="black", alpha=0.8)
axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] residuals.png")

# =============================================================
# 9. SAMPLE PREDICTIONS
# =============================================================
print("\n" + "=" * 60)
print("STEP 7 — SAMPLE PREDICTIONS")
print("=" * 60)

sample   = df.sample(10, random_state=7)
X_sample = sample[X.columns]
preds_s  = best_model.predict(
    scaler.transform(X_sample) if best_name in LINEAR else X_sample.values)

print(pd.DataFrame({
    "Movie Name"    : sample["Name"].values,
    "Year"          : sample["Year"].values,
    "Actual Avg Rtg": sample["avg_rating"].round(3).values,
    "Predicted Rtg" : preds_s.round(3),
    "Error"         : (sample["avg_rating"].values - preds_s).round(3),
}).to_string(index=False))

# =============================================================
# 10. FINAL SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
bm = results[best_name]
print(f"Best model : {best_name}")
print(f"  MAE      : {bm['MAE']}")
print(f"  RMSE     : {bm['RMSE']}")
print(f"  R2       : {bm['R2']}")
print(f"  CV MAE   : {bm['CV_MAE']}")
print("""
-------------------------------------------------------
Metric Guide
  MAE    — avg prediction error in rating points
  RMSE   — penalises large errors more than MAE
  R2     — 1.0=perfect | 0.0=no better than mean
  CV MAE — cross-validated MAE (generalisation proxy)

Saved output files
  eda_plots.png           EDA visualisations
  model_comparison.png    All model metrics side-by-side
  best_model_analysis.png Actual vs Predicted + importance
  residuals.png           Residual diagnosis plots
-------------------------------------------------------
""")
