"""
Spam Email Classifier
======================
Cybersecurity project: Intelligent spam filter using ML.

Dataset : SpamAssassin Public Corpus
          https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus

Techniques:
  - Email parsing (headers + body extraction)
  - Text preprocessing (cleaning, tokenising, stop-word removal)
  - Feature extraction: Bag-of-Words (BoW) AND TF-IDF
  - Classifiers: Naive Bayes, Logistic Regression, SVM, Random Forest
  - Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

Requirements:
    pip install scikit-learn pandas numpy matplotlib

HOW TO USE:
    Option A — Automatic (recommended):
        Extract easy_ham_2.zip so the folder "easy_ham" sits next to this script.
        The script finds it automatically.

    Option B — Manual:
        Set HAM_DIR below to the full path of your ham email folder.
        Optionally set SPAM_DIR if you have a real spam folder.

    No spam folder? No problem — the script generates realistic synthetic spam.
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────
import os, re, email, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1. PATHS  —  edit these if needed
# ─────────────────────────────────────────────────────────────
# The script tries several common locations automatically.
# Override by setting an explicit path string, e.g.:
#   HAM_DIR  = r"C:\Users\You\Downloads\easy_ham"
#   SPAM_DIR = r"C:\Users\You\Downloads\spam"

HAM_DIR  = None   # <- set to your ham folder path, or leave None for auto-detect
SPAM_DIR = None   # <- set to your spam folder path, or leave None to use synthetic spam

def _find_dir(candidates):
    """Return the first existing directory from a list of candidates."""
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None

if HAM_DIR is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    HAM_DIR = _find_dir([
        os.path.join(script_dir, "easy_ham"),
        os.path.join(script_dir, "data", "easy_ham"),
        os.path.join(script_dir, "easy_ham_2", "easy_ham"),
        os.path.join(script_dir, "dataset", "easy_ham"),
        "easy_ham",
        "data/easy_ham",
        "data\\easy_ham",
    ])

print("=" * 65)
print("SPAM EMAIL CLASSIFIER")
print("=" * 65)
print(f"\nHAM_DIR  : {HAM_DIR}")
print(f"SPAM_DIR : {SPAM_DIR}")

# ─────────────────────────────────────────────────────────────
# 2. EMAIL PARSER
# ─────────────────────────────────────────────────────────────
def parse_email_file(filepath):
    """Extract subject + body text from a raw email file."""
    try:
        with open(filepath, "rb") as f:
            msg = email.message_from_bytes(f.read())
        subject = msg.get("Subject", "") or ""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp  = str(part.get("Content-Disposition", ""))
                if ctype == "text/plain" and "attachment" not in disp:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode("utf-8", errors="ignore")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")
        return (subject + " " + body).strip()
    except Exception:
        return ""

def load_folder(directory, label):
    """Load all emails in a directory with a given label (0=ham, 1=spam)."""
    texts, labels = [], []
    if not directory or not os.path.isdir(directory):
        return texts, labels
    for fname in sorted(os.listdir(directory)):
        if fname.startswith(".") or fname.startswith("_"):
            continue
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            t = parse_email_file(fpath)
            if t.strip():
                texts.append(t)
                labels.append(label)
    return texts, labels

# ─────────────────────────────────────────────────────────────
# 3. SYNTHETIC SPAM GENERATOR
# ─────────────────────────────────────────────────────────────
SPAM_TEMPLATES = [
    "CONGRATULATIONS! You WON {prize}! Click HERE NOW to CLAIM your FREE {item}! "
    "Limited time! ACT FAST! Call {phone} or visit {url}",
    "URGENT: Your account SUSPENDED. Verify details at {url} to avoid closure. "
    "Reply with your password and credit card number immediately.",
    "Make ${amount} per day working from home! FREE training. Click {url} to START TODAY! {phone}",
    "You selected for FREE {item}! Send bank details to {email} — receive {prize} in 24 hours!",
    "HOT SINGLES in your area! Click {url} NOW. FREE access for {hours} hours only!",
    "WEIGHT LOSS MIRACLE! Lose {amount} lbs in 2 weeks! Buy at {url} — 90% OFF today! {phone}",
    "Your PayPal account shows unusual activity. VERIFY NOW at {url} or account locked!",
    "Microsoft Lottery selected your email. You WON {prize}. Send fee to claim. {phone}",
    "CHEAP MEDICATIONS online! No prescription needed. 80% off. Order at {url}.",
    "Investment opportunity! Turn ${amount} into ${prize} in 30 days GUARANTEED. {email}",
    "FREE iPhone {item} giveaway! Only {amount} spots left. Register at {url} NOW!",
    "Nigerian prince with {prize} needing transfer. Share 40%. Send bank details to {email}.",
    "WINNER! Your email address won our monthly draw. Prize: {prize}. Claim: {url}",
    "Lose weight fast! Our pill burns fat while you sleep. Order now at {url}. Only ${amount}!",
    "Your subscription expires today! Renew at {url} or lose access. Credit card needed.",
]

def make_spam(n):
    prizes  = ["$1,000,000","$500 Gift Card","a Brand New Car","$10,000 CASH","$250,000"]
    items   = ["iPhone 15","MacBook Pro","PS5","iPad","Samsung TV","AirPods Pro"]
    phones  = ["1-800-555-0199","0800-123-456","+1-555-987-6543","1-888-999-0000"]
    urls    = ["http://bit.ly/claimNow","http://free-gift.xyz/win",
               "http://verify-now.net/secure","http://deals99.top/offer",
               "http://prize-centre.biz/claim"]
    emails_ = ["support@prizecentre.com","noreply@winnings.biz","help@verify.net"]
    amounts = ["500","1000","5000","250","99","2500"]
    hours   = ["24","48","6","12","3"]
    spams = []
    for _ in range(n):
        t = random.choice(SPAM_TEMPLATES).format(
            prize=random.choice(prizes), item=random.choice(items),
            phone=random.choice(phones), url=random.choice(urls),
            email=random.choice(emails_), amount=random.choice(amounts),
            hours=random.choice(hours),
        )
        spams.append(t)
    return spams

# ─────────────────────────────────────────────────────────────
# 4. LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1 — LOADING DATA")
print("=" * 65)

ham_texts, ham_labels = load_folder(HAM_DIR, label=0)
print(f"  Ham emails loaded  : {len(ham_texts):,}")

if len(ham_texts) == 0:
    print("\n  [WARNING] No ham emails found in the detected path.")
    print("  Using 2,500 synthetic ham emails for demonstration.")
    HAM_SYNTH_TEMPLATES = [
        "Hi {name}, please find the meeting notes attached. Let me know if you have questions.",
        "The {project} report is ready for review. I have uploaded it to the shared drive.",
        "Reminder: team standup tomorrow at {time}. Agenda includes {topic}.",
        "Thanks for your message. I will get back to you by {day}.",
        "Python {version} has been released with improvements to {feature}. Check the changelog.",
        "Your order #{order} has been shipped. Expected delivery: {day}.",
        "Please review the pull request for {project}. Feedback needed by {day}.",
        "Conference call scheduled for {time} on {day}. Dial-in details below.",
        "The server maintenance window is {time} on {day}. Brief downtime expected.",
        "New article: {topic} — key insights for {project} teams.",
    ]
    names    = ["John","Sarah","Mike","Emily","David","Lisa","James","Anna"]
    projects = ["Q3 report","website redesign","data pipeline","API integration","security audit"]
    times    = ["9:00 AM","2:00 PM","10:30 AM","3:00 PM","11:00 AM"]
    days     = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    topics   = ["machine learning","cloud migration","product roadmap","budget review","team KPIs"]
    versions = ["3.12","3.13","2.0","4.1"]
    features = ["async support","type hints","performance","error handling"]
    orders   = [str(random.randint(10000,99999)) for _ in range(20)]

    for _ in range(2500):
        t = random.choice(HAM_SYNTH_TEMPLATES).format(
            name=random.choice(names), project=random.choice(projects),
            time=random.choice(times), day=random.choice(days),
            topic=random.choice(topics), version=random.choice(versions),
            feature=random.choice(features), order=random.choice(orders),
        )
        ham_texts.append(t)
        ham_labels.append(0)

if SPAM_DIR and os.path.isdir(SPAM_DIR):
    spam_texts, spam_labels = load_folder(SPAM_DIR, label=1)
    print(f"  Spam emails loaded : {len(spam_texts):,}")
else:
    print("  No spam folder found — generating synthetic spam...")
    spam_texts  = make_spam(n=len(ham_texts))
    spam_labels = [1] * len(spam_texts)
    print(f"  Synthetic spam     : {len(spam_texts):,}")

all_texts  = ham_texts  + spam_texts
all_labels = ham_labels + spam_labels

df = pd.DataFrame({"text": all_texts, "label": all_labels})
df = df[df["text"].str.strip().str.len() > 10].copy()   # drop near-empty rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  Total emails  : {len(df):,}")
print(f"  Ham  (0)      : {(df['label']==0).sum():,}")
print(f"  Spam (1)      : {(df['label']==1).sum():,}")

# ─────────────────────────────────────────────────────────────
# 5. STOP WORDS (no NLTK needed)
# ─────────────────────────────────────────────────────────────
STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","as","at","be","because","been","before","being","below","between",
    "both","but","by","can","cannot","could","did","do","does","doing","down",
    "during","each","few","for","from","further","get","got","had","has","have",
    "having","he","her","here","him","his","how","i","if","in","into","is","it",
    "its","itself","just","let","may","me","more","most","my","no","nor","not",
    "now","of","off","on","once","only","or","other","our","out","own","same",
    "she","should","so","some","such","than","that","the","their","them","then",
    "there","these","they","this","those","through","to","too","under","until",
    "up","us","very","was","we","were","what","when","where","which","while",
    "who","whom","why","will","with","would","you","your","yours",
}

# ─────────────────────────────────────────────────────────────
# 6. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+",        " emailaddr ", text)
    text = re.sub(r"<[^>]+>",        " ", text)
    text = re.sub(r"\$[\d,]+",       " moneymention ", text)
    text = re.sub(r"\d+",            " ", text)
    text = re.sub(r"[^a-z\s]",       " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

print("\n" + "=" * 65)
print("STEP 2 — TEXT PREPROCESSING")
print("=" * 65)

df["clean_text"] = df["text"].apply(clean_text)

# Drop any rows that became empty after cleaning
df = df[df["clean_text"].str.strip().str.len() > 5].reset_index(drop=True)
print(f"  Emails after cleaning : {len(df):,}")

# Safe sample display
ham_df   = df[df["label"] == 0].reset_index(drop=True)
spam_df  = df[df["label"] == 1].reset_index(drop=True)

print("\n  Sample HAM  (cleaned) :")
print("  " + ham_df["clean_text"].iloc[0][:200] + "...")
print("\n  Sample SPAM (cleaned) :")
print("  " + spam_df["clean_text"].iloc[0][:200] + "...")

ham_words  = " ".join(ham_df["clean_text"]).split()
spam_words = " ".join(spam_df["clean_text"]).split()
top_ham    = Counter(ham_words).most_common(15)
top_spam   = Counter(spam_words).most_common(15)

print(f"\n  Top 5 HAM  words : {[w for w,_ in top_ham[:5]]}")
print(f"  Top 5 SPAM words : {[w for w,_ in top_spam[:5]]}")

# ─────────────────────────────────────────────────────────────
# 7. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — FEATURE EXTRACTION (BoW + TF-IDF)")
print("=" * 65)

X = df["clean_text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

bow_vec = CountVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train_bow  = bow_vec.fit_transform(X_train)
X_test_bow   = bow_vec.transform(X_test)
print(f"\n  BoW   matrix : {X_train_bow.shape}")

tfidf_vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                             min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf  = tfidf_vec.transform(X_test)
print(f"  TF-IDF matrix: {X_train_tfidf.shape}")

# ─────────────────────────────────────────────────────────────
# 8. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — TRAINING & EVALUATING CLASSIFIERS")
print("=" * 65)

MODELS = {
    "Naive Bayes (BoW)"      : (MultinomialNB(alpha=0.1),                          X_train_bow,   X_test_bow),
    "Naive Bayes (TF-IDF)"   : (MultinomialNB(alpha=0.1),                          X_train_tfidf, X_test_tfidf),
    "Logistic Reg (BoW)"     : (LogisticRegression(max_iter=1000, C=1.0),           X_train_bow,   X_test_bow),
    "Logistic Reg (TF-IDF)"  : (LogisticRegression(max_iter=1000, C=1.0),           X_train_tfidf, X_test_tfidf),
    "Linear SVM (TF-IDF)"    : (LinearSVC(max_iter=2000, C=1.0),                   X_train_tfidf, X_test_tfidf),
    "Random Forest (TF-IDF)" : (RandomForestClassifier(n_estimators=100,
                                                        random_state=42, n_jobs=-1), X_train_tfidf, X_test_tfidf),
}

results       = {}
trained_clfs  = {}

for name, (clf, Xtr, Xte) in MODELS.items():
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(Xte)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(Xte)
    else:
        scores = preds.astype(float)

    results[name] = {
        "Accuracy" : round(accuracy_score(y_test, preds),       4),
        "Precision": round(precision_score(y_test, preds),      4),
        "Recall"   : round(recall_score(y_test, preds),         4),
        "F1-Score" : round(f1_score(y_test, preds),             4),
        "ROC-AUC"  : round(roc_auc_score(y_test, scores),       4),
    }
    trained_clfs[name] = (clf, Xte, scores)
    r = results[name]
    print(f"\n  {name}")
    print(f"    Acc={r['Accuracy']:.4f}  Prec={r['Precision']:.4f}  "
          f"Rec={r['Recall']:.4f}  F1={r['F1-Score']:.4f}  AUC={r['ROC-AUC']:.4f}")

results_df = pd.DataFrame(results).T.sort_values("F1-Score", ascending=False)
print("\n\n── Leaderboard (sorted by F1-Score) ──")
print(results_df.to_string())

best_name = results_df.index[0]
print(f"\n  Best model: {best_name}")

# ─────────────────────────────────────────────────────────────
# 9. DETAILED CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 5 — DETAILED REPORT: {best_name}")
print("=" * 65)

best_clf, Xte_best, _ = trained_clfs[best_name]
best_preds = best_clf.predict(Xte_best)
print("\n" + classification_report(y_test, best_preds, target_names=["Ham", "Spam"]))

# ─────────────────────────────────────────────────────────────
# 10. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
print("Generating plots ...")

# Figure 1 — EDA
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Spam Classifier — EDA", fontsize=15, fontweight="bold")

counts = df["label"].value_counts().sort_index()
axes[0].bar(["Ham", "Spam"], counts.values, color=["#4d96ff", "#e50914"], width=0.5)
axes[0].set_title("Class Distribution"); axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, f"{v:,}", ha="center", fontweight="bold")

df["word_count"] = df["clean_text"].str.split().str.len()
axes[1].hist(df[df["label"]==0]["word_count"].clip(upper=500),
             bins=50, alpha=0.7, color="#4d96ff", label="Ham", density=True)
axes[1].hist(df[df["label"]==1]["word_count"].clip(upper=500),
             bins=50, alpha=0.7, color="#e50914", label="Spam", density=True)
axes[1].set_title("Email Length Distribution")
axes[1].set_xlabel("Word Count (clipped 500)"); axes[1].set_ylabel("Density")
axes[1].legend()

s_words, s_freqs = zip(*top_spam[:12])
axes[2].barh(list(s_words), list(s_freqs), color="#e50914")
axes[2].set_title("Top 12 Spam Words"); axes[2].set_xlabel("Frequency")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [Saved] eda_plots.png")

# Figure 2 — Model Comparison
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")

for i, metric in enumerate(metrics):
    vals   = results_df[metric]
    colors = ["#e50914" if n == best_name else "#4d96ff" for n in results_df.index]
    labels = [n.replace(" (", "\n(") for n in results_df.index]
    axes[i].barh(labels, vals, color=colors)
    axes[i].set_title(metric); axes[i].set_xlim(0.4, 1.05); axes[i].invert_yaxis()
    for j, v in enumerate(vals):
        axes[i].text(v + 0.003, j, f"{v:.3f}", va="center", fontsize=7)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [Saved] model_comparison.png")

# Figure 3 — Best model deep dive
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f"Best Model: {best_name}", fontsize=14, fontweight="bold")

# Confusion matrix
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# ROC curves
palette = ["#e50914","#ff6b6b","#4d96ff","#6bcb77","#ffd93d","#b5a0f5"]
for idx, (mname, (clf, Xte, sc)) in enumerate(trained_clfs.items()):
    if mname == "Linear SVM (TF-IDF)":
        continue   # LinearSVC has no probability — skip clean ROC
    fpr, tpr, _ = roc_curve(y_test, sc)
    auc = results[mname]["ROC-AUC"]
    lw  = 2.5 if mname == best_name else 1.2
    axes[1].plot(fpr, tpr, color=palette[idx % len(palette)],
                 linewidth=lw, label=f"{mname} ({auc:.3f})")
axes[1].plot([0,1],[0,1],"k--", lw=0.8)
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("ROC Curves"); axes[1].legend(fontsize=7, loc="lower right")

# Feature importance
feat_names = tfidf_vec.get_feature_names_out()
if hasattr(best_clf, "coef_"):
    coef = best_clf.coef_[0] if best_clf.coef_.ndim > 1 else best_clf.coef_
    top_sp  = np.argsort(coef)[-15:]
    top_ham2= np.argsort(coef)[:15]
    all_idx = list(top_ham2) + list(top_sp)
    all_w   = [feat_names[i] for i in all_idx]
    all_v   = [coef[i] for i in all_idx]
    bar_col = ["#4d96ff"]*15 + ["#e50914"]*15
    axes[2].barh(all_w, all_v, color=bar_col)
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_title("Top Features\n(red=spam  blue=ham)")
    axes[2].set_xlabel("Coefficient")
elif hasattr(best_clf, "feature_importances_"):
    top_idx = np.argsort(best_clf.feature_importances_)[-20:]
    axes[2].barh([feat_names[i] for i in top_idx],
                  best_clf.feature_importances_[top_idx], color="#e50914")
    axes[2].set_title("Top 20 Feature Importances")
else:
    axes[2].text(0.5,0.5,"Not available\nfor this model",
                 ha="center", va="center", transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig("best_model_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [Saved] best_model_analysis.png")

# Figure 4 — Ham vs Spam words
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Most Frequent Words — Ham vs Spam", fontsize=14, fontweight="bold")
h_w, h_f = zip(*top_ham[:15])
axes[0].barh(list(h_w), list(h_f), color="#4d96ff")
axes[0].set_title("Top 15 Ham Words"); axes[0].set_xlabel("Frequency"); axes[0].invert_yaxis()
s_w, s_f = zip(*top_spam[:15])
axes[1].barh(list(s_w), list(s_f), color="#e50914")
axes[1].set_title("Top 15 Spam Words"); axes[1].set_xlabel("Frequency"); axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig("word_frequency.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [Saved] word_frequency.png")

# ─────────────────────────────────────────────────────────────
# 11. LIVE PREDICTION DEMO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — LIVE PREDICTION DEMO")
print("=" * 65)

# Choose vectoriser matching best model
_vec = tfidf_vec if "TF-IDF" in best_name else bow_vec

def predict_email(text, clf=best_clf, vectorizer=_vec):
    cleaned  = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred     = clf.predict(features)[0]
    if hasattr(clf, "predict_proba"):
        conf = clf.predict_proba(features)[0][pred] * 100
    elif hasattr(clf, "decision_function"):
        s    = abs(clf.decision_function(features)[0])
        conf = s / (s + 1) * 100
    else:
        conf = 100.0
    return ("SPAM" if pred == 1 else "HAM"), round(conf, 1)

TEST_EMAILS = [
    ("HAM",  "Hi Sarah, just checking in on the project timeline. "
             "Can we set up a call for Thursday morning to review the Q3 report?"),
    ("HAM",  "The Python 3.13 release is out. New features include improved type hints "
             "and faster performance. Details in the changelog on python.org."),
    ("HAM",  "Your order #84726 has been shipped and will arrive by Friday. "
             "Track it using the link in your account dashboard."),
    ("SPAM", "CONGRATULATIONS!! You WON $1,000,000 CASH PRIZE! "
             "Click HERE NOW to CLAIM your FREE reward! ACT FAST limited time!!!"),
    ("SPAM", "URGENT: Your bank account has been compromised! "
             "Verify details immediately at http://secure-verify-now.xyz or lose access FOREVER!"),
    ("SPAM", "Make $5000 per week working from home! 100% FREE signup. "
             "No experience needed. Click the link now to start earning today!"),
]

print(f"\n  {'Expected':<8}  {'Predicted':<8}  {'Confidence':>10}  {'Match':<6}  Preview")
print("  " + "─" * 72)
correct = 0
for expected, text in TEST_EMAILS:
    label, conf = predict_email(text)
    match = "✓" if label == expected else "✗"
    if label == expected:
        correct += 1
    print(f"  {expected:<8}  {label:<8}  {conf:>9.1f}%  {match:<6}  {text[:55]}...")

print(f"\n  Demo accuracy: {correct}/{len(TEST_EMAILS)} correct")

# ─────────────────────────────────────────────────────────────
# 12. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — 5-FOLD CROSS VALIDATION")
print("=" * 65)

X_full = tfidf_vec.transform(X) if "TF-IDF" in best_name else bow_vec.transform(X)
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1  = cross_val_score(best_clf, X_full, y, cv=skf, scoring="f1")
cv_acc = cross_val_score(best_clf, X_full, y, cv=skf, scoring="accuracy")

print(f"\n  Model      : {best_name}")
print(f"  CV F1      : {cv_f1.mean():.4f}  (+/- {cv_f1.std():.4f})")
print(f"  CV Accuracy: {cv_acc.mean():.4f}  (+/- {cv_acc.std():.4f})")
print(f"  Fold F1s   : {[round(v,4) for v in cv_f1]}")

# ─────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
bm = results[best_name]
print(f"""
  Best Model   : {best_name}
  ─────────────────────────────────────────────────────
  Accuracy     : {bm['Accuracy']}
  Precision    : {bm['Precision']}  <- of flagged spam, how many truly are spam
  Recall       : {bm['Recall']}  <- of all real spam, how many we caught
  F1-Score     : {bm['F1-Score']}  <- harmonic mean of precision & recall
  ROC-AUC      : {bm['ROC-AUC']}  <- area under ROC curve (1.0 = perfect)

  Pipeline
  ─────────────────────────────────────────────────────
  1. Email parsing     raw files -> subject + body text
  2. Text cleaning     lowercase, strip URLs/HTML/numbers
  3. Tokenisation      remove stop words, short tokens
  4. BoW features      CountVectorizer, 1-2 grams, 10k features
  5. TF-IDF features   TfidfVectorizer, sublinear_tf=True
  6. 6 models trained  NB, LR, SVM, RF x BoW/TF-IDF
  7. Best selected     by F1-Score on held-out test set

  Saved Plots
  ─────────────────────────────────────────────────────
  eda_plots.png            class dist, lengths, top spam words
  model_comparison.png     all models x all metrics
  best_model_analysis.png  confusion matrix, ROC, top features
  word_frequency.png       ham vs spam word frequency
  ─────────────────────────────────────────────────────
""")
