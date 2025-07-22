import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
INPUT_FILE = "processed_datasets/combined_dataset_with_labels.csv"
TEXT_COL = "preprocessed_text"
LABEL_COL = "review_label"
SAMPLE_FRAC = 0.1
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 30000

# === LOAD & SAMPLE DATA ===
print("Loading data...")
df = pd.read_csv(INPUT_FILE, usecols=[TEXT_COL, LABEL_COL])
df = df[df[LABEL_COL].isin([0, 1, 2])].dropna()
df_sample = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)

# === SPLIT SAMPLE ===
X_train, X_val, y_train, y_val = train_test_split(
    df_sample[TEXT_COL],
    df_sample[LABEL_COL],
    test_size=0.2,
    stratify=df_sample[LABEL_COL],
    random_state=RANDOM_STATE
)

# === PIPELINE: TF-IDF + LOGISTIC REGRESSION ===
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("logreg", LogisticRegression(
        solver="saga",
        class_weight="balanced",
        multi_class="multinomial",
        n_jobs=-1
    ))
])

# === HYPERPARAMETER GRID ===
param_dist = {
    "logreg__C": np.logspace(-2, 1, 6),  # [0.01, 0.0316, ..., 10]
    "logreg__max_iter": [300, 500, 700],
}

# === RANDOMIZED SEARCH ===
print("Starting hyperparameter tuning...")
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=6,  # Try 6 combinations
    cv=3,
    scoring="f1_macro",
    verbose=2,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

random_search.fit(X_train, y_train)

# === BEST MODEL ===
print("\nBest Hyperparameters:")
print(random_search.best_params_)

print("\nValidation Performance:")
y_pred = random_search.predict(X_val)
print(classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

print("\nTraining final model on full dataset...")
final_vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    stop_words="english"
)
X_full_tfidf = final_vectorizer.fit_transform(df[TEXT_COL])

final_model = LogisticRegression(
    C=random_search.best_params_["logreg__C"],
    max_iter=random_search.best_params_["logreg__max_iter"],
    solver="saga",
    class_weight="balanced",
    multi_class="multinomial",
    n_jobs=-1
)
final_model.fit(X_full_tfidf, df[LABEL_COL])

# === SAVE FINAL MODEL ===
print("\nSaving model and vectorizer...")
joblib.dump(final_model, "tfidf_logreg_model.pkl")
joblib.dump(final_vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Done. Model and vectorizer saved.")
