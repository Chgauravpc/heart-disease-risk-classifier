"""Cardiac Risk Classifier - comparative study of six scaled ML pipelines.

Usage:
    python src/heart_risk.py                # uses data/heart-disease.csv, writes to results/
    python src/heart_risk.py --data path.csv --out results

Steps: EDA plots -> stratified split -> six StandardScaler pipelines with stratified
5-fold CV -> GridSearchCV on SVM and Logistic Regression -> confusion matrices, ROC and
precision-recall curves -> threshold study -> Random Forest and permutation importance.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
GREEN, RED = "#3d9970", "#e07a5f"


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    return df.drop("target", axis=1), df["target"], df


def eda(df, out):
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    for ax, col in zip(axs, ["age", "thalach", "chol", "oldpeak"]):
        sns.boxplot(data=df, x="target", y=col, hue="target", palette=[RED, GREEN], legend=False, ax=ax)
        ax.set_title(col)
    fig.suptitle("Key numeric features by diagnosis (0 = healthy, 1 = disease)")
    fig.savefig(out / "01_numeric_boxplots.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axs.ravel(), ["sex", "cp", "exang", "slope", "ca", "thal"]):
        (pd.crosstab(df[col], df.target, normalize="index") * 100).plot(
            kind="bar", stacked=True, ax=ax, color=[RED, GREEN], legend=False)
        ax.set_title(col); ax.set_ylabel("% of patients"); ax.tick_params(axis="x", rotation=0)
    axs[0, 0].legend(["Healthy", "Disease"], fontsize=8)
    fig.suptitle("Disease rate within each categorical feature"); fig.tight_layout()
    fig.savefig(out / "02_categorical_rates.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    corr = df.corr()["target"].drop("target").sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    corr.plot.barh(color=np.where(corr > 0, GREEN, RED), ax=ax)
    ax.set_title("Pearson correlation of each feature with target")
    fig.savefig(out / "03_target_correlation.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return corr


def candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN (k=5)": KNeighborsClassifier(),
        "SVM (RBF)": SVC(probability=True, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def compare(Xtr, Xte, ytr, yte, out):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv, rows = {}, []
    for name, est in candidates().items():
        pipe = make_pipeline(StandardScaler(), est)
        cv[name] = cross_val_score(pipe, Xtr, ytr, cv=skf)
        pipe.fit(Xtr, ytr)
        pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        rows.append([name, cv[name].mean(), accuracy_score(yte, pred), precision_score(yte, pred),
                     recall_score(yte, pred), f1_score(yte, pred), roc_auc_score(yte, proba)])
    table = pd.DataFrame(rows, columns=["Model", "CV acc (mean)", "Test acc", "Precision", "Recall", "F1", "ROC-AUC"]).round(3)
    table.to_csv(out / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 3)); ax.axis("off")
    t = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
    for (r, _), cell in t.get_celld().items():
        if r == 0: cell.set_facecolor("#1f4e79"); cell.set_text_props(color="w", weight="bold")
    ax.set_title("Model comparison (scaled pipelines, stratified 75/25 split)", weight="bold")
    fig.savefig(out / "04_model_comparison.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(list(cv.values()), tick_labels=list(cv))
    ax.set_ylabel("Accuracy"); ax.set_title("5-fold stratified CV accuracy per model"); plt.xticks(rotation=20)
    fig.savefig(out / "05_cv_boxplot.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return table, skf


def tune(Xtr, Xte, ytr, yte, skf, out):
    svm = GridSearchCV(make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
                       {"svc__C": [.1, .5, 1, 2, 5, 10], "svc__gamma": ["scale", .01, .03, .1]}, cv=skf).fit(Xtr, ytr)
    lr = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                      {"logisticregression__C": np.logspace(-3, 2, 20)}, cv=skf).fit(Xtr, ytr)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, model) in zip(axs, [("Logistic Regression (tuned)", lr), ("SVM RBF (tuned)", svm)]):
        ConfusionMatrixDisplay.from_estimator(model, Xte, yte, display_labels=["Healthy", "Disease"],
                                              cmap="Purples", ax=ax, colorbar=False)
        ax.set_title(title)
    fig.savefig(out / "06_confusion_matrices.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=SEED)).fit(Xtr, ytr)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, model in [("LogReg", lr), ("SVM", svm), ("RandForest", rf)]:
        RocCurveDisplay.from_estimator(model, Xte, yte, name=name, ax=axs[0])
        PrecisionRecallDisplay.from_estimator(model, Xte, yte, name=name, ax=axs[1])
    axs[0].plot([0, 1], [0, 1], "k--"); axs[0].set_title("ROC curves"); axs[1].set_title("Precision-Recall curves")
    fig.savefig(out / "07_roc_pr_curves.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    best = lr if lr.score(Xte, yte) > svm.score(Xte, yte) else svm
    summary = {
        "svm": {"best_params": svm.best_params_, "cv_acc": svm.best_score_, "test_acc": svm.score(Xte, yte),
                "confusion_matrix": confusion_matrix(yte, svm.predict(Xte)).tolist()},
        "logreg": {"best_params": lr.best_params_, "cv_acc": lr.best_score_, "test_acc": lr.score(Xte, yte),
                   "confusion_matrix": confusion_matrix(yte, lr.predict(Xte)).tolist()},
        "best_tuned": "SVM" if best is svm else "LogisticRegression",
    }
    return best, summary


def explain(best, X, Xtr, Xte, ytr, yte, out):
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(Xtr, ytr)
    imp = pd.Series(rf.feature_importances_, X.columns).sort_values()
    perm = permutation_importance(best, Xte, yte, n_repeats=30, random_state=SEED)
    pim = pd.Series(perm.importances_mean, X.columns).sort_values()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    imp.plot.barh(ax=axs[0], color=GREEN); axs[0].set_title("Random Forest impurity importance")
    pim.plot.barh(ax=axs[1], color="#5b6abf"); axs[1].set_title("Permutation importance (tuned best model)")
    fig.savefig(out / "08_feature_importance.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    proba = best.predict_proba(Xte)[:, 1]
    th = np.arange(.1, .91, .05)
    rec = [recall_score(yte, proba >= t) for t in th]
    pre = [precision_score(yte, proba >= t, zero_division=0) for t in th]
    plt.figure(figsize=(7, 4.5)); plt.plot(th, rec, label="Recall"); plt.plot(th, pre, label="Precision")
    plt.axvline(.5, c="gray", ls="--"); plt.xlabel("Decision threshold"); plt.title("Precision / recall vs threshold")
    plt.legend(); plt.savefig(out / "09_threshold_analysis.png", dpi=130, bbox_inches="tight"); plt.close()
    return imp.round(3).to_dict(), pim.round(3).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart-disease.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    X, y, df = load(a.data)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns; class balance: {y.value_counts().to_dict()}")
    corr = eda(df, out)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    table, skf = compare(Xtr, Xte, ytr, yte, out)
    print(table.to_string(index=False))
    best, summary = tune(Xtr, Xte, ytr, yte, skf, out)
    rf_imp, perm_imp = explain(best, X, Xtr, Xte, ytr, yte, out)
    summary.update(split={"train": len(Xtr), "test": len(Xte)}, target_correlation=corr.round(3).to_dict(),
                   rf_importance=rf_imp, permutation_importance=perm_imp)
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("svm", "logreg", "best_tuned")}, indent=2, default=float))
    print(f"Figures and tables written to {out.resolve()}")


if __name__ == "__main__":
    main()
