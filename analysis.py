import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

# ============================================================
# CONFIG
# ============================================================
DATASETS = [
    "hard_01_pure_noise.xlsx",
]
SHEET_NAME = "Section_Averages"
TRUTH_SHEET = "Teacher_Truth"      # optional sheet: Teacher | True_Strictness
N_REPEATS  = 10
# ============================================================

COLUMN_MAP = {
    "CIE 1 Marks":      "Avg_Internal_1_30",
    "CIE 2 Marks":      "Avg_Internal_2_30",
    "Assignment Marks": "Avg_Assignment_20",
    "Attendance":       "Avg_Attendance_Percent",
}
LABELS   = ["Strict", "Moderate", "Lenient"]
KEY_COLS = ["Academic_Year", "Semester", "Section", "Subject", "Teacher"]
MARKS    = ["Avg_Internal_1_30", "Avg_Internal_2_30", "Avg_Assignment_20"]

# ── TWO FEATURE SECTIONS (now confounder-ADJUSTED) ───────────
FEATURES_MARKS = [
    "Adj_Total", "Adj_Internal", "Adj_Assignment",
    "Adj_Gain", "Adj_Assign_vs_Internal", "Resid_Std",
]
FEATURES_MARKS_ATT = FEATURES_MARKS + ["Avg_Attendance"]

SECTIONS = {
    "A: Marks only":         FEATURES_MARKS,
    "B: Marks + Attendance": FEATURES_MARKS_ATT,
}


# ═══════════════════════════════════════════════════════════════
# 1. TEACHER LOADER
#    KEY IMPROVEMENT: instead of raw averages (which mix teacher
#    strictness with subject difficulty / section strength), fit an
#    additive model  mark ~ Teacher + Subject + Section + Year + Sem
#    and use the TEACHER coefficient. No labels are used here.
# ═══════════════════════════════════════════════════════════════
def teacher_effects(df, target):
    X = pd.get_dummies(
        df[["Teacher", "Subject", "Section", "Academic_Year", "Semester"]].astype(str),
        dtype=float)
    m = Ridge(alpha=0.5).fit(X, df[target] - df[target].mean())
    coef = pd.Series(m.coef_, index=X.columns)
    keep = [c for c in X.columns if c.startswith("Teacher_")]
    return coef[keep].rename(lambda s: s[len("Teacher_"):])


def load_teachers(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    df = df.rename(columns=COLUMN_MAP)

    if "Avg_Total_80" not in df:
        df["Avg_Total_80"] = df[MARKS].sum(axis=1)
    df = df.drop_duplicates(subset=KEY_COLS).dropna()
    df["Avg_Internal_Sum"] = df["Avg_Internal_1_30"] + df["Avg_Internal_2_30"]

    cnt  = df.groupby("Teacher")["Subject"].count()
    keep = cnt[cnt >= 3].index                       # Workload >= 3
    df   = df[df["Teacher"].isin(keep)]

    t = pd.DataFrame({
        "Adj_Total":      teacher_effects(df, "Avg_Total_80"),
        "Adj_Internal":   teacher_effects(df, "Avg_Internal_Sum"),
        "Adj_Assignment": teacher_effects(df, "Avg_Assignment_20"),
        "Adj_C1":         teacher_effects(df, "Avg_Internal_1_30"),
        "Adj_C2":         teacher_effects(df, "Avg_Internal_2_30"),
    })
    t["Adj_Gain"]            = t["Adj_C2"] - t["Adj_C1"]
    t["Adj_Assign_vs_Internal"] = t["Adj_Assignment"] / 20 - t["Adj_Internal"] / 60

    # consistency of a teacher after removing subject/section effects
    Xs  = pd.get_dummies(df[["Subject", "Section"]].astype(str), dtype=float)
    res = df["Avg_Total_80"] - Ridge(alpha=0.5).fit(Xs, df["Avg_Total_80"]).predict(Xs)
    t["Resid_Std"]      = res.groupby(df["Teacher"]).std().fillna(0)
    t["Avg_Attendance"] = df.groupby("Teacher")["Avg_Attendance_Percent"].mean()
    t["Workload_Rows"]  = cnt[keep]
    t = t.loc[keep]

    # ── LABEL ───────────────────────────────────────────────
    truth_ok = False
    try:
        truth = pd.read_excel(file_name, sheet_name=TRUTH_SHEET) \
                  .set_index("Teacher")["True_Strictness"]
        t["Strictness_Label"] = truth.reindex(t.index).values
        truth_ok = t["Strictness_Label"].notna().all()
    except Exception:
        pass
    if not truth_ok:
        # fallback: tertile of ADJUSTED performance (high effect = lenient)
        t["Strictness_Label"] = pd.qcut(t["Adj_Total"], 3, labels=LABELS[::-1]).astype(str)
        t.attrs["label_source"] = "tertile of adjusted performance (LEAKY: label derived from features)"
    else:
        t.attrs["label_source"] = "independent Teacher_Truth sheet"
    return t


# ═══════════════════════════════════════════════════════════════
# 2. MODEL FACTORY
# ═══════════════════════════════════════════════════════════════
def get_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=2,
            random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }


# ═══════════════════════════════════════════════════════════════
# 3. EVALUATE ONE SECTION
# ═══════════════════════════════════════════════════════════════
def evaluate_section(t, features):
    X, y = t[features], t["Strictness_Label"]
    metrics, matrices = [], {}

    for name, model in get_models().items():
        m, cm_sum = [], np.zeros((3, 3))
        for r in range(N_REPEATS):
            cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=r)
            pred = cross_val_predict(model, X, y, cv=cv)
            m.append([accuracy_score(y, pred),
                      precision_score(y, pred, average="weighted", zero_division=0),
                      recall_score(y, pred,    average="weighted", zero_division=0),
                      f1_score(y, pred,        average="weighted", zero_division=0)])
            cm_sum += confusion_matrix(y, pred, labels=LABELS)
        metrics.append([name] + list(np.mean(m, axis=0)))
        matrices[name] = pd.DataFrame(np.round(cm_sum / N_REPEATS).astype(int),
                                      index=LABELS, columns=LABELS)

    res = pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision",
                                         "Recall", "F1"])
    return res, matrices


# ═══════════════════════════════════════════════════════════════
# 4. TEACHER LISTING
# ═══════════════════════════════════════════════════════════════
def print_teacher_groups(t):
    print("\n" + "-" * 80)
    print("TEACHERS CLASSIFIED BY STRICTNESS")
    print("-" * 80)
    for lvl in LABELS:
        names = t.index[t["Strictness_Label"] == lvl].tolist()
        print(f"\n{lvl}  ({len(names)} teachers):")
        for n in names:
            print(f"   • {n}")


# ═══════════════════════════════════════════════════════════════
# 5. RUN ONE DATASET ACROSS BOTH SECTIONS
# ═══════════════════════════════════════════════════════════════
def run(file_name, sheet_name):
    print("\n" + "=" * 80)
    print(f"DATASET: {file_name}")
    print("=" * 80)

    if not os.path.exists(file_name):
        print(f"File not found: {file_name}. Skipping.")
        return None
    try:
        t = load_teachers(file_name, sheet_name)
    except Exception as e:
        print(f"ERROR loading '{file_name}': {e}")
        return None

    print(f"({len(t)} teachers after filtering Workload >= 3)")
    print(f"Label source: {t.attrs['label_source']}")
    print_teacher_groups(t)

    section_best = {}
    for sec_name, feats in SECTIONS.items():
        print("\n" + "=" * 80)
        print(f"SECTION {sec_name}")
        print("=" * 80)
        res, matrices = evaluate_section(t, feats)
        print(res.to_string(index=False, float_format="%.4f"))

        best_row = res.loc[res["F1"].idxmax()]
        print(f"\n🏆 BEST MODEL for Section {sec_name}: "
              f"{best_row['Model']}  (F1 = {best_row['F1']:.4f})")
        print(f"   Accuracy  = {best_row['Accuracy']:.4f}")
        print(f"   Precision = {best_row['Precision']:.4f}")
        print(f"   Recall    = {best_row['Recall']:.4f}")
        print(f"   F1-Score  = {best_row['F1']:.4f}")
        print(f"\nConfusion Matrix ({best_row['Model']}):")
        print(matrices[best_row["Model"]].to_string())

        section_best[sec_name] = dict(
            model=best_row["Model"], accuracy=best_row["Accuracy"],
            precision=best_row["Precision"], recall=best_row["Recall"],
            f1=best_row["F1"])

    winner_sec = max(section_best, key=lambda k: section_best[k]["f1"])
    win = section_best[winner_sec]
    print("\n" + "=" * 80)
    print("★ OVERALL BEST FOR THIS DATASET ★")
    print("=" * 80)
    print(f"Section   : {winner_sec}")
    print(f"Model     : {win['model']}")
    print(f"Accuracy  : {win['accuracy']:.4f}")
    print(f"Precision : {win['precision']:.4f}")
    print(f"Recall    : {win['recall']:.4f}")
    print(f"F1-Score  : {win['f1']:.4f}")

    return {
        "Dataset": file_name, "Best Section": winner_sec, "Best Model": win["model"],
        "Accuracy": win["accuracy"], "Precision": win["precision"],
        "Recall": win["recall"], "F1 Score": win["f1"],
        "Marks-Only F1": section_best["A: Marks only"]["f1"],
        "Marks+Att F1": section_best["B: Marks + Attendance"]["f1"],
        "Label Source": t.attrs["label_source"][:28],
    }


# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for file in DATASETS:
        run(file, SHEET_NAME)