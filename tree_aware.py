import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- Load assets ----------------
rf = joblib.load("rf_cu_classifier.pkl")
features = joblib.load("rf_feature_order.pkl")
df = pd.read_csv("dataset.csv")

# ---------------- Helper functions ----------------
from sklearn.tree import _tree

def get_leaf_paths_classifier(tree, feature_names):
    tree_ = tree.tree_
    paths = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feat = feature_names[tree_.feature[node]]
            thresh = tree_.threshold[node]
            recurse(tree_.children_left[node],
                    conditions + [(feat, "<=", thresh)])
            recurse(tree_.children_right[node],
                    conditions + [(feat, ">", thresh)])
        else:
            counts = tree_.value[node][0]
            prob_1 = counts[1] / counts.sum()
            paths.append({"conditions": conditions, "prob_1": prob_1})

    recurse(0, [])
    return paths

def get_candidate_leaves_classifier(rf, feature_names):
    candidates = []
    for est in rf.estimators_:
        for p in get_leaf_paths_classifier(est, feature_names):
            if p["prob_1"] >= 0.5:
                candidates.append(p)
    return candidates

def l1_distance(a, b):
    return np.sum(np.abs(a - b))

def compute_bounds(df, features):
    return {f: {"min": df[f].min(), "max": df[f].max()} for f in features}

bounds = compute_bounds(df, features)

def apply_leaf_conditions_bounded(x, leaf_conditions, fixed_features):
    x_cf = x.copy()
    for feat, op, thresh in leaf_conditions:
        if feat in fixed_features:
            continue
        idx = features.index(feat)
        if op == "<=" and x_cf[idx] > thresh:
            x_cf[idx] = thresh
        elif op == ">" and x_cf[idx] <= thresh:
            x_cf[idx] = thresh + 1e-4

        x_cf[idx] = np.clip(
            x_cf[idx],
            bounds[feat]["min"],
            bounds[feat]["max"]
        )
    return x_cf

def tree_aware_cf(x_orig, fixed_features):
    best_cf, best_dist = None, np.inf
    for c in get_candidate_leaves_classifier(rf, features):
        x_cf = apply_leaf_conditions_bounded(
            x_orig, c["conditions"], fixed_features
        )
        if rf.predict([x_cf])[0] == 1:
            dist = l1_distance(x_orig, x_cf)
            if dist < best_dist:
                best_cf, best_dist = x_cf, dist
    return best_cf

# ---------------- Streamlit UI ----------------
st.title("Tree-Aware Counterfactual Explorer (Cu Slag)")

row_id = st.slider(
    "Select data point index",
    0, len(df) - 1, 0
)

fixed_features = st.multiselect(
    "Fix these parameters (hold constant):",
    features
)

x_input = df.loc[row_id, features].values

st.subheader("Original Prediction")
st.write("Class:", rf.predict([x_input])[0])

if st.button("Generate Counterfactual"):
    x_cf = tree_aware_cf(x_input, fixed_features)

    if x_cf is None:
        st.error("No feasible counterfactual found.")
    else:
        st.success("Counterfactual found (bounded & feasible)")
        st.write("New Class:", rf.predict([x_cf])[0])

        impact_df = counterfactual_impact_table(
            x_input, x_cf, features
        )
        st.dataframe(impact_df)
