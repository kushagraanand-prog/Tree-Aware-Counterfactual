import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
from sklearn.tree import _tree

st.set_page_config(
    page_title="CL Slag Cu — Tree-Aware Counterfactuals",
    layout="wide"
)

# =====================================================
# Load artifacts
# =====================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("cu_rf_model.pkl")
    X_train = np.load("X_train.npy")
    feature_order = joblib.load("feature_order.pkl")
    return model, X_train, feature_order

try:
    model, X_train, FEATURE_ORDER = load_artifacts()
except Exception as e:
    st.error("❌ Failed to load artifacts. Check file names.")
    st.stop()

n_features = len(FEATURE_ORDER)
train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)

# =====================================================
# Tree-Aware CF helpers
# =====================================================
def get_tree_paths(tree, feature_names):
    tree_ = tree.tree_
    paths = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            fname = feature_names[tree_.feature[node]]
            thr = tree_.threshold[node]
            recurse(tree_.children_left[node],
                    conditions + [(fname, "<=", thr)])
            recurse(tree_.children_right[node],
                    conditions + [(fname, ">", thr)])
        else:
            paths.append({
                "conditions": conditions,
                "value": tree_.value[node][0]
            })

    recurse(0, [])
    return paths


def path_to_bounds(path_conditions):
    bounds = {
        f: [float(train_min[i]), float(train_max[i])]
        for i, f in enumerate(FEATURE_ORDER)
    }

    for feat, op, thr in path_conditions:
        if op == "<=":
            bounds[feat][1] = min(bounds[feat][1], thr)
        else:
            bounds[feat][0] = max(bounds[feat][0], thr)

    return bounds


def sample_candidate(bounds, input_map, locked_features):
    candidate = {}
    for f, (lb, ub) in bounds.items():
        if lb > ub:
            return None
        if f in locked_features:
            candidate[f] = input_map[f]
        else:
            candidate[f] = np.random.uniform(lb, ub)
    return candidate


def tree_aware_counterfactuals(
    model,
    input_vector,
    input_map,
    desired_class,
    locked_features,
    desired_prob=0.9,
    max_paths=40,
    samples_per_path=40,
    max_return=3
):
    forest = model.estimators_
    valid_paths = []

    for tree in forest:
        for p in get_tree_paths(tree, FEATURE_ORDER):
            if np.argmax(p["value"]) == desired_class:
                valid_paths.append(p)

    random.shuffle(valid_paths)
    results = []

    for path in valid_paths[:max_paths]:
        bounds = path_to_bounds(path["conditions"])

        for _ in range(samples_per_path):
            cand_map = sample_candidate(bounds, input_map, locked_features)
            if cand_map is None:
                continue

            cand_vec = np.array([[cand_map[f] for f in FEATURE_ORDER]])
            prob = model.predict_proba(cand_vec)[0]

            if prob[desired_class] >= desired_prob:
                dist = np.linalg.norm(cand_vec - input_vector)
                results.append({
                    "candidate": cand_vec.flatten(),
                    "prob": prob,
                    "dist": dist
                })

    results = sorted(results, key=lambda x: x["dist"])
    return results[:max_return]

# =====================================================
# UI — Input
# =====================================================
st.title("CL Slag Cu — Tree-Aware Counterfactuals (Random Forest)")

def mean_default(feat):
    idx = FEATURE_ORDER.index(feat)
    return float(X_train[:, idx].mean())

st.subheader("Input parameters")

input_map = {}
cols = st.columns(3)

for i, feat in enumerate(FEATURE_ORDER):
    with cols[i % 3]:
        input_map[feat] = st.number_input(
            feat,
            value=mean_default(feat)
        )

input_vector = np.array([[input_map[f] for f in FEATURE_ORDER]])

# =====================================================
# Prediction
# =====================================================
st.markdown("---")
pred_class = int(model.predict(input_vector)[0])
pred_prob = model.predict_proba(input_vector)[0]

if pred_class == 1:
    st.success(f"Prediction: **0.70–0.75 Cu%**  |  Prob = {pred_prob[1]:.3f}")
else:
    st.error(f"Prediction: **0.80–0.85 Cu%**  |  Prob = {pred_prob[0]:.3f}")

# =====================================================
# Counterfactual Controls
# =====================================================
st.markdown("---")
st.subheader("Tree-Aware Counterfactual Settings")

desired_choice = st.radio(
    "Target class",
    ["Same as current", "0.70–0.75 Cu%", "0.80–0.85 Cu%"]
)

if desired_choice == "Same as current":
    desired_class = pred_class
elif desired_choice == "0.70–0.75 Cu%":
    desired_class = 1
else:
    desired_class = 0

desired_prob = st.slider(
    "Required probability",
    0.5, 0.99, 0.90, 0.01
)

locked_features = st.multiselect(
    "Lock features (fixed values)",
    FEATURE_ORDER
)

show_only_changes = st.checkbox(
    "Show only changed features",
    value=True
)

# =====================================================
# Run Tree-Aware CF
# =====================================================
if st.button("Find Tree-Aware Counterfactuals"):
    with st.spinner("Searching tree-consistent counterfactuals..."):
        cfs = tree_aware_counterfactuals(
            model=model,
            input_vector=input_vector,
            input_map=input_map,
            desired_class=desired_class,
            locked_features=locked_features,
            desired_prob=desired_prob
        )

    if len(cfs) == 0:
        st.warning("No valid counterfactual found.")
    else:
        for i, cf in enumerate(cfs):
            st.markdown(f"### Counterfactual #{i+1}")

            df = pd.DataFrame({
                "feature": FEATURE_ORDER,
                "original": input_vector.flatten(),
                "counterfactual": cf["candidate"],
                "delta": cf["candidate"] - input_vector.flatten()
            })

            if show_only_changes:
                df = df[df["delta"].abs() > 1e-8]

            st.dataframe(
                df.style.format({
                    "original": "{:.4f}",
                    "counterfactual": "{:.4f}",
                    "delta": "{:.4f}"
                })
            )

            st.write(
                f"Distance: **{cf['dist']:.4f}**  |  "
                f"Probabilities: {np.array2string(cf['prob'], precision=4)}"
            )

        st.success("Tree-aware counterfactuals generated ✅")
