"""
Student Academic Performance Prediction System - Streamlit dashboard.

Run:  streamlit run app.py
(Train first with:  python -m src.train_models)
"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (AT_RISK_CATEGORIES, CATEGORICAL_FEATURES, CATEGORY_ORDER, DATA_PATH,
                        EXTRA_OPTIONS, FEATURES, GENDER_OPTIONS, MODEL_DIR, NUMERIC_FEATURES,
                        REPORT_DIR, TARGET)
from src.preprocessing import IQRClipper  # noqa: F401  (needed to unpickle the pipeline)
from src.utils import categorize_marks, categorize_series, support_recommendations

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

COLORS = {"Excellent": "#2E9E5B", "Very Good": "#7CB342", "Good": "#F9A825",
          "Average": "#FB8C00", "Needs Improvement": "#D32F2F"}


# --------------------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    pipe = joblib.load(MODEL_DIR / "best_pipeline.joblib")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return pipe, meta


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Performance_Category"] = categorize_series(df[TARGET])
    return df


def artefacts_ready() -> bool:
    return all((MODEL_DIR / f).exists() for f in ["best_pipeline.joblib", "metadata.json"]) and DATA_PATH.exists()


if not artefacts_ready():
    st.error("Model files not found. Run `python -m src.generate_dataset` and "
             "`python -m src.train_models` first (see README.md).")
    st.stop()

pipe, meta = load_model()
data = load_data()

# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict a Student", "Batch Prediction", "Data Explorer",
                                  "Model Comparison", "Factor Analysis", "About"])
st.sidebar.info(f"**Active model:** {meta['best_model']}\n\n**Data:** synthetic (demonstration only)")


def gauge(value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": " / 100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f3b63"},
               "steps": [{"range": [0, 40], "color": COLORS["Needs Improvement"]},
                         {"range": [40, 55], "color": COLORS["Average"]},
                         {"range": [55, 70], "color": COLORS["Good"]},
                         {"range": [70, 85], "color": COLORS["Very Good"]},
                         {"range": [85, 100], "color": COLORS["Excellent"]}]}))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=30, r=30))
    return fig


def predict_df(frame: pd.DataFrame) -> np.ndarray:
    return np.clip(pipe.predict(frame[FEATURES]), 0, 100)


# --------------------------------------------------------------------------- pages
if page == "Home":
    st.title("🎓 Student Academic Performance Prediction System")
    st.caption("Machine-learning dashboard to predict final exam marks and flag students who may need support.")
    st.warning("The training data is **synthetic** (computer-generated), not real student records. "
               "Predictions demonstrate the method and must not be used for real decisions about students.")
    tm = meta["test_metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students in dataset", f"{len(data):,}")
    c2.metric("Best model", meta["best_model"])
    c3.metric("Test R²", f"{tm['R2']:.3f}")
    c4.metric("Test RMSE", f"{tm['RMSE']:.2f} marks")
    c5.metric("Category accuracy", f"{meta['category_accuracy']:.1%}")

    left, right = st.columns(2)
    counts = data["Performance_Category"].value_counts().reindex(CATEGORY_ORDER).fillna(0).reset_index()
    counts.columns = ["Category", "Students"]
    left.plotly_chart(px.bar(counts, x="Category", y="Students", color="Category",
                             color_discrete_map=COLORS, title="Students per performance category"),
                      use_container_width=True)
    right.plotly_chart(px.histogram(data, x=TARGET, nbins=30, title="Final marks distribution",
                                    color_discrete_sequence=["#4C78A8"]), use_container_width=True)
    st.subheader("Performance categories")
    st.table(pd.DataFrame({
        "Category": ["Excellent", "Very Good", "Good", "Average", "Needs Improvement"],
        "Marks": ["85 – 100", "70 – 84.99", "55 – 69.99", "40 – 54.99", "Below 40"]}))

elif page == "Predict a Student":
    st.title("🔮 Predict a Student's Final Marks")
    rng = meta["feature_ranges"]
    with st.form("single"):
        a, b, c = st.columns(3)
        age = a.number_input("Age", 16, 40, 20)
        gender = a.selectbox("Gender", GENDER_OPTIONS)
        extra = a.selectbox("Extracurricular participation", EXTRA_OPTIONS, index=1)
        att = b.slider("Attendance (%)", 0.0, 100.0, 80.0, 0.5)
        study = b.slider("Study hours per day", 0.0, 12.0, 3.0, 0.1)
        sleep = b.slider("Sleep hours per night", 3.0, 12.0, 7.0, 0.1)
        prev = c.slider("Previous semester marks", 0.0, 100.0, 65.0, 0.5)
        internal = c.slider("Internal exam marks", 0.0, 100.0, 62.0, 0.5)
        assign = c.slider("Assignment score", 0.0, 100.0, 68.0, 0.5)
        done = a.slider("Assignments completed (of 10)", 0, 10, 8)
        part = b.slider("Class participation (0-10)", 0.0, 10.0, 5.5, 0.1)
        go_btn = st.form_submit_button("Predict", type="primary")

    if go_btn:
        row = {"Age": age, "Gender": gender, "Attendance_Percentage": att, "Study_Hours_Per_Day": study,
               "Previous_Semester_Marks": prev, "Assignment_Score": assign, "Internal_Exam_Marks": internal,
               "Assignments_Completed": done, "Class_Participation_Score": part, "Sleep_Hours": sleep,
               "Extracurricular_Participation": extra}
        pred = float(predict_df(pd.DataFrame([row]))[0])
        cat = categorize_marks(pred)
        m1, m2 = st.columns([1, 1])
        m1.plotly_chart(gauge(pred), use_container_width=True)
        with m2:
            st.metric("Predicted final marks", f"{pred:.1f}")
            st.markdown(f"### Category: <span style='color:{COLORS[cat]}'>{cat}</span>", unsafe_allow_html=True)
            rmse = meta["test_metrics"]["RMSE"]
            st.caption(f"Typical model error is about ±{rmse:.1f} marks, so the likely range is "
                       f"{max(0, pred - 1.96 * rmse):.0f} – {min(100, pred + 1.96 * rmse):.0f} (approx. 95%).")
            if cat in AT_RISK_CATEGORIES:
                st.error("This student may benefit from additional academic support.")
            else:
                st.success("This student is on track.")
        st.subheader("Suggestions")
        for tip in support_recommendations(row, pred):
            st.markdown(f"- {tip}")

elif page == "Batch Prediction":
    st.title("📂 Batch Prediction (CSV upload)")
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in FEATURES))
    template = data[FEATURES].head(5).to_csv(index=False).encode()
    st.download_button("Download a sample template", template, "template.csv", "text/csv")
    up = st.file_uploader("Choose a CSV file", type="csv")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df_in.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df_in["Predicted_Final_Marks"] = predict_df(df_in).round(1)
                df_in["Predicted_Category"] = categorize_series(df_in["Predicted_Final_Marks"])
                df_in["Needs_Support"] = df_in["Predicted_Category"].isin(AT_RISK_CATEGORIES)
                st.success(f"Predicted {len(df_in)} students; {int(df_in['Needs_Support'].sum())} flagged for support.")
                st.dataframe(df_in, use_container_width=True)
                fig = px.histogram(df_in, x="Predicted_Category", color="Predicted_Category",
                                   category_orders={"Predicted_Category": CATEGORY_ORDER}, color_discrete_map=COLORS)
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Download results", df_in.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
        except Exception as e:  # show a friendly message for malformed files
            st.error(f"Could not process the file: {e}")

elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.caption("Synthetic dataset – see data/DATA_DICTIONARY.md")
    f1, f2 = st.columns(2)
    cats = f1.multiselect("Performance category", CATEGORY_ORDER, default=CATEGORY_ORDER)
    genders = f2.multiselect("Gender", GENDER_OPTIONS, default=GENDER_OPTIONS)
    view = data[data["Performance_Category"].isin(cats) & (data["Gender"].isin(genders) | data["Gender"].isna())]
    st.write(f"{len(view)} students shown")
    st.dataframe(view, use_container_width=True, height=250)

    t1, t2, t3 = st.tabs(["Scatter", "Correlation", "Distributions"])
    with t1:
        x = st.selectbox("X axis", NUMERIC_FEATURES, index=1)
        fig = px.scatter(view, x=x, y=TARGET, color="Performance_Category", color_discrete_map=COLORS, opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        corr = view[NUMERIC_FEATURES + [TARGET]].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  aspect="auto"), use_container_width=True)
    with t3:
        col = st.selectbox("Feature", NUMERIC_FEATURES + [TARGET], key="dist")
        st.plotly_chart(px.box(view, x="Performance_Category", y=col, color="Performance_Category",
                               category_orders={"Performance_Category": CATEGORY_ORDER},
                               color_discrete_map=COLORS), use_container_width=True)

elif page == "Model Comparison":
    st.title("⚖️ Model Comparison")
    res = pd.read_csv(REPORT_DIR / "model_comparison.csv")
    st.success(f"Selected model: **{meta['best_model']}** – {meta['selection_rule']}.")
    show = res.drop(columns=["Best_Params"]).round(3)
    st.dataframe(show, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(res, x="Model", y="Test_RMSE", color="Model", title="Test RMSE (lower = better)"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(res, x="Model", y="Test_R2", color="Model", title="Test R² (higher = better)",
                           range_y=[0, 1]), use_container_width=True)
    st.plotly_chart(px.bar(res.melt(id_vars="Model", value_vars=["Test_MAE", "Test_RMSE"]), x="Model", y="value",
                           color="variable", barmode="group", title="MAE vs RMSE"), use_container_width=True)
    with st.expander("Best hyper-parameters per model"):
        st.dataframe(res[["Model", "Best_Params"]], use_container_width=True)
    st.info("Models are chosen using cross-validation on the training set; the test set is used only for "
            "the final report. On this synthetic data the relationships are largely linear, so a simple "
            "model can match or beat ensembles. Real data may rank them differently.")
    tp = pd.read_csv(REPORT_DIR / "test_predictions.csv")
    st.plotly_chart(px.scatter(tp, x="Actual", y="Predicted", opacity=0.6,
                               title=f"Actual vs predicted on test set ({meta['best_model']})")
                    .add_shape(type="line", x0=20, y0=20, x1=100, y1=100, line=dict(dash="dash")),
                    use_container_width=True)

elif page == "Factor Analysis":
    st.title("🔍 Factors Affecting Performance")
    imp = pd.read_csv(REPORT_DIR / "feature_importance.csv").sort_values("Importance")
    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", error_x="Std",
                           title="Permutation importance (drop in R² when a feature is shuffled)"),
                    use_container_width=True)
    corr = data[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values().reset_index()
    corr.columns = ["Feature", "Pearson_r"]
    st.plotly_chart(px.bar(corr, x="Pearson_r", y="Feature", orientation="h",
                           title="Correlation with final marks"), use_container_width=True)
    st.caption("Importance and correlation describe associations in the (synthetic) data, not proven causes.")
    st.subheader("What-if analysis")
    base = data[NUMERIC_FEATURES].median().to_dict()
    base.update({"Gender": "Male", "Extracurricular_Participation": "Occasional"})
    feat = st.selectbox("Vary this feature", ["Attendance_Percentage", "Study_Hours_Per_Day", "Internal_Exam_Marks",
                                              "Previous_Semester_Marks", "Assignment_Score", "Sleep_Hours",
                                              "Class_Participation_Score"])
    lo, hi, _ = meta["feature_ranges"][feat]
    grid = np.linspace(lo, hi, 40)
    frame = pd.DataFrame([{**base, feat: v} for v in grid])
    st.plotly_chart(px.line(x=grid, y=predict_df(frame), labels={"x": feat, "y": "Predicted marks"},
                            title=f"Predicted marks vs {feat} (other inputs at median)"), use_container_width=True)

else:
    st.title("ℹ️ About")
    st.markdown(f"""
**Student Academic Performance Prediction System Using Machine Learning** – B.Tech mini-project.

* **Task:** regression (final marks) + rule-based classification into five categories.
* **Models compared:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
* **Selection:** {meta['selection_rule']}.
* **Data:** *synthetic*, generated by `src/generate_dataset.py` – **not real student data**.
* **Trained on:** {meta['trained_on']} (train {meta['n_train']} / test {meta['n_test']}, seed {meta['random_state']}).

**Limitations:** results on synthetic data do not guarantee performance on real students. Predictions
are an aid for early support, not a basis for grading or disciplinary decisions.
""")
