import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Alzheimer AT(N) Simulator",
    layout="wide"
)

st.title("🧠 Alzheimer's Disease AT(N) Simulator")

# --------------------------------------------------
# Load data
# --------------------------------------------------

try:
    df_cross = pd.read_csv("data/synthetic_dataset.csv")
except:
    df_cross = None

try:
    df_long = pd.read_csv("data/longitudinal_atn.csv")
except:
    df_long = None

# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "AT(N) Simulator",
        "PCA Brain Space",
        "Longitudinal",
        "Survival"
    ]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.header("Dataset Overview")

    if df_cross is not None:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Subjects", len(df_cross))

        with col2:
            st.metric(
                "Mean Age",
                f"{df_cross['age'].mean():.1f}"
            )

        with col3:
            st.metric(
                "Mean MMSE",
                f"{df_cross['mmse'].mean():.1f}"
            )

        st.dataframe(df_cross.head())

    else:

        st.warning(
            "synthetic_dataset.csv not found"
        )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.header("AT(N) Disease Simulator")

    age = st.slider(
        "Age",
        50,
        95,
        70
    )

    amyloid = st.slider(
        "Amyloid",
        0.0,
        10.0,
        2.0
    )

    tau = st.slider(
        "Tau",
        0.0,
        10.0,
        1.0
    )

    neurodegeneration = (
        0.5 * tau +
        0.2 * amyloid
    )

    mmse = (
        30
        - 0.15 * (age - 60)
        - 1.5 * tau
        - 2.5 * neurodegeneration
    )

    mmse = np.clip(mmse, 0, 30)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Neurodegeneration",
            f"{neurodegeneration:.2f}"
        )

    with col2:
        st.metric(
            "Predicted MMSE",
            f"{mmse:.1f}/30"
        )

    st.markdown(
        """
        **Disease cascade**

        Amyloid → Tau → Neurodegeneration → Cognitive decline
        """
    )

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.header("PCA Brain Space")

    if df_cross is not None:

        features = [
            "hippocampus",
            "ventricles",
            "cortical_thickness",
            "mmse"
        ]

        X = df_cross[features]

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)

        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots()

        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=df_cross["diagnosis"],
            alpha=0.7
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Brain PCA Space")

        st.pyplot(fig)

    else:

        st.warning("Dataset unavailable")

# ==================================================
# TAB 4
# ==================================================

with tab4:

    st.header("Longitudinal Progression")

    if df_long is not None:

        progression = (
            df_long
            .groupby("timepoint")
            [["amyloid",
              "tau",
              "neurodegeneration",
              "mmse"]]
            .mean()
        )

        fig, ax = plt.subplots()

        ax.plot(
            progression.index,
            progression["amyloid"],
            label="Amyloid"
        )

        ax.plot(
            progression.index,
            progression["tau"],
            label="Tau"
        )

        ax.plot(
            progression.index,
            progression["neurodegeneration"],
            label="Neurodegeneration"
        )

        ax.legend()

        ax.set_title(
            "AT(N) Progression"
        )

        st.pyplot(fig)

        fig2, ax2 = plt.subplots()

        ax2.plot(
            progression.index,
            progression["mmse"]
        )

        ax2.set_title(
            "MMSE Evolution"
        )

        st.pyplot(fig2)

    else:

        st.warning(
            "longitudinal_atn.csv not found"
        )

# ==================================================
# TAB 5
# ==================================================

with tab5:

    st.header("Survival / Risk Exploration")

    if df_long is not None:

        risk = (
            0.4 * df_long["tau"]
            + 0.6 * df_long["neurodegeneration"]
        )

        fig, ax = plt.subplots()

        ax.hist(
            risk,
            bins=30
        )

        ax.set_title(
            "Conversion Risk Distribution"
        )

        st.pyplot(fig)

        st.write(
            "Higher values correspond to higher simulated conversion risk."
        )

    else:

        st.warning(
            "Longitudinal dataset unavailable."
        )
