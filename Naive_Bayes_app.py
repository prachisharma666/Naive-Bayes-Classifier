# Naive Bayes Classifier
# input(streamlit)-dataset,target and features,train,test split
# output-testing accuracy,confusion matrix

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def Naive_bayes_streamlit():

    st.title("Naive Bayes Classifier")

    problem_type = st.selectbox(
        "Select Problem Type",
        ["Classification"]
    )

    model_type = st.selectbox(
        "Select Model",
        ["Select Model", "Gaussian Naive Bayes"]
    )

    file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"]
    )

    if file is not None:

        df = pd.read_csv(file)

        st.write("Dataset Preview")
        st.write(df.head())

        target_col = st.selectbox(
            "Select Target Column",
            df.columns
        )

        target_data = df[target_col]

        n_classes = target_data.nunique()

    
        if n_classes > 20:
            st.error(
                "Target column has too many unique values."
            )
            st.stop()

        if target_data.dtype in ['int64','float64'] and n_classes > 10:
            st.error(
                "Numeric target detected."
            )
            st.stop()
        rs = st.number_input(
            "Random State",
            value=42
        )

        ts = st.slider(
            "Test Size",
            0.1,
            0.5,
            0.2
        )

        x_cols = st.multiselect(
            "Select Features (X)",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col]
        )


        if st.button("Run Naive Bayes"):

            X = df[x_cols]
            y = df[target_col]

            GNB = GaussianNB()

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                random_state=int(rs),
                test_size=ts
            )

            GNB.fit(X_train, y_train)

            y_pred_train = GNB.predict(X_train)
            y_pred_test = GNB.predict(X_test)

            train_acc = accuracy_score(
                y_train,
                y_pred_train
            )

            test_acc = accuracy_score(
                y_test,
                y_pred_test
            )

            cm_train = confusion_matrix(
                y_train,
                y_pred_train
            )

            cm_test = confusion_matrix(
                y_test,
                y_pred_test
            )

            st.subheader("Results")

            st.write("Training Accuracy:", train_acc)

            st.write("Testing Accuracy:", test_acc)

            st.write("Training Confusion Matrix")
            st.write(cm_train)

            st.write("Testing Confusion Matrix")
            st.write(cm_test)


Naive_bayes_streamlit()