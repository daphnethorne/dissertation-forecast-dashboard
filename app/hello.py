"""Sanity check that Streamlit launches correctly in the dashboard venv."""

import streamlit as st
import pandas as pd
import numpy as np

st.title("Dashboard foundation check")

st.write("If you can see this, the Streamlit foundation is working.")

st.subheader("Environment versions")
st.write({
    "streamlit": st.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
})

st.subheader("Tiny data sample")
sample = pd.DataFrame({
    "year": [2022, 2023, 2024],
    "average_payment_days": [38, 41, 39],
})
st.dataframe(sample)
st.line_chart(sample.set_index("year"))
