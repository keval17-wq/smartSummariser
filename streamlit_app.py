import streamlit as st
import os
from novelty import scenario_4, scenario_5_augmented
from evaluation import evaluate, save_to_csv

# --- Constants ---
DATA_FILE = "data_for_run.txt"
SC4_FILE = "scenario_4_summary.txt"
SC5_FILE = "scenario_5_summary.txt"
SOURCE_FILE = "reviews_sample.txt"
BENCHMARK_FILE = "benchmark.txt"

st.set_page_config(page_title="Smart Summariser", layout="wide")
st.title("🧠 Smart Summariser")

user_input = st.text_area("Paste your review(s):", height=300)

if st.button("Run Scenarios"):
    if not user_input.strip():
        st.error("Please enter some text.")
    else:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write(user_input)

        with st.spinner("Running Scenario 4..."):
            scenario_4()

        with st.spinner("Running Scenario 5..."):
            scenario_5_augmented(include_sentiment=True)

        # Load summaries
        sc4_summary = open(SC4_FILE, encoding="utf-8").read()
        sc5_summary = open(SC5_FILE, encoding="utf-8").read()

        st.subheader("📘 Scenario 4 Summary")
        st.text(sc4_summary)
        st.download_button("Download Scene 4", data=sc4_summary, file_name="scenario4_summary.txt")

        st.subheader("📗 Scenario 5 Summary")
        st.text(sc5_summary)
        st.download_button("Download Scene 5", data=sc5_summary, file_name="scenario5_summary.txt")

        # Evaluation
        st.subheader("📊 Evaluation")
        ref = open(SOURCE_FILE, encoding="utf-8").read()
        bench = open(BENCHMARK_FILE, encoding="utf-8").read()

        results = [
            evaluate(ref, sc4_summary, "Scene 4 vs Source"),
            evaluate(ref, sc5_summary, "Scene 5 vs Source"),
            evaluate(bench, sc4_summary, "Scene 4 vs Benchmark"),
            evaluate(bench, sc5_summary, "Scene 5 vs Benchmark")
        ]

        save_to_csv(results, "evaluation_results.csv")

        import pandas as pd
        st.dataframe(pd.DataFrame(results).style.format(precision=4))
        st.download_button("Download Evaluation CSV",
                           data=open("evaluation_results.csv").read(),
                           file_name="evaluation_results.csv")
