import re
import csv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# --------- Core Text Comparison Tools ---------
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def compute_cosine_similarity(text1, text2):
    texts = [clean_text(text1), clean_text(text2)]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

def compute_rouge_scores(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(clean_text(reference), clean_text(summary))
    return {
        "ROUGE-1 F1": scores['rouge1'].fmeasure,
        "ROUGE-2 F1": scores['rouge2'].fmeasure,
        "ROUGE-L F1": scores['rougeL'].fmeasure
    }

# --------- File Loader ---------
def load_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return ""

# --------- Evaluation Wrapper ---------
def evaluate(reference, summary, label=""):
    print(f"\n=== {label} ===")
    cosine = compute_cosine_similarity(reference, summary)
    rouge = compute_rouge_scores(reference, summary)
    for k, v in rouge.items():
        print(f"{k}: {v:.4f}")
    print(f"Cosine Similarity: {cosine:.4f}")
    return {
        "Label": label,
        "Cosine Similarity": cosine,
        **rouge
    }

# --------- CSV Writer ---------
def save_to_csv(results, path="evaluation_results.csv"):
    if not results:
        print("No results to save.")
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"[CSV] Evaluation results saved to: {path}")

# --------- MAIN ---------
if __name__ == "__main__":
    reviews_sample = load_file("reviews_sample.txt")
    benchmark = load_file("benchmark.txt")
    summary_4 = load_file("scenario_4_summary.txt")
    summary_5 = load_file("scenario_5_summary.txt")

    all_results = []

    # Compare Scenario 4 and Scenario 5 against the Original Reviews (source)
    all_results.append(evaluate(reviews_sample, summary_4, "Scenario 4 vs Source"))
    all_results.append(evaluate(reviews_sample, summary_5, "Scenario 5 vs Source"))

    # Compare Scenario 4 and Scenario 5 against the Benchmark Summary
    all_results.append(evaluate(benchmark, summary_4, "Scenario 4 vs Benchmark"))
    all_results.append(evaluate(benchmark, summary_5, "Scenario 5 vs Benchmark"))

    # Determine best result by ROUGE-L F1 (with simple cosine tie-breaker)
    best = all_results[0]
    for result in all_results[1:]:
        if result["ROUGE-L F1"] > best["ROUGE-L F1"]:
            best = result
        elif result["ROUGE-L F1"] == best["ROUGE-L F1"]:
            if result["Cosine Similarity"] > best["Cosine Similarity"]:
                best = result

    print(f"\n=== Best Combination ===")
    print(f"Label: {best['Label']}")
    print(f"ROUGE-L F1: {best['ROUGE-L F1']:.4f}")
    print(f"Cosine Similarity: {best['Cosine Similarity']:.4f}")

    # Save to CSV
    save_to_csv(all_results)
