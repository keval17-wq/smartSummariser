import sys
import re
import time
import numpy as np
from openai import OpenAI

# Your config file references

DATA_FILE_PATH = "data_for_run.txt"
SUMMARY_OUTPUT_TXT = "scenario_1_summary.txt"
SUMMARY_OUTPUT_EMB = "scenario_2_summary.txt"


# Initialize OpenAI client
client = OpenAI(api_key="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")

# Attempt to import Hugging Face pipeline for optional sentiment
try:
    from transformers import pipeline
    import torch
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    sentiment_pipeline = None
    print(f"[WARNING] Sentiment pipeline failed to load: {e}")

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------
# Common Helpers
# -------------------------------
def load_reviews(filename):
    """Load lines from a file, ignoring empty lines."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        reviews = [line.strip() for line in lines if line.strip()]
        print(f"Loaded {len(reviews)} reviews from {filename}")
        return reviews
    except Exception as e:
        print(f"Error loading reviews: {e}")
        return []

def gpt4_summarize(text, system_instruction="You are an expert summarizer focused on conciseness and clarity."):
    """Use GPT-4 to summarize text with a given system instruction."""
    if not text.strip():
        return "No meaningful text to summarize."
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def save_summaries(summaries, output_path):
    """Save multiple summary strings to a file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for s in summaries:
                f.write(s + "\n\n")
        print(f"Saved summaries to {output_path}")
    except Exception as e:
        print(f"Error saving summaries: {e}")


# -------------------------------
# Scenario 1
# -------------------------------
# def generate_summaries(reviews):
#     """Generate GPT-4 summaries for each review in the list (one by one)."""
#     results = []
#     for rev in reviews:
#         prompt = f"Please provide a concise summary of the following text:\n\n{rev}"
#         summary = gpt4_summarize(prompt)
#         if summary:
#             results.append(summary)
#             print(f"Original (first 60 chars): {rev[:60]}... | Summary (first 60 chars): {summary[:60]}...")
#         else:
#             results.append("Summary generation failed.")
#     return results

# def scenario_1():
#     """Summarize first 15 raw reviews individually."""
#     start_time = time.time()
#     reviews = load_reviews(DATA_FILE_PATH)[:15]
#     summaries = generate_summaries(reviews)
#     save_summaries(summaries, SUMMARY_OUTPUT_TXT)
#     elapsed = time.time() - start_time
#     print(f"Scenario 1 completed in {elapsed:.2f} seconds.")


# # -------------------------------
# # Scenario 2
# # -------------------------------
# def generate_embeddings(reviews):
#     """Generate embeddings (text-embedding-ada-002) for a list of reviews."""
#     embeddings = []
#     for r in reviews:
#         try:
#             resp = client.embeddings.create(input=r, model="text-embedding-ada-002")
#             embeddings.append(resp.data[0].embedding)
#         except Exception as e:
#             print(f"Error generating embedding for '{r[:60]}': {e}")
#             embeddings.append([0.0]*1536)
#     return np.array(embeddings)

# def cluster_reviews(embeddings, reviews, n_clusters=5):
#     """Cluster reviews by embeddings using K-Means."""
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(embeddings)
#     clusters = {}
#     for label, rev in zip(labels, reviews):
#         clusters.setdefault(label, []).append(rev)
#     print(f"Reviews clustered into {n_clusters} clusters.")
#     return clusters

# def scenario_2():
#     """Embed reviews -> K-Means -> Summarize each cluster with GPT-4."""
#     start_time = time.time()
#     reviews = load_reviews(DATA_FILE_PATH)
#     emb = generate_embeddings(reviews)
#     clusters_dict = cluster_reviews(emb, reviews, n_clusters=5)

#     summaries = []
#     for cid, revs_in_cluster in clusters_dict.items():
#         cluster_text = " ".join(revs_in_cluster)
#         prompt = f"Please summarize this cluster of user feedback:\n\n{cluster_text}"
#         s = gpt4_summarize(prompt, "You are an expert summarizer focusing on the main themes.")
#         summaries.append(f"Cluster {cid}:\n{s if s else 'Summary generation failed.'}")

#     save_summaries(summaries, SUMMARY_OUTPUT_EMB)
#     elapsed = time.time() - start_time
#     print(f"Scenario 2 completed in {elapsed:.2f} seconds.")


# # -------------------------------
# # Scenario 3
# # -------------------------------
# def traditional_cluster_reviews(reviews, n_clusters=5):
#     """Cluster reviews using TF-IDF + K-Means."""
#     vec = TfidfVectorizer(stop_words='english')
#     X = vec.fit_transform(reviews)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(X)
#     clusters = {}
#     for label, rev in zip(labels, reviews):
#         clusters.setdefault(label, []).append(rev)
#     print(f"Reviews clustered into {n_clusters} clusters using TF-IDF.")
#     return clusters

# def scenario_3():
#     """TF-IDF + K-Means -> Summarize each cluster with GPT-4."""
#     start_time = time.time()
#     reviews = load_reviews(DATA_FILE_PATH)
#     clusters = traditional_cluster_reviews(reviews, n_clusters=5)

#     results = []
#     for cid, revs in clusters.items():
#         text_block = " ".join(revs)
#         prompt = f"Please summarize this cluster of user feedback:\n\n{text_block}"
#         s = gpt4_summarize(prompt, "You are an expert summarizer focusing on the main themes.")
#         results.append(f"Cluster {cid}:\n{s if s else 'Summary generation failed.'}")

#     save_summaries(results, 'scenario_3_summaries.txt')
#     elapsed = time.time() - start_time
#     print(f"Scenario 3 completed in {elapsed:.2f} seconds.")


# -------------------------------
# Scenario 4
# -------------------------------
def scenario_4():
    """
    TF-IDF-based extractive step + final GPT-4 summarization.
    """
    start_time = time.time()
    reviews = load_reviews(DATA_FILE_PATH)
    all_text = " ".join(reviews)
    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        print("No sentences to process.")
        return

    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    sorted_idx = np.argsort(scores)[::-1]

    top_n = min(5, len(sorted_idx))
    top_sents = [sentences[i] for i in sorted_idx[:top_n]]
    extractive_draft = ". ".join(top_sents) + "."

    prompt = (
        f"These are the top sentences I extracted:\n\n{extractive_draft}\n\n"
        "Please generate a single-paragraph summary that captures all main themes, issues, and sentiments."
    )
    final_sum = gpt4_summarize(prompt, "You are an expert summarizer ensuring completeness and coherence.")
    if not final_sum:
        final_sum = "Summary generation failed."

    res = [
        "Extractive Draft Sentences:\n" + extractive_draft,
        "Final Abstractive Summary:\n" + final_sum
    ]
    save_summaries(res, "scenario_4_summary.txt")
    elapsed = time.time() - start_time
    print(f"Scenario 4 completed in {elapsed:.2f} seconds.")


# -------------------------------
# Scenario 5 (with debugging, optional sentiment)
# -------------------------------
def scenario_5_augmented(include_sentiment: bool = True,
                         top_n: int = 15,
                         alpha: float = 0.2):
    """
    Scenario 5 – TF‑IDF‑based extraction (+ optional sentiment weighting)  
    → single‑paragraph GPT‑4 summary that is **always shorter** than the
    extracted source text, purely factual, and neutral.

    Parameters
    ----------
    include_sentiment : bool
        Boost TF‑IDF scores by sentiment magnitude if True.
    top_n : int
        Number of highest‑scoring sentences to keep before summarisation.
    alpha : float
        Weight factor for sentiment (0 ≤ alpha ≤ 1).
    """
    import time, re, numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    start = time.time()
    print(f"[DEBUG] Scenario 5 started (sentiment={include_sentiment}, "
          f"top_n={top_n}, alpha={alpha})")

    # ------------------------------------------------------------------ #
    # 1 · Load reviews → split into sentences
    # ------------------------------------------------------------------ #
    reviews = load_reviews(DATA_FILE_PATH)                     # helper exists
    sentences = [s.strip() for s in
                 re.split(r"[.!?]+", " ".join(reviews)) if s.strip()]
    print(f"[DEBUG] Parsed {len(sentences)} sentences.")
    if not sentences:
        print("[DEBUG] No sentences found – exiting.")
        return

    # ------------------------------------------------------------------ #
    # 2 · TF‑IDF scoring
    # ------------------------------------------------------------------ #
    vec = TfidfVectorizer(stop_words="english")
    tfidf_scores = vec.fit_transform(sentences).sum(axis=1).A1

    # ------------------------------------------------------------------ #
    # 3 · Optional sentiment re‑weighting (internal only)
    # ------------------------------------------------------------------ #
    if include_sentiment and sentiment_pipeline:
        results = sentiment_pipeline(sentences)                # [{'score':…}]
        tfidf_scores = [
            tf * (1 + alpha * r["score"])
            for tf, r in zip(tfidf_scores, results)
        ]

    # ------------------------------------------------------------------ #
    # 4 · Select top‑N sentences
    # ------------------------------------------------------------------ #
    top_idx = np.argsort(tfidf_scores)[::-1][:min(top_n, len(tfidf_scores))]
    top_sentences = [sentences[i] for i in top_idx]
    print(f"[DEBUG] Selected {len(top_sentences)} top sentences.")

    # ------------------------------------------------------------------ #
    # 5 · Prompt for single plain‑English paragraph
    # ------------------------------------------------------------------ #
    prompt = (
        "You will receive user‑feedback sentences.  Compose ONE plain‑English "
        "paragraph that is strictly shorter than the combined input, purely "
        "factual, neutral in tone, and mentions each distinct positive and "
        "negative fact exactly once.  No bullets or headings.\n\n"
        "Sentences:\n" +
        "\n".join(f"- {s}" for s in top_sentences)
    )

    # ------------------------------------------------------------------ #
    # 6 · First GPT‑4 pass
    # ------------------------------------------------------------------ #
    summary = gpt4_summarize(
        prompt,
        "Return the requested paragraph only."
    ) or "Summary generation failed."

    # ------------------------------------------------------------------ #
    # 7 · Dynamic ceiling: summary must be shorter than source text
    # ------------------------------------------------------------------ #
    SOURCE_LEN = len("\n".join(top_sentences))      # character length of input

    def _too_long(txt: str) -> bool:
        return len(txt) >= SOURCE_LEN

    if _too_long(summary):
        compress_prompt = (
            "Rewrite the following paragraph so it remains purely factual but "
            "is STRICTLY SHORTER in character count than the original source "
            "sentences.  Keep all distinct facts.\n\n" + summary
        )
        summary = gpt4_summarize(
            compress_prompt,
            "Return the shorter paragraph only."
        ) or summary

    # Hard‑trim safeguard (rare)
    if _too_long(summary):
        summary = summary[:SOURCE_LEN - 1].rsplit(" ", 1)[0].rstrip(",.;: ") + "…"

    # ------------------------------------------------------------------ #
    # 8 · Persist extraction + final summary
    # ------------------------------------------------------------------ #
    save_summaries(
        [
            "Extracted Top Sentences (Scenario 5):\n" + "\n".join(top_sentences),
            "\nFinal Factual Summary:\n" + summary
        ],
        "scenario_5_summary.txt"
    )

    print(f"[DEBUG] Scenario 5 completed in {time.time() - start:.2f}s.")




# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("Select scenario to run:")
    print("1 - Summarize first 15 raw reviews (individual GPT-4 summaries)")
    print("2 - Embed + K-Means + Summarize clusters (GPT-4)")
    print("3 - TF-IDF + K-Means + Summarize clusters (GPT-4)")
    print("4 - Extractive (TF–IDF) + Abstractive (GPT-4) hybrid summary (entire dataset)")
    print("5 - TF–IDF extractive + (Optional) Sentiment + GPT-4 final summary")
    choice = input("Enter 1, 2, 3, 4, or 5: ")

    if choice == "1":
        scenario_1()
    elif choice == "2":
        scenario_2()
    elif choice == "3":
        scenario_3()
    elif choice == "4":
        scenario_4()
    elif choice == "5":
        # Toggle sentiment if you want
        scenario_5_augmented(include_sentiment=True)
    else:
        print("Invalid choice. Exiting.")
        sys.exit()
