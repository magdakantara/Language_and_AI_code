# main.py
import os
import re
import time
from pathlib import Path

import pandas as pd
import spacy


INPUT = "nationality.csv"
OUTPUT = "nationality_preprocessed.csv"

TEXT_COL = "post"
CHUNK_SIZE = 500

LOWERCASE = True
KEEP_PUNCTUATION = True
NORMALIZE_URLS = True
NORMALIZE_USERS = True
FLATTEN_NEWLINES = True

NROWS_TEST = None


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

print("Using BASE_DIR:", BASE_DIR)
print("Files in BASE_DIR:", os.listdir("."))

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
print("spaCy OK")


URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
USER_PATTERN = re.compile(r"@\w+")

def normalize_text(text) -> str:
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    if FLATTEN_NEWLINES:
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

    if LOWERCASE:
        text = text.lower()

    if NORMALIZE_URLS:
        text = URL_PATTERN.sub("<URL>", text)

    if NORMALIZE_USERS:
        text = USER_PATTERN.sub("<USER>", text)

    return text



# Reducers

def make_stopword_and_lemma(texts):
    texts_norm = [normalize_text(t) for t in texts]
    docs = nlp.pipe(texts_norm, batch_size=128)

    stop_out = []
    lemma_out = []

    for doc in docs:
        stop_tokens = []
        lemma_tokens = []

        for t in doc:
            if t.is_space:
                continue
            if (not KEEP_PUNCTUATION) and t.is_punct:
                continue

            # stopword version (spaCy-standard)
            if not t.is_stop:
                stop_tokens.append(t.text)

            # lemma version
            lemma_tokens.append(t.lemma_)

        stop_out.append(" ".join(stop_tokens))
        lemma_out.append(" ".join(lemma_tokens))

    return stop_out, lemma_out


# Sanity check (readable)

def sanity_check():
    df_small = pd.read_csv(INPUT, nrows=3)
    texts = df_small[TEXT_COL].fillna("").astype(str).tolist()

    raw = [normalize_text(t) for t in texts]
    stop_col, lemma_col = make_stopword_and_lemma(texts)

    def short(x, n=200):
        x = str(x)
        return x if len(x) <= n else x[:n] + "..."

    print("\n--- SANITY CHECK (first 3 rows, truncated) ---")
    for i in range(min(3, len(texts))):
        print(f"\nRow {i+1}:")
        print("raw  :", short(raw[i]))
        print("stop :", short(stop_col[i]))
        print("lemma:", short(lemma_col[i]))
    print("\n--------------------------------------------\n")

def main():
    in_path = Path(INPUT)
    out_path = Path(OUTPUT)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path} (in {os.getcwd()})")

    sanity_check()

    first = True
    total_rows = 0
    t0 = time.time()

    read_kwargs = dict(chunksize=CHUNK_SIZE)
    if NROWS_TEST is not None:
        read_kwargs["nrows"] = NROWS_TEST

    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        for chunk_i, chunk in enumerate(pd.read_csv(INPUT, **read_kwargs), start=1):
            texts = chunk[TEXT_COL].fillna("").astype(str).tolist()

            chunk["text_raw"] = [normalize_text(t) for t in texts]
            stop_col, lemma_col = make_stopword_and_lemma(texts)
            chunk["text_stopword"] = stop_col
            chunk["text_lemma"] = lemma_col

            chunk.to_csv(
                f_out,
                index=False,
                header=first,
                lineterminator="\n",
            )
            first = False

            total_rows += len(chunk)
            elapsed = time.time() - t0
            print(f"chunk {chunk_i} | rows {total_rows} | elapsed {elapsed:.1f}s")

    print("DONE ->", out_path)


if __name__ == "__main__":
    main()
