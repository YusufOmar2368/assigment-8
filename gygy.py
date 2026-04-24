"""
Lightweight analysis script for the RJ_* texts you pasted.
- Tokenize (NLTK), Porter stem, spaCy lemmatize
- Top-20 tokens / stems / lemmas per file
- Named-entity count + type breakdown (spaCy)
- Top-10 trigrams per file
- Pairwise TF-IDF cosine similarity and 3-gram Jaccard
Usage (Mac):
  python3 -m venv venv && source venv/bin/activate
  pip install -U pip
  pip install spacy nltk scikit-learn
  python -m spacy download en_core_web_sm
  python /Users/yomar/Downloads/analyze_rj_texts.py \
    /Users/yomar/Downloads/RJ_Martin.txt \
    /Users/yomar/Downloads/RJ_Tolkein.txt \
    /Users/yomar/Downloads/RJ_Lovecraft.txt \
    [optional /path/to/Text_4.txt]
"""
import sys
import re
from collections import Counter
from pathlib import Path

import nltk
from nltk.stem import PorterStemmer
from nltk.util import ngrams
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ensure punkt
nltk.download("punkt", quiet=True)

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

def read_text(path):
    return Path(path).read_text(encoding="utf-8")

def clean_for_tokenize(text):
    # keep basic punctuation separation, lowercase for token freq
    return re.sub(r'\s+', ' ', text.strip())

def tokenize(text):
    toks = nltk.word_tokenize(text)
    # keep alphanumeric tokens (drop pure punctuation)
    toks = [t.lower() for t in toks if any(c.isalnum() for c in t)]
    return toks

def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha]

def named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def trigram_counts(tokens):
    return Counter(ngrams(tokens, 3))

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

def analyze(paths):
    docs = []
    info = {}
    for p in paths:
        raw = read_text(p)
        cleaned = clean_for_tokenize(raw)
        tokens = tokenize(cleaned)
        stems = stem_tokens(tokens)
        lemmas = lemmatize(cleaned)
        ents = named_entities(raw)  # preserve case for NER
        tri_counts = trigram_counts(tokens)
        info[p] = {
            "raw": raw,
            "cleaned": cleaned,
            "tokens": tokens,
            "stems": stems,
            "lemmas": lemmas,
            "entities": ents,
            "tri_counts": tri_counts,
            "token_freq": Counter(tokens),
            "stem_freq": Counter(stems),
            "lemma_freq": Counter(lemmas),
        }
        docs.append(cleaned)
    return info, docs

def pairwise_tfidf(docs, paths):
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vect.fit_transform(docs)
    sims = cosine_similarity(X)
    results = {}
    n = len(paths)
    for i in range(n):
        for j in range(i+1, n):
            results[(paths[i], paths[j])] = float(sims[i, j])
    return results

def pairwise_trigram_jaccard(info, paths):
    tri_sets = {p: set(" ".join(t) for t in info[p]["tri_counts"].keys()) for p in paths}
    results = {}
    n = len(paths)
    for i in range(n):
        for j in range(i+1, n):
            results[(paths[i], paths[j])] = jaccard(tri_sets[paths[i]], tri_sets[paths[j]])
    return results

def short_print(info, paths):
    for p in paths:
        d = info[p]
        print("====", Path(p).name, "====")
        print("Top 20 tokens:", d["token_freq"].most_common(20))
        print("Top 20 stems:", d["stem_freq"].most_common(20))
        print("Top 20 lemmas:", d["lemma_freq"].most_common(20))
        ents = d["entities"]
        print("Named entities (count):", len(ents))
        if ents:
            types = Counter([t[1] for t in ents])
            print("Entity types:", types.most_common())
        print("Top 10 trigrams:", [" ".join(t) for t,_ in d["tri_counts"].most_common(10)])
        print()

def main(argv):
    if len(argv) < 2:
        print("Usage: python analyze_rj_texts.py <file1> <file2> <file3> [optional file4]")
        return
    paths = argv[1:]
    for p in paths:
        if not Path(p).exists():
            print("Missing file:", p)
            return
    info, docs = analyze(paths)
    short_print(info, paths)

    tfidf = pairwise_tfidf(docs, paths)
    tri_jac = pairwise_trigram_jaccard(info, paths)

    print("=== Pairwise TF-IDF cosine similarities ===")
    for k, v in sorted(tfidf.items(), key=lambda x: x[0]):
        print(f"{Path(k[0]).name} <-> {Path(k[1]).name}: {v:.4f}")
    print()

    print("=== Pairwise 3-gram Jaccard similarities ===")
    for k, v in sorted(tri_jac.items(), key=lambda x: x[0]):
        print(f"{Path(k[0]).name} <-> {Path(k[1]).name}: {v:.4f}")
    print()

    # If a fourth text provided, show which of first three is most similar
    if len(paths) >= 4:
        target = paths[-1]
        candidates = paths[:-1]
        best_tfidf = max(((c, tfidf.get((c, target)) or tfidf.get((target, c) ) or 0.0) for c in candidates),
                         key=lambda x: x[1])
        best_jac = max(((c, tri_jac.get((c, target)) or tri_jac.get((target, c)) or 0.0) for c in candidates),
                       key=lambda x: x[1])
        print("If Text_4 is", Path(target).name)
        print("Best TF-IDF match among Text_1-3:", Path(best_tfidf[0]).name, "score", f"{best_tfidf[1]:.4f}")
        print("Best 3-gram Jaccard match among Text_1-3:", Path(best_jac[0]).name, "score", f"{best_jac[1]:.4f}")
    else:
        print("No fourth text provided; authorship comparison skipped.")

if __name__ == "__main__":
    main(sys.argv)