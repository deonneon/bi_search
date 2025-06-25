import os, re, json
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from bertopic import BERTopic
import umap

# ──────────────────────────────────────────────────────────────────────────────
# Tweakables – change these for quick experimentation                           
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH      = "microstrategy_metadata.csv"   # CSV with cols: object_name, description, object_type
GLOSSARY_PATH = "glossary.json"                # optional JSON {"mtd": "month_to_date", ...}
VECTOR_SIZE   = 256    # embedding dimension for Doc2Vec
EPOCHS        = 40     # training passes – 40 is usually enough
MIN_COUNT     = 5      # ignore tokens appearing <5 times to keep vocab sane
OUT_DIR       = "."    # where to save the models

# ──────────────────────────────────────────────────────────────────────────────
# 1. Helpers                                                                    
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_GLOSSARY = {
    "yoq": "year_over_year",
    "ty" : "this_year",
    "ly" : "last_year",
    "mtd": "month_to_date",
}

def load_glossary(path: str) -> dict:
    """Merge user glossary (if present) with defaults and lower‑case everything."""
    if os.path.exists(path):
        with open(path) as f:
            user = {k.lower(): v.lower() for k, v in json.load(f).items()}
            return {**DEFAULT_GLOSSARY, **user}
    return DEFAULT_GLOSSARY

glossary = load_glossary(GLOSSARY_PATH)

# pre‑compiled regex for camelCase splitting (a tiny speed boost)
split_camel = re.compile(r"([a-z])([A-Z])")

def expand_acronyms(text: str) -> str:
    """Replace any glossary key *as a whole word* with its expansion (case‑insensitive)."""
    for short, long in glossary.items():
        text = re.sub(fr"\b{short}\b", long, text, flags=re.I)
    return text

def split_identifier(text: str) -> str:
    """Convert SalesPlan_QTY → "sales plan qty" for better tokenisation."""
    return split_camel.sub(r"\1 \2", text).replace("_", " ")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load CSV and create one concatenated text field                             
# ──────────────────────────────────────────────────────────────────────────────
print("[1/5] reading csv…")
df = pd.read_csv(CSV_PATH)

# Build a single column called "document" that holds the cleaned text per object
# The lambda is longish but clearer than a helper function for casual hacking

df["document"] = (
    df.apply(lambda r: expand_acronyms(
        f"{split_identifier(str(r.get('object_name', 'unknown')))} "
        f"{str(r.get('description', ''))} "
        f"{str(r.get('object_type', ''))}"
    ).lower(), axis=1)
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Detect common bigrams / trigrams and define a tokeniser                     
# ──────────────────────────────────────────────────────────────────────────────
print("[2/5] building bigram / trigram models…")
raw_tokens = [simple_preprocess(t) for t in df["document"]]
# Phrases learns which two‑word and three‑word combos should stick together
bigram  = Phraser(Phrases(raw_tokens, min_count=10, threshold=10))
trigram = Phraser(Phrases(bigram[raw_tokens], threshold=8))

def tokenize(text: str):
    """micro helper so we don’t repeat the trigram/bigram chain everywhere."""
    return trigram[bigram[simple_preprocess(text)]]

tokenised = [tokenize(t) for t in df["document"]]
tagged     = [TaggedDocument(words=tok, tags=[i]) for i, tok in enumerate(tokenised)]

# ──────────────────────────────────────────────────────────────────────────────
# 4. Train Doc2Vec – unsupervised document embeddings                            
# ──────────────────────────────────────────────────────────────────────────────
print("[3/5] training Doc2Vec… (grab coffee)")
d2v = Doc2Vec(
    vector_size=VECTOR_SIZE,
    window=10,              # context window
    min_count=MIN_COUNT,
    dm=1,                   # distributed memory (vs. DBOW)
    negative=10,            # negative sampling for speed
    workers=os.cpu_count(),
    epochs=EPOCHS,
)

d2v.build_vocab(tagged)
d2v.train(tagged, total_examples=d2v.corpus_count, epochs=d2v.epochs)
d2v.save(os.path.join(OUT_DIR, "mstr_doc2vec.bin"))

# ──────────────────────────────────────────────────────────────────────────────
# 5. Cluster with BERTopic (UMAP → HDBSCAN)                                      
# ──────────────────────────────────────────────────────────────────────────────
print("[4/5] clustering with BERTopic…")

def embed(docs):
    """Tiny wrapper so BERTopic can call embed([str, str]) → list[np.ndarray]."""
    return [d2v.infer_vector(tokenize(doc)) for doc in docs]

umap_model = umap.UMAP(
    n_neighbors=15,  # topic granularity – fewer neighbours = more clusters
    n_components=5,  # dimensionality going into HDBSCAN
    metric="cosine",
    random_state=42,
)

model = BERTopic(
    embedding_model=embed,
    umap_model=umap_model,
    calculate_probabilities=True,
)
model.fit_transform(df["document"].tolist())
model.save(os.path.join(OUT_DIR, "mstr_bertopic"))

# ──────────────────────────────────────────────────────────────────────────────
# 6. Quick sanity check                                                         
# ──────────────────────────────────────────────────────────────────────────────
print("[5/5] top topics (size, topic_id, name):")
print(model.get_topic_info().head(10))
print("\nDone ✔ – models written to", os.path.abspath(OUT_DIR))
