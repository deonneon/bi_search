import os, re, json
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from bertopic import BERTopic
import umap

# --- tweakables --------------------------------------------------------------
CSV_PATH      = "microstrategy_metadata.csv"   # path to your metadata export
GLOSSARY_PATH = "glossary.json"                # optional file of {acronym: full}
VECTOR_SIZE   = 256
EPOCHS        = 40
MIN_COUNT     = 5
OUT_DIR       = "."                              # where models get saved
# -----------------------------------------------------------------------------

# 1) tiny helpers --------------------------------------------------------------
DEFAULT_GLOSSARY = {
    "yoq": "year_over_year",
    "ty": "this_year", "ly": "last_year",
    "mtd": "month_to_date",
}

def load_glossary(path):
    if os.path.exists(path):
        with open(path) as f:
            user = {k.lower(): v.lower() for k, v in json.load(f).items()}
            return {**DEFAULT_GLOSSARY, **user}
    return DEFAULT_GLOSSARY

glossary = load_glossary(GLOSSARY_PATH)

split_camel = re.compile(r"([a-z])([A-Z])")

def expand(text):
    for short, long in glossary.items():
        text = re.sub(fr"\b{short}\b", long, text, flags=re.I)
    return text

def split_identifier(text):
    return split_camel.sub(r"\1 \2", text).replace("_", " ")

# 2) load & munge -------------------------------------------------------------
print("[1/5] reading csv…")
df = pd.read_csv(CSV_PATH)

df["document"] = (
    df.apply(lambda r: expand(f"{split_identifier(str(r.get('object_name','unknown')))} "
                              f"{str(r.get('description',''))} {str(r.get('object_type',''))}".lower()),
              axis=1)
)

# 3) phrases & tokenisation ----------------------------------------------------
print("[2/5] building bigram / trigram models…")
raw_tokens = [simple_preprocess(t) for t in df["document"]]
bigram   = Phraser(Phrases(raw_tokens, min_count=10, threshold=10))
trigram  = Phraser(Phrases(bigram[raw_tokens], threshold=8))

def tokenize(text):
    return trigram[bigram[simple_preprocess(text)]]

tokenised = [tokenize(t) for t in df["document"]]
tagged     = [TaggedDocument(words=tok, tags=[i]) for i, tok in enumerate(tokenised)]

# 4) Doc2Vec ------------------------------------------------------------------
print("[3/5] training Doc2Vec… (this can take a bit)")
d2v = Doc2Vec(vector_size=VECTOR_SIZE,
              window=10,
              min_count=MIN_COUNT,
              dm=1,
              negative=10,
              workers=os.cpu_count(),
              epochs=EPOCHS)
d2v.build_vocab(tagged)
d2v.train(tagged, total_examples=d2v.corpus_count, epochs=d2v.epochs)
d2v.save(os.path.join(OUT_DIR, "mstr_doc2vec.bin"))

# 5) BERTopic  -----------------------------------------------------------------
print("[4/5] clustering with BERTopic…")

def embed(docs):
    return [d2v.infer_vector(tokenize(doc)) for doc in docs]

umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42)

model = BERTopic(embedding_model=embed, umap_model=umap_model, calculate_probabilities=True)
model.fit_transform(df["document"].tolist())
model.save(os.path.join(OUT_DIR, "mstr_bertopic"))

# 6) quick peek ----------------------------------------------------------------
print("[5/5] top topics:")
print(model.get_topic_info().head(10))
print("\nDone ✔")
