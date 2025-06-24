# --- Your existing script starts here ---
import pandas as pd
import re

# ===================================================================
# ADVANCED PREPROCESSING PIPELINE
# ===================================================================

# Define your custom noise lists and mappings
VERSION_NOISE = {
    'v', 'ver', 'version', 'final', 'fina', 'draft', 'wip', 'test',
    'dev', 'prod', 'copy', 'old', 'new', 'update', 'updated',
    'v1', 'v2', 'v3', 'v4', 'v01', 'v02', 'bu', 'backup'
}

ACRONYM_MAP = {
    'rpt': 'report', 'rep': 'report', 'inv': 'inventory', 'fin': 'financial',
    'perf': 'performance', 'q1': 'quarter 1', 'q2': 'quarter 2',
    'q3': 'quarter 3', 'q4': 'quarter 4', 'yoy': 'year over year',
    'ar': 'accounts receivable', 'ap': 'accounts payable', 'gl': 'general ledger',
    'cust': 'customer', 'mkt': 'marketing'
}

def normalize_report_name(text):
    """A full pipeline to clean and standardize report names."""
    if not isinstance(text, str):
        return ""

    # Step 1: Foundational Cleaning
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Keep only letters, numbers, spaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # Step 2 & 3: Remove Noise and Expand Acronyms in one pass
    words = text.split()
    processed_words = []
    for word in words:
        if word not in VERSION_NOISE:
            # Get the expanded acronym, or the word itself if not in the map
            processed_words.append(ACRONYM_MAP.get(word, word))

    text = ' '.join(processed_words)

    # Step 4 (Optional but recommended): Canonicalize by sorting
    # This makes "sales report" and "report sales" identical
    text = ' '.join(sorted(text.split()))

    return text

# --- 1. Load and Prepare Data ---
print("\nStep 1: Loading and applying ADVANCED cleaning...")
df = pd.read_csv('microstrategy_reports.csv')

# Create a new column for the cleaned names
print("Original Name -> Cleaned Name")
print("---------------------------------")
for i, name in enumerate(df['name'].head(5)):
    cleaned = normalize_report_name(name)
    print(f"'{name}' -> '{cleaned}'")

df['name_cleaned'] = df['name'].apply(normalize_report_name)

# Combine the CLEANED name with the (lightly cleaned) description
df['description'] = df['description'].fillna('')
# Using a simpler cleaner for the free-text description
df['description_cleaned'] = df['description'].apply(lambda x: clean_baseline(str(x)))

# THE FINAL TEXT for the model is now much cleaner
df['documents'] = df['name_cleaned'] + '. ' + df['description_cleaned']
documents = df['documents'].tolist()

# --- Your script continues from here with Step 2 (TaggedDocument, etc.) ---
# ...