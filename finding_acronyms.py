import pandas as pd
import re
from collections import Counter
import nltk
# You may need to download these resources once
# nltk.download('words')
# nltk.download('punkt')

# --- Script to Discover Potential Acronyms ---

print("Finding potential acronyms from report names...")

# Load your raw data
df = pd.read_csv('microstrategy_reports.csv')
# Get a list of all report names, handling potential non-string data
all_names = ' '.join(df['name'].dropna().astype(str))

# Basic cleaning: lowercase and keep only letters and spaces
all_names = all_names.lower()
all_names = re.sub(r'[^a-z\s]', '', all_names)

# Get all unique English words to use as a filter
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

# Count the frequency of all words
word_counts = Counter(all_names.split())

# Find frequent words that are NOT in the English dictionary
potential_acronyms = []
for word, count in word_counts.items():
    # Heuristic: A good candidate is short, frequent, and not a real word.
    if len(word) > 1 and len(word) < 6 and count > 5 and word not in english_vocab:
        potential_acronyms.append((word, count))

# Sort by frequency to see the most important ones first
potential_acronyms.sort(key=lambda x: x[1], reverse=True)

print("\n--- Top Potential Acronyms/Abbreviations to Investigate ---")
print("These are frequent, short words in your data that are not standard English.")
print("Investigate these and add them to your ACRONYM_MAP.")
print("-" * 60)
print(f"{'Potential Acronym':<25} | {'Frequency':<10}")
print("-" * 60)

for acronym, freq in potential_acronyms[:25]: # Show top 25
    print(f"{acronym:<25} | {freq:<10}")