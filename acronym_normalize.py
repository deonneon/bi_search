import re

# Stage 1: Map variations of full phrases/words to their canonical form
# This is crucial for handling hyphenation and compound words that should be separate.
VARIATION_MAP = {
    'non-conformance report': 'non conformance report',
    'nonconformance report': 'non conformance report',
    'non-conformance': 'non conformance', # Handle just the compound term itself
    'nonconformance': 'non conformance'
    # Add other common variations you observe in your data
}

# Stage 2: Map acronyms to their canonical full form (after variations are handled)
ACRONYM_MAP = {
    'ncr': 'non conformance report', # Maps the acronym to the CANONICAL expanded form
    'rpt': 'report',
    'rep': 'report',
    'inv': 'inventory',
    'fin': 'financial',
    'perf': 'performance',
    'q1': 'quarter 1',
    'q2': 'quarter 2',
    'q3': 'quarter 3',
    'q4': 'quarter 4',
    'yoy': 'year over year',
    'ar': 'accounts receivable',
    'ap': 'accounts payable',
    'gl': 'general ledger'
}

def normalize_and_expand_text(text):
    text = text.lower() # Always lowercase first

    # Apply Stage 1: Normalize variations of phrases/words
    # Use regex.sub for robust whole-word/phrase replacement
    # Sort keys by length in descending order to avoid partial matches
    for k in sorted(VARIATION_MAP.keys(), key=len, reverse=True):
        # Use word boundaries to prevent replacing 'nonconformance' in 'supernonconformance'
        # \b matches word boundaries
        text = re.sub(r'\b' + re.escape(k) + r'\b', VARIATION_MAP[k], text)

    # Apply Stage 2: Expand acronyms
    words = text.split()
    expanded_words = []
    for word in words:
        expanded_words.append(ACRONYM_MAP.get(word, word)) # If word is an acronym, replace, else keep

    return ' '.join(expanded_words)

# Example:
text_in_1 = "NCR report details"
text_out_1 = normalize_and_expand_text(text_in_1)
print(f"'{text_in_1}' -> '{text_out_1}'") # Expected: 'non conformance report report details' (note: "report" duplication needs handling later if desired, but this is accurate for tokenization)

text_in_2 = "non-conformance report summary"
text_out_2 = normalize_and_expand_text(text_in_2)
print(f"'{text_in_2}' -> '{text_out_2}'") # Expected: 'non conformance report summary'

text_in_3 = "nonconformance report q1 perf"
text_out_3 = normalize_and_expand_text(text_in_3)
print(f"'{text_in_3}' -> '{text_out_3}'") # Expected: 'non conformance report quarter 1 performance'

text_in_4 = "non conformance report"
text_out_4 = normalize_and_expand_text(text_in_4)
print(f"'{text_in_4}' -> '{text_out_4}'") # Expected: 'non conformance report'