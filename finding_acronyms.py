(word) < 6 and count > 5 and word not in english_vocab:
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