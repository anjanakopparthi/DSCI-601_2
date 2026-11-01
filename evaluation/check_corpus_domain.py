import pandas as pd

c = pd.read_csv('data_collection/collected/tamil_corpus_clean.csv')
print("DAPT corpus kind:")
print(c['kind'].value_counts())
print()

def script(s):
    s = str(s)
    tam = sum(1 for ch in s if '\u0b80' <= ch <= '\u0bff')
    lat = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    if tam + lat == 0:
        return 'other'
    return 'native' if tam / (tam + lat) > 0.5 else 'latin'

t = pd.read_csv('processed/tamil_train_processed.csv')
print("HopeEDI tamil_train script mix:")
print(t['text'].map(script).value_counts(normalize=True).round(3))

print()
print("DAPT corpus script mix (independent of the 'kind' column):")
print(c['text'].map(script).value_counts(normalize=True).round(3))