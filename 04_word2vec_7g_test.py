from gensim.models import Word2Vec

# Load the existing Word2Vec model
model = Word2Vec.load("stackoverflow_7g_word2vec.model")

# Function to extract N-grams from the model's vocabulary
def extract_ngrams(model, n, top_n=10):
    ngrams = [word for word in model.wv.index_to_key if len(word.split('_')) == n]
    return ngrams[:top_n]

# Extract and print top 10 2-grams
top_2grams = extract_ngrams(model, 2)
print("Top 10 2-grams:")
for ngram in top_2grams:
    print(ngram)

# Extract and print top 10 7-grams
top_7grams = extract_ngrams(model, 7)
print("\nTop 10 7-grams:")
for ngram in top_7grams:
    print(ngram)