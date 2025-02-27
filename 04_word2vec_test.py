from gensim.models import Word2Vec

# Load the custom-trained Word2Vec model
model = Word2Vec.load("stackoverflow_word2vec.model")

# Define quality criteria
quality_criteria = ["performance", "security", "usability", "scalability", "maintainability"]

# Extract related words
related_words = {}
for criterion in quality_criteria:
    if criterion in model.wv:
        similar_words = model.wv.most_similar(criterion, topn=10)  # Top 10 similar words
        related_words[criterion] = [word for word, _ in similar_words]
    else:
        related_words[criterion] = ["❌ Word not in vocabulary"]

# Print extracted words
for criterion, words in related_words.items():
    print(f"🔹 Quality Criterion: {criterion}")
    print(f"   Related Words: {', '.join(words)}")
