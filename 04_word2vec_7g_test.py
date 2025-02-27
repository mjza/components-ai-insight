import gensim
import os

# Path to your trained model
MODEL_PATH = "stackoverflow_7g_word2vec.model"

# **Load Word2Vec Model**
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file '{MODEL_PATH}' not found!")
    exit()

print("🔄 Loading Word2Vec model...")
model = gensim.models.Word2Vec.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# **Check Vocabulary**
vocab = list(model.wv.index_to_key)  # Get all words/phrases
print(f"🧠 Vocabulary size: {len(vocab)} words/phrases")

# **Display Sample Words/Phrases**
print("\n🔹 Sample Words & Phrases from the Model:")
for word in vocab[:30]:  # Show first 30 words
    print(f"   {word}")

# **Check if Multi-Word Phrases Exist**
ngram_count = sum(1 for word in vocab if "_" in word)
print(f"\n📝 Found {ngram_count} multi-word phrases (n-grams) in the model!")

# **Search for N-Grams**
search_terms = ["machine_learning", "error_message", "database_query", "performance_optimization"]

print("\n🔍 Checking if some n-grams exist in the model:")
for term in search_terms:
    if term in model.wv:
        print(f"   ✅ '{term}' exists in the model!")
    else:
        print(f"   ❌ '{term}' not found.")

# **Find Similar Words/N-Grams**
while True:
    query = input("\n🔍 Enter a word/phrase to find similar words (or 'exit' to quit): ").strip()
    if query.lower() == "exit":
        break
    if query in model.wv:
        print(f"📌 Similar words to '{query}':")
        for word, score in model.wv.most_similar(query, topn=10):
            print(f"   {word} (score: {score:.4f})")
    else:
        print(f"❌ '{query}' not found in the model.")
