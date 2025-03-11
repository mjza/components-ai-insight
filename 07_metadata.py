import os
import argparse
import gensim

def load_word2vec_model(model_path):
    """Loads a Word2Vec model from the given path."""
    try:
        model = gensim.models.Word2Vec.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def get_model_metadata(model, model_file):
    """Extracts metadata from a Word2Vec model."""
    if hasattr(model, 'wv'):
        vocab = model.wv.index_to_key  # Access vocabulary correctly
    else:
        vocab = model.index_to_key  # For KeyedVectors-only models
    
    return f"{model_file},{len(vocab)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metadata from Word2Vec models.")
    parser.add_argument("--path", required=True, help="Path to the folder containing Word2Vec models.")
    args = parser.parse_args()
    
    model_folder = os.path.abspath(args.path)  # Get the full path from argument
    
    for root, _, files in os.walk(model_folder):
        for file in files:
            if file.endswith(".model"):
                model_path = os.path.join(root, file)
                model = load_word2vec_model(model_path)
                
                if model:
                    print(get_model_metadata(model, file))
