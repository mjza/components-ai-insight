import os
from gensim.models import Word2Vec

def load_word2vec_model(model_path):
    """Loads a Word2Vec model from the given path."""
    try:
        model = Word2Vec.load(model_path, mmap="r")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def get_model_metadata(model, model_name, model_file):
    """Extracts metadata from a Word2Vec model."""
    metadata = {
        "Model ID": model_file,
        "Model Name": model_name,
        "Vocabulary Size": len(model.index_to_key),
        "Vector Size": model.vector_size
    }
    return metadata

if __name__ == "__main__":
    model_folders = [
        "stackoverflow_7g_v2_word2vec_final",
        "stackoverflow_7g_word2vec_final",
        "stackoverflow_v2_word2vec_final",
        "stackoverflow_word2vec_final",
    ]
    
    metadata_list = []
    
    for folder in model_folders:
        model_files = [f for f in os.listdir(folder) if f.endswith(".model")]
        
        if not model_files:
            print(f"No model file found in {folder}")
            continue
        
        model_file = model_files[0]  # Load the first found model file
        model_path = os.path.join(folder, model_file)
        model = load_word2vec_model(model_path)
        
        if model:
            metadata = get_model_metadata(model, folder, model_file)
            metadata_list.append(metadata)
    
    for metadata in metadata_list:
        print("\n--- Model Metadata ---")
        for key, value in metadata.items():
            print(f"{key}: {value}")
