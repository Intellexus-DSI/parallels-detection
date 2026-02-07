
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    model_name = "sentence-transformers/LaBSE"
    print(f"Loading model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test sentences
    sentences = [
        "སངས་རྒྱས་ཀྱི་ཞབས།",  # 0: Buddha's feet (Tibetan)
        "སྲིན་བུ་པདྨ།",        # 1: Leech/worm (Tibetan - unrelated)
        "བཀྲ་ཤིས་བདེ་ལེགས།",   # 2: Hello/Good luck (Tibetan - common greeting)
        "Hello world",        # 3: English
        "The quick brown fox" # 4: English
    ]
    
    descriptions = [
        "Buddha's feet (Tibetan)",
        "Leech/worm (Tibetan)",
        "Hello (Tibetan)",
        "Hello world (English)",
        "The quick brown fox (English)"
    ]

    print("\nGenerating embeddings...")
    embeddings = model.encode(sentences, normalize_embeddings=True)
    
    print("\nCosine Similarity Matrix:")
    sim_matrix = cosine_similarity(embeddings)
    
    # Print header
    print(f"{'':<30} | {'0':<8} | {'1':<8} | {'2':<8} | {'3':<8} | {'4':<8}")
    print("-" * 80)
    
    for i in range(len(sentences)):
        row_str = f"{descriptions[i]:<30} | "
        for j in range(len(sentences)):
            row_str += f"{sim_matrix[i][j]:.4f}   | "
        print(row_str)

    # Check for collapse
    print("\nAnalysis:")
    tib_sim = sim_matrix[0][1] # Feet vs Worm
    print(f"Similarity between 'Buddha's feet' and 'Worm': {tib_sim:.4f}")
    
    if tib_sim > 0.9:
        print("ALERT: High similarity between unrelated Tibetan texts! Vector space collapse likely.")
    elif tib_sim > 0.8:
        print("WARNING: Moderately high similarity. Model might be weak.")
    else:
        print("OK: distinct Tibetan texts are separable.")

if __name__ == "__main__":
    main()
