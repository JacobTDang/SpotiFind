# normalize_embeddings.py
import numpy as np
from app import create_app
from models import SongEmbedding, db
from audio_utils import normalize_embedding

def normalize_all_embeddings():
    """
    Load every SongEmbedding, check its L2 norm,
    and if it’s not ≈1.0, replace it with the unit‐normalized vector.
    """
    app = create_app()
    with app.app_context():
        all_embeddings = SongEmbedding.query.all()
        count_fixed = 0

        for emb in all_embeddings:
            arr = np.array(emb.embedding)
            # Convert stored list to np.ndarray
            orig_norm = np.linalg.norm(arr)
            if abs(orig_norm - 1.0) > 0.01:
                # Normalize in Python
                unit_vec = normalize_embedding(arr)
                emb.embedding = unit_vec.tolist()
                count_fixed += 1

        if count_fixed > 0:
            db.session.commit()
            print(f"Normalized {count_fixed} embeddings.")
        else:
            print("All embeddings were already unit‐normed.")

if __name__ == "__main__":
    normalize_all_embeddings()
