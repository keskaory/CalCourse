from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


def main():
    courses = pd.read_csv(
        DATA_DIR / "recommendable_courses_fall_2026.csv"
    )

    courses["course"] = (
        courses["subject"].str.upper().str.strip()
        + " "
        + courses["course_number"].astype(str).str.upper().str.strip()
    )

    courses["text"] = (
        courses["title"].fillna("")
        + ". "
        + courses["description"].fillna("")
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        courses["text"].tolist(),
        show_progress_bar=True
    )

    np.save(
        DATA_DIR / "course_embeddings_fall_2026.npy",
        embeddings
    )

    courses[["course"]].to_csv(
        DATA_DIR / "course_embedding_index_fall_2026.csv",
        index=False
    )

    print(f"Saved embeddings: {embeddings.shape}")
    print(f"Saved course labels: {len(courses)}")


if __name__ == "__main__":
    main()