import pandas as pd

from sentence_transformers import SentenceTransformer

from recommender import (
    build_prerequisite_graph,
    filter_eligible_courses,
    rank_hybrid_courses,
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

def main():
    courses = pd.read_csv(
        DATA_DIR / "recommendable_courses_fall_2026.csv"
)

    prereqs = pd.read_csv(
        DATA_DIR / "prerequisites_fall_2026.csv"
)


    graph = build_prerequisite_graph(prereqs)

    completed_courses = {
        "DATA C8",
        "COMPSCI 61A",
        "MATH 1A",
        "MATH 1B",
    }

    interests = (
        "machine learning statistics data science "
        "product analytics"
    )

    preferred_subjects = {
        "DATA",
        "STAT",
        "COMPSCI",
        "UGBA",
        "INDENG",
    }

    eligible_courses = filter_eligible_courses(
        courses,
        completed_courses,
        graph
    )

    eligible_courses["text"] = (
        eligible_courses["title"].fillna("")
        + ". "
        + eligible_courses["description"].fillna("")
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    course_embeddings = model.encode(
        eligible_courses["text"].tolist(),
        show_progress_bar=True
    )

    recommendations = rank_hybrid_courses(
        eligible_courses,
        interests,
        preferred_subjects,
        model,
        course_embeddings
    )

    print(
        recommendations[
            ["course", "title", "final_score"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()