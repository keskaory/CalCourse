from pathlib import Path
import sys
import numpy as np 

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "processed"

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.recommender import (
    build_alias_map,
    build_prerequisite_graph,
    filter_eligible_courses,
    rank_hybrid_courses,
    apply_progression_filter
)

grouped_prereqs = pd.read_csv(
    DATA_DIR / "prerequisites_grouped_fall_2026.csv"
)


@st.cache_data
def load_data():
    courses = pd.read_csv(
        DATA_DIR / "recommendable_courses_fall_2026.csv"
    )

    prereqs = pd.read_csv(
        DATA_DIR / "prerequisites_fall_2026.csv"
    )

    grouped_prereqs = pd.read_csv(
        DATA_DIR / "prerequisites_grouped_fall_2026.csv"
    )

    embedding_index = pd.read_csv(
        DATA_DIR / "course_embedding_index_fall_2026.csv"
    )

    course_embeddings = np.load(
        DATA_DIR / "course_embeddings_fall_2026.npy"
    )

    return (
        courses,
        prereqs,
        grouped_prereqs,
        embedding_index,
        course_embeddings,
    )


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

(
    courses,
    prereqs,
    grouped_prereqs,
    embedding_index,
    course_embeddings,
) = load_data()
alias_map = build_alias_map(courses)

graph = build_prerequisite_graph(
    prereqs,
    alias_map
)
model = load_model()

st.title("CalCourse")

st.write(
    "Personalized UC Berkeley course recommendations based on "
    "completed coursework, interests, and subject preferences."
)

completed_input = st.text_input(
    "Completed courses",
    placeholder="DATA C8, COMPSCI 61A, MATH 1A, MATH 1B"
)

interests = st.text_area(
    "Interests",
    placeholder="machine learning, product management, marketing, GTM"
)

preferred_subjects = st.multiselect(
    "Preferred subjects",
    sorted(courses["subject"].unique())
)

if st.button("Recommend Courses"):

    if not interests.strip():
        st.warning("Enter at least one interest.")
        st.stop()

    completed_courses = {
        course.strip().upper()
        for course in completed_input.split(",")
        if course.strip()
    }

    eligible_courses = filter_eligible_courses(
        courses,
        completed_courses,
        graph,
        alias_map,
        grouped_prereqs
    )

    eligible_courses = apply_progression_filter(
        eligible_courses,
        completed_courses,
        alias_map
    )   

    eligible_courses = eligible_courses.merge(
        embedding_index.reset_index().rename(
         columns={"index": "embedding_idx"}
        ),
        on="course",
        how="left"
    )

    eligible_courses = eligible_courses.dropna(
        subset=["embedding_idx"]
    ).copy()

    eligible_courses["embedding_idx"] = (
        eligible_courses["embedding_idx"].astype(int)
    )

    eligible_embeddings = course_embeddings[
        eligible_courses["embedding_idx"].to_numpy()
    ]

    recommendations = rank_hybrid_courses(
        eligible_courses,
        interests,
        preferred_subjects,
        model,
        eligible_embeddings
    )

    results = recommendations[
        ["course", "title", "final_score"]
    ].head(10).copy()

    results = results.rename(
        columns={"final_score": "match_score"}
    )

    results["match_score"] = results["match_score"].round(3)

    st.subheader("Top Recommendations")

    st.caption(
        "Recommendations are filtered by parsed prerequisites and "
        "ranked using semantic similarity plus subject preferences."
    )

    st.dataframe(
        results,
        width="stretch",
        hide_index=True
    )