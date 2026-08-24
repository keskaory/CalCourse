from pathlib import Path
import sys

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

    return courses, prereqs


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


courses, prereqs = load_data()
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

    eligible_courses["text"] = (
        eligible_courses["title"].fillna("")
        + ". "
        + eligible_courses["description"].fillna("")
    )

    course_embeddings = model.encode(
        eligible_courses["text"].tolist(),
        show_progress_bar=False
    )

    recommendations = rank_hybrid_courses(
        eligible_courses,
        interests,
        preferred_subjects,
        model,
        course_embeddings
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