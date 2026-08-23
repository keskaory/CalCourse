import pandas as pd
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def build_prerequisite_graph(prereqs):
    prereqs = prereqs.copy()

    prereqs["course"] = (
        prereqs["subject"] + " " + prereqs["course_number"].astype(str)
    )

    prereqs["prerequisite"] = (
        prereqs["prereq_subject"] + " " + prereqs["prereq_number"].astype(str)
    )

    graph = nx.DiGraph()

    for _, row in prereqs.iterrows():
        graph.add_edge(
            row["prerequisite"],
            row["course"]
        )

    return graph

# helpers

def get_prerequisites(course, graph):
    if course not in graph:
        return set()

    return set(graph.predecessors(course))


def missing_prerequisites(course, completed_courses, graph):
    return get_prerequisites(course, graph) - set(completed_courses)

# eligibility filtering

def filter_eligible_courses(courses, completed_courses, graph):
    courses = courses.copy()

    courses["course"] = (
        courses["subject"]
        + " "
        + courses["course_number"].astype(str)
    )

    courses = courses[
        ~courses["course"].isin(completed_courses)
    ].copy()

    courses["missing_prereqs"] = courses["course"].apply(
        lambda course: missing_prerequisites(
            course,
            completed_courses,
            graph
        )
    )

    return courses[
        courses["missing_prereqs"].apply(len) == 0
    ].copy()

# semantic ranking

def rank_semantic_courses(
    courses,
    interests,
    model,
    course_embeddings
):
    profile_embedding = model.encode([interests])

    scores = cosine_similarity(
        profile_embedding,
        course_embeddings
    ).flatten()

    ranked = courses.copy()
    ranked["semantic_score"] = scores

    return ranked.sort_values(
        "semantic_score",
        ascending=False
    )

# hybrid ranker

def rank_hybrid_courses(
    courses,
    interests,
    preferred_subjects,
    model,
    course_embeddings
):
    profile_embedding = model.encode([interests])

    semantic_scores = cosine_similarity(
        profile_embedding,
        course_embeddings
    ).flatten()

    ranked = courses.copy()
    ranked["semantic_score"] = semantic_scores

    ranked["subject_fit"] = ranked["subject"].apply(
        lambda x: 1 if x in preferred_subjects else 0
    )

    ranked["final_score"] = (
        0.85 * ranked["semantic_score"]
        + 0.15 * ranked["subject_fit"]
    )

    return ranked.sort_values(
        "final_score",
        ascending=False
    )