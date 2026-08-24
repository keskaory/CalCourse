import pandas as pd
import networkx as nx
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def add_course_labels(courses):
    courses = courses.copy()

    courses["course"] = (
        courses["subject"].str.upper().str.strip()
        + " "
        + courses["course_number"].astype(str).str.upper().str.strip()
    )

    return courses


def build_alias_map(courses):
    """
    Identify likely cross-listed courses using identical
    course number, title, and description.

    Returns:
        dict mapping every course label to a canonical label.
    """

    courses = add_course_labels(courses)

    courses["title_norm"] = (
        courses["title"]
        .fillna("")
        .str.lower()
        .str.strip()
    )

    courses["description_norm"] = (
        courses["description"]
        .fillna("")
        .str.lower()
        .str.strip()
    )

    alias_map = {}

    grouped = courses.groupby(
        ["course_number", "title_norm", "description_norm"]
    )

    for _, group in grouped:
        labels = sorted(group["course"].unique())

        canonical = labels[0]

        for label in labels:
            alias_map[label] = canonical

    return alias_map


def normalize_course(course, alias_map):
    course = course.strip().upper()
    return alias_map.get(course, course)


def normalize_courses(course_list, alias_map):
    return {
        normalize_course(course, alias_map)
        for course in course_list
    }

def build_prerequisite_graph(prereqs, alias_map=None):
    prereqs = prereqs.copy()

    prereqs["course"] = (
        prereqs["subject"].str.upper().str.strip()
        + " "
        + prereqs["course_number"].astype(str).str.upper().str.strip()
    )

    prereqs["prerequisite"] = (
        prereqs["prereq_subject"].str.upper().str.strip()
        + " "
        + prereqs["prereq_number"].astype(str).str.upper().str.strip()
    )

    if alias_map is not None:
        prereqs["course"] = prereqs["course"].apply(
            lambda x: normalize_course(x, alias_map)
        )

        prereqs["prerequisite"] = prereqs["prerequisite"].apply(
            lambda x: normalize_course(x, alias_map)
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

def filter_eligible_courses(
    courses,
    completed_courses,
    graph,
    alias_map=None,
    grouped_prereqs=None,
):
    courses = add_course_labels(courses)

    if alias_map is None:
        alias_map = {}

    completed_normalized = normalize_courses(
        completed_courses,
        alias_map
    )

    courses["normalized_course"] = courses["course"].apply(
        lambda x: normalize_course(x, alias_map)
    )

    courses = courses[
        ~courses["normalized_course"].isin(completed_normalized)
    ].copy()

    if grouped_prereqs is not None:
        courses["eligible"] = courses["normalized_course"].apply(
            lambda course: is_eligible_grouped(
                course,
                completed_normalized,
                grouped_prereqs,
                alias_map
            )
        )

        return courses[courses["eligible"]].copy()

    courses["missing_prereqs"] = courses[
        "normalized_course"
    ].apply(
        lambda course: missing_prerequisites(
            course,
            completed_normalized,
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

def is_eligible_grouped(
    course,
    completed_courses,
    grouped_prereqs,
    alias_map=None
):
    if alias_map is None:
        alias_map = {}

    course = normalize_course(course, alias_map)

    course_prereqs = grouped_prereqs.copy()

    course_prereqs["normalized_course"] = (
        course_prereqs["course"]
        .apply(lambda x: normalize_course(x, alias_map))
    )

    course_prereqs["normalized_prerequisite"] = (
        course_prereqs["prerequisite"]
        .apply(lambda x: normalize_course(x, alias_map))
    )

    course_prereqs = course_prereqs[
        course_prereqs["normalized_course"] == course
    ]

    if course_prereqs.empty:
        return True

    for _, group in course_prereqs.groupby("requirement_group"):

        group_satisfied = False

        for _, option in group.groupby("option_id"):

            required_courses = set(
                option["normalized_prerequisite"]
            )

            if required_courses.issubset(completed_courses):
                group_satisfied = True
                break

        if not group_satisfied:
            return False

    return True

import re


def get_course_number_value(course_number):
    """
    Extract the numeric portion of a Berkeley course number.

    Examples:
        C100 -> 100
        61A  -> 61
        C8   -> 8
        142A -> 142
    """
    match = re.search(r"\d+", str(course_number))

    if match:
        return int(match.group())

    return None


def apply_progression_filter(
    courses,
    completed_courses,
    alias_map=None
):
    """
    If a student has completed an upper-division course (100+)
    in a subject, remove lower-division recommendations from
    that same subject.
    """

    courses = courses.copy()

    if alias_map is None:
        alias_map = {}

    # Make labels if they don't already exist
    if "course" not in courses.columns:
        courses = add_course_labels(courses)

    completed_normalized = normalize_courses(
        completed_courses,
        alias_map
    )

    advanced_subjects = set()

    for completed in completed_normalized:
        parts = completed.split(" ", 1)

        if len(parts) != 2:
            continue

        subject, number = parts

        numeric = get_course_number_value(number)

        if numeric is not None and numeric >= 100:
            advanced_subjects.add(subject)

    def should_keep(row):
        subject = row["subject"]
        numeric = get_course_number_value(
            row["course_number"]
        )

        if (
            subject in advanced_subjects
            and numeric is not None
            and numeric < 100
        ):
            return False

        return True

    return courses[
        courses.apply(should_keep, axis=1)
    ].copy()