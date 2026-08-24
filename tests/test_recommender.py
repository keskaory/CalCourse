import numpy as np
import pandas as pd
import networkx as nx

from src.recommender import (
    build_alias_map,
    build_prerequisite_graph,
    filter_eligible_courses,
    is_eligible_grouped,
    apply_progression_filter,
    rank_hybrid_courses,
)

# completed courses should never be recommended

def test_completed_course_not_recommended():
    courses = pd.DataFrame([
        {
            "subject": "DATA",
            "course_number": "C8",
            "title": "Foundations of Data Science",
            "description": "Introduction to data science.",
        },
        {
            "subject": "STAT",
            "course_number": "133",
            "title": "Concepts in Computing with Data",
            "description": "Statistical computing.",
        },
    ])

    graph = nx.DiGraph()

    eligible = filter_eligible_courses(
        courses,
        {"DATA C8"},
        graph,
        alias_map={},
        grouped_prereqs=None,
    )

    assert "DATA C8" not in set(eligible["course"])

# cross-listed equivalents should also be excluded

def test_crosslisted_course_not_recommended():
    courses = pd.DataFrame([
        {
            "subject": "DATA",
            "course_number": "C8",
            "title": "Foundations of Data Science",
            "description": "Introduction to data science.",
        },
        {
            "subject": "STAT",
            "course_number": "C8",
            "title": "Foundations of Data Science",
            "description": "Introduction to data science.",
        },
        {
            "subject": "STAT",
            "course_number": "133",
            "title": "Concepts in Computing with Data",
            "description": "Statistical computing.",
        },
    ])

    alias_map = build_alias_map(courses)
    graph = nx.DiGraph()

    eligible = filter_eligible_courses(
        courses,
        {"DATA C8"},
        graph,
        alias_map=alias_map,
        grouped_prereqs=None,
    )

    recommendations = set(eligible["course"])

    assert "DATA C8" not in recommendations
    assert "STAT C8" not in recommendations

# OR prerequisite groups should work

def test_or_prerequisite_group():
    grouped_prereqs = pd.DataFrame([
        {
            "course": "DATA C102",
            "requirement_group": 1,
            "option_id": 1,
            "prerequisite": "DATA C100",
        },
        {
            "course": "DATA C102",
            "requirement_group": 2,
            "option_id": 1,
            "prerequisite": "EECS 126",
        },
        {
            "course": "DATA C102",
            "requirement_group": 2,
            "option_id": 2,
            "prerequisite": "STAT 134",
        },
    ])

    completed = {
        "DATA C100",
        "STAT 134",
    }

    assert is_eligible_grouped(
        "DATA C102",
        completed,
        grouped_prereqs,
        alias_map={},
    )


# missing a required group should block the course

def test_missing_prerequisite_blocks_course():
    grouped_prereqs = pd.DataFrame([
        {
            "course": "DATA C102",
            "requirement_group": 1,
            "option_id": 1,
            "prerequisite": "DATA C100",
        },
        {
            "course": "DATA C102",
            "requirement_group": 2,
            "option_id": 1,
            "prerequisite": "STAT 134",
        },
        {
            "course": "DATA C102",
            "requirement_group": 3,
            "option_id": 1,
            "prerequisite": "MATH 54",
        },
    ])

    # missing the math requirement
    completed = {
        "DATA C100",
        "STAT 134",
    }

    assert not is_eligible_grouped(
        "DATA C102",
        completed,
        grouped_prereqs,
        alias_map={},
    )


# multi-course AND option should require both courses


def test_multi_course_option_requires_all_courses():
    grouped_prereqs = pd.DataFrame([
        {
            "course": "DATA C102",
            "requirement_group": 1,
            "option_id": 1,
            "prerequisite": "ELENG 16A",
        },
        {
            "course": "DATA C102",
            "requirement_group": 1,
            "option_id": 1,
            "prerequisite": "ELENG 16B",
        },
    ])

    incomplete = {"ELENG 16A"}

    complete = {
        "ELENG 16A",
        "ELENG 16B",
    }

    assert not is_eligible_grouped(
        "DATA C102",
        incomplete,
        grouped_prereqs,
        alias_map={},
    )

    assert is_eligible_grouped(
        "DATA C102",
        complete,
        grouped_prereqs,
        alias_map={},
    )

# progression filter should remove intro courses

def test_progression_filter_removes_lower_division_courses():
    courses = pd.DataFrame([
        {
            "subject": "DATA",
            "course_number": "C6",
            "title": "Intro Data Course",
            "description": "",
        },
        {
            "subject": "DATA",
            "course_number": "145",
            "title": "Evidence and Uncertainty",
            "description": "",
        },
        {
            "subject": "STAT",
            "course_number": "C8",
            "title": "Foundations of Data Science",
            "description": "",
        },
        {
            "subject": "STAT",
            "course_number": "159",
            "title": "Statistical Data Science",
            "description": "",
        },
    ])

    filtered = apply_progression_filter(
        courses,
        completed_courses={
            "DATA C100",
            "STAT 134",
        },
        alias_map={},
    )

    remaining = set(filtered["course"])

    assert "DATA C6" not in remaining
    assert "STAT C8" not in remaining

    assert "DATA 145" in remaining
    assert "STAT 159" in remaining


# hybrid ranker should sort by final score

class DummyModel:
    def encode(self, texts):
        return np.array([[1.0, 0.0]])


def test_hybrid_ranker_sorts_descending():
    courses = pd.DataFrame([
        {
            "course": "DATA 100",
            "subject": "DATA",
            "title": "Data Science",
        },
        {
            "course": "MUSIC 100",
            "subject": "MUSIC",
            "title": "Music",
        },
    ])

    # first course is perfectly aligned with profile embedding.
    course_embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    ranked = rank_hybrid_courses(
        courses,
        interests="data science",
        preferred_subjects=[],
        model=DummyModel(),
        course_embeddings=course_embeddings,
    )

    assert ranked.iloc[0]["course"] == "DATA 100"

    scores = ranked["final_score"].tolist()

    assert scores == sorted(scores, reverse=True)