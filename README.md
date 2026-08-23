# CalCourse

CalCourse is a personalized course planning system for UC Berkeley students. It helps students identify which courses best fit their academic background, interests, career goals, and practical constraints.

Rather than only helping users find classes, CalCourse focuses on the decision-making problem behind course selection: **what should a student take next, and why?**

## Overview

CalCourse builds a student-specific recommendation pipeline using academic and course-level information such as:

* completed coursework
* prerequisites and eligibility
* academic interests
* career goals
* course descriptions and subject areas
* semester availability
* workload and unit constraints

The goal is to generate **personalized course recommendations** and eventually **feasible semester plans** while making each recommendation explainable.

## Data Pipeline

Course and class information is collected from Berkeley course data sources and transformed into structured datasets for analysis and modeling.

The current pipeline retrieves Fall 2026 data through BerkeleyTime's GraphQL interface and separates course-level metadata from semester-specific class records.

The Fall 2026 dataset contains:

* 15,573 class records
* 4,287 unique courses
* 2,119 courses in the final undergraduate recommendation pool

## Prerequisite and Eligibility Modeling

Course requirements are stored as semi-structured text, so CalCourse includes a custom prerequisite parsing pipeline that:

* extracts explicit course references from requirement text
* normalizes common subject-name variations
* handles shorthand references such as `MATH 53, 54, 55`
* preserves non-course constraints such as instructor consent, GPA, standing, and auditions

Parsed prerequisite relationships are modeled as a directed graph using NetworkX.

This allows CalCourse to identify:

* prerequisites for a target course
* courses unlocked by completed coursework
* missing prerequisites
* whether a student satisfies parsed course requirements

Eligibility filtering is then applied before recommendation ranking so the system prioritizes courses a student can realistically take.

## Recommendation System

CalCourse is being developed as a ranking system rather than a simple course search.

```text
Student Profile
      ↓
Academic History + Interests + Career Goals
      ↓
Prerequisite & Eligibility Filtering
      ↓
Course Relevance Scoring
      ↓
Personalized Ranking
      ↓
Schedule Constraints
      ↓
Recommended Courses / Semester Plan
```

The first recommendation baseline will use content-based ranking with TF-IDF and cosine similarity.

Future ranking models will be evaluated against baseline strategies using metrics such as **Recall@K** and **NDCG@K**.

## Tech Stack

* Python
* pandas
* GraphQL
* NetworkX
* scikit-learn
* SQL
* Git/GitHub

## Project Structure

```text
CalCourse/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_prerequisite_parsing.ipynb
│   ├── 03_prerequisite_graph.ipynb
│   └── 04_eligibility_filtering.ipynb
├── src/
├── README.md
└── requirements.txt
```

## Status

Currently in development.

**Completed**

* GraphQL data ingestion pipeline
* Fall 2026 course and class dataset construction
* Exploratory data analysis and recommendation-pool filtering
* Prerequisite parsing and normalization
* Directed prerequisite graph construction
* Student-specific eligibility filtering

**Next**

* TF-IDF baseline recommender
* Personalized course ranking
* Recommendation evaluation
* Schedule optimization
* Interactive course planning interface
