# CalCourse

CalCourse is a personalized course planning system for UC Berkeley students. It helps students identify which courses best fit their academic background, interests, career goals, and practical constraints.

Rather than only helping users find classes, CalCourse focuses on the decision-making problem behind course selection: **what should a student take next, and why?**

## Overview

CalCourse builds a student-specific ranking of courses using academic and course-level information such as:

* completed coursework
* prerequisites and eligibility
* academic interests
* career goals
* course descriptions and subject areas
* semester availability
* workload and unit constraints

The long-term goal is to generate both **personalized course recommendations** and **feasible semester plans**, while explaining the reasoning behind each recommendation.

## Data Pipeline

Course and class information is collected from Berkeley course data sources and transformed into structured datasets for analysis and modeling.

The current pipeline retrieves Fall 2026 data through BerkeleyTime's GraphQL interface and separates the data into course-level and class-level records.

These datasets provide the foundation for prerequisite analysis, feature engineering, recommendation modeling, and schedule optimization.

## Recommendation System

CalCourse is being developed as a ranking problem rather than a simple course search.

The recommendation system will combine signals such as:

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

Recommendations will be evaluated using ranking metrics such as **Recall@K** and **NDCG@K** and compared against baseline recommendation strategies.

## Tech Stack

* Python
* pandas
* SQL
* GraphQL
* scikit-learn
* Git/GitHub

## Project Structure

```text
CalCourse/
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── notebooks/
├── README.md
└── requirements.txt
```

## Status

Currently in development.

**Completed**

* Course data source exploration
* GraphQL data ingestion pipeline
* Fall 2026 course and class dataset construction

**Next**

* Data cleaning and exploratory analysis
* Prerequisite relationship modeling
* Baseline recommendation system
* Personalized ranking model
* Recommendation evaluation
* Schedule optimization
* Interactive course planning interface
