# CalCourse

[![Tests](https://github.com/keskaory/CalCourse/actions/workflows/tests.yml/badge.svg)](https://github.com/keskaory/CalCourse/actions/workflows/tests.yml)

CalCourse is a personalized UC Berkeley course recommender that helps students decide what to take next based on their coursework, interests, and subject preferences.

## Quickstart

```bash
git clone https://github.com/keskaory/CalCourse.git
cd CalCourse

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m streamlit run app/app.py --server.fileWatcherType none
```

The app uses the Fall 2026 dataset already committed under `data/processed/` — no scraping or preprocessing required to run it. See [Data Pipeline](#data-pipeline) if you want to regenerate that data from scratch.

## How It Works

```text
Student Profile
      ↓
Completed Courses + Interests + Preferred Subjects
      ↓
Prerequisite & Eligibility Filtering
      ↓
Progression Filtering
      ↓
Semantic Similarity Scoring (sentence-transformers)
      ↓
Hybrid Ranking (semantic score + subject-preference fit)
      ↓
Top 10 Recommended Courses
```

### Eligibility

Course requirements are stored as semi-structured text, so CalCourse includes a custom prerequisite parsing pipeline that:

* extracts explicit course references from requirement text
* normalizes common subject-name variations, including cross-listed courses (e.g. `DATA C100` / `COMPSCI C100`) via a title/description-based alias map
* handles shorthand references such as `MATH 53, 54, 55`
* preserves requirement structure — including **AND/OR groups** (e.g. "STAT 134 or EECS 126, and MATH 54") rather than flattening every prerequisite into a single list

Parsed prerequisite relationships are modeled as a directed graph using NetworkX. This lets CalCourse determine, for a given student:

* which prerequisites a target course requires
* which courses are unlocked by their completed coursework
* which requirement groups are still unsatisfied
* whether they're eligible for a course, honoring OR-alternatives and multi-course AND options

A progression filter also removes lower-division courses in a subject once the student has completed an upper-division course in that same subject, so recommendations stay level-appropriate.

Eligibility filtering runs before ranking, so recommendations satisfy the explicit course prerequisites captured by the V1 parser. (Non-course constraints — instructor consent, GPA, standing, equivalent preparation — are parsed and preserved but not yet enforced.)

### Semantic Matching

Course descriptions and student interests are embedded with `sentence-transformers/all-MiniLM-L6-v2`; relevance is scored by cosine similarity between a student's interest text and each eligible course's description.

Course embeddings are precomputed offline (`course_embeddings_fall_2026.npy` + an index CSV) because course descriptions are static for a given semester snapshot — this avoids re-encoding thousands of course descriptions on every request, so the app only has to embed the student's interest text at query time.

### Hybrid Ranking

The final ranking blends semantic relevance with a subject-preference signal:

```text
final_score = 0.60 * semantic_score + 0.40 * subject_fit
```

## Evaluation

Three ranking approaches are compared in [`notebooks/06_recommender_evaluation.ipynb`](notebooks/06_recommender_evaluation.ipynb) against 10 hand-labeled student profiles, using **Recall@10** and **NDCG@10**:

| Method | Recall@10 | NDCG@10 |
| --- | --- | --- |
| TF-IDF | 0.35 | 0.32 |
| Semantic | 0.52 | 0.49 |
| **Hybrid** | **0.66** | **0.61** |

The hybrid ranker produced the strongest overall results. Its semantic/subject weighting was selected using **leave-one-profile-out cross-validation**: for each held-out profile, the weight was chosen using only the other nine profiles, then scored on the one it never saw. All folds selected a semantic weight of `0.60` — the held-out cross-validated score (0.664 Recall@10, 0.607 NDCG@10) matches the in-sample hybrid result above, indicating a stable optimum rather than an artifact of tuning against the evaluation set.

**Caveat:** this is a small (10-profile), hand-labeled evaluation set authored by the same person who built the ranker. Treat the results as a directional, controlled V1 benchmark — not a claim about real student outcomes.

## Data Pipeline

Course and class information is collected from Berkeley course data sources and transformed into structured datasets for analysis and modeling.

The pipeline (`src/fetch_berkeley.py`) retrieves Fall 2026 data through BerkeleyTime's GraphQL interface and separates course-level metadata from semester-specific class records. The processed Fall 2026 dataset contains:

* 15,573 class records
* 4,287 unique courses
* 2,077 courses in the final undergraduate recommendation pool

Raw scrape output (`data/raw/`, ~15MB) is excluded from version control; `data/processed/` — including the precomputed course embeddings — is committed so the app runs without rerunning the pipeline.

## Testing / CI

```bash
pytest -q
```

Tests cover the parts of the system most likely to fail silently: OR/AND prerequisite groups, cross-listed course exclusion, the progression filter, and hybrid ranking order. GitHub Actions runs the suite on every push and pull request to `main`.

## Tech Stack

* Python
* pandas / numpy
* GraphQL (BerkeleyTime API)
* NetworkX
* scikit-learn
* sentence-transformers
* Streamlit
* pytest / GitHub Actions

## Project Structure

```text
CalCourse/
├── .github/workflows/     # CI (pytest on push/PR)
├── app/
│   └── app.py             # Streamlit interface
├── data/
│   ├── raw/                # gitignored — regenerate via src/fetch_berkeley.py
│   └── processed/          # committed — what the app reads
│       ├── recommendable_courses_fall_2026.csv
│       ├── prerequisites_fall_2026.csv
│       ├── prerequisites_grouped_fall_2026.csv
│       ├── course_embeddings_fall_2026.npy      # precomputed sentence-transformer embeddings
│       └── course_embedding_index_fall_2026.csv # course ↔ embedding row mapping
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_prerequisite_parsing.ipynb
│   ├── 03_prerequisite_graph.ipynb
│   ├── 04_eligibility_filtering.ipynb
│   ├── 05_baseline_recommender.ipynb
│   └── 06_recommender_evaluation.ipynb
├── src/
│   ├── fetch_berkeley.py  # data ingestion
│   ├── recommender.py     # eligibility filtering + ranking
│   └── run_recommender.py
├── tests/
│   └── test_recommender.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Limitations

* Recommends individual courses for a single term — no scheduling, unit constraints, or multi-semester plan generation yet.
* Ranking uses stated interests and subject preferences only; there's no career-goal-specific signal.
* Non-course eligibility constraints (instructor consent, GPA, standing) are parsed but not enforced.
* Evaluation is a small, hand-labeled benchmark rather than real usage data.

## License

[MIT](LICENSE)
