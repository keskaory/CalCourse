import requests
import json
import pandas as pd
from pathlib import Path

url = "https://berkeleytime.com/api/graphql"

query = """
{
  catalog(year: 2026, semester: Fall) {
    year
    semester
    termId
    sessionId
    courseId
    subject
    courseNumber
    number
    title
    description
    unitsMin
    unitsMax
    gradingBasis
    finalExam
    viewCount

    course {
      courseId
      subject
      number
      title
      description
      requirements
      academicCareer
      academicOrganizationName
      departmentNicknames
      typicallyOffered
    }
  }
}
"""

response = requests.post(url, json={"query": query})
data = response.json()

classes = data["data"]["catalog"]

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save untouched raw API response
with open("data/raw/berkeleytime_fall_2026.json", "w") as f:
    json.dump(classes, f, indent=2)

class_rows = []

for c in classes:
    class_rows.append({
        "course_id": c["courseId"],
        "year": c["year"],
        "semester": c["semester"],
        "term_id": c["termId"],
        "session_id": c["sessionId"],
        "subject": c["subject"],
        "course_number": c["courseNumber"],
        "class_number": c["number"],
        "units_min": c["unitsMin"],
        "units_max": c["unitsMax"],
        "grading_basis": c["gradingBasis"],
        "final_exam": c["finalExam"],
        "view_count": c["viewCount"]
    })

classes_df = pd.DataFrame(class_rows)

course_rows = []

for c in classes:
    course = c["course"]

    course_rows.append({
        "course_id": course["courseId"],
        "subject": course["subject"],
        "course_number": course["number"],
        "title": course["title"],
        "description": course["description"],
        "requirements": course["requirements"],
        "academic_career": course["academicCareer"],
        "department": course["academicOrganizationName"],
        "department_nicknames": course["departmentNicknames"],
        "typically_offered": course["typicallyOffered"]
    })

courses_df = pd.DataFrame(course_rows)

# Same course appears in multiple class records,
# so keep one row per unique course.
courses_df = courses_df.drop_duplicates(subset="course_id")

classes_df.to_csv(
    "data/processed/classes_fall_2026.csv",
    index=False
)

courses_df.to_csv(
    "data/processed/courses_fall_2026.csv",
    index=False
)

print("Class records:", classes_df.shape)
print("Unique courses:", courses_df.shape)

print("\nClasses:")
print(classes_df.head())

print("\nCourses:")
print(courses_df.head())