"""Central configuration shared by all modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "Final_Exam_Marks"
ID_COL = "Student_ID"

NUMERIC_FEATURES = [
    "Age",
    "Attendance_Percentage",
    "Study_Hours_Per_Day",
    "Previous_Semester_Marks",
    "Assignment_Score",
    "Internal_Exam_Marks",
    "Assignments_Completed",
    "Class_Participation_Score",
    "Sleep_Hours",
]
CATEGORICAL_FEATURES = ["Gender", "Extracurricular_Participation"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

GENDER_OPTIONS = ["Male", "Female"]
EXTRA_OPTIONS = ["Not Involved", "Occasional", "Regular"]

# Performance categories (lower bound inclusive, upper bound exclusive).
# NOTE: the original brief was cut off after "Average: 40-"; the standard
# completion below was assumed: Average 40-54.99, Needs Improvement < 40.
CATEGORY_BANDS = [
    ("Excellent", 85.0, 100.01),
    ("Very Good", 70.0, 85.0),
    ("Good", 55.0, 70.0),
    ("Average", 40.0, 55.0),
    ("Needs Improvement", 0.0, 40.0),
]
CATEGORY_ORDER = [c[0] for c in CATEGORY_BANDS]
AT_RISK_CATEGORIES = ["Needs Improvement", "Average"]
