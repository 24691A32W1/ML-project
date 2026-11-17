"""Helper functions: performance classification and support recommendations."""
import numpy as np
import pandas as pd

from .config import CATEGORY_BANDS


def categorize_marks(marks):
    """Map a mark (0-100) to a performance category."""
    m = float(np.clip(marks, 0, 100))
    for name, low, high in CATEGORY_BANDS:
        if low <= m < high:
            return name
    return CATEGORY_BANDS[-1][0]


def categorize_series(series: pd.Series) -> pd.Series:
    return series.apply(categorize_marks)


def support_recommendations(row: dict, predicted: float) -> list:
    """Rule-based, human-readable advice using the student's own inputs."""
    tips = []
    if row["Attendance_Percentage"] < 75:
        tips.append("Attendance is below 75%. Regular class attendance is strongly linked to better marks.")
    if row["Study_Hours_Per_Day"] < 2:
        tips.append("Study time is low. Try a structured plan of at least 2-3 focused hours per day.")
    if row["Assignments_Completed"] < 7:
        tips.append("Several assignments are incomplete. Completing every assignment builds exam readiness.")
    if row["Internal_Exam_Marks"] < 50:
        tips.append("Internal exam marks are weak. Revisit the topics tested and attend doubt-clearing sessions.")
    if row["Class_Participation_Score"] < 5:
        tips.append("Participation is low. Asking questions and joining discussions improves understanding.")
    if row["Sleep_Hours"] < 6 or row["Sleep_Hours"] > 9:
        tips.append("Sleep is outside the healthy 6-9 hour range, which can hurt concentration and memory.")
    if predicted < 40 and not tips:
        tips.append("Predicted marks are low. Arrange mentoring or remedial classes.")
    if not tips:
        tips.append("No major risk factors found. Keep up the current routine.")
    return tips
