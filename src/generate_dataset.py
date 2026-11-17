"""
Synthetic dataset generator.

IMPORTANT: The data produced here is SYNTHETIC. It is simulated with a
documented statistical model and is NOT real student data. It exists to
demonstrate the ML workflow.

How records are generated
-------------------------
Each student gets a hidden "academic ability" and "diligence" factor.
Behavioural inputs (attendance, study hours, assignments, participation) are
generated from these factors plus noise. Previous marks, assignment score and
internal marks follow from ability/diligence. Final exam marks are a weighted
combination of the inputs plus random noise, so the relationship is learnable
but not perfect. About 1% of cells in a few columns are deliberately blanked to
demonstrate missing-value handling.

Run:  python -m src.generate_dataset
"""
import numpy as np
import pandas as pd

from .config import DATA_PATH, RANDOM_STATE

N_STUDENTS = 1200


def generate(n: int = N_STUDENTS, seed: int = RANDOM_STATE, missing_rate: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ability = rng.normal(0, 1, n)
    diligence = 0.35 * ability + rng.normal(0, 1, n) * np.sqrt(1 - 0.35 ** 2)

    age = rng.choice([18, 19, 20, 21, 22, 23], size=n, p=[0.12, 0.28, 0.30, 0.20, 0.08, 0.02])
    gender = rng.choice(["Male", "Female"], size=n, p=[0.54, 0.46])
    extra = rng.choice(["Not Involved", "Occasional", "Regular"], size=n, p=[0.35, 0.40, 0.25])
    extra_effect = pd.Series(extra).map({"Not Involved": 0.0, "Occasional": 0.3, "Regular": 0.0}).to_numpy()

    attendance = np.clip(77 + 9 * diligence + 3 * ability + rng.normal(0, 6, n), 35, 100)
    study = np.clip(3.0 + 1.0 * diligence + 0.3 * ability + rng.normal(0, 0.8, n), 0.3, 9.0)
    sleep = np.clip(rng.normal(7.0, 1.0, n) - 0.15 * (study - 3), 4.0, 10.0)

    prev = np.clip(62 + 14 * ability + 3 * diligence + rng.normal(0, 6, n), 15, 100)
    assignment_score = np.clip(66 + 9 * diligence + 7 * ability + rng.normal(0, 7, n), 20, 100)
    internal = np.clip(60 + 11 * ability + 5 * diligence + 0.15 * (attendance - 75) + rng.normal(0, 7, n), 15, 100)

    completed = np.clip(np.round(7.8 + 1.4 * diligence + rng.normal(0, 1.0, n)), 0, 10).astype(int)
    participation = np.clip(5.5 + 1.3 * diligence + 0.6 * ability + 0.4 * extra_effect + rng.normal(0, 1.2, n), 0, 10)

    sleep_penalty = 1.2 * np.abs(sleep - 7.5)
    final = (
        4.0
        + 0.26 * prev
        + 0.30 * internal
        + 0.10 * assignment_score
        + 0.11 * attendance
        + 1.4 * np.minimum(study, 6.0)
        + 0.5 * participation
        + 0.4 * completed
        - sleep_penalty
        + rng.normal(0, 4.0, n)
    )
    final = np.clip(final, 0, 100)

    df = pd.DataFrame({
        "Student_ID": [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)],
        "Age": age,
        "Gender": gender,
        "Attendance_Percentage": attendance.round(1),
        "Study_Hours_Per_Day": study.round(1),
        "Previous_Semester_Marks": prev.round(1),
        "Assignment_Score": assignment_score.round(1),
        "Internal_Exam_Marks": internal.round(1),
        "Assignments_Completed": completed,
        "Class_Participation_Score": participation.round(1),
        "Sleep_Hours": sleep.round(1),
        "Extracurricular_Participation": extra,
        "Final_Exam_Marks": final.round(1),
    })

    # Deliberately blank ~1% of a few columns (documented) to exercise imputation.
    for col in ["Attendance_Percentage", "Study_Hours_Per_Day", "Sleep_Hours",
                "Gender", "Extracurricular_Participation"]:
        idx = rng.choice(n, size=int(n * missing_rate), replace=False)
        df.loc[idx, col] = np.nan
    return df


def main():
    df = generate()
    assert not df.drop(columns=["Student_ID"]).duplicated().any(), "duplicate records generated"
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} synthetic records to {DATA_PATH}")
    print(df.describe().round(2).T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
