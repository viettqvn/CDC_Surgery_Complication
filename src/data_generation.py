"""Generate synthetic pre-operative surgery data with a medically-plausible
hidden risk score, used as a stand-in for real EMR data.

Ported from CDC_CCI_DataGeneration.ipynb (originally a Colab notebook writing
to Google Drive) into a reproducible, parameterized script.
"""

import argparse
import logging

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_surgery_data(n_samples: int = 5000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = np.clip(rng.normal(loc=55, scale=15, size=n_samples), 18, 90).astype(int)
    bmi = np.round(np.clip(rng.normal(loc=23.5, scale=3.5, size=n_samples), 15, 40), 1)

    asa_choices = ["I", "II", "III", "IV", "V"]
    asa_probs = [0.35, 0.45, 0.15, 0.04, 0.01]
    asa_score = rng.choice(asa_choices, size=n_samples, p=asa_probs)

    diabetes = rng.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    htn = rng.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    surgery_type = rng.choice(["Chương trình", "Cấp cứu"], size=n_samples, p=[0.85, 0.15])

    wbc = np.round(np.clip(rng.normal(loc=8.0, scale=3.5, size=n_samples), 3.0, 25.0), 1)
    albumin = np.round(np.clip(rng.normal(loc=3.8, scale=0.6, size=n_samples), 1.5, 5.5), 1)

    asa_numeric = np.array([{"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}[x] for x in asa_score])
    is_emergency = np.where(surgery_type == "Cấp cứu", 1, 0)

    hidden_risk_score = (
        (age * 0.05)
        + (bmi * 0.1)
        + (asa_numeric * 2.5)
        + (is_emergency * 3.0)
        + (diabetes * 1.5)
        + (htn * 1.0)
        - (albumin * 2.0)
        + (wbc * 0.1)
        + rng.normal(0, 2, n_samples)
    )

    threshold = np.percentile(hidden_risk_score, 90)
    high_risk_flag = np.where(hidden_risk_score >= threshold, 1, 0)

    df = pd.DataFrame(
        {
            "Visit_ID": [f"SURG_{str(i).zfill(5)}" for i in range(1, n_samples + 1)],
            "Age": age,
            "BMI": bmi,
            "ASA_Score": asa_score,
            "Has_Diabetes": diabetes,
            "Has_HTN": htn,
            "Surgery_Type": surgery_type,
            "PreOp_WBC": wbc,
            "PreOp_Albumin": albumin,
            "High_Risk_Flag": high_risk_flag,
        }
    )

    missing_wbc_idx = rng.choice(df.index, size=int(n_samples * 0.10), replace=False)
    missing_alb_idx = rng.choice(df.index, size=int(n_samples * 0.15), replace=False)
    df.loc[missing_wbc_idx, "PreOp_WBC"] = np.nan
    df.loc[missing_alb_idx, "PreOp_Albumin"] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=str, default=str(RAW_DATA_PATH))
    args = parser.parse_args()

    logger.info("Generating %d synthetic surgery records...", args.n_samples)
    df = generate_surgery_data(n_samples=args.n_samples, random_state=args.random_state)
    df.to_csv(args.output, index=False)

    logger.info("Saved to %s", args.output)
    logger.info(
        "High_Risk_Flag positive rate: %.1f%% (%d / %d)",
        df["High_Risk_Flag"].mean() * 100,
        df["High_Risk_Flag"].sum(),
        len(df),
    )


if __name__ == "__main__":
    main()
