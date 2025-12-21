import asyncio
import pandas as pd
from sqlalchemy import select

from app.db import AsyncSessionLocal, Vitals
from app.risk import compute_risk


def risk_to_label(score: float) -> str:
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    else:
        return "high"


async def fetch_vitals():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Vitals))
        return result.scalars().all()


async def build_dataset():
    vitals = await fetch_vitals()
    rows = []

    for v in vitals:
        risk_score = compute_risk(
            heart_rate=v.heart_rate,
            spo2=v.spo2,
            temp_c=v.temp_c,
            steps=v.steps,
        )

        rows.append({
            "device_id": v.device_id,
            "user_id": v.user_id,
            "timestamp": v.timestamp,
            "heart_rate": v.heart_rate,
            "spo2": v.spo2,
            "temp_c": v.temp_c,
            "steps": v.steps,
            "risk_score": risk_score,
            "risk_label": risk_to_label(risk_score),
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = asyncio.run(build_dataset())

    print("Dataset preview:")
    print(df.head())

    import os

OUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "training_data.csv"
)

df.to_csv(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH}")
