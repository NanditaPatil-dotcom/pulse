import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, func, select

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
print("DATABASE_URL =", repr(DATABASE_URL))


engine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class Vitals(Base):
    __tablename__ = "vitals"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    heart_rate = Column(Integer)
    spo2 = Column(Integer)
    temp_c = Column(Float, nullable=True)
    steps = Column(Integer, nullable=True)


class Predictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    vitals_id = Column(Integer, index=True)
    prediction_label = Column(String)
    probability = Column(Float)
    shap_json = Column(JSON)
    model_version = Column(String)
    created_at = Column(DateTime, default=func.now())



async def init_db():
    """Create tables (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session():
    """Dependency-style session generator if needed."""
    async with AsyncSessionLocal() as session:
        yield session


async def insert_vital(vital_obj):
    """
    Accept either a pydantic-like object with attributes
    or a dict. Returns inserted id.
    """

    if isinstance(vital_obj, dict):
        d = vital_obj
    else:

        d = {
            "device_id": getattr(vital_obj, "device_id", None),
            "user_id": getattr(vital_obj, "user_id", None),
            "timestamp": getattr(vital_obj, "timestamp", None),
            "heart_rate": getattr(vital_obj, "heart_rate", None),
            "spo2": getattr(vital_obj, "spo2", None),
            "temp_c": getattr(vital_obj, "temp_c", None),
            "steps": getattr(vital_obj, "steps", None),
        }

    async with AsyncSessionLocal() as session:
        v = Vitals(
            device_id=d.get("device_id"),
            user_id=d.get("user_id"),
            timestamp=d.get("timestamp") if d.get("timestamp") else datetime.utcnow(),
            heart_rate=int(d.get("heart_rate")) if d.get("heart_rate") is not None else None,
            spo2=int(d.get("spo2")) if d.get("spo2") is not None else None,
            temp_c=float(d.get("temp_c")) if d.get("temp_c") is not None else None,
            steps=int(d.get("steps")) if d.get("steps") is not None else None,
        )
        session.add(v)
        await session.commit()
        await session.refresh(v)
        return v.id


async def insert_prediction(vitals_id, label, prob, shap_json, version):
    async with AsyncSessionLocal() as session:
        p = Predictions(
            vitals_id=vitals_id,
            prediction_label=label,
            probability=prob,
            shap_json=shap_json,
            model_version=version,
        )
        session.add(p)
        await session.commit()
        await session.refresh(p)
        return p.id


async def fetch_latest_vital(user_id):
    async with AsyncSessionLocal() as session:
        q = select(Vitals).where(Vitals.user_id == user_id).order_by(Vitals.timestamp.desc()).limit(1)
        res = await session.execute(q)
        row = res.scalars().first()
        return row


async def fetch_history(user_id, hours=24):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    async with AsyncSessionLocal() as session:
        q = select(Vitals).where(Vitals.user_id == user_id, Vitals.timestamp >= cutoff).order_by(Vitals.timestamp)
        res = await session.execute(q)
        rows = res.scalars().all()
        return rows


async def fetch_metrics():
    async with AsyncSessionLocal() as session:
        q = select(func.count(Predictions.id))
        total = (await session.execute(q)).scalar()
        return {"total_predictions": int(total or 0)}
