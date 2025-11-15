import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, func
from datetime import datetime, timedelta

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession)

Base = declarative_base()

class Vitals(Base):
    __tablename__ = "vitals"

    id = Column(Integer, primary_key=True)
    device_id = Column(String)
    user_id = Column(String)
    timestamp = Column(DateTime)
    heart_rate = Column(Integer)
    spo2 = Column(Integer)
    temp_c = Column(Float, nullable=True)
    steps = Column(Integer, nullable=True)

class Predictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    vitals_id = Column(Integer)
    prediction_label = Column(String)
    probability = Column(Float)
    shap_json = Column(JSON)
    model_version = Column(String)
    created_at = Column(DateTime, default=func.now())

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_session():
    async with AsyncSessionLocal() as session:
        yield session

async def insert_vital(vital_obj):
    async with AsyncSessionLocal() as session:
        v = Vitals(
            device_id=vital_obj.device_id,
            user_id=vital_obj.user_id,
            timestamp=vital_obj.timestamp if vital_obj.timestamp else datetime.utcnow(),
            heart_rate=vital_obj.heart_rate,
            spo2=vital_obj.spo2,
            temp_c=vital_obj.temp_c,
            steps=vital_obj.steps,
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
    # placeholder - return simple counts and last prediction timestamp
    async with AsyncSessionLocal() as session:
        q = select(func.count(Predictions.id))
        total = (await session.execute(q)).scalar()
        return {"total_predictions": total}