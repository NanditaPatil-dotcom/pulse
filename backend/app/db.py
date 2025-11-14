import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class PulseRecord(Base):
    __tablename__ = "pulse_readings"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String)
    heart_rate = Column(Integer)
    spo2 = Column(Float)
    timestamp = Column(DateTime)