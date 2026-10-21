# database.py — SQLite setup
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///drugmirror.db")
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class QueryLog(Base):
    __tablename__ = "query_log"
    id = Column(Integer, primary_key=True)
    drug_name = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sections_returned = Column(String)

class PredictionCache(Base):
    __tablename__ = "prediction_cache"
    id = Column(Integer, primary_key=True)
    drug_name = Column(String, unique=True)
    profile_json = Column(String)
    cached_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)