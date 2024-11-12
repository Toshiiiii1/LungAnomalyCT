from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'history.db')}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class RecognitionHistory(Base):
    __tablename__ = "recognition_history"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, index=True)
    file1_name = Column(String, index=True)
    file2_name = Column(String, index=True)
    extracted_images = Column(String)  # Danh sách URL ảnh trích xuất dưới dạng JSON
    predicted_image = Column(String)   # URL ảnh dự đoán
    slice_index = Column(Integer)
    timestamp = Column(DateTime, default=datetime.now())

Base.metadata.create_all(bind=engine)
