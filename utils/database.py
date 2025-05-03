import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import json
import numpy as np

# Create SQLAlchemy engine and base
DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback to SQLite for local environments if PostgreSQL is not available
if not DATABASE_URL:
    print("DATABASE_URL not found, using SQLite database for local development")
    DATABASE_URL = "sqlite:///energy_anomaly.db"

engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class User(Base):
    """User model for storing user data"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    datasets = relationship("Dataset", back_populates="user")
    detection_results = relationship("DetectionResult", back_populates="user")
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active
        }

class Dataset(Base):
    """Dataset model for storing uploaded energy data"""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(200))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    row_count = Column(Integer)
    column_count = Column(Integer)
    file_path = Column(String(255))  # Path to actual CSV file
    data_summary = Column(Text)  # JSON string with data summary
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Relationships
    user = relationship("User", back_populates="datasets")
    energy_data = relationship("EnergyData", back_populates="dataset", cascade="all, delete-orphan")
    detection_results = relationship("DetectionResult", back_populates="dataset")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "file_path": self.file_path,
            "data_summary": json.loads(self.data_summary) if self.data_summary else None,
            "user_id": self.user_id
        }

class EnergyData(Base):
    """EnergyData model for storing individual energy consumption records"""
    __tablename__ = 'energy_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    consumption = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    meter_id = Column(String(50))
    location = Column(String(100))
    is_weekend = Column(Boolean)
    is_business_hours = Column(Boolean)
    hour = Column(Integer)
    day_of_week = Column(Integer)
    month = Column(Integer)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    
    # Relationships
    dataset = relationship("Dataset", back_populates="energy_data")
    anomaly_results = relationship("AnomalyResult", back_populates="energy_data", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "consumption": self.consumption,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "meter_id": self.meter_id,
            "location": self.location,
            "is_weekend": self.is_weekend,
            "is_business_hours": self.is_business_hours,
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "month": self.month,
            "dataset_id": self.dataset_id
        }

class DetectionModel(Base):
    """DetectionModel model for storing model information"""
    __tablename__ = 'detection_models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # isolation_forest, kmeans, autoencoder
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    parameters = Column(Text)  # JSON string with model parameters
    feature_importance = Column(Text)  # JSON string with feature importance (if available)
    model_path = Column(String(255))  # Path to saved model file
    
    # Relationships
    detection_results = relationship("DetectionResult", back_populates="model")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "parameters": json.loads(self.parameters) if self.parameters else None,
            "feature_importance": json.loads(self.feature_importance) if self.feature_importance else None,
            "model_path": self.model_path
        }

class DetectionResult(Base):
    """DetectionResult model for storing anomaly detection results"""
    __tablename__ = 'detection_results'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    model_id = Column(Integer, ForeignKey('detection_models.id'))
    user_id = Column(Integer, ForeignKey('users.id'))
    threshold = Column(Float)
    anomaly_count = Column(Integer)
    normal_count = Column(Integer)
    anomaly_percentage = Column(Float)
    selected_features = Column(Text)  # JSON string with selected features
    processing_time = Column(Float)  # Time taken to run detection in seconds
    result_metadata = Column(Text)  # Additional metadata as JSON
    
    # Relationships
    dataset = relationship("Dataset", back_populates="detection_results")
    model = relationship("DetectionModel", back_populates="detection_results")
    user = relationship("User", back_populates="detection_results")
    anomaly_results = relationship("AnomalyResult", back_populates="detection_result", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "user_id": self.user_id,
            "threshold": self.threshold,
            "anomaly_count": self.anomaly_count,
            "normal_count": self.normal_count,
            "anomaly_percentage": self.anomaly_percentage,
            "selected_features": json.loads(self.selected_features) if self.selected_features else None,
            "processing_time": self.processing_time,
            "metadata": json.loads(self.result_metadata) if self.result_metadata else None
        }

class AnomalyResult(Base):
    """AnomalyResult model for storing individual anomaly results"""
    __tablename__ = 'anomaly_results'
    
    id = Column(Integer, primary_key=True)
    energy_data_id = Column(Integer, ForeignKey('energy_data.id'))
    detection_result_id = Column(Integer, ForeignKey('detection_results.id'))
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float)
    
    # Relationships
    energy_data = relationship("EnergyData", back_populates="anomaly_results")
    detection_result = relationship("DetectionResult", back_populates="anomaly_results")
    
    def to_dict(self):
        return {
            "id": self.id,
            "energy_data_id": self.energy_data_id,
            "detection_result_id": self.detection_result_id,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score
        }

# Database helper functions
def initialize_database():
    """Create all tables in the database"""
    Base.metadata.create_all(engine)

def add_user(username, email, password_hash):
    """Add a new user to the database"""
    session = Session()
    try:
        user = User(
            username=username,
            email=email,
            password_hash=password_hash
        )
        session.add(user)
        session.commit()
        return user.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_user_by_username(username):
    """Get user by username"""
    session = Session()
    try:
        user = session.query(User).filter_by(username=username).first()
        return user
    finally:
        session.close()

def save_dataset(name, description, df, file_path, user_id):
    """Save uploaded dataset to database"""
    session = Session()
    try:
        # Create dataset record
        dataset = Dataset(
            name=name,
            description=description,
            row_count=len(df),
            column_count=len(df.columns),
            file_path=file_path,
            data_summary=json.dumps(get_data_summary(df)),
            user_id=user_id
        )
        session.add(dataset)
        session.flush()
        
        # Add energy data records
        for _, row in df.iterrows():
            energy_data = EnergyData(
                dataset_id=dataset.id,
                timestamp=pd.to_datetime(row.get('timestamp')),
                consumption=row.get('consumption'),
                temperature=row.get('temperature'),
                humidity=row.get('humidity'),
                meter_id=row.get('meter_id'),
                location=row.get('location'),
                hour=getattr(pd.to_datetime(row.get('timestamp')).dt, 'hour', None),
                day_of_week=getattr(pd.to_datetime(row.get('timestamp')).dt, 'dayofweek', None),
                month=getattr(pd.to_datetime(row.get('timestamp')).dt, 'month', None),
                is_weekend=getattr(pd.to_datetime(row.get('timestamp')).dt, 'dayofweek', 0) >= 5,
                is_business_hours=8 <= getattr(pd.to_datetime(row.get('timestamp')).dt, 'hour', 0) < 18
            )
            session.add(energy_data)
        
        session.commit()
        return dataset.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_data_summary(df):
    """Generate summary statistics for dataframe"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "numeric_columns": list(numeric_cols),
        "categorical_columns": list(df.select_dtypes(include=['object']).columns),
        "stats": {}
    }
    
    # Summary statistics for numeric columns
    for col in numeric_cols:
        summary['stats'][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
    
    return summary

def get_datasets_by_user(user_id):
    """Get all datasets for a user"""
    session = Session()
    try:
        datasets = session.query(Dataset).filter_by(user_id=user_id).all()
        return [dataset.to_dict() for dataset in datasets]
    finally:
        session.close()

def get_dataset_by_id(dataset_id):
    """Get dataset by ID"""
    session = Session()
    try:
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        return dataset
    finally:
        session.close()

def get_energy_data_by_dataset(dataset_id):
    """Get all energy data for a dataset"""
    session = Session()
    try:
        energy_data = session.query(EnergyData).filter_by(dataset_id=dataset_id).all()
        return energy_data
    finally:
        session.close()

def dataset_to_dataframe(dataset_id):
    """Convert dataset to pandas DataFrame"""
    energy_data = get_energy_data_by_dataset(dataset_id)
    data = [data.to_dict() for data in energy_data]
    return pd.DataFrame(data)

def save_detection_model(name, model_type, parameters, feature_importance=None, model_path=None):
    """Save detection model to database"""
    session = Session()
    try:
        model = DetectionModel(
            name=name,
            model_type=model_type,
            parameters=json.dumps(parameters),
            feature_importance=json.dumps(feature_importance) if feature_importance else None,
            model_path=model_path
        )
        session.add(model)
        session.commit()
        return model.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def save_detection_result(dataset_id, model_id, user_id, name, threshold, anomaly_count, 
                          normal_count, selected_features, processing_time, metadata=None):
    """Save detection result to database"""
    session = Session()
    try:
        anomaly_percentage = (anomaly_count / (anomaly_count + normal_count)) * 100 if (anomaly_count + normal_count) > 0 else 0
        
        result = DetectionResult(
            name=name,
            dataset_id=dataset_id,
            model_id=model_id,
            user_id=user_id,
            threshold=threshold,
            anomaly_count=anomaly_count,
            normal_count=normal_count,
            anomaly_percentage=anomaly_percentage,
            selected_features=json.dumps(selected_features),
            processing_time=processing_time,
            result_metadata=json.dumps(metadata) if metadata else None
        )
        session.add(result)
        session.flush()
        return result.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def save_anomaly_results(detection_result_id, energy_data_ids, is_anomalies, anomaly_scores):
    """Save individual anomaly results to database"""
    session = Session()
    try:
        for i, energy_data_id in enumerate(energy_data_ids):
            anomaly = AnomalyResult(
                energy_data_id=energy_data_id,
                detection_result_id=detection_result_id,
                is_anomaly=bool(is_anomalies[i]),
                anomaly_score=float(anomaly_scores[i])
            )
            session.add(anomaly)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_detection_results_by_user(user_id):
    """Get all detection results for a user"""
    session = Session()
    try:
        results = session.query(DetectionResult).filter_by(user_id=user_id).all()
        return [result.to_dict() for result in results]
    finally:
        session.close()

def get_detection_result_by_id(result_id):
    """Get detection result by ID"""
    session = Session()
    try:
        result = session.query(DetectionResult).filter_by(id=result_id).first()
        return result
    finally:
        session.close()

def get_anomaly_results_by_detection(detection_result_id):
    """Get all anomaly results for a detection result with energy data"""
    session = Session()
    try:
        # Join AnomalyResult with EnergyData to get full information
        query = session.query(AnomalyResult, EnergyData).\
            join(EnergyData, AnomalyResult.energy_data_id == EnergyData.id).\
            filter(AnomalyResult.detection_result_id == detection_result_id)
        
        results = []
        for anomaly, energy in query.all():
            result = energy.to_dict()
            result.update({
                "is_anomaly": anomaly.is_anomaly,
                "anomaly_score": anomaly.anomaly_score
            })
            results.append(result)
        
        return results
    finally:
        session.close()

def detection_results_to_dataframe(detection_result_id):
    """Convert detection results to pandas DataFrame"""
    anomaly_results = get_anomaly_results_by_detection(detection_result_id)
    return pd.DataFrame(anomaly_results)

# Initialize database tables
initialize_database()