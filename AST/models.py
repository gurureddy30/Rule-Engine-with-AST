# models.py
from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Rule(Base):
    __tablename__ = "rules"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    rule_string = Column(String(1000), nullable=False)
    ast = Column(JSON, nullable=False)