# main.py
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models import Rule, Base
from database import SessionLocal, engine
from rule_engine import create_rule, evaluate_rule
from pydantic import BaseModel, validator
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Create tables
try:
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")
except Exception as e:
    print(f"Error creating database tables: {e}")
    raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class RuleCreate(BaseModel):
    rule_string: str

@app.post("/rules/")
def create_new_rule(rule: RuleCreate, db: Session = Depends(get_db)):
    try:
        ast = create_rule(rule.rule_string)
        db_rule = Rule(rule_string=rule.rule_string, ast=ast)
        db.add(db_rule)
        db.commit()
        return {"message": "Rule created successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

class EvaluateData(BaseModel):
    data: dict

class RuleInput(BaseModel):
    data: Dict[str, Any]
    rule_id: int

    @validator('data')
    def validate_data_types(cls, v):
        for key, value in v.items():
            if not isinstance(value, (int, float, str, bool)):
                raise ValueError(f"Invalid type for {key}: {type(value)}. Must be int, float, str, or bool.")
        return v

@app.post("/evaluate/")
def evaluate_rules(eval_data: RuleInput, db: Session = Depends(get_db)):
    rule = db.query(Rule).filter(Rule.id == eval_data.rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule with id {eval_data.rule_id} not found")
    
    try:
        result = evaluate_rule(rule.ast, eval_data.data)
        return {rule.id: result}
    except Exception as e:
        return {rule.id: f"Error evaluating rule: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)


