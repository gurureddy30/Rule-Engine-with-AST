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

    @validator('data')
    def validate_data_types(cls, v):
        for key, value in v.items():
            if not isinstance(value, (int, float, str, bool)):
                raise ValueError(f"Invalid type for {key}: {type(value)}. Must be int, float, str, or bool.")
        return v

# In your FastAPI route
@app.post("/evaluate/")
def evaluate_rules(eval_data: RuleInput, db: Session = Depends(get_db)):
    rules = db.query(Rule).all()
    results = {}
    for rule in rules:
        try:
            result = evaluate_rule(rule.ast, eval_data.data)
            results[rule.id] = result
        except Exception as e:
            results[rule.id] = f"Error evaluating rule: {str(e)}"
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# rule_engine.py

import logging
import json
from typing import List, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, type: str, value: Optional[str] = None, left: Optional['Node'] = None, right: Optional['Node'] = None):
        self.type = type
        self.value = value
        self.left = left
        self.right = right

    def to_dict(self):
        return {
            "type": self.type,
            "value": self.value,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None
        }

def tokenize(rule_string: str) -> List[str]:
    """Convert rule string into tokens, preserving parentheses and operators."""
    # Add spaces around parentheses and operators while preserving quoted strings
    operators = ['AND', 'OR', '>', '<', '>=', '<=', '=', '!=']
    tokens = []
    current_token = ''
    in_quotes = False
    i = 0
    
    while i < len(rule_string):
        char = rule_string[i]
        
        if char == '"' or char == "'":
            in_quotes = not in_quotes
            current_token += char
        elif char in '()' and not in_quotes:
            if current_token.strip():
                tokens.append(current_token.strip())
            tokens.append(char)
            current_token = ''
        elif char.isspace() and not in_quotes:
            if current_token.strip():
                tokens.append(current_token.strip())
            current_token = ''
        else:
            current_token += char
            
            # Check for operators
            if not in_quotes:
                rest_of_string = rule_string[i - len(current_token) + 1:]
                for op in operators:
                    if rest_of_string.upper().startswith(op):
                        if current_token[:-len(op)].strip():
                            tokens.append(current_token[:-len(op)].strip())
                        tokens.append(op)
                        current_token = ''
                        i += len(op) - 1
                        break
        
        i += 1
    
    if current_token.strip():
        tokens.append(current_token.strip())
    
    return [token for token in tokens if token.strip()]

def parse_expression(tokens: List[str], start: int = 0, end: Optional[int] = None) -> tuple[Node, int]:
    """Parse a list of tokens into an AST, handling nested parentheses."""
    if end is None:
        end = len(tokens)
    
    def find_matching_parenthesis(tokens: List[str], start: int) -> int:
        count = 1
        i = start + 1
        while i < len(tokens) and count > 0:
            if tokens[i] == '(':
                count += 1
            elif tokens[i] == ')':
                count -= 1
            i += 1
        return i - 1

    def parse_condition(condition_tokens: List[str]) -> Node:
        if len(condition_tokens) != 3:
            raise ValueError(f"Invalid condition: {' '.join(condition_tokens)}")
        return Node(
            type="operator",
            value=condition_tokens[1],
            left=Node("operand", condition_tokens[0]),
            right=Node("operand", condition_tokens[2])
        )

    i = start
    stack = []
    current_tokens = []
    
    while i < end:
        token = tokens[i]
        
        if token == '(':
            closing_index = find_matching_parenthesis(tokens, i)
            subnode, _ = parse_expression(tokens, i + 1, closing_index)
            stack.append(subnode)
            i = closing_index + 1
            continue
        elif token in ['AND', 'OR']:
            if current_tokens:
                stack.append(parse_condition(current_tokens))
                current_tokens = []
            right_tokens = tokens[i + 1:]
            if right_tokens and right_tokens[0] == '(':
                closing_index = find_matching_parenthesis(right_tokens, 0)
                right_node, _ = parse_expression(right_tokens, 1, closing_index)
                i += closing_index + 2
            else:
                right_node, offset = parse_expression(tokens, i + 1, end)
                i = offset
            
            left_node = stack.pop() if stack else parse_condition(current_tokens)
            stack.append(Node("operator", token, left_node, right_node))
            continue
        else:
            current_tokens.append(token)
        
        i += 1
    
    if current_tokens:
        stack.append(parse_condition(current_tokens))
    
    if not stack:
        raise ValueError("Empty expression")
    
    return stack[0], i

def create_rule(rule_string: str) -> str:
    """Create an AST from a rule string and return it as a JSON string."""
    try:
        tokens = tokenize(rule_string)
        logger.info(f"Tokens: {tokens}")
        ast, _ = parse_expression(tokens)
        return json.dumps(ast.to_dict())
    except Exception as e:
        raise ValueError(f"Error parsing rule: {str(e)}")

def evaluate_rule(ast_string: str, data: dict) -> bool:
    """Evaluate a rule AST against input data."""
    ast = json.loads(ast_string)

    def convert_to_number(value: str) -> Union[int, float, str]:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value.strip("'\"")

    def evaluate_node(node: dict) -> bool:
        if node['type'] == "operand":
            if node['value'] in data:
                return convert_to_number(str(data[node['value']]))
            return convert_to_number(node['value'])
        
        left_val = evaluate_node(node['left'])
        right_val = evaluate_node(node['right'])
        
        logger.info(f"Evaluating {node['value']} operation: {left_val} ({type(left_val)}) {node['value']} {right_val} ({type(right_val)})")
        
        if node['value'] in ['>', '<', '>=', '<=']:
            if not (isinstance(left_val, (int, float)) and isinstance(right_val, (int, float))):
                raise ValueError(f"Cannot compare {left_val} ({type(left_val)}) and {right_val} ({type(right_val)}) using {node['value']}. Ensure both values are numeric.")
        
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '=': lambda x, y: x == y,
            '!=': lambda x, y: x != y,
            'AND': lambda x, y: x and y,
            'OR': lambda x, y: x or y
        }
        
        if node['value'] not in operators:
            raise ValueError(f"Unknown operator: {node['value']}")
            
        result = operators[node['value']](left_val, right_val)
        logger.info(f"Result of {left_val} {node['value']} {right_val}: {result}")
        return result

    try:
        final_result = evaluate_node(ast)
        logger.info(f"Final evaluation result: {final_result}")
        return final_result
    except Exception as e:
        logger.error(f"Error evaluating rule: {str(e)}")
        raise ValueError(f"Error evaluating rule: {str(e)}")
    
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Chaithu@9515")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "rule_engine")

# Ensure the password is properly escaped
DB_PASSWORD = DB_PASSWORD.replace("@", "%40")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

try:
    engine = create_engine(DATABASE_URL, echo=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Test the connection
    with engine.connect() as connection:
        print("Successfully connected to the database!")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    raise

print(f"Database URL: {DATABASE_URL}")  # Print the URL for debugging (remove in production)

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