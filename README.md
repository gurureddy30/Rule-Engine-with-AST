# 🚀 AST Project


## 📝 Overview
The **AST Project** is a Python-based application designed for seamless integration of business rules with database management and a web interface. This project leverages modular architecture, enabling users to define and process custom business rules, interact with a database, and serve web content efficiently.


## 📚 Table of Contents
- [📝 Overview](#-overview)
- [📂 Project Structure](#-project-structure)
- [⚙️ Setup Instructions](#️-setup-instructions)
- [✨ Features](#-features)
- [🚀 Usage Guide](#-usage-guide)
- [📄 File Descriptions](#-file-descriptions)
- [👥 Contributing](#-contributing)



## 📂 Project Structure
```bash
# Let's unzip and explore the content of the uploaded file to understand its structure.
import zipfile
import os

# Path to the uploaded zip file
zip_file_path = '/mnt/data/ASTzip file.zip'
extract_dir = '/mnt/data/AST_project/'

# Extracting the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Walking through the directory structure to display it
project_structure = []
for dirpath, dirnames, filenames in os.walk(extract_dir):
    level = dirpath.replace(extract_dir, '').count(os.sep)
    indent = ' ' * 4 * level
    project_structure.append(f'{indent}{os.path.basename(dirpath)}/')
    subindent = ' ' * 4 * (level + 1)
    for f in filenames:
        project_structure.append(f'{subindent}{f}')

project_structure = "\n".join(project_structure)
project_structure

```
The project has the following structure:

├── database.py # 🗄️ Manages database connections and queries.

├── hoo.py # 📦 Miscellaneous or utility module 

├── main.py # 🚪 Main entry point of the application 

├── models.py # 🏗️ Defines data models for the database

├── rule_engine.py # 🛠️ Implements the rule engine

├── sql.txt # 📜 SQL statements for database setup

├── templates/ │ └── index.html # 🖥️ HTML template for the web front-end 

└── pycache/ # 🧠 Compiled Python files



## ⚙️ Setup Instructions

Follow these instructions to set up and run the AST project on your local machine.

### 1. Clone the Repository
First, clone the repository from GitHub:

```bash
git clone <repository-url>
cd AST
```
### 2. Install Dependencies
Ensure that you have Python 3 installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt

```
### 3. Set up the Database
Run the SQL commands in sql.txt to set up the necessary tables in your database. You can use SQLite for this project:

```bash
sqlite3 your_database.db < sql.txt
```
### 4. Run the Application
Once the database is set up, start the application by running main.py:

```bash
python main.py
```

The application will initialize and be ready for processing business rules and serving content.


![Alt text](image_path)
![Screenshot (15)](https://github.com/user-attachments/assets/a978b777-915f-4d22-a805-b219efd2622e)





## ✨ Features
Rule Engine 🛠️
The rule engine provides a flexible way to define and process business logic, with custom rules that can be tailored to specific business needs. This allows for greater flexibility and extensibility.

Database Interaction 🗄️
Built-in database management enables seamless data storage and retrieval. Using database.py, this project manages SQL commands, queries, and connections, ensuring efficient data handling.

Web Interface 🖥️
The project includes a simple HTML template (index.html), making it easy to serve a user-friendly web front-end. This allows users to interact with the system through a web browser.


## 🚀 Usage Guide
 1. Adding Rules: To add a new business rule, define it within rule_engine.py. Follow the existing rule structure to ensure compatibility.

 2. Managing Data: Use database.py to add, update, or delete records. The sql.txt file provides the structure for initializing the database tables.

 3. Running the Server: Run main.py to initiate the application and serve content to the front-end. The default setup serves the web front-end on localhost, but this can be customized.

 4. Customizing the Front-End: Edit templates/index.html to adjust the appearance and structure of the web interface as needed.



## 📄 File Descriptions
 -> main.py: The main script that initializes and runs the application.
 
 -> database.py: Manages all database interactions, including connections and queries.
 
 -> models.py: Defines data models that represent the structure of the database.
 
 -> rule_engine.py: Contains the logic for evaluating and processing business rules.
 
 -> sql.txt: SQL commands used to set up the database, including table creation and constraints.
 
 -> templates/index.html: HTML file for the front-end interface, served by the application.


## 👥 Contributing
 Contributions are welcome! To contribute:

 Fork the repository.
 Create a new branch for your feature or bug fix.
 Submit a pull request with a description of your changes.
 Please ensure your code follows the project's coding style and passes all tests.







