import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize Streamlit page configuration
st.set_page_config(page_title="SQL Query Assistant", layout="wide")
st.title("SQL Query Assistant")

# Initialize Ollama
llm = Ollama(model="codellama")

# Database configuration
DB_PATH = 'gaming_data.db'

# Function to get database connection
def get_db_connection():
    return sqlite3.connect(DB_PATH)

# Create SQLite database and load data
def init_database():
    conn = get_db_connection()
    
    # Check if tables already exist
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = [table[0] for table in cursor.fetchall()]
    
    if 'bonus' not in existing_tables or 'player_kpis' not in existing_tables:
        # Read CSV files
        bonus_df = pd.read_csv('input/Bonus_Data.csv')
        kpis_df = pd.read_csv('input/Player_KPIs.csv')
        
        # Convert txn_datetime to proper datetime format
        def convert_date(date_str):
            try:
                return pd.to_datetime(date_str).strftime('%Y-%m-%d 00:00:00')
            except:
                return None
        
        # Convert dates in player_kpis
        kpis_df['txn_datetime'] = kpis_df['txn_datetime'].apply(convert_date)
        
        # Convert dates in bonus table
        date_columns = ['unlocked_datetime', 'expiry_datetime']
        for col in date_columns:
            bonus_df[col] = bonus_df[col].apply(convert_date)
        
        # Create player_kpis table with proper schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_kpis (
            txn_datetime TEXT NOT NULL,
            txn_month INTEGER,
            txn_year INTEGER,
            user_id INTEGER,
            win_amount REAL,
            win_count INTEGER,
            deposit_amount REAL,
            deposit_count INTEGER,
            total_unique_players INTEGER,
            table_count INTEGER,
            game_played_count INTEGER,
            wager_amount REAL,
            wager_rejoin_amount REAL,
            partner_wager_amount REAL,
            partner_rake_amount REAL,
            partner_won_amount REAL,
            other_cost REAL,
            rake_ REAL
        )
        """)
        
        # Create bonus table with proper schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bonus (
            user_id INTEGER,
            bonus_id INTEGER,
            transaction_id INTEGER,
            bonus_code TEXT,
            bonus_criteria TEXT,
            bonusCode TEXT,
            relatedTo TEXT,
            activity TEXT,
            tandcParagraph INTEGER,
            target_amount REAL,
            contribution REAL,
            pending REAL,
            expired_amount REAL,
            bonus_amount REAL,
            converted_to_cash REAL,
            redeemed_amount REAL,
            unlocked_datetime TEXT NOT NULL,
            expiry_datetime TEXT NOT NULL,
            bonus_status TEXT,
            bonus_type TEXT,
            status TEXT,
            bonus_category TEXT,
            bonus_mode TEXT
        )
        """)
        conn.commit()
        
        # Write to SQLite database
        kpis_df.to_sql('player_kpis', conn, if_exists='replace', index=False)
        bonus_df.to_sql('bonus', conn, if_exists='replace', index=False)
        
        # Create indices for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_datetime ON player_kpis(txn_datetime);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON player_kpis(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bonus_status ON bonus(bonus_status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_unlocked_datetime ON bonus(unlocked_datetime);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_expiry_datetime ON bonus(expiry_datetime);")
        conn.commit()
    
    conn.close()

# Function to get relevant schemas
def get_relevant_schemas(question):
    # Transform question using the same vectorizer
    question_vector = st.session_state.vectorizer.transform([question])
    
    # Calculate similarity scores
    similarities = cosine_similarity(question_vector, st.session_state.schema_vectors)
    
    # Get indices of top 2 most similar schemas
    top_indices = np.argsort(similarities[0])[-2:][::-1]
    
    # Get the relevant schemas
    relevant_schemas = [st.session_state.schemas[i] for i in top_indices]
    return "\n".join(relevant_schemas)

# Function to validate and sanitize SQL query
def validate_sql_query(query):
    # Convert to lowercase for easier checks
    query_lower = query.lower()
    
    # Check for dangerous operations
    dangerous_keywords = ['drop', 'delete', 'truncate', 'update', 'insert', 'alter', 'create']
    if any(keyword in query_lower for keyword in dangerous_keywords):
        return False, "Only SELECT queries are allowed for safety reasons."
    
    # Check if it's a SELECT query
    if not query_lower.strip().startswith('select'):
        return False, "Query must start with SELECT."
    
    # Check for common SQLite syntax issues
    if 'group by' in query_lower:
        # Ensure GROUP BY columns are valid
        group_by_clause = query_lower.split('group by')[1].split(';')[0].strip()
        if not group_by_clause or group_by_clause.isspace():
            return False, "Invalid GROUP BY clause - columns must be specified"
    
    if 'order by' in query_lower:
        # Ensure ORDER BY columns are valid
        order_by_clause = query_lower.split('order by')[1].split(';')[0].strip()
        if not order_by_clause or order_by_clause.isspace():
            return False, "Invalid ORDER BY clause - columns must be specified"
    
    # Check for proper date/time function usage
    if 'strftime' in query_lower:
        if not re.search(r"strftime\s*\(\s*'[^']*'\s*,\s*\w+(\.\w+)?\s*\)", query_lower):
            return False, "Invalid strftime usage. Format should be: strftime('format', column)"
    
    # Check for proper string concatenation
    if '||' in query:
        if not re.search(r"\w+\s*\|\|\s*\w+", query):
            return False, "Invalid string concatenation. Format should be: expr1 || expr2"
    
    # Try to validate the query by preparing it in SQLite
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # SQLite's prepare statement will catch most syntax errors
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        cursor.close()
        conn.close()
    except sqlite3.Error as e:
        error_msg = str(e)
        # Provide more user-friendly error messages for common issues
        if "no such column" in error_msg:
            return False, f"Invalid column name in query: {error_msg}"
        elif "no such table" in error_msg:
            return False, f"Invalid table name in query: {error_msg}"
        elif "syntax error" in error_msg:
            return False, f"SQL syntax error: {error_msg}"
        else:
            return False, f"SQLite error: {error_msg}"
    
    # Basic validation passed
    return True, ""

# Function to clean and format SQL query
def clean_sql_query(query):
    # Remove any markdown backticks
    query = query.replace('```sql', '').replace('```', '')
    
    # Remove any leading/trailing whitespace
    query = query.strip()
    
    # Remove any comments (both inline and multi-line)
    query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)  # Remove single line comments
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)  # Remove multi-line comments
    query = re.sub(r';.*$', ';', query, flags=re.MULTILINE)  # Remove anything after semicolon
    
    # Ensure the query ends with a semicolon
    if not query.rstrip().endswith(';'):
        query += ';'
    
    # Replace any smart quotes with regular quotes
    query = query.replace('"', '"').replace('"', '"')
    query = query.replace(''', "'").replace(''', "'")
    
    # Ensure consistent spacing around operators
    operators = ['=', '<', '>', '<=', '>=', '<>', '!=']
    for op in operators:
        query = query.replace(op, f' {op} ')
    
    # Remove multiple spaces
    query = ' '.join(query.split())
    
    return query

# Function to strip RTF formatting
def strip_rtf(text):
    # Remove RTF formatting
    text = re.sub(r'\\[a-z]{1,32}(-?\d{1,10})?[ ]?', ' ', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\\', '', text)
    return text.strip()

# Function to load team context
def load_team_context():
    team_context = []
    
    # Load questions from RTF file
    try:
        with open('input/questions.rtf', 'r', encoding='utf-8') as f:
            questions = strip_rtf(f.read())
            team_context.append("Common Team Questions:\n" + questions)
    except Exception as e:
        st.warning(f"Could not load questions file: {str(e)}")
    
    # Load formulas from text file
    try:
        with open('input/Formulas.txt', 'r', encoding='utf-8') as f:
            formulas = f.read()
            team_context.append("Common Formulas and Calculations:\n" + formulas)
    except Exception as e:
        st.warning(f"Could not load formulas file: {str(e)}")
    
    return "\n\n".join(team_context)

# Function to get actual values from database for filters
def get_filter_values(table_name, column_name, filter_value=None, limit=10):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get distinct values for the column
        if filter_value:
            # If a filter value is provided, find closest matching values
            query = f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE {column_name} LIKE ?
                LIMIT {limit}
            """
            cursor.execute(query, (f'%{filter_value}%',))
        else:
            # If no filter value, just get distinct values
            query = f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                LIMIT {limit}
            """
            cursor.execute(query)
        
        values = [row[0] for row in cursor.fetchall()]
        conn.close()
        return values
    except Exception as e:
        st.error(f"Error getting filter values: {str(e)}")
        return []

# Function to extract filter conditions from question
def extract_filter_conditions(question, context):
    filter_conditions = []
    
    # Extract table names from context
    table_pattern = r'CREATE TABLE (?:IF NOT EXISTS )?(\w+)'
    tables = re.findall(table_pattern, context)
    
    # For each table, check its columns in the context
    for table in tables:
        # Extract column definitions
        table_pattern = f'CREATE TABLE (?:IF NOT EXISTS )?{table} \((.*?)\)'
        table_match = re.search(table_pattern, context, re.DOTALL)
        if table_match:
            columns_text = table_match.group(1)
            # Extract column names
            columns = [col.strip().split()[0] for col in columns_text.split(',')]
            
            # Look for potential filter conditions in the question
            for column in columns:
                # Common filter patterns
                patterns = [
                    (r'where\s+' + column + r'\s*[=><]+\s*(["\']?\w+["\']?)', 'exact'),
                    (r'with\s+' + column + r'\s+(?:equal to|equals|is)\s+(["\']?\w+["\']?)', 'exact'),
                    (r'for\s+' + column + r'\s*[=><]+\s*(["\']?\w+["\']?)', 'exact'),
                    (r'like\s+["\']?%?(\w+)%?["\']?\s+in\s+' + column, 'like'),
                ]
                
                for pattern, match_type in patterns:
                    matches = re.finditer(pattern, question.lower())
                    for match in matches:
                        filter_value = match.group(1).strip('"\'')
                        # Get actual values from database
                        actual_values = get_filter_values(table, column, filter_value)
                        if actual_values:
                            # Use the closest matching value
                            filter_conditions.append({
                                'table': table,
                                'column': column,
                                'value': actual_values[0],
                                'type': match_type
                            })
    
    return filter_conditions

# Function to generate SQL query
def generate_sql_query(question):
    # Get relevant schema context
    context = get_relevant_schemas(question)
    
    # Get team context
    team_context = load_team_context()
    
    # Extract and validate filter conditions
    filter_conditions = extract_filter_conditions(question, context)
    
    # Add filter information to the context
    if filter_conditions:
        filter_context = "\nActual filter values found in database:"
        for condition in filter_conditions:
            filter_context += f"\n- For {condition['table']}.{condition['column']}, using value: {condition['value']}"
        context += filter_context
    
    # Generate SQL query
    chain = st.session_state.prompt | llm | StrOutputParser()
    sql_query = chain.invoke({
        "context": context,
        "team_context": team_context,
        "question": question,
        "filter_conditions": filter_conditions
    })
    
    # Clean and format the query
    sql_query = clean_sql_query(sql_query)
    
    # Validate the generated query
    is_valid, error_message = validate_sql_query(sql_query)
    if not is_valid:
        raise ValueError(f"Invalid SQL query generated: {error_message}")
    
    return sql_query

# Function to execute SQL query
def execute_query(sql_query):
    try:
        conn = get_db_connection()
        # Execute query and fetch all results
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

# Function to reset database
def reset_database():
    # Close any existing connections
    if 'conn' in st.session_state:
        try:
            st.session_state.conn.close()
        except:
            pass
        del st.session_state.conn
    
    # Reset initialization flag
    if 'db_initialized' in st.session_state:
        del st.session_state.db_initialized

# Initialize database if needed
if 'db_initialized' not in st.session_state:
    # Reset database first
    reset_database()
    # Initialize fresh database
    init_database()
    st.session_state.db_initialized = True

if 'schema_store' not in st.session_state:
    # Define table schemas
    schemas = [
        """
        Table: bonus
        Columns:
        - user_id: User identifier
        - bonus_id: Unique bonus identifier
        - transaction_id: Transaction identifier
        - bonus_code: Code for the bonus
        - bonus_criteria: Criteria for bonus
        - bonusCode: Bonus code reference
        - relatedTo: Relation type
        - activity: Activity type
        - tandcParagraph: Terms and conditions reference
        - target_amount: Target amount for bonus
        - contribution: Contribution amount
        - pending: Pending amount
        - expired_amount: Expired amount
        - bonus_amount: Bonus amount
        - converted_to_cash: Converted to cash amount
        - redeemed_amount: Redeemed amount
        - unlocked_datetime: DateTime when bonus was unlocked
        - expiry_datetime: DateTime when bonus expires
        - bonus_status: Status of the bonus
        - bonus_type: Type of bonus
        - status: Status of the bonus
        - bonus_category: Category of the bonus
        - bonus_mode: Mode of the bonus
        """,
        """
        Table: player_kpis
        Columns:
        - txn_datetime: Transaction datetime
        - txn_month: Transaction month
        - txn_year: Transaction year
        - user_id: User identifier
        - win_amount: Amount won
        - win_count: Number of wins
        - deposit_amount: Amount deposited
        - deposit_count: Number of deposits
        - total_unique_players: Total unique players
        - table_count: Number of tables
        - game_played_count: Number of games played
        - wager_amount: Amount wagered
        - wager_rejoin_amount: Amount wagered on rejoin
        - partner_wager_amount: Partner wager amount
        - partner_rake_amount: Partner rake amount
        - partner_won_amount: Partner won amount
        - other_cost: Other cost
        - rake_: Rake amount
        """
    ]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    schema_vectors = vectorizer.fit_transform(schemas)
    
    st.session_state.vectorizer = vectorizer
    st.session_state.schema_vectors = schema_vectors
    st.session_state.schemas = schemas

# Query template for SQL generation
QUERY_TEMPLATE = """
You are an expert SQL query generator for SQLite databases. Generate a clean, executable SQLite query that answers the user's question.

IMPORTANT:
1. Return ONLY the raw SQL query - no markdown formatting, no backticks, no explanations
2. Use SQLite syntax and functions
3. Use double quotes for table names and single quotes for string literals
4. End the query with a semicolon
5. Only write SELECT queries (no INSERT, UPDATE, DELETE, etc.)
6. For datetime fields:
   - Dates are stored in 'YYYY-MM-DD 00:00:00' format
   - Use date(txn_datetime) for date only
   - Use strftime('%Y-%m', txn_datetime) for year-month
   - Use strftime('%Y', txn_datetime) for year only
   - Use strftime('%m', txn_datetime) for month only
   - Use strftime('%W', txn_datetime) for week number (00-53)
   - Use strftime('%w', txn_datetime) for day of week (0-6, Sunday=0)

7. Bonus status values:
   - 'Locked': Bonus is not yet available
   - 'Unlocked': Bonus is active and available for use
   - 'Expired': Bonus has expired

Available tables and their schemas:
{context}

Common Team Questions and Formulas:
{team_context}

Example valid queries:
# Get monthly deposits with proper date formatting
SELECT 
    strftime('%Y-%m', txn_datetime) as month,
    strftime('%Y', txn_datetime) as year,
    strftime('%m', txn_datetime) as month_num,
    SUM(deposit_amount) as total_deposits
FROM player_kpis 
GROUP BY month
ORDER BY month;

# Get weekly deposits
SELECT 
    strftime('%Y-%W', txn_datetime) as year_week,
    SUM(deposit_amount) as total_deposits
FROM player_kpis 
GROUP BY year_week
ORDER BY year_week;

# Get active bonuses with expiry dates
SELECT 
    user_id,
    bonus_amount,
    date(unlocked_datetime) as start_date,
    date(expiry_datetime) as end_date
FROM bonus 
WHERE bonus_status = 'Unlocked'
ORDER BY bonus_amount DESC;

User Question: {question}

SQL Query:"""

# Create the prompt template
st.session_state.prompt = PromptTemplate(
    template=QUERY_TEMPLATE,
    input_variables=["context", "team_context", "question"]
)

# Streamlit UI
st.write("Ask questions about your gaming data and get SQL queries and results!")
st.write("Example questions you can ask:")
st.write("""
- Show me the top 10 players with the highest win amounts
- What is the total deposit amount by month?
- List all active bonuses and their amounts
- Show me players who have both deposits and bonuses
- Calculate the win rate (win_count/game_played_count) for each player
- What is the average bonus amount by bonus code?
- Show me players with deposits greater than $1000
- List expired bonuses from the last month
- What is the total wager amount by game type?
- Show me players who have unlocked bonuses but haven't used them
""")

question = st.text_area("Enter your question:", height=100)

if st.button("Generate and Execute Query"):
    if question:
        with st.spinner("Generating SQL query..."):
            # Generate SQL query
            sql_query = generate_sql_query(question)
            
            # Display the generated query
            st.subheader("Generated SQL Query:")
            st.code(sql_query, language="sql")
            
            # Execute query and show results
            results = execute_query(sql_query)
            if results is not None:
                st.write("Query Results:")
                # Display full dataframe with pagination
                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show row count
                st.write(f"Total rows: {len(results)}")
    else:
        st.warning("Please enter a question first.")
