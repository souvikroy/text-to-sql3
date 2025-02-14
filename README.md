# AI meet BI for mobile games

A RAG-based chatbot that helps generate and execute SQL queries for gaming data analysis.


https://github.com/user-attachments/assets/974c0b14-fe78-4d77-bb53-834e1d0d1b35


## Setup

1. Make sure you have Python 3.8+ installed
2. Install Ollama from https://ollama.ai/
3. Pull the CodeLlama model:
```bash
ollama pull codellama
```

4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure the input CSV files are in the `input` directory:
   - Bonus_Data.csv
   - Player_KPIs.csv

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Using the Application

1. Enter your question in natural language about the gaming data
2. Click "Generate and Execute Query" button
3. The application will:
   - Generate an appropriate SQL query
   - Display the generated query
   - Execute the query and show the results

## Example Questions

- Show me the top 10 players with highest win amounts
- What is the average deposit amount by month?
- List all bonuses that are expired
- Show total bonus amounts by bonus code
- Calculate the win rate (win_count/game_played_count) for each player
