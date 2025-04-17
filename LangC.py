import streamlit as st
import os
from dotenv import load_dotenv 
from langchain_community.utilities import SQLDatabase
import lcconfig
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, Annotated

# Load API key
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

# Load database config
config = lcconfig.load_config()
username, password, host, port, uri, db_name = (
    config['username'], config['password'], config['host'], 
    config['port'], config['uri'], config['DB']
)

st.set_page_config(layout='wide')
st.title("NLQ Bot")
st.write(f"Currently selected DB: {db_name}")

# Sidebar: Select Database
available_dbs = ["Netflix", "Payments", "User_accounts"]  # Replace with actual DB names
selected_db = st.sidebar.selectbox("Select your DB", available_dbs)

# Setup database connection
db = SQLDatabase.from_uri(f"{uri}://{username}:{password}@{host}:{port}/{db_name}")

# Define state class
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# Initialize chat model
llm = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=key)
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

# Define query generation function
class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# Define query execution function
def execute_query(state: State):
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

# Define answer generation function
def generate_answer(state: State):
    prompt = (
        f"Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# Create LangGraph pipeline
graph_builder = StateGraph(State).add_sequence([
    write_query, execute_query, generate_answer
])
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# Streamlit interface
question = st.text_input("Enter your query about the database:")


if st.button("Submit") and question:
    with st.spinner("Processing your query..."):
        for step in graph.stream({"question": question}, stream_mode="updates"): 
            with st.container(border=True):
                st.markdown(list(step.keys())[0])
                st.markdown(list(list(step.values())[0].values())[0])
                
