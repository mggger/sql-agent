import os
import streamlit as st
from pathlib import Path
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools.sql_database.tool import SQLDatabase
from langchain.agents import AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import pandas as pd

st.set_page_config(page_title="SQL Agent", page_icon="🦜")
st.title("🦜 SQL Agent")

# Sidebar for input settings
st.sidebar.header("Input Settings")
connection_type = st.sidebar.radio("Select input type:", ("MySQL", "Excel / CSV"))


# Function to handle database connection and table creation
@st.cache_resource(ttl="2h")
def setup_database(connection_type, file=None, host=None, port=None, user=None, password=None, database=None):
    if connection_type == "MySQL":
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        table_name = None
    else:  # Excel / CSV
        db_path = "local_sqlite.db"
        db_uri = f"sqlite:///{db_path}"
        engine = create_engine(db_uri)
        if file is not None:
            table_name = Path(file.name).stem.lower().replace(" ", "_")  # Get filename without extension
            file_extension = file.name.split(".")[-1].lower()
            if file_extension == "csv":
                df = pd.read_csv(file)
            else:  # xlsx or xls
                df = pd.read_excel(file)
            df.to_sql(table_name, engine, index=False, if_exists='replace')

    return SQLDatabase.from_uri(database_uri=db_uri), table_name

# Handle input based on connection type
if connection_type == "MySQL":
    st.sidebar.subheader("MySQL Connection Settings")
    host = st.sidebar.text_input("Host:")
    port = st.sidebar.text_input("Port:", "3306")
    user = st.sidebar.text_input("User:")
    password = st.sidebar.text_input("Password:", type="password")
    database = st.sidebar.text_input("Database:")

    if not host or not user or not password or not database:
        st.info("Please provide database connection information.")
        st.stop()

    db, _ = setup_database(connection_type, host=host, port=port, user=user, password=password, database=database)
else:  # Excel / CSV
    uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV file", type=["xlsx", "xls", "csv"])

    if not uploaded_file:
        st.info("Please upload a file.")
        st.stop()

    db, table_name = setup_database(connection_type, file=uploaded_file)
    if table_name:
        st.sidebar.success(f"Table '{table_name}' created successfully!")

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY is not set in the environment variables.")
    st.stop()

# Setup agent
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini", temperature=0, streaming=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# Chat interface
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with your database?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything about your database!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[st_cb])
        except Exception as e:
            print(e)
            response = "I Don't know."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)