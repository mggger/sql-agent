import streamlit as st
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.connectors import MySQLConnector
import os
from PIL import Image

# Dictionary to store the extracted dataframes
data = {}

def main():
    st.set_page_config(page_title="Data Chat Agent", page_icon="🐼")
    st.title("Chat with Your Data")

    # Side Menu Bar
    with st.sidebar:
        st.title("Configuration:⚙️")
        data_source = st.radio("Choose data source:", ("File Upload", "MySQL Database"))

        if data_source == "File Upload":
            file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
            st.markdown(":green[*Please ensure the first row has the column names.*]")
        else:
            st.subheader("MySQL Database Configuration")
            host = st.text_input("Host", "localhost")
            port = st.number_input("Port", value=3306)
            database = st.text_input("Database Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            table = st.text_input("Table Name")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = get_LLM(openai_api_key)

    if data_source == "File Upload" and file_upload is not None:
        data = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
        st.dataframe(data[df])

        if llm:
            analyst = get_agent(data, llm)
            chat_window(analyst)
    elif data_source == "MySQL Database" and all([host, port, database, username, password, table]):
        try:
            connector = MySQLConnector(
                config={
                    "host": host,
                    "port": port,
                    "database": database,
                    "username": username,
                    "password": password,
                    "table": table,
                }
            )
            if llm:
                analyst = Agent([connector], config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
                st.success("Successfully connected to the database!")
                chat_window(analyst)
        except Exception as e:
            st.error(f"Failed to connect to the database: {str(e)}")
    else:
        st.warning("Please upload your data or configure the database connection to start chatting.")

# Function to get LLM
def get_LLM(user_api_key):
    llm = OpenAI(api_token=user_api_key)
    return llm

# Function for chat window
def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("How can I help you with your data?🧐")

    # Initializing message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the message history on re-run
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                display_response(message['response'])
            elif 'error' in message:
                st.text(message['error'])

    # Getting the questions from the users
    user_question = st.chat_input("What are you curious about? ")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "question": user_question})

        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
                display_response(response)
                st.session_state.messages.append({"role": "assistant", "response": response})
        except Exception as e:
            st.error(f"⚠️Sorry, Couldn't generate the answer: {str(e)}")

    # Function to clear history
    def clear_chat_history():
        st.session_state.messages = []

    st.sidebar.button("CLEAR Chat history🗑️", on_click=clear_chat_history)

def display_response(response):
    if isinstance(response, str) and os.path.isfile(response) and response.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        image = Image.open(response)
        st.image(image, caption="Generated Image")
    else:
        st.write(response)

def get_agent(data, llm):
    agent = Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
    return agent

def extract_dataframes(raw_file):
    dfs = {}
    if raw_file.name.endswith('.csv'):
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df
    elif raw_file.name.endswith(('.xlsx', '.xls')):
        xls = pd.ExcelFile(raw_file)
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)
    return dfs

if __name__ == "__main__":
    main()