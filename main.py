import streamlit as st
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.connectors import MySQLConnector
import os
from PIL import Image
import uuid

# Dictionary to store the extracted dataframes
data = {}

# Dictionary to store the generated images
image_dict = {}

def main():
    st.set_page_config(page_title="Data Chat Agent", page_icon="🐼", layout="wide")

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
            color: #17213D;
        }

        .main {
            background-color: #f8f9fa;
        }

        .stApp {
            background-color: #f8f9fa;
        }

        .css-1d391kg {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(23, 33, 61, 0.1);
        }

        .stButton>button {
            background-color: #FCA311;
            color: #17213D;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #e39310;
            color: #ffffff;
        }

        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #17213D;
        }

        .stSelectbox>div>div>select {
            border-radius: 5px;
            border: 1px solid #17213D;
        }

        .stFileUploader>div>div>button {
            background-color: #FCA311;
            color: #17213D;
        }

        .stRadio>div {
            background-color: white;
            border-radius: 5px;
            padding: 0.5rem;
        }

        .stChat {
            border-radius: 15px;
            border: 1px solid #17213D;
            padding: 1rem;
        }

        h1, h2, h3 {
            color: #17213D;
        }

        .sidebar .sidebar-content {
            background-color: #17213D;
            color: #ffffff;
        }

        .sidebar .sidebar-content .stRadio label {
            color: #ffffff;
        }

        .sidebar .sidebar-content .stButton>button {
            background-color: #FCA311;
            color: #17213D;
        }

        .sidebar .sidebar-content .stButton>button:hover {
            background-color: #ffffff;
            color: #17213D;
        }

        .stSpinner>div>div {
            border-top-color: #FCA311 !important;
        }

        .stAlert {
            background-color: #FCA311;
            color: #17213D;
        }

        .stDataFrame {
            border: 1px solid #17213D;
        }

        .stDataFrame th {
            background-color: #17213D;
            color: #ffffff;
        }

        .stDataFrame td {
            background-color: #ffffff;
            color: #17213D;
        }

        .stDataFrame tr:nth-of-type(even) {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)

    # Create two columns: one for chat, one for image display
    chat_col, image_col = st.columns(2)

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "image_dict" not in st.session_state:
        st.session_state.image_dict = {}

    with chat_col:
        st.title("Chat with Your Data")

        # Side Menu Bar
        with st.sidebar:
            st.title("Configuration:⚙️")
            data_source = st.radio("Choose data source:", ("File Upload", "MySQL Database"))

            if data_source == "File Upload":
                file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False,
                                               type=['csv', 'xls', 'xlsx'])
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

            if llm:
                analyst = get_agent(data, llm)
                chat_window(analyst, chat_col, image_col)
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
                    analyst = Agent([connector],
                                    config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
                    st.success("Successfully connected to the database!")
                    chat_window(analyst, chat_col, image_col)
            except Exception as e:
                st.error(f"Failed to connect to the database: {str(e)}")
        else:
            st.warning("Please upload your data or configure the database connection to start chatting.")

    # Update image column
    update_image_column(image_col)

# Function to get LLM
def get_LLM(user_api_key):
    llm = OpenAI(api_token=user_api_key)
    return llm

# Function for chat window
def chat_window(analyst, chat_col, image_col):
    with chat_col:
        with st.chat_message("assistant"):
            st.text("How can I help you with your data?🧐")

        # Container for chat messages
        chat_container = st.container()

        # Displaying the message history on re-run
        with chat_container:
            for idx, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    if 'question' in message:
                        st.markdown(message["question"])
                    elif 'response' in message:
                        display_response(message['response'], idx, image_col)
                    elif 'error' in message:
                        st.text(message['error'])

        # Getting the questions from the users
        with st.container():
            user_question = st.chat_input("What are you curious about? ")

        if user_question:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_question)
                st.session_state.messages.append({"role": "user", "question": user_question})

                try:
                    with st.spinner("Analyzing..."):
                        response = analyst.chat(user_question)
                        idx = len(st.session_state.messages)
                        if isinstance(response, str) and response.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            if os.path.isfile(response):
                                new_filename = f"{uuid.uuid4()}{os.path.splitext(response)[1]}"
                                os.rename(response, new_filename)
                                st.session_state.image_dict[idx] = new_filename
                                response = new_filename
                        with st.chat_message("assistant"):
                            display_response(response, idx, image_col)
                        st.session_state.messages.append({"role": "assistant", "response": response})
                except Exception as e:
                    st.error(f"⚠️Sorry, Couldn't generate the answer: {str(e)}")

    # Function to clear history
    def clear_chat_history():
        st.session_state.messages = []
        st.session_state.current_image = None
        image_dict.clear()

    st.sidebar.button("CLEAR Chat history🗑️", on_click=clear_chat_history)

def display_response(response, idx, image_col):
    if isinstance(response, str) and response.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        st.session_state.current_image = response
        if st.button(f"Show Image {idx}"):
            st.session_state.current_image = st.session_state.image_dict.get(idx, response)
            with image_col:
                st.empty()
            update_image_column(image_col)
    else:
        st.write(response)

def update_image_column(image_col):
    with image_col:
        if st.session_state.current_image:
            try:
                st.image(Image.open(st.session_state.current_image), use_column_width=True)
            except FileNotFoundError:
                st.error(f"Image file not found: {st.session_state.current_image}")

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