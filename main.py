import streamlit as st
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os
from PIL import Image



# Dictionary to store the extracted dataframes
data = {}


def main():
    st.set_page_config(page_title="SQL Agent", page_icon="🐼")
    st.title("Chat with Your Data")
    # reading the csv file

    # Side Menu Bar
    with st.sidebar:
        st.title("Configuration:⚙️")
        # Activating Demo Data
        file_upload = st.sidebar.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])

        st.markdown(":green[*Please ensure the first row has the column names.*]")

    if file_upload is not None:
        data = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!",
                          tuple(data.keys()), index=0
                          )
        st.dataframe(data[df])

        openai_api_key = os.environ.get("OPENAI_API_KEY")

        llm = get_LLM(openai_api_key)

        if llm:
            # Instattiating PandasAI agent
            analyst = get_agent(data, llm)

            # starting the chat with the PandasAI agent
            chat_window(analyst)

    else:
        st.warning("Please upload your data first! You can upload a CSV or an Excel file.")


# Function to get LLM
def get_LLM(user_api_key):
    llm = OpenAI(api_token=user_api_key)
    return llm


# Functuion for chat window
def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("How can I help you with your data?🧐")

    # Initilizing message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the message history on re-reun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # priting the questions
            if 'question' in message:
                st.markdown(message["question"])
            # printing the code generated and the evaluated code
            elif 'response' in message:
                # getting the response
                display_response(message['response'])

            # retrieving error messages
            elif 'error' in message:
                st.text(message['error'])
    # Getting the questions from the users
    user_question = st.chat_input("What are you curious about? ")

    if user_question:
        # Displaying the user question in the chat message
        with st.chat_message("user"):
            st.markdown(user_question)
        # Adding user question to chat history
        st.session_state.messages.append({"role": "user", "question": user_question})

        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
                display_response(response)
                st.session_state.messages.append({"role": "assistant", "response": response})

        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"

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
    """
    The function creates an agent on the dataframes exctracted from the uploaded files
    Args:
        data: A Dictionary with the dataframes extracted from the uploaded data
        llm:  llm object based on the ll type selected
    Output: PandasAI Agent
    """
    agent = Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})

    return agent


def extract_dataframes(raw_file):
    """
    This function extracts dataframes from the uploaded file/files
    Args:
        raw_file: Upload_File object
    Processing: Based on the type of file read_csv or read_excel to extract the dataframes
    Output:
        dfs:  a dictionary with the dataframes

    """
    dfs = {}
    if raw_file.name.split('.')[1] == 'csv':
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df

    elif (raw_file.name.split('.')[1] == 'xlsx') or (raw_file.name.split('.')[1] == 'xls'):
        # Read the Excel file
        xls = pd.ExcelFile(raw_file)

        # Iterate through each sheet in the Excel file and store them into dataframes
        dfs = {}
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)

    # return the dataframes
    return dfs


if __name__ == "__main__":
    main()

