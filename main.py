from tempfile import NamedTemporaryFile;
import streamlit as st;
from langchain.agents import initialize_agent;
from langchain.chat_models import ChatOpenAI;
from langchain.chains.conversation.memory import  ConversationBufferWindowMemory;
from tools import imageCaptionTool,ObjectDetectionTool;
tools = [imageCaptionTool(), ObjectDetectionTool()]

# Set up conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Initialize the language model
llm = ChatOpenAI(
    openai_api_key='sk-proj-pOht1XuFukQn57Qc9olTT3BlbkFJvA5zxO5GJzWYkOszeBuT',
    temperature=0,
    model_name="gpt-3.5-turbo"
)

# Initialize the agent
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopped_method='generate'
)
st.title("Ask a question to an image");
st.header("Please upload an image");



file=st.file_uploader("",type=["jpeg","jpg","png"]);
if file:
    st.image(file,use_column_width=True);
    user_question=st.text_input("Ask a question ");
    with NamedTemporaryFile(dir='.') as f:
        img_path = f.name;
        f.write(file.getbuffer());

    if user_question and user_question != "":
        with st.spinner(text="In progress... "):
            response = agent.run('{},this is the image path: {}'.format(user_question, img_path));

            st.write(response);








