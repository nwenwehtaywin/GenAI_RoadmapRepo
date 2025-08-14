import streamlit as st
from langchain_fireworks import ChatFireworks
from langchain.schema import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv

import os

load_dotenv()
os.environ["FIREWORKS_API_KEY"] = os.getenv("FIREWORKS_API_KEY")

model = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")
config = {"configurable": {"thread_id": "abc567"}}
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


st.title("Chatbot Application")

# Create a session state to store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input text box for user question
question = st.text_input("Enter your question:")

# Submit button
if st.button("Submit"):
    if question:
        input_messages = [HumanMessage(question)]
        output = app.invoke({"messages": input_messages}, config)

        st.session_state.messages.append({"role": "user", "content": question})
        
        response_content=output["messages"][-1].content
        st.session_state.messages.append({"role": "bot", "content": response_content})
        
        # Clear the input box
        st.experimental_rerun()

# Display messages
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")
