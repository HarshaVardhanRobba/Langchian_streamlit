import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_community.chains import LLMChain, SequentialChain
from langchain_community.memory import ConversationSummaryBufferMemory
import streamlit as st

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("Gemini Chatbot")

# Initialize the chat model
model = init_chat_model("gemini-2.5-flash", temperature=0.7, model_provider="google_genai")

person_memory = ConversationSummaryBufferMemory(llm = model, input_key='name', memory_key='movies_history', max_token_limit=1000)
dob_memory = ConversationSummaryBufferMemory(llm = model, input_key='name', memory_key='dob_history', max_token_limit=1000)
father_memory = ConversationSummaryBufferMemory(llm = model,input_key='dob', memory_key='father_history', max_token_limit=1000)

# Prompt to get movies by a person's name
input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me all the movies done by {name}.",
)

chain1 = LLMChain(
    llm=model, 
    prompt=input_prompt, 
    verbose=True, 
    memory=person_memory,
    output_key="movies"
)


# Prompt to get the birth date of a person
input_prompt2 = PromptTemplate(
    input_variables=['name'],
    template="When was {name} born?",
)

chain2 = LLMChain(
    llm=model, 
    prompt=input_prompt2, 
    verbose=True, 
    memory=dob_memory,
    output_key="dob")

# Prompt to get the father of the person based on their birth date
input_prompt3 = PromptTemplate(
    input_variables=['dob'],
    template="Who was the father of this {dob}?",
)

chain3 = LLMChain(
    llm=model, 
    prompt=input_prompt3, 
    verbose=True, 
    memory=father_memory, 
    output_key="father")

# Combine the chains into a sequential chain
combined_chain = SequentialChain(
    chains=[chain1, chain2, chain3], 
    input_variables=['name'], 
    output_variables=['movies', 'dob', 'father'] ,
    verbose=True)

input_text = st.text_input("You: ", "Enter a name:")

if st.button("Send"):
    if input_text:
        try:
            # Run the combined chain directly
            final_output = combined_chain({"name":input_text})  # Directly call combined_chain
            
            st.subheader("Results")
            st.write(f"**Films:** {final_output['movies']}")
            st.write(f"**Date of Birth:** {final_output['dob']}")
            st.write(f"**Father is:** {final_output['father']}")

            with st.expander("Conversation History"):
                st.markdown("**Movies Chain History:**")
                st.info(person_memory.load_memory_variables({})['movies_history'])

            with st.expander("DOB Chain History"):    
                st.markdown("**DOB Chain History:**")
                st.info(dob_memory.load_memory_variables({})['dob_history'])

            with st.expander("Father Chain History"):
                st.markdown("**Father Chain History:**")
                st.info(father_memory.load_memory_variables({})['father_history'])

        except Exception as e:
            st.error(f"An error occurred: {e}")

            