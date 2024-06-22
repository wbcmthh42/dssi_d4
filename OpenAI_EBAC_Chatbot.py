import os
# from dotenv import load_dotenv
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# file_path = '.env'
# load_dotenv(file_path)

class chatbot():
      
      def __init__(self):
            pass
      
      def ophai_chatbot(self):
            
            st.sidebar.header('Configure your chatbot')
            OPENAI_API_KEY = st.sidebar.text_input("Enter your OpenAI API key to start the bot:", type='password', key='api_key')
            
            loader = WebBaseLoader("https://sweetspot.straitstimes.com/education/sharpen-your-business-acumen-with-nus-iss/")

            data = loader.load()

        #     OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectors = FAISS.from_documents(data, embeddings)

            chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY),
                                                                                retriever=vectors.as_retriever())

            def conversational_chat(query):
                    
                    result = chain({"question": query, "chat_history": st.session_state['history']})
                    st.session_state['history'].append((query, result["answer"]))
                    
                    return result["answer"]
                
            if 'history' not in st.session_state:
                    st.session_state['history'] = []

            if 'generated' not in st.session_state:
                    st.session_state['generated'] = ["Hello! My name is EBAC Chatbot! Ask me questions about the NUS-ISS EBAC here!" + " 🤗"]

            if 'past' not in st.session_state:
                    st.session_state['past'] = ["Hey ! 👋"]
                    
            #container for the chat history
            response_container = st.container()
            #container for the user's text input
            container = st.container()

            with container:
                    with st.form(key='my_form', clear_on_submit=True):
                        
                        user_input = st.text_input("Query:", placeholder="Ask me questions about the NUS-ISS EBAC here (:", key='input')
                        submit_button = st.form_submit_button(label='Send')
                        
                    if submit_button and user_input:
                        output = conversational_chat(user_input)
                        
                        st.session_state['past'].append(user_input)
                        st.session_state['generated'].append(output)

            if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                            message(st.session_state["generated"][i], key=str(i), avatar_style="big-smile")

if __name__ == '__main__':
  ophai_chat = chatbot()
  st.header("Welcome to NUS-ISS EBAC Chatbot")
  '''
  -------------------------------
  '''
  st.subheader("Built with OpenAI")
  ophai_chat.ophai_chatbot()
