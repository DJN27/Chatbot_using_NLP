import os
import json
import datetime
import csv
import ssl
import random
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ssl._create_default_https_context=ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt", quiet=True)
file_path=os.path.abspath("intents.json")
with open(file_path,"r",encoding="utf-8") as file:
    intents=json.load(file)
tags,patterns=zip(*[(intent["tag"],pattern)for intent in intents for pattern in intent["patterns"]])
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(patterns)
clf=LogisticRegression(random_state=0,max_iter=10000)
clf.fit(X,tags)
def chatbot(input_text):
    input_vector=vectorizer.transform([input_text])
    tag=clf.predict(input_vector)[0]
    return random.choice(next(intent["responses"]for intent in intents if intent["tag"]==tag))
def log_chat(user_input,response):
    timestamp=datetime.datetime.now().strftime(f"%d-%m-%Y %H:%M:%S")
    with open("chat_log.csv","a",newline="",encoding="utf-8")as csvfile:
        csv.writer(csvfile).writerow([user_input,response,timestamp])
def display_chat_history():
    if os.path.exists("chat_log.csv"):
        with open("chat_log.csv","r",encoding="utf-8")as csvfile:
            csv_reader=csv.reader(csvfile)
            next(csv_reader,None)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    else:
        st.write("No conversation history found.")
def main():
    st.title("Chatbot with NLP & Logistic Regression")
    menu=["Home","Conversation History","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="Home":
        st.write("Welcome! Ask your queries below.")
        user_input=st.text_input("You:")
        if user_input:
            response=chatbot(user_input)
            st.text_area("Chatbot:",value=response,height=120)
            log_chat(user_input,response)
            if response.lower()in["goodbye","bye","seeya"]:
                st.write("Thank you for chatting! Have a great day!")
                st.stop()
    elif choice=="Conversation History":
        st.header("Conversation History")
        display_chat_history()
    elif choice=="About":
        st.subheader("About the Project")
        st.write("""
        This chatbot is built using Natural Language Processing (NLP) and Logistic Regression for intent classification. 
        It provides a user-friendly interface using Streamlit.
        
        **Features:**
        - Trained on predefined intents for accurate responses.
        - Interactive web-based interface.
        - Saves conversation history for later reference.
        
        **Future Enhancements:**
        - Expanding dataset for better accuracy.
        - Implementing more advanced NLP models.
        - Exploring deep learning for improved understanding.
        
        This project demonstrates the integration of NLP techniques with machine learning to create an intelligent chatbot.
        """)
if __name__=="__main__":
    main()