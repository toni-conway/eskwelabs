import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import openai
import joblib

from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
import spacy
import contractions
import string

from wordcloud import WordCloud

st.image('s4g4-waits-moodguard-banner.png')
st.write("MoodGuard is not a replacement for professional mental health guidance; rather, it is an exploration of how LLMs can contribute to mental health services. Seeking assistance from professionals is highly recommended")
st.write("Instructions: Upload the input file through the sidebar and then select the \"Start Analyze Data\" button.")

#--------------------------------------------------------------------------------------------------

#######################################################
# Initialize session state
#######################################################

# First Initialization
if "nlp" not in st.session_state:
    st.session_state.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

    nltk.download('punkt')     # Downloads the Punkt tokenizer models
    nltk.download('stopwords') # Downloads the list of stopwords
    nltk.download('wordnet')   # Downloads the WordNet lemmatizer data
    nltk.download('averaged_perceptron_tagger')

# Open AI Model
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"

    # Set OpenAI Keys
    openai.api_key = st.secrets["OPENAI-API-KEY"]
    SKLLMConfig.set_openai_key(openai.api_key)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.GPT_instuction = """
        You are warmth and approachable mental health expert and therapist, your expertise is in helping people in thier teens overcome obstacle
        regarding motivation, career, school, relationships and self esteem and you have done this for a few decades. Your task is to provide the best advice for
        helping improve mental health. Answer in concise and bullet form. Format your response for markdown processor
        """
        


#--------------------------------------------------------------------------------------------------

#######################################################
# MAIN Program
#######################################################
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://drive.google.com/file/d/18honVLHoQZ5iFU_zz6P51ASyC8ajvByL/view?usp=drive_link)
""")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    #st.sidebar.button("Start Analyzing Data", on_click=analyze_data(input_df))


#--------------------------------------------------------------------------------------------------
