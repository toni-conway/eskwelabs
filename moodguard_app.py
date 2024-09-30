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
#import spacy
import contractions
import string

from wordcloud import WordCloud




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
