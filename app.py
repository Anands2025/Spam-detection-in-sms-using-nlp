import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page configuration and styling
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
    }
    .stTitle {
        color: #00b4d8;
        text-align: center;
        padding-bottom: 20px;
    }
    .stHeader {
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .spam {
        background-color: #dc3545;
    }
    .not-spam {
        background-color: #00b4d8;
    }
    .credit-text {
        text-align: center;
        color: #90e0ef;
        font-style: italic;
    }
    .stTextInput {
        background-color: #2d2d2d;
        border-radius: 5px;
        padding: 10px;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #00b4d8;
        color: white;
        width: 100%;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Centered title with emoji
st.markdown("<h1 style='text-align: center;'>📱 SMS Spam Detection Model</h1>", unsafe_allow_html=True)
st.markdown("<p class='credit-text'>Made by Anand S</p>", unsafe_allow_html=True)

# Add some spacing
st.write("")
st.write("")

# Create columns for better layout
col1, col2, col3 = st.columns([1,2,1])
with col2:
    input_sms = st.text_area("Enter the SMS message:", height=100)
    predict_button = st.button('Analyze Message')

# Add some spacing
st.write("")

if predict_button and input_sms:
    # Create a spinner while processing
    with st.spinner('Analyzing message...'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tk.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        st.write("")
        if result == 1:
            st.markdown("<h2 class='stHeader spam'>🚨 SPAM DETECTED!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 class='stHeader not-spam'>✅ NOT SPAM</h2>", unsafe_allow_html=True)
elif predict_button and not input_sms:
    st.error("Please enter a message to analyze!")

# Add footer
st.markdown("---")
st.markdown("<p class='credit-text'>© 2024 SMS Spam Detection System | Developed by Anand S</p>", unsafe_allow_html=True)
