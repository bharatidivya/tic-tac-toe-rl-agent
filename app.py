import streamlit as st
import numpy as np
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="Reinforcement Learning Tic-Tac-Toe Arena",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #ffffff;
        background: linear-gradient(90deg, #FF4B4B, #FF914D);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        font-size: 30px !important;
        font-weight: bold !important;
        height: 85px !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

#
