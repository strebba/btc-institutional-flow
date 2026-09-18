import streamlit as st

from src.dashboard.tabs.validation import _tab_validation

_tab_validation(st.session_state["merged_df"], st.session_state["barriers"])
