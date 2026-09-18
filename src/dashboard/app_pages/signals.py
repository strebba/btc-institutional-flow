import streamlit as st

from src.dashboard.tabs.signals import _tab_signals

_tab_signals(
    st.session_state["snap"],
    st.session_state["merged_df"],
    st.session_state["barriers"],
)
