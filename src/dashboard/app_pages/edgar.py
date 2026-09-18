import streamlit as st

from src.dashboard.tabs.edgar import _tab_edgar_monitor

_tab_edgar_monitor(st.session_state["barriers"], st.session_state["merged_df"])
