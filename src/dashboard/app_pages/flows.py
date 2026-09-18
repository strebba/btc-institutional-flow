import streamlit as st

from src.dashboard.tabs.flows import _tab_flows

_tab_flows(st.session_state["merged_df"])
