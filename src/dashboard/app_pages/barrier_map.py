import streamlit as st

from src.dashboard.tabs.barrier_map import _tab_barrier_map

_tab_barrier_map(st.session_state["barriers"], st.session_state["snap"])
