#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 14:53:52 2025

@author: limeilangdemacbook
"""

import streamlit as st
import pandas as pd
from functions import topic_data, summarize_youtube_video, main

st.set_page_config(page_title="YouTube AI Toolkit", layout="wide")
st.title("📺 YouTube Videos Explorer")

page = st.sidebar.selectbox("What can I help you with?", ["👁️Explore a topic on Youtube", "🧐Summarize a Youtube video", "💬Discover public sentiment under a video"])
API_KEY = st.sidebar.text_input("🔑 Enter YouTube API Key", type="password")

if page == "👁️Explore a topic on Youtube":
    st.header("🔍 Search for popular videos under this topic")
    search_query = st.text_input("Enter your keyword", value=" ")
    max_results = st.slider("Number of videos collected", 5, 50, 10)
    video_duration = st.selectbox("Customize duration of videos targetting", [None, "short", "medium", "long"])
    published_after = st.date_input("Search for videos published after")
    published_before = st.date_input("Search for videos published before")

    if st.button("🔎 save results as a csv file"):
        output_path = "search_results.csv"
        topic_data(
            api_key=API_KEY,
            search_query=search_query,
            max_results=max_results,
            video_duration=video_duration,
            published_after=published_after.isoformat() + "T00:00:00Z",
            published_before=published_before.isoformat() + "T00:00:00Z",
            output_path=output_path
        )
        df = pd.read_csv(output_path)
        st.success("finished!")
        st.dataframe(df)
        st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="youtube_search.csv")

elif page == "🧐Summarize a YouTube video":
    st.header("🧠 Generate a summary for any YouTube video")
    video_id = st.text_input("Enter a video ID")
    if st.button("📄 Summarize it!"):
        with st.spinner("working on it..."):
            summary = summarize_youtube_video(video_id)
            st.success("finished!")
            st.text_area("📄 Summary", summary, height=300)

elif page == "💬Discover public sentiment under a video":
    st.header("💬 What's the public reflection towards a YouTube video?")
    video_id = st.text_input("Enter a video ID", value=" ")
    max_comments = st.slider("Number of comments collected", 10, 200, 100)
    order = st.selectbox("Collecting comments according to...", ["relevance", "time"])

    if st.button("🚀 Start analysing"):
        with st.spinner("Working on it..."):
            df = main(video_id=video_id, max_comments=max_comments, order=order, api_key=API_KEY)
        st.success("finished!")
        st.dataframe(df)
        st.download_button("📥 Download the results", data=df.to_csv(index=False), file_name="youtube_emotion_analysis.csv")
