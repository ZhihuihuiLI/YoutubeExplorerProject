#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 14:46:13 2025

@author: limeilangdemacbook
"""
import requests
import pandas as pd
import time
from transformers import pipeline
import re
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from nltk.probability import FreqDist
from wordcloud import WordCloud
from collections import Counter
import nltk
import streamlit as st
##############################################
## Function 1: Popular videos under a topic ##
##############################################
def topic_data(api_key, search_query, max_results, output_path,
               published_after=None, published_before=None, video_duration=None):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "snippet",
        "q": search_query,
        "type": "video",
        "maxResults": max_results,
        "order": "viewCount",
        "relevanceLanguage": "en",
        "videoDuration": video_duration,
        "publishedAfter": published_after,
        "publishedBefore": published_before,
        "key": api_key
    }
    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()
    video_ids = [item["id"]["videoId"] for item in search_data.get("items", [])]

    if not video_ids:
        print("no video found")
        return

    videos_url = "https://www.googleapis.com/youtube/v3/videos"
    videos_params = {
        "part": "snippet,statistics",
        "id": ",".join(video_ids),
        "key": api_key
    }
    video_response = requests.get(videos_url, params=videos_params)
    video_data = video_response.json()

    videos_info = []
    channel_ids = set()

    for item in video_data["items"]:
        video = {
            'video_id': item["id"],
            'title': item["snippet"].get("title", ""),
            'published_date': item["snippet"]["publishedAt"],
            'views_count': int(item["statistics"].get("viewCount", 0)),
            'likes': int(item["statistics"].get("likeCount", 0)),
            'commentCount': int(item["statistics"].get("commentCount", 0)),
            'channel_id': item["snippet"]["channelId"],
            'channel_title': item["snippet"]["channelTitle"]
        }
        channel_ids.add(video["channel_id"])
        videos_info.append(video)

    channel_info_map = {}
    channel_id_list = list(channel_ids)
    channels_url = "https://www.googleapis.com/youtube/v3/channels"

    for i in range(0, len(channel_id_list), 50):
        batch = channel_id_list[i:i+50]
        channels_params = {
            "part": "statistics,brandingSettings",
            "id": ",".join(batch),
            "key": api_key
        }
        channel_response = requests.get(channels_url, params=channels_params)
        channel_data = channel_response.json()
        for item in channel_data.get("items", []):
            cid = item["id"]
            subscribers = int(item["statistics"].get("subscriberCount", 0))
            keywords = item["brandingSettings"]["channel"].get("keywords", "")
            channel_info_map[cid] = {
                "subscribers": subscribers,
                "keywords": keywords
            }
        time.sleep(1)

    for video in videos_info:
        cid = video["channel_id"]
        video["channel_subscribers"] = channel_info_map.get(cid, {}).get("subscribers", 0)
        video["channel_keywords"] = channel_info_map.get(cid, {}).get("keywords", "")

    df = pd.DataFrame(videos_info)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("file saved!")
###################################
## Function 2: Summarize a video ##
###################################
MODEL_NAME = "facebook/bart-large-cnn"
MAX_TOKENS_PER_CHUNK = 1000
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 50

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'\b(music|applause|laughter)\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chunks(text, max_char_length=4000):
    sentences = text.split('.')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_char_length:
            chunk += sentence + '.'
        else:
            chunks.append(chunk.strip())
            chunk = sentence + '.'
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def summarize_youtube_video(video_id, language_code='en'):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
    text = ' '.join([line['text'] for line in transcript])
    cleaned_text = clean_text(text)
    chunks = split_into_chunks(cleaned_text)
    summarizer = pipeline("summarization", model=MODEL_NAME)
    partial_summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=MAX_SUMMARY_LENGTH,
                             min_length=MIN_SUMMARY_LENGTH, do_sample=False)[0]['summary_text']
        partial_summaries.append(summary)
    if len(partial_summaries) > 1:
        final_summary = summarizer(" ".join(partial_summaries),
                                   max_length=MAX_SUMMARY_LENGTH,
                                   min_length=MIN_SUMMARY_LENGTH, do_sample=False)[0]['summary_text']
    else:
        final_summary = partial_summaries[0]
    return final_summary
####################################
##  Function 3: Comment analysis  ##
####################################
emotion_model = pipeline("sentiment-analysis")

def get_comments(video_id, max_comments=100, order='relevance', api_key=None):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=min(100, max_comments - len(comments)),
            textFormat='plainText',
            order=order
        ).execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comments

def analyze_emotions(comments):
    results = []
    for comment in comments:
        try:
            result = emotion_model(comment)[0]
            results.append({
                'comment': comment,
                'sentiment': result['label'],
                'score': result['score']
            })
        except:
            continue
    return pd.DataFrame(results)

def preprocessing_text(comments):
    text = ' '.join(comments).lower()
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    content_words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    stems = [stemmer.stem(word) for word in content_words]
    pos_tags = pos_tag(content_words)
    def lookup_pos(pos):
        tag = pos[0].lower()
        return tag if tag in ('a', 'n', 'v', 'r') else 'n'
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, lookup_pos(pos)) for word, pos in pos_tags]
    frequencies = FreqDist(lemmas)
    return frequencies

def generate_wordcloud(frequencies):
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(frequencies)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Keyword WordCloud from Comments', fontsize=16)
    st.pyplot(plt)

def main(video_id, max_comments=150, order='relevance', api_key=None):
    comments = get_comments(video_id, max_comments=max_comments, order=order, api_key=api_key)
    df = analyze_emotions(comments)
    frequencies = preprocessing_text(comments)
    generate_wordcloud(frequencies)
    return df
