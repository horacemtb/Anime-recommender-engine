#to run: streamlit run web-ui.py

import streamlit as st
import pandas as pd
import requests

with st.sidebar:
    st.image("https://cdn.realsport101.com/images/ncavvykf/gfinityesports/dd7e707686dad010341c1fc6bbfab1f715ebe155-1920x1080.jpg?rect=1,0,1919,1080&w=700&h=394&dpr=2")
    st.title("My Otaku Friend")
    st.info("This is an anime social networking and cataloging website that lets you share your feedback on anime and uses AI to give you recommendations for new anime based on your preferences and product similarity")
    st.info("There are currently 14353 unique anime and 13102 registered users on our portal!")
    choice = st.radio("Navigation", ["Login/Signup", "Preferences", "Find similar"])


if choice == "Login/Signup":
    st.title("Log in if you have an account or Sign up as a new user")
    st.write("Expected format: 1565")
    user_id = st.text_input("Enter your user_id")
    
    if user_id:
        try:
            r = requests.post('http://158.160.42.24:80/recommend/', json={'user_input': int(user_id), 'user_preferences': {}},
            headers={'content-type': 'application/json'}, timeout = 10)
            
            j = r.json()

            try:
                watched = pd.read_json(j['watched'])
                recs = pd.read_json(j['recommendations'])
                st.write("You watched:")
                st.dataframe(watched[['Name', 'rating']].reset_index(drop=True))
                st.write("You might like:")
                st.dataframe(recs[['Name']].reset_index(drop=True))
            
            except:
                recs = pd.read_json(j['recommendations'])
                st.write("Looks like you're a new user! You might like:")
                st.dataframe(recs[['Name']].reset_index(drop=True))
        except:
            st.write("Looks like something went wrong. Please make sure your input is in the right format. If this doesn't fix the issue, try again later")

    else:
        st.write("Your recommendations will show up here")

if choice == "Preferences":
    st.title("Enter your genre preferences here to get recommendations")
    st.write("Expected format: Action, Adventure, School")
    st.write("Available genres: Action, Adventure, Cars, Comedy, Dementia, Demons, Drama, Ecchi, Fantasy, Game, Harem, Hentai,\
    Historical, Horror, Josei, Kids, Magic, Martial Arts, Mecha, Military, Music, Mystery, Parody, Police,\
    Psychological, Romance, Samurai, School, Sci-Fi, Seinen, Shoujo, Shoujo Ai, Shounen, Shounen Ai, Slice of Life,\
    Space, Sports, Super Power, Supernatural, Thriller, Unknown, Vampire, Yaoi, Yuri")
    user_pref = st.text_input("Enter your genres")

    if user_pref:
        try:
            user_pref_dict = {}
            user_pref = user_pref.split(',')
            for pref in user_pref:
                pref = pref.strip()
                if pref in ['Action', 'Adventure', 'Cars', 'Comedy', 'Dementia',
                'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai',
                'Historical', 'Horror', 'Josei', 'Kids', 'Magic', 'Martial Arts',
                'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police',
                'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen',
                'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai', 'Slice of Life',
                'Space', 'Sports', 'Super Power', 'Supernatural', 'Thriller', 'Unknown',
                'Vampire', 'Yaoi', 'Yuri']:
                    user_pref_dict[pref] = [1]

            r = requests.post('http://158.160.42.24:80/recommend/', json={'user_input': 1234567890, 'user_preferences': user_pref_dict},
            headers={'content-type': 'application/json'}, timeout = 10)
            
            j = r.json()

            recs = pd.read_json(j['recommendations'])
            st.write("Based on the genres you entered, you might like:")
            st.dataframe(recs[['Name']].reset_index(drop=True))
        except:
            st.write("Looks like something went wrong. Please make sure your input is in the right format. If this doesn't fix the issue, try again later")
        
    else:
        st.write("Your recommendations will show up here")

if choice == "Find similar":
    st.title("Enter your favorite anime here to find similar series")
    st.write("Expected format: Neon Genesis Evangelion")
    user_input = st.text_input("Enter your anime")

    if user_input:
        try:
            r = requests.post('http://158.160.42.24:80/recommend/', json={'user_input': user_input, 'user_preferences': {}},
            headers={'content-type': 'application/json'}, timeout = 10)
            
            j = r.json()

            recs = pd.read_json(j['recommendations'])
            st.write(f"If you like {user_input}, you might also like")
            st.dataframe(recs[['Name', 'Genres']].reset_index(drop=True))
        except:
            st.write("Looks like something went wrong. Please make sure your input is in the right format. If this doesn't fix the issue, try again later")
        
    else:
        st.write("Your recommendations will show up here")
