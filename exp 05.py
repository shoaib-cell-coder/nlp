import streamlit as st
import nltk
import pandas as pd
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize




# =========================================================
# NLTK SETUP
# =========================================================


@st.cache_resource
def ensure_nltk_data():
   """Download required NLTK tokenizer packages only once."""
   for pkg in ("punkt", "punkt_tab"):
       try:
           nltk.download(pkg, quiet=True)
       except Exception:
           pass




ensure_nltk_data()




# =========================================================
# PAGE CONFIGURATION
# =========================================================


st.set_page_config(
   page_title="N-Gram Probability Analyzer",
   page_icon="📊",
   layout="wide"
)




# =========================================================
# CUSTOM CSS
# =========================================================


st.markdown(
   """
   <style>


   .block-container {
       padding-top: 2rem;
   }


   div[data-testid="stMetricValue"] {
       font-size: 1.8rem;
       font-weight: 700;
   }


   .ngram-pill {
       display: inline-block;
       background: #eef2ff;
       color: #3730a3;
       border-radius: 999px;
       padding: 6px 14px;
       margin: 4px;
       font-family: monospace;
       font-size: 0.9rem;
       border: 1px solid #c7d2fe;
   }


   </style>
   """,
   unsafe_allow_html=True
)




# =========================================================
# EXAMPLE TEXT
# =========================================================


EXAMPLE_TEXT = (
   "I love natural language processing. "
   "I love machine learning. "
   "I love deep learning. "
   "I enjoy natural language processing. "
   "I enjoy machine learning. "
   "Natural language processing is interesting. "
   "Machine learning is useful. "
   "Deep learning is powerful. "
   "I love learning new things."
)




# =========================================================
# SESSION STATE
# =========================================================


if "text_content" not in st.session_state:
   st.session_state.text_content = EXAMPLE_TEXT




# =========================================================
# N-GRAM ANALYSIS
# =========================================================


def run_analysis(raw_text, remove_punct):


   # Convert to lowercase
   text = raw_text.lower()


   # Tokenize
   tokens = word_tokenize(text)


   # Remove punctuation
   if remove_punct:
       tokens = [
           token
           for token in tokens
           if any(character.isalnum() for character in token)
       ]


   # Create N-grams
   unigrams = list(ngrams(tokens, 1))
   bigrams = list(ngrams(tokens, 2))
   trigrams = list(ngrams(tokens, 3))


   # Count N-grams
   unigram_counts = Counter(unigrams)
   bigram_counts = Counter(bigrams)
   trigram_counts = Counter(trigrams)


   total_words = len(tokens)




   # =====================================================
   # UNIGRAM PROBABILITY
   # P(w) = Count(w) / Total Words
   # =====================================================


   unigram_rows = []


   if total_words > 0:


       for unigram, count in unigram_counts.items():


           probability = count / total_words


           unigram_rows.append({
               "Unigram": unigram[0],
               "Count": count,
               "Probability": round(probability, 4)
           })




   # =====================================================
   # BIGRAM PROBABILITY
   # P(w2 | w1) = Count(w1,w2) / Count(w1)
   # =====================================================


   bigram_rows = []


   for (w1, w2), count in bigram_counts.items():


       previous_count = unigram_counts[(w1,)]


       probability = count / previous_count


       bigram_rows.append({
           "Bigram": f"{w1} {w2}",
           "Count": count,
           "Previous Word Count": previous_count,
           "Probability": round(probability, 4)
       })




   # =====================================================
   # TRIGRAM PROBABILITY
   # P(w3 | w1,w2)
   # =
   # Count(w1,w2,w3) / Count(w1,w2)
   # =====================================================


   trigram_rows = []


   for (w1, w2, w3), count in trigram_counts.items():


       previous_bigram_count = bigram_counts[(w1, w2)]


       probability = count / previous_bigram_count


       trigram_rows.append({
           "Trigram": f"{w1} {w2} {w3}",
           "Count": count,
           "Previous Bigram Count": previous_bigram_count,
           "Probability": round(probability, 4)
       })




   # =====================================================
   # DATAFRAMES
   # =====================================================


   unigram_df = pd.DataFrame(unigram_rows)
   bigram_df = pd.DataFrame(bigram_rows)
   trigram_df = pd.DataFrame(trigram_rows)




   # Sort by probability


   if not unigram_df.empty:


       unigram_df = (
           unigram_df
           .sort_values("Probability", ascending=False)
           .reset_index(drop=True)
       )


   if not bigram_df.empty:


       bigram_df = (
           bigram_df
           .sort_values("Probability", ascending=False)
           .reset_index(drop=True)
       )


   if not trigram_df.empty:


       trigram_df = (
           trigram_df
           .sort_values("Probability", ascending=False)
           .reset_index(drop=True)
       )




   return {
       "tokens": tokens,
       "total_words": total_words,
       "unigram_counts": unigram_counts,
       "bigram_counts": bigram_counts,
       "trigram_counts": trigram_counts,
       "unigram_df": unigram_df,
       "bigram_df": bigram_df,
       "trigram_df": trigram_df
   }




# =========================================================
# NEXT WORD PREDICTION
# =========================================================


def predict_next_word(analysis, phrase, top_n=8):


   phrase_tokens = word_tokenize(
       phrase.lower()
   )


   if not phrase_tokens:
       return None


   unigram_counts = analysis["unigram_counts"]
   bigram_counts = analysis["bigram_counts"]
   trigram_counts = analysis["trigram_counts"]




   # =====================================================
   # TRIGRAM PREDICTION
   # =====================================================


   if len(phrase_tokens) >= 2:


       w1 = phrase_tokens[-2]
       w2 = phrase_tokens[-1]


       candidates = []


       for (a, b, w3), count in trigram_counts.items():


           if a == w1 and b == w2:


               probability = (
                   count /
                   bigram_counts[(w1, w2)]
               )


               candidates.append(
                   (w3, probability)
               )




       if candidates:


           candidates.sort(
               key=lambda x: x[1],
               reverse=True
           )


           return {
               "context": f"{w1} {w2}",
               "model": "Trigram",
               "candidates": candidates[:top_n]
           }




   # =====================================================
   # BIGRAM FALLBACK
   # =====================================================


   last_word = phrase_tokens[-1]


   if (last_word,) in unigram_counts:


       candidates = []


       for (w1, w2), count in bigram_counts.items():


           if w1 == last_word:


               probability = (
                   count /
                   unigram_counts[(w1,)]
               )


               candidates.append(
                   (w2, probability)
               )




       if candidates:


           candidates.sort(
               key=lambda x: x[1],
               reverse=True
           )


           return {
               "context": last_word,
               "model": "Bigram",
               "candidates": candidates[:top_n]
           }




   return None




# =========================================================
# CSV DOWNLOAD
# =========================================================


def download_csv(df, label, filename):


   st.download_button(
       label=f"⬇️ Download {label} CSV",
       data=df.to_csv(index=False).encode("utf-8"),
       file_name=filename,
       mime="text/csv",
       use_container_width=True
   )




# =========================================================
# SIDEBAR
# =========================================================


st.sidebar.header("🔧 Settings")




remove_punctuation = st.sidebar.checkbox(
   "Remove Punctuation Tokens",
   value=True
)




top_predictions = st.sidebar.slider(
   "Maximum Suggestions",
   min_value=3,
   max_value=15,
   value=8
)




rows_to_display = st.sidebar.slider(
   "Rows to Display",
   min_value=5,
   max_value=100,
   value=20
)




chart_items = st.sidebar.slider(
   "Chart Items",
   min_value=5,
   max_value=20,
   value=10
)




# =========================================================
# TITLE
# =========================================================


st.title(
   "📊 N-Gram Probability Analyzer & Predictor"
)


st.write(
   "Analyze text using Unigrams, Bigrams and Trigrams, "
   "calculate their probabilities, and predict the next word."
)




# =========================================================
# TEXT INPUT
# =========================================================


col1, col2 = st.columns([4, 1])




with col1:


   text_input = st.text_area(
       "📝 Input Corpus Text",
       value=st.session_state.text_content,
       height=180
   )




with col2:


   st.write("### Text Controls")




   if st.button(
       "🗑️ Clear Input",
       use_container_width=True
   ):


       st.session_state.text_content = ""


       st.rerun()




   if st.button(
       "📋 Load Example",
       use_container_width=True
   ):


       st.session_state.text_content = EXAMPLE_TEXT


       st.rerun()




# =========================================================
# ANALYSIS
# =========================================================


if text_input.strip():


   analysis = run_analysis(
       text_input,
       remove_punctuation
   )




   # =====================================================
   # SUMMARY
   # =====================================================


   st.divider()


   st.subheader("📈 Corpus Summary")




   m1, m2, m3, m4 = st.columns(4)




   with m1:


       st.metric(
           "Total Tokens",
           analysis["total_words"]
       )




   with m2:


       st.metric(
           "Unique Unigrams",
           len(analysis["unigram_counts"])
       )




   with m3:


       st.metric(
           "Unique Bigrams",
           len(analysis["bigram_counts"])
       )




   with m4:


       st.metric(
           "Unique Trigrams",
           len(analysis["trigram_counts"])
       )




   # =====================================================
   # TABS
   # =====================================================


   tab_tokens, tab_uni, tab_bi, tab_tri, tab_predict = st.tabs(
       [
           "🔤 Tokens",
           "1️⃣ Unigrams",
           "2️⃣ Bigrams",
           "3️⃣ Trigrams",
           "🔮 Next Word Prediction"
       ]
   )




   # =====================================================
   # TOKENS TAB
   # =====================================================


   with tab_tokens:


       st.subheader("🔤 Tokenized Text")


       if analysis["tokens"]:


           pills_html = ""


           for token in analysis["tokens"]:


               pills_html += (
                   f'<span class="ngram-pill">'
                   f'{token}'
                   f'</span>'
               )


           st.markdown(
               pills_html,
               unsafe_allow_html=True
           )


       else:


           st.warning(
               "No tokens available."
           )




   # =====================================================
   # UNIGRAM TAB
   # =====================================================


   with tab_uni:


       st.subheader(
           "1️⃣ Unigram Probabilities"
       )


       st.write(
           "Formula: "
           "**P(w) = Count(w) / Total Words**"
       )




       unigram_df = analysis["unigram_df"]




       if unigram_df.empty:


           st.warning(
               "No unigram data available."
           )


       else:


           st.dataframe(
               unigram_df.head(rows_to_display),
               use_container_width=True,
               hide_index=True
           )




           st.subheader(
               "📊 Unigram Probability Chart"
           )




           chart_data = (
               unigram_df
               .head(chart_items)
               .set_index("Unigram")["Probability"]
           )




           st.bar_chart(chart_data)




           download_csv(
               unigram_df,
               "Unigrams",
               "unigrams.csv"
           )




   # =====================================================
   # BIGRAM TAB
   # =====================================================


   with tab_bi:


       st.subheader(
           "2️⃣ Bigram Probabilities"
       )


       st.write(
           "Formula: "
           "**P(w₂ | w₁) = Count(w₁,w₂) / Count(w₁)**"
       )




       bigram_df = analysis["bigram_df"]




       if bigram_df.empty:


           st.warning(
               "Not enough tokens to create bigrams."
           )


       else:


           st.dataframe(
               bigram_df.head(rows_to_display),
               use_container_width=True,
               hide_index=True
           )




           st.subheader(
               "📊 Bigram Probability Chart"
           )




           chart_data = (
               bigram_df
               .head(chart_items)
               .set_index("Bigram")["Probability"]
           )




           st.bar_chart(chart_data)




           download_csv(
               bigram_df,
               "Bigrams",
               "bigrams.csv"
           )




   # =====================================================
   # TRIGRAM TAB
   # =====================================================


   with tab_tri:


       st.subheader(
           "3️⃣ Trigram Probabilities"
       )


       st.write(
           "Formula: "
           "**P(w₃ | w₁,w₂) = "
           "Count(w₁,w₂,w₃) / Count(w₁,w₂)**"
       )




       trigram_df = analysis["trigram_df"]




       if trigram_df.empty:


           st.warning(
               "Not enough tokens to create trigrams."
           )


       else:


           st.dataframe(
               trigram_df.head(rows_to_display),
               use_container_width=True,
               hide_index=True
           )




           st.subheader(
               "📊 Trigram Probability Chart"
           )




           chart_data = (
               trigram_df
               .head(chart_items)
               .set_index("Trigram")["Probability"]
           )




           st.bar_chart(chart_data)




           download_csv(
               trigram_df,
               "Trigrams",
               "trigrams.csv"
           )




   # =====================================================
   # NEXT WORD PREDICTION TAB
   # =====================================================


   with tab_predict:


       st.subheader(
           "🔮 Predict Next Word"
       )




       st.write(
           "Enter one or more words. "
           "The system first tries a **Trigram** model "
           "using the last two words. If no trigram match "
           "is found, it uses a **Bigram** model."
       )




       prediction_phrase = st.text_input(
           "Enter context:",
           value="I love",
           placeholder="Example: I love"
       )




       if prediction_phrase.strip():


           prediction = predict_next_word(
               analysis,
               prediction_phrase,
               top_n=top_predictions
           )




           if prediction is None:


               st.warning(
                   "No predictive matches found. "
                   "Try another phrase that appears "
                   "in the training text."
               )




           else:


               best_word, best_probability = (
                   prediction["candidates"][0]
               )




               st.success(
                   f"Predicted Next Word: "
                   f"**{best_word}**"
               )




               p1, p2 = st.columns(2)




               with p1:


                   st.metric(
                       "Predicted Word",
                       best_word
                   )




               with p2:


                   st.metric(
                       "Probability",
                       f"{best_probability:.4f}"
                   )




               st.info(
                   f"Model used: **{prediction['model']}**  \n"
                   f"Context: **{prediction['context']}**"
               )




               # -----------------------------------------
               # POSSIBLE NEXT WORDS
               # -----------------------------------------


               st.subheader(
                   "📋 Possible Next Words"
               )




               prediction_df = pd.DataFrame(
                   prediction["candidates"],
                   columns=[
                       "Next Word",
                       "Probability"
                   ]
               )




               prediction_df["Probability"] = (
                   prediction_df["Probability"]
                   .round(4)
               )




               st.dataframe(
                   prediction_df,
                   use_container_width=True,
                   hide_index=True
               )




               st.subheader(
                   "📊 Prediction Probability Chart"
               )




               st.bar_chart(
                   prediction_df
                   .set_index("Next Word")["Probability"]
               )




# =========================================================
# EMPTY INPUT MESSAGE
# =========================================================


else:


   st.info(
       "Enter text in the input box above "
       "to start the N-gram analysis."
   )




# =========================================================
# FOOTER
# =========================================================


st.divider()


st.caption(
   "N-Gram Probability Analyzer | "
   "NLP Practical Project"
)
