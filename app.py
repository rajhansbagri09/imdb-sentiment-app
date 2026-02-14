import streamlit as st
import pickle

# load model and vectorizer
model = pickle.load(open("sentiment_model.pkl","rb"))
tfidf = pickle.load(open("vectorizer.pkl","rb"))

st.title("🎬 IMDB Movie Sentiment Analyzer")
st.caption("AI Powered Sentiment Analysis Web App")
st.write("Enter a movie review to check sentiment")


review = st.text_area("Enter Review")

if st.button("Predict Sentiment"):
    if review.strip()=="":
        st.warning("Please enter review")
    else:
        vector = tfidf.transform([review])
        result = model.predict(vector)

        if result[0] == 1:
            st.success("Positive Review 😄")
        else:
            st.error("Negative Review 😡")
st.markdown("---")
st.markdown(
"""
<div style="text-align:center; padding:10px; font-size:14px;">
© 2026 Rajhans Bagri | AI & ML Developer 🤖 <br>
Built with ❤️ using NLP, TF-IDF & Logistic Regression <br>
</div>
""",
unsafe_allow_html=True
)
