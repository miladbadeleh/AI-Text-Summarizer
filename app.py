# app.py
import streamlit as st
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import requests
from rouge_score import rouge_scorer




# app.py (continued)
def text_summarizer_spacy(raw_doc, summary_percent=0.3):
    """
    Extractive summarization using word frequency.
    """
    # Load the model
    nlp = spacy.load('en_core_web_sm')
    # Process the text
    doc = nlp(raw_doc)

    # Calculate word frequencies, ignoring stop words and punctuation
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS) and word.text.lower() not in punctuation and word.text not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    # Normalize frequencies
    max_frequency = max(word_frequencies.values()) if word_frequencies else 1
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Score sentences based on word frequencies
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_frequencies:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text]
                else:
                    sentence_scores[sent] += word_frequencies[word.text]

    # Select top sentences based on the summary percentage
    select_length = int(len(list(doc.sents)) * summary_percent)
    summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([sent.text for sent in summary_sentences])
    return summary





# app.py (continued)
def summarize_bart(long_text, max_length=130, min_length=30):
    """
    Abstractive summarization using Hugging Face's BART model.
    """
    # Use the pipeline API for simplicity
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Handle long text by chunking (BART has a max input length)
    # This is a simple chunking approach. For a robust project, you'd need more complex handling.
    if len(long_text) > 1024:
        long_text = long_text[:1024] # Truncate for demo purposes. Better: split into chunks and summarize recursively.

    summary = summarizer(long_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']






# app.py (continued)
def scrape_text_from_url(url):
    """
    A simple function to scrape text from a URL. For demo purposes.
    In a real project, use a dedicated library like newspaper3k or bs4.
    """
    try:
        response = requests.get(url)
        response.raise_for_status() # Raises an error for bad status codes

        # Very basic text extraction. This will get messy HTML.
        # For a portfolio, it's okay to say "this is a simple demo scraper".
        return response.text[:5000] # Limit text for demo
    except requests.exceptions.RequestException as e:
        return f"Error: Unable to fetch URL. {e}"





# app.py (continued)
def main():
    st.set_page_config(page_title="Text Summarizer Pro", page_icon="✂️")
    st.title("✂️ Text Summarizer Pro")
    st.markdown("Extract key insights from long articles and documents using AI.")

    # Input method selection
    input_method = st.radio("Choose your input method:", ("Enter Text", "Enter URL"))

    raw_text = ""
    if input_method == "Enter Text":
        raw_text = st.text_area("Paste your text here:", height=250, placeholder="Paste a long article, report, or any text you want to summarize...")
    else:
        url = st.text_input("Paste the article URL here:", placeholder="https://example.com/article/")
        if url:
            with st.spinner("Scraping content from URL..."):
                raw_text = scrape_text_from_url(url)
            if raw_text.startswith("Error"):
                st.error(raw_text)
            else:
                st.text_area("Scraped Text (Preview):", value=raw_text[:500] + "...", height=150)

    # Model selection
    summarization_type = st.selectbox(
        "Choose summarization method:",
        ("BART (Abstractive - AI-Powered)", "SpaCy (Extractive - Frequency-based)")
    )

    # Summary length slider (only show for BART)
    if summarization_type.startswith("BART"):
        max_len = st.slider("Maximum summary length (words):", min_value=50, max_value=200, value=130)
        min_len = st.slider("Minimum summary length (words):", min_value=10, max_value=100, value=30)

    # Process button
    if st.button("Generate Summary") and raw_text:
        with st.spinner(f'Generating summary using {summarization_type}...'):
            try:
                if summarization_type.startswith("BART"):
                    summary = summarize_bart(raw_text, max_length=max_len, min_length=min_len)
                else:
                    summary = text_summarizer_spacy(raw_text)

                # Display results
                st.subheader("📋 Summary")
                st.success(summary)

                # Show statistics
                col1, col2, col3 = st.columns(3)
                original_words = len(raw_text.split())
                summary_words = len(summary.split())
                reduction = (1 - (summary_words / original_words)) * 100 if original_words > 0 else 0

                col1.metric("Original Length", f"{original_words} words")
                col2.metric("Summary Length", f"{summary_words} words")
                col3.metric("Reduction", f"{reduction:.1f}%")

            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool demonstrates two approaches to **Automatic Text Summarization**:

        - **🤖 BART (Abstractive)**: Understands meaning and generates new sentences. *More fluent, like a human.*
        - **🔍 SpaCy (Extractive)**: Selects key sentences from the text. *More factual, faster.*

        Built with `spaCy`, `transformers`, and `streamlit`.
        """)

if __name__ == "__main__":
    main()
