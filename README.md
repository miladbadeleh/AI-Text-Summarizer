# AI Text Summarizer

A web application that condenses long articles, documents, and web pages into concise summaries using Natural Language Processing.

## 🚀 Features

*   **Dual Summarization Techniques:**
    *   **Abstractive Summarization:** Uses Facebook's BART model to generate new, fluent paraphrases.
    *   **Extractive Summarization:** Uses spaCy to identify and extract the most important sentences based on word frequency.
*   **Web Scraping:** Can directly pull text from a provided URL for summarization.
*   **Interactive Web UI:** Built with Streamlit for a clean and user-friendly experience.
*   **Summary Metrics:** Displays original length, summary length, and compression ratio.

## 🛠️ Tech Stack

*   **NLP Libraries:** Hugging Face Transformers, spaCy
*   **Models:** facebook/bart-large-cnn, en_core_web_sm
*   **Web Framework:** Streamlit
*   **Evaluation:** ROUGE metrics (optional)

## 📦 Installation & Usage

1.  Clone the repo and install dependencies:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
2.  Run the application:
    ```bash
    streamlit run app.py
    ```
3.  Paste text or a URL, choose a method, and generate your summary!

## 🎯 How It Works

*   **Abstractive (BART):** The BART model is a transformer-based encoder-decoder (seq2seq) model. It is pre-trained on a large corpus of text and fine-tuned specifically for summarization. It reads the input text and generates a new sequence of words that represents a summary.
*   **Extractive (spaCy):** This method calculates the frequency of non-stopwords in the text, scores each sentence based on the sum of its word frequencies, and then selects the top N highest-scoring sentences to form the summary.

## 🔮 Future Improvements

*   Implement more robust web scraping using `newspaper3k`.
*   Add a "Text Similarity" score between original and summary.
*   Add support for multiple languages.
*   Deploy the app publicly on Streamlit Community Cloud.
