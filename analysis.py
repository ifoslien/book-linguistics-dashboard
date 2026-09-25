# analysis.py holds the shared logic so the notebook and dashboard stay in sync
import nltk
import textstat
import pandas as pd
from nltk.corpus import gutenberg

nltk.download('gutenberg', quiet=True)
nltk.download('punkt_tab', quiet=True)

books = [
    'austen-emma.txt',
    'austen-persuasion.txt',
    'austen-sense.txt',
    'melville-moby_dick.txt',
    'carroll-alice.txt',
    'chesterton-ball.txt',
    'chesterton-thursday.txt'
]

def analyze_book(book, ttr_sample_size=10000):
    words = gutenberg.words(book)
    sents = gutenberg.sents(book)
    vocab = set(w.lower() for w in words if w.isalpha())
    raw_text = gutenberg.raw(book)

    word_count = len(words)
    sentence_count = len(sents)
    vocab_size = len(vocab)

    alpha_words = [w.lower() for w in words if w.isalpha()]
    sample = alpha_words[:ttr_sample_size]
    sample_vocab = set(sample)

    return {
        'book': book,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'vocab_size': vocab_size,
        'type_token_ratio_raw': round(vocab_size / word_count, 4),
        'type_token_ratio_sample': round(len(sample_vocab) / len(sample), 4),
        'avg_sentence_length': round(word_count / sentence_count, 2),
        # higher score means easier to read
        'flesch_reading_ease': round(textstat.flesch_reading_ease(raw_text), 2),
        # roughly the US school grade level needed to understand the text
        'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(raw_text), 2)
    }

def build_dataframe():
    results = [analyze_book(book) for book in books]
    return pd.DataFrame(results)




# adds sentiment scoring, ported from the notebook's sentiment arc section
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

def sentiment_arc(book, num_bins=20):
    sents = gutenberg.sents(book)
    sentence_scores = []
    for sent in sents:
        text = ' '.join(sent)
        score = sia.polarity_scores(text)['compound']
        sentence_scores.append(score)

    bin_size = len(sentence_scores) // num_bins
    binned_scores = []
    for i in range(num_bins):
        start = i * bin_size
        end = start + bin_size
        bin_scores = sentence_scores[start:end]
        binned_scores.append(sum(bin_scores) / len(bin_scores))
    return binned_scores

def build_arc_dataframe():
    arc_rows = []
    for book in books:
        scores = sentiment_arc(book)
        for i, score in enumerate(scores):
            arc_rows.append({'book': book, 'chunk': i + 1, 'sentiment': score})
    return pd.DataFrame(arc_rows)