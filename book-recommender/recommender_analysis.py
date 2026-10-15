# recommender_analysis.py is self-contained on purpose, it does not import from analysis.py
# so the book recommender can stay a separate project from the original dashboard
import nltk
import textstat
import pandas as pd
from nltk.corpus import gutenberg
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('gutenberg', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# the recommender searches the full nltk gutenberg corpus, 18 texts, the largest pool available locally
books = gutenberg.fileids()

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
        'type_token_ratio_sample': round(len(sample_vocab) / len(sample), 4),
        'avg_sentence_length': round(word_count / sentence_count, 2),
        'flesch_reading_ease': round(textstat.flesch_reading_ease(raw_text), 2),
        'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(raw_text), 2)
    }

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

def build_recommender_dataframe():
    base_rows = [analyze_book(book) for book in books]
    base_df = pd.DataFrame(base_rows)

    arc_rows = []
    for book in books:
        scores = sentiment_arc(book)
        for i, score in enumerate(scores):
            arc_rows.append({'book': book, 'chunk': i + 1, 'sentiment': score})
    arc_df = pd.DataFrame(arc_rows)

    sentiment_summary = arc_df.groupby('book')['sentiment'].agg(['mean', 'std']).reset_index()
    sentiment_summary.columns = ['book', 'avg_sentiment', 'sentiment_volatility']
    return base_df.merge(sentiment_summary, on='book')

# finds the n books whose style features are closest to the target book
# distance is euclidean over five features, each min-max normalized first so no single feature dominates
def find_similar_books(target_book, df, n=3):
    features = ['type_token_ratio_sample', 'avg_sentence_length', 'flesch_kincaid_grade', 'avg_sentiment', 'sentiment_volatility']

    normalized = df.copy()
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        span = max_val - min_val
        normalized[feature] = (df[feature] - min_val) / span if span else 0

    target_row = normalized.loc[normalized['book'] == target_book, features].iloc[0]
    normalized['distance'] = ((normalized[features] - target_row) ** 2).sum(axis=1) ** 0.5

    ranked = normalized[normalized['book'] != target_book].sort_values('distance').head(n)
    return df.merge(ranked[['book', 'distance']], on='book').sort_values('distance').reset_index(drop=True)
