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

def build_recommender_arc_dataframe():
    arc_rows = []
    for book in books:
        scores = sentiment_arc(book)
        for i, score in enumerate(scores):
            arc_rows.append({'book': book, 'chunk': i + 1, 'sentiment': score})
    return pd.DataFrame(arc_rows)

# takes the arc dataframe as an argument instead of recomputing it, since recommender.py needs
# the per chunk detail for the sentiment arc chart and building it twice would mean scoring every
# sentence in all 18 books twice over
def build_recommender_dataframe(arc_df):
    base_rows = [analyze_book(book) for book in books]
    base_df = pd.DataFrame(base_rows)

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

# the longest possible distance between two books, every feature normalized to 0 to 1 and maximally
# different on all five at once, used to turn a raw distance into a percentage
MAX_DISTANCE = 5 ** 0.5

# converts a raw distance into a linguistic similarity percentage, a measure of how close the five
# writing style features land rather than how alike the books' content or themes are
def similarity_percent(distance):
    return round(max(0, 1 - distance / MAX_DISTANCE) * 100)

# finds a short passage whose local score on the matched feature sits closest to the book's own
# measured value for it, so the quote shown is typical of why the book scored the way it did
# sentiment volatility is the one exception, there the passage with the largest swing away from
# the book's average sentiment is what demonstrates the volatility, not a passage near the average
def representative_excerpt(book, feature, target, window=3):
    sents = list(gutenberg.sents(book))
    if len(sents) <= window:
        return ' '.join(' '.join(s) for s in sents)

    # samples a bounded number of windows spread across the book instead of checking every single one,
    # so a long book like the Bible or Moby Dick doesn't make this slow
    span = len(sents) - window
    num_samples = min(200, span)
    step = max(1, span // num_samples)

    best_start, best_score = 0, None
    for i in range(0, span, step):
        chunk_sents = sents[i:i + window]
        chunk_words = [w for s in chunk_sents for w in s]
        alpha_words = [w.lower() for w in chunk_words if w.isalpha()]
        if len(alpha_words) < 15:
            continue
        chunk_text = ' '.join(' '.join(s) for s in chunk_sents)

        if feature == 'Vocabulary Richness':
            local_value = len(set(alpha_words)) / len(alpha_words)
        elif feature == 'Sentence Length':
            local_value = len(chunk_words) / window
        elif feature == 'Reading Difficulty':
            local_value = textstat.flesch_kincaid_grade(chunk_text)
        else:
            local_value = sia.polarity_scores(chunk_text)['compound']

        # every feature but volatility wants the closest match to the book's own average, volatility
        # wants the opposite, the furthest swing away from it
        if feature == 'Sentiment Volatility':
            score = abs(local_value - target)
        else:
            score = -abs(local_value - target)

        if best_score is None or score > best_score:
            best_score, best_start = score, i

    return ' '.join(' '.join(s) for s in sents[best_start:best_start + window])