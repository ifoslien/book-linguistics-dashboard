import nltk

nltk.download('gutenberg')
nltk.download('punkt_tab')

from nltk.corpus import gutenberg

books = ['austen-emma.txt', 'melville-moby_dick.txt', 'carroll-alice.txt']

def analyze_book(book):
    words = gutenberg.words(book)
    sents = gutenberg.sents(book)
    vocab = set(w.lower() for w in words if w.isalpha())

    word_count = len(words)
    sentence_count = len(sents)
    vocab_size = len(vocab)

    return {
        'book': book,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'vocab_size': vocab_size,
        'type_token_ratio': round(vocab_size / word_count, 4),
        'avg_sentence_length': round(word_count / sentence_count, 2)
    }

results = [analyze_book(book) for book in books]

for r in results:
    print(r['book'])
    print('word count', r['word_count'])
    print('sentence count', r['sentence_count'])
    print('unique words', r['vocab_size'])
    print('type token ratio', r['type_token_ratio'])
    print('avg sentence length', r['avg_sentence_length'])
    print()