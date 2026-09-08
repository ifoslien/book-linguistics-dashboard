import nltk

nltk.download('gutenberg')
nltk.download('punkt_tab')

from nltk.corpus import gutenberg

books = ['austen-emma.txt', 'melville-moby_dick.txt', 'carroll-alice.txt']

for book in books:
    words = gutenberg.words(book)
    sents = gutenberg.sents(book)
    vocab = set(w.lower() for w in words if w.isalpha())

    print(book)
    print('word count', len(words))
    print('sentence count', len(sents))
    print('unique words', len(vocab))
    print()