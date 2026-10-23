# recommender.py is its own separate Dash app, kept apart from dashboard.py on purpose
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from recommender_analysis import build_recommender_dataframe, build_recommender_arc_dataframe, find_similar_books, books

arc_df = build_recommender_arc_dataframe()
df = build_recommender_dataframe(arc_df)

# same five features find_similar_books uses to compute distance, shown here so the match is visible, not just claimed
radar_features = ['type_token_ratio_sample', 'avg_sentence_length', 'flesch_kincaid_grade', 'avg_sentiment', 'sentiment_volatility']
radar_labels = ['Vocabulary Richness', 'Sentence Length', 'Reading Difficulty', 'Avg Sentiment', 'Sentiment Volatility']
feature_label_map = dict(zip(radar_features, radar_labels))

# plain language gloss for each feature, shown the first time that feature is named in a given result
feature_definitions = {
    'Vocabulary Richness': 'how varied its word choice is',
    'Sentence Length': 'how long its sentences run on average',
    'Reading Difficulty': 'how difficult the text is to read',
    'Avg Sentiment': 'how positive or negative its tone is overall',
    'Sentiment Volatility': 'how much the tone shifts throughout the book'
}

# title and author for each book in the corpus, pulled from each text's own gutenberg header
# the bible has no single author, so it maps to None and gets left out of the sentence
book_info = {
    'austen-emma.txt': ('Emma', 'Jane Austen'),
    'austen-persuasion.txt': ('Persuasion', 'Jane Austen'),
    'austen-sense.txt': ('Sense and Sensibility', 'Jane Austen'),
    'bible-kjv.txt': ('The King James Bible', None),
    'blake-poems.txt': ('Poems', 'William Blake'),
    'bryant-stories.txt': ('Stories to Tell to Children', 'Sara Cone Bryant'),
    'burgess-busterbrown.txt': ('The Adventures of Buster Bear', 'Thornton W. Burgess'),
    'carroll-alice.txt': ("Alice's Adventures in Wonderland", 'Lewis Carroll'),
    'chesterton-ball.txt': ('The Ball and The Cross', 'G.K. Chesterton'),
    'chesterton-brown.txt': ('The Wisdom of Father Brown', 'G.K. Chesterton'),
    'chesterton-thursday.txt': ('The Man Who Was Thursday', 'G.K. Chesterton'),
    'edgeworth-parents.txt': ("The Parent's Assistant", 'Maria Edgeworth'),
    'melville-moby_dick.txt': ('Moby Dick', 'Herman Melville'),
    'milton-paradise.txt': ('Paradise Lost', 'John Milton'),
    'shakespeare-caesar.txt': ('The Tragedie of Julius Caesar', 'William Shakespeare'),
    'shakespeare-hamlet.txt': ('The Tragedie of Hamlet', 'William Shakespeare'),
    'shakespeare-macbeth.txt': ('The Tragedie of Macbeth', 'William Shakespeare'),
    'whitman-leaves.txt': ('Leaves of Grass', 'Walt Whitman')
}

# builds the "'Title' by Author (fileid.txt)" phrase used in the match sentences, and leaves out the author when there isn't one
def book_label(book):
    title, author = book_info[book]
    if author:
        return f"'{title}' by {author} ({book})"
    return f"'{title}' ({book})"

# the chosen book gets a dark line to stand out, the three matches get the lighter bookish palette
chosen_color = '#2b241c'
match_palette = ['#A9C6E0', '#D9A182', '#8FBFAA']

# grid positions for the five raw value subplots, in a 2 row by 3 col layout with one cell left empty
raw_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

# short writeup explaining what this tool does, kept to one paragraph
about_panel = html.Div([
    html.H2('About This Recommender'),
    html.P('This tool searches the full NLTK Gutenberg corpus, 18 books across 12 authors, and surfaces the three books whose writing style sits closest to whichever book you pick. Each book is scored on five features: vocabulary richness, average sentence length, reading difficulty, and the average and volatility of its sentiment arc, scored the same way as the original dashboard. Every feature is min-max normalized across the 18 books, and similarity is the Euclidean distance between those normalized scores, so no single feature dominates the match just for having a bigger scale.')
], className='card')

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Book Recommender'),
    about_panel,
    html.Div([
        html.H2('Pick a Book'),
        dcc.Dropdown(
            id='target-book-dropdown',
            options=[{'label': book, 'value': book} for book in sorted(books)],
            value='austen-emma.txt'
        )
    ], className='card'),
    html.Div(id='matches-panel', className='card')
], className='page')

# reruns whenever a different book is picked, shows the three closest matches by style distance
@app.callback(
    Output('matches-panel', 'children'),
    Input('target-book-dropdown', 'value')
)
def update_matches(selected_book):
    matches = find_similar_books(selected_book, df, n=3)
    compare_books = [selected_book] + matches['book'].tolist()
    compare_colors = [chosen_color] + match_palette

    # normalized against the full 18-book pool, same as find_similar_books, so the radar reflects real position in the corpus
    normalized = df.copy()
    for feature in radar_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        span = max_val - min_val
        normalized[feature] = (df[feature] - min_val) / span if span else 0

    # a distance number says how close a match is but not why, this names the single feature each match is nearest on
    target_row_norm = normalized.loc[normalized['book'] == selected_book, radar_features].iloc[0]

    def closest_feature(match_book):
        match_row_norm = normalized.loc[normalized['book'] == match_book, radar_features].iloc[0]
        diffs = (target_row_norm - match_row_norm).abs()
        return feature_label_map[diffs.idxmin()]

    # one sentence per match instead of one sentence covering all three, since the three matches can split
    # across one, two, or three different closest features depending on the book, and a single sentence
    # chaining a variable number of reasons together gets hard to follow
    chosen_label = book_label(selected_book)
    explained_features = set()
    match_sentences = []
    for i, row in enumerate(matches.itertuples()):
        feature = closest_feature(row.book)
        match_label = book_label(row.book)
        if feature in explained_features:
            reason = feature
        else:
            reason = f'{feature}, {feature_definitions[feature]}'
            explained_features.add(feature)
        if i == 0:
            match_sentences.append(f'{chosen_label} most closely matches {match_label} in {reason}.')
        else:
            match_sentences.append(f"It's also closest to {match_label} in {reason}.")

    radar_fig = go.Figure()
    for book, color in zip(compare_books, compare_colors):
        row = normalized.loc[normalized['book'] == book, radar_features].iloc[0]
        values = row.tolist()
        values.append(values[0])
        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_labels + [radar_labels[0]],
            name=book,
            line=dict(color=color, width=3 if book == selected_book else 2),
            mode='lines'
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template='plotly_white',
        title=f'Style Comparison, {selected_book} vs Closest Matches'
    )

    # neither the radar nor the raw values grid show how tone moves across the story, just where it lands on average
    # this traces the same sentiment arc the original dashboard uses, filtered down to just the books being compared
    compare_arc_df = arc_df[arc_df['book'].isin(compare_books)]
    arc_fig = px.line(
        compare_arc_df,
        x='chunk',
        y='sentiment',
        color='book',
        category_orders={'book': compare_books},
        color_discrete_map=dict(zip(compare_books, compare_colors)),
        title=f'Sentiment Across the Story, {selected_book} vs Closest Matches',
        labels={'chunk': 'Story Progress (chunk)', 'sentiment': 'Sentiment Score'}
    )
    # the chosen book gets a thicker line so it still stands out here, same idea as the radar
    for trace in arc_fig.data:
        trace.line.width = 4 if trace.name == selected_book else 2
    arc_fig.update_layout(template='plotly_white')

    # the radar shows shape but hides the real numbers, since every feature gets squeezed onto a 0 to 1 scale
    # this grid shows the actual value for each feature instead, one subplot per feature since they don't share a scale
    raw_fig = make_subplots(rows=2, cols=3, subplot_titles=radar_labels)
    for i, (feature, (r, c)) in enumerate(zip(radar_features, raw_positions)):
        for book, color in zip(compare_books, compare_colors):
            value = df.loc[df['book'] == book, feature].iloc[0]
            raw_fig.add_trace(
                go.Bar(
                    x=[book],
                    y=[value],
                    marker=dict(color=color),
                    name=book,
                    legendgroup=book,
                    # only shows each book once in the legend, on the first subplot, instead of five times over
                    showlegend=(i == 0),
                    text=[f'{value:.2f}'],
                    textposition='outside'
                ),
                row=r, col=c
            )
    raw_fig.update_xaxes(showticklabels=False)
    raw_fig.update_layout(
        template='plotly_white',
        title=f'Raw Values, {selected_book} vs Closest Matches',
        height=650
    )

    return [
        html.H2(f'Closest Matches to {selected_book}'),
        html.Div(' '.join(match_sentences), className='finding-row'),
        dcc.Graph(figure=radar_fig),
        dcc.Graph(figure=arc_fig),
        dcc.Graph(figure=raw_fig)
    ]

if __name__ == '__main__':
    app.run(debug=True, port=8051)