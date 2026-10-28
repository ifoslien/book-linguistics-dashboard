# recommender.py is its own separate Dash app, kept apart from dashboard.py on purpose
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from recommender_analysis import build_recommender_dataframe, build_recommender_arc_dataframe, find_similar_books, similarity_percent, representative_excerpt, books

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

# maps each feature label to the dataframe column representative_excerpt should measure a passage against
# sentiment volatility has no column of its own to sample against, a passage's sentiment still gets compared
# to the book's average sentiment, just judged by how far it swings away from that average instead of how close
feature_to_column = {
    'Vocabulary Richness': 'type_token_ratio_sample',
    'Sentence Length': 'avg_sentence_length',
    'Reading Difficulty': 'flesch_kincaid_grade',
    'Avg Sentiment': 'avg_sentiment',
    'Sentiment Volatility': 'avg_sentiment'
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
    html.P('This tool searches the full NLTK Gutenberg corpus, 18 books across 12 authors, and surfaces the three books whose writing style sits closest to whichever book you pick. Each book is scored on five features: vocabulary richness, how varied its word choice is, average sentence length, how long its sentences run, reading difficulty, how hard the text is to read, average sentiment, how positive or negative its tone is overall, and sentiment volatility, how much that tone shifts through the story. Every feature is min-max normalized across the 18 books, and similarity is the Euclidean distance between those normalized scores, turned into a linguistic similarity percentage so no single feature dominates the match just for having a bigger scale. That percentage reflects how close the five writing style features land, not how alike the books are in content or theme, so two very different stories can still score a high linguistic similarity.')
], className='book-page')

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Book Recommender'),
    about_panel,
    html.Div([
        html.H2('Pick a Book'),
        dcc.Dropdown(
            id='target-book-dropdown',
            # shows the real title and author instead of the bare fileid, sorted alphabetically by that same label
            options=[{'label': book_label(book), 'value': book} for book in sorted(books, key=lambda b: book_label(b))],
            value='austen-emma.txt'
        )
    ], className='book-picker'),
    html.Div(id='matches-panel', className='book-page')
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

    # each match gets its own heading and a short explanation instead of one shared paragraph, so every
    # book gets its own small showcase rather than being folded into a single block of text
    chosen_title = book_info[selected_book][0]
    chosen_row = df.loc[df['book'] == selected_book].iloc[0]
    explained_features = set()
    # caches the chosen book's excerpt by feature, so if two matches share the same closest feature the
    # chosen book's passage for that feature only gets computed once instead of twice
    chosen_excerpts = {}
    match_entries = []
    for row in matches.itertuples():
        feature = closest_feature(row.book)
        column = feature_to_column[feature]
        pct = similarity_percent(row.distance)

        if feature in explained_features:
            reason = feature
        else:
            reason = f'{feature}, {feature_definitions[feature]}'
            explained_features.add(feature)
        sentence = f"{pct}% linguistic similarity to '{chosen_title}', closest in {reason}."

        if feature not in chosen_excerpts:
            chosen_excerpts[feature] = representative_excerpt(selected_book, feature, chosen_row[column])

        match_row = df.loc[df['book'] == row.book].iloc[0]
        match_excerpt = representative_excerpt(row.book, feature, match_row[column])

        match_entries.append({
            'heading': book_label(row.book),
            'sentence': sentence,
            'chosen_title': chosen_title,
            'chosen_excerpt': chosen_excerpts[feature],
            'match_title': book_info[row.book][0],
            'match_excerpt': match_excerpt
        })

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
        title=f'Style Comparison, {selected_book} vs Closest Matches',
        # only the headline title goes serif, the axis and legend labels stay in the default sans since
        # serif gets harder to read at the smaller sizes those use
        title_font_family='Lora, serif',
        title_font_size=18
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
    arc_fig.update_layout(template='plotly_white', title_font_family='Lora, serif', title_font_size=18)

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
        title_font_family='Lora, serif',
        title_font_size=18,
        height=650
    )
    # the five subplot titles act like mini headlines too, so they get the serif treatment along with the main title
    raw_fig.update_annotations(font=dict(family='Lora, serif', size=13))

    return [
        html.H2(f'Closest Matches to {selected_book}'),
        html.Div([
            html.Div([
                html.H3(entry['heading']),
                html.P(entry['sentence']),
                html.Div([
                    html.P(f"“{entry['chosen_excerpt']}”", className='excerpt'),
                    html.P(f"— {entry['chosen_title']}", className='excerpt-attribution')
                ], className='excerpt-group'),
                html.Div([
                    html.P(f"“{entry['match_excerpt']}”", className='excerpt'),
                    html.P(f"— {entry['match_title']}", className='excerpt-attribution')
                ], className='excerpt-group')
            ], className='match-entry')
            for entry in match_entries
        ]),
        dcc.Tabs(value='radar', className='classic-tabs', children=[
            dcc.Tab(label='Style Comparison', value='radar', children=[dcc.Graph(figure=radar_fig)]),
            dcc.Tab(label='Sentiment Arc', value='arc', children=[dcc.Graph(figure=arc_fig)]),
            dcc.Tab(label='Raw Values', value='raw', children=[dcc.Graph(figure=raw_fig)])
        ])
    ]

if __name__ == '__main__':
    app.run(debug=True, port=8051)