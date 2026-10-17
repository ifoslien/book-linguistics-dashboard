# recommender.py is its own separate Dash app, kept apart from dashboard.py on purpose
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from recommender_analysis import build_recommender_dataframe, find_similar_books, books

df = build_recommender_dataframe()

# same five features find_similar_books uses to compute distance, shown here so the match is visible, not just claimed
radar_features = ['type_token_ratio_sample', 'avg_sentence_length', 'flesch_kincaid_grade', 'avg_sentiment', 'sentiment_volatility']
radar_labels = ['Vocabulary Richness', 'Sentence Length', 'Reading Difficulty', 'Avg Sentiment', 'Sentiment Volatility']

# the chosen book gets a dark line to stand out, the three matches get the lighter bookish palette
chosen_color = '#2b241c'
match_palette = ['#A9C6E0', '#D9A182', '#8FBFAA']

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Book Recommender'),
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

    return [
        html.H2(f'Closest Matches to {selected_book}'),
        html.Div([
            html.Div(f'{row.book} (distance {row.distance:.3f})', className='finding-row')
            for row in matches.itertuples()
        ]),
        dcc.Graph(figure=radar_fig)
    ]

if __name__ == '__main__':
    app.run(debug=True, port=8051)