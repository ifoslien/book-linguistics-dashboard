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

# the chosen book gets a dark line to stand out, the three matches get the lighter bookish palette
chosen_color = '#2b241c'
match_palette = ['#A9C6E0', '#D9A182', '#8FBFAA']

# grid positions for the five raw value subplots, in a 2 row by 3 col layout with one cell left empty
raw_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

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
        html.Div([
            html.Div(f'{row.book} (distance {row.distance:.3f})', className='finding-row')
            for row in matches.itertuples()
        ]),
        dcc.Graph(figure=radar_fig),
        dcc.Graph(figure=arc_fig),
        dcc.Graph(figure=raw_fig)
    ]

if __name__ == '__main__':
    app.run(debug=True, port=8051)