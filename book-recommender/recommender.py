# recommender.py is its own separate Dash app, kept apart from dashboard.py on purpose
from dash import Dash, dcc, html, Input, Output
from recommender_analysis import build_recommender_dataframe, find_similar_books, books

df = build_recommender_dataframe()

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
    return [
        html.H2(f'Closest Matches to {selected_book}'),
        html.Div([
            html.Div(f'{row.book} (distance {row.distance:.3f})', className='finding-row')
            for row in matches.itertuples()
        ])
    ]

if __name__ == '__main__':
    app.run(debug=True, port=8051)
