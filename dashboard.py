# dashboard.py is the interactive dashboard
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from analysis import build_arc_dataframe, build_full_dataframe, build_top_words_dataframe

df = build_full_dataframe()
arc_df = build_arc_dataframe()
top_words_df = build_top_words_dataframe()

# maps each column name to a readable label for the dropdown and chart title
metric_options = {
    'type_token_ratio_sample': 'Vocabulary Richness (Type Token Ratio)',
    'avg_sentence_length': 'Average Sentence Length',
    'flesch_kincaid_grade': 'Reading Difficulty (Flesch-Kincaid Grade)'
}

# muted palette matching the bookish theme, one color per book
muted_palette = ['#A9C6E0', '#D9A182', '#8FBFAA', '#D8B979', '#D9A6BC', '#7A9B6E', '#A79BD1']

sentiment_fig = px.line(
    arc_df,
    x='chunk',
    y='sentiment',
    color='book',
    title='Sentiment Across the Story by Book',
    labels={'chunk': 'Story Progress (chunk)', 'sentiment': 'Sentiment Score'},
    color_discrete_sequence=muted_palette
)
sentiment_fig.update_layout(template='plotly_white')

# normalizes each style feature to a 0 to 1 scale so they can share one radar axis
radar_features = ['type_token_ratio_sample', 'avg_sentence_length', 'flesch_kincaid_grade', 'avg_sentiment', 'sentiment_volatility']
radar_labels = ['Vocabulary Richness', 'Sentence Length', 'Reading Difficulty', 'Avg Sentiment', 'Sentiment Volatility']

normalized = df.copy()
for feature in radar_features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    normalized[feature] = (df[feature] - min_val) / (max_val - min_val)

radar_fig = go.Figure()
for i, row in normalized.iterrows():
    values = row[radar_features].tolist()
    values.append(values[0])
    radar_fig.add_trace(go.Scatterpolar(
        r=values,
        theta=radar_labels + [radar_labels[0]],
        name=row['book'],
        line=dict(color=muted_palette[i % len(muted_palette)]),
        mode='lines'
    ))
radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    template='plotly_white',
    title='Style Profile by Book'
)

# pulls a few headline facts from the existing data for the summary row
num_books = len(df)
richest_vocab_book = df.loc[df['type_token_ratio_sample'].idxmax(), 'book']
easiest_read_book = df.loc[df['flesch_kincaid_grade'].idxmin(), 'book']
most_volatile_book = df.loc[df['sentiment_volatility'].idxmax(), 'book']

stat_tiles = html.Div([
    html.Div([
        html.Div('Books Analyzed', className='stat-label'),
        html.Div(str(num_books), className='stat-value')
    ], className='stat-tile'),
    html.Div([
        html.Div('Richest Vocabulary', className='stat-label'),
        html.Div(richest_vocab_book, className='stat-value stat-value-text')
    ], className='stat-tile'),
    html.Div([
        html.Div('Easiest to Read', className='stat-label'),
        html.Div(easiest_read_book, className='stat-value stat-value-text')
    ], className='stat-tile'),
    html.Div([
        html.Div('Most Volatile Sentiment', className='stat-label'),
        html.Div(most_volatile_book, className='stat-value stat-value-text')
    ], className='stat-tile')
], className='stat-row')

# curated findings written during the earlier analysis, not recomputed here
findings = [
    'Sense and Sensibility has the highest sentiment volatility in the set, despite sharing an author with the steadier Emma and Persuasion.',
    'Moby Dick sits alone on the style map, distant from every other book on nearly every metric.',
    'The two Chesterton novels form the tightest style cluster of any pair in the set.',
    'Austen\'s three novels all surface formal social titles like mr and mrs among their top words, while neither Chesterton novel does.'
]

findings_panel = html.Div([
    html.H2('Notable Findings'),
    html.Div([
        html.Div(finding, className='finding-row')
        for finding in findings
    ])
], className='card')

# short writeup explaining the data source and methods for anyone landing on the dashboard cold
about_panel = html.Div([
    html.H2('About This Project'),
    html.P('This dashboard compares the linguistic style of seven classic novels pulled from the NLTK Gutenberg corpus, spanning Jane Austen, Herman Melville, Lewis Carroll, and G.K. Chesterton. The set was chosen for contrast since it pairs three novels from one author against standalone outliers, which makes it possible to see whether a book\'s style owes more to its author or to the book itself.'),
    html.P('Vocabulary richness comes from the type token ratio on a 10,000 word sample per book, sentence length is the average words per sentence, and reading difficulty is the Flesch-Kincaid grade level. Sentiment is scored sentence by sentence with VADER, then binned into 20 chunks per book to trace how the tone shifts across the story and how much it varies.'),
    html.P('Built as part of a 100 days of code challenge.')
], className='card')

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Linguistic Style Dashboard'),
    about_panel,
    stat_tiles,
    html.Div([
        dcc.Dropdown(
            id='metric-dropdown',
            options=[{'label': label, 'value': metric} for metric, label in metric_options.items()],
            value='type_token_ratio_sample'
        ),
        dcc.Graph(id='metric-chart')
    ], className='card'),
    html.Div([
        dcc.Graph(figure=sentiment_fig)
    ], className='card'),
    html.Div([
        dcc.Graph(figure=radar_fig)
    ], className='card'),
    html.Div([
        html.H2('Top Words by Book'),
        dcc.Dropdown(
            id='book-dropdown',
            options=[{'label': book, 'value': book} for book in df['book']],
            value='austen-emma.txt'
        ),
        dcc.Graph(id='top-words-chart')
    ], className='card'),
    findings_panel
], className='page')

# this function reruns automatically whenever the dropdown value changes
@app.callback(
    Output('metric-chart', 'figure'),
    Input('metric-dropdown', 'value')
)
def update_chart(selected_metric):
    fig = px.bar(
        df,
        x='book',
        y=selected_metric,
        title=metric_options[selected_metric],
        labels={'book': 'Book', selected_metric: metric_options[selected_metric]},
        color_discrete_sequence=['#A9C6E0']
    )
    fig.update_layout(template='plotly_white', showlegend=False)
    return fig

# reruns whenever a different book is picked, sorted so the most frequent word sits on top
@app.callback(
    Output('top-words-chart', 'figure'),
    Input('book-dropdown', 'value')
)
def update_top_words_chart(selected_book):
    book_words = top_words_df[top_words_df['book'] == selected_book].sort_values('count')
    fig = px.bar(
        book_words,
        x='count',
        y='word',
        orientation='h',
        title=f'Most Frequent Words in {selected_book}',
        labels={'count': 'Occurrences', 'word': 'Word'},
        color_discrete_sequence=['#D9A182']
    )
    # tall enough and forced to one tick per bar, otherwise plotly skips every other word label
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        height=max(400, len(book_words) * 28)
    )
    fig.update_yaxes(tickmode='linear')
    return fig

if __name__ == '__main__':
    app.run(debug=True)