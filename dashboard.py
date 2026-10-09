# dashboard.py is the interactive dashboard
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from analysis import build_dataframe, build_arc_dataframe

df = build_dataframe()
arc_df = build_arc_dataframe()

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

# pulls a few headline facts from the existing data for the summary row
num_books = len(df)
richest_vocab_book = df.loc[df['type_token_ratio_sample'].idxmax(), 'book']
easiest_read_book = df.loc[df['flesch_kincaid_grade'].idxmin(), 'book']
most_volatile_book = arc_df.groupby('book')['sentiment'].std().idxmax()

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

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Linguistic Style Dashboard'),
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

if __name__ == '__main__':
    app.run(debug=True)