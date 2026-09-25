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

sentiment_fig = px.line(
    arc_df,
    x='chunk',
    y='sentiment',
    color='book',
    title='Sentiment Across the Story by Book',
    labels={'chunk': 'Story Progress (chunk)', 'sentiment': 'Sentiment Score'},
    color_discrete_sequence=['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
)
sentiment_fig.update_layout(template='plotly_white')

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Linguistic Style Dashboard'),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[{'label': label, 'value': metric} for metric, label in metric_options.items()],
        value='type_token_ratio_sample'
    ),
    dcc.Graph(id='metric-chart'),
    dcc.Graph(figure=sentiment_fig)
])

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
        color_discrete_sequence=['#2a78d6']
    )
    fig.update_layout(template='plotly_white', showlegend=False)
    return fig

if __name__ == '__main__':
    app.run(debug=True)