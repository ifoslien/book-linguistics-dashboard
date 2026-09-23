# dashboard.py is the start of the interactive dashboard
from dash import Dash, dcc, html
import plotly.express as px
from analysis import build_dataframe

df = build_dataframe()

fig = px.bar(
    df,
    x='book',
    y='type_token_ratio_sample',
    title='Vocabulary Richness by Book (10,000 word sample)',
    labels={'book': 'Book', 'type_token_ratio_sample': 'Type Token Ratio'},
    color_discrete_sequence=['#2a78d6']
)
fig.update_layout(template='plotly_white', showlegend=False)

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Linguistic Style Dashboard'),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run(debug=True)
