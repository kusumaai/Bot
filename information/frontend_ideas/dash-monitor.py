from dash import Dash, html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H1('KillaBot Monitor'),
    
    # Portfolio Value Chart
    dcc.Graph(id='equity-chart'),
    
    # Active Positions Table
    html.Div(id='positions-table'),
    
    # Performance Metrics
    html.Div([
        html.H3('Performance'),
        html.Div(id='metrics')
    ]),
    
    dcc.Interval(id='interval', interval=5000)
])

@app.callback(
    [Output('equity-chart', 'figure'),
     Output('positions-table', 'children'),
     Output('metrics', 'children')],
    Input('interval', 'n_intervals')
)
def update_metrics(n):
    # Reuse your existing DB queries from portfolio.py
    with DBConnection(ctx.db_pool) as conn:
        trades_df = pd.read_sql("SELECT * FROM trades ORDER BY entry_time", conn)
        positions_df = pd.read_sql("SELECT * FROM positions", conn)
    
    # Create equity chart
    fig = go.Figure(data=[
        go.Scatter(x=trades_df['entry_time'], 
                  y=trades_df['cumulative_pnl'],
                  fill='tozeroy')
    ])
    
    # Format positions table
    positions_table = generate_dash_table(positions_df)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(trades_df)
    
    return fig, positions_table, metrics

if __name__ == '__main__':
    app.run_server(debug=True)