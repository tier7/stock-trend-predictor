import plotly.graph_objects as go

def create_candlestick_chart(data, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=data["date"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"]
    )
    ])
    chart_html = fig.to_html(full_html=False)
    return chart_html
