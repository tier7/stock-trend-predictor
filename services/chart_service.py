import pandas as pd
import plotly.graph_objects as go


def create_candlestick_chart(data, ticker):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])

    fig = go.Figure(data=[
        go.Candlestick(
            x=data["date"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            increasing_line_color="#0ecb81",
            decreasing_line_color="#f6465d",
            name=ticker
        )
    ])

    fig.update_layout(
        title=f"{ticker} — wykres świecowy",
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        dragmode="pan",
        showlegend=False,
        margin=dict(l=20, r=70, t=60, b=40),

        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ]
            ),
            rangebreaks=[
                dict(bounds=["sat", "mon"])
            ]
        ),

        yaxis=dict(
            title="Cena",
            side="right",
            fixedrange=False
        )
    )

    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True
    )

    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True
    )

    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "responsive": True
    }
    chart_id = f"{ticker.lower()}-chart"

    post_script = """
    var chart = document.getElementById('{plot_id}');
    var updatingY = false;

    function autoscaleY() {
        if (!chart || !chart.layout || !chart.layout.xaxis) {
            return;
        }

        var xRange = chart.layout.xaxis.range;

        if (!xRange || xRange.length < 2) {
            return;
        }

        var trace = chart.data[0];

        if (!trace || !trace.x || !trace.low || !trace.high) {
            return;
        }

        var xData = trace.x;
        var lowData = trace.low;
        var highData = trace.high;

        var start = new Date(xRange[0]).getTime();
        var end = new Date(xRange[1]).getTime();

        var minY = Infinity;
        var maxY = -Infinity;

        for (var i = 0; i < xData.length; i++) {
            var currentDate = new Date(xData[i]).getTime();

            if (currentDate >= start && currentDate <= end) {
                var low = Number(lowData[i]);
                var high = Number(highData[i]);

                if (!isNaN(low) && low < minY) {
                    minY = low;
                }

                if (!isNaN(high) && high > maxY) {
                    maxY = high;
                }
            }
        }

        if (!isFinite(minY) || !isFinite(maxY)) {
            return;
        }

        var padding = (maxY - minY) * 0.08;

        if (padding === 0) {
            padding = maxY * 0.01 || 1;
        }

        updatingY = true;

        Plotly.relayout(chart, {
            "yaxis.range": [minY - padding, maxY + padding]
        }).then(function() {
            updatingY = false;
        });
    }

    chart.on("plotly_relayout", function(eventData) {
        if (updatingY) {
            return;
        }

        var changedX =
            eventData["xaxis.range"] !== undefined ||
            eventData["xaxis.range[0]"] !== undefined ||
            eventData["xaxis.range[1]"] !== undefined ||
            eventData["xaxis.autorange"] !== undefined;

        if (changedX) {
            autoscaleY();
        }
    });
    """

    return fig.to_html(full_html=False, config=config, div_id=chart_id, post_script=post_script)