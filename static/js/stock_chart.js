const chartElement = document.getElementById("stock-chart");

if (chartElement) {
    const ticker = chartElement.dataset.ticker;
    const chart = LightweightCharts.createChart(chartElement, {
        width: chartElement.clientWidth,
        height: 650,

        layout: {
            background: {
                type: "solid",
                color: "#0b0e11"
            },
            textColor: "#d1d4dc"
        },

        grid: {
            vertLines: {
                color: "#1e2329"
            },
            horzLines: {
                color: "#1e2329"
            }
        },

        rightPriceScale: {
            borderColor: "#2b3139",
            scaleMargins: {
                top: 0.08,
                bottom: 0.08
            }
        },

        timeScale: {
            borderColor: "#2b3139",
            timeVisible: true,
            secondsVisible: false
        },

        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal
        },

        handleScroll: {
            mouseWheel: true,
            pressedMouseMove: true,
            horzTouchDrag: true,
            vertTouchDrag: false
        },

        handleScale: {
            axisPressedMouseMove: true,
            mouseWheel: true,
            pinch: true
        }
    });

    const candleOptions = {
        upColor: "#0ecb81",
        downColor: "#f6465d",
        borderUpColor: "#0ecb81",
        borderDownColor: "#f6465d",
        wickUpColor: "#0ecb81",
        wickDownColor: "#f6465d"
    };

    const candleSeries = chart.addSeries
        ? chart.addSeries(LightweightCharts.CandlestickSeries, candleOptions)
        : chart.addCandlestickSeries(candleOptions);

    fetch(`/api/stock/${ticker}/candles`)
        .then(response => response.json())
        .then(data => {
            candleSeries.setData(data);
            chart.timeScale().fitContent();
            setupRangeButtons(chart, data);
        })
        .catch(error => {
            console.error("Chart data loading error:", error);
        });

    window.addEventListener("resize", () => {
        chart.applyOptions({
            width: chartElement.clientWidth
        });
    });
}

function showAllData(chart, data) {
    if (!data || data.length === 0) {
        return;
    }

    chart.timeScale().setVisibleLogicalRange({
        from: 0,
        to: data.length - 1
    });
}

function setVisibleDays(chart, data, days) {
    if (!data || data.length === 0) {
        return;
    }

    const lastIndex = data.length - 1;
    const lastDate = new Date(data[lastIndex].time);

    const startDate = new Date(lastDate);
    startDate.setDate(startDate.getDate() - days);

    let fromIndex = data.findIndex(candle => {
        return new Date(candle.time) >= startDate;
    });

    if (fromIndex === -1) {
        fromIndex = 0;
    }

    chart.timeScale().setVisibleLogicalRange({
        from: fromIndex,
        to: lastIndex
    });
}

function setupRangeButtons(chart, data) {
    const buttons = document.querySelectorAll(".range-buttons button");

    buttons.forEach(button => {
        button.addEventListener("click", () => {
            const range = button.dataset.range;

            buttons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");

            if (range === "ALL") {
            showAllData(chart, data);
            return;
            }

            const daysByRange = {
                "1M": 30,
                "6M": 180,
                "1Y": 365,
                "5Y": 365 * 5
            };

            setVisibleDays(chart, data, daysByRange[range]);

            const days = daysByRange[range];

            if (!days || data.length === 0) {
                return;
            }

            const lastCandle = data[data.length - 1];
            const lastDate = new Date(lastCandle.time);

            const startDate = new Date(lastDate);
            startDate.setDate(startDate.getDate() - days);

            const from = startDate.toISOString().split("T")[0];
            const to = lastDate.toISOString().split("T")[0];

            chart.timeScale().setVisibleRange({
                from: from,
                to: to
            });
        });
    });
}