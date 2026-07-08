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
            secondsVisible: false,
            minBarSpacing: 0.01,
            rightOffset: 3
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

    const rangeConfig = {
        "1M": {
            interval: "1d",
            days: 30
        },
        "6M": {
            interval: "1d",
            days: 180
        },
        "1Y": {
            interval: "1d",
            days: 365
        },
        "5Y": {
            interval: "1w",
            days: 365 * 5
        },
        "ALL": {
            interval: "1mo",
            days: null
        }};

    function setVisibleDays(data, days) {
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

    function loadCandles(range) {
        const config = rangeConfig[range];

        if (!config) {
            return;
        }

        fetch(`/api/stock/${ticker}/candles?interval=${config.interval}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error("Failed to load chart data");
                }

                return response.json();
            })
            .then(data => {
                console.log("Range:", range);
                console.log("Interval:", config.interval);
                console.log("Candles count:", data.length);
                console.log("First candle:", data[0]);
                console.log("Last candle:", data[data.length - 1]);

                candleSeries.setData(data);

                if (!data || data.length === 0) {
                    return;
                }

                if (range === "ALL") {
                    chart.timeScale().fitContent();
                } else {
                    setVisibleDays(data, config.days);
                }
            })
            .catch(error => {
                console.error("Chart data loading error:", error);
            });
    }

    const buttons = document.querySelectorAll(".range-buttons button");

    buttons.forEach(button => {
        button.addEventListener("click", () => {
            const range = button.dataset.range;

            buttons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");

            loadCandles(range);
        });
    });

    const defaultRange = "1Y";
    const defaultButton = document.querySelector(`.range-buttons button[data-range="${defaultRange}"]`);

    if (defaultButton) {
        defaultButton.classList.add("active");
    }

    loadCandles(defaultRange);

    window.addEventListener("resize", () => {
        chart.applyOptions({
            width: chartElement.clientWidth
        });
    });
}