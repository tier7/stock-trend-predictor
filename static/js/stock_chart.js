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

    const tooltip = document.createElement("div");
    tooltip.className = "chart-tooltip";
    chartElement.appendChild(tooltip);
    const chartLegend = document.getElementById("chart-legend");

    let currentInterval = "1d";
    const indicatorsCache = {};
    const indicatorSeries = {};
    const indicatorDefinitions = {
        sma_12: ["sma_12"],
        sma_26: ["sma_26"],
        ema_12: ["ema_12"],
        ema_26: ["ema_26"],
        bollinger_bands: ["bollinger_upper", "bollinger_mid", "bollinger_lower"]
    };
    const indicatorStyles = {
        sma_12: {
            color: "#f0b90b",
            lineWidth: 2
        },
        sma_26: {
            color: "#f6465d",
            lineWidth: 2
        },
        ema_12: {
            color: "#0ecb81",
            lineWidth: 2
        },
        ema_26: {
            color: "#3f8cff",
            lineWidth: 2
        },
        bollinger_upper: {
            color: "#b7c0cc",
            lineWidth: 1
        },
        bollinger_mid: {
            color: "#f0b90b",
            lineWidth: 1
        },
        bollinger_lower: {
            color: "#b7c0cc",
            lineWidth: 1
        }
    };
    const indicatorLegendLabels = {
        sma_12: "Średnia krocząca z 12 sesji",
        sma_26: "Średnia krocząca z 26 sesji",
        ema_12: "Wykładnicza średnia krocząca z 12 sesji",
        ema_26: "Wykładnicza średnia krocząca z 26 sesji",
        bollinger_upper: "Bollinger górne pasmo",
        bollinger_mid: "Bollinger średnia z 20 sesji",
        bollinger_lower: "Bollinger dolne pasmo"
    };

    function updateIndicatorLegend() {
        if (!chartLegend) {
            return;
        }

        chartLegend.replaceChildren();

        Object.values(indicatorSeries).flat().forEach(item => {
            const style = indicatorStyles[item.name] || {};
            const legendItem = document.createElement("span");
            const legendLine = document.createElement("span");
            const legendLabel = document.createElement("span");

            legendItem.className = "chart-legend-item";
            legendLine.className = "chart-legend-line";
            legendLine.style.borderTopColor = style.color || "#b7c0cc";
            legendLine.style.borderTopWidth = `${style.lineWidth || 2}px`;
            legendLabel.textContent = indicatorLegendLabels[item.name] || item.name;

            legendItem.append(legendLine, legendLabel);
            chartLegend.appendChild(legendItem);
        });
    }

    function createLineSeries(options) {
        return chart.addSeries
            ? chart.addSeries(LightweightCharts.LineSeries, options)
            : chart.addLineSeries(options);
    }

    function removeLineSeries(series) {
        if (chart.removeSeries) {
            chart.removeSeries(series);
            return;
        }

        series.setData([]);
    }
    function formatPrice(value) {
        if (value === undefined || value === null) {
            return "-";
        }

        return Number(value).toFixed(2);
    }

    chart.subscribeCrosshairMove(param => {
        if (
            !param.point ||
            param.point.x < 0 ||
            param.point.y < 0 ||
            param.point.x > chartElement.clientWidth ||
            param.point.y > chartElement.clientHeight
        ) {
            tooltip.style.display = "none";
            return;
        }

        const candle = param.seriesData.get(candleSeries);

        if (!candle) {
            tooltip.style.display = "none";
            return;
        }

        tooltip.innerHTML = `
            <strong>${candle.time}</strong><br>
            Open: ${formatPrice(candle.open)}<br>
            High: ${formatPrice(candle.high)}<br>
            Low: ${formatPrice(candle.low)}<br>
            Close: ${formatPrice(candle.close)}
        `;

        let left = param.point.x + 15;
        let top = param.point.y + 15;

        if (left + 170 > chartElement.clientWidth) {
            left = param.point.x - 180;
        }

        if (top + 120 > chartElement.clientHeight) {
            top = param.point.y - 130;
        }

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
        tooltip.style.display = "block";
    });

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

    const intervalLabels = {
    "1d": "1D",
    "1w": "1W",
    "1mo": "1M"
    };

    const rangeLabel = document.getElementById("chart-range-label");
    const candleLabel = document.getElementById("chart-candle-label");

    function updateChartLabels(range, interval) {
        if (rangeLabel) {
            rangeLabel.textContent = `Widok: ${range}`;
            rangeLabel.style.display = "inline";
        }

        if (candleLabel) {
            candleLabel.textContent = `Świece: ${intervalLabels[interval]}`;
            candleLabel.style.display = "inline";
        }
    }

    function hideRangeLabel() {
        if (!rangeLabel) {
            return;
        }

        rangeLabel.style.display = "none";
    }
    chartElement.addEventListener("wheel", hideRangeLabel, {
        passive: true
    });

    chartElement.addEventListener("mousedown", hideRangeLabel);

    chartElement.addEventListener("touchstart", hideRangeLabel, {
        passive: true
    });

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

        currentInterval = config.interval;
        updateChartLabels(range, config.interval);

        fetch(`/api/stock/${ticker}/candles?interval=${config.interval}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error("Nie udało się pobrać danych dla tego tickera");
                }

                return response.json();
            })
            .then(data => {
                candleSeries.setData(data);

                if (range === "ALL") {
                    chart.timeScale().fitContent();
                } else {
                    setVisibleDays(data, config.days);
                }

                refreshActiveIndicators();
            })
            .catch(error => {
                console.error("Chart data loading error:", error);
                chartElement.innerHTML = "<p style='color: white;'>Nie znaleziono danych dla tego tickera.</p>";
            });
    }

    function loadIndicators(interval) {
        if (indicatorsCache[interval]) {
            return Promise.resolve(indicatorsCache[interval]);
        }

        return fetch(`/api/stock/${ticker}/indicators?interval=${interval}`, {
            cache: "no-store"
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error("Nie udało się pobrać wskaźników");
                }

                return response.json();
            })
            .then(data => {
                indicatorsCache[interval] = data;
                return indicatorsCache[interval];
            })
            .catch(error => {
                console.error("Indicators loading error:", error);
                return null;
            });
    }

    function toggleIndicator(indicatorName, button) {
        if (indicatorSeries[indicatorName]) {
            indicatorSeries[indicatorName].forEach(item => {
                removeLineSeries(item.series);
            });

            delete indicatorSeries[indicatorName];
            updateIndicatorLegend();
            button.classList.remove("active");
            return;
        }

        button.classList.add("loading");

        loadIndicators(currentInterval).then(data => {
            const seriesNames = indicatorDefinitions[indicatorName] || [indicatorName];

            if (!data || seriesNames.some(seriesName => !data[seriesName])) {
                return;
            }

            indicatorSeries[indicatorName] = seriesNames.map(seriesName => {
                const lineSeries = createLineSeries(indicatorStyles[seriesName] || {});
                lineSeries.setData(data[seriesName]);

                return {
                    name: seriesName,
                    series: lineSeries
                };
            });

            updateIndicatorLegend();
            button.classList.add("active");
        }).finally(() => {
            button.classList.remove("loading");
        });
    }

    function refreshActiveIndicators() {
        const activeIndicatorNames = Object.keys(indicatorSeries);

        if (activeIndicatorNames.length === 0) {
            return;
        }

        loadIndicators(currentInterval).then(data => {
            if (!data) {
                return;
            }

            activeIndicatorNames.forEach(indicatorName => {
                indicatorSeries[indicatorName].forEach(item => {
                    if (data[item.name]) {
                        item.series.setData(data[item.name]);
                    }
                });
            });
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

    const indicatorButtons = document.querySelectorAll(".indicator-tabs button[data-indicator]");

    const predictionStartInput = document.getElementById("start");
const predictionButton = document.querySelector('button[data-indicator="start_prediction"]');

let predictionResultElement = document.getElementById("prediction-result");

if (!predictionResultElement && predictionButton) {
    predictionResultElement = document.createElement("p");
    predictionResultElement.id = "prediction-result";
    predictionButton.insertAdjacentElement("afterend", predictionResultElement);
}

if (predictionButton && predictionStartInput) {
    predictionButton.addEventListener("click", () => {
        const startDate = predictionStartInput.value;

        if (!startDate) {
            predictionResultElement.textContent = "Wybierz date startu predykcji.";
            return;
        }

        predictionButton.disabled = true;
        predictionResultElement.textContent = "Trenowanie modelu...";

        fetch(`/api/stock/${ticker}/prediction?start_date=${encodeURIComponent(startDate)}`, {
            cache: "no-store"
        })
            .then(response => {
                return response.json().then(data => {
                    if (!response.ok) {
                        throw new Error(data.error || "Nie udalo sie uruchomic predykcji");
                    }

                    return data;
                });
            })
            .then(data => {
                const accuracyPercent = (data.accuracy * 100).toFixed(2);

                predictionResultElement.textContent =
                    `Skutecznosc modelu: ${accuracyPercent}% (${data.correct_predictions}/${data.total_predictions})`;

                console.log("Prediction result:", data);
            })
            .catch(error => {
                console.error("Prediction error:", error);
                predictionResultElement.textContent = error.message;
            })
            .finally(() => {
                predictionButton.disabled = false;
            });
    });
}

    indicatorButtons.forEach(button => {
        button.addEventListener("click", event => {
            event.preventDefault();
            toggleIndicator(button.dataset.indicator, button);
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
