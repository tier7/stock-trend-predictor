const chartElement = document.getElementById("stock-chart");
const stockChartCardElement = document.querySelector(".stock-layout .chart-card");
const stockToolsElement = document.querySelector(".stock-tools");
let stockLayoutResizeObserver = null;

function syncStockToolsHeight() {
    if (!stockChartCardElement || !stockToolsElement) {
        return;
    }

    if (window.innerWidth <= 760) {
        stockToolsElement.style.removeProperty("height");
        return;
    }

    stockToolsElement.style.height = `${stockChartCardElement.offsetHeight}px`;
}

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
    const predictionEndInput = document.getElementById("end");
    const predictionButton = document.getElementById("simulation-run-button");
    const simulationModelSelect = document.getElementById("simulation-model");
    const modelDescription = document.getElementById("simulation-model-description");
    const confidenceThresholdInput = document.getElementById("confidence-threshold");
    const confidenceThresholdValue = document.getElementById("confidence-threshold-value");
    const confidenceThresholdDescription = document.getElementById("confidence-threshold-description");
    let activeGlobalModel = null;
    let globalModelLoading = true;
    let globalModelError = "";
    let simulationBusy = false;

    const backtestResultsElement = document.getElementById("backtest-results");
    const modelReturnElement = document.getElementById("model-return");
    const buyHoldReturnElement = document.getElementById("buy-hold-return");
    const modelCagrElement = document.getElementById("model-cagr");
    const buyHoldCagrElement = document.getElementById("buy-hold-cagr");
    const modelDrawdownElement = document.getElementById("model-drawdown");
    const buyHoldDrawdownElement = document.getElementById("buy-hold-drawdown");
    const modelExposureElement = document.getElementById("model-exposure");
    const modelFinalValueElement = document.getElementById("model-final-value");
    const buyHoldFinalValueElement = document.getElementById("buy-hold-final-value");
    const modelTransactionsElement = document.getElementById("model-transactions");
    const portfolioChartCard = document.getElementById("portfolio-chart-card");
    const portfolioChartElement = document.getElementById("portfolio-chart");
    const modelStatsElement = document.getElementById("model-stats");
    const modelClosedTransactionsElement = document.getElementById("model-closed-transactions");
    const modelProfitableElement = document.getElementById("model-profitable");
    const modelUnprofitableElement = document.getElementById("model-unprofitable");
    const modelWinRatioElement =document.getElementById("model-winratio");
    const sidebarTabButtons = Array.from(
        document.querySelectorAll("[data-sidebar-tab]")
    );
    const sidebarTabPanels = Array.from(
        document.querySelectorAll("[data-sidebar-panel]")
    );
    let portfolioChart = null;

    function activateSidebarTab(tabName) {
        sidebarTabButtons.forEach(button => {
            const isActive = button.dataset.sidebarTab === tabName;
            button.classList.toggle("active", isActive);
            button.setAttribute("aria-selected", String(isActive));
            button.tabIndex = isActive ? 0 : -1;
        });

        sidebarTabPanels.forEach(panel => {
            panel.hidden = panel.dataset.sidebarPanel !== tabName;
        });

        if (tabName === "simulator" && portfolioChart && !portfolioChartCard.hidden) {
            requestAnimationFrame(() => {
                portfolioChart.applyOptions({
                    width: portfolioChartElement.clientWidth
                });
            });
        }
    }

    sidebarTabButtons.forEach((button, buttonIndex) => {
        button.addEventListener("click", () => {
            activateSidebarTab(button.dataset.sidebarTab);
        });

        button.addEventListener("keydown", event => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
                return;
            }

            event.preventDefault();

            let nextIndex = buttonIndex;

            if (event.key === "ArrowLeft") {
                nextIndex = (buttonIndex - 1 + sidebarTabButtons.length) % sidebarTabButtons.length;
            } else if (event.key === "ArrowRight") {
                nextIndex = (buttonIndex + 1) % sidebarTabButtons.length;
            } else if (event.key === "Home") {
                nextIndex = 0;
            } else if (event.key === "End") {
                nextIndex = sidebarTabButtons.length - 1;
            }

            const nextButton = sidebarTabButtons[nextIndex];
            activateSidebarTab(nextButton.dataset.sidebarTab);
            nextButton.focus();
        });
    });

    function formatPercent(value) {
        return `${(Number(value) * 100).toFixed(2)}%`;
    }
    function formatMoney(value) {
        return Number(value).toFixed(2);
    }

    function renderPortfolioChart(modelHistory, buyHoldHistory) {
    if (
        !portfolioChartCard ||
        !portfolioChartElement ||
        !modelHistory ||
        !buyHoldHistory
    ) {
        return;
    }

    portfolioChartCard.hidden = false;

    if (portfolioChart) {
        portfolioChart.remove();
        portfolioChart = null;
    }

    portfolioChartElement.replaceChildren();

    portfolioChart = LightweightCharts.createChart(
        portfolioChartElement,
        {
            width: portfolioChartElement.clientWidth,
            height: 280,

            layout: {
                background: {
                    type: "solid",
                    color: "#0b111a"
                },
                textColor: "#d1d4dc"
            },

            grid: {
                vertLines: {
                    color: "#1e2735"
                },
                horzLines: {
                    color: "#1e2735"
                }
            },

            rightPriceScale: {
                borderColor: "#263141"
            },

            timeScale: {
                borderColor: "#263141"
            },

            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal
            },

            localization: {
                priceFormatter: value => Number(value).toFixed(2)
            }
        }
    );

        const modelSeriesOptions = {
            color: "#0ecb81",
            lineWidth: 2,
            priceLineVisible: false
        };

        const buyHoldSeriesOptions = {
            color: "#f0b90b",
            lineWidth: 2,
            priceLineVisible: false
        };

        const modelSeries = portfolioChart.addSeries
            ? portfolioChart.addSeries(
                LightweightCharts.LineSeries,
                modelSeriesOptions
            )
            : portfolioChart.addLineSeries(modelSeriesOptions);

        const buyHoldSeries = portfolioChart.addSeries
            ? portfolioChart.addSeries(
                LightweightCharts.LineSeries,
                buyHoldSeriesOptions
            )
            : portfolioChart.addLineSeries(buyHoldSeriesOptions);

    const modelData = modelHistory.map(point => ({
        time: point.date,
        value: Number(point.value)
    }));

    const buyHoldData = buyHoldHistory.map(point => ({
        time: point.date,
        value: Number(point.value)
    }));

    modelSeries.setData(modelData);
    buyHoldSeries.setData(buyHoldData);

    portfolioChart.timeScale().fitContent();
}

    let predictionResultElement = document.getElementById("prediction-result");

    if (!predictionResultElement && predictionButton) {
        predictionResultElement = document.createElement("p");
        predictionResultElement.id = "prediction-result";
        predictionButton.insertAdjacentElement("afterend", predictionResultElement);
    }

    async function fetchJson(url, options = {}) {
        const response = await fetch(url, { cache: "no-store", ...options });
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("Serwer nie zwrócił wyników. Sprawdź, czy aplikacja działa.");
        }
        if (!response.ok) {
            throw new Error(data.error || "Nie udało się wykonać operacji.");
        }
        return data;
    }

    function updateSimulationControls() {
        const isGlobal = simulationModelSelect.value === "global";
        predictionButton.disabled = simulationBusy || (isGlobal && (globalModelLoading || !activeGlobalModel));
        simulationModelSelect.disabled = simulationBusy;
        confidenceThresholdInput.disabled = simulationBusy;
        predictionButton.textContent = simulationBusy ? "Trwa symulacja…" : "Uruchom symulację";
        if (isGlobal && activeGlobalModel) {
            const cutoff = activeGlobalModel.training_cutoff;
            predictionStartInput.min = cutoff;
            if (!predictionStartInput.value) predictionStartInput.value = cutoff;
            modelDescription.textContent =
                `Gotowy model uczony na ${activeGlobalModel.train_tickers.length} spółkach. Symulacja od ${cutoff}, bez ponownego uczenia.`;
        } else {
            predictionStartInput.removeAttribute("min");
            modelDescription.textContent = isGlobal
                ? (globalModelLoading ? "Wczytywanie gotowego modelu globalnego…" : globalModelError)
                : `Model uczy się na historii ${ticker} i ponawia uczenie co 20 sesji.`;
        }
    }

    async function refreshActiveGlobalModel() {
        try {
            const data = await fetchJson("/api/models/global/active");
            activeGlobalModel = data.model;
        } catch (error) {
            globalModelError = error.message;
        } finally {
            globalModelLoading = false;
            updateSimulationControls();
        }
    }

    function clearSimulationResults() {
        predictionResultElement.textContent = "";
        backtestResultsElement.hidden = true;
        portfolioChartCard.hidden = true;
        modelStatsElement.hidden = true;
    }

    function describeConfidenceThreshold() {
        const percentage = Number(confidenceThresholdInput.value);
        confidenceThresholdValue.value = `${percentage}%`;
        confidenceThresholdDescription.textContent = percentage === 50
            ? "Kupno: szansa wzrostu > 50%. Sprzedaż: szansa wzrostu < 50%. Przy dokładnie 50% utrzymujesz obecną pozycję."
            : `Kupno: szansa wzrostu ≥ ${percentage}%. Sprzedaż: szansa wzrostu ≤ ${100 - percentage}%. Pomiędzy progami utrzymujesz obecną pozycję.`;
        confidenceThresholdInput.setAttribute("aria-valuetext", confidenceThresholdDescription.textContent);
    }

    function renderSimulation(data, modelLabel) {
        predictionResultElement.textContent =
            `${modelLabel}: skuteczność ${(data.accuracy * 100).toFixed(2)}% (${data.correct_predictions}/${data.total_predictions}). Próg: ${Math.round(data.model_summary.confidence_threshold * 100)}%.`;
        const modelSummary = data.model_summary;
        const buyHoldSummary = data.buyandhold_summary;
        modelClosedTransactionsElement.textContent = modelSummary.closed_trades.length;
        modelProfitableElement.textContent = modelSummary.profitable_trades;
        modelUnprofitableElement.textContent = modelSummary.losing_trades;
        modelWinRatioElement.textContent = formatPercent(modelSummary.win_rate);
        modelReturnElement.textContent = formatPercent(modelSummary.total_return);
        buyHoldReturnElement.textContent = formatPercent(buyHoldSummary.total_return);
        modelCagrElement.textContent = formatPercent(modelSummary.cagr);
        buyHoldCagrElement.textContent = formatPercent(buyHoldSummary.cagr);
        modelDrawdownElement.textContent = formatPercent(modelSummary.max_drawdown);
        buyHoldDrawdownElement.textContent = formatPercent(buyHoldSummary.max_drawdown);
        modelExposureElement.textContent = formatPercent(modelSummary.market_exposure);
        modelFinalValueElement.textContent = formatMoney(modelSummary.final_value);
        buyHoldFinalValueElement.textContent = formatMoney(buyHoldSummary.final_value);
        modelTransactionsElement.textContent = modelSummary.total_transactions;
        backtestResultsElement.hidden = false;
        modelStatsElement.hidden = false;
        renderPortfolioChart(modelSummary.equity_history, buyHoldSummary.equity_history);
    }

    async function runSimulation() {
        if (simulationBusy) return;
        const mode = simulationModelSelect.value;
        const startDate = predictionStartInput.value;
        const endDate = predictionEndInput.value;

        if (!startDate || !endDate) {
            predictionResultElement.textContent = "Wybierz datę początku i końca testu.";
            return;
        }

        if (startDate >= endDate) {
            predictionResultElement.textContent = "Data końcowa musi być późniejsza od początkowej.";
            return;
        }
        const feeInput = document.getElementById("fee-value");
        const cashInput = document.getElementById("money-value");
        const feePercent = Number(feeInput.value);
        const initialCash = Number(cashInput.value);
        const confidencePercent = Number(confidenceThresholdInput.value);
        if (!feeInput.value || !cashInput.value || !Number.isFinite(feePercent) || feePercent < 0 || !Number.isFinite(initialCash) || initialCash <= 0) {
            predictionResultElement.textContent = "Podaj dodatni kapitał i nieujemną prowizję.";
            return;
        }
        if (!Number.isFinite(confidencePercent) || confidencePercent < 50 || confidencePercent > 100) {
            predictionResultElement.textContent = "Ustaw próg pewności od 50% do 100%.";
            return;
        }
        if (mode === "global") {
            if (!activeGlobalModel || startDate < activeGlobalModel.training_cutoff) {
                predictionResultElement.textContent = activeGlobalModel
                    ? `Ten model można testować od ${activeGlobalModel.training_cutoff}. Wybierz późniejszy początek symulacji.`
                    : "Brak gotowego modelu globalnego. Wybierz model lokalny.";
                return;
            }
        }
        const parameters = {
            start_date: startDate,
            end_date: endDate,
            fee_rate: feePercent / 100,
            initial_cash: initialCash,
            confidence_threshold: confidencePercent / 100
        };

        simulationBusy = true;
        updateSimulationControls();
        clearSimulationResults();
        predictionResultElement.textContent = mode === "global"
            ? "Symulacja gotowym modelem globalnym…"
            : "Uczenie modelu lokalnego i symulacja…";
        try {
            let data;
            let modelLabel;
            if (mode === "global") {
                data = await fetchJson(`/api/stock/${ticker}/global-simulation`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ ...parameters, model_id: activeGlobalModel.model_id })
                });
                modelLabel = "Model globalny";
            } else {
                data = await fetchJson(`/api/stock/${ticker}/prediction?${new URLSearchParams(parameters)}`);
                modelLabel = "Model lokalny";
            }
            renderSimulation(data, modelLabel);
        } catch (error) {
            predictionResultElement.textContent = error.message;
        } finally {
            simulationBusy = false;
            updateSimulationControls();
        }
    }

    if (predictionButton && simulationModelSelect && predictionStartInput && predictionEndInput) {
        predictionButton.addEventListener("click", runSimulation);
        confidenceThresholdInput.addEventListener("input", () => {
            describeConfidenceThreshold();
            clearSimulationResults();
        });
        simulationModelSelect.addEventListener("change", () => {
            clearSimulationResults();
            updateSimulationControls();
        });
        updateSimulationControls();
        describeConfidenceThreshold();
        refreshActiveGlobalModel();
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
    syncStockToolsHeight();

    if (typeof ResizeObserver !== "undefined" && stockChartCardElement) {
        stockLayoutResizeObserver = new ResizeObserver(syncStockToolsHeight);
        stockLayoutResizeObserver.observe(stockChartCardElement);
    }

    window.addEventListener("resize", () => {
        syncStockToolsHeight();

        chart.applyOptions({
            width: chartElement.clientWidth
        });

        if (portfolioChart && portfolioChartCard && !portfolioChartCard.hidden) {
            portfolioChart.applyOptions({
                width: portfolioChartElement.clientWidth
            });
        }
    });
}
