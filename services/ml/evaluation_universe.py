UNIVERSE_AS_OF = "2026-11-02"
UNIVERSE_SOURCE_URL = "https://www.kadoa.com/companies"
GROUP_SPLIT_SEED = 26
UNIVERSE_MIN_HISTORY_START = "2021-01-01"
UNIVERSE_REPLACEMENTS = {"GEV": "ETN", "CEG": "ETR"}
EVALUATION_PERIODS = {
"train_start": "2021-01-01",
"validation_start": "2025-01-01",
"calibration_start": "2025-07-01",
"test_start": "2026-01-01",
"test_end": "2026-11-01"}

#7 największych spółek dla danego sektora z historią od przynajmniej 2021 roku[ranking 02.11.2026]
EVALUATION_UNIVERSE = {
    "information_technology": {"seen": ("AAPL", "MSFT", "AVGO", "MU", "PLTR"), "unseen": ("NVDA", "AMD")},
    "financials": {"seen": ("BRK-B", "V", "MA", "BAC", "GS"), "unseen": ("JPM", "MS")},
    "health_care": {"seen": ("LLY", "JNJ", "ABBV", "TMO", "AMGN"), "unseen": ("MRK", "UNH")},
    "consumer_discretionary": {"seen": ("TSLA", "HD", "MCD", "LOW", "SBUX"), "unseen": ("AMZN", "TJX")},
    "consumer_staples": {"seen": ("WMT", "KO", "PM", "PEP", "MO"), "unseen": ("COST", "PG")},
    "industrials": {"seen": ("CAT", "RTX", "UNP", "LMT", "ETN"), "unseen": ("GE", "DE")},
    "communication_services": {"seen": ("GOOGL", "META", "NFLX", "DIS", "T"), "unseen": ("VZ", "TMUS")},
    "energy": {"seen": ("XOM", "COP", "MPC", "VLO", "WMB"), "unseen": ("CVX", "PSX")},
    "materials": {"seen": ("LIN", "NEM", "FCX", "SHW", "CRH"), "unseen": ("ECL", "APD")},
    "utilities": {"seen": ("NEE", "DUK", "AEP", "SRE", "ETR"), "unseen": ("SO", "D")},
    "real_estate": {"seen": ("WELL", "PLD", "EQIX", "SPG", "PSA"), "unseen": ("AMT", "DLR")}
}

SEEN_TICKERS = tuple(ticker for cohorts in EVALUATION_UNIVERSE.values() for ticker in cohorts["seen"])
UNSEEN_TICKERS = tuple(ticker for cohorts in EVALUATION_UNIVERSE.values() for ticker in cohorts["unseen"])
ALL_TICKERS = SEEN_TICKERS + UNSEEN_TICKERS
SECTOR_ASSIGNMENTS = {ticker: sector for sector, cohorts in EVALUATION_UNIVERSE.items() for tickers in cohorts.values() for ticker in tickers}
