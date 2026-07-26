# Architektura projektu

## Cel dokumentu

Dokument opisuje budowę aplikacji, odpowiedzialność jej głównych elementów
oraz przepływ danych pomiędzy frontendem, backendem, bazą danych,
modelem predykcyjnym i symulatorem transakcji.

## Główne elementy systemu

Aplikacja jest podzielona na kilka warstw. Każda warstwa odpowiada za inny
rodzaj operacji i komunikuje się tylko z wybranymi częściami systemu.

| Element | Lokalizacja | Odpowiedzialność                                                                    |
|---|---|-------------------------------------------------------------------------------------|
| Frontend | `templates/`, `static/` | Wyświetla strony, wykresy i wyniki oraz wysyła zapytania do API.                    |
| Aplikacja Flask | `app.py` | Odbiera zapytania użytkownika, wywołuje odpowiednie serwisy, zwraca HTML albo JSON. |
| Pobieranie danych | `services/data_fetcher.py` | Pobiera dane giełdowe i informacje o spółkach.                                      |
| Aktualizacja danych | `services/data_updater.py` | Sprawdza dostępność danych i zapisuje brakujące rekordy.                            |
| Baza danych | `services/db.py` | Odpowiada za połączenie z bazą, odczytywanie i zapisywanie danych.                  |
| Wskaźniki techniczne | `services/technical_indicator_service.py` | Oblicza wskaźniki giełdowe.                                                         |
| Zmiana interwału | `services/resample_service.py` | Łączy dane dzienne w większe świece, na przykład tygodniowe.                        |
| Przygotowanie datasetu | `services/ml/dataset_service.py` | Przygotowuje cechy i target wykorzystywane do trenowania modelu.                    |
| Model predykcyjny | `services/ml/model_service.py` | Trenuje model, wykonuje walk-forward i przygotowuje predykcje oraz metryki.         |
| Symulator transakcji | `services/simulator_service.py` | Symuluje grę na giełdzie korzystając z predykcji modelu oraz innych strategii.      |
| Informacje o spółkach | `services/company_service.py` | Pobiera i zapisuje nazwy spółek.                                                    |
| Testowanie modelu | `evaluate_model.py` | Testy modelu i symulatora dla wielu spółek.

## Przepływ otwierania strony spółki

Poniższy diagram pokazuje przepływ od wpisania tickera na stronie głównej
do wyświetlenia strony spółki i narysowania wykresu.

```mermaid
flowchart TD
    A["Użytkownik wpisuje ticker"] --> B["Wysłanie formularza POST /"]
    B --> C["Funkcja index()"]
    C --> D{"Ticker ma poprawny format?"}

    D -->|Nie| E["Wyświetlenie index.html z błędem"]
    D -->|Tak| F["Przekierowanie do /stock/ticker"]

    F --> G["Funkcja stock(ticker)"]
    G --> H["ensure_stock_data()"]
    H --> I{"Dane są dostępne i aktualne?"}

    I -->|Nie| J["fetch_stock_data()"]
    J --> K["save_stock_data()"]
    I -->|Tak| L["ensure_stock_name()"]
    K --> L

    L --> M["get_latest_stock_summary()"]
    M --> N["get_stock_data()"]
    N --> O["get_latest_indicators()"]
    O --> P["render_template(stock.html)"]

    P --> Q["Przeglądarka uruchamia stock_chart.js"]
    Q --> R["GET /api/stock/ticker/candles"]
    R --> S["JavaScript rysuje wykres"]
```


## Przepływ uruchomienia modelu i symulatora

Proces rozpoczyna się po kliknięciu przycisku „Uruchom model”.
Frontend wysyła zakres dat do API, backend przygotowuje dataset,
wykonuje walk-forward, uruchamia dwie strategie i zwraca wyniki do strony.

```mermaid
flowchart TD
    subgraph FRONTEND["Frontend - stock_chart.js"]
        A["Kliknięcie: Uruchom model"] --> B["Odczyt start_date i end_date"]
        B --> C{"Daty są poprawne?"}
        C -->|Nie| D["Wyświetlenie komunikatu o błędzie"]
        C -->|Tak| E["GET /api/stock/ticker/prediction"]
    end

    subgraph API["Backend - app.py"]
        F["stock_prediction_api()"] --> G["ensure_stock_data()"]
        G --> H["get_stock_data()"]
        H --> I["train_and_predict()"]
    end

    E --> F

    subgraph DATASET["Przygotowanie danych - dataset_service.py"]
        J["prepare_dataset()"] --> K["calculate_all_indicators()"]
        K --> L["Utworzenie dodatkowych cech"]
        L --> M["Utworzenie target i danych następnego dnia"]
        M --> N["Usunięcie niepełnych rekordów"]
    end

    I --> J

    subgraph MODEL["Model - model_service.py"]
        O["split_dataset_by_date()"] --> P["Podział na train i test"]
        P --> Q["Przejście po kolejnych dniach testowych"]
        Q --> R{"Czy minęło 20 predykcji?"}
        R -->|Tak| S["Ponowne trenowanie Random Forest"]
        R -->|Nie| T["Użycie obecnego modelu"]
        S --> U["Wykonanie predykcji"]
        T --> U
        U --> V{"Czy pozostały kolejne dni?"}
        V -->|Tak| Q
        V -->|Nie| W["Obliczenie metryk i utworzenie prediction_records"]
    end

    N --> O

    subgraph SIMULATOR["Symulator - simulator_service.py"]
        X["run_backtest()"] --> Z["model_summary"]
        Y["run_buy_and_hold()"] --> AA["buyandhold_summary"]
    end

    W --> X
    W --> Y

    Z --> AB["Zbudowanie odpowiedzi"]
    AA --> AB
    AB --> AC["jsonify()"]
    AC --> AD["Odpowiedź JSON wraca do stock_chart.js"]
    AD --> AE["Wyświetlenie skuteczności i tabeli"]
    AD --> AF["Narysowanie wykresu wartości portfela"]
```

## Przepływ danych giełdowych

Aplikacja nie pobiera całej historii z yfinance przy każdym zapytaniu.
Najpierw sprawdza dane zapisane w PostgreSQL, a następnie pobiera tylko
brakującą historię.

```mermaid
flowchart TD
    subgraph TRIGGERS["Miejsca wymagające danych giełdowych"]
        A["Strona główna"]
        B["Strona spółki"]
        C["API świec"]
        D["API wskaźników"]
        E["API modelu"]
    end

    A --> F["ensure_stock_data()"]
    B --> F
    C --> F
    D --> F
    E --> F

    subgraph UPDATER["Aktualizacja danych - data_updater.py"]
        F --> G["Odczyt ostatniej daty z tabeli stock_prices"]
        G --> H{"Czy spółka ma zapisane dane?"}

        H -->|Nie| I["fetch_stock_data(period='max')"]
        H -->|Tak| J{"Czy ostatnia data jest dzisiejsza?"}

        J -->|Tak| K["Dane są aktualne"]
        J -->|Nie| L["Wyznaczenie następnego dnia"]
        L --> M["fetch_stock_data(start_date=next_day)"]
    end

    subgraph FETCHER["Pobieranie danych - data_fetcher.py"]
        I --> N["yfinance.download()"]
        M --> N
        N --> O["Zmiana nazw kolumn"]
        O --> P["Dodanie ticker i interval"]
        P --> Q["Konwersja daty"]
        Q --> R{"Czy pobrano rekordy?"}
    end

    R -->|Tak| S["save_stock_data()"]
    R -->|Nie| T{"Czy wcześniejsze dane istnieją?"}

    T -->|Tak| K
    T -->|Nie| U["Brak danych - zwrócenie błędu"]

    subgraph DATABASE["Baza danych - db.py"]
        S --> V[("PostgreSQL: stock_prices")]
        K --> V
        V --> W["get_recent_stock_prices()"]
        V --> X["get_latest_stock_summary()"]
        V --> Y["get_stock_data()"]
    end

    W --> Z["Kafelki na stronie głównej"]
    X --> AA["Podsumowanie sesji na stronie spółki"]

    Y --> AB["Świece na wykresie"]
    Y --> AC["Wskaźniki techniczne"]
    Y --> AD["Dataset modelu"]
```

## Endpointy aplikacji

Poniższa tabela przedstawia trasy udostępniane przez aplikację Flask.
Trasy zwracają gotową stronę HTML, przekierowanie albo dane JSON.

| Metoda | Endpoint | Funkcja | Parametry | Wynik                                                              |
|---|---|---|---|--------------------------------------------------------------------|
| `GET` | `/` | `index()` | Opcjonalny parametr `error` | Strona główna z wyszukiwarką                                       |
| `POST` | `/` | `index()` | Pole formularza `ticker` | Przekierowanie na stronę wybranej spółki albo komunikat o błędzie. |
| `GET` | `/stock/<ticker>` | `stock()` | Ticker umieszczony w adresie | Strona spółki z podsumowaniem i wskaźnikami.                       |
| `GET` | `/api/stock/<ticker>/candles` | `stock_candles_api()` | Opcjonalny `interval`, domyślnie `1d` | JSON zawierający świece dla wykresu.                               |
| `GET` | `/api/stock/<ticker>/indicators` | `stock_indicators_api()` | Opcjonalny `interval`, domyślnie `1d` | JSON zawierający serie wskaźników technicznych.                    |
| `GET` | `/api/stock/<ticker>/prediction` | `stock_prediction_api()` | Wymagane `start_date` i `end_date` | JSON zawierający predykcje, metryki modelu i wyniki symulatora.    |

