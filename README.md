# Klasyfikacja artykułów z podziałem na kategorie / System RAG z dokumentów o ML

*Celem tego ćwiczenia jest rozszerzenie wiedzy praktycznej o NLP/RAG.*


---

## Dataset

Zbiór danych podzielony jest na 1490 rekordów do celów szkoleniowych i 735 do testów. Celem jest zbudowanie systemu, który będzie mógł precyzyjnie klasyfikować wcześniej niepublikowane artykuły do odpowiedniej kategorii.

Kategorie docelowe:
- business,
- tech,
- politics,
- sport,
- entertainment.

Część przykładowego tekstu (kategoria: business):
*worldcom ex-boss launches defence lawyers defending former worldcom chief bernie ebbers against a battery of fraud charges have called a company whistleblower as their first witness. cynthia cooper worldcom s ex-head of internal accounting alerted directors to irregular accounting practices at the us telecoms giant in 2002. her warnings led to the collapse of the firm following the discovery of an $11bn (£5.7bn) accounting fraud. mr ebbers has pleaded not guilty to charges of fraud and conspiracy.*

## Budowa prostego modelu predykcyjnego (TF-IDF, SVM)

### Struktura

Struktura projektu przedstawia się następująco:

```nlp_proj/
├── README.md
├── pyproject.toml
├── .gitignore
├── img/
│
├── data/
│   ├── emb/        # pliki pickle zawierające osadzenia kontekstu do RAG
│   └── raw/        # oryginalne pliki CSV
│
├── docs/           # wyjaśnienia pojęć o ML w formacie .md
│
├── src/
│   └── text_lab/
│       ├── __init__.py
│       ├── io.py           # wczytywanie/zapis danych
│       ├── clean.py        # czyszczenie tekstu
│       ├── features.py     # TfidfVectorizer, przygotowanie X, y
│       ├── tokenization.py # stemming, Lametyzacja
│       ├── train.py        # trening + zapis modelu
│       ├── train.py        # trening + zapis modelu
│       ├── train.py        # trening + zapis modelu
│       └── predict.py      # predykcja na nowym tekście
│
├── notebooks/
│   └── 01_eda.ipynb    # eksploracja danych
│
├── tests/
│   ├── test_clean.py
│   ├── test_rag.py
│   ├── test_train.py
│   └── test_features.py
│
│── outputs/         # utworzone pliki CSV
├── models/          # zapisany pipeline (.joblib)
└── main.py
```

Utworzone w ten sposób moduły pozwoliły na stworzenie przejrzystego i prostego do testowania kodu.

---

### Eksporacyjna Analiza Danych (EDA)

Analiza długości tekstu:

![Długośc tekstu od kategorii](img/length_distribution.png)

Wnioski z całości:
- Obecność duplikatów.
- Lekkie niezbalansowanie klas:

| category | no_cat | perc_of_data |
| :--- | :---: | :---: |
| sport | 342 | 23.75% |
| business | 335 | 23.26% |
| politics | 266 | 18.47% |
| entertainment | 263 | 18.26% |
| tech | 234 | 16.25% |

- Konieczność usunięcia stopwords oraz własnych propozycji słów odrzuconych po analizie.
- Długość tekstu ma wpływ na jego przynależność do danej kategorii.
- Należy ustawić odpowiednią liczbę `max_features` w TF-IDF (34541 unikalnych słów).

---

### Opis rozwiązania

1. **Ładowanie danych**
   - Wykonywane za pomocą `io.py`. Odpowiednia obsługa błędów, podział na typ danych ('train' lub 'test'), normalizacja nazw kolumn.

2. **Przygotowanie zbiorów danych**
   - Wydzielenie zbioru treningowego oraz testowego wraz z podziałem na wartości wejściowe i wyjściowe. Ostatecznie zwracana jest krotka (X_train, X_test, y_train, y_test).

3. **Trenowanie modelu**
   - Zbudowanie potoku danych (pipeline):
     - Wektoryzator słów TF-IDF ze szczegółowo dobraną bazą słów stop (na podstawie analizy) oraz własnym preprocessorem, który został zaimplementowany w module `clean.py` i przetestowany przy użyciu *pytest*.
     - Maszyna wektorów nośnych (SVM). Następnie dobrano zakres parametrów dla obu obiektów w potoku, aby wykonać dostrajanie hiperparametrów za pomocą `GridSearchCV`. Skorzystano z 5-warstwowej walidacji krzyżowej z warstwowym podziałem klas (*stratify=True*). Optymalizowaną metryką było '*f1_macro*'.

4. **Walidacja**

    - Odpowiednio dostrojony model wykorzystano do oceny na zbiorze testowym. Poniżej znajdują się uzyskane wyniki.

        **Raport klasyfikacji**

        | Klasa | Precision | Recall | F1-Score | Support |
        | :--- | :---: | :---: | :---: | :---: |
        | **business** | 0.97 | 0.97 | 0.97 | 67 |
        | **entertainment** | 0.96 | 1.00 | 0.98 | 55 |
        | **politics** | 0.98 | 0.96 | 0.97 | 55 |
        | **sport** | 0.99 | 1.00 | 0.99 | 69 |
        | **tech** | 0.98 | 0.94 | 0.96 | 52 |
        | | | | | |
        | **accuracy** | | | **0.98** | **298** |
        | **macro avg** | 0.98 | 0.98 | 0.98 | 298 |
        | **weighted avg** | 0.98 | 0.98 | 0.98 | 298 |


    - Nagłówki kolumn (*Predicted*) i wierszy (*True*) odpowiadają kolejno klasom: *business, entertainment, politics, sport, tech*.

<!-- 
        **Macierz pomyłek (Confusion Matrix)**

        | Rzeczywiste \ Przewidywane | business | entertainment | politics | sport | tech |
        | :--- | :---: | :---: | :---: | :---: | :---: |
        | **business** | **65** | 0 | 1 | 1 | 0 |
        | **entertainment** | 0 | **55** | 0 | 0 | 0 |
        | **politics** | 1 | 0 | **53** | 0 | 1 |
        | **sport** | 0 | 0 | 0 | **69** | 0 |
        | **tech** | 1 | 2 | 0 | 0 | **49** |
-->

![Macierz Pomyłek](img/conf_matrix.png)

5. **Lematyzacja i Stemming**
    - Zaimoplementowano możliwość wykonania operacji normalizacji tekstu przez odpowiednie flagi parsera. Wykorzystano bibliotekę PyStemmer dla stemmingu oraz spaCy dla lematyzacji, dla której wykorzystano model '*en_core_web_trf*' opierający się na transformerze. Poniżej znajdują się macierze pomyłek dla poszczególnych wersji:

    **Stemming**

    ![Stemming](img/stem_conf_matrix.png)

    **Lematyzacja**

    ![Lematyzacja](img/lem_conf_matrix.png)

6. **Dodanie nieliniowego SVM**
    - Postanowiono przetestować nieliniową wersję modelu SVM. Sprawdzono jądra radialnej funcji bazowej (rbf) oraz wielomianowej (poly). Po dostrajaniu hiperparametrów najlepszym wyborem była rbf.

7. **Wysłanie na Kaggle**
   - Przygotowany zbiór testowy z konkursu na platformie Kaggle, składający się z 722 rekordów, został przepuszczony przez wytrenowane modele. Uzyskane predykcje zostały wysłane w formacie .csv jako *Late Submission*. Obowiązującą metryką w wyzwaniu była dokładność (*Accuracy*). Udało się osiągnąć wynik na poziomie **0.98367** dla modelu bez normalizacji tekstu. Użycie stemmingu pogorszyło wynik do **0.94421**, a lematyzacja do **0.97959**. Uzyskany wynik jest wyższy niż najwyższy rezultat znajdujący się obecnie w tabeli wyników (**0.98231**). Oznacza to, że w tym przypadku odpowiednie czyszczenie danych tekstowych było najlepszym rozwiązaniem.

## RAG

W części RAG udało się zbudować prosty mechanizm wyszukiwania i odpowiedzi na pytania w oparciu o embeddingi:

- dokumenty z katalogu `docs/` oraz artykuły BBC są dzielone na nakładające się chunki,
- chunki są zamieniane na wektory przy użyciu modeli `sentence-transformers`,
- podobieństwo między zapytaniem a fragmentami tekstu jest liczone przez iloczyn skalarny na znormalizowanych embeddingach,
- funkcja `retrieve()` zwraca najbardziej pasujące fragmenty, a `answer()` przekazuje je do modelu `llama3.2` przez `ollama`,
- całość pozwala odpowiadać wyłącznie na podstawie znalezionego kontekstu, bez zgadywania spoza danych.

Przykładowe pytania i uzyskane odpowiedzi z folderu `docs/`:
```
>> uv run main.py --rag
```
```
Pytanie: Czym jest nadmierne dopasowanie?
Odpowiedź: Odpowiedź: Przeuczenie (overfitting) występuje, gdy model zbyt dokładnie dopasowuje się do danych treningowych, w tym do szumu, i słabo działa na nowych danych.
--------------------
Pytanie: Co to overfitting?
Odpowiedź: Overfitting występuje, gdy model zbyt dokładnie dopasowuje się do danych treningowych, w tym do szumu, i słabo działa na nowych danych. Typowym objawem jest duża różnica między wysokim wynikiem na zbiorze treningowym a niższym na testowym.
--------------------
Pytanie: Kiedy się ustawia i co kontrolują hiperparametry?
Odpowiedź: Hiperparametry ustawia się zwykle przed treningiem modelu i kontrolują proces uczenia. Kontrolują różne parametry, takie jak:

- learning rate (stopień uczenia)
- siła regularyzacji (C w SVM)
- `max_depth` w drzewie decyzyjnym
- wagi w modelu liniowym
- wektory nośne w SVM

Te hiperparametry zwykle stroi się walidacją, taką jak `GridSearchCV`, aby znaleźć najbardziej optymalne wartości.
--------------------
Pytanie: Czy mogę użyć danych testowych do trenowania modelu?
Odpowiedź: Nie, nie można użyć danych testowych do trenowania modelu. Użycie tych samych danych do obu etapów procesu uczenia może powodować wyciek danych (data leakage), który może skutecznie spowodować fałszywe wyniki ewaluacji modelu.
```

## Źródło

> Bijoy Bose. BBC News Classification. https://kaggle.com/competitions/learn-ai-bbc, 2019. Kaggle.
> Oryginalny zbiór danych: D. Greene and P. Cunningham. "Practical Evaluation of Recommender Systems," In Proc. 3rd International Conference on Mobile Ubiquitous Computing, Systems, Services and Technologies (UBICOMM'09), 2009.
