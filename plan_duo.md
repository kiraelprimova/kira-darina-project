# План разработки итоговой работы (plan\_duo)

# 

# Этап 1. Подготовка структуры данных

**Задача:** Загрузить твиты, объединить их и оставить только нужные колонки.

* **Откуда берем пример:** В тетрадке `2025/models s bank(5).ipynb` (самое начало, ячейки 3-10). Мы там читали CSV с помощью `pd.read\_csv` и отбирали нужные данные.
* **Как это выглядит в коде:**

&#x20;   ```python
    import pandas as pd
    
    # Загружаем позитивные и негативные твиты (они разделены через ';')
    cols = \['id', 'date', 'name', 'text', 'type', 'rep', 'rtw', 'fav', 'st1', 'usr', 'st2', 'st3']
    df\_pos = pd.read\_csv('итоговая работа/pos.csv', sep=';', names=cols, header=None)
    df\_neg = pd.read\_csv('итоговая работа/neg.csv', sep=';', names=cols, header=None)
    
    # Ставим метку класса: 1 - позитив, 0 - негатив
    df\_pos\['label'] = 1
    df\_neg\['label'] = 0
    
    # Соединяем в один большой датасет
    df = pd.concat(\[df\_pos, df\_neg], ignore\_index=True)
    
    # Оставляем только текст и метку, остальное нам не нужно для тональности
    df = df\[\['text', 'label']]
    ```

\---

## Этап 2. Предобработка текстов (Очистка и Лемматизация)

**Задача:** Очистить текст от мусора (ссылки, знаки препинания) и привести слова к начальной форме.

* **Откуда берем пример:** Тетрадка `2026/notebook/pdf/nlp movie.ipynb` (ячейки с 14 по 32). Там мы использовали регулярные выражения для очистки, стоп-слова из `nltk` и лемматизацию через `pymorphy3`.
* **Как это выглядит в коде:**

&#x20;   ```python
    import re
    import pymorphy3
    from nltk.corpus import stopwords
    
    morph = pymorphy3.MorphAnalyzer()
    stop\_words = set(stopwords.words('russian'))
    
    def clean\_text(text):
        text = text.lower()
        text = re.sub(r'http\\S+|@\\S+', '', text)  # удаляем ссылки и упоминания
        text = re.sub(r'\[^а-яё\\s]', ' ', text)     # оставляем только русские буквы
        words = text.split()
        # Лемматизируем (приводим к начальной форме) и убираем стоп-слова
        cleaned = \[morph.parse(w)\[0].normal\_form for w in words if w not in stop\_words]
        return " ".join(cleaned)
        
    df\['clean\_text'] = df\['text'].apply(clean\_text)
    ```

\---

## Этап 3. Тематическое моделирование и Облако слов

**Задача:** Нарисовать облако слов и обучить модель LDA/NMF, чтобы понять, о чём пишут люди.

* **Откуда берем пример:** Тетрадка `2026/notebook/pdf/nlp movie.ipynb` (ячейки 42-66 для WordCloud и 70-85 для LDA и NMF моделей).
* **Как это выглядит в коде:**

&#x20;   ```python
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature\_extraction.text import CountVectorizer
    
    # Облако слов
    pos\_text = " ".join(df\[df\['label'] == 1]\['clean\_text'])
    wc = WordCloud(width=800, height=400, background\_color='white').generate(pos\_text)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    
    # LDA для поиска тем
    cv = CountVectorizer(max\_features=1000)
    dtm = cv.fit\_transform(df\['clean\_text'])
    lda = LatentDirichletAllocation(n\_components=5, random\_state=42)
    lda.fit(dtm)
    ```

\---

## Этап 4. Векторизация текста (TF-IDF)

**Задача:** Перевести очищенные тексты в числовые вектора.

* **Откуда берем пример:** Тетрадка `2026/notebook/pdf/nlp movie.ipynb` (ячейки 54-55). Там мы делали векторизацию через `TfidfVectorizer`.
* **Как это выглядит в коде:**

&#x20;   ```python
    from sklearn.feature\_extraction.text import TfidfVectorizer
    
    # Переводим в вектора
    tfidf = TfidfVectorizer(max\_features=5000, min\_df=2)
    X = tfidf.fit\_transform(df\['clean\_text'])
    y = df\['label']
    ```

\---

## Этап 5. Обучение классификаторов и сравнение моделей

**Задача:** Обучить 3+ модели и проверить, какая лучше предсказывает тональность.

* **Откуда берем пример:** Тетрадка `2025/models s bank(5).ipynb` (ячейки 81-132). Там мы делили данные, обучали `LogisticRegression`, `RandomForestClassifier` и `SVC`, считали `f1\_score`, строили матрицу ошибок и рисовали графики ROC AUC.
* **Как это выглядит в коде:**

&#x20;   ```python
    from sklearn.model\_selection import train\_test\_split
    from sklearn.linear\_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification\_report
    
    X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=42)
    
    # Обучаем логистическую регрессию
    lr = LogisticRegression()
    lr.fit(X\_train, y\_train)
    
    # Смотрим качество
    preds = lr.predict(X\_test)
    print(classification\_report(y\_test, preds))
    ```

\---

## Этап 6. Парсинг новых отзывов и дообучение (Активное обучение)

**Задача:** Собрать не менее 500 отзывов с сайта (например, Кинопоиск или Хабр). Использовать нашу модель, чтобы отобрать самые "уверенные" предсказания и добавить их в обучающий датасет, а потом переобучить модель.

* **Откуда берем пример:** Тетрадки `2026/notebook/parsingKinopoisk.ipynb` (Selenium-парсер) и `2026/notebook/pdf/habr\_parser.ipynb`.
* **Как это выглядит в коде:**

&#x20;   ```python
    # Делаем прогноз вероятностей для спарсенных текстов
    probs = lr.predict\_proba(X\_parsed)\[:, 1]
    
    # Отбираем только те, где вероятность > 0.8 (явно позитивный) или < 0.2 (явно негативный)
    confident\_mask = (probs > 0.8) | (probs < 0.2)
    
    # Добавляем их в наш обучающий корпус и переобучаем вектормайзер и модель!
    ```

\---

## Этап 7. Разработка API

**Задача:** Написать простой бэкенд на Flask/FastAPI, который принимает текст и возвращает его тональность.

* **Откуда берем пример:** Идея взята из связки скриптов и парсера в `2026/notebook/parsingKinopoisk.ipynb`.
* **Как это выглядит в коде:**

&#x20;   ```python
    from flask import Flask, request, jsonify
    app = Flask(\_\_name\_\_)
    
    @app.route('/predict', methods=\['POST'])
    def predict():
        data = request.json
        text = data\['text']
        cleaned = clean\_text(text)
        vector = tfidf.transform(\[cleaned])
        prob = lr.predict\_proba(vector)\[0]\[1]
        sentiment = 'positive' if prob > 0.5 else 'negative'
        return jsonify({'sentiment': sentiment, 'probability': prob})
        
    if \_\_name\_\_ == '\_\_main\_\_':
        app.run(port=5000)
    ```

\---

## Этап 8. Графическое приложение (GUI) на PyQt5

**Задача:** Сделать оконную программу на PyQt5. В ней будут текстовое поле для ввода отзыва, кнопка проверки, вкладка со статистикой (графики, средняя длина постов, доли классов) и вкладка со справкой.

* **Как мы сделаем:** Используем элементы `QWidget`, `QPushButton`, `QTabWidget` и библиотеку `matplotlib` для встраивания красивых графиков в окно программы.

