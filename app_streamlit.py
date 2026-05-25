import streamlit as st
import requests
import json
import os

# Сбрасываем прокси-переменные, как учили в презентации, 
# чтобы запросы к локальному API не шли через учебный прокси
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''

# Настройка страницы
st.set_page_config(
    page_title="Анализ тональности твитов",
    page_icon="🐦",
    layout="wide"
)

# Красивое оформление заголовка
st.markdown("<h1 style='text-align: center; color: #1DA1F2;'>🐦 Анализ тональности русскоязычных твитов</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em;'>Учебное приложение для определения эмоциональной окраски текста с помощью модели машинного обучения</p>", unsafe_allow_html=True)
st.markdown("---")

# Функция загрузки статистики из JSON
@st.cache_data
def load_stats():
    # Файл лежит в той же папке, что и этот скрипт
    stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Не удалось прочитать файл статистики: {e}")
    return None

stats_data = load_stats()

# Создаем три вкладки
tab_predict, tab_stats, tab_help = st.tabs([
    "🔮 Анализ текста", 
    "📊 Статистика датасета", 
    "ℹ️ Справка по командам"
])

# ==================== ВКЛАДКА 1: АНАЛИЗ ТЕКСТА ====================
with tab_predict:
    st.header("Определение тональности текста")
    st.write("Введите любой текст на русском языке, и наша модель определит, положительный он или отрицательный.")
    
    # Поле ввода текста
    user_input = st.text_area(
        "Ваш комментарий или твит:", 
        placeholder="Напишите что-нибудь хорошее или плохое...",
        height=120
    )
    
    # Кнопка для запуска анализа
    if st.button("Проанализировать тональность", type="primary"):
        if not user_input.strip():
            st.warning("Пожалуйста, введите текст для анализа!")
        else:
            # Адрес нашего локального FastAPI
            api_url = "http://127.0.0.1:8000/predict"
            
            with st.spinner("Отправка запроса к API модели..."):
                try:
                    response = requests.post(api_url, json={"text": user_input}, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Извлекаем данные из ответа API
                        sentiment = result.get("sentiment")
                        sentiment_code = result.get("sentiment_code")
                        probability = result.get("probability")
                        cleaned_text = result.get("clean_text")
                        
                        st.markdown("### Результаты анализа:")
                        
                        # Отображение в зависимости от тональности
                        if sentiment_code == 1:
                            st.success(f"**Тональность:** Положительная 😊 (Класс {sentiment_code})")
                            # Вероятность положительного класса
                            st.write(f"Вероятность положительного класса: **{probability:.2%}**")
                            st.progress(probability)
                        else:
                            st.error(f"**Тональность:** Отрицательная 😔 (Класс {sentiment_code})")
                            st.write(f"Вероятность положительного класса: **{probability:.2%}** (ближе к 0)")
                            st.progress(probability)
                            
                        # Вывод очищенного текста для справки
                        with st.expander("Посмотреть текст после предобработки (очистки и лемматизации):"):
                            st.code(cleaned_text, language="")
                            st.write("_Примечание: Именно в таком виде текст отправляется в модель классификации._")
                            
                    else:
                        st.error(f"Ошибка API (Код {response.status_code}): {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Не удалось подключиться к API бэкенда! Убедитесь, что сервер FastAPI запущен командой `uvicorn main:app --reload` на порту 8000.")
                except Exception as e:
                    st.error(f"Произошла непредвиденная ошибка: {e}")

# ==================== ВКЛАДКА 2: СТАТИСТИКА ДАТАСЕТА ====================
with tab_stats:
    st.header("Статистика исходного Twitter-датасета")
    st.write("Эти данные были рассчитаны на основе размеченных корпусов положительных (`pos.csv`) и отрицательных (`neg.csv`) твитов.")
    
    if stats_data is None:
        st.warning("Файл `dataset_stats.json` со статистикой не найден или поврежден. Пожалуйста, запустите расчет статистики.")
    else:
        # Карточки с основными метриками
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего постов в датасете", f"{stats_data['total_count']:,}".replace(",", " "))
        with col2:
            st.metric("Положительных постов (pos.csv)", f"{stats_data['pos_count']:,}".replace(",", " "), f"{stats_data['share_pos']}%")
        with col3:
            st.metric("Отрицательных постов (neg.csv)", f"{stats_data['neg_count']:,}".replace(",", " "), f"-{stats_data['share_neg']}%")
            
        st.markdown("---")
        
        # Длина постов
        st.subheader("Средняя длина сообщений в твиттере")
        col_len1, col_len2 = st.columns(2)
        with col_len1:
            st.info(f"**Положительные посты:**\n- Средняя длина в символах: **{stats_data['pos_mean_len_chars']}**\n- Среднее количество слов: **{stats_data['pos_mean_len_words']}**")
        with col_len2:
            st.info(f"**Отрицательные посты:**\n- Средняя длина в символах: **{stats_data['neg_mean_len_chars']}**\n- Среднее количество слов: **{stats_data['neg_mean_len_words']}**")
            
        st.markdown("---")
        
        # Частотный анализ слов (визуализация)
        st.subheader("Часто встречающиеся слова (топ-15 без стоп-слов)")
        
        import pandas as pd
        
        col_words1, col_words2 = st.columns(2)
        
        with col_words1:
            st.write("**😊 Положительные слова:**")
            pos_words = stats_data["pos_top_words"][:15]
            df_pos = pd.DataFrame(pos_words, columns=["Слово", "Частота"])
            # Строим красивый горизонтальный график
            st.bar_chart(data=df_pos, x="Слово", y="Частота", color="#2ca02c")
            
        with col_words2:
            st.write("**😔 Отрицательные слова:**")
            neg_words = stats_data["neg_top_words"][:15]
            df_neg = pd.DataFrame(neg_words, columns=["Слово", "Частота"])
            st.bar_chart(data=df_neg, x="Слово", y="Частота", color="#d62728")

# ==================== ВКЛАДКА 3: СПРАВКА ПО КОМАНДАМ ====================
with tab_help:
    st.header("Справка по командам и параметрам")
    
    st.subheader("🛠️ Как запустить приложение и бэкенд")
    st.markdown("""
    Для полноценной работы системы требуется запустить два компонента:
    1. **Запуск API сервера (FastAPI)**:
       Откройте терминал в папке проекта и выполните команду:
       ```bash
       uvicorn main:app --reload --port 8000
       ```
       *Параметры:*
       * `--reload` — автоматически перезапускает сервер при изменении кода (удобно для разработки).
       * `--port 8000` — указывает порт, на котором будет доступно наше API.
       
    2. **Запуск графического интерфейса (Streamlit)**:
       Откройте второй терминал в папке проекта и запустите команду:
       ```bash
       streamlit run app_streamlit.py
       ```
       Приложение автоматически откроется в вашем браузере по адресу `http://localhost:8501`.
    """)
    
    st.markdown("---")
    
    st.subheader("🔌 Описание API (эндпоинты и параметры)")
    st.markdown("""
    Наше FastAPI API предоставляет следующие возможности:
    
    * **`GET /`** — Проверка статуса сервера. Возвращает информацию о версии и типе загруженной модели.
    * **`POST /predict`** — Основной метод для классификации текста.
      * **Входные параметры (JSON):**
        * `text` (строка, обязательно) — текст комментария или твита для анализа.
      * **Возвращаемые параметры (JSON):**
        * `text` (строка) — исходный переданный текст.
        * `clean_text` (строка) — текст после очистки, удаления ссылок, стоп-слов и приведения слов к нормальной форме (лемме).
        * `sentiment` (строка) — текстовый вердикт тональности: `"Положительный"` или `"Отрицательный"`.
        * `sentiment_code` (число) — код тональности: `1` для положительного и `0` для отрицательного.
        * `probability` (число) — вероятность положительного класса (от `0.0` до `1.0`). Значения выше `0.5` классифицируются как положительные, ниже `0.5` — как отрицательные.
        
    Интерактивную документацию Swagger можно открыть по адресу: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
    """)
