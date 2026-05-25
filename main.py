import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from text_utils import clean_text

# Определение путей к файлам модели и векторизатора
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# Проверяем наличие файлов перед запуском
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден по пути: {MODEL_PATH}")
if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Файл векторизатора не найден по пути: {VECTORIZER_PATH}")

# Загрузка векторизатора и модели
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Инициализация FastAPI приложения
app = FastAPI(
    title="API Анализа Тональности Русскоязычных Постов",
    description="Это API использует обученную модель логистической регрессии и TF-IDF векторизатор для определения тональности текста (позитивный или негативный).",
    version="1.0.0"
)

# Модели Pydantic для валидации входных и выходных данных
class TextRequest(BaseModel):
    text: str = Field(..., example="Я так люблю эту жизнь, сегодня прекрасный день!")

class SentimentResponse(BaseModel):
    text: str = Field(..., description="Исходный текст")
    clean_text: str = Field(..., description="Текст после очистки и лемматизации")
    sentiment: str = Field(..., description="Тональность текста ('Положительный' или 'Отрицательный')")
    sentiment_code: int = Field(..., description="Код тональности (1 - положительный, 0 - отрицательный)")
    probability: float = Field(..., description="Вероятность положительного класса (от 0 до 1)")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Добро пожаловать в API анализа тональности! Перейдите на /docs для интерактивного тестирования.",
        "model_type": "Logistic Regression",
        "features_limit": 5000
    }

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: TextRequest):
    raw_text = request.text
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Текст запроса не должен быть пустым.")
    
    # 1. Предобработка текста
    cleaned = clean_text(raw_text)
    
    # Если после очистки ничего не осталось (например, были только цифры или смайлики)
    if not cleaned.strip():
        cleaned = "пустой_текст" # Заглушка, чтобы векторизатор не выдал ошибку на пустой строке
    
    # 2. Векторизация текста
    text_vector = vectorizer.transform([cleaned])
    
    # 3. Предсказание тональности и получение вероятностей
    prob_positive = float(model.predict_proba(text_vector)[0][1])
    prediction = int(model.predict(text_vector)[0])
    
    # 4. Формирование ответа
    sentiment_label = "Положительный" if prediction == 1 else "Отрицательный"
    
    return SentimentResponse(
        text=raw_text,
        clean_text=cleaned,
        sentiment=sentiment_label,
        sentiment_code=prediction,
        probability=prob_positive
    )
