import re
import nltk
import pymorphy3
from nltk.corpus import stopwords

# Скачиваем стоп-слова, если они еще не скачаны
try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')

morph = pymorphy3.MorphAnalyzer(lang='ru')
stop_words = set(stopwords.words('russian'))

# Дополнительные стоп-слова и слова-паразиты из ноутбука
extra_stopwords = {
    'это', 'весь', 'который', 'свой', 'ещё',
    'очень', 'сегодня', 'завтра', 'просто',
    'вообще', 'почему', 'твой', 'мой',
    'наш', 'ваш', 'самый', 'сам',
    'мочь', 'хотеть', 'знать',
    'сказать', 'сделать', 'всё',
    'пока', 'давать', 'кстати', 'rt',
    'день', 'год', 'человек', 'время',
    'идти', 'делать', 'думать',
    'пойти', 'писать', 'смотреть',
    'первый', 'новый', 'говорить',
    'ждать', 'дом', 'жизнь', 'час',
    
    # Twitter-специфичные слова и другие слова-паразиты
    'твит', 'твиттер', 'прям', 'таки', 'тип', 
    'типа', 'ага', 'уж', 'фотка', 'фото', 'видео',
    'эх', 'эхх', 'ээх', 'азаза', 'азазаза', 'охохо', 
    'хехе', 'хихи', 'хохо', 'хаха',
    
    # Дополнительный разговорный мусор и сокращения
    'блин', 'кек', 'лол', 'оч', 'ща', 'щас', 'хз', 
    'мда', 'спс', 'плз', 'плиз', 'аккаунт', 'блог'
}

stop_words.update(extra_stopwords)

def clean_text(text):
    text = str(text).lower()
    # Удаление ссылок
    text = re.sub(r'http\S+', ' ', text)
    # Удаление упоминаний пользователей (@username)
    text = re.sub(r'@\w+', ' ', text)
    # Удаление хэштегов
    text = re.sub(r'#\w+', ' ', text)
    # Сокращение повторяющихся букв (3 и более подряд -> 1 буква)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Удаление цифр
    text = re.sub(r'\d+', ' ', text)
    # Удаление всех символов кроме русских букв и пробелов
    text = re.sub(r'[^а-яё\s]', ' ', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    cleaned_words = []
    
    for word in words:
        if len(word) < 3:
            continue
            
        # Удаляем смех (ахаха, азаза, хехе и т.д.) через проверку множества букв
        word_letters = set(word)
        if word_letters <= {'а', 'х'} or word_letters <= {'а', 'з'} or word_letters <= {'х', 'е'} or word_letters <= {'х', 'и'} or word_letters <= {'х', 'о'} or word_letters <= {'ы'}:
            continue
            
        if word in stop_words:
            continue
            
        lemma = morph.parse(word)[0].normal_form
        
        # Снова проверяем лемму после нормализации
        lemma_letters = set(lemma)
        if lemma_letters <= {'а', 'х'} or lemma_letters <= {'а', 'з'} or lemma_letters <= {'х', 'е'} or lemma_letters <= {'х', 'и'} or lemma_letters <= {'х', 'о'} or lemma_letters <= {'ы'}:
            continue
            
        if lemma not in stop_words and len(lemma) >= 3:
            cleaned_words.append(lemma)
            
    return ' '.join(cleaned_words)
