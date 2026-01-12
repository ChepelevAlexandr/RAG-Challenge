# RAG-Challenge — Retrieval-Augmented Generation System

Данный репозиторий содержит решение домашнего задания **RAG-Challenge**,
в рамках которого реализована Retrieval-Augmented Generation (RAG) система
для работы с реальными годовыми отчётами компаний в формате PDF.

## Описание задачи

Цель задания — разработать RAG-систему, которая:

1. Извлекает текст из PDF-документов (годовые отчёты компаний).
2. Индексирует полученный текст для последующего поиска.
3. Находит релевантные фрагменты под каждый вопрос.
4. Генерирует ответы **исключительно на основе найденного контекста**.
5. Формирует submission-файл в заданном формате без ручных правок.

Система реализована в соответствии с требованиями задания:
- ответы не генерируются напрямую без retrieval,
- индексы и сабмишны не хранятся в репозитории,
- используется автоматическая генерация submission-файла.

## Архитектура решения

Решение построено как классический RAG-пайплайн:

1. **PDF parsing**  
   Извлечение текста из PDF с использованием библиотеки **PyMuPDF**.

2. **Индексация и поиск**  
   Реализован гибридный retrieval:
   - BM25
   - TF-IDF (word и char n-grams)

3. **Answer generation**  
   - rule-based извлечение чисел и логических фактов,
   - опционально используется GigaChat API,
   - все ответы сопровождаются ссылками на страницы источников.

4. **Submission generation**  
   Формирование JSON-файла в формате, требуемом системой проверки.

## Структура проекта
rag-challenge/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│ └── questions.json
├── src/
│ ├── pdf_index.py
│ ├── rag_answer.py
│ ├── submission_api.py
│ └── gigachat_client.py
└── tools/

## Установка и запуск

### 1. Создание виртуального окружения
```bash
python -m venv .venv

Активация:

Windows:

.venv\Scripts\activate


Linux / macOS:

source .venv/bin/activate

2. Установка зависимостей
pip install -r requirements.txt

3. Сборка индекса
python main.py build-index --data-dir ./data --out-dir ./index

4. Генерация submission-файла
python main.py make-submission \
  --index-dir ./index \
  --questions ./data/questions.json \
  --out ./submission_Chepelev_v2.json \
  --team-email your_email@example.com \
  --submission-name Chepelev_v2