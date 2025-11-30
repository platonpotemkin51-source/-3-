import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import ast


# Функция загрузки датасета
def load_embeddings_to_dataframe(filepath):
    """
    Загружает данные эмбеддингов из CSV файла в pandas DataFrame

    Args:
        filepath (str): Путь к файлу с данными

    Returns:
        pd.DataFrame: DataFrame с загруженными данными
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Загружено {len(df)} записей из файла {filepath}")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"\nПервые 3 записи:")
        print(df.head(3))
        return df
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return None


# Загрузка данных
df = load_embeddings_to_dataframe("arxiv_embeddings202505211515.csv")

# Просмотр информации о датасете
print(f"Размер датасета: {df.shape}")
print(f"\nТипы данных:\n{df.dtypes}")
print(f"\nПроверка пропущенных значений:\n{df.isnull().sum()}")


def prepare_documents(df, limit=1000):
    """
    Преобразует DataFrame в список документов LangChain

    Args:
        df: DataFrame с данными статей
        limit: Количество документов для обработки

    Returns:
        List[Document]: Список документов
    """
    documents = []

    # Возьмите первые limit записей
    df_subset = df.head(limit)

    for idx, row in df_subset.iterrows():
        # Формируйте текст документа из доступных полей
        page_content = ''

        # Адаптируйте под структуру вашего датасета
        if 'title' in row:
            page_content += f"Название: {row['title']}\n"
        if 'abstract' in row:
            page_content += f"Аннотация: {row['abstract']}\n"
        if 'authors' in row:
            page_content += f"Авторы: {row['authors']}\n"

        # Метаданные
        metadata = {}
        if 'categories' in row:
            metadata['categories'] = row['categories']
        if 'main_category' in row:
            metadata['main_category'] = row['main_category']
        if 'year' in row:
            metadata['year'] = row['year']
        if 'article_id' in row:
            metadata['article_id'] = row['article_id']

        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))

    print(f"Подготовлено {len(documents)} документов")
    return documents


# Подготовка документов
documents = prepare_documents(df)

# Просмотр примера документа
print("Пример документа:")
print(f"Содержимое: {documents[0].page_content[:200]}...")
print(f"Метаданные: {documents[0].metadata}")

# Конфигурация OpenAI клиента для получения эмбеддингов
client = OpenAI(
    api_key="ZmJhMjUwZTItMDg0ZC00N2E3LWIyNDktYjA4MTQyZGFmMGE4.97f6d089a16317c3aa93b365eda739a8",
    base_url="https://foundation-models.api.cloud.ru/v1"
)


def get_embedding(text: str, model="BAAI/bge-m3") -> list:
    """Получает эмбеддинг текста"""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


# Тестирование функции эмбеддингов
test_embedding = get_embedding("Тестовый запрос")
print(f"Размерность эмбеддинга: {len(test_embedding)}")

from langchain_core.embeddings import Embeddings
from typing import List


class CustomEmbeddings(Embeddings):
    """Кастомный класс эмбеддингов для работы с API"""

    def __init__(self, client, model="BAAI/bge-m3"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для списка документов"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Получение эмбеддинга для запроса"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding


# Создание экземпляра эмбеддингов
embeddings = CustomEmbeddings(client)

# Создание векторного хранилища ChromaDB
# print("Создание векторного хранилища...")
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     collection_name="arxiv_papers",
#     persist_directory="./chroma_db"  # Директория для сохранения
# )
# print("Векторное хранилище успешно создано!")


# Проверка работы хранилища
# test_query = "машинное обучение и нейронные сети"
# results = vectorstore.similarity_search(test_query, k=3)
# print(f"\nРезультаты поиска по запросу '{test_query}':")
# for i, doc in enumerate(results, 1):
#     print(f"\n{i}. {doc.page_content[:200]}...")
#     print(f"   Метаданные: {doc.metadata}")

# Задание 1
# 1. Вывод статистики по категориям статей
# print("Статистика по категориям статей:")
# category_stats = df['categories'].value_counts()
# print(category_stats)

# 2. Выбор подмножества из 1000 статей
# subset_df = df.sample(n=1000, random_state=33)
# print(f"Размер подмножества: {subset_df.shape}")

# Задание 2
# 1. Разные значения параметра k
# print("1. Разные значения k:")
# for k in [1, 3, 5, 10]:
#     results = vectorstore.similarity_search(test_query, k=k)
#     print(f"k={k}: найдено {len(results)} документов")

# 2. Поиск по нескольким тематическим запросам
# print("2. Поиск по нескольким тематическим запросам:")
# queries = [
#     "глубокое обучение",
#     "обработка естественного языка",
#     "компьютерное зрение",
#     "нейронные сети"
# ]
#
# for query in queries:
#     results = vectorstore.similarity_search(query, k=2)
#     print(f"Запрос: '{query}'")
#     print(f"Найдено документов: {len(results)}")
#     for i, doc in enumerate(results, 1):
#         print(f"  {i}. {doc.page_content[:100]}...")

# 3. Метод similarity_search_with_score()
# print("3. Поиск с оценками схожести:")
# results_with_scores = vectorstore.similarity_search_with_score(test_query, k=3)
# print(f"Запрос: '{test_query}'")
# for i, (doc, score) in enumerate(results_with_scores, 1):
#     print(f"\n{i}. Оценка схожести: {score:.4f}")
#     print(f"   Содержимое: {doc.page_content[:150]}...")
#     print(f"   Метаданные: {doc.metadata}")

print("Загрузка векторного хранилища...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="arxiv_papers"
)
print("Векторное хранилище загружено!")

# Задание 3
# Создание ретривера с поиском по схожести
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Возвращать топ-5 документов
)

# Тестирование ретривера
query = "глубокое обучение для обработки изображений"
retrieved_docs = retriever.invoke(query)

print(f"Найдено документов: {len(retrieved_docs)}")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nДокумент {i}:")
    print(doc.page_content[:150] + "...")

# MMR балансирует между релевантностью и разнообразием результатов
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Количество документов для первичной выборки
        "lambda_mult": 0.5  # Баланс между релевантностью (1.0) и разнообразием (0.0)
    }
)

# Сравнение результатов
query = "обработка естественного языка"
docs_similarity = retriever.invoke(query)
docs_mmr = retriever_mmr.invoke(query)

print("Сравнение результатов поиска:")
print("\n=== Similarity Search ===")
for i, doc in enumerate(docs_similarity[:3], 1):
    print(f"{i}. {doc.page_content[:100]}...")

print("\n=== MMR Search ===")
for i, doc in enumerate(docs_mmr[:3], 1):
    print(f"{i}. {doc.page_content[:100]}...")

# Ретривер с фильтрацией по оценке схожести
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # Минимальная оценка схожести
        "k": 10
    }
)

# ВАЖНО: Не все векторные хранилища поддерживают score_threshold
# В случае ChromaDB может потребоваться другой подход

# 1. Поэкспериментируйте с параметром lambda_mult в MMR
print("1. Эксперименты с lambda_mult в MMR:")

for lambda_val in [0.0, 0.5, 1.0]:
    retriever_mmr_test = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 50,
            "lambda_mult": lambda_val
        }
    )
    docs = retriever_mmr_test.invoke("нейронные сети")
    print(f"lambda_mult={lambda_val}: найдено {len(docs)} документов")
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        metadata = doc.metadata

        # Вытащим заголовок
        title = "Не найден"
        if "Название:" in content:
            title = content.split("Название:")[1].split("\n")[0].strip()

        # Вытащим категорию
        category = metadata.get('categories', 'Нет категории')

        print(f"   Документ {i}:")
        print(f"      Заголовок: {title[:80]}...")
        print(f"      Категория: {category}")

# 2. Создайте функцию для сравнения времени работы разных типов поиска
print("\n2. Сравнение времени работы:")

import time


def compare_search_times(query, vectorstore):
    search_types = {
        "similarity": {"search_type": "similarity", "search_kwargs": {"k": 5}},
        "mmr": {"search_type": "mmr", "search_kwargs": {"k": 5, "fetch_k": 20, "lambda_mult": 0.5}},
    }

    for name, config in search_types.items():
        retriever = vectorstore.as_retriever(**config)
        start_time = time.time()
        docs = retriever.invoke(query)
        end_time = time.time()
        print(f"{name}: {len(docs)} документов за {end_time - start_time:.3f} сек")


compare_search_times("машинное обучение", vectorstore)

# Пример конфигурации (используйте свои данные)

llm = GigaChat(
    credentials="OThhZGViNTgtN2E0Mi00YmExLTgzMTctM2YwNjFmNGI0NzNkOmM2YzYzMGJlLTczMGQtNDk3MC04MjRlLWQwZjBkZWRkM2U5Mg==",
    scope="GIGACHAT_API_B2B",
    model="GigaChat-Pro",
    verify_ssl_certs=False,
    timeout=30
)

# Тест языковой модели
test_response = llm.invoke("Привет! Ответь кратко: что такое машинное обучение?")
print(f"Ответ модели: {test_response.content}")

# Шаблон промпта для RAG
prompt_template = """Ты -- научный ассистент, специализирующийся на анализе научных статей.
Твоя задача -- отвечать на вопросы пользователя, основываясь ТОЛЬКО на предоставленном контексте из научных статей ArXiv.

Правила:
1. Используй только информацию из контекста ниже
2. Если в контексте нет информации для ответа, честно скажи об этом
3. Указывай, из каких статей взята информация (если есть метаданные)
4. Отвечай на русском языке, четко и структурированно
5. Если вопрос касается технических деталей, будь точным

Контекст из научных статей:
{context}

Вопрос пользователя: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(prompt_template)


def format_docs(docs):
    """
    Форматирует список документов в единую строку контекста

    Args:
        docs: Список документов Document

    Returns:
        str: Форматированный контекст
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Документ {i}]")
        context_parts.append(doc.page_content)
        if doc.metadata:
            context_parts.append(f"Метаданные: {doc.metadata}")
        context_parts.append("")  # Пустая строка для разделения
    return "\n".join(context_parts)


# Тест форматирования
test_docs = retriever.invoke("нейронные сети")
formatted_context = format_docs(test_docs[:2])
print("Пример форматированного контекста:")
print(formatted_context[:500] + "...")
# Создание RAG-цепочки
rag_chain = (
    {
        "context": retriever | format_docs,  # Извлекаем и форматируем документы
        "question": RunnablePassthrough()     # Передаем вопрос как есть
    }
    | prompt      # Формируем промпт
    | llm         # Отправляем в языковую модель
)

# Тестирование RAG-системы
questions = [
    "Какие методы машинного обучения используются для обработки изображений?",
    "Расскажи о применении трансформеров в обработке естественного языка",
    "Какие существуют подходы к обучению нейронных сетей?"
]

print("=== Тестирование RAG-системы ===\n")
for i, question in enumerate(questions, 1):
    print(f"Вопрос {i}: {question}")
    print("-" * 80)

    try:
        response = rag_chain.invoke(question)
        print(f"Ответ: {response.content}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")


def interactive_rag_qa():
    """Интерактивная система вопросов-ответов"""
    print("=== Интерактивная RAG-система для научных статей ArXiv ===")
    print("Введите 'выход' для завершения\n")

    while True:
        question = input("Ваш вопрос: ").strip()

        if question.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break

        if not question:
            continue

        try:
            # Получаем релевантные документы
            docs = retriever.invoke(question)
            print(f"\n📚 Найдено релевантных документов: {len(docs)}")

            # Генерируем ответ
            response = rag_chain.invoke(question)
            print(f"\n🤖 Ответ:\n{response.content}\n")

            # Показываем источники
            show_sources = input("Показать источники? (да/нет): ").strip().lower()
            if show_sources in ['да', 'yes', 'y', 'д']:
                print("\n📖 Источники:")
                for i, doc in enumerate(docs[:3], 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
                    print(f"   Метаданные: {doc.metadata}")

            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f"❌ Ошибка: {e}\n")


print("Шаг 4.6")

# Всего 2 дополнительных вопроса вместо 5
diverse_questions = [
    "Что такое квантовые вычисления?",
    "Какие методы используются в компьютерном зрении?"
]

print("Тестирование на разных тематиках:\n")
for i, question in enumerate(diverse_questions, 1):
    print(f"Вопрос {i}: {question}")
    print("-" * 60)

    try:
        response = rag_chain.invoke(question)
        print(f"Ответ: {response.content}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")

# Один эксперимент с разным k
print("Эксперимент с разным количеством документов:")
for k in [3, 6]:  # всего 2 значения вместо 3
    custom_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    custom_rag_chain = (
            {"context": custom_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm
    )

    response = custom_rag_chain.invoke("нейронные сети")
    print(f"k={k}: ответ {len(response.content)} символов")

# Запуск интерактивной системы (раскомментируйте для использования)
interactive_rag_qa()
