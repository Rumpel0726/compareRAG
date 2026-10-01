# -*- coding: utf-8 -*-
"""
Тестовые вопросы для оценки RAG на документации PostgreSQL.

Сгенерированы автоматически через scripts/generate_postgresql_questions.py.
Документ: data/PostrgreSQL/postgres_part1_2.pdf
20 лёгких вопросов (1 страница) + 10 средних (2-3 страницы).
Номера страниц гарантированно корректны: модель видела реальные [Страница N].
"""

from src.evaluation.questions import EvalQuestion


EVAL_QUESTIONS_POSTGRESQL: list[EvalQuestion] = [

    # 1. Лёгкий — Внешние ключи
    EvalQuestion(
        question="Как в PostgreSQL создать внешний ключ, который связывает таблицу weather с таблицей cities?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [14],
        }
    ),

    # 2. Лёгкий — Правила сортировки
    EvalQuestion(
        question="Как в PostgreSQL переопределить правило сортировки для выражения?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [40],
        }
    ),

    # 3. Лёгкий — Изменение владельца
    EvalQuestion(
        question="Какая команда используется для изменения владельца таблицы в PostgreSQL?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [66],
        }
    ),

    # 4. Лёгкий — Создание правил
    EvalQuestion(
        question="Какой командой можно создать правило для автоматического направления новых строк в дочернюю таблицу measurement_y2006m02?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [91, 92],
        }
    ),

    # 5. Лёгкий — Группировка с DISTINCT
    EvalQuestion(
        question="Как избежать дублирующихся наборов группировок в запросе с ROLLUP?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [118],
        }
    ),

    # 6. Лёгкий — Тип данных bytea
    EvalQuestion(
        question="Какой байт при выводе дублируется в режиме escape для типа данных bytea?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [144],
        }
    ),

    # 7. Лёгкий — оператор вхождения jsonb
    EvalQuestion(
        question="Какой оператор в PostgreSQL используется для проверки, содержит ли один документ jsonb другой документ?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [170],
        }
    ),

    # 8. Лёгкий — Ограничения-исключения
    EvalQuestion(
        question="Как создать ограничение-исключение для предотвращения пересечения диапазонов в таблице reservation?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [196],
        }
    ),

    # 9. Лёгкий — Регулярные выражения
    EvalQuestion(
        question="Какая функция в PostgreSQL позволяет извлекать подстроку, соответствующую регулярному выражению, и возвращать N-е вхождение?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [222, 240],
        }
    ),

    # 10. Лёгкий — Жадность регулярных выражений
    EvalQuestion(
        question="Как определяется жадность регулярного выражения, образованного из двух или более ветвей, соединённых оператором |?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [248],
        }
    ),

    # 11. Лёгкий — Функции часовых поясов
    EvalQuestion(
        question="Какая функция в PostgreSQL эквивалентна SQL-совместимой конструкции 'время AT TIME ZONE часовой_пояс'?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [274],
        }
    ),

    # 12. Лёгкий — Параметр пространств имён
    EvalQuestion(
        question="Какой параметр функций xpath и xpath_exists в PostgreSQL используется для определения пространств имён?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [300],
        }
    ),

    # 13. Лёгкий — Преобразование в целое число
    EvalQuestion(
        question="Какая функция в PostgreSQL преобразует строковое значение JSON в целое число?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [326],
        }
    ),

    # 14. Лёгкий — агрегатные функции
    EvalQuestion(
        question="Какая функция в PostgreSQL является стандартным SQL аналогом bool_and?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [352, 354],
        }
    ),

    # 15. Лёгкий — Функции создания объектов
    EvalQuestion(
        question="Какая функция в PostgreSQL используется для восстановления команды создания функции или процедуры?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [378],
        }
    ),

    # 16. Лёгкий — Статистика таблиц
    EvalQuestion(
        question="Какая функция в PostgreSQL используется для обновления статистики на уровне таблиц, особенно после восстановления?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [404, 405],
        }
    ),

    # 17. Лёгкий — Типы индексов
    EvalQuestion(
        question="Какой тип индекса создается по умолчанию при использовании команды CREATE INDEX в PostgreSQL?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [430],
        }
    ),

    # 18. Лёгкий — Функция ts_headline
    EvalQuestion(
        question="Какая функция PostgreSQL используется для выделения фрагментов документа с выделенными словами из запроса при полнотекстовом поиске?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [456],
        }
    ),

    # 19. Лёгкий — Шаблоны текстового поиска
    EvalQuestion(
        question="Какая команда используется для просмотра шаблонов текстового поиска в PostgreSQL?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [482],
        }
    ),

    # 20. Лёгкий — EXPLAIN ANALYZE
    EvalQuestion(
        question="Какую дополнительную информацию выводит команда EXPLAIN ANALYZE для узлов Index Scan в PostgreSQL?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [508],
        }
    ),

    # 21. Средний — Приведение типов и оконные функции
    EvalQuestion(
        question="Как приведение типов данных влияет на использование интервалов в рамках оконных функций PostgreSQL?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [39, 40],
        }
    ),

    # 22. Средний — Секционирование наследование
    EvalQuestion(
        question="Какие методы используются для автоматического направления вставляемых данных в дочерние таблицы при секционировании через наследование, и какие особенности у каждого подхода?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [92, 93, 94],
        }
    ),

    # 23. Средний — Точность типов данных
    EvalQuestion(
        question="Как в PostgreSQL задается точность (p) для типов данных времени и даты, и как это влияет на ввод и хранение значений?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [145, 146, 147],
        }
    ),

    # 24. Средний — Псевдотипы и OID-псевдонимы
    EvalQuestion(
        question="Как псевдотипы и типы-псевдонимы OID взаимодействуют в функциях PostgreSQL, и как это влияет на зависимости объектов?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [198, 199, 200],
        }
    ),

    # 25. Средний — Функции форматирования
    EvalQuestion(
        question="Какие функции форматирования данных и коды форматирования даты/времени доступны в PostgreSQL?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [252, 253],
        }
    ),

    # 26. Средний — XML-преобразования и схемы
    EvalQuestion(
        question="Как параметры tableforest, nulls и targetns влияют на формат XML-документов, генерируемых функциями PostgreSQL, и как эти документы могут быть использованы в сочетании с XML-схемами и XSLT-преобразованиями?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [304, 305, 306],
        }
    ),

    # 27. Средний — Ранжирование и процентили
    EvalQuestion(
        question="Какие функции в PostgreSQL позволяют вычислять ранги и процентили в контексте агрегации и оконных функций, и как они различаются по применению?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [357, 358, 359],
        }
    ),

    # 28. Средний — Блокировки и триггеры
    EvalQuestion(
        question="Какие встроенные функции PostgreSQL позволяют управлять рекомендательными блокировками и предотвращать ненужные обновления строк, и как они применяются?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [410, 411, 412],
        }
    ),

    # 29. Средний — Анализаторы и словари
    EvalQuestion(
        question="Как стандартный анализатор PostgreSQL обрабатывает составные слова с дефисами и как это взаимодействует с конфигурацией словарей для полнотекстового поиска?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [463, 464, 465],
        }
    ),

    # 30. Средний — Расширенная статистика
    EvalQuestion(
        question="Какие типы расширенной статистики доступны в PostgreSQL и для каких целей они применяются?",
        expected_sources=["postgres_part1_2.pdf"],
        expected_pages={
            "postgres_part1_2.pdf": [516, 517, 518],
        }
    ),

]
