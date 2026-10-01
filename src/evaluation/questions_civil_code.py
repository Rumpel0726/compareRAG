# -*- coding: utf-8 -*-
"""
Тестовые вопросы для оценки RAG на Гражданском кодексе РФ.

Сгенерированы автоматически через scripts/generate_civil_code_questions.py.
Документ: data/Civil_Code/gkodeksrf.pdf
20 лёгких вопросов (1 страница) + 10 средних (2-3 страницы).
Номера страниц соответствуют pypdf нумерации (1-indexed).
"""

from src.evaluation.questions import EvalQuestion


EVAL_QUESTIONS_CIVIL_CODE: list[EvalQuestion] = [

    # Лёгкий — вопрос 1 (стр. 17)
    EvalQuestion(
        question="Какие виды сделок с имуществом подопечного не может совершать опекун без предварительного разрешения органа опеки и попечительства?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [17],
        }
    ),

    # Лёгкий — вопрос 2 (стр. 50)
    EvalQuestion(
        question="Какие сведения должен содержать учредительный договор полного товарищества согласно статье 70 Гражданского кодекса РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [50],
        }
    ),

    # Лёгкий — вопрос 3 (стр. 83)
    EvalQuestion(
        question="Какие лица могут быть учредителями учреждения согласно Гражданскому кодексу РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [83],
        }
    ),

    # Лёгкий — вопрос 4 (стр. 116)
    EvalQuestion(
        question="Какой минимальный процент участников гражданско-правового сообщества должен отправить документы для принятия решения без проведения заседания?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [116],
        }
    ),

    # Лёгкий — вопрос 5 (стр. 149)
    EvalQuestion(
        question="Можно ли продать земельный участок, находящийся в пожизненном наследуемом владении?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [149],
        }
    ),

    # Лёгкий — вопрос 6 (стр. 182)
    EvalQuestion(
        question="Какие имущества автоматически считаются находящимися в залоге согласно статье 345 ГК РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [182],
        }
    ),

    # Лёгкий — вопрос 7 (стр. 215)
    EvalQuestion(
        question="Как определяется текущая цена при возмещении убытков при досрочном расторжении договора?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [215],
        }
    ),

    # Лёгкий — вопрос 8 (стр. 248)
    EvalQuestion(
        question="Какие действия может предпринять покупатель, если товар передан без тары или в ненадлежащей упаковке?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [248],
        }
    ),

    # Лёгкий — вопрос 9 (стр. 281)
    EvalQuestion(
        question="Какой срок предусмотрен для уведомления плательщика ренты о выкупе постоянной ренты?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [281],
        }
    ),

    # Лёгкий — вопрос 10 (стр. 314)
    EvalQuestion(
        question="При каких условиях подрядчик может продать результат работы, если заказчик уклоняется от его принятия?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [314],
        }
    ),

    # Лёгкий — вопрос 11 (стр. 347)
    EvalQuestion(
        question="Как определяется период начисления процентов на сумму банковского вклада?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [347],
        }
    ),

    # Лёгкий — вопрос 12 (стр. 380)
    EvalQuestion(
        question="Какой срок хранения невостребованной вещи устанавливает ломбард до её продажи?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [380],
        }
    ),

    # Лёгкий — вопрос 13 (стр. 413)
    EvalQuestion(
        question="Какова ответственность доверительного управляющего за действия переданного им права управления другому лицу?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [413],
        }
    ),

    # Лёгкий — вопрос 14 (стр. 446)
    EvalQuestion(
        question="Как определяются доли наследников, если в завещании не указаны их части имущества?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [446],
        }
    ),

    # Лёгкий — вопрос 15 (стр. 479)
    EvalQuestion(
        question="Как определяется личный закон юридического лица согласно статье 1202 ГК РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [479],
        }
    ),

    # Лёгкий — вопрос 16 (стр. 512)
    EvalQuestion(
        question="Кто может быть зарегистрирован в качестве патентного поверенного в Российской Федерации?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [512],
        }
    ),

    # Лёгкий — вопрос 17 (стр. 545)
    EvalQuestion(
        question="Какие элементы включает знак правовой охраны смежных прав согласно статье 1305 ГК РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [545],
        }
    ),

    # Лёгкий — вопрос 18 (стр. 578)
    EvalQuestion(
        question="На каком языке должно подаваться заявление о выдаче патента на изобретение, полезную модель или промышленный образец, и какие требования к другим документам заявки?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [578],
        }
    ),

    # Лёгкий — вопрос 19 (стр. 611)
    EvalQuestion(
        question="Какой срок установлен для использования приоритета первой заявки на патент на селекционное достижение, поданной в иностранном государстве, согласно Гражданскому кодексу РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [611],
        }
    ),

    # Лёгкий — вопрос 20 (стр. 644)
    EvalQuestion(
        question="В течение какого срока можно оспорить регистрацию товарного знака, если правовая охрана предоставлена с нарушением требований пунктов 6, 7 и 10 статьи 1483 ГК РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [644],
        }
    ),

    # Средний — вопрос 21 (стр. 49-51)
    EvalQuestion(
        question="Каковы особенности субсидиарной ответственности участников при преобразовании полного товарищества в общество и в случае обязательств полного товарищества?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [49, 50, 51],
        }
    ),

    # Средний — вопрос 22 (стр. 115-117)
    EvalQuestion(
        question="Каковы сроки исковой давности для требований о признании недействительными ничтожных и оспоримых сделок, а также решений собраний, и как определяется начало их течения согласно Гражданскому кодексу РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [115, 116, 117],
        }
    ),

    # Средний — вопрос 23 (стр. 181-183)
    EvalQuestion(
        question="Какие последствия замены предмета залога предусмотрены законом, и как это влияет на обязанности сторон по страхованию и сохранности имущества?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [181, 182, 183],
        }
    ),

    # Средний — вопрос 24 (стр. 247-249)
    EvalQuestion(
        question="Какие последствия наступают, если продавец передал товар без необходимой комплектации, и какие действия должен предпринять покупатель для защиты своих прав?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [247, 248, 249],
        }
    ),

    # Средний — вопрос 25 (стр. 313-315)
    EvalQuestion(
        question="Какие последствия наступают, если заказчик уклоняется от приемки выполненной работы по договору подряда, и какие права имеет подрядчик в этой ситуации?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [313, 314, 315],
        }
    ),

    # Средний — вопрос 26 (стр. 379-381)
    EvalQuestion(
        question="Какие обязательные реквизиты должны содержать двойное и простое складские свидетельства, и в чем различия их правового режима согласно статьям 913 и 917 ГК РФ?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [379, 380, 381],
        }
    ),

    # Средний — вопрос 27 (стр. 445-447)
    EvalQuestion(
        question="Какие последствия наступают при разглашении тайны совместного завещания супругов до открытия наследства, и как это влияет на обязательства супругов по завещанию?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [445, 446, 447],
        }
    ),

    # Средний — вопрос 28 (стр. 511-513)
    EvalQuestion(
        question="Какие органы уполномочены рассматривать споры, связанные с защитой интеллектуальных прав в административном порядке, и кто устанавливает правила их рассмотрения?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [511, 512, 513],
        }
    ),

    # Средний — вопрос 29 (стр. 577-579)
    EvalQuestion(
        question="Каковы последствия непользования патентом на изобретение, выданным по государственному контракту, в течение двух лет для Российской Федерации и исполнителя, а также как регулируется передача исключительного права в этом случае?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [577, 578, 579],
        }
    ),

    # Средний — вопрос 30 (стр. 643-645)
    EvalQuestion(
        question="Каковы основания досрочного прекращения правовой охраны коллективного знака и как они связаны с оспариванием регистрации товарных знаков?",
        expected_sources=["gkodeksrf.pdf"],
        expected_pages={
            "gkodeksrf.pdf": [643, 644, 645],
        }
    ),

]
