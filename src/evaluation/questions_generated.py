# -*- coding: utf-8 -*-
"""
Вопросы сгенерированы автоматически через scripts/generate_questions.py.
Проверьте содержимое и скопируйте в src/evaluation/questions.py.
"""

from dataclasses import dataclass, field


@dataclass
class EvalQuestion:
    """Вопрос для оценки RAG системы."""
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_pages: dict[str, list[int]] = field(default_factory=dict)

    def has_page_annotations(self) -> bool:
        """Проверка наличия page-based аннотаций."""
        return bool(self.expected_pages)


EVAL_QUESTIONS: list[EvalQuestion] = [

    # Вопрос 1 — analiz-struktury-hirurgicheskih-vmeshatelstv-pri-pahovyh-gryzhah-v-usloviyah-vysokopotokovogo-gerniologicheskogo-tsentra.pdf
    EvalQuestion(
        question="Какой процент пациентов с паховыми грыжами в исследовании перенесли ТАРР герниопластику?",
        expected_sources=["analiz-struktury-hirurgicheskih-vmeshatelstv-pri-pahovyh-gryzhah-v-usloviyah-vysokopotokovogo-gerniologicheskogo-tsentra.pdf"],
        expected_pages={
            "analiz-struktury-hirurgicheskih-vmeshatelstv-pri-pahovyh-gryzhah-v-usloviyah-vysokopotokovogo-gerniologicheskogo-tsentra.pdf": [1, 3],
        }
    ),

    # Вопрос 2 — aplaziya-vlagalischa-i-matki-i-tazovaya-distopiya-pochki-taktika-vedeniya-i-vozmozhnosti-hirurgicheskoy-korrektsii-poroka-razvitiya-polovyh-organov.pdf
    EvalQuestion(
        question="Какая длина неовлагалища была у пациентки К. после операции?",
        expected_sources=["aplaziya-vlagalischa-i-matki-i-tazovaya-distopiya-pochki-taktika-vedeniya-i-vozmozhnosti-hirurgicheskoy-korrektsii-poroka-razvitiya-polovyh-organov.pdf"],
        expected_pages={
            "aplaziya-vlagalischa-i-matki-i-tazovaya-distopiya-pochki-taktika-vedeniya-i-vozmozhnosti-hirurgicheskoy-korrektsii-poroka-razvitiya-polovyh-organov.pdf": [3],
        }
    ),

    # Вопрос 3 — binokulyarnaya-implantatsiya-novoy-trifokalnoy-difraktsionnoy-intraokulyarnoy-linzy-dlya-korrektsii-presbiopii.pdf
    EvalQuestion(
        question="Какова частота достижения сфероэквивалента в диапазоне ± 0,5 Дптр после имплантации ИОЛ AT LISA tri 839MP?",
        expected_sources=["binokulyarnaya-implantatsiya-novoy-trifokalnoy-difraktsionnoy-intraokulyarnoy-linzy-dlya-korrektsii-presbiopii.pdf"],
        expected_pages={
            "binokulyarnaya-implantatsiya-novoy-trifokalnoy-difraktsionnoy-intraokulyarnoy-linzy-dlya-korrektsii-presbiopii.pdf": [1, 4],
        }
    ),

    # Вопрос 4 — bipolyarnoe-affektivnoe-rasstroystvo-ii-tipa.pdf
    EvalQuestion(
        question="Какова частота биполярного аффективного расстройства II типа в общей популяции согласно данным, приведённым в статье?",
        expected_sources=["bipolyarnoe-affektivnoe-rasstroystvo-ii-tipa.pdf"],
        expected_pages={
            "bipolyarnoe-affektivnoe-rasstroystvo-ii-tipa.pdf": [1],
        }
    ),

    # Вопрос 5 — chastichnaya-transplantatsiya-destsemetovoy-membrany-s-endoteliem-i-dmek.pdf
    EvalQuestion(
        question="Какова средняя острота зрения спустя 3 месяца после операции частичной DMEK?",
        expected_sources=["chastichnaya-transplantatsiya-destsemetovoy-membrany-s-endoteliem-i-dmek.pdf"],
        expected_pages={
            "chastichnaya-transplantatsiya-destsemetovoy-membrany-s-endoteliem-i-dmek.pdf": [1, 3],
        }
    ),

    # Вопрос 6 — diagnostika-i-profilaktika-zabolevaniy-parodonta-u-bolnyh-s-perelomami-nizhney-chelyusti.pdf
    EvalQuestion(
        question="Какова была экспрессия гена hBD-2 у пациентов с здоровым пародонтом до шинирования?",
        expected_sources=["diagnostika-i-profilaktika-zabolevaniy-parodonta-u-bolnyh-s-perelomami-nizhney-chelyusti.pdf"],
        expected_pages={
            "diagnostika-i-profilaktika-zabolevaniy-parodonta-u-bolnyh-s-perelomami-nizhney-chelyusti.pdf": [1, 2],
        }
    ),

    # Вопрос 7 — digoksin-u-bolnyh-s-hronicheskoy-serdechnoy-nedostatochnostyu-pro-i-contra.pdf
    EvalQuestion(
        question="Какая доза дигоксина рекомендуется для пациентов с массой тела более 85 кг?",
        expected_sources=["digoksin-u-bolnyh-s-hronicheskoy-serdechnoy-nedostatochnostyu-pro-i-contra.pdf"],
        expected_pages={
            "digoksin-u-bolnyh-s-hronicheskoy-serdechnoy-nedostatochnostyu-pro-i-contra.pdf": [1],
        }
    ),

    # Вопрос 8 — dinamika-markerov-fibrozirovaniya-s-uchetom-osobennostey-metabolicheskogo-fenotipa-u-patsientov-s-infarktom-mikarda.pdf
    EvalQuestion(
        question="Какова концентрация PICP у пациентов с нормальным весом на 1-е сутки заболевания?",
        expected_sources=["dinamika-markerov-fibrozirovaniya-s-uchetom-osobennostey-metabolicheskogo-fenotipa-u-patsientov-s-infarktom-mikarda.pdf"],
        expected_pages={
            "dinamika-markerov-fibrozirovaniya-s-uchetom-osobennostey-metabolicheskogo-fenotipa-u-patsientov-s-infarktom-mikarda.pdf": [3],
        }
    ),

    # Вопрос 9 — dreniruyuschaya-autoklapannaya-limbosklerektomiya-v-lechenii-posttravmaticheskoy-glaukomy.pdf
    EvalQuestion(
        question="Какое значение внутриглазного давления (ВГД) было зафиксировано у пациента К. на следующий день после операции субтотальной витрэктомии и имплантации интраокулярной линзы?",
        expected_sources=["dreniruyuschaya-autoklapannaya-limbosklerektomiya-v-lechenii-posttravmaticheskoy-glaukomy.pdf"],
        expected_pages={
            "dreniruyuschaya-autoklapannaya-limbosklerektomiya-v-lechenii-posttravmaticheskoy-glaukomy.pdf": [1, 2],
        }
    ),

    # Вопрос 10 — embrionalnyy-i-postnatalnyy-gistogenez-zubov-u-krys-v-usloviyah-zagryazneniya-okruzhayuschey-sredy.pdf
    EvalQuestion(
        question="Какова общая внутриутробная смертность в III группе животных, подвергнутых воздействию пестицидов и диоксидов серы и азота?",
        expected_sources=["embrionalnyy-i-postnatalnyy-gistogenez-zubov-u-krys-v-usloviyah-zagryazneniya-okruzhayuschey-sredy.pdf"],
        expected_pages={
            "embrionalnyy-i-postnatalnyy-gistogenez-zubov-u-krys-v-usloviyah-zagryazneniya-okruzhayuschey-sredy.pdf": [2],
        }
    ),

    # Вопрос 11 — gemofiltratsiya-pri-infuzionnoy-terapii-tyazhelogo-ostrogo-pankreatita.pdf
    EvalQuestion(
        question="Какой объем инфузионной терапии потребовался пациентам без гемофильтрации через 72 часа?",
        expected_sources=["gemofiltratsiya-pri-infuzionnoy-terapii-tyazhelogo-ostrogo-pankreatita.pdf"],
        expected_pages={
            "gemofiltratsiya-pri-infuzionnoy-terapii-tyazhelogo-ostrogo-pankreatita.pdf": [2],
        }
    ),

    # Вопрос 12 — gialuronidaza-eksperimentalnoe-podtverzhdenie-svoystv-endolimfaticheskogo-provodnika.pdf
    EvalQuestion(
        question="Какова концентрация цефотаксима в плазме крови кроликов через 1,5 часа после введения гиалуронидазы и антибиотика?",
        expected_sources=["gialuronidaza-eksperimentalnoe-podtverzhdenie-svoystv-endolimfaticheskogo-provodnika.pdf"],
        expected_pages={
            "gialuronidaza-eksperimentalnoe-podtverzhdenie-svoystv-endolimfaticheskogo-provodnika.pdf": [2],
        }
    ),

    # Вопрос 13 — himioterapiya-tuberkuleza-organov-dyhaniya-u-detey-i-podrostkov-nauchnye-podhody-k-resheniyu-problemy.pdf
    EvalQuestion(
        question="Какова продолжительность химиотерапии для пациентов с «малыми» формами туберкулеза органов дыхания согласно таблице 1?",
        expected_sources=["himioterapiya-tuberkuleza-organov-dyhaniya-u-detey-i-podrostkov-nauchnye-podhody-k-resheniyu-problemy.pdf"],
        expected_pages={
            "himioterapiya-tuberkuleza-organov-dyhaniya-u-detey-i-podrostkov-nauchnye-podhody-k-resheniyu-problemy.pdf": [2],
        }
    ),

    # Вопрос 14 — integratsiya-stomatologicheskoy-determinanty-v-programmu-aktivnogo-dolgoletiya.pdf
    EvalQuestion(
        question="Какой процент населения старше 60 лет будет в мире к 2050 году?",
        expected_sources=["integratsiya-stomatologicheskoy-determinanty-v-programmu-aktivnogo-dolgoletiya.pdf"],
        expected_pages={
            "integratsiya-stomatologicheskoy-determinanty-v-programmu-aktivnogo-dolgoletiya.pdf": [1, 2],
        }
    ),

    # Вопрос 15 — intestinalnaya-limfangiektaziya.pdf
    EvalQuestion(
        question="Какова масса тела девочки 17 лет при поступлении в клинику 3 апреля 2014 года?",
        expected_sources=["intestinalnaya-limfangiektaziya.pdf"],
        expected_pages={
            "intestinalnaya-limfangiektaziya.pdf": [3],
        }
    ),

    # Вопрос 16 — issledovanie-mediko-sotsialnyh-aspektov-zabolevaemosti-sredi-vzroslogo-i-detskogo-naseleniya-v-arabskih-stranah-blizhnego-vostoka.pdf
    EvalQuestion(
        question="Какой процент населения в Йемене, Ираке и Сирии составляет уровень бедности?",
        expected_sources=["issledovanie-mediko-sotsialnyh-aspektov-zabolevaemosti-sredi-vzroslogo-i-detskogo-naseleniya-v-arabskih-stranah-blizhnego-vostoka.pdf"],
        expected_pages={
            "issledovanie-mediko-sotsialnyh-aspektov-zabolevaemosti-sredi-vzroslogo-i-detskogo-naseleniya-v-arabskih-stranah-blizhnego-vostoka.pdf": [2],
        }
    ),

    # Вопрос 17 — issledovanie-sravnitelnoy-farmakokinetiki-i-bioekvivalentnosti-preparatov-amlodipin-valsartan-i-eksforzh.pdf
    EvalQuestion(
        question="Какие 90%-ные доверительные интервалы были получены для AUC0-t и Cmax амлодипина в исследовании биоэквивалентности препаратов Амлодипин+Валсартан и Эксфорж®?",
        expected_sources=["issledovanie-sravnitelnoy-farmakokinetiki-i-bioekvivalentnosti-preparatov-amlodipin-valsartan-i-eksforzh.pdf"],
        expected_pages={
            "issledovanie-sravnitelnoy-farmakokinetiki-i-bioekvivalentnosti-preparatov-amlodipin-valsartan-i-eksforzh.pdf": [1, 2],
        }
    ),

    # Вопрос 18 — izmeneniya-faktorov-angiogeneza-i-markerov-endotelialnoy-disfunktsii-pri-bolezni-vilsona-konovalova-u-detey-i-podrostkov.pdf
    EvalQuestion(
        question="Какова концентрация VEGF-A у детей с болезнью Вильсона–Коновалова, согласно результатам исследования?",
        expected_sources=["izmeneniya-faktorov-angiogeneza-i-markerov-endotelialnoy-disfunktsii-pri-bolezni-vilsona-konovalova-u-detey-i-podrostkov.pdf"],
        expected_pages={
            "izmeneniya-faktorov-angiogeneza-i-markerov-endotelialnoy-disfunktsii-pri-bolezni-vilsona-konovalova-u-detey-i-podrostkov.pdf": [2],
        }
    ),

    # Вопрос 19 — izolirovannyy-tuberkulez-sinovialnoy-obolochki-kolennogo-sustava.pdf
    EvalQuestion(
        question="Какой объем синовиальной жидкости был определен при пальпации коленного сустава пациента в сентябре, через полгода после начала заболевания?",
        expected_sources=["izolirovannyy-tuberkulez-sinovialnoy-obolochki-kolennogo-sustava.pdf"],
        expected_pages={
            "izolirovannyy-tuberkulez-sinovialnoy-obolochki-kolennogo-sustava.pdf": [2],
        }
    ),

    # Вопрос 20 — izuchenie-osobenostey-semi-s-rebenkom-doshkolnogo-vozrasta.pdf
    EvalQuestion(
        question="Какова заболеваемость детей, проживающих в курящих семьях?",
        expected_sources=["izuchenie-osobenostey-semi-s-rebenkom-doshkolnogo-vozrasta.pdf"],
        expected_pages={
            "izuchenie-osobenostey-semi-s-rebenkom-doshkolnogo-vozrasta.pdf": [1],
        }
    ),

    # Вопрос 21 — izuchenie-otdalennyh-rezultatov-profilakticheskoy-pomoschi-patsientam-rabotosposobnogo-vozrasta-s-arterialnoy-gipertoniey-pri-uchastii-srednih-meditsinskih-rabotnikov.pdf
    EvalQuestion(
        question="Какой процент пациентов в группе наблюдения через 6 месяцев после обучения начал регулярно принимать гипотензивные лекарственные средства?",
        expected_sources=["izuchenie-otdalennyh-rezultatov-profilakticheskoy-pomoschi-patsientam-rabotosposobnogo-vozrasta-s-arterialnoy-gipertoniey-pri-uchastii-srednih-meditsinskih-rabotnikov.pdf"],
        expected_pages={
            "izuchenie-otdalennyh-rezultatov-profilakticheskoy-pomoschi-patsientam-rabotosposobnogo-vozrasta-s-arterialnoy-gipertoniey-pri-uchastii-srednih-meditsinskih-rabotnikov.pdf": [2],
        }
    ),

    # Вопрос 22 — izuchenie-udovletvorennosti-patsientami-i-ih-rodstvennikami-kachestvom-usloviy-prebyvaniya-v-otdelenii-sestrinskogo-uhoda.pdf
    EvalQuestion(
        question="Какой процент пациентов, лечившихся три и более месяцев в отделении сестринского ухода, составляют по данным исследования?",
        expected_sources=["izuchenie-udovletvorennosti-patsientami-i-ih-rodstvennikami-kachestvom-usloviy-prebyvaniya-v-otdelenii-sestrinskogo-uhoda.pdf"],
        expected_pages={
            "izuchenie-udovletvorennosti-patsientami-i-ih-rodstvennikami-kachestvom-usloviy-prebyvaniya-v-otdelenii-sestrinskogo-uhoda.pdf": [1],
        }
    ),

    # Вопрос 23 — kastratsionno-rezistentnyy-rak-predstatelnoy-zhelezy-novye-perspektivy-lekarstvennoy-terapii.pdf
    EvalQuestion(
        question="Какова медиана выживаемости без метастазов при применении апалутамида по сравнению с плацебо в исследовании SPARTAN?",
        expected_sources=["kastratsionno-rezistentnyy-rak-predstatelnoy-zhelezy-novye-perspektivy-lekarstvennoy-terapii.pdf"],
        expected_pages={
            "kastratsionno-rezistentnyy-rak-predstatelnoy-zhelezy-novye-perspektivy-lekarstvennoy-terapii.pdf": [3],
        }
    ),

    # Вопрос 24 — klinicheskie-proyavleniya-i-effektivnost-lecheniya-tuberkulyoza-lyogkih-s-mnozhestvennoy-lekarstvennoy-ustoychivostyu-vozbuditelya-u-bolnyh-saharnym-diabetom.pdf
    EvalQuestion(
        question="Какова эффективность прекращения бактериовыделения у больных с сахарным диабетом 1-го типа при лечении туберкулёза лёгких с множественной лекарственной устойчивостью возбудителя?",
        expected_sources=["klinicheskie-proyavleniya-i-effektivnost-lecheniya-tuberkulyoza-lyogkih-s-mnozhestvennoy-lekarstvennoy-ustoychivostyu-vozbuditelya-u-bolnyh-saharnym-diabetom.pdf"],
        expected_pages={
            "klinicheskie-proyavleniya-i-effektivnost-lecheniya-tuberkulyoza-lyogkih-s-mnozhestvennoy-lekarstvennoy-ustoychivostyu-vozbuditelya-u-bolnyh-saharnym-diabetom.pdf": [1, 3],
        }
    ),

    # Вопрос 25 — klinicheskiy-polimorfizm-pri-narusheniyah-v-rabote-pischevaritelnoy-sistemy-i-metody-ih-korrektsii-u-detey-rannego-vozrasta-rodivshihsya-s-ekstremalno-nizkoy-i-ochen-nizkoy-massoy-tela-v-katamneze.pdf
    EvalQuestion(
        question="Какой процент детей, рожденных с очень низкой массой тела (ОНМТ), имеет недостаточный прирост массы тела в первые месяцы жизни?",
        expected_sources=["klinicheskiy-polimorfizm-pri-narusheniyah-v-rabote-pischevaritelnoy-sistemy-i-metody-ih-korrektsii-u-detey-rannego-vozrasta-rodivshihsya-s-ekstremalno-nizkoy-i-ochen-nizkoy-massoy-tela-v-katamneze.pdf"],
        expected_pages={
            "klinicheskiy-polimorfizm-pri-narusheniyah-v-rabote-pischevaritelnoy-sistemy-i-metody-ih-korrektsii-u-detey-rannego-vozrasta-rodivshihsya-s-ekstremalno-nizkoy-i-ochen-nizkoy-massoy-tela-v-katamneze.pdf": [2],
        }
    ),

    # Вопрос 26 — klinika-diagnostika-i-lechenie-vrozhdennogo-komedonovogo-nevusa.pdf
    EvalQuestion(
        question="Какого размера был комедоновый невус у пациентки А. на момент обращения в 32 года?",
        expected_sources=["klinika-diagnostika-i-lechenie-vrozhdennogo-komedonovogo-nevusa.pdf"],
        expected_pages={
            "klinika-diagnostika-i-lechenie-vrozhdennogo-komedonovogo-nevusa.pdf": [2],
        }
    ),

    # Вопрос 27 — kliniko-farmakokineticheskie-paralleli-perioperatsionnoy-antibiotikoprofilaktiki-v-abdominalnoy-hirurgii.pdf
    EvalQuestion(
        question="Какова концентрация цефоперазона в крови через 20 минут после его введения?",
        expected_sources=["kliniko-farmakokineticheskie-paralleli-perioperatsionnoy-antibiotikoprofilaktiki-v-abdominalnoy-hirurgii.pdf"],
        expected_pages={
            "kliniko-farmakokineticheskie-paralleli-perioperatsionnoy-antibiotikoprofilaktiki-v-abdominalnoy-hirurgii.pdf": [1, 2],
        }
    ),

    # Вопрос 28 — kliniko-fiziologicheskie-pokazateli-ispolzovaniya-preryvistoy-pnevmokompressii-dlya-profilaktiki-tromboza-glubokih-ven-i-tromboembolii-lyogochnyh-arteriy.pdf
    EvalQuestion(
        question="Какой процент пациентов с острым нарушением мозгового кровообращения имеет риск тромбоэмболии?",
        expected_sources=["kliniko-fiziologicheskie-pokazateli-ispolzovaniya-preryvistoy-pnevmokompressii-dlya-profilaktiki-tromboza-glubokih-ven-i-tromboembolii-lyogochnyh-arteriy.pdf"],
        expected_pages={
            "kliniko-fiziologicheskie-pokazateli-ispolzovaniya-preryvistoy-pnevmokompressii-dlya-profilaktiki-tromboza-glubokih-ven-i-tromboembolii-lyogochnyh-arteriy.pdf": [1],
        }
    ),

    # Вопрос 29 — kompleksnyy-regionarnyy-bolevoy-sindrom.pdf
    EvalQuestion(
        question="Какой препарат был назначен пациентке в составе терапии при комплексном регионарном болевом синдроме?",
        expected_sources=["kompleksnyy-regionarnyy-bolevoy-sindrom.pdf"],
        expected_pages={
            "kompleksnyy-regionarnyy-bolevoy-sindrom.pdf": [3],
        }
    ),

    # Вопрос 30 — lechenie-patsientov-s-arterialnoy-gipertenziey-v-realnoy-ambulatornoy-praktike.pdf
    EvalQuestion(
        question="Какой процент пациентов, получающих гипотензивную терапию, достигают целевых значений артериального давления в реальной амбулаторной практике?",
        expected_sources=["lechenie-patsientov-s-arterialnoy-gipertenziey-v-realnoy-ambulatornoy-praktike.pdf"],
        expected_pages={
            "lechenie-patsientov-s-arterialnoy-gipertenziey-v-realnoy-ambulatornoy-praktike.pdf": [1],
        }
    ),

]


# ============================================================================
# ПРИМЕР АННОТИРОВАННОГО ВОПРОСА (для копирования при аннотировании)
# ============================================================================
# EvalQuestion(
#     question="Ваш вопрос здесь?",
#     expected_sources=["document.pdf"],
#     expected_pages={
#         "document.pdf": [5, 7, 12]
#     }
# ),
