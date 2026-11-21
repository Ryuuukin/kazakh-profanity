"""
kz_regex — словарь + regex для поиска мата в казахских / смешанных казахско-русских отзывах.

Использование:
    import kz_regex
    kz_regex.tag("Пиздес ойын екен")            # -> (1, ['ru_pizd:пиздес'], False)
    kz_regex.tag("Блин көшірілмейді")           # -> (-1, ['ru_euph:блин'], False)
    df = kz_regex.tag_series(df["text"], version=1)  # колонки regex_label, regex_terms, regex_masked

Метки:
     1  — найдено однозначно обсценное / вульгарное слово
    -1  — только «слабое» / двусмысленное слово (блин, жалап, боқтау...) -> нужна ручная проверка
     0  — ничего не найдено

Версии:
    version=1 — исходный словарь; им отбиралась выборка для разметки (нужен для воспроизводимости).
    version=2 — улучшенный словарь: добавлены формы, которые v1 пропускал (наху, зайбал, далбаёб,
                шешең амы, шшс, ташақ, ...). Его лучше использовать для новых данных.

Модуль назван kz_regex, а не regex.py, чтобы не перекрыть популярный пакет `regex`
(его импортируют transformers / tokenizers).
"""
import re

import pandas as pd

# ---------------------------------------------------------------- нормализация
LAT2CYR = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х",
                         "y": "у", "k": "к", "m": "м", "t": "т", "b": "в", "h": "н"})
LEET = str.maketrans({"@": "а", "0": "о", "3": "з", "$": "с", "€": "е"})
CYR = re.compile(r"[а-яёәғқңөұүһі]")
MASK = "*#%•"  # ! и ? не считаем маской: «ху!» — это междометие


def norm_token(tok: str) -> str:
    # бл.я / х-у-й  -> бля / хуй  (только если между каждой буквой разделитель)
    if re.fullmatch(r"(?:\w[.\-_]){2,}\w", tok):
        tok = re.sub(r"[.\-_]", "", tok)
    # цифры/символы вместо букв, только внутри слов с буквами
    if re.search(r"[a-zа-яё]", tok):
        tok = tok.translate(LEET)
    # смешанная письменность: латинские двойники -> кириллица
    if CYR.search(tok) and re.search(r"[a-z]", tok):
        tok = tok.translate(LAT2CYR)
    # растянутые буквы: бляяяя -> бля
    tok = re.sub(r"(.)\1{2,}", r"\1", tok)
    return tok


def normalize(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    return " ".join(norm_token(t) for t in text.split())


# ---------------------------------------------------------------- словарь
B = r"(?<![а-яёәғқңөұүһіa-z])"   # начало слова
E = r"(?![а-яёәғқңөұүһіa-z])"    # конец слова
W = r"[а-яёәғқңөұүһіa-z]*"       # хвост слова

STRONG = {
    # --- русский мат (кириллица)
    "ru_xuj":   rf"(?<!стра)(?<!застра)х[уy][йеёияю]{W}",
    "ru_pizd":  rf"п[иеi]зд{W}",
    "ru_eb":    rf"{B}(?:за|на|у|до|въ|съ|вы|по|от|раз|об|долбо|при|про|из|под)?[её]б(?:а[тлнш]|ал|ан|ну|ук|уч|ыв|ись|ош|ло|ля|л[иао]){W}",
    "ru_yob":   rf"{B}(?:за|на|у|по|раз)?ёб{W}",
    "ru_blya":  rf"{B}бл[яa](?:д|т|ха|ть|дь|динa?)?{W}",
    "ru_mud":   rf"{B}муд[аои]к{W}|{B}мудил{W}",
    "ru_pidor": rf"{B}п[иеі]д[оаe]?р{W}",
    "ru_gandon": rf"{B}г[ао]нд[оа]н{W}",
    "ru_zalup": rf"залуп{W}",
    "ru_shluh": rf"{B}шлюх{W}",
    "ru_suka":  rf"{B}с[уy]к[аиуеоа](?:н|ми|лар|лал)?{W}|{B}сучк{W}|{B}сучар{W}",
    "ru_govno": rf"{B}г[ао]вн{W}|{B}дерьм{W}",
    "ru_eblan": rf"{B}[её]блан{W}",
    # --- русский мат латиницей
    "ru_lat":   rf"{B}(?:na|po)?(?:hu[iy]|xu[iy]|hue|xue){W}|{B}pi[sz]d{W}|{B}bl[yi]a(?:t|d)?{W}|{B}blyat{W}|{B}(?:za|na|u|vy|do|po)?[eyo]ba[tln]{W}|{B}eblan{W}|{B}suk[ai]{E}|{B}pid[ao]r{W}",
    # --- казахский: сік / сіг-
    "kk_sik":   rf"{B}с[іи]к(?:{E}|т[іи]{W}|[іи]м{E}|кен{W}|[іи]п{W}|е[йи]{W})"
                rf"|{B}с[іи]г[еэ]й?[іи]н{W}|{B}с[іи]г[еэ]м{E}|{B}с[іи]г[іи]п{W}|{B}с[іи]г[іи]л{W}",
    "kk_sik_lat": rf"{B}s[iy]k(?:{E}|t[iy]{W}|eyin|ein|en)|{B}s[iy]g(?:eyin|ein|em|ip){W}",
    # --- казахский: қотақ
    "kk_qotaq": rf"{B}(?:қ|к|q|k)[оo][тt][аa](?:қ|к|q|k|ғ|г){W}",
    # --- казахский: ам- (только формы с притяжательным окончанием)
    "kk_am":    rf"{B}ам[ыі]ң{W}|{B}ам[ыи]нд[ыа]{W}|{B}ам[ыі]на{E}|{B}am[yi]n(?:dy|ny|a){W}|{B}amyń{W}",
    # --- казахский: «шешеңді ұрайын» и подобные конструкции
    "kk_mother": rf"(?:шешең|шешен|әкең|акен|әкен|анаң|апаң|sheshen|aken)\w*\s+(?:ұр|ур|ur){W}",
    # --- казахский: көт- (зад, вульгарно) — только формы, которые не путаются с «көтеру»
    "kk_kot":   rf"{B}к[өо]т[іи]ң{W}|{B}к[өо]т[іи]н[еді]{E}|{B}к[өо]т[іи]не{W}",
    # --- казахский: боқ (только с буквой қ, т.к. «бок» — русское слово)
    # --- добавлено в v2 по итогам проверки LLM (см. README)
    "ru_nahu":  rf"{B}(?:на|по|ни|не|да)х[уy](?:й|и|я)?{E}|{B}нах{E}|{B}nah[uy]{W}",
    "ru_zaebal": rf"за[йи]б[аa]л{W}|{B}зайб{W}|za[iy]bal{W}",
    "ru_dolboeb": rf"{B}д[ао]лб[ао](?:[её]|й[оеё]|ю|и[оё])?[бп]{W}|{B}д[ао]лбан{W}|{B}dalba(?:e|yo|io)[bp]{W}",
    "ru_yopt":  rf"{B}(?:[её]|[ий]о)пт(?:[ыи]|ваю|вай)?{W}|{B}[её]п[ -]?тв[ао][йю]{W}|{B}ёп{E}",
    "ru_eba":   rf"{B}еба+{E}|{B}еба[йи]т{W}|{B}ebait{W}",
    "ru_blya2": rf"{B}бл[әə]+{E}|{B}бл[әə]+[тд]{W}|{B}бл[иі]я?а?т{W}|{B}былат{E}|{B}бл\*+",
    "ru_gandon_lat": rf"{B}g[ae]nd[oa]n{W}",
    "abbr_shshs": rf"{B}(?:шшс|щщс|шщс|щшс|ššs|shhs){E}",
    "kk_amy":   rf"{B}амы{E}|{B}ам{E}|{B}амдар{W}|{B}амш[ыі][қк]{W}|{B}amsh[iy][qk]{W}|{B}амы?с+(?:[іи]г|ки){W}"
                rf"|(?:шеше|енең|анаң|апаң)\w{{0,5}}\s?амы{W}",
    "kk_sgin":  rf"(?:{B}|шеше[ңн]?\w{{0,3}}|анас+|әнәс+|амы|мамас+)с{{1,2}}[іи]?г(?:[еэ]й?[іи]?н|[іи]н|ем|ейн){W}",
    "kk_tashaq": rf"{B}та[шс]а[қк]{W}|{B}tasha[qk]{W}",
    "kk_kott":  rf"{B}к[өо]тт[еі]р{W}",
    "kk_zhalep": rf"{B}ж[аә]л[еә]п{W}|{B}жалаптар{W}|{B}zhalep{W}",
    "kk_boq":   rf"{B}боқ(?:{E}|[ыі]{E}|қа{E}|тың{E}|сың{E}|ты{E})|{B}boq{E}",
}

# Слабые/двусмысленные: отправляем на ручную проверку, не ставим 1 автоматически.
WEAK = {
    "kk_zhalap": rf"{B}(?:жалап|jalap|zhalap){E}",          # «проститутка» ИЛИ «облизывая»
    "ru_bla":    rf"{B}бла+{E}|{B}білә{E}",                 # «бла» = «бля» или «бла-бла»
    "abbr_zb":   rf"{B}зб{E}",                              # «зб» = «заебал», но коротко и двусмысленно
    "ru_her":    rf"{B}(?:на|по)?хер{W}",                   # «хер» — эвфемизм, часто мягко
    "ru_euph":   rf"{B}(?:блин|блиин|капец|пипец|офиге\w*|фиг(?:ня|ни|не|а|е)?{E})",
    "kk_boqtau": rf"{B}боқта{W}",                           # «боқтау» = ругаться (упоминание)
}

# Маскированные формы: х*й, п**да, бл*ть, с*гейін, е**ть ...
MASKED = {
    "m_xuj":   rf"х[{re.escape(MASK)}]+[йяеи]|ху[{re.escape(MASK)}]+",
    "m_pizd":  rf"п[{re.escape(MASK)}]+зд|пи[{re.escape(MASK)}]+[дц]|п[{re.escape(MASK)}]{{2,}}[ад]",
    "m_blya":  rf"{B}бл[{re.escape(MASK)}]+[тдя]|{B}б[{re.escape(MASK)}]+я{E}|{B}б[{re.escape(MASK)}]+дь",
    "m_eb":    rf"{B}(?:за|на|у|по)?[её][{re.escape(MASK)}]+(?:ал|ать|ан|ну)",
    "m_sik":   rf"{B}с[{re.escape(MASK)}]+г[еэ]й?[іи]н|{B}с[іи][{re.escape(MASK)}]+[еэ]й?[іи]н|{B}с[{re.escape(MASK)}]+к{E}",
    "m_qotaq": rf"{B}[қк][{re.escape(MASK)}]+т[аa][қк]|{B}[қк]о[{re.escape(MASK)}]+[аa][қк]",
    "m_suka":  rf"{B}с[{re.escape(MASK)}]+ка{E}",
}


# ---------------------------------------------------------------- версия 1
V2_ADDED = {"ru_nahu", "ru_zaebal", "ru_dolboeb", "ru_yopt", "ru_eba", "ru_blya2", "ru_gandon_lat",
            "abbr_shshs", "kk_amy", "kk_sgin", "kk_tashaq", "kk_kott", "kk_zhalep"}
STRONG_V1 = {k: v for k, v in STRONG.items() if k not in V2_ADDED}
STRONG_V1["kk_qotaq"] = rf"{B}(?:қ|к|q|k)[оo]т[аa](?:қ|к|q|k){W}"
STRONG_V1["ru_suka"] = rf"{B}с[уy]к[аиуеоа](?:н|ми)?{E}|{B}сучк{W}|{B}сучар{W}"
WEAK_V1 = {k: v for k, v in WEAK.items() if k not in ("ru_bla", "abbr_zb")}
WEAK_V1["kk_zhalap"] = rf"{B}(?:ж[аә]л[аеә]п|jalap|zhalap){W}"


def _compile(d):
    return {k: re.compile(v) for k, v in d.items()}


RULES = {
    1: (_compile(STRONG_V1), _compile(MASKED), _compile(WEAK_V1)),
    2: (_compile(STRONG), _compile(MASKED), _compile(WEAK)),
}


# ---------------------------------------------------------------- API
def tag(text: str, version: int = 2):
    """Возвращает (метка, список сработавших правил 'правило:слово', было_ли_маскировано)."""
    strong, masked_rx, weak = RULES[version]
    t = normalize(text)
    terms, masked = [], False
    for name, rx in strong.items():
        m = rx.search(t)
        if m:
            terms.append(f"{name}:{m.group(0)}")
    for name, rx in masked_rx.items():
        m = rx.search(t)
        if m:
            terms.append(f"{name}:{m.group(0)}")
            masked = True
    if terms:
        return 1, terms, masked
    found = []
    for name, rx in weak.items():
        m = rx.search(t)
        if m:
            found.append(f"{name}:{m.group(0)}")
    if found:
        return -1, found, False
    return 0, [], False


def is_profane(text: str, version: int = 2) -> bool:
    return tag(text, version)[0] == 1


def tag_series(texts: pd.Series, version: int = 2) -> pd.DataFrame:
    """Размечает колонку текстов; возвращает DataFrame с regex_label, regex_terms, regex_masked."""
    res = texts.astype(str).map(lambda t: tag(t, version))
    return pd.DataFrame({
        "regex_label": res.map(lambda r: r[0]),
        "regex_terms": res.map(lambda r: "|".join(r[1])),
        "regex_masked": res.map(lambda r: r[2]),
    }, index=texts.index)


def rule_counts(terms: pd.Series) -> pd.Series:
    """Сколько раз сработало каждое правило (по колонке regex_terms)."""
    t = terms[terms.fillna("") != ""].str.split("|").explode()
    return t.str.split(":").str[0].value_counts()
