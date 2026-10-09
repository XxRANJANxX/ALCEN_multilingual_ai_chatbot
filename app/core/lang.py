import re
from langdetect import DetectorFactory, detect_langs
DetectorFactory.seed = 0

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "ar": "Arabic", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu", "it": "Italian", "tr": "Turkish", "id": "Indonesian",
    "nl": "Dutch",
}

REFUSALS = {
    "en": "I couldn't find enough information in my knowledge base to answer that reliably.",
    "hi": "मुझे इसका भरोसेमंद उत्तर देने के लिए अपने ज्ञान-आधार में पर्याप्त जानकारी नहीं मिली।",
    "es": "No encontré suficiente información en mi base de conocimiento para responder con fiabilidad.",
    "fr": "Je n'ai pas trouvé assez d'informations dans ma base de connaissances pour répondre de façon fiable.",
    "de": "Ich habe in meiner Wissensbasis nicht genügend Informationen gefunden, um das zuverlässig zu beantworten.",
    "pt": "Não encontrei informações suficientes na minha base de conhecimento para responder com segurança.",
}


_EN_WORDS = {
    "the", "is", "are", "what", "how", "does", "of", "to", "for", "and", "can", "my",
    "this", "that", "which", "who", "when", "where", "why", "with", "be", "was",
    "will", "about", "from", "by", "or", "not", "have", "has",
}


def detect_language(text: str) -> str:
    """langdetect is unreliable on short English questions, so bias Latin-script text to English."""
    letters = re.findall(r"[^\W\d_]", text)
    is_latin = bool(letters) and all(ord(c) < 0x250 for c in letters)
    if is_latin:
        words = re.findall(r"[a-z']+", text.lower())
        hits = sum(w in _EN_WORDS for w in words)
        if hits >= 2 or (words and len(words) <= 3 and hits >= 1):
            return "en"
    try:
        best = detect_langs(text)[0]
    except Exception:
        return "en"
    if is_latin and best.prob < 0.85:
        return "en"
    return best.lang


def language_name(code: str) -> str:
    return LANG_NAMES.get(code.lower(), code)


def refusal_message(code: str) -> str:
    return REFUSALS.get(code.lower().split("-")[0], REFUSALS["en"])
