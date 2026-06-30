"""Domain-aware anchor extraction and normalization helpers."""
import re
from typing import Dict, Iterable, List, Optional


DOMAIN_KEYWORDS = {
    "enron": ("enron", "email", "business", "company", "corporate", "finance"),
    "health": ("health", "healthcare", "medical", "medicine", "clinical", "patient"),
    "pokemon": ("pokemon", "species", "move", "evolution", "trainer"),
    "literature": ("harry", "potter", "fiction", "literature", "book", "novel", "fantasy"),
}


GENERIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "here", "him", "his", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "please", "she",
    "so", "that", "the", "their", "them", "then", "there", "these", "this", "those", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "about", "after", "again", "available", "because", "before", "being", "details", "during",
    "focused", "general", "include", "provide", "relevant", "retrieved", "should", "specific", "through",
    "within", "while", "first", "item", "items",
}


DOMAIN_STOPWORDS = {
    "enron": {
        "subject", "sender", "recipient", "recipients", "file", "from", "to", "cc", "bcc", "sent", "date",
        "message", "email", "doc", "memo", "original message", "forwarded", "forwarded by", "pm to", "am to",
        "ect", "ena", "corp", "inc", "http", "www",
    },
    "health": {
        "hi", "hello", "thanks", "thank", "doctor", "dr", "sir", "madam", "background", "question",
        "answer", "really", "right", "year", "years", "month", "months", "week", "weeks", "day", "days",
    },
    "pokemon": {
        "pokemon", "generation", "version", "game", "games", "episode", "episodes", "player",
        "battle", "type", "types",
    },
    "literature": {
        "chapter", "book", "page", "novel", "harry potter", "said", "asked", "looked", "moment",
    },
    "general": set(),
}


ACTION_WORDS = (
    "approve", "approved", "schedule", "scheduled", "meet", "met", "send", "sent", "review", "reviewed",
    "confirm", "confirmed", "cancel", "cancelled", "trade", "traded", "settle", "settled", "execute",
    "executed", "discuss", "discussed", "update", "updated", "draft", "drafted", "sign", "signed",
)


MEDICAL_TERMS = (
    "pain", "fever", "rash", "cough", "bleeding", "swelling", "ache", "nausea", "vomit", "dizzy",
    "headache", "infection", "disease", "syndrome", "cancer", "diabetes", "asthma", "allergy", "test",
    "scan", "xray", "mri", "ct", "blood", "urine", "tablet", "capsule", "dose", "mg", "drug",
    "medicine", "medication", "heart", "chest", "stomach", "abdomen", "skin", "eye", "ear", "throat",
    "male", "female", "pregnant", "child", "infant", "teen", "adult",
)


POKEMON_TYPES = (
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
    "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
)


POKEMON_MECHANICS = (
    "evolve", "evolves", "evolution", "mega", "gigantamax", "dynamax", "stone", "badge", "gym", "route",
    "city", "town", "cave", "forest", "island", "league", "move", "ability", "item", "berry", "ball",
)


def resolve_anchor_domain(dataset_name: str = "", topic_word: str = "", preset: str = "") -> str:
    text = f"{dataset_name or ''} {topic_word or ''} {preset or ''}".lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return domain
    return "general"


def normalize_anchor(anchor: str, domain: str = "general") -> Optional[str]:
    anchor = str(anchor or "")
    anchor = re.sub(r"[\r\n\t]+", " ", anchor)
    anchor = re.sub(r"^(context|chunk)\s*\d+\s*:\s*", "", anchor, flags=re.IGNORECASE)
    anchor = re.sub(r"^(re|fw|fwd)\s*:\s*", "", anchor, flags=re.IGNORECASE)
    anchor = re.sub(r"\s+", " ", anchor).strip(" .,:;()[]{}\"'")

    if not anchor:
        return None

    email_match = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", anchor, flags=re.IGNORECASE)
    if domain == "enron" and email_match:
        return email_match.group(0).lower()

    date_match = re.search(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{0,4})\b",
        anchor,
        flags=re.IGNORECASE,
    )
    if domain == "enron" and date_match:
        return _title_preserving_acronyms(date_match.group(0))

    anchor = re.sub(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", " ", anchor, flags=re.IGNORECASE)
    anchor = re.sub(r"\b(?:am|pm)\s+to\b", " ", anchor, flags=re.IGNORECASE)
    anchor = re.sub(r"\s+", " ", anchor).strip(" .,:;()[]{}\"'")
    anchor = _domain_rewrite(anchor, domain)
    anchor = _strip_edge_noise(anchor, domain)

    if _is_noisy(anchor, domain):
        return None

    tokens = anchor.split()
    if len(tokens) > 6:
        anchor = " ".join(tokens[:6]).strip(" .,:;()[]{}\"'")
    if len(anchor) > 80:
        anchor = anchor[:80].rsplit(" ", 1)[0].strip(" .,:;()[]{}\"'")

    if _is_noisy(anchor, domain):
        return None
    return _title_preserving_acronyms(anchor)


def _domain_rewrite(anchor: str, domain: str) -> str:
    lowered = anchor.lower()
    if domain == "enron":
        if re.search(r"\b(?:com|net|org)\s+on\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", lowered):
            return ""
        anchor = re.sub(r"^(?:and|or|please)\s+", "", anchor, flags=re.IGNORECASE)
    elif domain == "health":
        anchor = re.sub(r"^(?:hi|hello|thanks|thank you)\s+(?:doctor|dr|sir|madam)?\b", "", anchor, flags=re.IGNORECASE)
        anchor = re.sub(r"\b(?:hi|hello|thanks|thank you)\s+(?:doctor|dr|sir|madam)\b", "", anchor, flags=re.IGNORECASE)
    elif domain == "pokemon":
        anchor = re.sub(r"^(?:first\s+)?items?\s+include\s+", "", anchor, flags=re.IGNORECASE)
        anchor = re.sub(r"^first\s+", "", anchor, flags=re.IGNORECASE)
        anchor = re.sub(r"\s+on$", "", anchor, flags=re.IGNORECASE)
        words = anchor.split()
        lowered_words = [word.lower().strip(".,:;()[]{}\"'") for word in words]
        if (
            len(words) > 2
            and lowered_words[0] not in POKEMON_TYPES
            and lowered_words[0] not in POKEMON_MECHANICS
            and any(word in POKEMON_TYPES for word in lowered_words[1:])
        ):
            anchor = words[0]
    elif domain == "literature":
        anchor = re.sub(r"^(?:near|before|after|inside|outside|at|in|the)\s+", "", anchor, flags=re.IGNORECASE)
        anchor = re.sub(r"\s+(?:near|before|after|inside|outside|at|in|the)$", "", anchor, flags=re.IGNORECASE)
    anchor = re.sub(r"\b[Aa]\b", " ", anchor)
    return re.sub(r"\s+", " ", anchor).strip(" .,:;()[]{}\"'")


def _strip_edge_noise(anchor: str, domain: str) -> str:
    edge_noise = set(GENERIC_STOPWORDS)
    edge_noise.update(DOMAIN_STOPWORDS.get(domain, set()))
    words = anchor.split()
    while words and words[0].lower().strip(".,:;()[]{}\"'") in edge_noise:
        words.pop(0)
    while words and words[-1].lower().strip(".,:;()[]{}\"'") in edge_noise:
        words.pop()
    return " ".join(words).strip(" .,:;()[]{}\"'")


def extract_anchors_from_texts(
    texts: Iterable[str],
    domain: str = "general",
    max_anchors: int = 12,
    existing: Iterable[str] = (),
) -> List[str]:
    counts: Dict[str, int] = {}
    seen_existing = {str(item).lower() for item in existing if item}

    for raw_text in texts:
        text = str(raw_text or "")
        if not text.strip():
            continue
        for candidate, weight in _candidate_phrases(text, domain):
            anchor = normalize_anchor(candidate, domain)
            if not anchor or anchor.lower() in seen_existing:
                continue
            counts[anchor] = counts.get(anchor, 0) + weight

    ranked = sorted(counts, key=lambda item: (counts[item], _domain_specificity(item, domain), len(item)), reverse=True)
    return ranked[:max_anchors]


def normalized_anchor_focus(chunk: str, domain: str = "general", max_anchors: int = 5) -> str:
    anchors = extract_anchors_from_texts([chunk], domain=domain, max_anchors=max_anchors)
    if not anchors:
        normalized = normalize_anchor(chunk, domain)
        return normalized or ""
    return ", ".join(anchors)


def _candidate_phrases(text: str, domain: str):
    candidates = []

    for email in re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE):
        candidates.append((email, 5))

    for date in re.findall(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{0,4})\b",
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append((date, 4 if domain == "enron" else 1))

    for phrase in re.findall(r"\b(?:[A-Z][a-zA-Z0-9&'-]+\s+){0,5}[A-Z][a-zA-Z0-9&'-]+\b", text):
        candidates.append((phrase, 3))

    for phrase in re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]*(?:\s+[a-zA-Z][a-zA-Z0-9'-]*){1,4}\b", text):
        score = _domain_specificity(phrase, domain)
        if score > 0:
            candidates.append((phrase, 2 + score))

    for term in re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{3,}\b", text):
        score = _domain_specificity(term, domain)
        if score > 0:
            candidates.append((term, 1 + score))

    return candidates


def _domain_specificity(anchor: str, domain: str) -> int:
    lowered = anchor.lower()
    score = 0
    if domain == "enron":
        if "@" in lowered:
            score += 4
        if any(word in lowered for word in ACTION_WORDS):
            score += 2
        if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|\d{1,2}[/-]\d{1,2})", lowered):
            score += 2
    elif domain == "health":
        if any(word in lowered for word in MEDICAL_TERMS):
            score += 3
        if re.search(r"\b\d+\s*(?:yo|year|years|month|months|week|weeks|day|days)\b", lowered):
            score += 2
    elif domain == "pokemon":
        if any(word in lowered for word in POKEMON_TYPES):
            score += 2
        if any(word in lowered for word in POKEMON_MECHANICS):
            score += 3
    elif domain == "literature":
        if any(word in lowered for word in ("harry", "hermione", "ron", "dumbledore", "voldemort", "hogwarts", "wand", "spell")):
            score += 3
        if any(word in lowered for word in ("hall", "tower", "forest", "chamber", "stone", "cup", "map", "battle", "lesson")):
            score += 2
    else:
        if re.search(r"[A-Z]", anchor):
            score += 1
    return score


def _is_noisy(anchor: str, domain: str) -> bool:
    lowered = anchor.lower().strip()
    if not lowered:
        return True
    tokens = lowered.split()
    if len(anchor) < 3 or len(tokens) > 8:
        return True
    if lowered in GENERIC_STOPWORDS or lowered in DOMAIN_STOPWORDS.get(domain, set()):
        return True
    if any(part in lowered for part in ("original message", "forwarded by", "http://", "https://")):
        return True
    if re.search(r"\.(doc|xls|xlsx|ppt|pdf|txt)\b", lowered):
        return True
    if re.fullmatch(r"[\d\s:./,-]+", lowered):
        return True
    if len(tokens) == 1 and tokens[0] in GENERIC_STOPWORDS:
        return True
    if domain in {"enron", "health", "pokemon", "literature"} and len(tokens) == 1:
        token = tokens[0]
        if token in DOMAIN_STOPWORDS.get(domain, set()):
            return True
        if token in GENERIC_STOPWORDS:
            return True
    return False


def _title_preserving_acronyms(text: str) -> str:
    words = []
    for word in text.split():
        if "@" in word or word.isupper() or re.fullmatch(r"\d+[/-]\d+[/-]\d+", word):
            words.append(word)
        elif re.fullmatch(r"[A-Za-z]\.?[A-Za-z]?\.", word):
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)
