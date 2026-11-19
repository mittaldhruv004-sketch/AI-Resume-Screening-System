"""Deterministic skill extraction with optional spaCy phrase extraction."""
import re
from typing import List
from src.utils.keyword_mapping import SKILL_GROUPS, map_keyword_to_group


def normalize_text(text: str, preserve_newlines: bool = False) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if preserve_newlines:
        lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text: str) -> List[str]:
    """Return canonical skill groups found anywhere in the supplied text."""
    source = (text or "").lower()
    found = []
    for group, aliases in SKILL_GROUPS.items():
        for alias in sorted(set(aliases + [group]), key=len, reverse=True):
            escaped = re.escape(alias.lower()).replace(r"\ ", r"\s+")
            if re.search(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])", source):
                found.append(group)
                break
    return sorted(set(found))


def extract_keywords(text: str, top_k: int = 15, use_synonyms: bool = True) -> List[str]:
    """Prioritize skills detected across the complete text; retain useful fallback phrases."""
    skills = extract_skills(text)
    if skills:
        # Canonical skill categories are collected from the complete text; do not truncate them.
        return skills
    # Lightweight fallback avoids downloading a spaCy model at application startup.
    words = re.findall(r"\b[a-z][a-z0-9+#.-]{2,}\b", (text or "").lower())
    stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "was", "are", "you", "our", "their", "will", "years", "experience", "work", "team", "role", "responsibilities", "requirements"}
    freq = {}
    for word in words:
        if word not in stop:
            freq[word] = freq.get(word, 0) + 1
    return [word for word, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
