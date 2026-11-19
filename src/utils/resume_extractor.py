"""Structured resume feature extraction with conservative, explainable rules."""
from __future__ import annotations
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_TOKEN = rf"(?:{MONTH}\s+)?(?:19\d{{2}}|20\d{{2}})"
RANGE_RE = re.compile(rf"(?P<start>{DATE_TOKEN})\s*(?:-|–|—|to)\s*(?P<end>present|current|now|{DATE_TOKEN})", re.I)


def _year_month(value: str) -> Tuple[int, int]:
    value = value.strip().lower()
    year_match = re.search(r"(19\d{2}|20\d{2})", value)
    year = int(year_match.group(1)) if year_match else datetime.now().year
    month = 1
    if value and not value[:1].isdigit():
        token = value.split()[0][:3]
        month = {m: i for i, m in enumerate(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], 1)}.get(token, 1)
    return year, month


def _merge_intervals(intervals: List[Tuple[int, int]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    merged = []
    start, end = intervals[0]
    for s, e in intervals[1:]:
        if s <= end:
            end = max(end, e)
        else:
            merged.append((start, end)); start, end = s, e
    merged.append((start, end))
    months = sum(max(0, e - s) for s, e in merged)
    return round(months / 12.0, 1)


def extract_experience_years(text: str) -> Optional[float]:
    """Prefer explicit experience statements; otherwise merge work-history ranges."""
    if not text:
        return None
    lower = text.lower()
    explicit = [
        r"(?<!\w)(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:professional\s+)?(?:work\s+)?experience\b",
        r"\bexperience\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
    ]
    for pattern in explicit:
        m = re.search(pattern, lower)
        if m:
            years = float(m.group(1))
            if 0 <= years <= 50:
                return years
    intervals = []
    for match in RANGE_RE.finditer(text):
        # Avoid education date spans such as "2019 - 2023 B.Tech".
        line_start = lower.rfind("\n", 0, match.start()) + 1
        line_end = lower.find("\n", match.end())
        if line_end < 0:
            line_end = len(lower)
        context = lower[max(line_start, match.start()-65):match.start()]
        tail = lower[match.end():min(line_end, match.end()+45)]
        if re.search(r"\b(education|academic|university|college|b\.?(?:tech|sc|e)|m\.?(?:tech|sc|e)|bachelor|master|degree)\b", context):
            continue
        if re.match(r"\s*(?:b\.?tech|m\.?tech|bachelor|master|degree|university|college)", tail):
            continue
        sy, sm = _year_month(match.group("start"))
        if match.group("end").lower() in {"present", "current", "now"}:
            ey, em = datetime.now().year, datetime.now().month
        else:
            ey, em = _year_month(match.group("end"))
        if 1950 <= sy <= datetime.now().year and (ey, em) >= (sy, sm):
            intervals.append((sy * 12 + sm - 1, ey * 12 + em - 1))
    return _merge_intervals(intervals) if intervals else None


def _degree_present(text: str, degree: str) -> bool:
    # Short abbreviations are accepted only in uppercase or with punctuation/context.
    patterns = {
        "phd": [r"\bph\.?d\.?\b", r"\bdoctor(?:ate|al)\b"],
        "masters": [r"\bmaster(?:'s)?\s+(?:degree|of\s+[a-z])", r"\bmasters\s+degree\b", r"\bm\.?s\.?c?\b(?=\s+(?:in|degree|,|$))", r"\bm\.?tech\b", r"\bmba\b", r"\bmca\b"],
        "bachelors": [r"\bbachelor(?:'s|s)?\b", r"\bb\.?tech\b", r"\bb\.?sc\.?\b", r"\bbca\b", r"\bbcom\b"],
        "diploma": [r"\bdiploma\b"],
    }
    for pattern in patterns.get(degree, []):
        if re.search(pattern, text, re.I):
            return True
    # Bare BE/BS/MS/BA are highly ambiguous; require uppercase source spelling and nearby degree context.
    if degree == "bachelors" and re.search(r"(?<![A-Za-z])B\.?E\.?(?![A-Za-z])(?=\s+(?:degree|in|computer|mechanical|electrical|civil))", text):
        return True
    return False


def extract_education(text: str) -> Dict[str, bool]:
    return {level: _degree_present(text or "", level) for level in ["phd", "masters", "bachelors", "diploma"]}


def extract_certifications(text: str) -> List[str]:
    if not text:
        return []
    patterns = [
        r"\bAWS Certified [A-Za-z0-9][A-Za-z0-9 +\-]{2,55}",
        r"\bMicrosoft Certified:?\s+[A-Za-z0-9][A-Za-z0-9 +\-]{2,55}",
        r"\bGoogle Cloud Certified [A-Za-z0-9][A-Za-z0-9 +\-]{2,55}",
        r"\b(?:PMP|CISSP|CEH|CCNA|CCNP|OCI|CKA|CKAD)\b",
        r"\bCertified Kubernetes Administrator\b",
    ]
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = re.sub(r"\s+", " ", match.group(0)).strip(" .,-")
            if value and value.lower() not in {x.lower() for x in found}:
                found.append(value)
    return found[:20]


def extract_recent_roles(text: str, top_n: int = 5) -> List[str]:
    if not text:
        return []
    roles = []
    title_pattern = re.compile(r"\b(data analyst|data scientist|machine learning engineer|software engineer|developer|business analyst|project manager|product manager|systems analyst|devops engineer|cloud engineer|intern|research assistant|technical lead|team lead)\b", re.I)
    for match in title_pattern.finditer(text):
        role = match.group(0).title()
        if role.lower() not in {r.lower() for r in roles}:
            roles.append(role)
    return roles[:top_n]


def extract_all_resume_features(text: str) -> Dict:
    return {"experience_years": extract_experience_years(text), "education": extract_education(text), "certifications": extract_certifications(text), "recent_roles": extract_recent_roles(text)}
