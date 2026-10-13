import re
from dataclasses import dataclass, field, asdict

DROP_TOKENS = {"ltd", "limited", "pvt", "private", "corporation", "corp", "inc", "co", "company"}

CITY_ALIASES = {
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "delhi": "New Delhi",
    "new delhi": "New Delhi",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "poonah": "Pune",
    "poona": "Pune",
    "pune": "Pune",
    "madras": "Chennai",
    "chennai": "Chennai",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "secunderabad": "Hyderabad",
    "hyderabad": "Hyderabad",
    "noida": "Noida",
    "thana": "Thane",
    "thane": "Thane",
}


@dataclass
class Record:
    source_type: str
    source_file: str
    invoice_id: str = ""
    vendor: str = ""
    amount: int = 0
    date: str = ""
    city: str = ""
    tax_id: str = ""
    confidence: float = 1.0
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def normalize_name(name):
    if not name:
        return ""
    words = re.split(r"[^a-z0-9]+", name.lower())
    return " ".join(word for word in words if word and word not in DROP_TOKENS)


def canon_city(city):
    if not city:
        return ""
    return CITY_ALIASES.get(city.lower().strip(), city.strip().title())


def parse_amount(value):
    if value is None:
        return 0
    text = str(value).lower()
    for token in ("inr", "rs", "rupees", ",", "amount payable:", "amount:", "total"):
        text = text.replace(token, "")
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0
