from dataclasses import dataclass, field, asdict

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
    text = name.lower().strip()
    for token in ("ltd.", "ltd", "limited", "pvt", "private", "corporation", "corp", "inc", "co", "company", "&", "."):
        text = text.replace(token, " ")
    return " ".join(text.split())


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
