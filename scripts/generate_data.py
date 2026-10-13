import csv
import glob
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont
from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
GOLD = os.path.join(ROOT, "data", "gold")

VENDORS = [
    ("Acme Ltd", "27AABCU9603R1ZM", "Mumbai", "Bombay"),
    ("Globex Corporation", "29AAGCG1234P1Z5", "Bengaluru", "Bangalore"),
    ("Initech Solutions", "07AAECI5678Q1Z2", "New Delhi", "Delhi"),
    ("Umbrella Traders", "19AAFCU9012R1Z8", "Kolkata", "Calcutta"),
    ("Wayne Enterprises", "33AABCW3456S1Z1", "Pune", "Poona"),
    ("Stark Industries", "24AADCS7890T1Z4", "Chennai", "Madras"),
    ("Daily Planet Media", "06AAECD2345U1Z7", "Gurugram", "Gurgaon"),
    ("Cyberdyne Systems", "36AAEEC6789V1Z3", "Hyderabad", "Secunderabad"),
    ("Soylent Foods", "09AAFCS1234W1Z6", "Noida", "Noida"),
    ("Hooli Services", "27AAGCH5678X1Z9", "Thane", "Thana"),
    ("Pied Piper", "29AAHCP1122Y1Z1", "Bengaluru", "Bangalore"),
    ("Aperture Labs", "07AAICA3344Z1Z4", "New Delhi", "Delhi"),
    ("Wonka Industries", "19AAJCW5566A1Z7", "Kolkata", "Calcutta"),
    ("Vandelay Industries", "33AAKCV7788B1Z2", "Pune", "Poona"),
    ("Dunder Mifflin", "24AALCD9900C1Z5", "Chennai", "Madras"),
    ("Sterling Cooper", "06AAMCS1234D1Z8", "Gurugram", "Gurgaon"),
    ("Oscorp", "36AANCO5678E1Z3", "Hyderabad", "Secunderabad"),
    ("Massive Dynamic", "09AAOCM9012F1Z6", "Noida", "Noida"),
    ("Tyrell Corp", "27AAPCT3456G1Z9", "Thane", "Thana"),
    ("InGen", "29AAQCI7890H1Z1", "Mumbai", "Bombay"),
    ("Zenith Traders", "27AABZK1122M1Z5", "Mumbai", "Bombay"),
    ("Zenith Trading Company", "29AACZL3344N1Z8", "Pune", "Poona"),
    ("Orion Systems", "07AADZO5566P1Z2", "New Delhi", "Delhi"),
    ("Orion Systematics", "33AAEZO7788Q1Z5", "Chennai", "Madras"),
    ("Vertex Analytics", "19AAFZV9900R1Z8", "Kolkata", "Calcutta"),
    ("Vertex Analysis", "36AAGZV2233S1Z1", "Hyderabad", "Secunderabad"),
    ("Nova Retail", "24AAHZN4455T1Z4", "Noida", "Noida"),
    ("Nova Retailers", "09AAIZN6677U1Z7", "Thane", "Thana"),
    ("Pioneer Logistics", "06AAJZP8899V1Z3", "Gurugram", "Gurgaon"),
    ("Pioneer Logistical", "27AAKZP1122W1Z6", "Bengaluru", "Bangalore"),
]

VARIANT_RULES = [
    lambda n: n,
    lambda n: n.upper(),
    lambda n: n.replace(" Ltd", " Limited"),
    lambda n: n.replace(" Corporation", " Corp"),
    lambda n: n.replace(" Solutions", ""),
    lambda n: n.replace(" Private", ""),
]


def font(size):
    for path in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def variant(name, mode):
    if mode == 0:
        return name
    if mode == 1:
        return name.upper()
    if mode == 2:
        return name.replace(" Ltd", " Limited").replace(" Corporation", " Corp")
    if mode == 3:
        return name.split()[0]
    return name


def bench_questions(canonical, unique, vendors_per_type=8):
    by_vendor = {}
    for t in unique.values():
        by_vendor.setdefault(t["vendor_id"], []).append(t)

    def spend(vid):
        return sum(t["amount"] for t in by_vendor.get(vid, []))

    selected = random.sample(canonical, min(vendors_per_type, len(canonical)))
    questions = []
    for v in selected:
        questions.append(
            {
                "q": "What is the total spend with {} in INR?".format(v["name"]),
                "answer": spend(v["vendor_id"]),
                "type": "sum",
            }
        )
        questions.append(
            {
                "q": "What is the total spend with {} in INR?".format(v["name"].upper()),
                "answer": spend(v["vendor_id"]),
                "type": "sum_upper_variant",
            }
        )
        questions.append(
            {
                "q": "How many transactions involve {}?".format(v["name"]),
                "answer": len(by_vendor.get(v["vendor_id"], [])),
                "type": "count",
            }
        )
        questions.append(
            {
                "q": "What city is {} registered in?".format(v["name"]),
                "answer": v["city"],
                "type": "city",
            }
        )
        questions.append(
            {
                "q": "What is the GSTIN of {}?".format(v["name"]),
                "answer": v["tax_id"],
                "type": "tax",
            }
        )
    questions.append(
        {"q": "How many distinct vendors appear in the data?", "answer": len(canonical), "type": "global_count"}
    )
    questions.append(
        {
            "q": "What is the total spend across all vendors in INR?",
            "answer": sum(t["amount"] for t in unique.values()),
            "type": "global_sum",
        }
    )
    random.shuffle(questions)
    return questions


def build(emit=None, seed=None):
    say = emit or (lambda s: None)
    if seed is None:
        seed = random.randrange(1, 1000000)
    random.seed(seed)
    say("generator seed {}".format(seed))
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(GOLD, exist_ok=True)

    for sub, pattern in (("invoices", "*.pdf"), ("receipts", "*.png")):
        folder = os.path.join(RAW, sub)
        os.makedirs(folder, exist_ok=True)
        for stale in glob.glob(os.path.join(folder, pattern)):
            os.remove(stale)

    canonical = []
    for idx, (name, tax, city, alt_city) in enumerate(VENDORS, start=1):
        canonical.append(
            {
                "vendor_id": "V{:03d}".format(idx),
                "name": name,
                "tax_id": tax,
                "city": city,
                "alt_city": alt_city,
            }
        )

    transactions = []
    inv = 0
    for v in canonical:
        count = random.randint(8, 12)
        for _ in range(count):
            inv += 1
            transactions.append(
                {
                    "invoice_id": "INV-{:04d}".format(inv),
                    "vendor_id": v["vendor_id"],
                    "amount": random.choice([12500, 24800, 50000, 7300, 96000, 15800, 34200]),
                    "date": "2025-{:02d}-{:02d}".format(random.randint(1, 12), random.randint(1, 28)),
                }
            )

    dup_targets = transactions[::2]
    for t in dup_targets:
        t["dup"] = True

    used_sources = {}
    for i, t in enumerate(transactions):
        used_sources[t["invoice_id"]] = "xlsx" if i % 2 == 0 else "pdf"
    for t in dup_targets:
        t["second"] = "pdf" if used_sources[t["invoice_id"]] == "xlsx" else "xlsx"

    conflicts = []

    by_vendor = {}
    for t in transactions:
        by_vendor.setdefault(t["vendor_id"], []).append(t["amount"])

    def amount_repeats(t):
        return sum(1 for a in by_vendor[t["vendor_id"]] if a == t["amount"]) > 1

    claimed = set()

    def pick(candidates, n):
        out = []
        for t in candidates:
            key = (t["vendor_id"], t["amount"])
            if key in claimed or not amount_repeats(t):
                continue
            claimed.add(key)
            out.append(t)
            if len(out) == n:
                break
        return out

    three_copy = pick((t for t in transactions if used_sources[t["invoice_id"]] == "xlsx"), 8)
    three_ids = {t["invoice_id"] for t in three_copy}

    two_copy = pick(
        (t for t in transactions if t.get("second") and t["invoice_id"] not in three_ids), 12
    )

    def inject(t, wrong_source, factor):
        wrong = t["amount"] * factor
        t["wrong"] = (wrong_source, wrong)
        conflicts.append(
            {
                "invoice_id": t["invoice_id"],
                "vendor_id": t["vendor_id"],
                "true_amount": t["amount"],
                "wrong_amount": wrong,
                "wrong_source": wrong_source,
                "copies": ["xlsx", "pdf", "receipt"] if t["invoice_id"] in three_ids else ["xlsx", "pdf"],
            }
        )

    for t in two_copy[:8]:
        inject(t, "pdf", 10)
    for t in two_copy[8:12]:
        inject(t, "xlsx", 10)
    for i, t in enumerate(three_copy):
        inject(t, "xlsx" if i % 3 == 0 else "pdf", 10)

    say(
        "injected {} cross-source conflicts, {} of them in the most reliable source".format(
            len(conflicts), sum(1 for c in conflicts if c["wrong_source"] == "xlsx")
        )
    )

    say("seed {} unique invoices, {} duplicate targets".format(len({t["invoice_id"] for t in transactions}), len(dup_targets)))

    def amount_for(t, source):
        wrong = t.get("wrong")
        return wrong[1] if wrong and wrong[0] == source else t["amount"]

    with open(os.path.join(RAW, "customers.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "company_name", "gstin", "registered_city"])
        for i, v in enumerate(canonical):
            w.writerow([v["vendor_id"], variant(v["name"], i % 3), v["tax_id"], v["alt_city"]])
    say("wrote customers.csv with {} vendor rows".format(len(canonical)))

    wb = Workbook()
    ws = wb.active
    ws.title = "ledger"
    ws.append(["invoice_id", "vendor", "gstin", "amount_inr", "date"])
    for t in transactions:
        if used_sources[t["invoice_id"]] == "xlsx" or t.get("second") == "xlsx":
            v = canonical[int(t["vendor_id"][1:]) - 1]
            ws.append([t["invoice_id"], variant(v["name"], 3), v["tax_id"], amount_for(t, "xlsx"), t["date"]])
    wb.save(os.path.join(RAW, "transactions.xlsx"))
    say("wrote transactions.xlsx ledger")

    pdfs = [t for t in transactions if used_sources[t["invoice_id"]] == "pdf" or t.get("second") == "pdf"]
    say("writing {} pdf invoices".format(len(pdfs)))
    for i, t in enumerate(pdfs, start=1):
        v = canonical[int(t["vendor_id"][1:]) - 1]
        path = os.path.join(RAW, "invoices", "{}.pdf".format(t["invoice_id"]))
        c = canvas.Canvas(path, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(60, 780, "TAX INVOICE")
        c.setFont("Helvetica", 11)
        c.drawString(60, 740, "Invoice No: {}".format(t["invoice_id"]))
        c.drawString(60, 720, "Supplier: {}".format(variant(v["name"], 2)))
        c.drawString(60, 700, "GSTIN: {}".format(v["tax_id"]))
        c.drawString(60, 680, "Place of Supply: {}".format(v["alt_city"]))
        c.drawString(60, 660, "Date: {}".format(t["date"]))
        c.drawString(60, 620, "Amount Payable: INR {}".format(amount_for(t, "pdf")))
        c.save()
        if i % 50 == 0 or i == len(pdfs):
            say("pdf {}/{} written".format(i, len(pdfs)))

    receipts = three_copy
    say("writing {} receipt images".format(len(receipts)))
    for t in receipts:
        v = canonical[int(t["vendor_id"][1:]) - 1]
        img = Image.new("RGB", (520, 300), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([10, 10, 510, 290], outline="black", width=2)
        lines = [
            "PAYMENT RECEIPT",
            "",
            "Vendor: {}".format(variant(v["name"], 1)),
            "Receipt No: {}".format(t["invoice_id"]),
            "Date: {}".format(t["date"]),
            "Amount: INR {}".format(amount_for(t, "receipt")),
            "",
            "Thank you for your business",
        ]
        y = 30
        for i, line in enumerate(lines):
            d.text((30, y), line, fill="black", font=font(22 if i == 0 else 18))
            y += 30
        img.save(os.path.join(RAW, "receipts", "{}.png".format(t["invoice_id"])))
    say("wrote {} receipt images".format(len(receipts)))

    gold_vendors = [
        {"vendor_id": v["vendor_id"], "name": v["name"], "tax_id": v["tax_id"], "city": v["city"]}
        for v in canonical
    ]
    unique = {t["invoice_id"]: t for t in transactions}
    total = sum(t["amount"] for t in unique.values())

    def spend(vid):
        return sum(t["amount"] for t in unique.values() if t["vendor_id"] == vid)

    questions = [
        {"q": "How many distinct vendors appear in the data?", "answer": len(canonical), "type": "number"},
        {"q": "What is the total spend across all vendors in INR?", "answer": total, "type": "number"},
        {"q": "What is the total spend with Acme Ltd in INR?", "answer": spend("V001"), "type": "number"},
        {"q": "What is the total spend with Globex Corporation in INR?", "answer": spend("V002"), "type": "number"},
        {"q": "What city is Acme Ltd registered in?", "answer": "Mumbai", "type": "text"},
        {"q": "What city is Globex Corporation registered in?", "answer": "Bengaluru", "type": "text"},
        {"q": "What is the GSTIN of Initech Solutions?", "answer": "07AAECI5678Q1Z2", "type": "text"},
        {"q": "How many transactions involve Stark Industries?", "answer": sum(1 for t in unique.values() if t["vendor_id"] == "V006"), "type": "number"},
        {"q": "What is the total spend with Hooli Services in INR?", "answer": spend("V010"), "type": "number"},
        {"q": "What city is Wayne Enterprises registered in?", "answer": "Pune", "type": "text"},
    ]

    with open(os.path.join(GOLD, "canonical_vendors.json"), "w", encoding="utf-8") as f:
        json.dump(gold_vendors, f, indent=2)
    with open(os.path.join(GOLD, "questions.json"), "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    with open(os.path.join(GOLD, "questions_bench.json"), "w", encoding="utf-8") as f:
        json.dump(bench_questions(canonical, unique), f, indent=2)
    with open(os.path.join(GOLD, "conflicts.json"), "w", encoding="utf-8") as f:
        json.dump(conflicts, f, indent=2)
    with open(os.path.join(GOLD, "invoice_truth.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "invoice_id": t["invoice_id"],
                    "vendor_id": t["vendor_id"],
                    "amount": t["amount"],
                    "date": t["date"],
                }
                for t in sorted(unique.values(), key=lambda t: t["invoice_id"])
            ],
            f,
            indent=2,
        )
    with open(os.path.join(GOLD, "receipts.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "invoice_id": t["invoice_id"],
                    "vendor": variant(canonical[int(t["vendor_id"][1:]) - 1]["name"], 1),
                    "amount": amount_for(t, "receipt"),
                    "date": t["date"],
                }
                for t in receipts
            ],
            f,
            indent=2,
        )

    raw_rows = sum(1 for t in transactions if used_sources[t["invoice_id"]] == "xlsx" or t.get("second") == "xlsx")
    pdf_rows = sum(1 for t in transactions if used_sources[t["invoice_id"]] == "pdf" or t.get("second") == "pdf")
    with open(os.path.join(GOLD, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"seed": seed, "vendors": len(canonical), "unique_invoices": len(unique), "raw_rows": raw_rows + pdf_rows},
            f,
            indent=2,
        )
    print("vendors:", len(canonical))
    print("unique invoices:", len(unique))
    print("raw transaction rows written (xlsx+pdf):", raw_rows + pdf_rows)
    print("receipts:", len(receipts))
    print("duplicate transactions:", len(dup_targets))
    print("gold total spend:", total)
    say("wrote gold/canonical_vendors.json and gold/questions.json")
    say("generate done: {} vendors, {} unique invoices, {} raw rows".format(len(canonical), len(unique), raw_rows + pdf_rows))
    return len(canonical), len(unique), raw_rows + pdf_rows


if __name__ == "__main__":
    build()
