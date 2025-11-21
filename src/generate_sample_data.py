# src/generate_sample_data.py
import csv, random, os

random.seed(42)

rows = [
    ("Starbucks 123", "dining"),
    ("AMAZON MKTPLACE PMTS", "shopping"),
    ("SHELL OIL 456", "fuel"),
    ("WALMART SUPERCENTER", "groceries"),
    ("ELECTRICITY BILLNEW", "utilities"),
    ("UBER TRIP HELP.UBER.COM", "other"),
    ("Starbucks Store #45", "dining"),
    ("Amazon Web Services", "shopping"),
    ("SHELL GAS STATION", "fuel"),
    ("Whole Foods Market", "groceries"),
    ("MOVIE THEATRE TICKET", "entertainment"),
    ("CINEMA - IMAX", "entertainment"),
    ("NETFLIX.COM", "entertainment"),
    ("UDACITY COURSE FEE", "education"),
    ("ONLINE COURSE - COURSERA", "education"),
    ("GENERAL HOSPITAL PAYMENT", "healthcare"),
    ("CVS PHARMACY", "healthcare"),
    ("DOCTOR CONSULTATION", "healthcare"),
]


# expand with noisy variants
def noisy(t):
    variants = [t.lower(), t.upper(), t + " POS", t.replace(" ", "  "), t + " #4455"]
    return random.choice(variants)


if not os.path.exists("data/raw"):
    os.makedirs("data/raw")

with open("data/raw/sample_transactions.csv", "w", newline="", encoding="utf8") as f:
    writer = csv.writer(f)
    writer.writerow(["transaction", "label"])
    for _ in range(2000):  # generate many examples by perturbation
        t, lbl = random.choice(rows)
        writer.writerow([noisy(t), lbl])
print("Wrote data/raw/sample_transactions.csv")
