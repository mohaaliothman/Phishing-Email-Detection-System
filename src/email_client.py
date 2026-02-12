"""
email_client.py
---------------
Reads the latest unseen emails from Gmail (IMAP), extracts features, loads your saved
ML pipeline (phishing_detection_pipeline.pkl), predicts phishing probability,
and sends alert emails (SMTP) for suspicious messages.

Requirements:
- pip install beautifulsoup4
- Enable Gmail App Password (recommended) and set env vars:
    PHISHING_EMAIL=your_email@gmail.com
    PHISHING_APP_PASSWORD=your_app_password
"""

import os
import re
import imaplib
import email
import smtplib
import pickle
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from email.message import EmailMessage
from email.header import decode_header
from email.utils import parsedate_to_datetime

import torch
import torch.nn as nn


# =========================
# 0) Config
# =========================
EMAIL_ADDR = os.environ.get("PHISHING_EMAIL", "").strip()
APP_PASSWORD = os.environ.get("PHISHING_APP_PASSWORD", "").strip()

IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

N_UNSEEN = int(os.environ.get("PHISHING_N_UNSEEN", "5"))

# Risk thresholds
HIGH_THRESHOLD = float(os.environ.get("PHISHING_HIGH_THRESHOLD", "0.85"))
MEDIUM_THRESHOLD = float(os.environ.get("PHISHING_MEDIUM_THRESHOLD", "0.60"))

# Filepaths (put the pkl inside project root or adjust)
PIPELINE_PATH = os.environ.get("PHISHING_PIPELINE_PATH", "phishing_detection_pipeline.pkl")


# =========================
# 1) Helper functions
# =========================
def decode_mime_header(value: str) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    out = ""
    for part, enc in parts:
        if isinstance(part, bytes):
            try:
                out += part.decode(enc or "utf-8", errors="ignore")
            except Exception:
                out += part.decode("utf-8", errors="ignore")
        else:
            out += part
    return out.strip()


def clean_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def extract_links(text: str):
    return re.findall(r"https?://[^\s]+", text or "")


def safe_parse_date(date_str: str):
    try:
        dt = parsedate_to_datetime(date_str)
        return dt
    except Exception:
        return None


def send_alert_email(subject: str, body: str):
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDR
    msg["To"] = EMAIL_ADDR
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(EMAIL_ADDR, APP_PASSWORD)
        server.send_message(msg)


# =========================
# 2) Keyword lists (same as your notebook)
# =========================
URGENT_KEYWORDS = [
    "urgent", "urgently", "immediately", "immediate", "immediate action",
    "respond immediately", "important", "high importance", "high priority",
    "priority", "action required", "act now", "respond now", "asap",
    "verify now", "confirm now", "update now", "review now",
    "expire", "expires", "expires today", "expiring soon", "last warning",
    "limited time", "within 24 hours", "within 48 hours", "within hours",
    "time is running out", "offer expires", "today only",
    "urgent response needed", "critical update", "attention required",
    "final reminder", "final warning", "immediate response needed",
    "deadline", "due today", "due now", "time-sensitive", "time sensitive",
    "respond asap", "quickly", "fast action required", "take action now",
    "do this immediately", "important notice", "emergency action",
    "critical issue", "must act now", "respond without delay",
    "urgent attention", "requires immediate action"
]

THREAT_KEYWORDS = [
    "suspend", "suspended", "locked", "locked out", "close your account",
    "account closure", "legal action", "legal notice", "final notice",
    "termination", "deactivate", "deactivation", "restricted", "restriction",
    "blocked", "frozen", "compromised", "breach", "security breach",
    "hacked", "unauthorized", "unusual activity",
    "violation", "policy violation", "will be closed", "forced closure",
    "lose access", "permanently removed",
    "account disabled", "security suspension", "threat detected",
    "security risk", "security warning", "final attempt", "failure to respond",
    "account will be terminated", "forced deactivation", "breach detected",
    "critical security issue", "account compromised", "service interruption",
    "security alert", "high-risk login", "multiple failed attempts"
]

MONEY_PRIZE_KEYWORDS = [
    "win", "winner", "won", "prize", "grand prize", "cash", "cash prize",
    "reward", "bonus", "lottery", "jackpot", "million", "million dollars",
    "refund", "tax refund", "tax rebate", "rebate", "unclaimed",
    "compensation", "inheritance", "beneficiary", "grant", "funds",
    "payment released", "claim your", "claim now", "redeem now",
    "congratulations you", "free money", "you've been selected",
    "$$$", "financial award", "monetary reward", "payout", "transfer",
    "urgent refund", "deposit available", "funds available",
    "cash transfer", "unexpected funds", "reward waiting", "prize awaiting",
    "selected as winner", "lucky draw", "special payout",
    "exclusive reward", "free bonus", "instant winnings"
]

URGENCY_PHRASES = [
    "act now", "don't wait", "hurry", "last chance", "time is running out",
    "offer expires", "today only", "now or never", "don't miss out",
    "limited offer", "limited-time", "while supplies last",
    "urgent response needed", "expires soon", "limited time", "dont miss",
    "final opportunity", "respond quickly", "immediate attention required",
    "before it’s too late", "claim before expiry",
    "offer ends soon", "only hours left", "only today",
    "limited quantity", "final hours", "respond before deadline"
]

GENERIC_SUSPICIOUS = [
    "click here", "click below", "click link", "open link", "login",
    "log in", "update", "update account", "verify", "verify your",
    "confirm", "confirm your", "validate", "review", "reactivate",
    "password", "security alert", "dear customer",
    "dear user", "valued customer", "account holder", "update payment",
    "billing information", "payment method", "credit card", "card details",
    "reset password", "identity verification", "authentication required",
    "verify immediately", "important update", "account review",
    "confirm identity", "provide details", "submit information",
    "resolve issue", "confirm account", "login required",
    "unlock account", "secure login",
    "update credentials", "verification process", "restore access",
    "protect your account", "account verification"
]

COMPANY_IMPERSONATION = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "instagram", "twitter", "linkedin", "bank", "irs", "fedex", "usps",
    "ups", "dhl", "netflix", "ebay",
    "wells fargo", "chase", "bank of america", "citibank", "hsbc",
    "visa", "mastercard", "amex",
    "office365", "outlook", "teams",
    "amazon support", "microsoft support", "google security",
    "bank security"
]

SPOOFING_INDICATORS = [
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "support@", "admin@", "security@", "verification@", "alert@",
    "service@", "info@", "notification@", "update@", "mailer@", "robot@",
    "helpdesk@", "accounts@", "billing@", "customerservice@", "system@"
]

CREDENTIAL_REQUESTS = [
    "enter your password", "provide your password", "confirm password",
    "password", "username and password", "username",
    "account number", "credit card", "card details", "cvv", "pin",
    "date of birth", "two-factor code", "otp", "one-time password",
    "authentication code", "verify identity",
    "enter your credentials", "provide login details", "reauthenticate",
    "submit credentials"
]

FREE_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "yandex.com", "gmx.com", "icloud.com",
    "live.com", "msn.com"
]

attachment_keywords = [
    "attached", "attachment", "find attached", "see attached", "enclosed",
    "invoice", "statement", "report", "form", "pdf", "doc", "docx",
    "xls", "xlsx", "ppt", "pptx", "zip", "rar"
]

personal_pronouns = [
    " i ", " me ", " my ", " mine ",
    " you ", " your ", " yours ",
    " we ", " us ", " our ",
    " he ", " him ", " his ",
    " she ", " her ", " hers ",
    " they ", " them ", " their "
]

action_words = [
    "click here", "verify now", "update now", "confirm now", "act now",
    "login", "log in", "sign in", "reset password",
    "download", "open attachment", "open file", "enable macros",
    "submit information", "provide details", "approve", "authenticate"
]

common_misspellings = [
    "urgnet", "accout", "pasword", "verifiy", "securty",
    "offical", "addres", "identitiy", "authenticaion",
    "loggin", "passwrod", "confim", "securrity"
]


# =========================
# 3) Feature extraction (same logic as your notebook)
# =========================
def parse_date_text(date_str):
    # robust fallback
    dt = safe_parse_date(date_str)
    if dt is None:
        weekday, day, month, year, hour, minute, second = 0, 1, 1, 2025, 0, 0, 0
    else:
        weekday, day, month, year = dt.weekday(), dt.day, dt.month, dt.year
        hour, minute, second = dt.hour, dt.minute, dt.second

    def cyclical(val, max_val):
        return (np.sin(2 * np.pi * val / max_val), np.cos(2 * np.pi * val / max_val))

    weekday_sin, weekday_cos = cyclical(weekday, 7)
    day_sin, day_cos = cyclical(day, 31)
    month_sin, month_cos = cyclical(month, 12)
    hour_sin, hour_cos = cyclical(hour, 24)
    minute_sin, minute_cos = cyclical(minute, 60)
    second_sin, second_cos = cyclical(second, 60)

    return pd.Series(
        [weekday_sin, weekday_cos, day_sin, day_cos, month_sin, month_cos,
         hour_sin, hour_cos, minute_sin, minute_cos, second_sin, second_cos, year],
        index=["weekday_sin","weekday_cos","day_sin","day_cos","month_sin","month_cos",
               "hour_sin","hour_cos","minute_sin","minute_cos","second_sin","second_cos","year"]
    )


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    features_df = df.copy()

    # normalize
    features_df["subject"] = features_df["subject"].fillna("").astype(str)
    features_df["body"] = features_df["body"].fillna("").astype(str)
    features_df["from"] = features_df["from"].fillna("").astype(str)
    features_df["cc"] = features_df["cc"].fillna("").astype(str)
    features_df["bcc"] = features_df["bcc"].fillna("").astype(str)
    features_df["url"] = features_df["url"].fillna("").astype(str)

    subject_lower = features_df["subject"].str.lower()
    body_lower = features_df["body"].str.lower()
    from_lower = features_df["from"].str.lower()
    urls_lower = features_df["url"].str.lower()
    url_str_col = features_df["url"]

    features_df["has_cc"] = (features_df["cc"].str.len() > 0).astype(int)
    features_df["has_bcc"] = (features_df["bcc"].str.len() > 0).astype(int)
    features_df["has_url"] = (features_df["url"].str.len() > 0).astype(int)

    features_df["cc_count"] = features_df["cc"].apply(lambda x: len([e for e in re.split(r"[,;]", x) if "@" in e]))
    features_df["bcc_count"] = features_df["bcc"].apply(lambda x: len([e for e in re.split(r"[,;]", x) if "@" in e]))

    features_df["total_recipients"] = features_df["cc_count"] + features_df["bcc_count"] + 1
    features_df["is_mass_email"] = (features_df["total_recipients"] > 5).astype(int)

    features_df["subject_length"] = features_df["subject"].apply(len)
    features_df["subject_word_count"] = features_df["subject"].apply(lambda x: len(x.split()))

    features_df["subject_urgent_keywords"] = subject_lower.apply(lambda x: sum(1 for kw in URGENT_KEYWORDS if kw in x))
    features_df["subject_threat_keywords"] = subject_lower.apply(lambda x: sum(1 for kw in THREAT_KEYWORDS if kw in x))
    features_df["subject_money_keywords"] = subject_lower.apply(lambda x: sum(1 for kw in MONEY_PRIZE_KEYWORDS if kw in x))
    features_df["subject_generic_keywords"] = subject_lower.apply(lambda x: sum(1 for kw in GENERIC_SUSPICIOUS if kw in x))
    features_df["subject_company_keywords"] = subject_lower.apply(lambda x: sum(1 for kw in COMPANY_IMPERSONATION if kw in x))

    features_df["subject_excessive_punctuation"] = features_df["subject"].apply(lambda x: int(bool(re.search(r"!{2,}|\?{2,}|\${2,}", x))))
    features_df["subject_exclamation_count"] = features_df["subject"].apply(lambda x: x.count("!"))
    features_df["subject_question_count"] = features_df["subject"].apply(lambda x: x.count("?"))
    features_df["subject_dollar_count"] = features_df["subject"].apply(lambda x: x.count("$"))

    features_df["subject_all_caps_ratio"] = features_df["subject"].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    features_df["subject_has_re_fwd"] = subject_lower.apply(lambda x: int(bool(re.match(r"^(re:|fw:|fwd:)", x))))

    features_df["sender_is_freemail"] = from_lower.apply(lambda x: int(any(dom in x for dom in FREE_EMAIL_DOMAINS)))
    features_df["sender_has_numbers"] = features_df["from"].apply(lambda x: int(bool(re.search(r"\d", x))))
    features_df["sender_number_count"] = features_df["from"].apply(lambda x: len(re.findall(r"\d", x)))
    features_df["sender_length"] = features_df["from"].apply(len)

    features_df["sender_has_spoofing_indicator"] = from_lower.apply(lambda x: int(any(ind in x for ind in SPOOFING_INDICATORS)))
    features_df["sender_multiple_separators"] = features_df["from"].apply(lambda x: int(bool(re.search(r"\.{2,}|-{2,}|_{2,}", x))))

    features_df["body_length"] = features_df["body"].apply(len)
    features_df["body_word_count"] = features_df["body"].apply(lambda x: len(x.split()))

    features_df["body_urgent_keywords"] = body_lower.apply(lambda x: sum(1 for kw in URGENT_KEYWORDS if kw in x))
    features_df["body_threat_keywords"] = body_lower.apply(lambda x: sum(1 for kw in THREAT_KEYWORDS if kw in x))
    features_df["body_money_keywords"] = body_lower.apply(lambda x: sum(1 for kw in MONEY_PRIZE_KEYWORDS if kw in x))
    features_df["body_generic_keywords"] = body_lower.apply(lambda x: sum(1 for kw in GENERIC_SUSPICIOUS if kw in x))
    features_df["body_company_keywords"] = body_lower.apply(lambda x: sum(1 for kw in COMPANY_IMPERSONATION if kw in x))
    features_df["body_urgency_phrases"] = body_lower.apply(lambda x: sum(1 for ph in URGENCY_PHRASES if ph in x))
    features_df["body_credential_requests"] = body_lower.apply(lambda x: sum(1 for kw in CREDENTIAL_REQUESTS if kw in x))

    features_df["body_excessive_punctuation"] = features_df["body"].apply(lambda x: int(bool(re.search(r"!{2,}|\?{2,}|\${2,}", x))))
    features_df["body_exclamation_count"] = features_df["body"].apply(lambda x: x.count("!"))
    features_df["body_question_count"] = features_df["body"].apply(lambda x: x.count("?"))
    features_df["body_dollar_count"] = features_df["body"].apply(lambda x: x.count("$"))

    features_df["body_has_html"] = features_df["body"].apply(lambda x: int(bool(re.search(r"<[^>]+>", x))))
    features_df["body_has_misspellings"] = body_lower.apply(lambda x: int(any(w in x for w in common_misspellings)))

    features_df["url_count"] = url_str_col.apply(lambda x: len(re.findall(r"https?://", x)))

    shorteners = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"]
    features_df["url_is_shortened"] = urls_lower.apply(lambda x: int(any(s in x for s in shorteners)))

    features_df["url_avg_length"] = url_str_col.apply(
        lambda x: (sum(len(u) for u in re.findall(r"https?://[^\s]+", x)) /
                   max(1, len(re.findall(r"https?://", x))))
    )

    features_df["url_many_subdomains"] = url_str_col.apply(
        lambda x: int(any(len(re.findall(r"\.", u.split("/")[2])) > 3 for u in re.findall(r"https?://[^\s]+", x) if "/" in u))
    )
    features_df["url_has_at_symbol"] = url_str_col.apply(lambda x: int("@" in x and "http" in x))

    features_df["sender_company_mismatch"] = (
        (features_df["sender_is_freemail"] == 1) &
        ((features_df["subject_company_keywords"] > 0) | (features_df["body_company_keywords"] > 0))
    ).astype(int)

    features_df["total_urgent_keywords"] = features_df["subject_urgent_keywords"] + features_df["body_urgent_keywords"]
    features_df["total_threat_keywords"] = features_df["subject_threat_keywords"] + features_df["body_threat_keywords"]
    features_df["total_money_keywords"] = features_df["subject_money_keywords"] + features_df["body_money_keywords"]
    features_df["total_generic_keywords"] = features_df["subject_generic_keywords"] + features_df["body_generic_keywords"]
    features_df["total_company_keywords"] = features_df["subject_company_keywords"] + features_df["body_company_keywords"]

    features_df["urgent_keyword_ratio"] = features_df["total_urgent_keywords"] / (features_df["body_word_count"] + 1)
    features_df["threat_keyword_ratio"] = features_df["total_threat_keywords"] / (features_df["body_word_count"] + 1)
    features_df["money_keyword_ratio"] = features_df["total_money_keywords"] / (features_df["body_word_count"] + 1)
    features_df["generic_keyword_ratio"] = features_df["total_generic_keywords"] / (features_df["body_word_count"] + 1)

    features_df["url_to_text_ratio"] = features_df["url_count"] / (features_df["body_word_count"] + 1)
    features_df["is_reply_or_forward"] = features_df["subject_has_re_fwd"]

    features_df["avg_word_length"] = features_df["body"].apply(lambda x: sum(len(w) for w in x.split()) / max(1, len(x.split())))
    features_df["body_uppercase_ratio"] = features_df["body"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(1, len(x)))
    features_df["body_special_char_count"] = features_df["body"].apply(lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", x)))

    features_df["mentions_attachment"] = body_lower.apply(lambda x: int(any(kw in x for kw in attachment_keywords)))
    features_df["body_personal_pronoun_count"] = body_lower.apply(lambda x: sum(1 for pr in personal_pronouns if pr in (" " + x + " ")))
    features_df["body_action_request_count"] = body_lower.apply(lambda x: sum(1 for act in action_words if act in x))

    features_df["has_signature"] = body_lower.apply(lambda x: int(bool(re.search(r"(regards|sincerely|best wishes|thanks|thank you)", x))))
    features_df["body_has_phone"] = features_df["body"].apply(lambda x: int(bool(re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", x))))
    features_df["body_is_very_short"] = (features_df["body_word_count"] < 10).astype(int)

    features_df["subject_body_overlap"] = features_df.apply(
        lambda row: len(set(row["subject"].lower().split()) & set(row["body"].lower().split())) / max(1, len(row["subject"].split())),
        axis=1
    )

    features_df["link_text_no_url"] = ((features_df["url_count"] == 0) & body_lower.apply(lambda x: ("click" in x) or ("link" in x) or ("http" in x))).astype(int)

    # date cyclical
    features_df[[
        "weekday_sin","weekday_cos","day_sin","day_cos","month_sin","month_cos",
        "hour_sin","hour_cos","minute_sin","minute_cos","second_sin","second_cos","year"
    ]] = features_df["date"].apply(parse_date_text)

    return features_df


# =========================
# 4) Model class (must match training)
# =========================
class PhishingDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(self.relu(self.bn4(self.fc4(x))))
        x = self.sigmoid(self.fc5(x))
        return x


# =========================
# 5) Pipeline loader (compatible with your saved pkl)
# =========================
class PhishingDetectionPipeline:
    def __init__(self):
        self.tfidf_subject = None
        self.tfidf_body = None
        self.scaler = None
        self.svd = None
        self.kmeans = None
        self.model = None
        self.subject_weight = 2
        self.numeric_features = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def load(cls, filepath: str, model_instance: nn.Module):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        pipe = cls()
        pipe.tfidf_subject = data["tfidf_subject"]
        pipe.tfidf_body = data["tfidf_body"]
        pipe.scaler = data["scaler"]
        pipe.svd = data["svd"]
        pipe.kmeans = data["kmeans"]
        pipe.subject_weight = data.get("subject_weight", 2)
        pipe.numeric_features = data.get("numeric_features", [])

        pipe.model = model_instance
        pipe.model.load_state_dict(data["model_state_dict"])
        pipe.model.to(pipe.device)
        pipe.model.eval()
        return pipe

    def transform_features(self, df: pd.DataFrame):
        # TF-IDF
        X_subject = self.tfidf_subject.transform(df["subject"])
        X_subject_weighted = X_subject * self.subject_weight

        X_body = self.tfidf_body.transform(df["body"])

        # combine
        X_tfidf = np.hstack([X_subject_weighted.toarray(), X_body.toarray()])

        # numeric
        X_other = df[self.numeric_features].copy()
        X_other_scaled = self.scaler.transform(X_other)

        # SVD on tfidf
        X_tfidf_reduced = self.svd.transform(X_tfidf)

        # final combined
        X_combined = np.hstack([X_tfidf_reduced, X_other_scaled])

        # clustering
        cluster_labels = self.kmeans.predict(X_combined)

        return X_combined, cluster_labels

    def predict_proba(self, features_df: pd.DataFrame):
        X_combined, clusters = self.transform_features(features_df)

        # IMPORTANT: you used cluster as a feature during training
        X_with_cluster = np.hstack([X_combined, clusters.reshape(-1, 1)])

        X_tensor = torch.FloatTensor(X_with_cluster).to(self.device)
        with torch.no_grad():
            probs = self.model(X_tensor).squeeze().cpu().numpy()

        return probs, clusters


# =========================
# 6) Read unseen emails from Gmail
# =========================
def fetch_unseen_emails(n_unseen: int) -> pd.DataFrame:
    mail = imaplib.IMAP4_SSL(IMAP_HOST)
    mail.login(EMAIL_ADDR, APP_PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, "UNSEEN")
    if status != "OK":
        mail.logout()
        return pd.DataFrame([])

    email_ids = messages[0].split()[-n_unseen:]
    emails_data = []

    for e_id in email_ids:
        _, msg_data = mail.fetch(e_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        subject = decode_mime_header(msg.get("Subject", ""))
        sender = decode_mime_header(msg.get("From", ""))
        to = decode_mime_header(msg.get("To", ""))
        cc = decode_mime_header(msg.get("Cc", ""))
        bcc = decode_mime_header(msg.get("Bcc", ""))
        date_str = msg.get("Date", "")

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                text = payload.decode(errors="ignore")

                ctype = part.get_content_type()
                if ctype == "text/plain":
                    body += text + "\n"
                elif ctype == "text/html":
                    body += clean_html(text) + "\n"
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = clean_html(payload.decode(errors="ignore"))

        urls = extract_links(body)

        emails_data.append({
            "subject": subject,
            "from": sender,
            "to": to,
            "cc": cc,
            "bcc": bcc,
            "body": body,
            "date": date_str,
            "url": " ".join(urls),
        })

        # mark as seen
        mail.store(e_id, "+FLAGS", "\\Seen")

    mail.logout()
    return pd.DataFrame(emails_data)


# =========================
# 7) Main
# =========================
def main():
    if not EMAIL_ADDR or not APP_PASSWORD:
        raise ValueError("Missing env vars: PHISHING_EMAIL and/or PHISHING_APP_PASSWORD")

    print(f"[+] Fetching up to {N_UNSEEN} unseen emails...")
    raw_df = fetch_unseen_emails(N_UNSEEN)

    if raw_df.empty:
        print("[i] No unseen emails found.")
        return

    print(f"[+] Fetched {len(raw_df)} emails. Extracting features...")
    feats = extract_features(raw_df)

    # Load pipeline
    print(f"[+] Loading pipeline from: {PIPELINE_PATH}")
    
    candidate_dims = [182, 183, 181, 184]
    last_err = None
    pipeline = None

    for dim in candidate_dims:
        try:
            model = PhishingDetector(dim)
            pipeline = PhishingDetectionPipeline.load(PIPELINE_PATH, model_instance=model)
            # quick dry-run on 1 row to validate shape
            _ = pipeline.predict_proba(feats.head(1))
            print(f"[✓] Loaded pipeline with model input_dim={dim}")
            break
        except Exception as e:
            last_err = e
            pipeline = None

    if pipeline is None:
        raise RuntimeError(f"Failed to load pipeline / shape mismatch. Last error: {last_err}")

    print("[+] Predicting phishing probabilities...")
    probs, clusters = pipeline.predict_proba(feats)

    # Decide + alert
    alerts_sent = 0
    for i, row in raw_df.iterrows():
        p = float(probs[i])

        if p < MEDIUM_THRESHOLD:
            continue

        if p >= HIGH_THRESHOLD:
            subj = "🚨 HIGH RISK PHISHING EMAIL"
            risk = "HIGH"
        else:
            subj = "⚠️ SUSPICIOUS EMAIL DETECTED"
            risk = "MEDIUM"

        body = (
            f"From: {row['from']}\n"
            f"Subject: {row['subject']}\n\n"
            f"Phishing Probability: {p:.3f}\n"
            f"Risk Level: {risk}\n"
            f"Cluster: {int(clusters[i])}\n\n"
            f"Extracted URLs:\n{row['url']}\n"
        )

        send_alert_email(subj, body)
        alerts_sent += 1

    print(f"[✓] Done. Alerts sent: {alerts_sent} / {len(raw_df)}")


if __name__ == "__main__":
    main()
