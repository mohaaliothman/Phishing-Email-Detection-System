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
        [weekday_sin, we_]()
