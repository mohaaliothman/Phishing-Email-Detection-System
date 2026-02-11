import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime

# 1) Keyword lists
# ======================
URGENT_KEYWORDS = [
    "urgent" , "urgently" , "immediately" , "immediate" , "immediate action" ,
    "respond immediately" , "important" , "high importance" , "high priority" ,
    "priority" , "action required" , "act now" , "respond now" , "asap" ,
    "verify now" , "confirm now" , "update now" , "review now" ,
    "expire" , "expires" , "expires today" , "expiring soon" , "last warning" ,
    "limited time" , "within 24 hours" , "within 48 hours" , "within hours" ,
    "time is running out" , "offer expires" , "today only" ,
    "urgent response needed" , "critical update" , "attention required" ,
    "final reminder" , "final warning" , "immediate response needed" ,
    "deadline" , "due today" , "due now" , "time-sensitive" , "time sensitive" ,
    "respond asap" , "quickly" , "fast action required" , "take action now" ,
    "do this immediately" , "important notice" , "emergency action" ,
    "critical issue" , "must act now" , "respond without delay" ,
    "urgent attention" , "requires immediate action"
]

THREAT_KEYWORDS = [
    "suspend" , "suspended" , "locked" , "locked out" , "close your account" ,
    "account closure" , "legal action" , "legal notice" , "final notice" ,
    "termination" , "deactivate" , "deactivation" , "restricted" , "restriction" ,
    "blocked" , "frozen" , "compromised" , "breach" , "security breach" ,
    "hacked" , "unauthorized" , "unauthorized access" , "unusual activity" ,
    "violation" , "policy violation" , "will be closed" , "forced closure" ,
    "take action or" , "lose access" , "permanently removed" ,
    "account disabled" , "security suspension" , "threat detected" ,
    "security risk" , "security warning" , "final attempt" , "failure to respond" ,
    "account will be terminated" , "forced deactivation" , "breach detected" ,
    "critical security issue" , "account compromised" , "service interruption" ,
    "security alert" , "high-risk login" , "multiple failed attempts"
]

MONEY_PRIZE_KEYWORDS = [
    "win" , "winner" , "won" , "prize" , "grand prize" , "cash" , "cash prize" ,
    "reward" , "bonus" , "lottery" , "jackpot" , "million" , "million dollars" ,
    "refund" , "tax refund" , "tax rebate" , "rebate" , "unclaimed" ,
    "compensation" , "inheritance" , "beneficiary" , "grant" , "funds" ,
    "payment released" , "claim your" , "claim now" , "redeem now" ,
    "congratulations you" , "free money" , "you've been selected" ,
    "$$$" , "financial award" , "monetary reward" , "payout" , "transfer" ,
    "urgent refund" , "deposit available" , "funds available" ,
    "cash transfer" , "unexpected funds" , "reward waiting" , "prize awaiting" ,
    "selected as winner" , "lucky draw" , "special payout" ,
    "exclusive reward" , "free bonus" , "instant winnings"
]

URGENCY_PHRASES = [
    "act now" , "don't wait" , "hurry" , "last chance" , "time is running out" ,
    "offer expires" , "today only" , "now or never" , "don't miss out" ,
    "limited offer" , "limited-time" , "while supplies last" ,
    "urgent response needed" , "expires soon" , "limited time" , "dont miss" ,
    "final opportunity" , "respond quickly" , "immediate attention required" ,
    "before it’s too late" , "claim before expiry" ,
    "offer ends soon" , "only hours left" , "only today" ,
    "limited quantity" , "final hours" , "respond before deadline"
]

GENERIC_SUSPICIOUS = [
    "click here" , "click below" , "click link" , "open link" , "login" ,
    "log in" , "update" , "update account" , "verify" , "verify your" ,
    "confirm" , "confirm your" , "validate" , "review" , "reactivate" ,
    "re-activate" , "password" , "security alert" , "dear customer" ,
    "dear user" , "valued customer" , "account holder" , "update payment" ,
    "billing information" , "payment method" , "credit card" , "card details" ,
    "reset password" , "identity verification" , "authentication required" ,
    "verify immediately" , "important update" , "account review" ,
    "confirm identity" , "provide details" , "submit information" ,
    "resolve issue" , "confirm account" , "login required" ,
    "reset your account" , "unlock account" , "secure login" ,
    "update credentials" , "verification process" , "restore access" ,
    "protect your account" , "account verification"
]

COMPANY_IMPERSONATION = [
    "paypal" , "amazon" , "apple" , "microsoft" , "google" , "facebook" ,
    "instagram" , "twitter" , "linkedin" , "bank" , "irs" , "fedex" , "usps" ,
    "ups" , "dhl" , "netflix" , "ebay" , "ssa" , "social security" ,
    "wells fargo" , "chase" , "bank of america" , "citibank" , "hsbc" ,
    "capital one" , "barclays" , "royal mail" , "revolut" , "stripe" ,
    "coinbase" , "binance" , "mcdonalds" , "spotify" , "icloud" ,
    "adobe" , "dropbox" , "onedrive" , "office365" , "outlook" , "teams" ,
    "zoho" , "intuit" , "quickbooks" ,
    "amazon support" , "microsoft support" , "google security" ,
    "bank security" , "visa" , "mastercard" , "amex" ,
    "national lottery" , "telecom" , "azure" , "aws" , "gmail team" ,
    "facebook security"
]

SPOOFING_INDICATORS = [
    "noreply" , "no-reply" , "donotreply" , "do-not-reply" ,
    "support@" , "admin@" , "security@" , "verification@" , "alert@" ,
    "service@" , "info@" , "notification@" , "update@" , "mailer@" , "robot@" ,
    "support-team@" , "supportdesk@" , "helpdesk@" , "accounts@" ,
    "billing@" , "customerservice@" , "compliance@" , "system@" , "system-mail@" ,
    "alerts@" , "noreplymail@" , "auto-mailer@" , "webmaster@" , "it-support@"
]

CREDENTIAL_REQUESTS = [
    "enter your password" , "provide your password" , "confirm password" ,
    "password" , "username and password" , "username" , "user id" ,
    "social security number" , "ssn" , "account number" , "credit card" ,
    "credit card number" , "card details" , "cvv" , "pin number" , "pin" ,
    "date of birth" , "dob" , "mother's maiden name" , "security question" ,
    "full name and address" , "bank details" , "routing number" ,
    "id card" , "passport number" , "driver license" , "two-factor code" ,
    "otp" , "one-time password" , "authentication code" , "verify identity" ,
    "enter your credentials" , "provide login details" , "reauthenticate" ,
    "input verification code" , "enter 2fa code" , "banking password" ,
    "enter digits" , "identity confirmation" , "submit credentials"
]

FREE_EMAIL_DOMAINS = [
    "gmail.com" , "yahoo.com" , "hotmail.com" , "outlook.com" , "aol.com" ,
    "mail.com" , "protonmail.com" , "yandex.com" , "gmx.com" , "icloud.com" ,
    "live.com" , "msn.com" , "inbox.com" , "fastmail.com" , "zoho.com" ,
    "hushmail.com" , "tutanota.com" , "yahoo.co.uk" , "outlook.co.uk" ,
    "hotmail.co.uk" , "googlemail.com" , "mail.ru" , "gmx.net" ,
    "yandex.ru" , "usa.com" , "europe.com"
]

personal_pronouns = [
    " i " , " me " , " myself " , " my " , " mine " ,
    " you " , " your " , " yours " , " yourself " , " yourselves " ,
    " we " , " us " , " our " , " ours " , " ourselves " ,
    " he " , " him " , " his " , " himself " ,
    " she " , " her " , " hers " , " herself " ,
    " they " , " them " , " their " , " theirs " , " themselves " ,
    " i'm " , " you're " , " we're " , " they're " ,
    " i'd " , " you'd " , " we'd " , " they'd " ,
    " i'll " , " you'll " , " we'll " , " they'll " ,
    " i've " , " you've " , " we've " , " they've " ,
    " u " , " ur " , " im " , " id " , " youll " , " weve " ,
    " ya " , " u r " , " u're "
]

action_words = [
    "click here" , "verify now" , "update now" , "confirm now" , "act now" ,
    "login" , "log in" , "sign in" , "sign-in" , "reset password" ,
    "unlock account" , "reactivate account" , "open attachment" ,
    "open file" , "download" , "run file" , "install" , "enable macros" ,
    "enable content" , "open this link" , "visit link" , "review document" ,
    "access portal" , "complete form" , "submit info" , "submit information" ,
    "provide details" , "authorize" , "approve request" , "authenticate" ,
    "verify identity" , "take action" , "respond now" , "reply now" ,
    "urgent action required" , "redeem now" , "claim reward" , "claim prize" ,
    "claim bonus" , "authenticate now" , "fix account" , "resolve issue" ,
    "update details" , "check status" , "check your account" , "security check" ,
    "identity check" , "invoice due" , "payment required" , "confirm payment" ,
    "review invoice" , "download invoice" , "open invoice" , "final warning" ,
    "last notice" , "immediate attention" , "time-sensitive" , "expires today" ,
    "expires soon" , "within 24 hours" , "within 48 hours" , "verify details" ,
    "confirm identity" , "update credentials" , "reactivate immediately" ,
    "urgent verification" , "review activity" ,
    "review payment" , "open secure message" , "open secure file" ,
    "download secure document" , "complete verification" ,
    "follow instructions" , "press the button" , "tap to verify" ,
    "accept request" , "approve payment" , "review balance"
]

attachment_keywords = [
    "attached" , "attachment" , "find attached" , "see attached" , "enclosed" ,
    "included" , "attached file" , "attached document" , "attached invoice" ,
    "file" , "document" , "invoice" , "statement" , "report" , "form" , "pdf" ,
    "doc" , "docx" , "xls" , "xlsx" , "ppt" , "pptx" , "zip" , "rar" , "7z" , "tar" ,
    "gz" , "invoice.pdf" , "statement.pdf" , "payment.docx" , "document.pdf" ,
    "remittance.pdf" , "receipt.pdf" , "scan.pdf" , "scanned document" ,
    "secure document" , "protected file" , "important file" ,
    "download attachment" , "download file" , "open document" , "open report" ,
    "review attachment" , "package attached" ,
    "secure pdf" , "encrypted file" , "password-protected file" ,
    "confidential document" , "urgent document" , "delivery note" ,
    "shipping label" , "invoice copy" , "wire details" , "account statement"
]

common_misspellings = [
    "urgnet" , "accout" , "pasword" , "verifiy" , "confirim" , "securty" ,
    "recieve" , "bussiness" , "offical" , "adress" , "addres" , "identiy" ,
    "identitiy" , "authenticaion" , "updatte" , "verfication" , "verificatoin" ,
    "verifiction" , "logn" , "loggin" , "loggin in" , "passwrod" , "paswrod" ,
    "pssword" , "confrim" , "confim" , "confurm" , "securrity" , "securitty" ,
    "acount" , "acoount" , "accuont" , "accoubt" , "acc0unt" , "passw0rd" ,
    "ver1fy" , "conf1rm" , "secur1ty" , "verlfy" , "veriry" , "authentlcate" ,
    "l0gin" , "resp0nd" , "paymnet" , "invocie" , "docuemnt" , "statment" ,
    "recipt" , "notcie" , "activatoin" , "restircted" , "suspened" , "prizee" ,
    "reawrd" , "bannk" , "accuont" , "micorsoft" , "microsofft" ,
    "protction" , "verificaiton" , "confurmation" , "identificatoin" ,
    "autheticatoin" ,
    "passowrd" , "passworld" , "loggin" , "verifcation" , "immediatly" ,
    "urjent" , "supension" , "restrction" , "invoie" , "documant" ,
    "statemant" , "bankk" , "paymnent" , "invocie" , "accpunt" ,
    "vaccant" , "authentcation"
]

# ======================
# 2) Helpers
# ======================
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())

def extract_links(text: str):
    return re.findall(r'https?://[^\s]+', str(text))

# ======================
# 3) Date parsing + cyclical encoding
# ======================
def parse_date_text(date_str):
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    date_str = str(date_str).strip().lower()

    weekday = day = month = year = hour = minute = second = None

    if len(date_str) >= 3:
        weekday_str = date_str[:3].title()
        if weekday_str in weekdays:
            weekday = weekdays.index(weekday_str)

    match = re.search(r'(\d{1,2})\s+([a-z]{3})\s+(\d{4})', date_str)
    if match:
        day = int(match.group(1))
        month = month_map.get(match.group(2))
        year = int(match.group(3))

    match_time = re.search(r'(\d{2}):(\d{2})(?::(\d{2}))?', date_str)
    if match_time:
        hour = int(match_time.group(1))
        minute = int(match_time.group(2))
        second = int(match_time.group(3)) if match_time.group(3) else 0

    # fallback
    if None in [weekday, day, month, year, hour, minute, second]:
        try:
            dt = parsedate_to_datetime(date_str)
            if dt:
                weekday = weekday if weekday is not None else dt.weekday()
                day = day if day is not None else dt.day
                month = month if month is not None else dt.month
                year = year if year is not None else dt.year
                hour = hour if hour is not None else dt.hour
                minute = minute if minute is not None else dt.minute
                second = second if second is not None else dt.second
        except Exception:
            pass

    weekday = 0 if weekday is None else weekday
    day = 1 if day is None else day
    month = 1 if month is None else month
    year = 2025 if year is None else year
    hour = 0 if hour is None else hour
    minute = 0 if minute is None else minute
    second = 0 if second is None else second

    def cyclical(val, max_val):
        return (np.sin(2 * np.pi * val / max_val),
                np.cos(2 * np.pi * val / max_val))

    weekday_sin, weekday_cos = cyclical(weekday, 7)
    day_sin, day_cos = cyclical(day, 31)
    month_sin, month_cos = cyclical(month, 12)
    hour_sin, hour_cos = cyclical(hour, 24)
    minute_sin, minute_cos = cyclical(minute, 60)
    second_sin, second_cos = cyclical(second, 60)

    return pd.Series([
        weekday_sin, weekday_cos,
        day_sin, day_cos,
        month_sin, month_cos,
        hour_sin, hour_cos,
        minute_sin, minute_cos,
        second_sin, second_cos,
        year
    ], index=[
        'weekday_sin', 'weekday_cos',
        'day_sin', 'day_cos',
        'month_sin', 'month_cos',
        'hour_sin', 'hour_cos',
        'minute_sin', 'minute_cos',
        'second_sin', 'second_cos',
        'year'
    ])

# ======================
# 4) Feature Engineering
# ======================
def extract_features(df):
    features_df = df.copy()
    
    subject_lower = features_df["subject"].astype(str).str.lower()
    body_lower = features_df["body"].astype(str).str.lower()
    from_lower = features_df["from"].astype(str).str.lower()
    
    url_col = "url" if "url" in features_df.columns else "urls"
    urls_lower = features_df[url_col].fillna("").astype(str).str.lower()
    url_str_col = features_df[url_col].fillna("").astype(str)
    
    features_df["has_cc"] = (~features_df["cc"].isna()).astype(int)
    features_df["has_bcc"] = (~features_df["bcc"].isna()).astype(int)
    features_df["has_url"] = (~features_df[url_col].isna()).astype(int)
    
    features_df["cc_count"] = features_df["cc"].fillna("").astype(str).apply(
        lambda x: len([e for e in re.split(r"[,;]", x) if "@" in e])
    )
    features_df["bcc_count"] = features_df["bcc"].fillna("").astype(str).apply(
        lambda x: len([e for e in re.split(r"[,;]", x) if "@" in e])
    )
    
    features_df["total_recipients"] = features_df["cc_count"] + features_df["bcc_count"] + 1
    features_df["is_mass_email"] = (features_df["total_recipients"] > 5).astype(int)
    
    features_df["subject_length"] = features_df["subject"].astype(str).apply(len)
    features_df["subject_word_count"] = features_df["subject"].astype(str).apply(lambda x: len(x.split()))
    
    features_df["subject_urgent_keywords"] = subject_lower.apply(
        lambda x: sum(1 for kw in URGENT_KEYWORDS if kw in x)
    )
    features_df["subject_threat_keywords"] = subject_lower.apply(
        lambda x: sum(1 for kw in THREAT_KEYWORDS if kw in x)
    )
    features_df["subject_money_keywords"] = subject_lower.apply(
        lambda x: sum(1 for kw in MONEY_PRIZE_KEYWORDS if kw in x)
    )
    features_df["subject_generic_keywords"] = subject_lower.apply(
        lambda x: sum(1 for kw in GENERIC_SUSPICIOUS if kw in x)
    )
    features_df["subject_company_keywords"] = subject_lower.apply(
        lambda x: sum(1 for kw in COMPANY_IMPERSONATION if kw in x)
    )
    
    features_df["subject_excessive_punctuation"] = features_df["subject"].astype(str).apply(
        lambda x: int(bool(re.search(r"!{2,}|\?{2,}|\${2,}", x)))
    )
    features_df["subject_exclamation_count"] = features_df["subject"].astype(str).apply(lambda x: x.count("!"))
    features_df["subject_question_count"] = features_df["subject"].astype(str).apply(lambda x: x.count("?"))
    features_df["subject_dollar_count"] = features_df["subject"].astype(str).apply(lambda x: x.count("$"))
    
    features_df["subject_all_caps_ratio"] = features_df["subject"].astype(str).apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    features_df["subject_has_re_fwd"] = subject_lower.apply(
        lambda x: int(bool(re.match(r"^(re:|fw:|fwd:)", x)))
    )
    
    features_df["sender_is_freemail"] = from_lower.apply(
        lambda x: int(any(dom in x for dom in FREE_EMAIL_DOMAINS))
    )
    features_df["sender_has_numbers"] = features_df["from"].astype(str).apply(
        lambda x: int(bool(re.search(r"\d" , x)))
    )
    features_df["sender_number_count"] = features_df["from"].astype(str).apply(
        lambda x: len(re.findall(r"\d" , x))
    )
    features_df["sender_length"] = features_df["from"].astype(str).apply(len)
    
    features_df["sender_has_spoofing_indicator"] = from_lower.apply(
        lambda x: int(any(indicator in x for indicator in SPOOFING_INDICATORS))
    )
    
    features_df["sender_multiple_separators"] = features_df["from"].astype(str).apply(
        lambda x: int(bool(re.search(r"\.{2,}|-{2,}|_{2,}", x)))
    )
    
    features_df["body_length"] = features_df["body"].astype(str).apply(len)
    features_df["body_word_count"] = features_df["body"].astype(str).apply(lambda x: len(x.split()))
    
    features_df["body_urgent_keywords"] = body_lower.apply(
        lambda x: sum(1 for kw in URGENT_KEYWORDS if kw in x)
    )
    features_df["body_threat_keywords"] = body_lower.apply(
        lambda x: sum(1 for kw in THREAT_KEYWORDS if kw in x)
    )
    features_df["body_money_keywords"] = body_lower.apply(
        lambda x: sum(1 for kw in MONEY_PRIZE_KEYWORDS if kw in x)
    )
    features_df["body_generic_keywords"] = body_lower.apply(
        lambda x: sum(1 for kw in GENERIC_SUSPICIOUS if kw in x)
    )
    features_df["body_company_keywords"] = body_lower.apply(
        lambda x: sum(1 for kw in COMPANY_IMPERSONATION if kw in x)
    )
    features_df["body_urgency_phrases"] = body_lower.apply(
        lambda x: sum(1 for phrase in URGENCY_PHRASES if phrase in x)
    )
    features_df["body_credential_requests"] = body_lower.apply(
        lambda x: sum(1 for kw in CREDENTIAL_REQUESTS if kw in x)
    )
    
    features_df["body_excessive_punctuation"] = features_df["body"].astype(str).apply(
        lambda x: int(bool(re.search(r"!{2,}|\?{2,}|\${2,}" , x)))
    )
    features_df["body_exclamation_count"] = features_df["body"].astype(str).apply(lambda x: x.count("!"))
    features_df["body_question_count"] = features_df["body"].astype(str).apply(lambda x: x.count("?"))
    features_df["body_dollar_count"] = features_df["body"].astype(str).apply(lambda x: x.count("$"))
    
    features_df["body_has_html"] = features_df["body"].astype(str).apply(
        lambda x: int(bool(re.search(r"<[^>]+>" , x)))
    )
    
    features_df["body_has_misspellings"] = body_lower.apply(
        lambda x: int(any(word in x for word in common_misspellings))
    )
    
    features_df["url_count"] = url_str_col.apply(
        lambda x: len(re.findall(r"https?://" , x))
    )
    
    shorteners = ["bit.ly" , "tinyurl" , "goo.gl" , "t.co" , "ow.ly" , "is.gd" , "buff.ly"]
    features_df["url_is_shortened"] = urls_lower.apply(
        lambda x: int(any(short in x for short in shorteners))
    )
    
    features_df["url_avg_length"] = url_str_col.apply(
        lambda x: sum(len(url) for url in re.findall(r"https?://[^\s]+", x)) / max(1 , len(re.findall(r"https?://" , x)))
    )
    
    features_df["url_many_subdomains"] = url_str_col.apply(
        lambda x: int(any(len(re.findall(r"\." , url.split("/")[2])) > 3 
                         for url in re.findall(r"https?://[^\s]+" , x) if "/" in url))
    )
    
    features_df["url_has_at_symbol"] = url_str_col.apply(
        lambda x: int("@" in x and "http" in x)
    )
    
    features_df["sender_company_mismatch"] = (
        (features_df["sender_is_freemail"] == 1) &
        ((features_df["subject_company_keywords"] > 0) | 
         (features_df["body_company_keywords"] > 0))
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
    
    features_df["avg_word_length"] = features_df["body"].astype(str).apply(
        lambda x: sum(len(word) for word in x.split()) / max(1 , len(x.split()))
    )
    
    features_df["body_uppercase_ratio"] = features_df["body"].astype(str).apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(1 , len(x))
    )
    
    features_df["body_special_char_count"] = features_df["body"].astype(str).apply(
        lambda x: len(re.findall(r"[^a-zA-Z0-9\s]" , x))
    )
    
    features_df["mentions_attachment"] = body_lower.apply(
        lambda x: int(any(kw in x for kw in attachment_keywords))
    )
    
    features_df["body_personal_pronoun_count"] = body_lower.apply(
        lambda x: sum(1 for pronoun in personal_pronouns if pronoun in " " + x + " ")
    )
    
    features_df["body_action_request_count"] = body_lower.apply(
        lambda x: sum(1 for action in action_words if action in x)
    )
    
    features_df["has_signature"] = body_lower.apply(
        lambda x: int(bool(re.search(r"(regards|sincerely|best wishes|thanks|thank you)" , x)))
    )
    
    features_df["body_has_phone"] = features_df["body"].astype(str).apply(
        lambda x: int(bool(re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b" , x)))
    )
    
    features_df["body_is_very_short"] = (features_df["body_word_count"] < 10).astype(int)
    
    features_df["subject_body_overlap"] = features_df.apply(
        lambda row: len(set(str(row["subject"]).lower().split()) & 
                       set(str(row["body"]).lower().split())) / 
                   max(1 , len(str(row["subject"]).split())),
        axis = 1
    )
    
    features_df["link_text_no_url"] = (
        (features_df["url_count"] == 0) & 
        (body_lower.apply(lambda x: "click" in x or "link" in x or "http" in x))
    ).astype(int)
    
    return features_df
