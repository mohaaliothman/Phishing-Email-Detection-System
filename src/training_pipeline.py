import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

from features import extract_features, parse_date_text
from model import PhishingDetector
from pipeline import PhishingDetectionPipeline

def auto_phishing_label(row):
    score = 0
    score += 5 * row.get("body_credential_requests", 0)
    score += 4 * row.get("sender_company_mismatch", 0)
    score += 4 * row.get("url_has_ip", 0)
    score += 3 * row.get("total_threat_keywords", 0)
    score += 2 * row.get("total_urgent_keywords", 0)
    score += 2 * row.get("body_urgency_phrases", 0)
    score += 3 * row.get("total_money_keywords", 0)
    score += 2 * row.get("url_is_shortened", 0)
    score += 2 * row.get("url_many_subdomains", 0)
    score += 3 * row.get("url_has_at_symbol", 0)

    avg_len = row.get("url_avg_length", 0)
    if avg_len > 100:
        score += 2
    elif avg_len < 15 and row.get("url_count", 0) > 0:
        score += 1

    if row.get("url_to_text_ratio", 0) > 0.5:
        score += 2

    score += 2 * row.get("sender_has_spoofing_indicator", 0)
    score += 1 * row.get("sender_is_freemail", 0)
    score += 1 * row.get("sender_multiple_separators", 0)

    if row.get("sender_number_count", 0) > 3:
        score += 2
    elif row.get("sender_has_numbers", 0):
        score += 1

    sender_len = row.get("sender_length", 0)
    if sender_len < 8 or sender_len > 60:
        score += 1

    score += 1 * row.get("total_generic_keywords", 0)

    if row.get("subject_excessive_punctuation", 0):
        score += 1
    if row.get("body_excessive_punctuation", 0):
        score += 1

    score += min(row.get("subject_exclamation_count", 0), 3)
    score += min(row.get("body_exclamation_count", 0), 3)
    score += min(row.get("subject_dollar_count", 0), 2)
    score += min(row.get("body_dollar_count", 0), 2)

    caps = row.get("subject_all_caps_ratio", 0)
    if caps > 0.6:
        score += 2
    elif caps > 0.3:
        score += 1

    if row.get("urgent_keyword_ratio", 0) > 0.05:
        score += 2
    if row.get("threat_keyword_ratio", 0) > 0.03:
        score += 2
    if row.get("money_keyword_ratio", 0) > 0.03:
        score += 2

    return 1 if score >= 18 else 0

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb).squeeze()
        yb = yb.squeeze()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (out > 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / len(loader), correct / max(1, total)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb).squeeze()
            yb = yb.squeeze()
            loss = criterion(out, yb)

            total_loss += loss.item()
            preds = (out > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return total_loss / len(loader), correct / max(1, total)

def main():
    df = pd.read_csv("Enron_Emails.csv")
    df = df.drop_duplicates()
    for c in ["to", "subject", "body", "from", "cc", "bcc", "url"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
        else:
            df[c] = ""

    date_cols = [
        'weekday_sin','weekday_cos','day_sin','day_cos','month_sin','month_cos',
        'hour_sin','hour_cos','minute_sin','minute_cos','second_sin','second_cos','year'
    ]
    df[date_cols] = df["date"].apply(parse_date_text)

    engineered = extract_features(df)

    # TF-IDF
    tfidf_subject = TfidfVectorizer(max_features=100, min_df=100, max_df=0.5, stop_words="english")
    X_subject = tfidf_subject.fit_transform(engineered["subject"])
    subject_weight = 2
    X_subject_weighted = X_subject * subject_weight
    df_subject_tfidf = pd.DataFrame(
        X_subject_weighted.toarray(),
        columns=[f"subj_{w}" for w in tfidf_subject.get_feature_names_out()],
        index=engineered.index
    )

    tfidf_body = TfidfVectorizer(max_features=300, min_df=100, max_df=0.5, stop_words="english")
    X_body = tfidf_body.fit_transform(engineered["body"])
    df_body_tfidf = pd.DataFrame(
        X_body.toarray(),
        columns=[f"body_{w}" for w in tfidf_body.get_feature_names_out()],
        index=engineered.index
    )

    engineered = pd.concat([engineered, df_subject_tfidf, df_body_tfidf], axis=1)

    # label
    engineered["possible_phishing"] = engineered.apply(auto_phishing_label, axis=1)

    # drop non-features
    drop_cols = ["file_path", "date", "subject", "from", "to", "cc", "bcc", "body", "url"]
    engineered = engineered.drop(columns=[c for c in drop_cols if c in engineered.columns])

    # Outliers
    row_norms = np.linalg.norm(engineered.drop(columns=["possible_phishing"]).values, axis=1)
    outliers = np.where(row_norms > np.percentile(row_norms, 99.99))[0]
    engineered = engineered.drop(index=outliers).reset_index(drop=True)

    tfidf_cols = [c for c in engineered.columns if c.startswith("subj_") or c.startswith("body_")]
    y = engineered["possible_phishing"].copy()

    X_other = engineered.drop(columns=tfidf_cols + ["possible_phishing"])
    X_tfidf = engineered[tfidf_cols]

    numeric_features = [
        "subject_length","subject_word_count","body_length","body_word_count",
        "cc_count","bcc_count","total_recipients","sender_length","sender_number_count",
        "subject_exclamation_count","subject_question_count","subject_dollar_count",
        "body_exclamation_count","body_question_count","body_dollar_count","avg_word_length",
        "url_count","url_avg_length","total_urgent_keywords","total_threat_keywords",
        "total_money_keywords","total_generic_keywords","total_company_keywords"
    ]

    scaler = StandardScaler()
    X_other_scaled = X_other.copy()
    X_other_scaled[numeric_features] = scaler.fit_transform(X_other_scaled[numeric_features])

    # SVD + KMeans
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_tfidf_reduced = svd.fit_transform(X_tfidf)

    X_combined = pd.DataFrame(
        np.hstack([X_tfidf_reduced, X_other_scaled.values]),
        columns=[f"tfidf_svd_{i}" for i in range(X_tfidf_reduced.shape[1])] + X_other_scaled.columns.tolist()
    )

    kmeans = KMeans(n_clusters=3, max_iter=50, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_combined)

    # add cluster to training features
    X_combined["cluster"] = labels

    # undersample
    data = X_combined.copy()
    data["target"] = y.values
    majority = data[data["target"] == 0]
    minority = data[data["target"] == 1]
    majority_under = resample(majority, replace=False, n_samples=len(minority)*2, random_state=42)
    data_balanced = pd.concat([majority_under, minority]).sample(frac=1, random_state=42).reset_index(drop=True)

    Xb = data_balanced.drop(columns=["target"])
    yb = data_balanced["target"]

    X_train, X_test, y_train, y_test = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values).unsqueeze(1)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val.values), torch.FloatTensor(y_val.values).unsqueeze(1)),
        batch_size=32, shuffle=False
    )

    model = PhishingDetector(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    best_val = float("inf")
    patience, pc = 4, 0

    for epoch in range(30):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            pc = 0
            torch.save(model.state_dict(), "best_phishing_model.pth")
        else:
            pc += 1
            if pc >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load("best_phishing_model.pth", map_location=device))

    # Save pipeline
    pipe = PhishingDetectionPipeline().fit(
        tfidf_subject=tfidf_subject,
        tfidf_body=tfidf_body,
        scaler=scaler,
        svd=svd,
        kmeans=kmeans,
        model=model
    )
    pipe.save("phishing_detection_pipeline.pkl")
    print("Saved phishing_detection_pipeline.pkl")

if __name__ == "__main__":
    main()
