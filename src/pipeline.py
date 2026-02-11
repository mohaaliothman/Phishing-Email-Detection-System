# src/pipeline.py
import pickle
import numpy as np
import pandas as pd
import torch

class PhishingDetectionPipeline:
    def __init__(self):
        self.tfidf_subject = None
        self.tfidf_body = None
        self.scaler = None
        self.svd = None
        self.kmeans = None
        self.model = None
        self.subject_weight = 2
        self.numeric_features = [
            "subject_length", "subject_word_count", "body_length", "body_word_count",
            "cc_count", "bcc_count", "total_recipients", "sender_length", "sender_number_count",
            "subject_exclamation_count", "subject_question_count", "subject_dollar_count",
            "body_exclamation_count", "body_question_count", "body_dollar_count", "avg_word_length",
            "url_count", "url_avg_length", "total_urgent_keywords", "total_threat_keywords",
            "total_money_keywords", "total_generic_keywords", "total_company_keywords"
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, tfidf_subject, tfidf_body, scaler, svd, kmeans, model):
        self.tfidf_subject = tfidf_subject
        self.tfidf_body = tfidf_body
        self.scaler = scaler
        self.svd = svd
        self.kmeans = kmeans
        self.model = model.to(self.device).eval()
        return self

    def transform_features(self, df: pd.DataFrame):
        X_subject = self.tfidf_subject.transform(df["subject"])
        X_subject_weighted = X_subject * self.subject_weight
        df_subject_tfidf = pd.DataFrame(
            X_subject_weighted.toarray(),
            columns=[f"subj_{w}" for w in self.tfidf_subject.get_feature_names_out()],
            index=df.index
        )

        X_body = self.tfidf_body.transform(df["body"])
        df_body_tfidf = pd.DataFrame(
            X_body.toarray(),
            columns=[f"body_{w}" for w in self.tfidf_body.get_feature_names_out()],
            index=df.index
        )

        X_tfidf = pd.concat([df_subject_tfidf, df_body_tfidf], axis=1)

        X_other = df[self.numeric_features].copy()
        X_other_scaled = X_other.copy()
        X_other_scaled[self.numeric_features] = self.scaler.transform(X_other[self.numeric_features])

        X_tfidf_reduced = self.svd.transform(X_tfidf)

        tfidf_cols = [f"tfidf_svd_{i}" for i in range(X_tfidf_reduced.shape[1])]
        other_cols = X_other_scaled.columns.tolist()

        X_combined = pd.DataFrame(
            np.hstack([X_tfidf_reduced, X_other_scaled.values]),
            index=df.index,
            columns=tfidf_cols + other_cols
        )

        cluster_labels = self.kmeans.predict(X_combined)

        return X_combined, cluster_labels

    def predict(self, df: pd.DataFrame, return_proba=False, threshold=0.5):
        X_combined, cluster_labels = self.transform_features(df)

       
        X_model = X_combined.copy()
        X_model["cluster"] = cluster_labels

        X_tensor = torch.FloatTensor(X_model.values).to(self.device)

        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy().flatten() 

        if return_proba:
            return probs, cluster_labels

        preds = (probs >= threshold).astype(int)
        return preds, cluster_labels

    def save(self, filepath: str):
        pipeline_data = {
            "tfidf_subject": self.tfidf_subject,
            "tfidf_body": self.tfidf_body,
            "scaler": self.scaler,
            "svd": self.svd,
            "kmeans": self.kmeans,
            "model_state_dict": self.model.state_dict(),
            "subject_weight": self.subject_weight,
            "numeric_features": self.numeric_features,
        }
        with open(filepath, "wb") as f:
            pickle.dump(pipeline_data, f)

    @classmethod
    def load(cls, filepath: str, model_instance):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        pipe = cls()
        pipe.tfidf_subject = data["tfidf_subject"]
        pipe.tfidf_body = data["tfidf_body"]
        pipe.scaler = data["scaler"]
        pipe.svd = data["svd"]
        pipe.kmeans = data["kmeans"]
        pipe.subject_weight = data["subject_weight"]
        pipe.numeric_features = data["numeric_features"]

        pipe.model = model_instance
        pipe.model.load_state_dict(data["model_state_dict"])
        pipe.model.to(pipe.device).eval()
        return pipe
