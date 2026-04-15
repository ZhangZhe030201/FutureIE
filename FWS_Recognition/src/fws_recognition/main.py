import os
import re
import random
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# -------- NLTK (for sklearn pipeline preprocessing) --------
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_recall_fscore_support

from .logging_utils import setup_file_logger, make_print_and_log
from .report_utils import save_eval_result

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = setup_file_logger(PROJECT_ROOT / "logs" / "run.log")
print = make_print_and_log(LOGGER)
REPORTS_DIR = PROJECT_ROOT / "reports"

# Optional tqdm (progress bars)
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# -------- Transformers (for SciBERT) --------
_HAS_TRANSFORMERS = False
try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        set_seed as hf_set_seed,
    )
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# -----------------------------
# NLTK downloads (more compatible)
# -----------------------------
def _safe_nltk_download(pkg: str):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


_safe_nltk_download("punkt")
_safe_nltk_download("averaged_perceptron_tagger")
_safe_nltk_download("wordnet")
_safe_nltk_download("omw-1.4")


# -----------------------------
# Data balancing (same as yours)
# -----------------------------
def JustPercent(texts, labels, neg_cap=9009, seed=42):
    """
    Keep all positive samples, subsample negatives up to neg_cap.
    """
    neg_texts, pos_texts = [], []
    for t, y in zip(texts, labels):
        if int(y) == 0:
            neg_texts.append(t)
        else:
            pos_texts.append(t)

    rnd = random.Random(seed)
    rnd.shuffle(neg_texts)

    keep = min(neg_cap, len(neg_texts))
    neg_texts = neg_texts[:keep]

    texts_new = neg_texts + pos_texts
    labels_new = [0] * len(neg_texts) + [1] * len(pos_texts)
    return texts_new, labels_new


# -----------------------------
# Text preprocessing (for sklearn path)
# -----------------------------
def remove_punctuation(line):
    line = str(line).strip()
    line = line.replace("-", " ")
    line = line.replace("'s", "")

    rule2 = re.compile(r"\(.*?\)|{.*?}|\[.*?]")
    line = rule2.sub("", line)

    rule1 = re.compile(r"[^a-zA-Z\s+]")
    line = rule1.sub("", line).lower()
    return line


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return None


def lemm(sentence):
    tokens = word_tokenize(sentence)
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmas = []
    for w, t in tagged_sent:
        wn_pos = get_wordnet_pos(t) or wordnet.NOUN
        lemmas.append(wnl.lemmatize(w, pos=wn_pos))
    return " ".join(lemmas)


def preprocess_texts(texts):
    if _HAS_TQDM:
        cleaned = [remove_punctuation(t) for t in tqdm(texts, desc="clean", leave=False)]
        lemmatized = [lemm(t) for t in tqdm(cleaned, desc="lemmatize", leave=False)]
        return lemmatized
    else:
        cleaned = [remove_punctuation(t) for t in texts]
        lemmatized = [lemm(t) for t in cleaned]
        return lemmatized


class PreprocessTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapper for preprocess_texts so it can be inside a Pipeline and pickled."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return preprocess_texts(list(X))


# -----------------------------
# Safe SelectKBest: avoid k > n_features crash
# -----------------------------
class SafeSelectKBest(SelectKBest):
    def __init__(self, score_func=chi2, k=14000):
        super().__init__(score_func=score_func, k=k)

    def fit(self, X, y):
        if isinstance(self.k, int):
            self.k = min(self.k, X.shape[1])
        return super().fit(X, y)


# -----------------------------
# sklearn Classifier builder
# -----------------------------
def build_clf(model_name: str):
    if model_name == "svm":
        return LinearSVC(max_iter=5000, C=1.0)
    if model_name == "naive-bayes":
        return BernoulliNB(alpha=0.0001)
    if model_name == "random-forest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    if model_name == "logistic-regression":
        return LR(max_iter=1000, n_jobs=-1)
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str, k_features: int):
    return Pipeline(steps=[
        ("prep", PreprocessTransformer()),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=3,
            max_df=1.0,
            sublinear_tf=True,
            analyzer="word",
        )),
        ("select", SafeSelectKBest(chi2, k=k_features)),
        ("clf", build_clf(model_name))
    ])


def build_pipeline_no_prep(model_name: str, k_features: int):
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=3,
            max_df=1.0,
            sublinear_tf=True,
            analyzer="word",
        )),
        ("select", SafeSelectKBest(chi2, k=k_features)),
        ("clf", build_clf(model_name))
    ])


# -----------------------------
# Load CSV helpers
# -----------------------------
def load_labeled_csv(path: str, text_col="text", label_col="label"):
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Labeled CSV must contain columns: {text_col}, {label_col}")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return df, texts, labels


def load_unlabeled_csv(path: str, text_col="text"):
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Unlabeled CSV must contain column: {text_col}")
    texts = df[text_col].astype(str).tolist()
    return df, texts


# ============================================================
# sklearn train/eval/predict (unchanged behavior)
# ============================================================
def train_and_save_sklearn(train_csv: str, model_name: str, model_path: str,
                           text_col="text", label_col="label",
                           do_balance=True, neg_cap=9009, seed=42, k_features=14000):
    print("[train][sklearn] step 1/4: loading labeled csv...")
    _, texts, labels = load_labeled_csv(train_csv, text_col, label_col)
    print(f"[train][sklearn] loaded: {len(texts)} samples")

    print("[train][sklearn] step 2/4: balancing (optional)...")
    if do_balance:
        texts, labels = JustPercent(texts, labels, neg_cap=neg_cap, seed=seed)
        print(f"[train][sklearn] after balance: {len(texts)} samples (neg_cap={neg_cap})")
    else:
        print("[train][sklearn] balance disabled")

    print("[train][sklearn] step 3/4: fitting pipeline (prep->tfidf->select->clf)...")
    pipe = build_pipeline(model_name, k_features)
    pipe.fit(texts, labels)

    print("[train][sklearn] step 4/4: saving model bundle (.pkl)...")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "model_name": model_name,
        "text_col": text_col,
        "label_col": label_col,
        "do_balance": do_balance,
        "neg_cap": neg_cap,
        "seed": seed,
        "k_features": k_features,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[OK] Trained and saved sklearn model bundle to: {model_path}")


def eval_cv_sklearn(train_csv: str, model_name: str, cv: int,
                    text_col="text", label_col="label",
                    do_balance=True, neg_cap=9009, seed=42, k_features=14000):

    print("[eval][sklearn] loading labeled csv...")
    _, texts, labels = load_labeled_csv(train_csv, text_col, label_col)

    if do_balance:
        texts, labels = JustPercent(texts, labels, neg_cap=neg_cap, seed=seed)

    print("[eval][sklearn] preprocessing texts (once)...")
    texts = preprocess_texts(texts)

    print(f"[eval][sklearn] running {cv}-fold CV ...")
    pipe = build_pipeline_no_prep(model_name, k_features)

    scores = cross_validate(
        pipe, texts, labels,
        cv=cv, n_jobs=-1,
        scoring=["precision_macro", "recall_macro", "f1_macro"],
        return_train_score=False
    )

    p = scores["test_precision_macro"]
    r = scores["test_recall_macro"]
    f = scores["test_f1_macro"]

    print("\n[eval][sklearn] macro results (per-fold):")
    print(" precision_macro:", p)
    print(" recall_macro   :", r)
    print(" f1_macro       :", f)

    print("\n[eval][sklearn] macro results (mean):")
    print(" precision_macro mean:", p.mean())
    print(" recall_macro    mean:", r.mean())
    print(" f1_macro        mean:", f.mean())

    save_eval_result(
        {
            "task": "fws_recognition",
            "mode": "eval_cv",
            "model": model_name,
            "train_csv": train_csv,
            "eval_csv": "",
            "cv": cv,
            "sample_size": len(texts),
            "precision": "",
            "recall": "",
            "f1": "",
            "precision_macro_mean": float(p.mean()),
            "recall_macro_mean": float(r.mean()),
            "f1_macro_mean": float(f.mean()),
            "notes": f"do_balance={do_balance}; neg_cap={neg_cap}; k_features={k_features}",
        },
        REPORTS_DIR,
    )


def eval_saved_sklearn(model_path: str, eval_csv: str,
                       text_col="text", label_col="label",
                       average="macro"):
    """
    Evaluate a trained sklearn model (.pkl) on a labeled validation set.
    """
    print("[eval_saved][sklearn] loading trained model...")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    pipe = bundle["pipeline"]

    print("[eval_saved][sklearn] loading eval csv:", eval_csv)
    df = pd.read_csv(eval_csv)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Eval CSV must contain columns: {text_col}, {label_col}")

    X = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(int).tolist()

    print(f"[eval_saved][sklearn] evaluating on {len(X)} samples...")
    y_pred = pipe.predict(X)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    print("\n[eval_saved][sklearn] results:")
    print(" precision:", float(p))
    print(" recall   :", float(r))
    print(" f1       :", float(f1))

    save_eval_result(
        {
            "task": "fws_recognition",
            "mode": "eval_saved",
            "model": bundle.get("model_name", "unknown"),
            "train_csv": "",
            "eval_csv": eval_csv,
            "cv": "",
            "sample_size": len(X),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "precision_macro_mean": "",
            "recall_macro_mean": "",
            "f1_macro_mean": "",
            "notes": f"model_path={model_path}",
        },
        REPORTS_DIR,
    )


def predict_csv_sklearn(model_path: str, input_csv: str, out_csv: str, text_col="text"):
    print("[predict][sklearn] step 1/4: loading model bundle...")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    pipe: Pipeline = bundle["pipeline"]
    print(f"[predict][sklearn] loaded model: {bundle.get('model_name')}")

    print("[predict][sklearn] step 2/4: reading input csv...")
    df, texts = load_unlabeled_csv(input_csv, text_col=text_col)
    print(f"[predict][sklearn] loaded: {len(texts)} samples")

    print("[predict][sklearn] step 3/4: predicting labels...")
    pred = pipe.predict(texts)
    df["pred_label"] = pred

    print("[predict][sklearn] step 4/4: adding scores (if available) and saving...")
    clf = pipe.named_steps["clf"]

    if hasattr(clf, "decision_function"):
        try:
            df["score"] = pipe.decision_function(texts)
        except Exception:
            pass

    if hasattr(clf, "predict_proba"):
        try:
            proba = pipe.predict_proba(texts)
            if proba.shape[1] >= 2:
                df["prob_0"] = proba[:, 0]
                df["prob_1"] = proba[:, 1]
        except Exception:
            pass

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved predictions to: {out_csv}")


# ============================================================
# SciBERT train/eval/predict (new)
# ============================================================
SCIBERT_CKPT = "microsoft/deberta-v3-large"


def _require_transformers():
    if not _HAS_TRANSFORMERS:
        raise RuntimeError(
            "Transformers path requires extra deps. Run:\n"
            "  pip install -U transformers datasets evaluate accelerate torch\n"
        )


def _make_hf_dataset(texts, labels=None, text_col="text"):
    if labels is None:
        df = pd.DataFrame({text_col: texts})
    else:
        df = pd.DataFrame({text_col: texts, "label": labels})
    return Dataset.from_pandas(df)


def _scibert_tokenize(ds, tokenizer, text_col="text", max_length=256):
    def tok(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    ds = ds.map(tok, batched=True)
    keep_cols = ["input_ids", "attention_mask"]
    if "label" in ds.column_names:
        keep_cols.append("label")
    ds.set_format(type="torch", columns=keep_cols)
    return ds


def _compute_metrics_factory(average="binary"):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds,
            average=average,
            zero_division=0
        )
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}

    return compute_metrics


def eval_scibert_from_saved(model_dir: str, eval_csv: str, hf_ckpt: str,
                            text_col="text", label_col="label",
                            max_length=None, metric_average="binary"):
    _require_transformers()

    print("[eval_saved][scibert] loading model/tokenizer from:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    if max_length is None:
        max_length = 256
        meta_path = os.path.join(model_dir, "meta.pkl")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                max_length = int(meta.get("max_length", max_length))
                metric_average = meta.get("metric_average", metric_average)
            except Exception:
                pass

    print("[eval_saved][scibert] reading eval csv:", eval_csv)
    df = pd.read_csv(eval_csv)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Eval CSV must contain columns: {text_col}, {label_col}")

    texts = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(int).tolist()
    print(f"[eval_saved][scibert] eval samples: {len(texts)}")

    ds = _make_hf_dataset(texts, y_true, text_col=text_col)
    ds = _scibert_tokenize(ds, tokenizer, text_col=text_col, max_length=max_length)

    args = TrainingArguments(
        output_dir="data/_scibert_eval_saved_tmp",
        per_device_eval_batch_size=64,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer)

    print("[eval_saved][scibert] predicting...")
    pred = trainer.predict(ds)
    logits = pred.predictions
    y_pred = np.argmax(logits, axis=-1)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=metric_average, zero_division=0
    )

    print("\n[eval_saved][scibert] results:")
    print(" precision:", float(p))
    print(" recall   :", float(r))
    print(" f1       :", float(f1))

    save_eval_result(
        {
            "task": "fws_recognition",
            "mode": "eval_saved",
            "model": "scibert",
            "train_csv": "",
            "eval_csv": eval_csv,
            "cv": "",
            "sample_size": len(texts),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "precision_macro_mean": "",
            "recall_macro_mean": "",
            "f1_macro_mean": "",
            "notes": f"model_dir={model_dir}; hf_ckpt={hf_ckpt}",
        },
        REPORTS_DIR,
    )


def train_and_save_scibert(train_csv: str, model_dir: str, hf_ckpt: str,
                           text_col="text", label_col="label",
                           do_balance=True, neg_cap=9009, seed=42,
                           max_length=256,
                           lr=2e-5, epochs=3,
                           train_bs=16, eval_bs=32,
                           weight_decay=0.01,
                           eval_split=0.2,
                           metric_average="binary"):
    _require_transformers()
    hf_set_seed(seed)

    print("[train][scibert] loading labeled csv...")
    _, texts, labels = load_labeled_csv(train_csv, text_col, label_col)
    print(f"[train][scibert] loaded: {len(texts)} samples")

    if do_balance:
        print("[train][scibert] balancing (optional)...")
        texts, labels = JustPercent(texts, labels, neg_cap=neg_cap, seed=seed)
        print(f"[train][scibert] after balance: {len(texts)} samples (neg_cap={neg_cap})")

    print("[train][scibert] train/val split...")
    X_tr, X_ev, y_tr, y_ev = train_test_split(
        texts, labels, test_size=eval_split, random_state=seed, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_ckpt, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(hf_ckpt, num_labels=2)

    train_ds = _make_hf_dataset(X_tr, y_tr, text_col=text_col)
    eval_ds = _make_hf_dataset(X_ev, y_ev, text_col=text_col)
    train_ds = _scibert_tokenize(train_ds, tokenizer, text_col=text_col, max_length=max_length)
    eval_ds = _scibert_tokenize(eval_ds, tokenizer, text_col=text_col, max_length=max_length)

    os.makedirs(model_dir, exist_ok=True)
    args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics_factory(metric_average),
    )

    print("[train][scibert] training...")
    trainer.train()

    print("[train][scibert] final eval...")
    metrics = trainer.evaluate()
    print("[train][scibert] metrics:", metrics)

    print("[train][scibert] saving model + tokenizer...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    meta = {
        "model": "scibert",
        "checkpoint": SCIBERT_CKPT,
        "text_col": text_col,
        "label_col": label_col,
        "do_balance": do_balance,
        "neg_cap": neg_cap,
        "seed": seed,
        "max_length": max_length,
        "metric_average": metric_average,
    }
    with open(os.path.join(model_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"[OK] Trained and saved SciBERT to directory: {model_dir}")


def eval_scibert(train_csv: str, hf_ckpt: str,
                 text_col="text", label_col="label",
                 do_balance=True, neg_cap=9009, seed=42,
                 max_length=256,
                 lr=2e-5, epochs=3,
                 train_bs=16, eval_bs=32,
                 weight_decay=0.01,
                 eval_split=0.2,
                 metric_average="binary"):
    """
    为了“像以前一样一条命令评估”，这里默认做一次 holdout（train/val split）
    并打印 precision/recall/f1。更严格的 k-fold SciBERT 也能做，但会非常耗时。
    """
    _require_transformers()
    hf_set_seed(seed)

    print("[eval][scibert] loading labeled csv...")
    _, texts, labels = load_labeled_csv(train_csv, text_col, label_col)

    if do_balance:
        texts, labels = JustPercent(texts, labels, neg_cap=neg_cap, seed=seed)

    print("[eval][scibert] train/val split...")
    X_tr, X_ev, y_tr, y_ev = train_test_split(
        texts, labels, test_size=eval_split, random_state=seed, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_ckpt, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(hf_ckpt, num_labels=2)

    train_ds = _make_hf_dataset(X_tr, y_tr, text_col=text_col)
    eval_ds = _make_hf_dataset(X_ev, y_ev, text_col=text_col)
    train_ds = _scibert_tokenize(train_ds, tokenizer, text_col=text_col, max_length=max_length)
    eval_ds = _scibert_tokenize(eval_ds, tokenizer, text_col=text_col, max_length=max_length)

    tmp_out = "data/_scibert_eval_tmp"
    os.makedirs(tmp_out, exist_ok=True)

    args = TrainingArguments(
        output_dir=tmp_out,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics_factory(metric_average),
    )

    print("[eval][scibert] training (for evaluation)...")
    trainer.train()

    print("[eval][scibert] evaluating...")
    metrics = trainer.evaluate()

    p = metrics.get("eval_precision", None)
    r = metrics.get("eval_recall", None)
    f = metrics.get("eval_f1", None)

    print("\n[eval][scibert] results (holdout):")
    print(" precision:", p)
    print(" recall   :", r)
    print(" f1       :", f)

    save_eval_result(
        {
            "task": "fws_recognition",
            "mode": "eval_holdout",
            "model": "scibert",
            "train_csv": train_csv,
            "eval_csv": "",
            "cv": "",
            "sample_size": len(texts),
            "precision": float(p) if p is not None else "",
            "recall": float(r) if r is not None else "",
            "f1": float(f) if f is not None else "",
            "precision_macro_mean": "",
            "recall_macro_mean": "",
            "f1_macro_mean": "",
            "notes": f"hf_ckpt={hf_ckpt}; do_balance={do_balance}; neg_cap={neg_cap}; eval_split={eval_split}",
        },
        REPORTS_DIR,
    )


def predict_csv_scibert(model_dir: str, input_csv: str, out_csv: str, hf_ckpt: str = None, text_col="text"):
    _require_transformers()

    print("[predict][scibert] loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    max_length = 256
    meta_path = os.path.join(model_dir, "meta.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            max_length = int(meta.get("max_length", max_length))
        except Exception:
            pass

    print("[predict][scibert] reading input csv...")
    df, texts = load_unlabeled_csv(input_csv, text_col=text_col)
    print(f"[predict][scibert] loaded: {len(texts)} samples")

    ds = _make_hf_dataset(texts, labels=None, text_col=text_col)
    ds = _scibert_tokenize(ds, tokenizer, text_col=text_col, max_length=max_length)

    args = TrainingArguments(
        output_dir="data/_scibert_pred_tmp",
        per_device_eval_batch_size=64,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer)

    print("[predict][scibert] predicting...")
    preds = trainer.predict(ds)
    logits = preds.predictions
    pred_label = np.argmax(logits, axis=-1)
    df["pred_label"] = pred_label

    try:
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        if probs.shape[1] >= 2:
            df["prob_0"] = probs[:, 0]
            df["prob_1"] = probs[:, 1]
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved predictions to: {out_csv}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FWS recognition: train/eval/predict (sklearn + scibert)")

    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval", "predict", "eval_saved"],
                        help="train: fit final model and save; eval: evaluate; predict: predict unlabeled csv")

    parser.add_argument("--train_csv", type=str, default="Corpus_For_FWS_Recognition.csv",
                        help="Labeled training CSV path (text,label)")
    parser.add_argument("--model", type=str, default="svm",
                        choices=["svm", "naive-bayes", "random-forest", "logistic-regression", "scibert"],
                        help="Model type")

    parser.add_argument("--cv", type=int, default=10, help="CV folds (sklearn eval only)")

    parser.add_argument("--model_path", type=str, default="data/fws_model.pkl",
                        help="sklearn: .pkl file; scibert: directory path (e.g., data/scibert_fws)")

    parser.add_argument("--input_csv", type=str, default=None, help="Unlabeled CSV path (predict mode)")
    parser.add_argument("--out_csv", type=str, default="data/predictions.csv", help="Output CSV path")

    parser.add_argument("--text_col", type=str, default="text", help="Text column name")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name (labeled csv)")
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=None,
        help="Labeled validation CSV for evaluating a saved model"
    )

    parser.add_argument("--k_features", type=int, default=14000, help="chi2 SelectKBest k (sklearn only)")

    parser.add_argument("--do_balance", action="store_true", help="Enable negative downsampling (like original)")
    parser.add_argument("--neg_cap", type=int, default=9009, help="Max number of negative samples kept")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--max_length", type=int, default=256, help="SciBERT max_length")
    parser.add_argument("--lr", type=float, default=5e-5, help="SciBERT learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="SciBERT epochs")
    parser.add_argument("--train_bs", type=int, default=16, help="SciBERT train batch size per device")
    parser.add_argument("--eval_bs", type=int, default=32, help="SciBERT eval batch size per device")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="SciBERT weight decay")
    parser.add_argument("--eval_split", type=float, default=0.2, help="SciBERT holdout split ratio in eval/train")
    parser.add_argument("--metric_average", type=str, default="binary", choices=["binary", "macro"],
                        help="SciBERT metric averaging (binary or macro)")
    parser.add_argument(
        "--hf_ckpt",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="HuggingFace pretrained checkpoint for transformer training (e.g., roberta-base, microsoft/deberta-v3-base)"
    )

    args = parser.parse_args()

    if not os.path.exists("data/"):
        os.mkdir("data/")

    if args.model != "scibert":
        if args.mode == "train":
            train_and_save_sklearn(
                train_csv=args.train_csv,
                model_name=args.model,
                model_path=args.model_path,
                text_col=args.text_col,
                label_col=args.label_col,
                do_balance=args.do_balance,
                neg_cap=args.neg_cap,
                seed=args.seed,
                k_features=args.k_features
            )
        elif args.mode == "eval":
            eval_cv_sklearn(
                train_csv=args.train_csv,
                model_name=args.model,
                cv=args.cv,
                text_col=args.text_col,
                label_col=args.label_col,
                do_balance=args.do_balance,
                neg_cap=args.neg_cap,
                seed=args.seed,
                k_features=args.k_features
            )
        elif args.mode == "predict":
            if args.input_csv is None:
                raise ValueError("--input_csv is required in predict mode")
            predict_csv_sklearn(
                model_path=args.model_path,
                input_csv=args.input_csv,
                out_csv=args.out_csv,
                text_col=args.text_col
            )
        elif args.mode == "eval_saved":
            if args.eval_csv is None:
                raise ValueError("--eval_csv is required in eval_saved mode")
            eval_saved_sklearn(
                model_path=args.model_path,
                eval_csv=args.eval_csv,
                text_col=args.text_col,
                label_col=args.label_col,
                average="macro"
            )
    else:
        if args.mode == "train":
            train_and_save_scibert(
                train_csv=args.train_csv,
                model_dir=args.model_path,
                hf_ckpt=args.hf_ckpt,
                text_col=args.text_col,
                label_col=args.label_col,
                do_balance=args.do_balance,
                neg_cap=args.neg_cap,
                seed=args.seed,
                max_length=args.max_length,
                lr=args.lr,
                epochs=args.epochs,
                train_bs=args.train_bs,
                eval_bs=args.eval_bs,
                weight_decay=args.weight_decay,
                eval_split=args.eval_split,
                metric_average=args.metric_average,
            )
        elif args.mode == "eval":
            eval_scibert(
                train_csv=args.train_csv,
                text_col=args.text_col,
                hf_ckpt=args.hf_ckpt,
                label_col=args.label_col,
                do_balance=args.do_balance,
                neg_cap=args.neg_cap,
                seed=args.seed,
                max_length=args.max_length,
                lr=args.lr,
                epochs=args.epochs,
                train_bs=args.train_bs,
                eval_bs=args.eval_bs,
                weight_decay=args.weight_decay,
                eval_split=args.eval_split,
                metric_average=args.metric_average,
            )
        elif args.mode == "predict":
            if args.input_csv is None:
                raise ValueError("--input_csv is required in predict mode")
            predict_csv_scibert(
                model_dir=args.model_path,
                input_csv=args.input_csv,
                hf_ckpt=args.hf_ckpt,
                out_csv=args.out_csv,
                text_col=args.text_col
            )
        elif args.mode == "eval_saved":
            if args.eval_csv is None:
                raise ValueError("--eval_csv is required in eval_saved mode")
            eval_scibert_from_saved(
                model_dir=args.model_path,
                hf_ckpt=args.hf_ckpt,
                eval_csv=args.eval_csv,
                text_col=args.text_col,
                label_col=args.label_col,
                max_length=args.max_length,
                metric_average=args.metric_average
            )


if __name__ == "__main__":
    main()
