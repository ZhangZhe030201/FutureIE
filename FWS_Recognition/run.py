# -*- coding: utf-8 -*-
"""
Unified entry point for FWS recognition.

Examples:
1) Train sklearn SVM
python run.py --mode train --model svm --train_csv data/train1.csv --model_path models/fws_svm.pkl --do_balance

2) 10-fold CV evaluation for sklearn SVM
python run.py --mode eval --model svm --train_csv data/train1.csv --cv 10 --do_balance

3) Evaluate a saved sklearn logistic regression model
python run.py --mode eval_saved --model logistic-regression --model_path models/fws_logistic.pkl --eval_csv data/train1.5.csv

4) Predict with a saved sklearn SVM model
python run.py --mode predict --model svm --model_path models/fws_svm.pkl --input_csv data/test.csv --out_csv reports/svm_predictions.csv

5) Train a transformer route (default checkpoint can be changed with --hf_ckpt)
python run.py --mode train --model scibert --hf_ckpt microsoft/deberta-v3-large --train_csv data/train1.csv --model_path models/deberta_fws --do_balance
"""
# python run.py --mode predict --model scibert --model_path D:\future_work_sentence\code\FWS_Recognition\scibert_scivocab_uncased\1yangbenlvscibert --input_csv D:\future_work_sentence\code\FWS_Recognition\data\candidate_acl_fws_sentences.csv --out_csv D:\future_work_sentence\code\FWS_Recognition\reports\scibert_predictions.csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fws_recognition.main import main


if __name__ == "__main__":
    main()
