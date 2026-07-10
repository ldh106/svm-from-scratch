# main.py
# 실행 엔트리포인트: 학습 -> 평가 -> test_set 예측
import os

from data_utils import (
    read_csv_xy, read_csv_x, StandardScaler,
    to_svm_labels, from_svm_label, accuracy, confusion_matrix
)
from svm_smo import LinearSVM_SMO


def train_and_eval(train_path, label_col, feature_cols=None, C=1.0):
    X, y01, used_features = read_csv_xy(train_path, feature_cols=feature_cols, label_col=label_col)

    # 표준화 (SVM 최적화 안정성 확보)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 레이블 변환: 0/1 -> -1/+1
    y_pm1 = to_svm_labels(y01)

    model = LinearSVM_SMO(C=C, tol=1e-3, eps=1e-5, max_passes=15, seed=42)
    model.fit(Xs, y_pm1)

    pred_pm1 = model.predict_pm1(Xs)
    pred01 = [from_svm_label(v) for v in pred_pm1]

    acc = accuracy(y01, pred01)
    cm = confusion_matrix(y01, pred01)
    n_sv = model.support_vector_count()

    print("=== TRAIN SUMMARY ===")
    print("train file      :", train_path)
    print("features        :", used_features)
    print("label           :", label_col)
    print("params          :", model.get_params())
    print(f"train accuracy  : {acc*100:.2f}%")
    print("confusion matrix [[TN,FP],[FN,TP]] =", cm)
    print(f"support vectors : {n_sv} / {len(y01)} ({n_sv/len(y01)*100:.2f}%)")
    print("w =", model.w)
    print("b =", model.b)
    print("=====================")
    return model, scaler, used_features


def predict_test(model, scaler, test_path, feature_cols=None):
    Xt, used_features = read_csv_x(test_path, feature_cols=feature_cols)
    Xts = scaler.transform(Xt)
    pred_pm1 = model.predict_pm1(Xts)
    pred01 = [from_svm_label(v) for v in pred_pm1]

    print("\n=== TEST PREDICTIONS ===")
    print("test file :", test_path)
    print("features  :", used_features)
    print("index,predicted_converted")
    for i, p in enumerate(pred01):
        print(f"{i},{p}")
    print("========================")
    return pred01


def main():
    default_label = "converted"
    default_features = ["response_time_min", "quote_amount_10k", "qna_count"]

    train_path = "train_set.csv"
    test_path = "test_set.csv"

    if not os.path.exists(train_path):
        print(f"[ERROR] 학습 파일이 없습니다: {train_path}")
        return

    model, scaler, used_features = train_and_eval(
        train_path=train_path,
        label_col=default_label,
        feature_cols=default_features,
        C=1.0
    )

    if os.path.exists(test_path):
        predict_test(model, scaler, test_path, feature_cols=default_features)
    else:
        print("[INFO] test_set.csv가 없어서 테스트 예측은 생략합니다.")


if __name__ == "__main__":
    main()
