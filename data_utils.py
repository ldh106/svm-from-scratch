# data_utils.py
# 데이터 로딩 / 표준화 / 평가 유틸 (외부 라이브러리 미사용)
import csv
import math


def read_csv_xy(path, feature_cols=None, label_col=None):
    """
    학습용 CSV에서 입력 특성 행렬 X와 레이블 벡터 y를 분리해 반환한다.
    - feature_cols가 None이면 label_col을 제외한 나머지 전부를 feature로 사용
    - label_col은 반드시 지정해야 함 (지도학습이므로 y가 무엇인지 명시 필요)
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("CSV header가 없습니다.")
        if label_col is None:
            raise ValueError("label_col을 지정해야 합니다. 예: label_col='converted'")
        if feature_cols is None:
            feature_cols = [h for h in headers if h != label_col]

        X, y = [], []
        for row in reader:
            X.append([float(row[c]) for c in feature_cols])
            y.append(int(float(row[label_col])))
        return X, y, feature_cols


def read_csv_x(path, feature_cols=None):
    """
    테스트용 CSV(레이블 없음)에서 입력 X만 읽는다.
    feature_cols가 None이면 헤더 전체를 feature로 사용한다.
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("CSV header가 없습니다.")
        if feature_cols is None:
            feature_cols = headers

        X = []
        for row in reader:
            X.append([float(row[c]) for c in feature_cols])
        return X, feature_cols


class StandardScaler:
    """
    각 feature(열)를 평균 0, 표준편차 1로 표준화한다.
    x' = (x - mean) / std
    SVM은 스케일에 민감하므로, 학습 안정성 확보를 위해 사용.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        n = len(X)
        d = len(X[0])
        self.mean_ = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        self.std_ = []
        for j in range(d):
            var = sum((X[i][j] - self.mean_[j]) ** 2 for i in range(n)) / n
            self.std_.append(math.sqrt(var) if var > 0 else 1.0)
        return self

    def transform(self, X):
        return [
            [(x[j] - self.mean_[j]) / self.std_[j] for j in range(len(x))]
            for x in X
        ]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def to_svm_labels(y01):
    """0/1 -> -1/+1 변환 (SVM 이론 표준 레이블)"""
    return [1 if v == 1 else -1 for v in y01]


def from_svm_label(y_pm1):
    """-1/+1 -> 0/1 변환"""
    return 1 if y_pm1 >= 0 else 0


def accuracy(y_true01, y_pred01):
    if not y_true01:
        return 0.0
    correct = sum(1 for t, p in zip(y_true01, y_pred01) if t == p)
    return correct / len(y_true01)


def confusion_matrix(y_true01, y_pred01):
    # [[TN, FP], [FN, TP]]
    tn = fp = fn = tp = 0
    for t, p in zip(y_true01, y_pred01):
        if t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tp += 1
    return [[tn, fp], [fn, tp]]
