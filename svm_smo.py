# svm_smo.py
# Soft-margin Linear SVM을 SMO(Sequential Minimal Optimization)로 학습.
# sklearn.svm 등 기성 SVM 함수를 사용하지 않고, 표준 라이브러리만으로 구현.
import random


def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def add_scaled(w, x, scale):
    # w += scale * x  (증분 업데이트용)
    for i in range(len(w)):
        w[i] += scale * x[i]


class LinearSVM_SMO:
    """
    Soft-margin Linear SVM (선형 커널)을 SMO로 학습한다.

    최적화 대상은 Dual Problem:
        max_a  sum(a_i) - 1/2 * sum_i sum_j a_i a_j y_i y_j K(x_i, x_j)
        s.t.   0 <= a_i <= C,  sum(a_i y_i) = 0

    한 번에 두 개의 라그랑주 승수(a_i, a_j)만 선택해 반복적으로 갱신하며,
    KKT 조건을 위반하는 샘플을 우선적으로 갱신 대상으로 삼는다.
    """

    def __init__(self, C=1.0, tol=1e-3, eps=1e-5, max_passes=15, seed=42):
        self.C = float(C)
        self.tol = float(tol)
        self.eps = float(eps)
        self.max_passes = int(max_passes)
        self.seed = int(seed)

        self.alphas = None
        self.b = 0.0
        self.w = None
        self.X = None
        self.y = None

    def decision_function_one(self, x):
        # f(x) = w^T x + b
        return dot(self.w, x) + self.b

    def _E(self, i):
        # E_i = f(x_i) - y_i
        return self.decision_function_one(self.X[i]) - self.y[i]

    def fit(self, X, y_pm1):
        """
        X: list[list[float]]
        y_pm1: list[int]  (값이 -1 또는 +1)
        """
        random.seed(self.seed)

        m = len(X)
        if m == 0:
            raise ValueError("X가 비어있습니다.")
        n_features = len(X[0])

        self.X = X
        self.y = y_pm1
        self.alphas = [0.0] * m
        self.b = 0.0
        self.w = [0.0] * n_features

        passes = 0
        while passes < self.max_passes:
            num_changed = 0

            for i in range(m):
                Ei = self._E(i)
                yi = self.y[i]
                ai = self.alphas[i]

                # KKT 조건 위반 여부 검사
                if (yi * Ei < -self.tol and ai < self.C) or (yi * Ei > self.tol and ai > 0.0):

                    # i와 다른 j를 랜덤 선택
                    j = i
                    while j == i:
                        j = random.randrange(0, m)

                    Ej = self._E(j)
                    yj = self.y[j]
                    aj = self.alphas[j]

                    ai_old = ai
                    aj_old = aj

                    # alpha_j의 허용 범위 [L, H]
                    if yi != yj:
                        L = max(0.0, aj - ai)
                        H = min(self.C, self.C + aj - ai)
                    else:
                        L = max(0.0, ai + aj - self.C)
                        H = min(self.C, ai + aj)

                    if L == H:
                        continue

                    # 선형 커널: K(xi, xj) = xi · xj
                    Kii = dot(self.X[i], self.X[i])
                    Kjj = dot(self.X[j], self.X[j])
                    Kij = dot(self.X[i], self.X[j])

                    eta = 2.0 * Kij - Kii - Kjj
                    if eta >= 0.0:
                        continue

                    # alpha_j 갱신 + clipping
                    aj_new = aj_old - (yj * (Ei - Ej)) / eta
                    if aj_new > H:
                        aj_new = H
                    elif aj_new < L:
                        aj_new = L

                    if abs(aj_new - aj_old) < self.eps:
                        continue

                    # alpha_i 갱신 (결합 제약 유지)
                    ai_new = ai_old + yi * yj * (aj_old - aj_new)

                    # b 갱신
                    b_old = self.b
                    b1 = b_old - Ei - yi * (ai_new - ai_old) * Kii - yj * (aj_new - aj_old) * Kij
                    b2 = b_old - Ej - yi * (ai_new - ai_old) * Kij - yj * (aj_new - aj_old) * Kjj

                    if 0.0 < ai_new < self.C:
                        b_new = b1
                    elif 0.0 < aj_new < self.C:
                        b_new = b2
                    else:
                        b_new = 0.5 * (b1 + b2)

                    self.alphas[i] = ai_new
                    self.alphas[j] = aj_new

                    # w는 매번 전체 합을 다시 계산하지 않고, 변화량만 증분 반영
                    # w = sum(alpha_i * y_i * x_i)
                    add_scaled(self.w, self.X[i], (ai_new - ai_old) * yi)
                    add_scaled(self.w, self.X[j], (aj_new - aj_old) * yj)

                    self.b = b_new
                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        return self

    def predict_pm1(self, X):
        preds = []
        for x in X:
            val = dot(self.w, x) + self.b
            preds.append(1 if val >= 0.0 else -1)
        return preds

    def support_vector_count(self, threshold=1e-6):
        """alpha_i > threshold 인 데이터 개수 (서포트 벡터 개수)"""
        return sum(1 for a in self.alphas if a > threshold)

    def get_params(self):
        return {"C": self.C, "tol": self.tol, "eps": self.eps, "max_passes": self.max_passes}
