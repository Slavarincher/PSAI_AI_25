import numpy as np
import matplotlib.pyplot as plt

SURNAME = "Трошин"
VARIANT = 13
LETTERS = ["Т", "Р", "О"]

ROWS, COLS = 7, 5
N_IN = ROWS * COLS
N_OUT = 3

H_BASE = 4
ETA = 0.10
M_MAX = 6
EE = 0.05
MAX_EPOCHS = 5000
TAU = 0.6
N_RUNS = 5
N_NOISE = 50

#данные
def read_dataset(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    X, labels, insts = [], [], []
    for i in range(0, len(lines), ROWS + 1):
        cls, inst = map(int, lines[i].split())
        grid = lines[i + 1:i + 1 + ROWS]
        X.append(grid_to_vector(grid))
        labels.append(cls - 1)
        insts.append(inst)
    return np.array(X), np.array(labels), np.array(insts)

def grid_to_vector(grid):
    return np.array([1.0 if ch == "#" else 0.0 for row in grid for ch in row])

def vector_to_grid(x):
    return ["".join("#" if x[r * COLS + c] > 0.5 else "." for c in range(COLS))
            for r in range(ROWS)]

def print_grid(x):
    for row in vector_to_grid(x):
        print(row)

def one_hot(labels, n):
    Y = np.zeros((len(labels), n))
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

def dataset_quality(X, labels):
    """N_fill и средние попарные расстояния Хэмминга (внутри классов и между ними)."""
    n_fill = int(X.sum())
    D = np.sum(X[:, None, :] != X[None, :, :], axis=2)
    upper = np.triu(np.ones_like(D, dtype=bool), 1)
    same = labels[:, None] == labels[None, :]
    intra = D[same & upper].mean()
    inter = D[~same & upper].mean()
    print(f"N_fill = {n_fill}")
    print(f"Среднее расстояние Хэмминга внутри классов: {intra:.3f}")
    print(f"Среднее расстояние Хэмминга между классами: {inter:.3f}")
    for c in range(N_OUT):
        m = (labels[:, None] == c) & (labels[None, :] == c)
        print(f"  внутри класса {LETTERS[c]}: {D[m & upper].mean():.3f}")
    for a in range(N_OUT):
        for b in range(a + 1, N_OUT):
            m = (labels[:, None] == a) & (labels[None, :] == b)
            print(f"  между {LETTERS[a]} и {LETTERS[b]}: {D[m].mean():.3f}")
    if intra < 3:
        print("Внутриклассовое среднее меньше 3: экземпляры слишком похожи")
    return n_fill

#сеть
def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

class MLP:
    def __init__(self, n_hidden, rng):
        self.W1 = rng.uniform(-0.5, 0.5, (n_hidden, N_IN))
        self.T1 = rng.uniform(-0.5, 0.5, n_hidden)
        self.W2 = rng.uniform(-0.5, 0.5, (N_OUT, n_hidden))
        self.T2 = rng.uniform(-0.5, 0.5, N_OUT)

    def params(self):
        return [self.W1, self.T1, self.W2, self.T2]

    def n_params(self):
        return sum(p.size for p in self.params())

    def forward(self, x):
        h = sigmoid(self.W1 @ x - self.T1)
        p = softmax(self.W2 @ h - self.T2)
        return h, p

    def backward(self, x, y, h, p):
        dz = p - y                      # dCE/dz для softmax + кросс-энтропии
        ds = (self.W2.T @ dz) * h * (1.0 - h)
        return [np.outer(ds, x), -ds, np.outer(dz, h), -dz]

def cross_entropy(p, y):
    return -np.sum(y * np.log(p + 1e-12))

def evaluate(net, X):
    return np.array([net.forward(x)[1] for x in X])

def errors(net, X, Y):
    """Es = среднее по выборке 0.5*sum((y-p)^2); CE – средняя кросс-энтропия."""
    P = evaluate(net, X)
    es = np.mean(0.5 * np.sum((Y - P) ** 2, axis=1))
    ce = np.mean(-np.sum(Y * np.log(P + 1e-12), axis=1))
    return es, ce

def train(net, X, Y, rng):
    hist_es, hist_ce = [], []
    for epoch in range(1, MAX_EPOCHS + 1):
        for i in rng.permutation(len(X)):
            h, p = net.forward(X[i])
            grads = net.backward(X[i], Y[i], h, p)
            for prm, g in zip(net.params(), grads):
                prm -= ETA * g
        es, ce = errors(net, X, Y)
        hist_es.append(es)
        hist_ce.append(ce)
        if es <= EE:
            return hist_es, hist_ce, epoch, True
    return hist_es, hist_ce, MAX_EPOCHS, False

#метрики
def predict(net, X):
    return np.argmax(evaluate(net, X), axis=1)

def accuracy(true, pred):
    return float(np.mean(true == pred))

def confusion_matrix(true, pred):
    cm = np.zeros((N_OUT, N_OUT), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm

def confidence_stats(net, X, labels):
    P = evaluate(net, X)
    conf = P.max(axis=1)
    ok = np.argmax(P, axis=1) == labels
    mean_ok = conf[ok].mean() if ok.any() else float("nan")
    mean_bad = conf[~ok].mean() if (~ok).any() else float("nan")
    return mean_ok, mean_bad

#численный градиент
def numeric_gradients(net, x, y, eps=1e-6):
    grads = []
    for prm in net.params():
        g = np.zeros_like(prm)
        for k in range(prm.size):
            old = prm.flat[k]
            prm.flat[k] = old + eps
            lp = cross_entropy(net.forward(x)[1], y)
            prm.flat[k] = old - eps
            lm = cross_entropy(net.forward(x)[1], y)
            prm.flat[k] = old
            g.flat[k] = (lp - lm) / (2 * eps)
        grads.append(g)
    return grads

def check_gradients(net, x, y):
    h, p = net.forward(x)
    analytic = net.backward(x, y, h, p)
    numeric = numeric_gradients(net, x, y)
    names = ["W1", "T1", "W2", "T2"]
    for name, a, n in zip(names, analytic, numeric):
        print(f"  {name}: max|аналитический - численный| = {np.max(np.abs(a - n)):.2e}")

#шум, отказ
def add_noise(x, m, rng):
    y = x.copy()
    idx = rng.choice(N_IN, m, replace=False)
    y[idx] = 1.0 - y[idx]
    return y

def noise_test(net, X, labels, rng):
    acc_all = np.zeros(M_MAX + 1)
    acc_cls = np.zeros((M_MAX + 1, N_OUT))
    for m in range(M_MAX + 1):
        correct = np.zeros(len(X))
        for i, x in enumerate(X):
            for _ in range(N_NOISE):
                _, p = net.forward(add_noise(x, m, rng))
                correct[i] += np.argmax(p) == labels[i]
        correct /= N_NOISE
        acc_all[m] = correct.mean()
        for c in range(N_OUT):
            acc_cls[m, c] = correct[labels == c].mean()
    return acc_all, acc_cls

def decide(p):
    """Решение с порогом отказа: None, если максимальная вероятность меньше TAU."""
    return None if p.max() < TAU else int(np.argmax(p))

def reject_stats(net, X, labels, name):
    P = evaluate(net, X)
    rejected = P.max(axis=1) < TAU
    would_be_ok = rejected & (np.argmax(P, axis=1) == labels)
    print(f"{name}: отвергнуто {rejected.sum()} из {len(X)}, "
          f"из них верно классифицировались бы {would_be_ok.sum()}")

#режим функционирования
def describe(net, x, title):
    _, p = net.forward(x)
    print(f"--- {title} ---")
    print_grid(x)
    print("Вероятности: " + ", ".join(f"p{k + 1} = {p[k]:.4f}" for k in range(N_OUT)))
    print(f"Класс: {LETTERS[int(np.argmax(p))]}, уверенность: {p.max():.4f}")
    d = decide(p)
    print("Решение: " + ("не распознано" if d is None else LETTERS[d]))


def functioning_mode(net):
    print(f"Введите {ROWS} строк по {COLS} символов (# и .):")
    grid = [input().strip() for _ in range(ROWS)]
    describe(net, grid_to_vector(grid), "Введённый пример")


#графики
def title_suffix(n_fill):
    return f"{SURNAME}, вариант {VARIANT}, N_fill = {n_fill}"


def plot_convergence(study, n_fill):
    plt.figure(figsize=(8, 5))
    for i, (h, runs) in enumerate(study.items()):
        es, ce = runs[0]["hist_es"], runs[0]["hist_ce"]
        plt.semilogy(range(1, len(es) + 1), es, color=f"C{i}", label=f"Es, H = {h}")
        plt.semilogy(range(1, len(ce) + 1), ce, color=f"C{i}", linestyle="--",
                     label=f"CE, H = {h}")
    plt.axhline(EE, color="gray", linestyle=":")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.title("Сходимость\n" + title_suffix(n_fill))
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("convergence.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_epochs_scatter(study, n_fill):
    plt.figure(figsize=(6, 5))
    for i, (h, runs) in enumerate(study.items()):
        plt.scatter([i] * len(runs), [r["epochs"] for r in runs], s=50)
    plt.xticks(range(len(study)), [f"H = {h}" for h in study])
    plt.ylabel("Число эпох")
    plt.title("Разброс числа эпох по 5 запускам\n" + title_suffix(n_fill))
    plt.grid(True, alpha=0.3)
    plt.savefig("epochs_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, n_fill):
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(N_OUT):
        for j in range(N_OUT):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.xticks(range(N_OUT), LETTERS)
    plt.yticks(range(N_OUT), LETTERS)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок (тест)\n" + title_suffix(n_fill))
    plt.savefig("confusion_test.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_noise(acc_all, acc_cls, n_fill):
    ms = range(M_MAX + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(ms, acc_all * 100, "k-o", linewidth=2.5, label="Все классы")
    for c in range(N_OUT):
        plt.plot(ms, acc_cls[:, c] * 100, "--o", label=f"Класс {LETTERS[c]}")
    plt.axhline(80, color="gray", linestyle=":")
    plt.xlabel("Число испорченных клеток m")
    plt.ylabel("Accuracy, %")
    plt.title("Устойчивость к шуму\n" + title_suffix(n_fill))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("noise_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_weight_maps(net, n_fill):
    h = net.W1.shape[0]
    vmax = np.abs(net.W1).max()
    fig, axes = plt.subplots(1, h, figsize=(3 * h, 4.2))
    for j, ax in enumerate(np.atleast_1d(axes)):
        im = ax.imshow(net.W1[j].reshape(ROWS, COLS), cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Нейрон {j + 1}")
        ax.set_xticks(range(COLS))
        ax.set_xticklabels(range(1, COLS + 1))
        ax.set_yticks(range(ROWS))
        ax.set_yticklabels(range(1, ROWS + 1))
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("Карты весов скрытых нейронов\n" + title_suffix(n_fill))
    fig.savefig("weight_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_uncertain(items, n_fill):
    fig, axes = plt.subplots(1, len(items), figsize=(3.5 * len(items), 4.5))
    for ax, (x, letter, inst, conf) in zip(np.atleast_1d(axes), items):
        ax.imshow(x.reshape(ROWS, COLS), cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"{letter} №{inst}, p_max = {conf:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Самые неуверенные тестовые примеры\n" + title_suffix(n_fill))
    fig.savefig("uncertain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


#основная программа
def main():
    X, labels, inst = read_dataset("dataset.txt")
    Y = one_hot(labels, N_OUT)
    tr, te = inst <= 7, inst >= 8
    Xtr, Ytr, ltr = X[tr], Y[tr], labels[tr]
    Xte, Yte, lte, ite = X[te], Y[te], labels[te], inst[te]

    print("Контроль качества датасета")
    n_fill = dataset_quality(X, labels)

    print("\nПроверка градиентов")
    check_gradients(MLP(H_BASE, np.random.default_rng(0)), Xtr[0], Ytr[0])

    #исследование по H (первый запуск H = H_BASE является базовым экспериментом)
    study = {}
    for h in (2, H_BASE, 2 * H_BASE):
        study[h] = []
        for seed in range(1, N_RUNS + 1):
            rng = np.random.default_rng(seed)
            net = MLP(h, rng)
            hist_es, hist_ce, epochs, reached = train(net, Xtr, Ytr, rng)
            study[h].append({
                "net": net, "hist_es": hist_es, "hist_ce": hist_ce,
                "epochs": epochs, "reached": reached,
                "test_acc": accuracy(lte, predict(net, Xte)),
            })

    base = study[H_BASE][0]
    net = base["net"]
    pred_tr, pred_te = predict(net, Xtr), predict(net, Xte)
    print(f"\nБазовый эксперимент (H = {H_BASE}, eta = {ETA})")
    print(f"Эпох: {base['epochs']}, критерий достигнут: {base['reached']}")
    print(f"Итоговая Es: {base['hist_es'][-1]:.5f}")
    print(f"Accuracy обучающая: {accuracy(ltr, pred_tr):.4f}")
    print(f"Accuracy тестовая: {accuracy(lte, pred_te):.4f}")
    cm_tr, cm_te = confusion_matrix(ltr, pred_tr), confusion_matrix(lte, pred_te)
    print("Матрица ошибок (обучающая), строки – истина, столбцы – предсказание:")
    print(cm_tr)
    print("Матрица ошибок (тестовая):")
    print(cm_te)
    ok, bad = confidence_stats(net, Xte, lte)
    print(f"Средняя уверенность на верных тестовых: {ok:.4f}, на неверных: {bad:.4f}")

    print("\nИсследование скрытого слоя")
    print("H | эпохи: среднее [min-max] | test acc: среднее [min-max] | не достигли | параметров")
    for h, runs in study.items():
        ep = np.array([r["epochs"] for r in runs])
        ac = np.array([r["test_acc"] for r in runs])
        fails = sum(not r["reached"] for r in runs)
        print(f"{h} | {ep.mean():.1f} [{ep.min()}-{ep.max()}] | "
              f"{ac.mean():.4f} [{ac.min():.4f}-{ac.max():.4f}] | {fails} | "
              f"{runs[0]['net'].n_params()}")

    rng = np.random.default_rng(100)

    print("\nУстойчивость к шуму")
    acc_all, acc_cls = noise_test(net, Xte, lte, rng)
    for m in range(M_MAX + 1):
        row = ", ".join(f"{LETTERS[c]}: {acc_cls[m, c]:.3f}" for c in range(N_OUT))
        print(f"m = {m}: accuracy = {acc_all[m]:.3f} ({row})")
    m80 = next((m for m in range(M_MAX + 1) if acc_all[m] < 0.8), None)
    print("Наименьшее m, при котором accuracy < 80 %: "
          + (str(m80) if m80 is not None else f"нет (до m = {M_MAX})"))
    fragile = int(np.argmin(acc_cls[1:].mean(axis=0)))
    print(f"Самая хрупкая буква: {LETTERS[fragile]}")

    print("\nКарты весов")
    for j in range(H_BASE):
        letter = LETTERS[int(np.argmax(net.W2[:, j]))]
        top = np.argsort(-np.abs(net.W1[j]))[:3]
        cells = ", ".join(f"({k // COLS + 1}, {k % COLS + 1}): {net.W1[j, k]:+.2f}" for k in top)
        v = ", ".join(f"v{k + 1}{j + 1} = {net.W2[k, j]:+.2f}" for k in range(N_OUT))
        print(f"Нейрон {j + 1}: работает на {letter}; наибольшие |w| (строка, столбец): {cells}; {v}")

    print("\nСамые неуверенные тестовые примеры")
    P_te = evaluate(net, Xte)
    order = np.argsort(P_te.max(axis=1))[:3]
    uncertain = []
    for i in order:
        print(f"Класс {LETTERS[lte[i]]}, экземпляр №{ite[i]}:")
        print_grid(Xte[i])
        print("Вероятности: " + ", ".join(f"{LETTERS[k]} = {P_te[i, k]:.4f}" for k in range(N_OUT)))
        uncertain.append((Xte[i], LETTERS[lte[i]], ite[i], P_te[i].max()))

    print("\nКросс-тест на чужом почерке")
    Xo, lo, _ = read_dataset("other.txt")
    acc_other = accuracy(lo, predict(net, Xo))
    print(f"Accuracy на чужом почерке: {acc_other:.4f}")
    print(f"Accuracy на своём тесте: {accuracy(lte, pred_te):.4f}")
    print(confusion_matrix(lo, predict(net, Xo)))

    print("\n«Чужая» буква и порог отказа")
    Xf, lf, _ = read_dataset("foreign.txt")
    _, pf = net.forward(Xf[0])
    print(f"Класс: {LETTERS[int(np.argmax(pf))]}, вероятность: {pf.max():.4f}")
    reject_stats(net, Xte, lte, "Тестовая выборка")
    reject_stats(net, Xo, lo, "Чужой почерк")
    reject_stats(net, Xf, lf, "«Чужая» буква")

    plot_convergence(study, n_fill)
    plot_epochs_scatter(study, n_fill)
    plot_confusion(cm_te, n_fill)
    plot_noise(acc_all, acc_cls, n_fill)
    plot_weight_maps(net, n_fill)
    plot_uncertain(uncertain, n_fill)

    print("\nРежим функционирования (демонстрация)")
    for c in range(N_OUT):
        describe(net, Xtr[ltr == c][0], f"Обучающий пример, класс {LETTERS[c]}")
    for c in range(N_OUT):
        describe(net, Xte[lte == c][0], f"Тестовый пример, класс {LETTERS[c]}")
    for src, m in ((Xte[0], 2), (Xte[4], 4)):
        describe(net, add_noise(src, m, rng), f"Искажённый пример, изменено клеток: {m}")
    describe(net, Xf[0], "«Чужая» буква")
    describe(net, Xo[0], "Чужой почерк, пример 1")
    describe(net, Xo[3], "Чужой почерк, пример 2")

    functioning_mode(net)


if __name__ == "__main__":
    main()
