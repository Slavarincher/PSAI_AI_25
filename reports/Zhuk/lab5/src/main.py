from pathlib import Path
import csv
import math
import random
import sys

import numpy as np
import matplotlib.pyplot as plt


#   НАСТРОЙКИ ВАРИАНТА  

SURNAME = "Жук"
VARIANT = 5
H_BASE = 4
LEARNING_RATE = 0.1
EE = 0.05
MAX_EPOCHS = 5000
M_MAX = 8
NOISE_TRIALS = 50
REJECTION_THRESHOLD = 0.6

# В датасете записаны технические метки c1/c2/c3.
# Если известны реальные буквы, замените значения, например: {"c1": "А", ...}.
CLASS_NAMES = {"c1": "c1", "c2": "c2", "c3": "c3"}
CLASS_IDS = ["c1", "c2", "c3"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_IDS)}

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.txt"
DATASET2_PATH = BASE_DIR / "dataset2.txt"
RESULTS_DIR = BASE_DIR / "results"


#  ДАТАСЕТ

def parse_dataset(path):
    """Читает датасет вручную заданного формата и возвращает список объектов."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    objects = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        parts = line.replace(" ", "").split(",")
        if len(parts) != 2 or parts[0] not in CLASS_TO_INDEX:
            raise ValueError(
                f"{path.name}, строка {i + 1}: ожидался заголовок вида 'c1, 1', "
                f"получено: {lines[i]!r}"
            )

        class_id = parts[0]
        try:
            instance_id = int(parts[1])
        except ValueError:
            raise ValueError(f"{path.name}, строка {i + 1}: неверный номер экземпляра.")

        if i + 7 >= len(lines):
            raise ValueError(f"{path.name}, объект {class_id}, {instance_id}: недостаточно строк.")

        grid_lines = [lines[i + 1 + k].strip() for k in range(7)]
        for row_number, row in enumerate(grid_lines, start=1):
            if len(row) != 5 or any(ch not in "#." for ch in row):
                raise ValueError(
                    f"{path.name}, объект {class_id}, {instance_id}, строка сетки {row_number}: "
                    "нужно ровно 5 символов '#' или '.'."
                )

        grid = np.array([[1 if ch == "#" else 0 for ch in row] for row in grid_lines],
                        dtype=np.int8)
        objects.append({
            "class_id": class_id,
            "label": CLASS_TO_INDEX[class_id],
            "instance": instance_id,
            "grid": grid,
            "x": grid.reshape(35).astype(float)
        })
        i += 8

    if not objects:
        raise ValueError(f"В файле {path.name} нет объектов.")
    return objects


def read_dataset(path=DATASET_PATH):
    """Читает основной датасет; поддерживает загруженное имя dataset (1).txt."""
    path = Path(path)
    if not path.exists() and path.name == "dataset.txt":
        alternative = path.with_name("dataset (1).txt")
        if alternative.exists():
            path = alternative
    objects = parse_dataset(path)

    seen = set()
    for obj in objects:
        key = (obj["class_id"], obj["instance"])
        if key in seen:
            raise ValueError(f"Повторяется объект {key}.")
        seen.add(key)

    found_classes = {obj["class_id"] for obj in objects}
    if found_classes != set(CLASS_IDS):
        raise ValueError(f"Ожидаются классы {CLASS_IDS}, найдены {sorted(found_classes)}.")
    return objects


def check_dataset_quality(objects):
    """Считает Nfill и средние попарные расстояния Хэмминга."""
    nfill = sum(int(np.sum(obj["grid"])) for obj in objects)
    same_distances = []
    different_distances = []

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            distance = int(np.sum(objects[i]["x"] != objects[j]["x"]))
            if objects[i]["label"] == objects[j]["label"]:
                same_distances.append(distance)
            else:
                different_distances.append(distance)

    same_mean = float(np.mean(same_distances)) if same_distances else float("nan")
    different_mean = float(np.mean(different_distances)) if different_distances else float("nan")

    print("\nКОНТРОЛЬ КАЧЕСТВА ДАТАСЕТА")
    print(f"Количество объектов: {len(objects)}")
    print(f"Nfill (общее число закрашенных клеток): {nfill}")
    print(f"Среднее расстояние Хэмминга внутри класса: {same_mean:.3f}")
    print(f"Среднее расстояние Хэмминга между классами: {different_mean:.3f}")
    if same_mean < 3:
        print("ПРЕДУПРЕЖДЕНИЕ: внутриклассовое среднее меньше 3. "
              "Проверьте и при необходимости перепишите похожие экземпляры.")
    return nfill, same_mean, different_mean


def split_dataset(objects):
    """Экземпляры 1–7 каждого класса — обучение, 8–10 — тест."""
    train = [o for o in objects if 1 <= o["instance"] <= 7]
    test = [o for o in objects if 8 <= o["instance"] <= 10]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Для разбиения нужны экземпляры 1–7 и 8–10.")
    for class_id in CLASS_IDS:
        tr_count = sum(o["class_id"] == class_id for o in train)
        te_count = sum(o["class_id"] == class_id for o in test)
        if tr_count == 0 or te_count == 0:
            raise ValueError(f"В классе {class_id} отсутствует train- или test-часть.")
    return train, test


def objects_to_arrays(objects):
    x = np.array([o["x"] for o in objects], dtype=float)
    y = np.array([o["label"] for o in objects], dtype=int)
    return x, y


# НЕЙРОННАЯ СЕТЬ 35-H-3  

def sigmoid(z):
    """Сигмоида скрытого слоя."""
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """Самостоятельная численно устойчивая реализация softmax."""
    z = np.asarray(z, dtype=float)
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def make_target(label):
    """Ручное формирование целевого вектора без one-hot из библиотек."""
    target = np.zeros(3, dtype=float)
    target[int(label)] = 1.0
    return target


def initialize_network(hidden_size, seed):
    """Инициализация весов и смещений."""
    rng = np.random.default_rng(seed)
    # Небольшая случайная инициализация; входы имеют значения 0/1.
    w1 = rng.uniform(-0.35, 0.35, size=(35, hidden_size))
    b1 = np.zeros(hidden_size, dtype=float)
    w2 = rng.uniform(-0.35, 0.35, size=(hidden_size, 3))
    b2 = np.zeros(3, dtype=float)
    return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}


def forward_pass(x, network):
    """Прямой проход: 35 -> H (sigmoid) -> 3 (softmax)."""
    hidden_net = np.asarray(x) @ network["W1"] + network["b1"]
    hidden = sigmoid(hidden_net)
    output_net = hidden @ network["W2"] + network["b2"]
    probabilities = softmax(output_net)
    return {
        "input": np.asarray(x),
        "hidden_net": hidden_net,
        "hidden": hidden,
        "output_net": output_net,
        "probabilities": probabilities
    }


def cross_entropy(probabilities, target):
    """Кросс-энтропия для одного объекта."""
    p = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(target * np.log(p)))


def backward_pass(network, cache, target):
    """Обратный проход; возвращает градиенты CE по всем параметрам."""
    p = cache["probabilities"]
    hidden = cache["hidden"]

    # Для softmax + cross-entropy: dL/dz_output = p - target.
    delta_output = p - target
    grad_w2 = np.outer(hidden, delta_output)
    grad_b2 = delta_output

    delta_hidden = (network["W2"] @ delta_output) * hidden * (1.0 - hidden)
    grad_w1 = np.outer(cache["input"], delta_hidden)
    grad_b1 = delta_hidden

    return {"W1": grad_w1, "b1": grad_b1, "W2": grad_w2, "b2": grad_b2}


def update_network(network, gradients, learning_rate):
    """Шаг стохастического градиентного спуска."""
    for key in ("W1", "b1", "W2", "b2"):
        network[key] -= learning_rate * gradients[key]


def calculate_dataset_errors(network, x, y):
    """Считает среднюю ошибку Es (MSE) и среднюю CE на выборке."""
    squared_errors = []
    ce_values = []
    for sample, label in zip(x, y):
        p = forward_pass(sample, network)["probabilities"]
        target = make_target(label)
        squared_errors.append(0.5 * float(np.sum((target - p) ** 2)))
        ce_values.append(cross_entropy(p, target))
    return float(np.mean(squared_errors)), float(np.mean(ce_values))


def train_network(x, y, hidden_size=H_BASE, learning_rate=LEARNING_RATE,
                  seed=1, max_epochs=MAX_EPOCHS, ee=EE, verbose=False):
    """Онлайн-обучение с перемешиванием примеров в каждой эпохе."""
    network = initialize_network(hidden_size, seed)
    rng = np.random.default_rng(seed + 100_003)
    es_history = []
    ce_history = []
    stopped = False

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(x))
        for index in order:
            cache = forward_pass(x[index], network)
            target = make_target(y[index])
            gradients = backward_pass(network, cache, target)
            update_network(network, gradients, learning_rate)

        es, ce = calculate_dataset_errors(network, x, y)
        es_history.append(es)
        ce_history.append(ce)

        if verbose and (epoch == 1 or epoch % 250 == 0):
            print(f"Эпоха {epoch:4d}: Es={es:.6f}, CE={ce:.6f}")

        if es <= ee:
            stopped = True
            break

    return {
        "network": network,
        "epochs": len(es_history),
        "Es": es_history[-1],
        "CE": ce_history[-1],
        "es_history": es_history,
        "ce_history": ce_history,
        "stopped": stopped,
        "hidden_size": hidden_size,
        "seed": seed
    }


#   МЕТРИКИ 

def predict_one(network, x):
    probabilities = forward_pass(x, network)["probabilities"]
    return int(np.argmax(probabilities)), probabilities


def confusion_matrix(y_true, y_pred, number_of_classes=3):
    """Самостоятельное построение матрицы ошибок."""
    matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def calculate_metrics(network, x, y, objects=None):
    """Accuracy, матрица ошибок, уверенность на верных и неверных ответах."""
    predictions = []
    confidences = []
    correct_confidences = []
    wrong_confidences = []
    probabilities = []

    for sample, true_label in zip(x, y):
        predicted, p = predict_one(network, sample)
        confidence = float(np.max(p))
        predictions.append(predicted)
        confidences.append(confidence)
        probabilities.append(p)
        if predicted == int(true_label):
            correct_confidences.append(confidence)
        else:
            wrong_confidences.append(confidence)

    predictions = np.array(predictions, dtype=int)
    accuracy = float(np.mean(predictions == y))
    matrix = confusion_matrix(y, predictions)
    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "confidences": np.array(confidences),
        "probabilities": np.array(probabilities),
        "matrix": matrix,
        "mean_confidence_correct": (float(np.mean(correct_confidences))
                                    if correct_confidences else float("nan")),
        "mean_confidence_wrong": (float(np.mean(wrong_confidences))
                                  if wrong_confidences else float("nan")),
        "correct_count": len(correct_confidences),
        "wrong_count": len(wrong_confidences)
    }


def print_metrics(title, metrics, y_true):
    print(f"\n{title}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Матрица ошибок (строки — истинный класс, столбцы — предсказанный):")
    print(metrics["matrix"])
    print(f"Средняя уверенность на верных ответах: "
          f"{metrics['mean_confidence_correct']:.4f}")
    print(f"Средняя уверенность на неверных ответах: "
          f"{metrics['mean_confidence_wrong']:.4f}")


def rejection_statistics(network, x, y=None, title="Выборка"):
    """Считает отказы. Для объектов с известной меткой — верные отказы."""
    rejected = 0
    rejected_would_be_correct = 0
    details = []
    for i, sample in enumerate(x):
        predicted, p = predict_one(network, sample)
        confidence = float(np.max(p))
        is_rejected = confidence < REJECTION_THRESHOLD
        if is_rejected:
            rejected += 1
            if y is not None and predicted == int(y[i]):
                rejected_would_be_correct += 1
        details.append((predicted, confidence, is_rejected, p))
    print(f"{title}: отказано {rejected} из {len(x)}.")
    if y is not None:
        print(f"Среди отвергнутых правильных классификаций было бы: "
              f"{rejected_would_be_correct}.")
    else:
        print("Правильность классификации неизвестна: истинный класс не задан.")
    return rejected, rejected_would_be_correct, details


#  ЧИСЛЕННЫЙ ГРАДИЕНТ

def loss_for_sample(network, x, label):
    p = forward_pass(x, network)["probabilities"]
    return cross_entropy(p, make_target(label))


def numerical_gradient(network, x, label, epsilon=1e-5):
    """Численный градиент центральной разностью для проверки backprop."""
    numerical = {}
    for key in ("W1", "b1", "W2", "b2"):
        numerical[key] = np.zeros_like(network[key])
        iterator = np.nditer(network[key], flags=["multi_index"], op_flags=["readwrite"])
        while not iterator.finished:
            index = iterator.multi_index
            original = network[key][index]
            network[key][index] = original + epsilon
            loss_plus = loss_for_sample(network, x, label)
            network[key][index] = original - epsilon
            loss_minus = loss_for_sample(network, x, label)
            network[key][index] = original
            numerical[key][index] = (loss_plus - loss_minus) / (2.0 * epsilon)
            iterator.iternext()
    return numerical


def check_gradients():
    """Сравнивает аналитический и численный градиенты на маленькой сети."""
    rng = np.random.default_rng(123)
    network = initialize_network(2, 123)
    sample = rng.integers(0, 2, size=35).astype(float)
    label = 1
    cache = forward_pass(sample, network)
    analytical = backward_pass(network, cache, make_target(label))
    numerical = numerical_gradient(network, sample, label)
    max_difference = max(float(np.max(np.abs(analytical[k] - numerical[k])))
                         for k in analytical)
    print(f"\nПроверка градиента (центральная разность): "
          f"максимальное расхождение = {max_difference:.10e}")
    if max_difference > 1e-5:
        print("ПРЕДУПРЕЖДЕНИЕ: проверьте реализацию обратного прохода.")
    else:
        print("Проверка градиента пройдена.")
    return max_difference


#  ИССЛЕДОВАНИЕ H 

def analyze_hidden_layer(x_train, y_train, hidden_values=(2, 4, 8),
                         seeds=(11, 22, 33, 44, 55)):
    """По пять запусков для каждого H; сохраняет историю первого запуска."""
    all_results = {}
    for hidden_size in hidden_values:
        runs = []
        for seed in seeds:
            print(f"Обучение: H={hidden_size}, seed={seed}")
            run = train_network(x_train, y_train, hidden_size=hidden_size,
                                learning_rate=LEARNING_RATE, seed=seed)
            runs.append(run)
        all_results[hidden_size] = runs

    print("\nИССЛЕДОВАНИЕ ЧИСЛА СКРЫТЫХ НЕЙРОНОВ")
    print("H | эпохи среднее (min–max) | test accuracy среднее (min–max) "
          "| не достигли Ee | параметры")
    for hidden_size, runs in all_results.items():
        epoch_values = np.array([r["epochs"] for r in runs], dtype=float)
        accuracies = []
        for run in runs:
            metrics = calculate_metrics(run["network"], X_TEST_GLOBAL, Y_TEST_GLOBAL)
            accuracies.append(metrics["accuracy"] * 100.0)
        accuracies = np.array(accuracies, dtype=float)
        not_converged = sum(not r["stopped"] for r in runs)
        parameter_count = 35 * hidden_size + hidden_size + hidden_size * 3 + 3
        print(f"{hidden_size:2d} | {np.mean(epoch_values):.1f} "
              f"({np.min(epoch_values):.0f}–{np.max(epoch_values):.0f}) | "
              f"{np.mean(accuracies):.2f}% "
              f"({np.min(accuracies):.2f}–{np.max(accuracies):.2f}) | "
              f"{not_converged} | {parameter_count}")
    return all_results


# Глобальные ссылки используются только в анализе H для тестовой accuracy.
X_TEST_GLOBAL = None
Y_TEST_GLOBAL = None
TRAIN_OBJECTS_GLOBAL = []


#  УСТОЙЧИВОСТЬ К ШУМУ 

def add_noise(grid_or_vector, number_of_cells, rng):
    """Инвертирует m различных случайных клеток одного примера."""
    result = np.array(grid_or_vector, dtype=np.int8).reshape(35).copy()
    if number_of_cells < 0 or number_of_cells > 35:
        raise ValueError("Число изменяемых клеток должно быть от 0 до 35.")
    if number_of_cells:
        indices = rng.choice(35, size=number_of_cells, replace=False)
        result[indices] = 1 - result[indices]
    return result


def test_noise_robustness(network, test_objects, m_max=M_MAX,
                          trials=NOISE_TRIALS, seed=2026):
    """Для каждого m — 50 независимых зашумлений каждого тестового примера."""
    rng = np.random.default_rng(seed)
    overall = []
    per_class = {class_id: [] for class_id in CLASS_IDS}

    for m in range(m_max + 1):
        all_correct = []
        class_correct = {class_id: [] for class_id in CLASS_IDS}
        for obj in test_objects:
            for _ in range(trials):
                noisy = add_noise(obj["grid"], m, rng)
                predicted, _ = predict_one(network, noisy.astype(float))
                correct = int(predicted == obj["label"])
                all_correct.append(correct)
                class_correct[obj["class_id"]].append(correct)
        overall.append(float(np.mean(all_correct)))
        for class_id in CLASS_IDS:
            per_class[class_id].append(float(np.mean(class_correct[class_id])))

    below = next((m for m, accuracy in enumerate(overall) if accuracy < 0.80), None)
    print("\nУСТОЙЧИВОСТЬ К ШУМУ")
    for m, accuracy in enumerate(overall):
        print(f"m={m}: accuracy={accuracy * 100:.2f}%")
    if below is None:
        print("На диапазоне m=0..m_max accuracy не опустилась ниже 80%.")
    else:
        print(f"Первое m, при котором accuracy ниже 80%: {below}")
    return np.array(overall), {k: np.array(v) for k, v in per_class.items()}, below


#  ВИЗУАЛИЗАЦИЯ  

def plot_confusion_matrix(matrix, nfill):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), [CLASS_NAMES[c] for c in CLASS_IDS])
    ax.set_yticks(range(3), [CLASS_NAMES[c] for c in CLASS_IDS])
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(f"{SURNAME}, вариант {VARIANT}, Nfill={nfill}\nМатрица ошибок тестовой выборки")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.show()


def plot_convergence(hidden_results, nfill):
    fig, ax = plt.subplots(figsize=(9, 6))
    for hidden_size, runs in hidden_results.items():
        run = runs[0]
        epochs = np.arange(1, len(run["es_history"]) + 1)
        ax.plot(epochs, run["es_history"], label=f"Es, H={hidden_size}")
        ax.plot(epochs, run["ce_history"], linestyle="--",
                label=f"CE, H={hidden_size}")
    ax.set_yscale("log")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Значение ошибки (логарифмическая шкала)")
    ax.set_title(f"{SURNAME}, вариант {VARIANT}, Nfill={nfill}\nСходимость для разных H")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence.png", dpi=160)
    plt.show()


def plot_epoch_scatter(hidden_results, nfill):
    fig, ax = plt.subplots(figsize=(8, 5))
    for position, (hidden_size, runs) in enumerate(hidden_results.items()):
        values = [run["epochs"] for run in runs]
        x_values = np.full(len(values), position, dtype=float)
        # Небольшой горизонтальный сдвиг точек, чтобы они не перекрывались.
        x_values += np.linspace(-0.08, 0.08, len(values))
        ax.scatter(x_values, values, label=f"H={hidden_size}", s=55)
    ax.set_xticks(range(len(hidden_results)), [f"H={h}" for h in hidden_results])
    ax.set_ylabel("Количество эпох")
    ax.set_title(f"{SURNAME}, вариант {VARIANT}, Nfill={nfill}\nРазброс числа эпох по пяти запускам")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "epochs_scatter.png", dpi=160)
    plt.show()


def plot_noise_curve(overall, per_class, nfill):
    fig, ax = plt.subplots(figsize=(8, 5))
    m_values = np.arange(len(overall))
    ax.plot(m_values, overall * 100, marker="o", linewidth=2, label="Общая accuracy")
    for class_id in CLASS_IDS:
        ax.plot(m_values, per_class[class_id] * 100, marker="s",
                label=f"Класс {CLASS_NAMES[class_id]}")
    ax.axhline(80, linestyle="--", label="Порог 80%")
    ax.set_xticks(m_values)
    ax.set_xlabel("Количество инвертированных клеток m")
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(0, 105)
    ax.set_title(f"{SURNAME}, вариант {VARIANT}, Nfill={nfill}\nУстойчивость к шуму")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "noise_accuracy.png", dpi=160)
    plt.show()


def plot_weight_maps(network, nfill):
    weights = network["W1"].T.reshape(-1, 7, 5)
    hidden_count = weights.shape[0]
    fig, axes = plt.subplots(1, hidden_count, figsize=(3.2 * hidden_count, 4))
    if hidden_count == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        max_abs = float(np.max(np.abs(weights[j])))
        if max_abs == 0:
            max_abs = 1.0
        image = ax.imshow(weights[j], cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        ax.set_title(f"Нейрон {j + 1}")
        ax.set_xlabel("Столбец")
        ax.set_ylabel("Строка")
        ax.set_xticks(range(5))
        ax.set_yticks(range(7))
        fig.colorbar(image, ax=ax, fraction=0.046)
        flat_order = np.argsort(np.abs(weights[j]).reshape(-1))[::-1]
        top = []
        for index in flat_order[:5]:
            row, col = divmod(int(index), 5)
            top.append(f"({row + 1},{col + 1})")
        signs = np.sign(weights[j])
        # Интерпретация по знакам входных весов:
        # положительный вес поддерживает закрашенную клетку, отрицательный —
        # пустую. Сравниваем этот шаблон со средними образами каждого класса.
        class_scores = []
        for class_index in range(3):
            class_samples = [o["x"] for o in TRAIN_OBJECTS_GLOBAL
                             if o["label"] == class_index]
            prototype = np.mean(class_samples, axis=0)
            signed_template = 2.0 * prototype - 1.0
            class_scores.append(float(np.sum(weights[j].reshape(35) * signed_template)))
        likely_class = int(np.argmax(class_scores))
        print(f"Скрытый нейрон {j + 1}: по знакам входных весов больше "
              f"соответствует классу {CLASS_NAMES[CLASS_IDS[likely_class]]}; "
              f"5 максимальных по модулю клеток: {', '.join(top)}")
        print(f"  Знак входных весов по сетке:\n{np.where(signs.reshape(7, 5) >= 0, '+', '-')}")
    fig.suptitle(f"{SURNAME}, вариант {VARIANT}, Nfill={nfill}\nКарты весов скрытого слоя")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "weight_maps.png", dpi=160)
    plt.show()


def draw_grid(ax, grid, title):
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.7)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=9)


def show_uncertain_examples(network, test_objects, nfill):
    ranked = []
    for obj in test_objects:
        predicted, probabilities = predict_one(network, obj["x"])
        ranked.append((float(np.max(probabilities)), obj, predicted, probabilities))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[:3]

    fig, axes = plt.subplots(1, len(selected), figsize=(10, 4))
    if len(selected) == 1:
        axes = [axes]
    print("\nТРИ САМЫХ НЕУВЕРЕННЫХ ТЕСТОВЫХ ПРИМЕРА")
    for i, (confidence, obj, predicted, p) in enumerate(selected):
        title = (f"{obj['class_id']} #{obj['instance']}\n"
                 f"предсказано {CLASS_IDS[predicted]}, уверенность={confidence:.4f}")
        draw_grid(axes[i], obj["grid"], title)
        print(f"{obj['class_id']} #{obj['instance']}: "
              f"p1={p[0]:.4f}, p2={p[1]:.4f}, p3={p[2]:.4f}; "
              f"предсказание={CLASS_IDS[predicted]}")
    fig.suptitle(f"{SURNAME}, вариант {VARIANT}, Nfill={nfill}\nСамые неуверенные примеры")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "uncertain_examples.png", dpi=160)
    plt.show()
    return selected


# ЧУЖОЙ ПОЧЕРК И БУКВА

def test_foreign_handwriting(network, path=DATASET2_PATH):
    """Проверяет dataset2.txt без дообучения."""
    if not Path(path).exists():
        print(f"\nФайл {Path(path).name} пока отсутствует. "
              "Проверка чужого почерка будет пропущена.")
        return None

    objects = parse_dataset(path)
    x, y = objects_to_arrays(objects)
    metrics = calculate_metrics(network, x, y)
    print_metrics("КРОСС-ТЕСТ НА ЧУЖОМ ПОЧЕРКЕ (без дообучения)", metrics, y)
    rejection_statistics(network, x, y, "Чужой почерк")
    print("\nПримеры чужого почерка:")
    for obj in objects[:2]:
        predicted, p = predict_one(network, obj["x"])
        print(f"{obj['class_id']} #{obj['instance']} -> {CLASS_IDS[predicted]}, "
              f"вероятности: {np.round(p, 4)}")
    return objects, metrics


def read_grid_from_console():
    """Запрашивает у пользователя 7 строк по 5 символов."""
    print("Введите 7 строк сетки по 5 символов (# — закрашено, . — пусто).")
    rows = []
    for row_number in range(1, 8):
        while True:
            row = input(f"Строка {row_number}: ").strip()
            if len(row) == 5 and all(ch in "#." for ch in row):
                rows.append(row)
                break
            print("Ошибка: введите ровно 5 символов '#' или '.'.")
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows],
                    dtype=np.int8)


def print_recognition_result(network, grid, true_label=None, title="Распознавание"):
    predicted, probabilities = predict_one(network, grid.reshape(35).astype(float))
    confidence = float(np.max(probabilities))
    print(f"\n{title}")
    print("\n".join("".join("#" if cell else "." for cell in row) for row in grid))
    print(f"p1={probabilities[0]:.4f}, p2={probabilities[1]:.4f}, "
          f"p3={probabilities[2]:.4f}")
    print(f"Предполагаемый класс: {CLASS_NAMES[CLASS_IDS[predicted]]}")
    print(f"Уверенность: {confidence:.4f}")
    if confidence < REJECTION_THRESHOLD:
        print("Итоговое решение: не распознано")
    else:
        print(f"Итоговое решение: {CLASS_NAMES[CLASS_IDS[predicted]]}")
    if true_label is not None:
        print(f"Истинный класс: {CLASS_NAMES[CLASS_IDS[int(true_label)]]}")
    return predicted, probabilities


def recognition_mode(network):
    """Интерактивный режим распознавания."""
    while True:
        answer = input("\nВвести сетку для распознавания? (y/n): ").strip().lower()
        if answer not in ("д", "да", "y", "yes"):
            break
        grid = read_grid_from_console()
        print_recognition_result(network, grid)


#  ДЕМОНСТРАЦИОННЫЕ ПРИМЕРЫ 

def demonstrate_examples(network, train_objects, test_objects, foreign_objects=None):
    print("\nДЕМОНСТРАЦИЯ НА ОБУЧАЮЩИХ ПРИМЕРАХ")
    for obj in train_objects[:3]:
        print_recognition_result(network, obj["grid"], obj["label"],
                                 f"Обучающий пример {obj['class_id']} #{obj['instance']}")

    print("\nДЕМОНСТРАЦИЯ НА ТЕСТОВЫХ ПРИМЕРАХ")
    for obj in test_objects[:3]:
        print_recognition_result(network, obj["grid"], obj["label"],
                                 f"Тестовый пример {obj['class_id']} #{obj['instance']}")

    print("\nДЕМОНСТРАЦИЯ НА ДВУХ ИСКАЖЁННЫХ ПРИМЕРАХ")
    rng = np.random.default_rng(909)
    for obj, m in zip(test_objects[:2], (2, 5)):
        noisy = add_noise(obj["grid"], m, rng).reshape(7, 5)
        print(f"Число изменённых клеток: {m}")
        print_recognition_result(network, noisy, obj["label"], f"Искажённый пример, m={m}")

    if foreign_objects:
        print("\nДЕМОНСТРАЦИЯ ЧУЖОГО ПОЧЕРКА")
        for obj in foreign_objects[:2]:
            print_recognition_result(network, obj["grid"], obj["label"],
                                     f"Чужой почерк {obj['class_id']} #{obj['instance']}")


def ask_foreign_letter(network):
    print("\nПРОВЕРКА ЧУЖОЙ БУКВЫ")
    print("Нарисуйте сетку 7×5 для буквы, которой нет среди трёх классов.")
    grid = read_grid_from_console()
    predicted, p = print_recognition_result(network, grid, title="Чужая буква")
    rejection_statistics(network, grid.reshape(1, 35), y=None,
                         title="Чужая буква")
    print(f"Сеть отнесла её к {CLASS_IDS[predicted]} с вероятностью {np.max(p):.4f}.")
    print("Правильность классификации чужой буквы без заданной истинной метки "
          "автоматически определить нельзя.")


#  СОХРАНЕНИЕ РЕЗУЛЬТАТОВ 

def save_experiment_summary(path, nfill, quality, base_result, train_metrics,
                            test_metrics, hidden_results, noise_result):
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Фамилия: {SURNAME}\nВариант: {VARIANT}\nNfill: {nfill}\n")
        file.write(f"Среднее расстояние внутри класса: {quality[0]:.4f}\n")
        file.write(f"Среднее расстояние между классами: {quality[1]:.4f}\n\n")
        file.write("БАЗОВАЯ СЕТЬ\n")
        file.write(f"H={H_BASE}, эпохи={base_result['epochs']}, "
                   f"Es={base_result['Es']:.6f}, CE={base_result['CE']:.6f}\n")
        file.write(f"Train accuracy={train_metrics['accuracy'] * 100:.2f}%\n")
        file.write(f"Test accuracy={test_metrics['accuracy'] * 100:.2f}%\n")
        file.write(f"Train confusion matrix:\n{train_metrics['matrix']}\n")
        file.write(f"Test confusion matrix:\n{test_metrics['matrix']}\n\n")
        file.write("ИССЛЕДОВАНИЕ H\n")
        for h, runs in hidden_results.items():
            epochs = [r["epochs"] for r in runs]
            accuracies = [calculate_metrics(r["network"], X_TEST_GLOBAL,
                                            Y_TEST_GLOBAL)["accuracy"] * 100
                          for r in runs]
            file.write(f"H={h}: epochs mean={np.mean(epochs):.2f}, "
                       f"range={min(epochs)}–{max(epochs)}; "
                       f"test accuracy mean={np.mean(accuracies):.2f}%, "
                       f"range={min(accuracies):.2f}–{max(accuracies):.2f}; "
                       f"не достигли Ee={sum(not r['stopped'] for r in runs)}; "
                       f"параметров={35*h+h+3*h+3}\n")
        overall, per_class, below = noise_result
        file.write("\nУСТОЙЧИВОСТЬ К ШУМУ\n")
        for m, acc in enumerate(overall):
            file.write(f"m={m}: {acc * 100:.2f}%\n")
        file.write(f"Первое m ниже 80%: {below}\n")


#  MAIN 

def main():
    global X_TEST_GLOBAL, Y_TEST_GLOBAL, TRAIN_OBJECTS_GLOBAL

    RESULTS_DIR.mkdir(exist_ok=True)

    objects = read_dataset()
    nfill, same_mean, different_mean = check_dataset_quality(objects)
    train_objects, test_objects = split_dataset(objects)
    TRAIN_OBJECTS_GLOBAL = train_objects
    x_train, y_train = objects_to_arrays(train_objects)
    x_test, y_test = objects_to_arrays(test_objects)
    X_TEST_GLOBAL, Y_TEST_GLOBAL = x_test, y_test

    print(f"\nОбучающая выборка: {len(x_train)} объектов.")
    print(f"Тестовая выборка: {len(x_test)} объектов.")
    print(f"Параметры варианта: H={H_BASE}, шаг={LEARNING_RATE}, m_max={M_MAX}.")
    check_gradients()

    print("\nОБУЧЕНИЕ БАЗОВОЙ СЕТИ")
    base_result = train_network(x_train, y_train, hidden_size=H_BASE,
                                learning_rate=LEARNING_RATE, seed=42, verbose=True)
    base_network = base_result["network"]
    print(f"\nБазовая сеть: эпох={base_result['epochs']}, "
          f"Es={base_result['Es']:.6f}, CE={base_result['CE']:.6f}, "
          f"критерий достигнут: {'да' if base_result['stopped'] else 'нет'}")

    train_metrics = calculate_metrics(base_network, x_train, y_train)
    test_metrics = calculate_metrics(base_network, x_test, y_test)
    print_metrics("МЕТРИКИ ОБУЧАЮЩЕЙ ВЫБОРКИ", train_metrics, y_train)
    print_metrics("МЕТРИКИ ТЕСТОВОЙ ВЫБОРКИ", test_metrics, y_test)

    hidden_values = (2, H_BASE, 2 * H_BASE)
    hidden_results = analyze_hidden_layer(x_train, y_train, hidden_values)
    noise_result = test_noise_robustness(base_network, test_objects)

    plot_convergence(hidden_results, nfill)
    plot_epoch_scatter(hidden_results, nfill)
    plot_confusion_matrix(test_metrics["matrix"], nfill)
    plot_noise_curve(noise_result[0], noise_result[1], nfill)
    plot_weight_maps(base_network, nfill)
    show_uncertain_examples(base_network, test_objects, nfill)

    foreign_result = test_foreign_handwriting(base_network)
    foreign_objects = foreign_result[0] if foreign_result else None
    if foreign_objects:
        demonstrate_examples(base_network, train_objects, test_objects, foreign_objects)
        fx, fy = objects_to_arrays(foreign_objects)
        rejection_statistics(base_network, fx, fy, "Чужой почерк")
    else:
        demonstrate_examples(base_network, train_objects, test_objects)

    rejection_statistics(base_network, x_test, y_test, "Тестовая выборка")
    save_experiment_summary(
        RESULTS_DIR / "summary.txt", nfill, (same_mean, different_mean),
        base_result, train_metrics, test_metrics, hidden_results, noise_result
    )

    print(f"\nРезультаты и графики сохранены в папке: {RESULTS_DIR}")
    while True:
        choice = input("\nВведите 'o' — чужая буква, 'f' — режим распознавания, "
                       "'e' — выход: ").strip().lower()
        if choice in ("o", "other"):
            ask_foreign_letter(base_network)
        elif choice in ("f", "functioning"):
            recognition_mode(base_network)
        elif choice in ("e", "exit", ""):
            break
        else:
            print("Неизвестная команда.")

    print("Работа программы завершена.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nОШИБКА: {error}")
        print("Проверьте структуру dataset.txt и наличие всех необходимых файлов.")
        sys.exit(1)
