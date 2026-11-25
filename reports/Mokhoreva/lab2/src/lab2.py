import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

K0, K1 = 3.0, -8.0
SAMPLE = np.array([[K0, K0], [K0, K1], [K1, K0], [K1, K1]], dtype=float)
RAW_Y = np.array([K0, K1, K1, K0], dtype=float)
CLS = (RAW_Y == K1).astype(float)                      # 0 для c0, 1 для c1

ETA = 0.5
EP_LIM = 50_000
TOL_MSE = 0.001
TOL_BCE = 0.1
SEEDS = [0, 1, 2, 3, 4]
MSE_LO, MSE_HI = 0.1, 0.9

IN = SAMPLE / 10.0
sig = lambda v: 1.0 / (1.0 + np.exp(-v))

def back_B(o): return K0 + o * (K1 - K0)
def back_A(o): return K0 + (o - MSE_LO) / (MSE_HI - MSE_LO) * (K1 - K0)

Y_MSE = MSE_LO + CLS * (MSE_HI - MSE_LO)
Y_BCE = CLS


class TinyNet:
    def __init__(self, gen):
        self.V1 = gen.uniform(-0.5, 0.5, (2, 2)); self.c1 = gen.uniform(-0.5, 0.5, 2)
        self.V2 = gen.uniform(-0.5, 0.5, 2);      self.c2 = gen.uniform(-0.5, 0.5)

    def feed(self, x):
        self.hid = sig(x @ self.V1 + self.c1)
        self.out = sig(self.hid @ self.V2 + self.c2)
        return self.out

    def learn(self, x, goal, eta, kind):
        o = self.feed(x)
        d_o = (o - goal) * o * (1 - o) if kind == "mse" else (o - goal)
        d_h = d_o * self.V2 * self.hid * (1 - self.hid)
        self.V2 -= eta * d_o * self.hid; self.c2 -= eta * d_o
        self.V1 -= eta * np.outer(x, d_h); self.c1 -= eta * d_h


def err_mse(n, G):
    return 0.5 * sum((G[i] - n.feed(IN[i])) ** 2 for i in range(4))

def err_bce(n, G):
    eps = 1e-12; acc = 0.0
    for i in range(4):
        p = np.clip(n.feed(IN[i]), eps, 1 - eps)
        acc += -(G[i] * np.log(p) + (1 - G[i]) * np.log(1 - p))
    return acc

VARIANTS = {
    "А (MSE)": dict(kind="mse", goal=Y_MSE, err=err_mse, tol=TOL_MSE, back=back_A),
    "Б (BCE)": dict(kind="bce", goal=Y_BCE, err=err_bce, tol=TOL_BCE, back=back_B),
}


def run_training(cfg, seed):
    gen = np.random.default_rng(seed)
    n = TinyNet(gen); curve = []
    for e in range(1, EP_LIM + 1):
        for i in gen.permutation(4):
            n.learn(IN[i], cfg["goal"][i], ETA, cfg["kind"])
        es = float(cfg["err"](n, cfg["goal"])); curve.append(es)
        if es <= cfg["tol"]:
            return n, e, es, True, curve
    return n, EP_LIM, es, False, curve


def scores(n, cfg):
    vals = np.array([cfg["back"](n.feed(IN[i])) for i in range(4)])
    mae = float(np.mean(np.abs(vals - RAW_Y)))
    acc = np.mean([(abs(v - K0) < abs(v - K1)) == (y == K0) for v, y in zip(vals, RAW_Y)])
    return mae, float(acc)

def pick(y): return "c0" if abs(y - K0) < abs(y - K1) else "c1"


runs = {k: [] for k in VARIANTS}
for k, cfg in VARIANTS.items():
    for s in SEEDS:
        net, ep, es, done, curve = run_training(cfg, s)
        mae, acc = scores(net, cfg)
        runs[k].append(dict(seed=s, net=net, ep=ep, es=es, done=done, curve=curve, mae=mae, acc=acc))

print(f"Пороги: MSE={TOL_MSE}, BCE={TOL_BCE}; скорость обучения={ETA}; макс. эпох={EP_LIM}\n")
for k, lst in runs.items():
    print(f"=== Конфигурация {k} ===")
    print(f"{'seed':>4} {'эпох':>7} {'Es':>10} {'сошлась':>8} {'accuracy':>9} {'MAE':>8}")
    for r in lst:
        print(f"{r['seed']:>4} {r['ep']:>7} {r['es']:>10.5f} {('да' if r['done'] else 'НЕТ'):>8} "
              f"{r['acc']*100:>8.0f}% {r['mae']:>8.3f}")
    ok_eps = [r["ep"] for r in lst if r["done"]]
    print(f"Сошлось: {len(ok_eps)} из {len(lst)}", end="")
    if ok_eps:
        print(f"; эпох: min={min(ok_eps)}, max={max(ok_eps)}, "
              f"среднее={np.mean(ok_eps):.0f}, std={np.std(ok_eps):.0f}")
    else:
        print()
    print()


def best(lst):
    ok = [r for r in lst if r["done"]] or lst
    return sorted(ok, key=lambda r: r["ep"])[len(ok) // 2]

rep = {k: best(v) for k, v in runs.items()}
print("Представительные запуски:", {k: f"seed={r['seed']}, эпох={r['ep']}" for k, r in rep.items()})

palette = {"А (MSE)": "tab:blue", "Б (BCE)": "tab:red"}

fig, ax = plt.subplots(figsize=(8, 5))
for k, r in rep.items():
    ax.plot(range(1, len(r["curve"]) + 1), r["curve"], color=palette[k],
            label=f"{k}, seed={r['seed']}")
    ax.axhline(VARIANTS[k]["tol"], color=palette[k], ls=":", alpha=.7,
               label=f"порог {k} = {VARIANTS[k]['tol']}")
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xlabel("Номер эпохи"); ax.set_ylabel("Суммарная ошибка Es (лог. шкала)")
ax.set_title("Сходимость: MSE vs BCE"); ax.legend(); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("fig1_convergence.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 5)); w = 0.38
for j, (k, lst) in enumerate(runs.items()):
    xs = np.arange(len(lst)) + (j - 0.5) * w
    for x, r in zip(xs, lst):
        ax.bar(x, r["ep"], w, color=palette[k],
               hatch=None if r["done"] else "//", edgecolor="black",
               alpha=1 if r["done"] else .45)
        ax.text(x, r["ep"], str(r["ep"]) + ("" if r["done"] else "\nне сошлась"),
                ha="center", va="bottom", fontsize=7)
ax.set_xticks(np.arange(len(SEEDS))); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
ax.set_yscale("log"); ax.set_ylabel("Число эпох (лог. шкала)"); ax.set_xlabel("Запуск")
ax.set_title("Устойчивость сходимости (штриховка — критерий не достигнут)")
ax.legend(handles=[Patch(color=palette[k], label=k) for k in runs] +
          [Patch(facecolor="white", edgecolor="black", hatch="//", label="не сошлась")])
ax.grid(alpha=.3, axis="y"); fig.tight_layout(); fig.savefig("fig2_epochs.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
grid_v = np.linspace(-10, 10, 200); GA, GB = np.meshgrid(grid_v, grid_v)
stacked = np.stack([GA / 10, GB / 10], axis=-1)
lo, hi = min(K0, K1), max(K0, K1)
for ax, (k, r) in zip(axs[:2], rep.items()):
    net = r["net"]
    Hmap = sig(stacked @ net.V1 + net.c1); Zmap = sig(Hmap @ net.V2 + net.c2)
    Zscaled = np.clip(VARIANTS[k]["back"](Zmap), lo, hi)
    cs = ax.contourf(GA, GB, Zscaled, levels=np.linspace(lo, hi, 23), cmap="RdYlBu", vmin=lo, vmax=hi)
    ax.contour(GA, GB, Zscaled, levels=[(K0 + K1) / 2], colors="k", linewidths=1.5)
    for a, b, y in zip(SAMPLE[:, 0], SAMPLE[:, 1], RAW_Y):
        ax.scatter(a, b, s=140, c="tab:green" if y == K0 else "tab:purple",
                   edgecolors="k", zorder=3, marker="o" if y == K0 else "s")
    ax.set_xlabel("A"); ax.set_ylabel("B"); ax.set_title(f"{k}, seed={r['seed']}")
    fig.colorbar(cs, ax=ax, label="выход в шкале [c0; c1]")
axs[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color="tab:green", mec="k", label=f"класс c0 ({K0})"),
                       plt.Line2D([], [], marker="s", ls="", color="tab:purple", mec="k", label=f"класс c1 ({K1})"),
                       plt.Line2D([], [], color="k", label="граница классов")], loc="upper right", fontsize=8)
mae_avg = {k: np.mean([r["mae"] for r in v if r["done"]] or [r["mae"] for r in v]) for k, v in runs.items()}
bars = axs[2].bar(list(mae_avg), list(mae_avg.values()), color=[palette[k] for k in mae_avg])
for b_, v in zip(bars, mae_avg.values()):
    axs[2].text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
axs[2].set_ylabel("Средняя |ошибка| в шкале [c0; c1]")
axs[2].set_title("MAE в шкале [c0; c1]\n(среднее по сошедшимся запускам)")
fig.tight_layout(); fig.savefig("fig3_surface_mae.png", dpi=150); plt.close()


def report(net, cfg, a, b):
    out = float(net.feed(np.array([a / 10, b / 10])))
    unit = (out - MSE_LO) / (MSE_HI - MSE_LO) if cfg["kind"] == "mse" else out
    real = cfg["back"](out)
    print(f"A={a:>6}, B={b:>6} | ŷ(0;1)={unit:.4f} | в шкале [c0;c1]={real:8.4f} | ближе к {pick(real)}")

net_b, cfg_b = rep["Б (BCE)"]["net"], VARIANTS["Б (BCE)"]
print("\n--- Режим функционирования (конфигурация Б, BCE) ---")
print("Обучающие примеры:")
for a, b in zip(SAMPLE[:, 0], SAMPLE[:, 1]): report(net_b, cfg_b, a, b)
print("Дополнительные пары:")
for a, b in [(5, -2), (9, 9), (-6, 4), (0, 0)]: report(net_b, cfg_b, a, b)

print("\nВведите свои значения. Пустая строка — выход.")
while True:
    try: s = input("Введите A и B из [-10; 10] через пробел: ").strip()
    except EOFError: break
    if not s: break
    try:
        a, b = map(float, s.split()); assert -10 <= a <= 10 and -10 <= b <= 10
    except Exception: print("Некорректный ввод. Нужны два числа из [-10; 10]."); continue
    report(net_b, cfg_b, a, b)