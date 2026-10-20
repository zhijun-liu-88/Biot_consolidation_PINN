import os

import deepxde as dde
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

dde.backend.set_default_backend("pytorch")


OUTPUT_DIR = "precision_compare_outputs"
SEED = 42
ADAM_ITERS = 15000

s, minute, hr, day = 1.0, 60.0, 60.0**2, 24 * 60.0**2
g, kg = 1.0e-3, 1.0
mm, cm, m, km = 1.0e-3, 1.0e-2, 1.0, 1.0e3
Pa, kPa, MPa, GPa = 1.0, 1.0e3, 1.0e6, 1.0e9
micro, milli, centi, kilo, mega = 1.0e-6, 1.0e-3, 1.0e-2, 1.0e3, 1.0e6
poise = 0.1 * Pa * s
Darcy = 9.869233e-13 * m**2

S_overburden = -0.1 * MPa
P_ini = 0.0 * MPa
mu = 6.0 * milli * poise
k = 1e-12
E = 120.0 * MPa
nu = 0.25
b = 1.0
M = 3e11
Lx = 1 * m
Ly = 50 * m

Lmbd = E * nu / (1 + nu) / (1 - 2 * nu)
Gshr = E / 2 / (1 + nu)
Kdr = E / (3 * (1 - 2 * nu))
B = b * M / (Kdr + M * (b**2))
nu_u = (3 * nu + b * B * (1 - 2 * nu)) / (3 - b * B * (1 - 2 * nu))

p_star = (-1) * S_overburden
x_star = Ly
t_star = (b**2 / Kdr + 1 / M) * mu * (x_star**2) / k
nu_star = (1 - 2 * nu) / (1 + nu)
u_star = p_star * x_star / Kdr

td_0 = 0.0
td_f = 15000.0 / t_star
yd_min = 0.0
yd_max = Ly / x_star
NUM_SAMPLES = 20000


def configure_precision(dtype_name):
    if dtype_name == "float32":
        dtype = torch.float32
    elif dtype_name == "float64":
        dtype = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    dde.config.set_default_float(dtype_name)
    dde.config.set_random_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_default_dtype(dtype)
    return dtype


def pde_system(X, Y):
    p_t = dde.grad.jacobian(Y, X, i=0, j=1)
    p_y = dde.grad.jacobian(Y, X, i=0, j=0)
    p_yy = dde.grad.hessian(Y, X, component=0, i=0, j=0)
    v_yy = dde.grad.hessian(Y, X, component=1, i=0, j=0)

    Mp = p_t - p_yy
    Msy = (1 + 2 * nu_star) * v_yy - b * p_y
    return [Mp, Msy]


def build_model():
    geom = dde.geometry.Interval(yd_min, yd_max)
    timedomain = dde.geometry.TimeDomain(td_0, td_f)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    n_b = 2500
    n_i = 2500
    eps_y = 1e-3

    y_bottom = np.full((n_b, 1), yd_min)
    t_bottom = np.random.uniform(td_0, td_f, (n_b, 1))
    x_bottom = np.hstack((y_bottom, t_bottom))

    y_top = np.full((n_b, 1), yd_max)
    t_top = np.random.uniform(td_0, td_f, (n_b, 1))
    x_top = np.hstack((y_top, t_top))

    y_init = np.random.uniform(yd_min, yd_max - eps_y, (n_i, 1))
    t_init = np.full((n_i, 1), td_0)
    x_init = np.hstack((y_init, t_init))

    bcs = [
        dde.PointSetBC(x_top, np.zeros((n_b, 1)), component=0),
        dde.PointSetOperatorBC(
            x_bottom,
            np.zeros((n_b, 1)),
            lambda X, Y, _: dde.grad.jacobian(Y, X, i=0, j=0),
        ),
        dde.PointSetBC(x_bottom, np.zeros((n_b, 1)), component=1),
        dde.PointSetOperatorBC(
            x_top,
            np.zeros((n_b, 1)),
            lambda X, Y, _: (1 + 2 * nu_star) * dde.grad.jacobian(Y, X, i=1, j=0)
            - b * Y[:, 0:1]
            - S_overburden / p_star,
        ),
        dde.PointSetBC(x_init, np.ones((n_i, 1)), component=0),
        dde.PointSetBC(x_init, np.zeros((n_i, 1)), component=1),
    ]

    data = dde.data.TimePDE(
        geomtime,
        pde_system,
        bcs,
        num_domain=int(NUM_SAMPLES / 2),
        num_test=5000,
    )

    net = dde.nn.FNN([2] + 4 * [32] + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    return model


def build_evaluation_grids():
    ny = 100
    nt = 200
    ys = np.linspace(yd_min, yd_max, ny * 3)
    ts = (np.logspace(0.0, np.log10(1.0 + td_f * t_star), nt * 3) - 1.0) / t_star

    yd_test, td_test = np.meshgrid(ys, ts, indexing="ij")
    ys_line = np.linspace(yd_min, yd_max, ny * 3)
    ts_line = np.array(
        [
            td_f * 0.01,
            td_f * 0.025,
            td_f * 0.05,
            td_f * 0.1,
            td_f * 0.25,
            td_f * 0.5,
            td_f * 0.75,
            td_f * 1.0,
        ]
    )
    yd_line, td_line = np.meshgrid(ys_line, ts_line, indexing="ij")
    return yd_test, td_test, yd_line, td_line, ys_line, ts_line


def analytic_solution(yd_test, td_test, yd_line, td_line):
    p_true_contour = np.zeros_like(yd_test)
    p_true_line = np.zeros_like(yd_line)

    for i in range(100):
        ci = 4 * (-1) ** i / (np.pi * (2 * i + 1))
        lisq = np.pi * (2 * i + 1) / 2
        p_true_contour += ci * np.cos(lisq * yd_test) * np.exp(-(lisq**2) * td_test)
        p_true_line += ci * np.cos(lisq * yd_line) * np.exp(-(lisq**2) * td_line)

    f2_contour = np.zeros_like(yd_test)
    f2_line = np.zeros_like(yd_line)
    for i in range(100):
        ci = 8 / ((np.pi * (2 * i + 1)) ** 2)
        lisq = np.pi * (2 * i + 1) / 2
        f2_contour += ci * np.cos(lisq * (1 - yd_test)) * (1 - np.exp(-(lisq**2) * td_test))
        f2_line += ci * np.cos(lisq * (1 - yd_line)) * (1 - np.exp(-(lisq**2) * td_line))

    term1_contour = S_overburden * Ly * (1 - 2 * nu_u) * yd_test / (2 * Gshr * (1 - nu_u))
    term2 = S_overburden * Ly * (nu_u - nu) / (2 * Gshr * (1 - nu_u) * (1 - nu))
    v_true_contour = term1_contour + term2 * f2_contour

    term1_line = S_overburden * Ly * (1 - 2 * nu_u) * yd_line / (2 * Gshr * (1 - nu_u))
    v_true_line = term1_line + term2 * f2_line

    return {
        "p_true_contour": p_true_contour * p_star / MPa,
        "p_true_line": p_true_line * p_star / MPa,
        "v_true_contour": v_true_contour,
        "v_true_line": v_true_line,
    }


def train_and_evaluate(dtype_name):
    configure_precision(dtype_name)
    model = build_model()

    model.compile("adam", lr=1e-3)
    model.train(iterations=ADAM_ITERS, display_every=1000)
    model.compile("L-BFGS")
    losshistory, _ = model.train()

    steps = np.asarray(losshistory.steps, dtype=float)
    loss_train = np.asarray(losshistory.loss_train, dtype=float)
    total_loss = np.sum(loss_train, axis=1)

    yd_test, td_test, yd_line, td_line, ys_line, ts_line = build_evaluation_grids()
    truth = analytic_solution(yd_test, td_test, yd_line, td_line)

    input_test = np.hstack((yd_test.reshape(-1, 1), td_test.reshape(-1, 1)))
    input_line = np.hstack((yd_line.reshape(-1, 1), td_line.reshape(-1, 1)))

    pred_all = model.predict(input_test)
    pred_line = model.predict(input_line)

    ny_grid = yd_line.shape[0]
    nt_line = yd_line.shape[1]

    p_grid = pred_all[:, 0:1].reshape(yd_test.shape) * p_star / MPa
    v_grid = pred_all[:, 1:2].reshape(yd_test.shape) * u_star / m
    p_line = pred_line[:, 0:1].reshape(ny_grid, nt_line) * p_star / MPa
    v_line = pred_line[:, 1:2].reshape(ny_grid, nt_line) * u_star / m

    err_p = np.linalg.norm(p_grid - truth["p_true_contour"]) / np.linalg.norm(truth["p_true_contour"])
    err_v = np.linalg.norm(v_grid - truth["v_true_contour"]) / np.linalg.norm(truth["v_true_contour"])

    return {
        "dtype_name": dtype_name,
        "steps": steps,
        "total_loss": total_loss,
        "p_grid": p_grid,
        "v_grid": v_grid,
        "p_line": p_line,
        "v_line": v_line,
        "truth": truth,
        "yd_test": yd_test,
        "td_test": td_test,
        "ys_line": ys_line,
        "ts_line": ts_line,
        "err_p": err_p,
        "err_v": err_v,
    }


def plot_loss(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in results:
        ax.plot(
            result["steps"],
            result["total_loss"],
            linewidth=1.8,
            label=f'{result["dtype_name"]} total loss',
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Total training loss")
    ax.set_title("PINN loss comparison: float32 vs float64")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_compare_32_64.png"), dpi=200)
    plt.close(fig)


def plot_line_results(results):
    truth = results[0]["truth"]
    ys_line = results[0]["ys_line"] * x_star
    ts_line = results[0]["ts_line"]
    selected_indices = [0, 3, 7]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, idx in enumerate(selected_indices):
        time_label = f"t={ts_line[idx] * t_star:.2f}s"

        ax_p = axes[0, col]
        ax_p.plot(ys_line, truth["p_true_line"][:, idx], color="black", linewidth=2.0, label="exact")
        ax_p.plot(ys_line, results[0]["p_line"][:, idx], linestyle="--", linewidth=1.8, label="float32")
        ax_p.plot(ys_line, results[1]["p_line"][:, idx], linestyle="-.", linewidth=1.8, label="float64")
        ax_p.set_title(f"Pressure, {time_label}")
        ax_p.set_ylabel("p (MPa)")
        ax_p.grid(True, linestyle="--", alpha=0.4)

        ax_v = axes[1, col]
        ax_v.plot(ys_line, truth["v_true_line"][:, idx], color="black", linewidth=2.0, label="exact")
        ax_v.plot(ys_line, results[0]["v_line"][:, idx], linestyle="--", linewidth=1.8, label="float32")
        ax_v.plot(ys_line, results[1]["v_line"][:, idx], linestyle="-.", linewidth=1.8, label="float64")
        ax_v.set_title(f"Displacement, {time_label}")
        ax_v.set_xlabel("y (m)")
        ax_v.set_ylabel("v (m)")
        ax_v.grid(True, linestyle="--", alpha=0.4)

    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "line_compare_32_64.png"), dpi=200)
    plt.close(fig)


def plot_contours(results):
    truth = results[0]["truth"]
    yd_test = results[0]["yd_test"]
    td_test = results[0]["td_test"]
    y_plot = yd_test * x_star
    t_plot = np.log10(np.maximum(td_test * t_star, 1e-12))

    p_vmin = np.min(truth["p_true_contour"])
    p_vmax = np.max(truth["p_true_contour"])
    v_vmin = np.min(truth["v_true_contour"])
    v_vmax = np.max(truth["v_true_contour"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    contour_specs = [
        (truth["p_true_contour"], "Exact p [MPa]", "jet_r", p_vmin, p_vmax),
        (results[0]["p_grid"], "float32 p [MPa]", "jet_r", p_vmin, p_vmax),
        (results[1]["p_grid"], "float64 p [MPa]", "jet_r", p_vmin, p_vmax),
        (truth["v_true_contour"], "Exact v [m]", "jet_r", v_vmin, v_vmax),
        (results[0]["v_grid"], "float32 v [m]", "jet_r", v_vmin, v_vmax),
        (results[1]["v_grid"], "float64 v [m]", "jet_r", v_vmin, v_vmax),
    ]

    for ax, (field, title, cmap, vmin, vmax) in zip(axes.flat, contour_specs):
        im = ax.pcolor(t_plot, y_plot, field, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("log10(t) (s)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(-1.5, np.max(t_plot))
        ax.set_ylim(np.min(y_plot), np.max(y_plot))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "contour_compare_32_64.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = [train_and_evaluate("float32"), train_and_evaluate("float64")]
    plot_loss(results)
    plot_line_results(results)
    plot_contours(results)

    summary_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        for result in results:
            fh.write(
                f'{result["dtype_name"]}: '
                f'rel_l2_p={result["err_p"]:.6e}, '
                f'rel_l2_v={result["err_v"]:.6e}\n'
            )

    for result in results:
        print(
            f'{result["dtype_name"]}: '
            f'rel_l2_p={result["err_p"]:.6e}, '
            f'rel_l2_v={result["err_v"]:.6e}'
        )
    print(f"Saved figures and metrics to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
