# deepxde_terzaghi.py
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import  torch
import os
import json
from datetime import datetime
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

dde.config.set_default_float("float64")  # DeepXDE 内部计算使用 float64
torch.set_default_dtype(torch.float64)   # PyTorch 默认张量用 float64
SEED = 42  # 你可以选任何喜欢的整数，比如 0, 42, 1234
dde.config.set_random_seed(SEED)  # 固定 DeepXDE 和 后端(Torch/TF) 的种子
np.random.seed(SEED)              # 固定 Numpy 的种子 (这对你手动采样的点至关重要)
# -------------------- Constants (same as your SciANN script) --------------------
s, minute, hr, day = 1., 60., 60.**2, 24*60.**2
g, kg = 1.e-3, 1.
mm, cm, m, km = 1e-3, 1e-2, 1.0, 1e3
Pa, kPa, MPa, GPa = 1.0, 1.e3, 1.e6, 1.e9
micro, milli, centi, kilo, mega = 1e-6, 1e-3, 1e-2, 1e3, 1e6
poise = 0.1*Pa*s
Darcy = 9.869233e-13*m**2

S_overburden = -0.1*MPa
P_ini = 0.*MPa
mu = 6. * milli * poise
k = 1e-12
E = 120.*MPa
nu = 0.25
b = 1.0
M = 3e11
Lx = 1*m
Ly = 50*m

Lmbd = E*nu/(1+nu)/(1-2*nu)
Gshr = E/2/(1+nu)
Kdr = E/(3*(1-2*nu))
B = b*M/(Kdr+M*(b**2))
nu_u = (3*nu+b*B*(1-2*nu))/(3-b*B*(1-2*nu))

p_star = (-1)*S_overburden
x_star = Ly
t_star = (b**2/Kdr + 1/M)*mu*(x_star**2)/(k)
nu_star = (1 - 2*nu)/(1 + nu)
u_star = p_star*x_star/Kdr

# nondimensional domains originally used in your SciANN script:
td_0 = 0.0
td_f = 15000.0 / t_star  # same as SciANN
yd_min = 0.0
yd_max = (Ly / x_star)    # should be 1.0 in your nondim choice
NUM_SAMPLES = 20000
USE_RAR = False
RAR_ITERS = 5
RAR_NUM_CANDIDATES = 10000
RAR_NUM_ADD = 500
RAR_TRAIN_ITERS = 3000

RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
    "terzaghi64_forward_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
)
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULT_DIR}")

# -------------------- PDE system in DeepXDE (dimensionless variables) --------------------
# Unknowns: p(y,t), v(y,t) both nondimensional variables (pd, vd from SciANN)
# PDEs (from your SciANN):
#   Mp: pd_td - diff(pd_yd, yd) = 0   (fluid diffusion in 1D time)
#   Msy: (1+2*nu_star)*diff(vd_yd, yd) - b*diff(pd, yd) = 0   (solid equilibrium in y)

def pde_system(X, Y):
    """
    X: (N,2) with columns [y, t]
    Y: (N,2) with columns [p, v]
    return residuals [Mp, Msy]
    """
    y = X[:, 0:1]
    t = X[:, 1:2]

    p = Y[:, 0:1]
    v = Y[:, 1:2]

    # first derivatives
    p_t = dde.grad.jacobian(Y, X, i=0, j=1)  # dp/dt
    p_y = dde.grad.jacobian(Y, X, i=0, j=0)  # dp/dy
    p_yy = dde.grad.hessian(Y, X, component=0, i=0, j=0)  # d2p/dy2

    v_y = dde.grad.jacobian(Y, X, i=1, j=0)  # dv/dy
    v_yy = dde.grad.hessian(Y, X, component=1, i=0, j=0)  # d2v/dy2

    Mp = p_t - p_yy
    Msy = (1 + 2*nu_star) * v_yy - b * p_y

    return [Mp, Msy]

# -------------------- Geometry and Data --------------------
geom = dde.geometry.Interval(yd_min, yd_max)
timedomain = dde.geometry.TimeDomain(td_0, td_f)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Number of training points
num_domain = int(NUM_SAMPLES / 2)
N_b = 2500  # 每条边界条件点数
N_i = 2500  # 每个初始条件点数
N_init_total =  N_i # 定义初始点总数 (2500)

# --- 定义安全距离 (Buffer) 以避开角点奇点 ---
eps_t = 1e-5  # 时间上的微小偏移 (避开 t=0)
eps_y = 1e-3  # 空间上的微小偏移 (避开 y=y_max)

# 1. Bottom boundary (y = yd_min)
# 底部没有冲突，保持原样
y_bottom = np.full((N_b, 1), yd_min)
t_bottom = np.random.uniform(td_0, td_f, (N_b, 1))
X_bottom = np.hstack((y_bottom, t_bottom))

# 2. Top boundary (y = yd_max)
# 【修改点】：时间从 td_0 + eps_t 开始采样，避开 t=0
y_top = np.full((N_b, 1), yd_max)
t_top = np.random.uniform(td_0 + eps_t, td_f, (N_b, 1))
X_top = np.hstack((y_top, t_top))

# 3. Initial (t = td_0)
# 【修改点】：空间只取到 yd_max - eps_y，避开 y=y_max
# 注意：这里使用 N_init_total (2500) 以匹配你之前的修改
y_init = np.random.uniform(yd_min, yd_max - eps_y, (N_init_total, 1))
t_init = np.full((N_init_total, 1), td_0)
X_init = np.hstack((y_init, t_init))

# -------------------- 定义边界条件 --------------------
# p(top) = 0
bcs = [
    dde.PointSetBC(X_top, np.zeros((N_b, 1)), component=0),
    # dp/dy(bottom) = 0
    dde.PointSetOperatorBC(
        X_bottom,
        np.zeros((N_b, 1)),
        lambda X, Y, _: dde.grad.jacobian(Y, X, i=0, j=0)
    ),
    # v(bottom) = 0
    dde.PointSetBC(X_bottom, np.zeros((N_b, 1)), component=1),
    # stress(top) operator
    dde.PointSetOperatorBC(
        X_top,
        np.zeros((N_b, 1)),
        lambda X, Y, _: (1 + 2 * nu_star) * dde.grad.jacobian(Y, X, i=1, j=0) - b * Y[:, 0:1] - S_overburden / p_star
    ),
    # initial p (注意使用 N_init_total)
    dde.PointSetBC(X_init, np.ones((N_init_total, 1)), component=0),
    # initial v (注意使用 N_init_total)
    dde.PointSetBC(X_init, np.zeros((N_init_total, 1)), component=1),
]

# -------------------- 构建 TimePDE 数据集 --------------------
data = dde.data.TimePDE(
    geomtime,
    pde_system,
    bcs,
    num_domain=int(NUM_SAMPLES / 2),
    num_test=5000,
)
print(f"u_star / m = {u_star / m}")
print(f"p_star / MPa = {p_star / MPa}")

# -------------------- Neural network --------------------
# architecture: keep similar to your SciANN 4*[20]
layer_size = [2] + 4 * [32] + [2]  # 2 inputs, two outputs (p,v)
act = "tanh"
net = dde.nn.FNN(layer_size, act, "Glorot normal")

model = dde.Model(data, net)
# loss_weights=[10, 100, 1, 10, 100, 10, 1, 100]
# 1. 普通训练 / 可选 RAR 训练
model.compile("adam", lr=1e-3) # 带权重
losshistory_adam, train_state_adam = model.train(iterations=15000)

if USE_RAR:
    for i in range(RAR_ITERS):
        print(f"\n[RAR] Iteration {i + 1}/{RAR_ITERS} ----------------------------")
        X_candidates = geomtime.random_points(RAR_NUM_CANDIDATES)
        residuals = model.predict(X_candidates, operator=pde_system)
        err_total = np.abs(residuals[0]) + np.abs(residuals[1])
        err_total = err_total.flatten()
        top_k_indices = np.argsort(err_total)[-RAR_NUM_ADD:]
        X_to_add = X_candidates[top_k_indices]

        max_err = np.max(err_total)
        mean_err = np.mean(err_total)
        print(f"[RAR] Found max residual: {max_err:.4e}, mean residual: {mean_err:.4e}")

        data.add_anchors(X_to_add)
        print(f"[RAR] Added {len(X_to_add)} anchors to training set.")

        model.train(iterations=RAR_TRAIN_ITERS)

model.compile("L-BFGS")
losshistory_lbfgs, train_state_lbfgs = model.train()


# ---评估网格和解析解计算--------
log10p = lambda x: np.log10(1. + x)
Nx, Ny, Nt = 5, 100, 200
ys = np.linspace(yd_min, yd_max, Ny * 3)  # y_d 网格，300点 [0,1]
ts = (np.logspace(0., log10p(td_f * t_star), Nt * 3) - 1.0) / t_star  # t_d 网格，对数间隔，600点

Ydtest, Tdtest = np.meshgrid(ys, ts, indexing='ij')
input_test = [Ydtest, Tdtest]

Ytest = Ydtest * x_star  # 物理 y (m)
Ttest = Tdtest * t_star  # 物理 t (s)

ys_line_plot = np.linspace(yd_min, yd_max, Ny * 3)  # 线图 y 网格，300点
ts_line_plot = [td_f * 0.01, td_f * 0.025, td_f * 0.05, td_f * 0.1, td_f * 0.25, td_f * 0.5, td_f * 0.75, td_f * 1.0]  # 8个无量纲 t 点
ts_line_plot1 = np.array(ts_line_plot)  # 转数组，便于索引
Ydtest_line, Tdtest_line = np.meshgrid(ys_line_plot, ts_line_plot1, indexing='ij')
input_test_line_plot = np.meshgrid(ys_line_plot, ts_line_plot1, indexing='ij')

# ----轮廓/线图解析孔压 p_true (无量纲 pd)----
p_true_contour = np.zeros_like(input_test[0])  #
for i in range(100):
    Ci = 4 * (-1) ** i / (np.pi * (2 * i + 1))
    Lisq = np.pi * (2 * i + 1) / 2
    Zz = np.cos(Lisq * input_test[0])
    Tt = np.exp(-Lisq ** 2 * input_test[1])
    p_true_contour += Ci * Zz * Tt

p_true_line = np.zeros_like(input_test_line_plot[0])  # (8,300)
for i in range(100):
    Ci = 4 * (-1) ** i / (np.pi * (2 * i + 1))
    Lisq = np.pi * (2 * i + 1) / 2
    Zz = np.cos(Lisq * input_test_line_plot[0])
    Tt = np.exp(-Lisq ** 2 * input_test_line_plot[1])
    p_true_line += Ci * Zz * Tt

# -----轮廓/线图解析位移 v_true
F2 = np.zeros_like(input_test[0])  #
for i in range(100):
    Ci = 8 / ((np.pi * (2 * i + 1)) ** 2)
    Lisq = np.pi * (2 * i + 1) / 2
    Zz = np.cos(Lisq * (1 - input_test[0]))
    Tt = 1 - np.exp(-(Lisq ** 2) * input_test[1])
    F2 += Ci * Zz * Tt

term1 = S_overburden * Ly * (1 - 2 * nu_u) * (input_test[0]) / (2 * Gshr * (1 - nu_u))
term2 = S_overburden * Ly * (nu_u - nu) / (2 * Gshr * (1 - nu_u) * (1 - nu))
v_true_contour = term1 + term2 * F2

F2 = np.zeros_like(input_test_line_plot[0])  # (8,300)
for i in range(100):
    Ci = 8 / ((np.pi * (2 * i + 1)) ** 2)
    Lisq = np.pi * (2 * i + 1) / 2
    Zz = np.cos(Lisq * (1 - input_test_line_plot[0]))
    Tt = 1 - np.exp(-(Lisq ** 2) * input_test_line_plot[1])
    F2 += Ci * Zz * Tt

term1 = S_overburden * Ly * (1 - 2 * nu_u) * (input_test_line_plot[0]) / (2 * Gshr * (1 - nu_u))
term2 = S_overburden * Ly * (nu_u - nu) / (2 * Gshr * (1 - nu_u) * (1 - nu))
v_true_line = term1 + term2 * F2

# ------------预测值计算（修正单位重复）--------
input_test1 = np.hstack((Ydtest.reshape(-1, 1), Tdtest.reshape(-1, 1)))  # (180000, 2)
input_test_line_plot1 = np.hstack((Ydtest_line.reshape(-1, 1), Tdtest_line.reshape(-1, 1)))  # (2400, 2)

pred_all = model.predict(input_test1)
pred_line_all = model.predict(input_test_line_plot1)

Ny_grid = Ydtest_line.shape[0]
Nt_line = Tdtest_line.shape[1]

p_line = pred_line_all[:, 0:1].reshape(Ny_grid, Nt_line)
v_line = pred_line_all[:, 1:2].reshape(Ny_grid, Nt_line)

# pred 字典（修正：[:,0] 取值，无重复缩放；v * u_star 无 /m）
pred = {
    'v': model.predict(input_test1)[:, 1:2] * u_star / m,  # (180000,1) m
    'v_line': v_line * u_star / m,  # 已 m，(300,8)
    'v_true_contour': v_true_contour,  # 已 m (600,300)? 实际 Ydtest (600t,300y)
    'v_true_line': v_true_line,  # 已 m (300,8)
    'p': model.predict(input_test1)[:, 0:1] * p_star / MPa,  # (180000,1) MPa
    'p_line': p_line * p_star / MPa,  # 已 MPa (300,8)
    'p_true_contour': p_true_contour * p_star / MPa ,
    'p_true_line': p_true_line * p_star / MPa ,
}

# 为下方 3 行 × 2 列图准备物理量数据
p_pred_grid = pred['p'][:, 0].reshape(Ydtest.shape)
p_true_grid = pred['p_true_contour']
p_error_grid = np.abs(p_pred_grid - p_true_grid)

v_pred_grid = pred['v'][:, 0].reshape(Ydtest.shape)
v_true_grid = pred['v_true_contour']
v_error_grid = np.abs(v_pred_grid - v_true_grid)

y_plot = input_test[0] * x_star
t_plot = np.log10(np.maximum(input_test[1] * t_star, 1e-12))

# -----------------------------------------------------------------------------
# 解析解与 PINN 预测曲线（或场）保留原始数据。
# -----------------------------------------------------------------------------
publication_rc = {
    'font.family': 'Times New Roman',
    'font.size': 19,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 26,
    'axes.titlesize': 26,
    'path.simplify': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'path',
}

with plt.rc_context(publication_rc):
    y_line_physical = ys_line_plot * x_star
    line_times_physical = np.asarray(ts_line_plot) * t_star
    spatial_ticks = np.arange(0, Ly / m + 1, 10)
    publication_colors = ['#ff0000', '#ffa500', '#0000ff', '#008000',
                          '#800080', '#00ffff', '#ffd700', '#000000']

    line_fields = [
        pred['v_true_line'],
        pred['p_true_line'],
        pred['v_line'],
        pred['p_line'],
        gaussian_filter1d(
            np.abs(pred['v_line'] - pred['v_true_line']),
            sigma=5, axis=0, mode='nearest'),
        gaussian_filter1d(
            np.abs(pred['p_line'] - pred['p_true_line']),
            sigma=5, axis=0, mode='nearest'),
    ]
    line_labels = [
        r'$v\;\mathrm{(m)}$', r'$p\;\mathrm{(MPa)}$',
        r'$\widehat{v}\;\mathrm{(m)}$', r'$\widehat{p}\;\mathrm{(MPa)}$',
        r'$|v-\widehat{v}|\;\mathrm{(m)}$',
        r'$|p-\widehat{p}|\;\mathrm{(MPa)}$',
    ]

    fig_line, axes_line = plt.subplots(3, 2, figsize=(17, 17.5))
    fig_line.subplots_adjust(
        left=.14, right=.81, top=.985, bottom=.065, wspace=.44, hspace=.24)
    for panel_index, (axis, field, label) in enumerate(
            zip(axes_line.flat, line_fields, line_labels)):
        for time_index, color in enumerate(publication_colors):
            axis.plot(
                y_line_physical, field[:, time_index], color=color,
                linewidth=2, linestyle=['-', '--', '-.'][panel_index // 2])
        axis.set(xlim=(-2.5, 52.5), xticks=spatial_ticks, ylabel=label)
        axis.tick_params(axis='both', pad=3.5)
        axis.grid(True, alpha=.65, linewidth=.8)
        if panel_index >= 4:
            axis.set_xlabel(r'$y\;\mathrm{(m)}$')
            axis.margins(y=.05)
            axis.ticklabel_format(axis='y', style='plain', useOffset=False)
        elif panel_index % 2 == 0:
            axis.set_ylim(-.032, .001)
        else:
            axis.set_ylim(-.003, .105)
    fig_line.legend(
        [Line2D([], [], color=color, linewidth=2, linestyle='-.')
         for color in publication_colors],
        [rf'$t={time_value:.0f}\;\mathrm{{(s)}}$'
         for time_value in line_times_physical],
        loc='center left', bbox_to_anchor=(.835, .56),
        frameon=False, fontsize=18)
    for extension in ('png', 'pdf', 'svg'):
        fig_line.savefig(
            os.path.join(RESULT_DIR, 'terzaghi_line_error.' + extension),
            dpi=300, facecolor='white')
    plt.close(fig_line)

    grid_fields = [
        v_true_grid,
        p_true_grid,
        v_pred_grid,
        p_pred_grid,
        np.abs(v_pred_grid - v_true_grid),
        np.abs(p_pred_grid - p_true_grid),
    ]

    
    positive_time_mask = Ttest[0, :] > 0
    source_log_time = np.log10(Ttest[0, positive_time_mask])
    uniform_log_time = np.linspace(
        max(-1.5, source_log_time.min()), source_log_time.max(), Ttest.shape[1])
    log_t, y = np.meshgrid(uniform_log_time, Ytest[:, 0])
    for error_index in (4, 5):
        interpolator = RegularGridInterpolator(
            (Ytest[:, 0], source_log_time),
            grid_fields[error_index][:, positive_time_mask])
        regular_error = interpolator(
            np.stack([y, log_t], axis=-1))
        grid_fields[error_index] = gaussian_filter(
            regular_error, sigma=(4, 4), mode='nearest')

    displacement_norm = Normalize(
        min(grid_fields[0].min(), grid_fields[2].min()),
        max(grid_fields[0].max(), grid_fields[2].max()))
    pressure_norm = Normalize(
        min(grid_fields[1].min(), grid_fields[3].min()),
        max(grid_fields[1].max(), grid_fields[3].max()))
    contour_labels = [
        r'$v\;\mathrm{(m)}$', r'$p\;\mathrm{(MPa)}$',
        r'$\widehat{v}\;\mathrm{(m)}$', r'$\widehat{p}\;\mathrm{(MPa)}$',
        r'$|v-\widehat{v}|\;\mathrm{(m)}$',
        r'$|p-\widehat{p}|\;\mathrm{(MPa)}$',
    ]
    safe_log_time = np.log10(np.maximum(Ttest, 1e-12))

    fig_contour, axes_contour = plt.subplots(
        3, 2, figsize=(17, 17.5))
    fig_contour.subplots_adjust(
        left=.14, right=.90, top=.95, bottom=.065, wspace=.38, hspace=.24)
    for panel_index, (axis, field, label) in enumerate(
            zip(axes_contour.flat, grid_fields,
                contour_labels)):
        if panel_index < 4:
            norm = displacement_norm if panel_index % 2 == 0 else pressure_norm
            plot_time, plot_y = safe_log_time, Ytest
        else:
            norm = Normalize(0, .00020 if panel_index == 4 else .08)
            plot_time, plot_y = log_t, y
        mesh = axis.pcolormesh(
            plot_time, plot_y, field, shading='nearest',
            cmap='jet', norm=norm, rasterized=True)
        axis.set(
            ylim=(0, 50), yticks=spatial_ticks,
            xlim=(-1.5, np.log10(Ttest.max())), title=label)
        if panel_index % 2 == 0:
            axis.set_ylabel(r'$y\;\mathrm{(m)}$')
        if panel_index >= 4:
            axis.set_xlabel(r'$\log_{10}(t)\;\mathrm{(s)}$')
        colorbar = fig_contour.colorbar(
            mesh, ax=axis, fraction=.046, pad=.025)
        if panel_index == 4:
            colorbar.set_ticks([0, .00005, .00010, .00015, .00020])
        elif panel_index == 5:
            colorbar.set_ticks([0, .02, .04, .06, .08])
        colorbar.ax.tick_params(labelsize=16)
    for extension in ('png', 'pdf', 'svg'):
        fig_contour.savefig(
            os.path.join(RESULT_DIR, 'terzaghi_contour_error.' + extension),
            dpi=300, facecolor='white')
    plt.close(fig_contour)

np.savez_compressed(
    os.path.join(RESULT_DIR, "terzaghi64_forward_data.npz"),
    ys=ys,
    ts=ts,
    ys_line_plot=ys_line_plot,
    ts_line_plot=np.asarray(ts_line_plot),
    Ydtest=Ydtest,
    Tdtest=Tdtest,
    Ytest=Ytest,
    Ttest=Ttest,
    y_plot=y_plot,
    t_plot=t_plot,
    input_test_points=input_test1,
    input_line_points=input_test_line_plot1,
    pred_all=pred_all,
    pred_line_all=pred_line_all,
    p_pred_grid=p_pred_grid,
    p_true_grid=p_true_grid,
    p_error_grid=p_error_grid,
    v_pred_grid=v_pred_grid,
    v_true_grid=v_true_grid,
    v_error_grid=v_error_grid,
    p_line=pred["p_line"],
    p_true_line=pred["p_true_line"],
    v_line=pred["v_line"],
    v_true_line=pred["v_true_line"],
    adam_steps=np.asarray(losshistory_adam.steps),
    adam_loss_train=np.asarray(losshistory_adam.loss_train),
    adam_loss_test=np.asarray(losshistory_adam.loss_test),
    lbfgs_steps=np.asarray(losshistory_lbfgs.steps),
    lbfgs_loss_train=np.asarray(losshistory_lbfgs.loss_train),
    lbfgs_loss_test=np.asarray(losshistory_lbfgs.loss_test),
)
metadata = {
    "script": "terzaghi-64.py",
    "seed": SEED,
    "NUM_SAMPLES": NUM_SAMPLES,
    "network_layers": layer_size,
    "activation": act,
    "td_0": td_0,
    "td_f": td_f,
    "yd_min": yd_min,
    "yd_max": yd_max,
    "x_star": x_star,
    "t_star": t_star,
    "p_star": p_star,
    "u_star": u_star,
    "S_overburden": S_overburden,
    "mu": mu,
    "k": k,
    "E": E,
    "nu": nu,
    "b": b,
    "M": M,
    "Ly": Ly,
}
with open(os.path.join(RESULT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved figures and data to: {RESULT_DIR}")

