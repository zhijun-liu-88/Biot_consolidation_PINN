import numpy as np
import deepxde as dde
import json

dde.backend.set_default_backend("pytorch")
dde.config.set_default_float("float64")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from datetime import datetime
from pathlib import Path

torch.set_default_dtype(torch.float64)

SEED = 1234  # 你可以选任何喜欢的整数，比如 0, 42, 1234
dde.config.set_random_seed(SEED)  # 固定 DeepXDE 和 后端(Torch/TF) 的种子
np.random.seed(SEED)              # 固定 Numpy 的种子 (这对你手动采样的点至关重要)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results_mandel_noev_64_{RUN_TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VAR_HISTORY_FILE = str(OUTPUT_DIR / f"variables_kbar_noev_64_{RUN_TIMESTAMP}.dat")
INVERSION_PLOT_FILE = str(OUTPUT_DIR / f"K_Inversion_Result_noev_64_{RUN_TIMESTAMP}.pdf")
CONTOUR_PLOT_FILE_TEMPLATE = str(OUTPUT_DIR / "Mandel_ContourPlot_itr{itr:02d}.jpg")
LINE_PLOT_FILE_TEMPLATE = str(OUTPUT_DIR / "Mandel_LinePlot_itr{itr:02d}.jpg")



# ==================== 参数定义和归一化 ====================
# 单位换算
s, minute, hr, day = 1., 60., 60.**2, 24*60.**2
g, kg = 1.e-3, 1.
mm, cm, m, km = 1e-3, 1e-2, 1.0, 1e3
Pa, kPa, MPa, GPa = 1.0, 1.e3, 1.e6, 1.e9
micro, milli, centi, kilo, mega = 1e-6, 1e-3, 1e-2, 1e3, 1e6
poise = 0.1*Pa*s
Darcy = 9.869233e-13*m**2

# 物理参数
S_overburden = -2*MPa
rho_b = 2400.0 * kg/m**3
rho_f = 1000. * kg/m**3

nu = 0.25
b = 1.0
M = 120e6
Lx = 50*m
Ly = 10*m

mu = 6. * milli * poise
mu_star = 6. * milli * poise
mu_bar = mu/mu_star

# 真实值 (用于生成观测数据)
k = 1e-12
k_star = 1e-12

E = 120e6
E_star = 120e6
E_bar = E/E_star

# 定义可学习参数 (Variables)
def inverse_softplus_scalar(value):
    return float(np.log(np.exp(value) - 1.0))


def positive_parameter(raw_var, min_value=1e-6):
    return torch.nn.functional.softplus(raw_var) + min_value


k_bar_raw = dde.Variable(
    torch.tensor(
        [inverse_softplus_scalar(2.0)],
        dtype=torch.float64,
        requires_grad=True,
    )
)


Kdr = E_bar*E_star/(3*(1-2*nu))
Kdr_star = E_star/(3*(1-2*nu))
Kdr_bar = Kdr/Kdr_star

# 计算派生参数

B = (b*M)/(Kdr_star+M*(b**2))
nu_u = (3*nu+b*B*(1-2*nu))/(3-b*B*(1-2*nu))
F = S_overburden*Ly*1*m
Gshr = E_star/2/(1+nu)
Gama = B*(1+nu_u)/(3*(1-nu_u))
p0 = (-F)*B*(1+nu_u)/(3*Ly)

# 特征尺度归一化
p_star = p0
x_star = Ly
t_star = (b**2/Kdr_star + 1/M)*mu_star*(x_star**2)/(k_star)
nu_star = (1 - 2*nu)/(1 + nu)
u_star = p_star*x_star/Kdr_star
D_star = b*M/((b**2)*M + Kdr_star)
DD_star = mu_star*(x_star**2)/Kdr_star/k_star/t_star

print(f"[float64] D_star = {D_star}, p0 = {p0/MPa} MPa")

td_0 = 0.0
td_f = 1000/t_star
xd_min, xd_max = 0.0, Lx/x_star
yd_min, yd_max = 0.0, Ly/x_star

geom = dde.geometry.Rectangle([xd_min, yd_min], [xd_max, yd_max])
timedomain = dde.geometry.TimeDomain(td_0, td_f)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pdes(x, y):
    """无独立 Ev 输出的 Mandel 反问题 PDE。"""
    p, u, v = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    xd, yd, td = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    # 计算梯度
    p_t = dde.grad.jacobian(y, x, i=0, j=2)  # dp/dt
    p_x = dde.grad.jacobian(y, x, i=0, j=0)  # dp/dx
    p_y = dde.grad.jacobian(y, x, i=0, j=1)  # dp/dy
    p_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)  # d²p/dx²
    p_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)  # d²p/dy²

    u_x = dde.grad.jacobian(y, x, i=1, j=0)
    u_y = dde.grad.jacobian(y, x, i=1, j=1)
    v_y = dde.grad.jacobian(y, x, i=2, j=1)
    v_x = dde.grad.jacobian(y, x, i=2, j=0)

    # 二阶导数
    u_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    u_xy = dde.grad.hessian(y, x, component=1, i=0, j=1)
    u_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)
    v_xy = dde.grad.hessian(y, x, component=2, i=1, j=0)
    v_yy = dde.grad.hessian(y, x, component=2, i=1, j=1)

    Ev = u_x + v_y
    Ev_t = dde.grad.jacobian(Ev, x, i=0, j=2)
    Ev_x = dde.grad.jacobian(Ev, x, i=0, j=0)
    Ev_y = dde.grad.jacobian(Ev, x, i=0, j=1)

    # 压力扩散方程: p_t - (p_xx + p_yy) + D_star * Svd_td = 0
    k_bar_value = positive_parameter(k_bar_raw)
    pde1 = p_t - k_bar_value * (p_xx + p_yy) + D_star * (Kdr_bar * Ev_t - b * p_t)

    pde2 = Kdr_bar * Ev_x + Kdr_bar * 1.5 * nu_star * (u_xx + u_yy) + Kdr_bar * (nu_star/2) * (u_xx + v_xy) - b * p_x
    pde3 = Kdr_bar * Ev_y + Kdr_bar * 1.5 * nu_star * (v_xx + v_yy) + Kdr_bar * (nu_star/2) * (u_xy + v_yy) - b * p_y

    return [pde1, pde2, pde3]


def get_mandel_roots(nu, nu_u, num_roots=100):
    def func(z):
        return np.tan(z) - ((1 - nu) / (nu_u - nu)) * z

    roots = []
    for i in range(num_roots):
        # 寻找区间的根
        guess = (i + 0.5) * np.pi + 0.1
        root = fsolve(func, guess)[0]
        if root > 0: roots.append(root)
    return np.unique(np.array(roots))[:num_roots]


# 预计算根
ROOTS = get_mandel_roots(nu, nu_u)


def get_true_values(X):
    """
    计算给定坐标点 X [x, y, t] 的解析解 p, u, v
    注意：这里使用 k_bar_true 和 E_bar_true 对应的物理参数
    """
    x_in = X[:, 0:1]  # 无量纲 x
    y_in = X[:, 1:2]  # 无量纲 y
    t_in = X[:, 2:3]  # 无量纲 t

    # 物理量转换 (解析解公式通常基于物理坐标)
    # 这里的公式需要对应你之前提供的 compute_analytical_solution 逻辑
    # 假设 Mandel 解是标准形式

    p_val = np.zeros_like(x_in)
    u_val = np.zeros_like(x_in)
    v_val = np.zeros_like(x_in)

    # --- 计算 p ---
    term_p = 2 * (-F) * B * (1 + nu_u) / (3 * Ly)
    sum_p = np.zeros_like(x_in)
    for alpha in ROOTS:
        T1 = np.sin(alpha) / (alpha - np.sin(alpha) * np.cos(alpha))
        T2 = np.cos(alpha * y_in) - np.cos(alpha)  # y 是短边
        T3 = np.exp(-(alpha ** 2) * t_in)
        sum_p += T1 * T2 * T3
    p_val = term_p * sum_p / p_star  # 归一化

    # --- 计算 u ---
    term_u1 = -F * (1 - nu) / (2 * Gshr * Ly)
    term_u2 = F * (1 - nu_u) / (Gshr * Ly)
    sum_u = np.zeros_like(x_in)
    for alpha in ROOTS:
        T1 = (np.sin(alpha) * np.cos(alpha)) / (alpha - np.sin(alpha) * np.cos(alpha))
        T2 = np.exp(-(alpha ** 2) * t_in)
        sum_u += T1 * T2
    # u 的物理位置依赖于 (Lx - x_phys)，即 (Lx - x*x_star)
    u_phys = (term_u1 + term_u2 * sum_u) * (Lx - x_in * x_star)
    u_val = u_phys / u_star

    # --- 计算 v ---
    term_v1 = F * nu / (2 * Gshr * Ly)
    term_v2 = F * nu_u / (Gshr * Ly)
    term_v3 = F / Gshr
    sum_v1 = np.zeros_like(x_in)
    sum_v2 = np.zeros_like(x_in)
    for alpha in ROOTS:
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)
        denom = alpha - sin_a * cos_a
        exp_t = np.exp(-(alpha ** 2) * t_in)

        sum_v1 += (sin_a * cos_a / denom) * exp_t
        sum_v2 += (cos_a / denom) * np.sin(alpha * y_in) * exp_t

    v_phys = (term_v1 - term_v2 * sum_v1) * (y_in * x_star) + term_v3 * sum_v2
    v_val = v_phys / u_star

    return p_val, u_val, v_val


# ==============================================================================
# 2. 生成特定的观测点数据 (Sensors)
# ==============================================================================

def build_front_loaded_time_grid(t0, tf, n_total, early_fraction=0.6, early_window=0.15):
    """前期密集、后期稀疏的观测时间网格。"""
    n_early = int(n_total * early_fraction)
    n_late = n_total - n_early
    t_cut = t0 + early_window * (tf - t0)
    t_early = np.linspace(t0 + 1e-3, t_cut, n_early, endpoint=False)
    t_late = np.linspace(t_cut, tf, n_late)
    return np.unique(np.concatenate([t_early, t_late]))


# 注意：反演时，时间维度的覆盖很重要
t_sensors = build_front_loaded_time_grid(td_0, td_f, 320, early_fraction=0.65, early_window=0.12)


def create_sensor_data(spatial_points):
    """将空间点与时间点通过 meshgrid 组合，生成 (N*T, 3) 的坐标数组"""
    X_list = []
    for pt in spatial_points:
        # pt = [x, y]
        for t in t_sensors:
            X_list.append([pt[0], pt[1], t])
    return np.array(X_list)


# ==============================================================================
# 1. 坐标定义 (这就是你刚才选的黄金组合)
# ==============================================================================
# P点集合 (只测压力)
pts_p_list = [
    [1.0, 0.0],
    [1.0, 0.2],
    [1.0, 0.5],
    [2.5, 0.0],
    [2.5, 0.15],
    [2.5, 0.4],
    [4.0, 0.0],
    [4.0, 0.2],
    [4.0, 0.5],
]

# 转换为传感器时空坐标 (假设已用 geomspace 生成了 t_sensors)
X_obs_p_all = []
p_true_all = []

for pt in pts_p_list:
    # 生成当前位置的时空坐标 (N_t, 3)
    X_tmp = create_sensor_data([pt])
    # 计算对应的真值
    p_tmp, _, _ = get_true_values(X_tmp)

    X_obs_p_all.append(X_tmp)
    p_true_all.append(p_tmp)

# --- 核心步骤：堆叠合并 (Stacking) ---
# 将所有 P 的观测点拼成一个大数组
X_obs_p_merged = np.vstack(X_obs_p_all)
p_true_merged = np.vstack(p_true_all)

print(f"[float64] Total Pressure Observation Points: {len(X_obs_p_merged)}")

# 3. 力学位置：顶部边缘
# 这里的位移最大，数据最清晰，用于“锁死”力学场
pts_top_surface = [[2.5, 1.0]] # 或者 [5.0, 1.0]


# 组 3: 顶部只看位移 (因为顶部 p=0，没意义)
X_obs_top = create_sensor_data(pts_top_surface)
_, u_true_top, v_true_top = get_true_values(X_obs_top)


def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], xd_min)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], xd_max)

def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1], yd_min)

def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], yd_max)

def boundary_initial(x, on_initial):
    return on_initial and np.isclose(x[2], td_0)

# 边界条件值
S_overburden_norm = S_overburden / p_star

# 初始条件 - 压力
def ic_func_p(x):
    return np.ones((len(x), 1))  # p = 1 (归一化)

def ic_sxx(x, y, X):
    p, u, v = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Exx = dde.grad.jacobian(y, x, i=1, j=0)
    Eyy = dde.grad.jacobian(y, x, i=2, j=1)
    Ev = Exx + Eyy
    Sxx = (1-nu_star)*Ev + 3*nu_star*Exx - b*(p)
    return Sxx - S_overburden_norm

def ic_syy(x, y, aux=None):
    p, u, v = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Exx = dde.grad.jacobian(y, x, i=1, j=0)
    Eyy = dde.grad.jacobian(y, x, i=2, j=1)
    Ev = Exx + Eyy
    Syy = (1-nu_star)*Ev + 3*nu_star*Eyy - b*(p)
    return Syy

# 边界条件 - 左侧应力
def bc_left_sxx(x, y, aux=None):
    p, u, v = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Exx = dde.grad.jacobian(y, x, i=1, j=0)
    Eyy = dde.grad.jacobian(y, x, i=2, j=1)
    Ev = Exx + Eyy
    Sxx = (1-nu_star)*Ev + 3*nu_star*Exx - b*(p)
    return Sxx - S_overburden_norm

# 边界条件 - 右侧位移
def bc_right_u(x):
    return np.zeros((len(x), 1))

# 边界条件 - 底部位移
def bc_bottom_v(x):
    return np.zeros((len(x), 1))

# 边界条件 - 顶部应力
def bc_top_syy(x, y, aux=None):
    p, u, v = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Exx = dde.grad.jacobian(y, x, i=1, j=0)
    Eyy = dde.grad.jacobian(y, x, i=2, j=1)
    Ev = Exx + Eyy
    Syy = (1-nu_star)*Ev + 3*nu_star*Eyy - b*(p)
    return Syy

# 边界条件 - 剪应力
def bc_shear_sxy(x, y, aux=None):
    # du/dy
    u_y = dde.grad.jacobian(y, x, i=1, j=1)
    # dv/dx
    v_x = dde.grad.jacobian(y, x, i=2, j=0)
    Exy = 0.5 * (u_y + v_x)
    Sxy = 3*nu_star*Exy
    return Sxy

# 定义采样数量
N_b = 1200   # 每条边界的总点数，适度缩减以降低训练成本
N_i = 4000   # 初始条件点数

# --- A. 初始点 (Initial, t = td_0) ---
# 定义空间缓冲 (仅针对 y 方向，用于初始孔压)
epsilon_y = 0.005 * (yd_max - yd_min)

# A1. 初始点 - 仅针对孔压 P (带缓冲: y < y_max - eps)
y_init_p = np.random.uniform(yd_min, yd_max - epsilon_y, (N_i, 1))
x_init_p = np.random.uniform(xd_min, xd_max, (N_i, 1))
t_init_p = np.full((N_i, 1), td_0)
X_init_p = np.hstack((x_init_p, y_init_p, t_init_p))

# A2. 初始点 - 针对力学量 (全域: y <= y_max)
y_init_mech = np.random.uniform(yd_min, yd_max, (N_i, 1))
x_init_mech = np.random.uniform(xd_min, xd_max, (N_i, 1))
t_init_mech = np.full((N_i, 1), td_0)
X_init_mech = np.hstack((x_init_mech, y_init_mech, t_init_mech))


# --- B. 边界点 (Boundaries) ---

# 1. Left Boundary (x = xd_min)
y_left = np.random.uniform(yd_min, yd_max, (N_b, 1))
t_left = np.random.uniform(td_0, td_f, (N_b, 1))
x_left = np.full((N_b, 1), xd_min)
X_left = np.hstack((x_left, y_left, t_left))

# 2. Right Boundary (x = xd_max)
y_right = np.random.uniform(yd_min, yd_max, (N_b, 1))
t_right = np.random.uniform(td_0, td_f, (N_b, 1))
x_right = np.full((N_b, 1), xd_max)
X_right = np.hstack((x_right, y_right, t_right))

# 3. Bottom Boundary (y = yd_min)
x_bot = np.random.uniform(xd_min, xd_max, (N_b, 1))
t_bot = np.random.uniform(td_0, td_f, (N_b, 1))
y_bot = np.full((N_b, 1), yd_min)
X_bottom = np.hstack((x_bot, y_bot, t_bot))

# 4. Top Boundary (y = yd_max) —— [修改核心: 分段均匀采样]
# 策略: 50% 的点集中在前 10% 的时间段，另外 50% 覆盖剩余 90% 的时间
N_early = int(0.5 * N_b)
N_late = N_b - N_early
t_cutoff = td_0 + 0.1 * (td_f - td_0) # 时间分割点

# (1) 早期密集采样 (捕捉奇异性)
t_top_early = np.random.uniform(td_0, t_cutoff, (N_early, 1))
# (2) 晚期稀疏采样 (捕捉稳态)
t_top_late = np.random.uniform(t_cutoff, td_f, (N_late, 1))
# 合并时间
t_top = np.vstack((t_top_early, t_top_late))


# 空间保持均匀
x_top = np.random.uniform(xd_min, xd_max, (N_b, 1))
y_top = np.full((N_b, 1), yd_max)

# 组合坐标
X_top = np.hstack((x_top, y_top, t_top))


def build_front_loaded_domain_anchors(n_points, x_min, x_max, y_min, y_max, t0, tf):
    """构造前期密集、后期稀疏的域内 anchors。"""
    n_early = int(0.6 * n_points)
    n_mid = int(0.25 * n_points)
    n_late = n_points - n_early - n_mid

    t_early_max = t0 + 0.1 * (tf - t0)
    t_mid_max = t0 + 0.4 * (tf - t0)

    x_early = np.random.uniform(x_min, x_max, (n_early, 1))
    y_early = np.random.uniform(y_min, y_max, (n_early, 1))
    t_early = np.random.uniform(t0, t_early_max, (n_early, 1))

    x_mid = np.random.uniform(x_min, x_max, (n_mid, 1))
    y_mid = np.random.uniform(y_min, y_max, (n_mid, 1))
    t_mid = np.random.uniform(t_early_max, t_mid_max, (n_mid, 1))

    x_late = np.random.uniform(x_min, x_max, (n_late, 1))
    y_late = np.random.uniform(y_min, y_max, (n_late, 1))
    t_late = np.random.uniform(t_mid_max, tf, (n_late, 1))

    X_early = np.hstack((x_early, y_early, t_early))
    X_mid = np.hstack((x_mid, y_mid, t_mid))
    X_late = np.hstack((x_late, y_late, t_late))
    return np.vstack((X_early, X_mid, X_late))
# ==============================================================================
# 2. 定义边界条件 (使用 PointSetBC)
# ==============================================================================

bcs = []

# --- 初始条件 ---
# 1. 孔压 P 使用带缓冲的点集 X_init_p
bcs.append(dde.PointSetBC(X_init_p, np.ones((N_i, 1)), component=0))

# 2. 应力 Sxx 使用全域点集 X_init_mech
bcs.append(dde.PointSetOperatorBC(X_init_mech, np.zeros((N_i, 1)), ic_sxx))

# 3. 应力 Syy 使用全域点集 X_init_mech
bcs.append(dde.PointSetOperatorBC(X_init_mech, np.zeros((N_i, 1)), ic_syy))


# --- 顶部边界 (Top) ---
# 这里 t 从 0 开始，p=0。
# 由于 X_init_p 最高只到 0.995, 而这里是 1.0, 两者不打架。
bcs.append(dde.PointSetBC(X_top, np.zeros((N_b, 1)), component=0)) # p=0
bcs.append(dde.PointSetOperatorBC(X_top, np.zeros((N_b, 1)), bc_top_syy))
bcs.append(dde.PointSetOperatorBC(X_top, np.zeros((N_b, 1)), bc_shear_sxy))

# --- 底部边界 (Bottom) ---
# Flux (dp/dy = 0). 法向是 -y，所以是 -dp/dy
bcs.append(dde.PointSetOperatorBC(X_bottom, np.zeros((N_b, 1)),
           lambda x, y, _: -dde.grad.jacobian(y, x, i=0, j=1)))
bcs.append(dde.PointSetBC(X_bottom, np.zeros((N_b, 1)), component=2)) # v=0
bcs.append(dde.PointSetOperatorBC(X_bottom, np.zeros((N_b, 1)), bc_shear_sxy)) # Sxy

# --- 左侧边界 (Left) ---
# Flux (dp/dx = 0). 法向是 -x，所以是 -dp/dx
bcs.append(dde.PointSetOperatorBC(X_left, np.zeros((N_b, 1)),
           lambda x, y, _: -dde.grad.jacobian(y, x, i=0, j=0)))
bcs.append(dde.PointSetOperatorBC(X_left, np.zeros((N_b, 1)), bc_left_sxx)) # Sxx
bcs.append(dde.PointSetOperatorBC(X_left, np.zeros((N_b, 1)), bc_shear_sxy)) # Sxy

# --- 右侧边界 (Right) ---
# Flux (dp/dx = 0). 法向是 +x
bcs.append(dde.PointSetOperatorBC(X_right, np.zeros((N_b, 1)),
           lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=0)))
bcs.append(dde.PointSetBC(X_right, np.zeros((N_b, 1)), component=1)) # u=0
bcs.append(dde.PointSetOperatorBC(X_right, np.zeros((N_b, 1)), bc_shear_sxy)) # Sxy

# 注意：因为点变多了，总 Loss 会变大，权重需要适当权衡
bcs.append(dde.PointSetBC(X_obs_p_merged, p_true_merged, component=0))

# 顶部变形权重高，确保力学场不跑偏
bcs.append(dde.PointSetBC(X_obs_top, u_true_top, component=1))
bcs.append(dde.PointSetBC(X_obs_top, v_true_top, component=2))

# ==============================================================================
# 3. 构建数据与模型
# ==============================================================================

NUM_DOMAIN_POINTS = 8000
X_domain_anchors = build_front_loaded_domain_anchors(
    NUM_DOMAIN_POINTS, xd_min, xd_max, yd_min, yd_max, td_0, td_f
)


data = dde.data.TimePDE(
    geomtime,
    pdes,
    bcs,
    num_domain=0,  # 内部点由手工 anchors 提供
    num_boundary=0,    # 边界点：使用 PointSetBC 手动提供的点
    num_initial=0,     # 初始点：使用 PointSetBC 手动提供的点
    anchors=X_domain_anchors
)

layer_size = [3] + 4*[128] + [3]
activation = "swish"
initializer = "He normal"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
# 注册外部变量
external_trainable_variables = [k_bar_raw]
fname_var = VAR_HISTORY_FILE
variable = dde.callbacks.VariableValue(
    external_trainable_variables,
    period=100,
    filename=fname_var
)

# ==============================================================================
# 解决方案：设置 batch_size
# 建议设置为 2048 或 4096 (根据显存情况调整)
# ==============================================================================
BATCH_SIZE = 1024


def get_phase_loss_weights(phase_name):
    if phase_name == "pretrain":
        w_pde = [2.0, 1.0, 1.0]
        w_bc = [25.0] * 15
        w_obs = [60.0, 25.0, 25.0]
    elif phase_name == "identify":
        w_pde = [8.0, 2.0, 2.0]
        w_bc = [20.0] * 15
        w_obs = [120.0, 35.0, 35.0]
    else:
        w_pde = [5.0, 5.0, 5.0]
        w_bc = [20.0] * 15
        w_obs = [100.0, 60.0, 60.0]
    return w_pde + w_bc + w_obs


def set_network_trainable(trainable):
    for param in model.net.parameters():
        param.requires_grad = trainable


class ParameterAccuracyCallback(dde.callbacks.Callback):
    def __init__(self, period=500):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration
        if iteration % self.period != 0:
            return

        loss_list = self.model.train_state.loss_train
        total_loss = float(np.sum(loss_list)) if loss_list is not None else float("nan")
        k_pred = positive_parameter(k_bar_raw).item()
        rel_error = abs(k_pred - 1.0) / 1.0
        accuracy = max(0.0, 100.0 * (1.0 - rel_error))

        print(f"[float64] Iteration {iteration}, Total Loss: {total_loss:.6e}")
        print(
            f"  [float64] k_bar = {k_pred:.5f}, Relative Error = {rel_error * 100:.2f}%, "
            f"Accuracy = {accuracy:.2f}%"
        )


monitor = ParameterAccuracyCallback(period=500)

# --- Phase 1: Pre-training ---
print("[float64] Phase 1: Pre-training (Frozen k_bar)...")
set_network_trainable(True)
model.compile(
    "adam",
    lr=1e-3,
    loss_weights=get_phase_loss_weights("pretrain"),
    external_trainable_variables=[],
)

# 【修改点】加入 batch_size
losshistory_pretrain, train_state_pretrain = model.train(epochs=4000, batch_size=BATCH_SIZE)


# --- Phase 2: Parameter Identification ---
print("[float64] Phase 2: Parameter Identification (network weakly updated, k_bar emphasized)...")
set_network_trainable(True)
model.compile(
    "adam",
    lr=2e-4,
    loss_weights=get_phase_loss_weights("identify"),
    external_trainable_variables=[k_bar_raw],
)

losshistory_identify, train_state_identify = model.train(epochs=3000, callbacks=[variable, monitor], batch_size=BATCH_SIZE)


# --- Phase 3: Joint Fine-tuning ---
print("[float64] Phase 3: Joint Fine-tuning (small LR)...")
set_network_trainable(True)
lr_schedule = ("step", 3000, 0.9)
model.compile(
    "adam",
    lr=8e-5,
    loss_weights=get_phase_loss_weights("finetune"),
    external_trainable_variables=[k_bar_raw],
    decay=lr_schedule,
)

# 【修改点】加入 batch_size
losshistory_finetune, train_state_finetune = model.train(epochs=13000, callbacks=[variable, monitor], batch_size=BATCH_SIZE)

# --- Phase 4: Field-Only Adam Fine-tuning ---
print("[float64] Phase 4: Field-Only Adam Fine-tuning (freeze k_bar, extra 5000 epochs)...")
set_network_trainable(True)
field_lr_schedule = ("step", 3000, 0.9)
model.compile(
    "adam",
    lr=8e-5,
    loss_weights=get_phase_loss_weights("finetune"),
    external_trainable_variables=[],
    decay=field_lr_schedule,
)
losshistory_field, train_state_field = model.train(epochs=5000, callbacks=[variable, monitor], batch_size=BATCH_SIZE)

# --- Phase 5: L-BFGS Fine-tuning ---
print("[float64] Phase 5: L-BFGS Fine-tuning (freeze k_bar)...")
set_network_trainable(True)
model.compile(
    "L-BFGS",
    loss_weights=get_phase_loss_weights("finetune"),
    external_trainable_variables=[],
)
losshistory_lbfgs, train_state_lbfgs = model.train(callbacks=[variable, monitor])

print("\n-------------------------------------------")
k_bar_final = positive_parameter(k_bar_raw).item()
final_rel_error = abs(k_bar_final - 1.0) / 1.0
final_accuracy = max(0.0, 100.0 * (1.0 - final_rel_error))
print(f"[float64] Final Estimated k_bar: {k_bar_final:.5f} (Target: 1.0)")
print(f"[float64] Final Relative Error: {final_rel_error * 100:.2f}%")
print(f"[float64] Final Accuracy: {final_accuracy:.2f}%")

print("-------------------------------------------\n")


def load_deepxde_variable_file(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        clean_data = []
        for line in lines:
            # 去掉DeepXDE输出格式中的括号和逗号
            clean_line = line.replace('[', ' ').replace(']', ' ').replace(',', ' ')
            items = [float(x) for x in clean_line.split()]
            if items:
                clean_data.append(items)
        return np.array(clean_data)
    except Exception as e:
        print(f"[float64] 读取文件失败: {e}")
        return np.array([])


# ==============================================================================
# 2. 主绘图逻辑
# ==============================================================================
fname_var = VAR_HISTORY_FILE
data_history = load_deepxde_variable_file(fname_var)

if len(data_history) > 0:
    # DeepXDE 格式: [step, var1, var2...]
    # 这里只有 var1 (k_bar_raw)，需要映射回物理 k_bar
    steps = data_history[:, 0]
    k_bar_raw_history = data_history[:, 1]
    k_bar_history = np.log1p(np.exp(k_bar_raw_history)) + 1e-6

    # --- 转换为物理单位 ---
    k_pred_physical = k_bar_history * k_star

    # --- 绘图：对齐 Barry 脚本格式 ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    ax.plot(steps, k_pred_physical, color='red', label='Predicted (float64)')
    ax.hlines(y=k, xmin=steps[0], xmax=steps[-1],
              color='black', linestyle='--', label='True')
    ax.set_title(r"$\kappa$ inversion history (float64)")
    ax.set_xlabel('epochs')
    ax.set_ylabel(r'$\kappa$ ($\mathrm{m}^2$)')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.xaxis.grid(ls='--')
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.92, wspace=0.3, hspace=0.25)

    final_k = k_pred_physical[-1]
    np.savez_compressed(
        OUTPUT_DIR / f"mandel_inverse_noev_64_data_{RUN_TIMESTAMP}.npz",
        variable_history=data_history,
        steps=steps,
        k_bar_raw_history=k_bar_raw_history,
        k_bar_history=k_bar_history,
        k_pred_physical=k_pred_physical,
        X_obs_p_merged=X_obs_p_merged,
        p_true_merged=p_true_merged,
        X_obs_top=X_obs_top,
        u_true_top=u_true_top,
        v_true_top=v_true_top,
        t_sensors=t_sensors,
        X_init_p=X_init_p,
        X_init_mech=X_init_mech,
        X_left=X_left,
        X_right=X_right,
        X_bottom=X_bottom,
        X_top=X_top,
        X_domain_anchors=X_domain_anchors,
        pretrain_steps=np.asarray(losshistory_pretrain.steps),
        pretrain_loss_train=np.asarray(losshistory_pretrain.loss_train),
        pretrain_loss_test=np.asarray(losshistory_pretrain.loss_test),
        identify_steps=np.asarray(losshistory_identify.steps),
        identify_loss_train=np.asarray(losshistory_identify.loss_train),
        identify_loss_test=np.asarray(losshistory_identify.loss_test),
        finetune_steps=np.asarray(losshistory_finetune.steps),
        finetune_loss_train=np.asarray(losshistory_finetune.loss_train),
        finetune_loss_test=np.asarray(losshistory_finetune.loss_test),
        field_steps=np.asarray(losshistory_field.steps),
        field_loss_train=np.asarray(losshistory_field.loss_train),
        field_loss_test=np.asarray(losshistory_field.loss_test),
        lbfgs_steps=np.asarray(losshistory_lbfgs.steps),
        lbfgs_loss_train=np.asarray(losshistory_lbfgs.loss_train),
        lbfgs_loss_test=np.asarray(losshistory_lbfgs.loss_test),
    )
    final_parameters = {
        "k_bar": float(k_bar_final),
        "k_physical_m2": float(k_bar_final * k_star),
        "target_k_bar": 1.0,
        "target_k_m2": float(k),
        "relative_error": float(final_rel_error),
        "accuracy_percent": float(final_accuracy),
    }
    with open(OUTPUT_DIR / "final_parameters.json", "w", encoding="utf-8") as f:
        json.dump(final_parameters, f, indent=2, default=float)
    metadata = {
        "script": "mandel_反问题_单参数_无Ev版_64.py",
        "seed": SEED,
        "network_layers": layer_size,
        "activation": activation,
        "initializer": initializer,
        "batch_size": BATCH_SIZE,
        "NUM_DOMAIN_POINTS": NUM_DOMAIN_POINTS,
        "N_b": N_b,
        "N_i": N_i,
        "num_pressure_observation_points": int(len(X_obs_p_merged)),
        "num_top_observation_points": int(len(X_obs_top)),
        "td_0": td_0,
        "td_f": td_f,
        "xd_min": xd_min,
        "xd_max": xd_max,
        "yd_min": yd_min,
        "yd_max": yd_max,
        "p_star": p_star,
        "x_star": x_star,
        "t_star": t_star,
        "u_star": u_star,
        "D_star": D_star,
        "DD_star": DD_star,
        "S_overburden": S_overburden,
        "mu": mu,
        "mu_star": mu_star,
        "k": k,
        "k_star": k_star,
        "E": E,
        "E_star": E_star,
        "nu": nu,
        "b": b,
        "M": M,
        "Lx": Lx,
        "Ly": Ly,
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=float)
    plt.savefig(INVERSION_PLOT_FILE, dpi=150)
    print(f"[float64] 绘图完成！最终反演值: {final_k:.4e} m^2")
    plt.close(fig)

else:
    print("[float64] 未找到数据或数据为空，请检查 variables_kbar_noev_64_*.dat")


