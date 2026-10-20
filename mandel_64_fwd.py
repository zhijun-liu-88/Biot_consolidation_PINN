import os
import json
from datetime import datetime

import numpy as np
import deepxde as dde

dde.backend.set_default_backend("pytorch")
dde.config.set_default_float("float64")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

torch.set_default_dtype(torch.float64)


SEED = 1234  # 你可以选任何喜欢的整数，比如 0, 42, 1234
dde.config.set_random_seed(SEED)  # 固定 DeepXDE 和 后端(Torch/TF) 的种子
np.random.seed(SEED)              # 固定 Numpy 的种子 (这对你手动采样的点至关重要)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("outputs", f"mandel64_forward_{RUN_TIMESTAMP}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(OUTPUT_DIR)}")

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
mu = 6. * milli * poise
k = 1e-12
E = 120e6
nu = 0.25
b = 1.0
M = 120e6
Lx = 50*m
Ly = 10*m

# 计算派生参数
Kdr = E/(3*(1-2*nu))
B = (b*M)/(Kdr+M*(b**2))
nu_u = (3*nu+b*B*(1-2*nu))/(3-b*B*(1-2*nu))
F = S_overburden*Ly*1*m
Gshr = E/2/(1+nu)
Gama = B*(1+nu_u)/(3*(1-nu_u))
p0 = (-F)*B*(1+nu_u)/(3*Ly)

# 特征尺度归一化
p_star = p0
x_star = Ly
t_star = (b**2/Kdr + 1/M)*mu*(x_star**2)/(k)
nu_star = (1 - 2*nu)/(1 + nu)
u_star = p_star*x_star/Kdr
D_star = b*M/((b**2)*M + Kdr)

print(f"D_star = {D_star}, p0 = {p0/MPa} MPa")

td_0 = 0.0
td_f = 1000/t_star
xd_min, xd_max = 0.0, Lx/x_star
yd_min, yd_max = 0.0, Ly/x_star
print(td_f)
geom = dde.geometry.Rectangle([xd_min, yd_min], [xd_max, yd_max])
timedomain = dde.geometry.TimeDomain(td_0, td_f)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


def pdes(x, y):
    """流体力学PDE: 压力扩散方程"""
    p, u, v, Ev = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    xd, yd, td = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    # 计算梯度
    p_t = dde.grad.jacobian(y, x, i=0, j=2)  # dp/dt
    p_x = dde.grad.jacobian(y, x, i=0, j=0)  # dp/dx
    p_y = dde.grad.jacobian(y, x, i=0, j=1)  # dp/dy
    p_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)  # d²p/dx²
    p_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)  # d²p/dy²

    # 体积应变的时间导数 (来自固体力学)
    Svd = Ev - b * p
    Svd_td = dde.grad.jacobian(Svd, x, i=0, j=2)

    # 压力扩散方程: p_t - (p_xx + p_yy) + D_star * Svd_td = 0
    pde1 = p_t - (p_xx + p_yy) + D_star * Svd_td

    u_x = dde.grad.jacobian(y, x, i=1, j=0)

    v_y = dde.grad.jacobian(y, x, i=2, j=1)

    # 二阶导数
    u_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    u_xy = dde.grad.hessian(y, x, component=1, i=0, j=1)
    u_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)
    v_xy = dde.grad.hessian(y, x, component=2, i=1, j=0)
    v_yy = dde.grad.hessian(y, x, component=2, i=1, j=1)
    Ev_x = dde.grad.jacobian(y, x, i=3, j=0)
    Ev_y = dde.grad.jacobian(y, x, i=3, j=1)

    pde2 = Ev_x + 1.5 * nu_star * (u_xx + u_yy) + (nu_star/2) * (u_xx + v_xy) - b * p_x
    pde3 = Ev_y + 1.5 * nu_star * (v_xx + v_yy) + (nu_star/2) * (u_xy + v_yy) - b * p_y
    pde4 = (u_x + v_y) - Ev

    return [pde1, pde2, pde3, pde4]

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

# 初始条件 - 应力
def ic_sxx(x, y, X):
    p, u, v, Ev = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    Exx = dde.grad.jacobian(y, x, i=1, j=0)
    Sxx = (1-nu_star)*Ev + 3*nu_star*Exx - b*(p)
    return Sxx - S_overburden_norm

def ic_syy(x, y, aux=None):
    p, u, v, Ev = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    Eyy = dde.grad.jacobian(y, x, i=2, j=1)
    Syy = (1-nu_star)*Ev + 3*nu_star*Eyy - b*(p)
    return Syy

# 边界条件 - 左侧应力
def bc_left_sxx(x, y, aux=None):
    p, u, v, Ev = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    Exx = dde.grad.jacobian(y, x, i=1, j=0)
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
    p, u, v, Ev = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    Eyy = dde.grad.jacobian(y, x, i=2, j=1)
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

# ==============================================================================
# 1. 采样与数据生成 (修改后)
# ==============================================================================

# 定义采样数量（与反问题脚本保持一致）
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

# ==============================================================================
# 3. 构建数据与模型
# ==============================================================================

# 注意：


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

    x_domain = np.vstack((x_early, x_mid, x_late))
    y_domain = np.vstack((y_early, y_mid, y_late))
    t_domain = np.vstack((t_early, t_mid, t_late))
    return np.hstack((x_domain, y_domain, t_domain))


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

# ----------------------- 网络 -----------------------
layer_size = [3] + 4*[128] + [4]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)


loss_weights = [1.0] * 4 + [100.0] * 15
RAR_ENABLED = True
rar_callback = dde.callbacks.PDEPointResampler(period=2000, pde_points=True, bc_points=True)
train_callbacks = [rar_callback] if RAR_ENABLED else []

model.compile("adam", lr=1e-3, loss_weights=loss_weights)
losshistory_adam, train_state_adam = model.train(epochs=15000, callbacks=train_callbacks)

# Adam 结束后切到 float64，再用 L-BFGS 收尾
dde.config.set_default_float("float64")
torch.set_default_dtype(torch.float64)
model.net.double()
model.compile("L-BFGS", loss_weights=loss_weights)
losshistory_lbfgs, train_state_lbfgs = model.train()


Nx, Ny, Nt = 11, 100, 200
log10p = lambda x: np.log10(1. + x)

xs = np.linspace(xd_min, xd_max, Nx)
ys = np.linspace(yd_min, yd_max, Ny * 3)
ts = (np.logspace(0.1, log10p(td_f * t_star), Nt * 3) - 1.0) / t_star
Xdtest, Ydtest, Tdtest = np.meshgrid(xs, ys, ts, indexing="ij")

# 用于 u 的更密集网格
xs_u = np.linspace(xd_min, xd_max, Nx * 10)
Xdtest_u_plot, Ydtest_u_plot, Tdtest_u_plot = np.meshgrid(xs_u, ys, ts, indexing='ij')

# contour 绘图用的坐标
input_test = [Xdtest[10, :, :], Ydtest[10, :, :], Tdtest[10, :, :]]
y_plot = input_test[1] * x_star
t_plot = np.log10(input_test[2] * t_star)

input_test_u = [Xdtest_u_plot[:, 5, :], Ydtest_u_plot[:, 5, :], Tdtest_u_plot[:, 5, :]]
xx_plot = input_test_u[0] * x_star
tt_plot = np.log10(input_test_u[2] * t_star)

# 线图网格（关键：按原始代码设置）
xs_p_line = xs[-1:]  # 只取最后一个 x 值
ys_p_line = np.linspace(yd_min, yd_max, Ny)
ts_p_line = [td_f * 0.01, td_f * 0.025, td_f * 0.05, td_f * 0.1, td_f * 0.25, td_f * 0.5, td_f * 0.75, td_f * 1.0]
input_test_p_line = np.meshgrid(xs_p_line, ys_p_line, ts_p_line, indexing='ij')

xs_u_line = np.linspace(xd_min, xd_max, Nx * 10)  # u 使用更密集的网格
ys_u_line = ys[50]  # 只取中间一个 y 值
ts_u_line = [td_f * 0.01, td_f * 0.025, td_f * 0.05, td_f * 0.1, td_f * 0.25, td_f * 0.5, td_f * 0.75, td_f * 1.0]
input_test_u_line = np.meshgrid(xs_u_line, ys_u_line, ts_u_line, indexing='ij')


# ==============================================================
# 计算 Mandel 问题真解
# ==============================================================

def mandel_roots(nu, nu_u):
    """求解 Mandel 问题中的特征根"""

    def func(z):
        return np.tan(z) - ((1 - nu) / (nu_u - nu)) * z

    a = np.around((np.linspace(1, 100, 100) * 2 - 1) * (np.pi / 2) * 0.999999834, decimals=6)
    roots = np.zeros_like(a)
    for i in range(100):
        roots[i] = fsolve(func, float(a[i]))[0]  # 修复：提取标量并取第一个元素
    return roots


def compute_analytical_solution():
    """计算完整的解析解"""
    global input_test, input_test_u, input_test_p_line, input_test_u_line
    global nu, nu_u, F, B, Ly, Lx, Gshr, MPa, m

    roots = mandel_roots(nu, nu_u)

    # ===== 压力 p =====
    # contour plot
    p_true = np.zeros_like(input_test[1])
    for i in range(100):
        alpha = roots[i]
        T1 = np.sin(alpha) / (alpha - np.sin(alpha) * np.cos(alpha))
        T2 = np.cos(alpha * (input_test[1])) - np.cos(alpha)
        T3 = np.exp(-(alpha ** 2) * input_test[2])
        p_true += T1 * T2 * T3
    term1 = 2 * (-F) * B * (1 + nu_u) / (3 * Ly)
    p_true = term1 * p_true

    # line plot
    p_true_line = np.zeros_like(input_test_p_line[1])
    for i in range(100):
        alpha = roots[i]
        T1 = np.sin(alpha) / (alpha - np.sin(alpha) * np.cos(alpha))
        T2 = np.cos(alpha * (input_test_p_line[1])) - np.cos(alpha)
        T3 = np.exp(-(alpha ** 2) * input_test_p_line[2])
        p_true_line += T1 * T2 * T3
    p_true_line = term1 * p_true_line / MPa

    # ===== 垂直位移 v =====
    # contour plot
    v_true_part1 = np.zeros_like(input_test[2])
    v_true_part2 = np.zeros_like(input_test[2])
    term1 = F * nu / (2 * Gshr * Ly)
    term2 = F * nu_u / (Gshr * Ly)
    term3 = F / Gshr
    for i in range(100):
        alpha = roots[i]
        T1 = (np.sin(alpha) * np.cos(alpha)) / (alpha - np.sin(alpha) * np.cos(alpha))
        T2 = np.exp(-(alpha ** 2) * input_test[2])
        T3 = (np.cos(alpha)) / (alpha - np.sin(alpha) * np.cos(alpha))
        T4 = np.sin(alpha * (input_test[1]))
        v_true_part1 += T1 * T2
        v_true_part2 += T3 * T4 * T2
    v_true = (np.ones_like(input_test[2]) * term1 - term2 * v_true_part1) * input_test[1] * Ly + term3 * v_true_part2

    # line plot
    v_true_part1 = np.zeros_like(input_test_p_line[2])
    v_true_part2 = np.zeros_like(input_test_p_line[2])
    for i in range(100):
        alpha = roots[i]
        T1 = (np.sin(alpha) * np.cos(alpha)) / (alpha - np.sin(alpha) * np.cos(alpha))
        T2 = np.exp(-(alpha ** 2) * input_test_p_line[2])
        T3 = (np.cos(alpha)) / (alpha - np.sin(alpha) * np.cos(alpha))
        T4 = np.sin(alpha * (input_test_p_line[1]))
        v_true_part1 += T1 * T2
        v_true_part2 += T3 * T4 * T2
    v_true_line = (np.ones_like(input_test_p_line[2]) * term1 - term2 * v_true_part1) * input_test_p_line[
        1] * Ly + term3 * v_true_part2

    # ===== 水平位移 u =====
    # contour plot
    u_true = np.zeros_like(input_test_u[2])
    term1 = -F * (1 - nu) / (2 * Gshr * Ly)
    term2 = F * (1 - nu_u) / (Gshr * Ly)
    for i in range(100):
        alpha = roots[i]
        T1 = (np.sin(alpha) * np.cos(alpha)) / (alpha - np.sin(alpha) * np.cos(alpha))
        T2 = np.exp(-(alpha ** 2) * input_test_u[2])
        u_true += T1 * T2
    u_true = (term1 + term2 * u_true) * (Lx - input_test_u[0] * x_star)

    # line plot
    u_true_line = np.zeros_like(input_test_u_line[2])
    for i in range(100):
        alpha = roots[i]
        T1 = (np.sin(alpha) * np.cos(alpha)) / (alpha - np.sin(alpha) * np.cos(alpha))
        T2 = np.exp(-(alpha ** 2) * input_test_u_line[2])
        u_true_line += T1 * T2
    u_true_line = (term1 + term2 * u_true_line) * (Lx - input_test_u_line[0] * x_star)

    return p_true, v_true, u_true, p_true_line, v_true_line, u_true_line


# ==============================================================
# 从模型预测
# ==============================================================

def predict_fields(model, Xd, Yd, Td, batch_size=10000):
    """统一的预测函数（分批处理避免显存溢出）"""
    X_input = np.hstack((Xd.flatten()[:, None], Yd.flatten()[:, None], Td.flatten()[:, None]))

    # 分批预测
    n_samples = X_input.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    y_pred_list = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_input = X_input[start_idx:end_idx]
        batch_pred = model.predict(batch_input)
        y_pred_list.append(batch_pred)

    y_pred = np.vstack(y_pred_list)

    p_pred = y_pred[:, 0:1].reshape(Xd.shape)
    u_pred = y_pred[:, 1:2].reshape(Xd.shape)
    v_pred = y_pred[:, 2:3].reshape(Xd.shape)
    Ev_pred = y_pred[:, 3:4].reshape(Xd.shape)
    return p_pred, u_pred, v_pred, Ev_pred


# ==============================================================
# 绘图函数（完全复现原始代码）
# ==============================================================

def plot_contours_and_lines(model, itr=1):
    """
    修改版: 强制所有子图为严格正方形 (Fixed Square Aspect Ratio)
    即使放大或调整窗口大小，长宽比也保持 1:1
    """
    import torch
    import gc
    import matplotlib.pyplot as plt
    import numpy as np

    # 引入全局变量
    global Xdtest, Ydtest, Tdtest, Xdtest_u_plot, Ydtest_u_plot, Tdtest_u_plot
    global input_test, input_test_u, input_test_p_line, input_test_u_line
    global y_plot, t_plot, xx_plot, tt_plot
    global xs_p_line, ys_p_line, ts_p_line, xs_u_line
    global p_star, u_star, x_star, t_star, MPa, m

    contour_dir = os.path.join(OUTPUT_DIR, "contours")
    line_dir = os.path.join(OUTPUT_DIR, "lines")
    os.makedirs(contour_dir, exist_ok=True)
    os.makedirs(line_dir, exist_ok=True)

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print("计算解析解...")
    p_true, v_true, u_true, p_true_line, v_true_line, u_true_line = compute_analytical_solution()

    print("模型预测 - p, u, v (主网格)...")
    p_pred, u_pred, v_pred, _ = predict_fields(model, Xdtest, Ydtest, Tdtest, batch_size=5000)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print("模型预测 - u (密集网格)...")
    _, u_pred_u, _, _ = predict_fields(model, Xdtest_u_plot, Ydtest_u_plot, Tdtest_u_plot, batch_size=5000)

    print("模型预测 - 线图数据...")
    p_line_pred, _, v_line_pred, _ = predict_fields(model, *input_test_p_line, batch_size=5000)
    _, u_line_pred, _, _ = predict_fields(model, *input_test_u_line, batch_size=5000)

    # 反归一化
    pred_itr = {
        'p': (p_pred[10, :, :]) * p_star / MPa,
        'p_true': p_true / MPa,
        'v': (v_pred[10, :, :]) * u_star / m,
        'v_true': -v_true / m,
        'u': (u_pred_u[:, 5, :]) * u_star / m,
        'u_true': u_true / m,
        'p_line': p_line_pred * p_star / MPa,
        'p_true_line': p_true_line,
        'v_line': v_line_pred * u_star / m,
        'v_true_line': -v_true_line / m,
        'u_line': u_line_pred * u_star / m,
        'u_true_line': u_true_line / m
    }
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"mandel64_forward_data_itr{itr:02d}.npz"),
        xs=xs,
        ys=ys,
        ts=ts,
        xs_u=xs_u,
        xs_p_line=xs_p_line,
        ys_p_line=ys_p_line,
        ts_p_line=np.asarray(ts_p_line),
        xs_u_line=xs_u_line,
        ys_u_line=np.asarray(ys_u_line),
        ts_u_line=np.asarray(ts_u_line),
        Xdtest=Xdtest,
        Ydtest=Ydtest,
        Tdtest=Tdtest,
        Xdtest_u_plot=Xdtest_u_plot,
        Ydtest_u_plot=Ydtest_u_plot,
        Tdtest_u_plot=Tdtest_u_plot,
        y_plot=y_plot,
        t_plot=t_plot,
        xx_plot=xx_plot,
        tt_plot=tt_plot,
        p_pred=pred_itr["p"],
        p_true=pred_itr["p_true"],
        p_error=np.abs(pred_itr["p"] - pred_itr["p_true"]),
        v_pred=pred_itr["v"],
        v_true=pred_itr["v_true"],
        v_error=np.abs(pred_itr["v"] - pred_itr["v_true"]),
        u_pred=pred_itr["u"],
        u_true=pred_itr["u_true"],
        u_error=np.abs(pred_itr["u"] - pred_itr["u_true"]),
        p_line=pred_itr["p_line"],
        p_true_line=pred_itr["p_true_line"],
        p_line_error=np.abs(pred_itr["p_line"] - pred_itr["p_true_line"]),
        v_line=pred_itr["v_line"],
        v_true_line=pred_itr["v_true_line"],
        v_line_error=np.abs(pred_itr["v_line"] - pred_itr["v_true_line"]),
        u_line=pred_itr["u_line"],
        u_true_line=pred_itr["u_true_line"],
        u_line_error=np.abs(pred_itr["u_line"] - pred_itr["u_true_line"]),
        adam_steps=np.asarray(losshistory_adam.steps),
        adam_loss_train=np.asarray(losshistory_adam.loss_train),
        adam_loss_test=np.asarray(losshistory_adam.loss_test),
        lbfgs_steps=np.asarray(losshistory_lbfgs.steps),
        lbfgs_loss_train=np.asarray(losshistory_lbfgs.loss_train),
        lbfgs_loss_test=np.asarray(losshistory_lbfgs.loss_test),
    )
    metadata = {
        "script": "mandel-64.py",
        "seed": SEED,
        "network_layers": layer_size,
        "activation": activation,
        "initializer": initializer,
        "loss_weights": loss_weights,
        "RAR_ENABLED": RAR_ENABLED,
        "NUM_DOMAIN_POINTS": NUM_DOMAIN_POINTS,
        "N_b": N_b,
        "N_i": N_i,
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
        "S_overburden": S_overburden,
        "mu": mu,
        "k": k,
        "E": E,
        "nu": nu,
        "b": b,
        "M": M,
        "Lx": Lx,
        "Ly": Ly,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=float)

    # =========================================================
    # Contour 绘图 (强制正方形)
    # =========================================================
    # figsize 设置为接近 1:1 的比例，比如 (12, 11)
    # 统一采用论文插图的字体和线宽。这里用局部 rc_context，避免影响训练过程中的其他图。
    publication_rc = {
        "font.family": "Times New Roman",
        "font.size": 16,
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    }

    # 经典时刻采用物理时间。若离散时间网格中没有完全相同的点，则选取最近的时间点。
    classic_times = np.asarray([0.25, 25.0, 50.0, 250.0], dtype=float)
    physical_times = np.asarray(ts, dtype=float) * t_star
    classic_indices = np.asarray(
        [int(np.argmin(np.abs(physical_times - value))) for value in classic_times],
        dtype=int,
    )
    selected_times = physical_times[classic_indices]
    x_xy = np.asarray(xs_u, dtype=float) * x_star
    y_xy = np.asarray(ys, dtype=float) * x_star

    def build_xy_fields(data, direction):
        """把 Mandel 一维剖面扩展成经典时刻的 x-y 场。"""
        data = np.asarray(data, dtype=float)
        fields = []
        for time_index in classic_indices:
            if direction == "y":
                profile = data[:, time_index]
                field = np.repeat(profile[:, None], x_xy.size, axis=1)
            else:
                profile = data[:, time_index]
                field = np.repeat(profile[None, :], y_xy.size, axis=0)
            fields.append(field)
        return np.asarray(fields)

    def save_classic_xy_figure(
        symbol,
        unit,
        exact_data,
        predicted_data,
        direction,
        value_limits,
        error_max,
        output_stem,
    ):
        """绘制 3×4 经典时刻云图。"""
        exact_fields = build_xy_fields(exact_data, direction)
        predicted_fields = build_xy_fields(predicted_data, direction)
        error_fields = np.abs(predicted_fields - exact_fields)

        with plt.rc_context(publication_rc):
            fig, axes = plt.subplots(
                3, 4, figsize=(13.6, 9.7), constrained_layout=False
            )

            row_specs = (
                (exact_fields, rf"${symbol}$", value_limits[0], value_limits[1]),
                (predicted_fields, rf"$\hat{{{symbol}}}$", value_limits[0], value_limits[1]),
                (error_fields, rf"$|{symbol}-\hat{{{symbol}}}|$", 0.0, error_max),
            )
            images = []
            for row, (fields, row_label, vmin, vmax) in enumerate(row_specs):
                for col, title_time in enumerate(classic_times):
                    axis = axes[row, col]
                    image = axis.pcolormesh(
                        x_xy, y_xy, fields[col],
                        shading="auto", cmap="jet", vmin=vmin, vmax=vmax,
                        rasterized=True,
                    )
                    images.append(image)
                    axis.set_xlim(float(x_xy.min()), float(x_xy.max()))
                    axis.set_ylim(float(y_xy.min()), float(y_xy.max()))
                    axis.set_box_aspect(1)
                    axis.tick_params(labelsize=12, direction="out", width=1.1, length=4)
                    for spine in axis.spines.values():
                        spine.set_linewidth(1.1)
                    if row == 0:
                        axis.set_title(
                            rf"$t={title_time:g}\ (\mathrm{{s}})$",
                            fontsize=15, pad=6,
                        )
                    if col == 0:
                        axis.set_ylabel(row_label + f" ({unit})\n" + r"$y$ (m)", fontsize=14)
                    if row == 2:
                        axis.set_xlabel(r"$x$ (m)", fontsize=14)

            fig.subplots_adjust(
                left=0.085, right=0.895, bottom=0.08, top=0.93,
                wspace=0.32, hspace=0.35,
            )
            for row in range(3):
                colorbar = fig.colorbar(
                    images[row * 4 + 3],
                    ax=axes[row, :],
                    fraction=0.018,
                    pad=0.015,
                    aspect=25,
                )
                colorbar.ax.tick_params(labelsize=11)
                if row == 2:
                    colorbar.set_ticks(np.linspace(0.0, error_max, 5))

            for extension in ("png", "pdf", "svg"):
                save_options = {"bbox_inches": "tight"}
                if extension == "png":
                    save_options["dpi"] = 300
                fig.savefig(os.path.join(OUTPUT_DIR, f"{output_stem}.{extension}"), **save_options)
            plt.close(fig)

        # 同步保存作图使用的离散数据，便于复核每一个像素场和所选时间点。
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"{output_stem}_data.npz"),
            x=x_xy,
            y=y_xy,
            title_times_s=classic_times,
            selected_times_s=selected_times,
            selected_indices=classic_indices,
            true_fields=exact_fields,
            pred_fields=predicted_fields,
            error_fields=error_fields,
            all_times_s=physical_times,
            cmap="jet",
            value_vmin=value_limits[0],
            value_vmax=value_limits[1],
            error_vmin=0.0,
            error_vmax=error_max,
        )

    # 分别输出压力、水平位移和竖向位移的经典时刻云图。
    save_classic_xy_figure(
        "p", "MPa", pred_itr["p_true"], pred_itr["p"], "y",
        (0.0, 0.60), 0.060, "mandel_p_xy_classic_times",
    )
    save_classic_xy_figure(
        "u", "m", pred_itr["u_true"], pred_itr["u"], "x",
        (0.0, 0.80), 0.014, "mandel_u_xy_classic_times",
    )
    save_classic_xy_figure(
        "v", "m", pred_itr["v_true"], pred_itr["v"], "y",
        (0.0, 0.08), 0.0022, "mandel_v_xy_classic_times",
    )

    # 线图按列排列 u、v、p，按行排列解析解、预测解和绝对误差。
    line_colors = ["red", "darkorange", "blue", "green", "purple", "cyan", "gold", "black"]
    line_times = np.asarray(ts_p_line, dtype=float) * t_star
    x_line = np.asarray(xs_u_line, dtype=float) * x_star
    y_line = np.asarray(ys_p_line, dtype=float) * x_star
    u_exact = np.asarray(pred_itr["u_true_line"])[:, 0, :]
    u_predicted = np.asarray(pred_itr["u_line"])[:, 0, :]
    v_exact = np.asarray(pred_itr["v_true_line"])[0, :, :]
    v_predicted = np.asarray(pred_itr["v_line"])[0, :, :]
    p_exact = np.asarray(pred_itr["p_true_line"])[0, :, :]
    p_predicted = np.asarray(pred_itr["p_line"])[0, :, :]

    line_columns = (
        (x_line, u_exact, u_predicted, "u", "x", "m", (0.0, 0.80)),
        (y_line, v_exact, v_predicted, "v", "y", "m", (0.0, 0.08)),
        (y_line, p_exact, p_predicted, "p", "y", "MPa", (0.0, 0.60)),
    )

    with plt.rc_context(publication_rc):
        fig, axes = plt.subplots(3, 3, figsize=(15.8, 11.8))
        for col, (coordinate, exact, predicted, symbol, coordinate_symbol, unit, value_ylim) in enumerate(line_columns):
            error = np.abs(predicted - exact)
            for index, color in enumerate(line_colors):
                axes[0, col].plot(coordinate, exact[:, index], color=color, lw=1.5, ls="-")
                axes[1, col].plot(coordinate, predicted[:, index], color=color, lw=1.5, ls="--")
                axes[2, col].plot(
                    coordinate, error[:, index], color=color, lw=1.5, ls="-.",
                    label=rf"$t={line_times[index]:g}$ s",
                )

            axes[0, col].set_ylabel(rf"${symbol}^*$ ({unit})")
            axes[1, col].set_ylabel(rf"$\hat{{{symbol}}}$ ({unit})")
            axes[2, col].set_ylabel(rf"$|\hat{{{symbol}}}-{symbol}^*|$ ({unit})")
            axes[0, col].set_ylim(*value_ylim)
            axes[1, col].set_ylim(*value_ylim)

            error_top = float(np.nanmax(error))
            if not np.isfinite(error_top) or error_top <= 0.0:
                error_top = 1.0
            axes[2, col].set_ylim(0.0, error_top * 1.08)

            for row in range(3):
                axes[row, col].set_xlabel(rf"${coordinate_symbol}$ (m)")
                axes[row, col].grid(True, color="0.82", lw=0.7, alpha=0.8)
                axes[row, col].tick_params(top=True, right=True, length=4)
                axes[row, col].set_box_aspect(0.82)

        handles, labels = axes[2, 2].get_legend_handles_labels()
        fig.legend(
            handles, labels, title="Time", loc="center left",
            bbox_to_anchor=(0.885, 0.50), frameon=False,
            fontsize=14, title_fontsize=15,
        )
        fig.subplots_adjust(left=0.075, right=0.875, top=0.97, bottom=0.075, wspace=0.34, hspace=0.34)

        line_stem = f"Mandel_LinePlot_itr{itr:02d}"
        for extension in ("png", "pdf", "svg"):
            save_options = {"bbox_inches": "tight"}
            if extension == "png":
                save_options["dpi"] = 600
            fig.savefig(os.path.join(line_dir, f"{line_stem}.{extension}"), **save_options)
            fig.savefig(os.path.join(OUTPUT_DIR, f"mandel_line_comparison_3x3.{extension}"), **save_options)
        plt.close(fig)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "mandel_line_comparison_3x3_data.npz"),
        times=line_times,
        x_u=x_line,
        y_vp=y_line,
        u_exact=u_exact,
        u_predicted=u_predicted,
        u_error=np.abs(u_predicted - u_exact),
        v_exact=v_exact,
        v_predicted=v_predicted,
        v_error=np.abs(v_predicted - v_exact),
        p_exact=p_exact,
        p_predicted=p_predicted,
        p_error=np.abs(p_predicted - p_exact),
    )

    print(f"✅ Postprocessing completed for iteration {itr}")
    print(f"Classic-time contour plots saved to: {OUTPUT_DIR}")
    print(f"Line plots saved to: {line_dir}")

    # (线图部分代码不需要动，通常线图长方形比较好看，如果也要正方形可以在线图循环里加 set_box_aspect(1))

# ==============================================================
# 使用方式
# ==============================================================

# 假设你已经训练好模型并且定义了所有必要的全局变量：
# xd_min, xd_max, yd_min, yd_max, td_f, t_star, x_star
# nu, nu_u, F, B, Ly, Lx, Gshr, MPa, m, p_star, u_star

# 直接调用：
plot_contours_and_lines(model, itr=10000)
