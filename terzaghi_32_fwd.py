# deepxde_terzaghi.py
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import tensorflow as tf

dde.backend.set_default_backend("pytorch")
print("当前 DeepXDE 后端:", dde.backend.backend_name)
print("GPU available:", tf.config.list_physical_devices('GPU'))

SEED = 1234  # 你可以选任何喜欢的整数，比如 0, 42, 1234
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

    # v_yyy = dde.grad.jacobian(v_yy, X, i=0, j=0)
    # v_yyt = dde.grad.jacobian(v_yy, X, i=0, j=1)
    # p_yt = dde.grad.hessian(Y, X, component=0, i=0, j=1)
    #
    # p_ty = dde.grad.hessian(Y, X, component=0, i=1, j=0)
    # p_tt = dde.grad.hessian(Y, X, component=0, i=1, j=1)
    # p_yyy = dde.grad.jacobian(p_yy, X, i=0, j=0)
    # p_yyt = dde.grad.jacobian(p_yy, X, i=0, j=1)


    Mp = p_t - p_yy
    Msy = (1 + 2*nu_star) * v_yy - b * p_y

    # Mp_y = p_ty - p_yyy
    # Mp_t = p_tt - p_yyt
    # Msy_y = (1 + 2*nu_star) * v_yyy - b * p_yy
    # Msy_t = (1 + 2 * nu_star) * v_yyt - b * p_yt

    return [Mp, Msy]
    # return [Mp, Msy, Msy_y, Msy_t, Mp_y, Mp_t]


# -------------------- Geometry and Data --------------------
geom = dde.geometry.Interval(yd_min, yd_max)
timedomain = dde.geometry.TimeDomain(td_0, td_f)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Number of training points
num_domain = int(NUM_SAMPLES / 2)
N_b = 2500  # 每条边界条件点数
N_i = 2500  # 每个初始条件点数
N_init_total =  N_i # 定义初始点总数 (5000)

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
t_top = np.random.uniform(td_0 , td_f, (N_b, 1))
X_top = np.hstack((y_top, t_top))

# 3. Initial (t = td_0)
# 【修改点】：空间只取到 yd_max - eps_y，避开 y=y_max
# 注意：这里使用 N_init_total (5000) 以匹配你之前的修改
y_init = np.random.uniform(yd_min, yd_max - eps_y, (N_init_total, 1))
t_init = np.full((N_init_total, 1), td_0)
X_init = np.hstack((y_init, t_init))


p_init_values = np.tanh(75 * (yd_max - X_init[:, 0:1]))
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

# 1. 普通训练 / 可选 RAR 训练
model.compile("adam", lr=1e-3) # 带权重
model.train(iterations=15000)

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
model.train()


# ---评估网格和解析解计算--------
log10p = lambda x: np.log10(1. + x)
Nx, Ny, Nt = 5, 100, 200
ys = np.linspace(yd_min, yd_max, Ny * 3)  # y_d 网格，300点 [0,1]
ts = (np.logspace(0.1, log10p(td_f * t_star), Nt * 3) - 1.0) / t_star  # t_d 网格，对数间隔，600点

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

# -------绘图设置------
legend_settings = {
    'bbox_to_anchor': (1.05, 0.25),
    'loc': 'upper left',
    'fontsize': 'medium',
    'fancybox': True,
    'frameon': False}

subplot_adust = [0.1, 0.15, 0.85, 0.85, 0.45, 0.35]
line_color = ['red', 'orange', 'blue', 'green', 'purple', 'cyan', 'gold', 'black']

# --------------线图比较：PINN vs 解析解--------
fig, ax = plt.subplots(2, 3, figsize=(12, 7))

for i in range(len(ts_line_plot)):  # 8点
    time_plot = np.around(ts_line_plot[i] * t_star, decimals=2)  # 物理 t (s)
    # 上排 p
    ax[0, 0].plot(ys_line_plot, pred['p_true_line'][:, i], color=line_color[i], linestyle='-', label='t = ' + str(time_plot))
    ax[0, 1].plot(ys_line_plot, pred['p_line'][:, i], color=line_color[i], linestyle='--', label='t = ' + str(time_plot))
    ax[0, 2].plot(ys_line_plot, np.abs(pred['p_line'][:, i] - pred['p_true_line'][:, i]), color=line_color[i], linestyle='-.', label='t = ' + str(time_plot))
    # 下排 v
    ax[1, 0].plot(ys_line_plot, pred['v_true_line'][:, i], color=line_color[i], linestyle='-')
    ax[1, 1].plot(ys_line_plot, pred['v_line'][:, i], color=line_color[i], linestyle='--')
    ax[1, 2].plot(ys_line_plot, np.abs(pred['v_line'][:, i] - pred['v_true_line'][:, i]), color=line_color[i], linestyle='-.', label='t = ' + str(time_plot))

ax[0, 0].set_xlabel('y(m)'); ax[0, 0].set_ylabel('$p^*$ (MPa)')
ax[0, 1].set_xlabel('y(m)'); ax[0, 1].set_ylabel('$p$ (MPa)')
ax[0, 2].set_xlabel('y(m)'); ax[0, 2].set_ylabel('Error=|$p^* - p$| (MPa)')
ax[0, 0].set_ylim(0.0, 0.11); ax[0, 1].set_ylim(0.0, 0.11)
ax[1, 0].set_xlabel('y(m)'); ax[1, 0].set_ylabel('$v^*$ (m)')
ax[1, 1].set_xlabel('y(m)'); ax[1, 1].set_ylabel('$v$ (m)')
ax[1, 2].set_xlabel('y(m)'); ax[1, 2].set_ylabel('Error=|$v^* - v$| (m)')
ax[1, 0].set_ylim(-0.031, 0.001); ax[1, 1].set_ylim(-0.031, 0.001)
ax[0, 2].legend(**legend_settings)
ax[0, 0].grid(True); ax[0, 1].grid(True); ax[0, 2].grid(True)
ax[1, 0].grid(True); ax[1, 1].grid(True); ax[1, 2].grid(True)
plt.subplots_adjust(*subplot_adust)
plt.show()

#----------轮廓图：2行3列布局（绝对误差版）-------
# 1. 数据准备：计算绝对误差
p_pred_grid = pred['p'][:, 0].reshape(Ydtest.shape)
p_true_grid = pred['p_true_contour']
# 修改点：增加 np.abs()
p_error_grid = np.abs(p_pred_grid - p_true_grid)

v_pred_grid = pred['v'][:, 0].reshape(Ydtest.shape)
v_true_grid = pred['v_true_contour']
# 修改点：增加 np.abs()
v_error_grid = np.abs(v_pred_grid - v_true_grid)

# 坐标网格
y_plot = input_test[0] * x_star
t_plot = np.log10(np.maximum(input_test[1] * t_star, 1e-12))

# 创建画布
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

# ==========================================
# 第一行：孔压 p (MPa)
# ==========================================

# 1. 左图：精确解 p*
im00 = ax[0, 0].pcolor(t_plot, y_plot, p_true_grid, cmap='jet_r', vmin=0, vmax=0.1)
plt.colorbar(im00, ax=ax[0, 0])
ax[0, 0].set_ylabel('y (m)')
ax[0, 0].set_title('$p^*$ [MPa]')

# 2. 中图：预测解 p
im01 = ax[0, 1].pcolor(t_plot, y_plot, p_pred_grid, cmap='jet_r', vmin=0, vmax=0.1)
plt.colorbar(im01, ax=ax[0, 1])
ax[0, 1].set_title('$p$ [MPa]')

# 3. 右图：绝对误差 |p - p*|
# 修改点：cmap 改为 jet (蓝=0, 红=大), vmin=0
max_p_err = np.max(p_error_grid)
im02 = ax[0, 2].pcolor(t_plot, y_plot, p_error_grid, cmap='jet', vmin=0, vmax=0.04)
plt.colorbar(im02, ax=ax[0, 2])
ax[0, 2].set_title('Error $|p - p^*|$ [MPa]')


# ==========================================
# 第二行：位移 v (m)
# ==========================================
# 获取 v 的最大最小值以统一色标范围
v_min, v_max = np.min(v_true_grid), np.max(v_true_grid)

# 1. 左图：精确解 v*
im10 = ax[1, 0].pcolor(t_plot, y_plot, v_true_grid, cmap='jet_r', vmin=v_min, vmax=v_max)
plt.colorbar(im10, ax=ax[1, 0])
ax[1, 0].set_ylabel('y (m)')
ax[1, 0].set_xlabel('log10(t) (s)')
ax[1, 0].set_title('$v^*$ [m]')

# 2. 中图：预测解 v
im11 = ax[1, 1].pcolor(t_plot, y_plot, v_pred_grid, cmap='jet_r', vmin=v_min, vmax=v_max)
plt.colorbar(im11, ax=ax[1, 1])
ax[1, 1].set_xlabel('log10(t) (s)')
ax[1, 1].set_title('$v$ [m]')

# 3. 右图：绝对误差 |v - v*|
# 修改点：cmap 改为 jet, vmin=0
max_v_err = np.max(v_error_grid)
im12 = ax[1, 2].pcolor(t_plot, y_plot, v_error_grid, cmap='jet', vmin=0, vmax=max_v_err)
plt.colorbar(im12, ax=ax[1, 2])
ax[1, 2].set_xlabel('log10(t) (s)')
ax[1, 2].set_title('Error $|v - v^*|$ [m]')

# ==========================================
# 统一调整
t_phys_min = 10**0.1 - 1.0  # ≈ 0.258925 秒
t_plot_min = np.log10(t_phys_min)
# ==========================================
for i in range(2):
    for j in range(3):
        ax[i, j].set_ylim(np.min(y_plot), np.max(y_plot))
        ax[i, j].set_xlim(t_plot_min, np.max(t_plot))

plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.1, wspace=0.25, hspace=0.3)
plt.show()



