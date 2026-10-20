# deepxde_terzaghi.py
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import torch
import matplotlib.lines as mlines
import os
import json
from datetime import datetime
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

dde.backend.set_default_backend("pytorch")
dde.config.set_default_float("float64")  # DeepXDE internal tensors use float64
torch.set_default_dtype(torch.float64)   # PyTorch tensors default to float64
print("当前 DeepXDE 后端:", dde.backend.backend_name)

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
nu = 0.25
b = 1.0
M = 3e11
Lx = 1*m
Ly = 50*m

mu = 6. * milli * poise
mu_star = 6. * milli * poise
mu_bar = mu/mu_star

# 真实值 (用于生成观测数据)
k_true_val = 1e-12
k_star = 1e-12

E_true_val = 120.*MPa
E_star = 120.*MPa

# 定义可学习参数 (Variables)
k_bar = dde.Variable(2.0) # Initial guess
E_bar = dde.Variable(2.0) # Initial guess

Kdr = E_bar*E_star/(3*(1-2*nu))
Kdr_star = E_star/(3*(1-2*nu))


Lmbd = E_star*nu/(1+nu)/(1-2*nu)
Gshr = E_star/2/(1+nu)

B = b*M/(Kdr_star+M*(b**2))
nu_u = (3*nu+b*B*(1-2*nu))/(3-b*B*(1-2*nu))

p_star = (-1)*S_overburden
x_star = Ly
t_star = (b**2/Kdr_star + 1/M)*mu*(x_star**2)/(k_star)
nu_star = (1 - 2*nu)/(1 + nu)
u_star = p_star*x_star/Kdr_star
D_star = mu_star*(x_star**2)/Kdr_star/k_star/t_star
print(D_star)
print(Kdr_star)

# nondimensional domains originally used in your SciANN script:
td_0 = 0.0
td_f = 15000.0 / t_star  # same as SciANN
yd_min = 0.0
yd_max = (Ly / x_star)    # should be 1.0 in your nondim choice
NUM_SAMPLES = 20000

RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
    "terzaghi64_inverse_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
)
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULT_DIR}")

# 全部计算和保存完成后，自动用 Windows 默认图片查看器打开结果图
OPEN_FIGURES_AFTER_RUN = True


# -------------------- 辅助函数：计算解析解 (用于生成观测数据) --------------------
def get_analytical_solution(yd_arr, td_arr):
    """
    输入: 无量纲 yd, td (numpy array)
    输出: 无量纲 p (对应 p/p_star), 无量纲 v (对应 v/u_star)
    """
    # 1. 孔压 P (无量纲)
    p_true = np.zeros_like(yd_arr)
    # Series expansion
    for i in range(100):
        Ci = 4 * (-1) ** i / (np.pi * (2 * i + 1))
        Lisq = np.pi * (2 * i + 1) / 2
        Zz = np.cos(Lisq * yd_arr)
        Tt = np.exp(-Lisq ** 2 * td_arr)
        p_true += Ci * Zz * Tt

    # 2. 位移 V (物理量 -> 转无量纲)
    F2 = np.zeros_like(yd_arr)
    for i in range(100):
        Ci = 8 / ((np.pi * (2 * i + 1)) ** 2)
        Lisq = np.pi * (2 * i + 1) / 2
        Zz = np.cos(Lisq * (1 - yd_arr))
        Tt = 1 - np.exp(-(Lisq ** 2) * td_arr)
        F2 += Ci * Zz * Tt

    # 计算物理位移 (m)
    term1 = S_overburden * Ly * (1 - 2 * nu_u) * (yd_arr) / (2 * Gshr * (1 - nu_u))
    term2 = S_overburden * Ly * (nu_u - nu) / (2 * Gshr * (1 - nu_u) * (1 - nu))
    v_physical = term1 + term2 * F2

    # 转无量纲 v
    v_dimless = v_physical / u_star

    return p_true, v_dimless


def pde_system(X, Y):
    """
    X: (N,2) with columns [y, t]
    Y: (N,2) with columns [p, v]
    """
    y = X[:, 0:1]
    t = X[:, 1:2]

    p = Y[:, 0:1]
    v = Y[:, 1:2]

    # --- 【关键修改】在函数内部计算依赖参数 ---
    # 这样每次迭代都会建立新的计算图，避免 "Backward twice" 错误
    # 逻辑：Kdr = E_bar * ... -> Kdr_bar = Kdr / Kdr_star
    # 实际上 Kdr_bar 就等于 E_bar，但为了逻辑严谨，我们写完整：

    current_Kdr = E_bar * E_star / (3 * (1 - 2 * nu))
    current_Kdr_bar = current_Kdr / Kdr_star

    # 或者简单直接用 (因为数学上它是等价的):
    # current_Kdr_bar = E_bar

    # --- 导数计算 ---
    p_t = dde.grad.jacobian(Y, X, i=0, j=1)
    p_y = dde.grad.jacobian(Y, X, i=0, j=0)
    p_yy = dde.grad.hessian(Y, X, component=0, i=0, j=0)

    v_y = dde.grad.jacobian(Y, X, i=1, j=0)
    v_yy = dde.grad.hessian(Y, X, component=1, i=0, j=0)

    # 使用内部计算的 current_Kdr_bar 和全局的 k_bar
    Mp = (b ** 2 / current_Kdr_bar + Kdr_star / M) * D_star * p_t - k_bar * p_yy / mu_bar
    Msy = current_Kdr_bar * (1 + 2 * nu_star) * v_yy - b * p_y

    return [Mp, Msy]

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

# -------------------- 【新增】 生成观测数据 (Observation Data) --------------------
N_obs = 200 # 观测点数量

# 1. 孔压观测点: y = 25m (y_d = 0.5)
y_obs_p = np.full((N_obs, 1), 0.5)
t_obs_p = np.random.uniform(td_0 + eps_t, td_f, (N_obs, 1))
X_obs_p = np.hstack((y_obs_p, t_obs_p))

# 计算该位置的真实值 (Ground Truth)
p_target_obs, _ = get_analytical_solution(X_obs_p[:, 0:1], X_obs_p[:, 1:2])

# 2. 位移观测点: y = 50m (顶部, y_d = 1.0)
y_obs_v = np.full((N_obs, 1), 1.0)
t_obs_v = np.random.uniform(td_0 + eps_t, td_f, (N_obs, 1))
X_obs_v = np.hstack((y_obs_v, t_obs_v))

# 计算该位置的真实值 (Ground Truth)
_, v_target_obs = get_analytical_solution(X_obs_v[:, 0:1], X_obs_v[:, 1:2])
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
# 【新增】 观测数据约束
    dde.PointSetBC(X_obs_p, p_target_obs, component=0), # Fits p at y=25m
    dde.PointSetBC(X_obs_v, v_target_obs, component=1)  # Fits v at y=50m
]

# -------------------- 构建 TimePDE 数据集 --------------------
data = dde.data.TimePDE(
    geomtime,
    pde_system,
    bcs,
    num_domain=int(NUM_SAMPLES / 2),
    num_test=5000,
)

# -------------------- Neural network --------------------
# architecture: keep similar to your SciANN 4*[20]
layer_size = [2] + 4 * [64] + [2]  # 2 inputs, two outputs (p,v)
act = "tanh"
net = dde.nn.FNN(layer_size, act, "Glorot normal")

model = dde.Model(data, net)

# -------------------- Training with Callbacks --------------------
# 使用 VariableValue 回调来监控 k_bar 和 E_bar 的变化
fname_var = os.path.join(RESULT_DIR, "variables64.dat")
variable = dde.callbacks.VariableValue([k_bar, E_bar], period=100, filename=fname_var)

loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 100, 100]

# 1. Adam
# 注意: 这里的 external_trainable_variables 必须包含你的反演参数
model.compile(
    "adam",
    lr=1e-3,
    loss_weights=loss_weights,  # <--- 在这里传入权重
    external_trainable_variables=[k_bar, E_bar]
)

print(f"\n开始 Adam 训练 (Inverse Problem)...")
print(f"当前使用的损失权重: {loss_weights}")
print("(注: 最后两项 100 代表观测数据的权重，其余为 PDE 和 边界条件)")

# 训练
losshistory_adam, train_state_adam = model.train(iterations=20000, callbacks=[variable])

# 2. L-BFGS
model.compile("L-BFGS", external_trainable_variables=[k_bar, E_bar])
print("\n开始 L-BFGS 训练...")
losshistory_lbfgs, train_state_lbfgs = model.train(callbacks=[variable])

print("\n-------------------------------------------")
# 修改后 (PyTorch 正确写法)
print(f"Final Estimated k_bar: {k_bar.item():.5f} (Target: 1.0)")
print(f"Final Estimated E_bar: {E_bar.item():.5f} (Target: 1.0)")
print("-------------------------------------------\n")

# -------------------- 修正后的参数反演绘图代码 --------------------
import re


# 1. 重新定义读取函数 (防止格式报错)
def load_deepxde_variable_file(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        clean_data = []
        for line in lines:
            # 去掉 [, ] 和逗号，只留空格
            clean_line = line.replace('[', ' ').replace(']', ' ').replace(',', ' ')
            clean_data.append([float(x) for x in clean_line.split()])
        return np.array(clean_data)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return np.array([])


# 2. 读取数据
fname_var = os.path.join(RESULT_DIR, "variables64.dat")
data_history = load_deepxde_variable_file(fname_var)

if len(data_history) > 0:
    steps = data_history[:, 0]
    k_bar_history = data_history[:, 1]
    E_bar_history = data_history[:, 2]

    # 将无量纲反演参数换算为物理量
    k_pred_physical = k_bar_history * k_star
    k_target_physical = k_true_val
    Kdr_pred_mpa = E_bar_history * Kdr_star / MPa
    Kdr_target_mpa = E_true_val / (3 * (1 - 2 * nu)) / MPa

    parameter_rc = {
        'font.family': 'Times New Roman',
        'font.size': 19,
        'mathtext.fontset': 'stix',
        'axes.labelsize': 25,
        'pdf.fonttype': 42,
        'svg.fonttype': 'path',
    }
    with plt.rc_context(parameter_rc):
        fig_parameter, axes_parameter = plt.subplots(1, 2, figsize=(17, 6.8))

        axes_parameter[0].plot(
            steps, k_pred_physical, color='red', linewidth=2.6,
            label='Inversion')
        axes_parameter[0].axhline(
            k_target_physical, color='black', linestyle='--', linewidth=2.2,
            label='Target')
        axes_parameter[0].set_xlabel('epochs')
        axes_parameter[0].set_ylabel(r'$\kappa\;\mathrm{(m^2)}$')
        axes_parameter[0].ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0))

        axes_parameter[1].plot(
            steps, Kdr_pred_mpa, color='red', linewidth=2.6,
            label='Inversion')
        axes_parameter[1].axhline(
            Kdr_target_mpa, color='black', linestyle='--', linewidth=2.2,
            label='Target')
        axes_parameter[1].set_xlabel('epochs')
        axes_parameter[1].set_ylabel(r'$K_{dr}\;\mathrm{(MPa)}$')

        for axis in axes_parameter:
            axis.grid(True, color='#c8c8c8', linewidth=.8, alpha=.75)
            axis.legend(loc='upper right', frameon=False, fontsize=18)
            axis.set_xlim(steps.min(), steps.max())

        fig_parameter.subplots_adjust(
            left=.08, right=.98, bottom=.16, top=.96, wspace=.34)
        for extension in ('png', 'pdf', 'svg'):
            fig_parameter.savefig(
                os.path.join(
                    RESULT_DIR,
                    'Parameter_Inversion_Physical64.' + extension),
                dpi=300, facecolor='white')
        plt.close(fig_parameter)
else:
    print("未读取到参数历史数据，无法绘制参数反演图。")

# -------------------- Plotting (Same as original) --------------------
# (The plotting section remains largely the same, logic kept for validation)

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

# 为 3 行 × 2 列线图和云图准备物理量数据
p_pred_grid = pred['p'][:, 0].reshape(Ydtest.shape)
p_true_grid = pred['p_true_contour']
p_error_grid = np.abs(p_pred_grid - p_true_grid)
v_pred_grid = pred['v'][:, 0].reshape(Ydtest.shape)
v_true_grid = pred['v_true_contour']
v_error_grid = np.abs(v_pred_grid - v_true_grid)
y_plot = input_test[0] * x_star
t_plot = np.log10(np.maximum(input_test[1] * t_star, 1e-12))

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
    colors = ['#ff0000', '#ffa500', '#0000ff', '#008000',
              '#800080', '#00ffff', '#ffd700', '#000000']

    line_fields = [
        pred['v_true_line'], pred['p_true_line'],
        pred['v_line'], pred['p_line'],
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
        for time_index, color in enumerate(colors):
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
         for color in colors],
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
        v_true_grid, p_true_grid,
        v_pred_grid, p_pred_grid,
        np.abs(v_pred_grid - v_true_grid),
        np.abs(p_pred_grid - p_true_grid),
    ]

    positive_time_mask = Ttest[0, :] > 0
    source_log_time = np.log10(Ttest[0, positive_time_mask])
    uniform_log_time = np.linspace(
        source_log_time.min(), source_log_time.max(), Ttest.shape[1])
    log_t_grid, y_grid = np.meshgrid(uniform_log_time, Ytest[:, 0])
    for error_index in (4, 5):
        interpolator = RegularGridInterpolator(
            (Ytest[:, 0], source_log_time),
            grid_fields[error_index][:, positive_time_mask])
        regular_error = interpolator(
            np.stack([y_grid, log_t_grid], axis=-1))
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

    fig_contour, axes_contour = plt.subplots(3, 2, figsize=(17, 17.5))
    fig_contour.subplots_adjust(
        left=.14, right=.90, top=.95, bottom=.065, wspace=.38, hspace=.24)
    for panel_index, (axis, field, label) in enumerate(
            zip(axes_contour.flat, grid_fields, contour_labels)):
        if panel_index < 4:
            norm = displacement_norm if panel_index % 2 == 0 else pressure_norm
            plot_time, plot_y = safe_log_time, Ytest
        else:
            norm = Normalize(0, .00020 if panel_index == 4 else .08)
            plot_time, plot_y = log_t_grid, y_grid
        mesh = axis.pcolormesh(
            plot_time, plot_y, field, shading='nearest',
            cmap='jet', norm=norm, rasterized=True)
        axis.set(
            ylim=(0, 50), yticks=spatial_ticks,
            xlim=(uniform_log_time.min(), uniform_log_time.max()), title=label)
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

final_params = {
    "k_bar": float(k_bar.item()),
    "E_bar": float(E_bar.item()),
    "k_physical_m2": float(k_bar.item() * k_star),
    "Kdr_physical_pa": float(E_bar.item() * Kdr_star),
    "target_k_bar": 1.0,
    "target_E_bar": 1.0,
}
with open(os.path.join(RESULT_DIR, "final_parameters.json"), "w", encoding="utf-8") as f:
    json.dump(final_params, f, indent=2)

np.savez_compressed(
    os.path.join(RESULT_DIR, "terzaghi64_inverse_data.npz"),
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
    variable_history=data_history,
    X_obs_p=X_obs_p,
    p_target_obs=p_target_obs,
    X_obs_v=X_obs_v,
    v_target_obs=v_target_obs,
    adam_steps=np.asarray(losshistory_adam.steps),
    adam_loss_train=np.asarray(losshistory_adam.loss_train),
    adam_loss_test=np.asarray(losshistory_adam.loss_test),
    lbfgs_steps=np.asarray(losshistory_lbfgs.steps),
    lbfgs_loss_train=np.asarray(losshistory_lbfgs.loss_train),
    lbfgs_loss_test=np.asarray(losshistory_lbfgs.loss_test),
)
metadata = {
    "script": "terzaghi64-反.py",
    "seed": SEED,
    "NUM_SAMPLES": NUM_SAMPLES,
    "N_obs": N_obs,
    "network_layers": layer_size,
    "activation": act,
    "loss_weights": loss_weights,
    "td_0": td_0,
    "td_f": td_f,
    "yd_min": yd_min,
    "yd_max": yd_max,
    "x_star": x_star,
    "t_star": t_star,
    "p_star": p_star,
    "u_star": u_star,
    "D_star": D_star,
    "S_overburden": S_overburden,
    "mu": mu,
    "mu_star": mu_star,
    "k_true_val": k_true_val,
    "k_star": k_star,
    "E_true_val": E_true_val,
    "E_star": E_star,
    "Kdr_star": Kdr_star,
    "nu": nu,
    "b": b,
    "M": M,
    "Ly": Ly,
}
with open(os.path.join(RESULT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved figures and data to: {RESULT_DIR}")

# 训练与数据保存完成后再打开图片，避免绘图窗口阻塞计算过程
if OPEN_FIGURES_AFTER_RUN and hasattr(os, "startfile"):
    figure_files = [
        "Parameter_Inversion_Physical64.png",
        "terzaghi_line_error.png",
        "terzaghi_contour_error.png",
    ]
    for figure_name in figure_files:
        figure_path = os.path.join(RESULT_DIR, figure_name)
        if os.path.exists(figure_path):
            os.startfile(figure_path)



