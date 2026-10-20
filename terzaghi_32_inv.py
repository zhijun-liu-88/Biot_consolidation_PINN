# deepxde_terzaghi.py
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.lines as mlines

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
N_obs = 5000 # 观测点数量

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
fname_var = "variables.dat"
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
model.train(iterations=20000, callbacks=[variable])

# 2. L-BFGS
model.compile("L-BFGS", external_trainable_variables=[k_bar, E_bar])
print("\n开始 L-BFGS 训练...")
model.train(callbacks=[variable])

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
fname_var = "variables.dat"
data_history = load_deepxde_variable_file(fname_var)

if len(data_history) > 0:
    steps = data_history[:, 0]
    k_bar_history = data_history[:, 1]
    E_bar_history = data_history[:, 2]

    # 3. 物理量计算与准备
    # (1) K_dr 真实值与预测值 (Pa)
    # 根据你的定义: Kdr_star = E_star / (3*(1-2*nu))
    # 且 E_bar 实际上对应 Kdr_bar，即 Kdr_pred = E_bar * Kdr_star
    Kdr_star_val = E_star / (3 * (1 - 2 * nu))
    Kdr_true_val = E_true_val / (3 * (1 - 2 * nu))  # 真实物理值

    Kdr_pred_physical = E_bar_history * Kdr_star_val

    # (2) k 真实值与预测值 (m^2)
    # k_pred = k_bar * k_star
    k_true_physical = k_true_val
    k_pred_physical = k_bar_history * k_star

    # 4. 绘图 (完全仿照你的 Parameter_Plot_Itr 风格)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))

    # --- 左图: K_dr ---
    # 绘制红色预测线
    ax[0].plot(steps, Kdr_pred_physical, color='red', label='Predicted')
    # 绘制黑色虚线 (真实值)
    ax[0].hlines(y=Kdr_true_val, xmin=steps[0], xmax=steps[-1], color='black', linestyle='--', label='True')

    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('$K_{dr}~(Pa)$')

    # --- 右图: k ---
    # 绘制红色预测线
    ax[1].plot(steps, k_pred_physical, color='red', label='Predicted')
    # 绘制黑色虚线 (真实值)
    ax[1].hlines(y=k_true_physical, xmin=steps[0], xmax=steps[-1], color='black', linestyle='--', label='True')

    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('$k~(m^2)$')

    # --- 统一格式设置 ---
    for i in range(2):
        # 强制使用科学计数法 (因为数值很大或很小)
        ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # 你的参考图中对 x 轴并未强制科学计数，如果步数很多可以取消下面这行的注释
        # ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # 网格线
        ax[i].xaxis.grid(ls='--')

        # 如果需要显示图例，可以取消下面注释
        # ax[i].legend(loc='best', frameon=False, fontsize='small')

    # 调整布局
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92, wspace=0.3, hspace=0.25)

    # 保存并显示
    plt.savefig('Parameter_Inversion_Physical.pdf', dpi=150)
    print("反演参数物理量变化图已绘制完成。")
    plt.show()
else:
    print("未读取到数据，无法绘图。")
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

# 定义8种不同的标记形状，对应8个时间步
marker_list = ['o', 's', '^', 'D', 'v', '>', '<', 'p']
line_color = ['red', 'orange', 'blue', 'green', 'purple', 'cyan', 'gold', 'black']

# 设置画布：1行2列 (左图: P, 右图: V)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 遍历时间步进行绘图
for i in range(len(ts_line_plot)):
    # 格式化时间标签
    time_val = np.around(ts_line_plot[i] * t_star, decimals=2)
    time_label = f't = {time_val} s'

    # ==========================================
    # 左图：孔压 p (Pressure)
    # ==========================================
    # 1. PINN 预测值 -> 实线 (Line)
    ax[0].plot(ys_line_plot, pred['p_line'][:, i],
               color=line_color[i], linestyle='-', linewidth=1.5,
               label=time_label, zorder=1)

    # 2. 解析解真值 -> 散点 (Points)
    # linestyle='None': 不连线
    # markevery=15: 每隔15个点画一个符号，防止符号太密
    # markerfacecolor='white': 空心/白底符号，视觉上更清晰
    ax[0].plot(ys_line_plot, pred['p_true_line'][:, i],
               color=line_color[i], linestyle='None',
               marker=marker_list[i], markersize=3, markevery=15,
               markerfacecolor='white', markeredgewidth=1.5,
               zorder=2)

    # ==========================================
    # 右图：位移 v (Displacement)
    # ==========================================
    # 1. PINN 预测值 -> 实线
    ax[1].plot(ys_line_plot, pred['v_line'][:, i],
               color=line_color[i], linestyle='-', linewidth=1.5,
               zorder=1)

    # 2. 解析解真值 -> 散点
    ax[1].plot(ys_line_plot, pred['v_true_line'][:, i],
               color=line_color[i], linestyle='None',
               marker=marker_list[i], markersize=3, markevery=15,
               markerfacecolor='white', markeredgewidth=1.5,
               zorder=2)

# ------- 装饰与标签 -------

# 1. 设置左图 (Pressure) 属性
ax[0].set_xlabel('y (m)', fontsize=12)
ax[0].set_ylabel('Pore Pressure $p$ (MPa)', fontsize=12)
ax[0].set_title('Pore Pressure: PINN vs Analytical', fontsize=13)
ax[0].set_ylim(0.0, 0.11)
ax[0].grid(True, linestyle='--', alpha=0.6)

# 2. 设置右图 (Displacement) 属性
ax[1].set_xlabel('y (m)', fontsize=12)
ax[1].set_ylabel('Displacement $v$ (m)', fontsize=12)
ax[1].set_title('Displacement: PINN vs Analytical', fontsize=13)
ax[1].set_ylim(-0.031, 0.001)
ax[1].grid(True, linestyle='--', alpha=0.6)

# ------- 图例处理 (Legend) -------

# 图例1：右侧显示“时间-颜色”对应关系
# 我们只在左图(ax[0])生成了带label的plot，所以取左图的句柄
handles, labels = ax[0].get_legend_handles_labels()
# 过滤掉重复的label (如果有的话)，这里因为只plot了一次label，通常不需要过滤
legend1 = fig.legend(handles, labels,
                     loc='center right',
                     bbox_to_anchor=(0.99, 0.5),
                     title="Time Steps",
                     frameon=False)

# 图例2：顶部显示“线型-真值/预测”对应关系 (手动构建图例对象)
line_proxy = mlines.Line2D([], [], color='black', linestyle='-', linewidth=1.5, label='PINN (Prediction)')
point_proxy = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markerfacecolor='white', markeredgewidth=1.5, markersize=6, label='Exact (Analytical)')

legend2 = fig.legend(handles=[line_proxy, point_proxy],
                     loc='upper center',
                     bbox_to_anchor=(0.5, 0.98),
                     ncol=2,
                     frameon=False,
                     fontsize=11)

# 调整布局，为图例留出空间
plt.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.15, wspace=0.25)

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



