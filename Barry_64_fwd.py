"""
Forward modeling of nonisothermal multiphase poromechanics using Physics-Informed Neural Networks
Coupled training version with single network for all field variables
Based on DeepXDE 1.14.0 with PyTorch backend and GPU support

Original paper: https://arxiv.org/abs/2209.03276
"""

import numpy as np
import deepxde as dde
import torch
import matplotlib.pyplot as plt
import time
import os

# Set backend to PyTorch
dde.config.set_default_float('float64')

# Set random seeds for reproducibility
np.random.seed(124)
torch.manual_seed(124)
if torch.cuda.is_available():
    torch.cuda.manual_seed(124)
    torch.cuda.manual_seed_all(124)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float64
NP_DTYPE = np.float64
GP_TORCH_DTYPE = torch.float32
GP_NP_DTYPE = np.float32

print(f"Using device: {device}")
print("Default precision: float64")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# ===================== Unit Conversions =====================
s, minute, hr, day = 1., 60., 60.**2, 24*60.**2
g, kg = 1.e-3, 1.
mm, cm, m, km = 1e-3, 1e-2, 1.0, 1e3
Pa, kPa, MPa, GPa = 1.0, 1.e3, 1.e6, 1.e9
micro, milli, centi, kilo, mega = 1e-6, 1e-3, 1e-2, 1e3, 1e6
poise = 0.1*Pa*s
Darcy = 9.869233e-13*m**2

# ===================== Physical Parameters =====================
rho_b = 2400.0 * kg/m**3
rho_star = 2400. * kg/m**3
rho_b_bar = rho_b/rho_star

mu = 1. * centi * poise
mu_star = 1. * centi * poise
mu_bar = mu/mu_star

k = 1e-10
k_star = 1e-10
k_bar_true = k/k_star

E = 46.7e5
E_star = 46.7e5
E_bar_true = E/E_star

nu = 0.167
b = 1.0
Lx = 1*m
Ly = 1*m

Kdr_star = E/(3*(1-2*nu))
Lame1 = E*nu/((1+nu)*(1-2*nu))
Lame2 = E/(2*(1+nu))

# ===================== Dimensionless Parameters =====================
p_star = Lame1 + 2*Lame2
x_star = Lx
t_star = mu_star*(x_star**2)/(k_star*Kdr_star)
u_star = p_star*x_star/Kdr_star
D_star = mu_star*(x_star**2)/Kdr_star/k_star/t_star

beta = ((Lame1 + 2*Lame2)*k_star)/(mu_star*Lx**2)
f_star = 2*beta*mu_star*(x_star**2)/(k_star*p_star)

x_source = 0.25
y_source = 0.25

td_0 = 0.0
t_star_Barry = (mu_star*(x_star**2)/(p_star*k_star))
td_f = 2*np.pi*t_star_Barry/t_star
xd_min = 0.0
xd_max = (Lx/x_star)
yd_min = 0.0
yd_max = (Ly/x_star)

# ===================== Training Parameters =====================
BATCH_SIZE = 256  # Reduced to 256 for better convergence and memory efficiency
EPOCHS = 20000  # Total epochs for coupled training
n_domain = 10000  # Number of collocation points
n_bc_edge = 1000  # Points per boundary edge
n_initial = 2000  # Points at t=0

initial_lr = 1e-3
decay_steps = 5000
ENABLE_RAR = False
GP_PERIOD = 1000
GP_SAMPLE_SIZE = 256
GP_ALPHA = 0.9

print(f"\nTrue k_bar: {k_bar_true:.6f}")
print(f"True E_bar: {E_bar_true:.6f}")

# ===================== Data Generation =====================
# Grid for collocation points
xx = np.array([0.0,0.05,0.1,0.15,0.175,0.2,0.225,0.250,0.275,0.3,0.325,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0])
yy = np.array([0.0,0.05,0.1,0.15,0.175,0.2,0.225,0.250,0.275,0.3,0.325,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0])
tn = 41
tt = np.linspace(td_0, td_f, tn)
x_data, y_data, t_data = [v.reshape(-1,1) for v in np.meshgrid(xx, yy, tt, indexing='ij')]

# Sensor locations
x_sensor_ids = np.where(np.logical_and(x_data.flatten() == xd_max/2,
                                       np.logical_or(y_data.flatten() == 0.1,
                                                     np.logical_or(y_data.flatten() == 0.4, y_data.flatten() == 0.8))))[0]
y_sensor_ids = np.where(np.logical_and(y_data.flatten() == yd_max/2,
                                       np.logical_or(x_data.flatten() == 0.1,
                                                     np.logical_or(x_data.flatten() == 0.4, x_data.flatten() == 0.8))))[0]

# Test points for evaluation
Nx, Ny, Nt = 41, 41, 101
xs = np.linspace(xd_min, xd_max, Nx)
ys = np.linspace(yd_min, yd_max, Ny)
ts = np.linspace(td_0, td_f, Nt)

Xdtest, Ydtest, Tdtest = np.meshgrid(xs, ys, ts, indexing='ij')
input_test_point1 = np.hstack([Xdtest[10,30,:].reshape(-1,1), Ydtest[10,30,:].reshape(-1,1), Tdtest[10,30,:].reshape(-1,1)]).astype(NP_DTYPE)
input_test_point2 = np.hstack([Xdtest[30,30,:].reshape(-1,1), Ydtest[30,30,:].reshape(-1,1), Tdtest[30,30,:].reshape(-1,1)]).astype(NP_DTYPE)
input_test_point3 = np.hstack([Xdtest[30,10,:].reshape(-1,1), Ydtest[30,10,:].reshape(-1,1), Tdtest[30,10,:].reshape(-1,1)]).astype(NP_DTYPE)

# ===================== Analytical Solution =====================
print("\nComputing analytical solution...")
p_true_sensor = np.zeros_like(t_data)
u_true_sensor = np.zeros_like(t_data)
v_true_sensor = np.zeros_like(t_data)

p_true_point1 = np.zeros((len(input_test_point1), 1))
u_true_point1 = np.zeros((len(input_test_point1), 1))
v_true_point1 = np.zeros((len(input_test_point1), 1))

p_true_point2 = np.zeros((len(input_test_point2), 1))
u_true_point2 = np.zeros((len(input_test_point2), 1))
v_true_point2 = np.zeros((len(input_test_point2), 1))

p_true_point3 = np.zeros((len(input_test_point3), 1))
u_true_point3 = np.zeros((len(input_test_point3), 1))
v_true_point3 = np.zeros((len(input_test_point3), 1))

x0 = 0.25
y0 = 0.25

td_Barry_sensor = t_data*t_star*beta
td_Barry_point1 = input_test_point1[:, 2:3]*t_star*beta
td_Barry_point2 = input_test_point2[:, 2:3]*t_star*beta
td_Barry_point3 = input_test_point3[:, 2:3]*t_star*beta

for n in range(1, 100):
    lambda_n = n*np.pi/Lx
    for q in range(1, 100):
        lambda_q = q*np.pi/Ly
        lambda_nq = lambda_n**2 + lambda_q**2

        # Sensor data
        ps_sensor = - (2.0*np.sin(lambda_n*x0)*np.sin(lambda_q*y0)/(lambda_nq**2+1))*(lambda_nq*np.sin(td_Barry_sensor) - np.cos(td_Barry_sensor) + np.exp(-lambda_nq * td_Barry_sensor))
        us_sensor = lambda_n * ps_sensor / lambda_nq
        vs_sensor = lambda_q * ps_sensor / lambda_nq
        p_true_sensor += ps_sensor * np.sin(lambda_n * x_data)*np.sin(lambda_q * y_data)
        u_true_sensor += us_sensor * np.cos(lambda_n * x_data)*np.sin(lambda_q * y_data)
        v_true_sensor += vs_sensor * np.sin(lambda_n * x_data)*np.cos(lambda_q * y_data)

        # Point 1
        ps_point1 = - (2.0*np.sin(lambda_n*x0)*np.sin(lambda_q*y0)/(lambda_nq**2+1))*(lambda_nq*np.sin(td_Barry_point1) - np.cos(td_Barry_point1) + np.exp(-lambda_nq*td_Barry_point1))
        us_point1 = lambda_n*ps_point1/lambda_nq
        vs_point1 = lambda_q*ps_point1/lambda_nq
        p_true_point1 += ps_point1*np.sin(lambda_n*input_test_point1[:, 0:1])*np.sin(lambda_q*input_test_point1[:, 1:2])
        u_true_point1 += us_point1*np.cos(lambda_n*input_test_point1[:, 0:1])*np.sin(lambda_q*input_test_point1[:, 1:2])
        v_true_point1 += vs_point1*np.sin(lambda_n*input_test_point1[:, 0:1])*np.cos(lambda_q*input_test_point1[:, 1:2])

        # Point 2
        ps_point2 = - (2.0*np.sin(lambda_n*x0)*np.sin(lambda_q*y0)/(lambda_nq**2+1))*(lambda_nq*np.sin(td_Barry_point2) - np.cos(td_Barry_point2) + np.exp(-lambda_nq*td_Barry_point2))
        us_point2 = lambda_n*ps_point2/lambda_nq
        vs_point2 = lambda_q*ps_point2/lambda_nq
        p_true_point2 += ps_point2*np.sin(lambda_n*input_test_point2[:, 0:1])*np.sin(lambda_q*input_test_point2[:, 1:2])
        u_true_point2 += us_point2*np.cos(lambda_n*input_test_point2[:, 0:1])*np.sin(lambda_q*input_test_point2[:, 1:2])
        v_true_point2 += vs_point2*np.sin(lambda_n*input_test_point2[:, 0:1])*np.cos(lambda_q*input_test_point2[:, 1:2])

        # Point 3
        ps_point3 = - (2.0*np.sin(lambda_n*x0)*np.sin(lambda_q*y0)/(lambda_nq**2+1))*(lambda_nq*np.sin(td_Barry_point3) - np.cos(td_Barry_point3) + np.exp(-lambda_nq*td_Barry_point3))
        us_point3 = lambda_n*ps_point3/lambda_nq
        vs_point3 = lambda_q*ps_point3/lambda_nq
        p_true_point3 += ps_point3*np.sin(lambda_n*input_test_point3[:, 0:1])*np.sin(lambda_q*input_test_point3[:, 1:2])
        u_true_point3 += us_point3*np.cos(lambda_n*input_test_point3[:, 0:1])*np.sin(lambda_q*input_test_point3[:, 1:2])
        v_true_point3 += vs_point3*np.sin(lambda_n*input_test_point3[:, 0:1])*np.cos(lambda_q*input_test_point3[:, 1:2])

# Scale solutions
p_true_sensor = -(Lame1 + 2*Lame2)*p_true_sensor*4/Lx/Ly
u_true_sensor = Lx*u_true_sensor*4/Lx/Ly
v_true_sensor = Ly*v_true_sensor*4/Lx/Ly

pd_true_sensor_Xprescribed = p_true_sensor[x_sensor_ids]/p_star
pd_true_sensor_Yprescribed = p_true_sensor[y_sensor_ids]/p_star
ud_true_sensor_Xprescribed = u_true_sensor[x_sensor_ids]/u_star
ud_true_sensor_Yprescribed = u_true_sensor[y_sensor_ids]/u_star
vd_true_sensor_Xprescribed = v_true_sensor[x_sensor_ids]/u_star
vd_true_sensor_Yprescribed = v_true_sensor[y_sensor_ids]/u_star

p_true_point1 = -(Lame1 + 2*Lame2)*p_true_point1*4/Lx/Ly/MPa
u_true_point1 = Lx*u_true_point1*4/Lx/Ly/m
v_true_point1 = Ly*v_true_point1*4/Lx/Ly/m

p_true_point2 = -(Lame1 + 2*Lame2)*p_true_point2*4/Lx/Ly/MPa
u_true_point2 = Lx*u_true_point2*4/Lx/Ly/m
v_true_point2 = Ly*v_true_point2*4/Lx/Ly/m

p_true_point3 = -(Lame1 + 2*Lame2)*p_true_point3*4/Lx/Ly/MPa
u_true_point3 = Lx*u_true_point3*4/Lx/Ly/m
v_true_point3 = Ly*v_true_point3*4/Lx/Ly/m

print("Analytical solution computed.")

# ===================== Coupled PDEs =====================
def injection_source(xd, yd, td):
    """Point source injection"""
    a = 0.04
    source = (1/(np.pi*a**2)) * torch.exp(-((xd-x_source)/a)**2) * torch.exp(-((yd-y_source)/a)**2)
    time_term = torch.sin(td*t_star/t_star_Barry)
    return source * time_term

def pde_coupled(X, outputs):
    """
    Coupled poromechanics PDEs for PyTorch backend
    outputs: [pd, ud, vd, Evd] - all field variables from single network
    Returns: [Mp, Msx, Msy, MEv] - all PDE residuals
    """
    xd = X[:, 0:1]
    yd = X[:, 1:2]
    td = X[:, 2:3]

    pd = outputs[:, 0:1]    # Pressure
    ud = outputs[:, 1:2]    # Displacement x
    vd = outputs[:, 2:3]    # Displacement y
    Evd = outputs[:, 3:4]   # Volumetric strain

    # ===== Compute derivatives using PyTorch autograd =====
    # Pressure derivatives
    pd_xd = dde.grad.jacobian(outputs, X, i=0, j=0)
    pd_yd = dde.grad.jacobian(outputs, X, i=0, j=1)
    pd_td = dde.grad.jacobian(outputs, X, i=0, j=2)
    pd_xd_xd = dde.grad.hessian(outputs, X, component=0, i=0, j=0)
    pd_yd_yd = dde.grad.hessian(outputs, X, component=0, i=1, j=1)

    # Displacement u derivatives
    ud_xd = dde.grad.jacobian(outputs, X, i=1, j=0)
    ud_yd = dde.grad.jacobian(outputs, X, i=1, j=1)
    ud_xd_xd = dde.grad.hessian(outputs, X, component=1, i=0, j=0)
    ud_yd_yd = dde.grad.hessian(outputs, X, component=1, i=1, j=1)

    # Displacement v derivatives
    vd_xd = dde.grad.jacobian(outputs, X, i=2, j=0)
    vd_yd = dde.grad.jacobian(outputs, X, i=2, j=1)
    vd_xd_xd = dde.grad.hessian(outputs, X, component=2, i=0, j=0)
    vd_yd_yd = dde.grad.hessian(outputs, X, component=2, i=1, j=1)

    # Volumetric strain derivatives
    Evd_xd = dde.grad.jacobian(outputs, X, i=3, j=0)
    Evd_yd = dde.grad.jacobian(outputs, X, i=3, j=1)
    Evd_td = dde.grad.jacobian(outputs, X, i=3, j=2)

    # Cross derivatives
    ud_xd_yd = dde.grad.hessian(outputs, X, component=1, i=0, j=1)
    vd_xd_yd = dde.grad.hessian(outputs, X, component=2, i=0, j=1)

    # ===== Material parameters =====
    # Use TRUE parameters for forward problem
    Kdr_bar = E_bar_true * E_star / (3*(1-2*nu)) / Kdr_star
    
    # ===== Compute Svd and its time derivative =====
    Svd = Kdr_bar * Evd - b * pd
    Svd_td = Kdr_bar * Evd_td - b * pd_td

    # ===== PDE 1: Fluid mass balance =====
    injc = injection_source(xd, yd, td)

    Mp = (b**2/Kdr_bar * D_star * pd_td +
          b/Kdr_bar * D_star * Svd_td -
          k_bar_true /mu_bar * (pd_xd_xd + pd_yd_yd) -
          f_star * injc)

    # ===== PDE 2: Momentum balance in x-direction =====
    Msx = (Kdr_bar * Evd_xd +
           Kdr_bar * (1/2) * (1-2*nu)/(1+nu) * (ud_xd_xd + vd_xd_yd) +
           Kdr_bar * (3/2) * (1-2*nu)/(1+nu) * (ud_xd_xd + ud_yd_yd) -
           b * pd_xd)

    # ===== PDE 3: Momentum balance in y-direction =====
    Msy = (Kdr_bar * Evd_yd +
           Kdr_bar * (1/2) * (1-2*nu)/(1+nu) * (ud_xd_yd + vd_yd_yd) +
           Kdr_bar * (3/2) * (1-2*nu)/(1+nu) * (vd_xd_xd + vd_yd_yd) -
           b * pd_yd)

    # ===== PDE 4: Volumetric strain definition =====
    MEv = ud_xd + vd_yd - Evd

    return [Mp, Msx, Msy, MEv]

# ===================== Boundary and Initial Conditions =====================
def boundary_initial(X, on_boundary):
    """Initial condition at t=0"""
    return np.isclose(X[2], td_0)

def boundary_xmin(X, on_boundary):
    return on_boundary and np.isclose(X[0], xd_min)

def boundary_xmax(X, on_boundary):
    return on_boundary and np.isclose(X[0], xd_max)

def boundary_ymin(X, on_boundary):
    return on_boundary and np.isclose(X[1], yd_min)

def boundary_ymax(X, on_boundary):
    return on_boundary and np.isclose(X[1], yd_max)

def boundary_all(X, on_boundary):
    """All spatial boundaries"""
    return on_boundary and (np.isclose(X[0], xd_min) or np.isclose(X[0], xd_max) or
                            np.isclose(X[1], yd_min) or np.isclose(X[1], yd_max))


class DecoupledNet(torch.nn.Module):
    """
    解耦的多头神经网络 (Decoupled Multi-head Network) - No Fourier Features
    输入(3) -> [分支1: p] -> pd
            -> [分支2: u] -> ud
            -> [分支3: v] -> vd
            -> [分支4: Ev] -> Evd
    """

    def __init__(self, layer_sizes, activation='tanh'):
        super().__init__()
        # layer_sizes 仅作为参考
        self.regularizer = None  # DeepXDE 需要此属性
        self.input_dim = 7  # x, y, t, sin(wt), cos(wt), sin(2wt), cos(2wt)
        hidden_dim = 128  # 提升网络宽度以增强表达能力
        depth = 3

        # 定义激活函数
        self.activation = torch.nn.Tanh()
        
        # === 创建4个独立的子网络 ===
        self.net_p = self._make_subnet(self.input_dim, hidden_dim, depth)
        self.net_u = self._make_subnet(self.input_dim, hidden_dim, depth)
        self.net_v = self._make_subnet(self.input_dim, hidden_dim, depth)
        self.net_Ev = self._make_subnet(self.input_dim, hidden_dim, depth)

    def _make_subnet(self, in_dim, hidden_dim, depth):
        layers = []
        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        layers.append(self.activation)
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        layers.append(torch.nn.Linear(hidden_dim, 1))  # 输出单变量
        return torch.nn.Sequential(*layers)

    def forward(self, inputs):
        # 双精度版本中统一强制转换为 float64，避免出现混合精度张量
        param = next(self.parameters())
        inputs = inputs.to(device=param.device, dtype=param.dtype)
        
        # 提取时间维度 td (第3列)
        td = inputs[:, 2:3]
        
        # 计算周期频率特征 (引入先验知识，破解谱偏差)
        omega = t_star / t_star_Barry
        
        sin_t = torch.sin(td * omega)
        cos_t = torch.cos(td * omega)
        sin_2t = torch.sin(2.0 * td * omega)
        cos_2t = torch.cos(2.0 * td * omega)
        
        # 将原始输入与周期特征拼接
        enhanced_inputs = torch.cat([inputs, sin_t, cos_t, sin_2t, cos_2t], dim=1)
        
        # 独立前向传播，使用增强后的特征
        pd = self.net_p(enhanced_inputs)
        ud = self.net_u(enhanced_inputs)
        vd = self.net_v(enhanced_inputs)
        Evd = self.net_Ev(enhanced_inputs)
        
        # 拼接输出 [pd, ud, vd, Evd]
        return torch.cat([pd, ud, vd, Evd], dim=1)


def build_shadow_state_dict(state_dict, dtype):
    """
    将主网络权重复制到影子网络，并统一转换为指定精度。
    """
    shadow_state = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            shadow_state[key] = value.detach().to(device=device, dtype=dtype)
        else:
            shadow_state[key] = value
    return shadow_state


# 创建网络
# 使用解耦网络 (4 subnets, 3x128 hidden)
net = DecoupledNet(layer_sizes=[3, 128, 128, 128, 4])

if torch.cuda.is_available():
    net = net.to(device)

print("\nNeural network created (DecoupledNet, No Fourier)")
print(f"Architecture: 4 independent subnets (Input 3 -> 3x128 -> Output 1)")

# ===================== Setup Geometry and Problem =====================
geom = dde.geometry.Rectangle([xd_min, yd_min], [xd_max, yd_max])
timedomain = dde.geometry.TimeDomain(td_0, td_f)

# === 自定义几何类以支持手动点 ===
class ManualGeometryXTime(dde.geometry.GeometryXTime):
    def __init__(self, geom, timedomain, ic_points, bc_points):
        super().__init__(geom, timedomain)
        self.ic_points = torch.tensor(ic_points, dtype=TORCH_DTYPE) if isinstance(ic_points, np.ndarray) else ic_points
        self.bc_points = torch.tensor(bc_points, dtype=TORCH_DTYPE) if isinstance(bc_points, np.ndarray) else bc_points
        # 转换回 numpy，因为 deepxde 内部几何采样通常返回 numpy
        if torch.is_tensor(self.ic_points): self.ic_points = self.ic_points.cpu().detach().numpy()
        if torch.is_tensor(self.bc_points): self.bc_points = self.bc_points.cpu().detach().numpy()

    def random_initial_points(self, n=0, random='pseudo', criteria=None):
        return self.ic_points

    def random_boundary_points(self, n=0, random='pseudo', criteria=None):
        return self.bc_points

# === 混合采样与手动点生成 (Mixed Sampling Strategy) ===
print("\nGenerating manual points for Domain, ICs and BCs...")

# 1. Domain Points (Anchors): 50% Uniform + 50% Gaussian around source
n_uni = n_domain // 2
n_gauss = n_domain - n_uni

# Uniform points
x_uni = np.random.uniform(xd_min, xd_max, n_uni)
y_uni = np.random.uniform(yd_min, yd_max, n_uni)
t_uni = np.random.uniform(td_0, td_f, n_uni)
X_domain_uni = np.stack((x_uni, y_uni, t_uni), axis=1)

# Gaussian points around source (0.25, 0.25)
# Using sigma=0.04 to match source width 'a'
sigma = 0.04
x_gauss = np.random.normal(0.25, sigma, n_gauss)
y_gauss = np.random.normal(0.25, sigma, n_gauss)
# Clip to domain boundaries to avoid invalid points
x_gauss = np.clip(x_gauss, xd_min, xd_max)
y_gauss = np.clip(y_gauss, yd_min, yd_max)
t_gauss = np.random.uniform(td_0, td_f, n_gauss)
X_domain_gauss = np.stack((x_gauss, y_gauss, t_gauss), axis=1)

# Combined Domain Anchors
X_anchors = np.vstack((X_domain_uni, X_domain_gauss)).astype(NP_DTYPE)
print(f"  Domain anchors: {len(X_anchors)} (Mixed: 50% Uniform + 50% Gaussian)")

# 2. Initial Points (t=0): Uniform spatial coverage
idx_ic = np.random.uniform(xd_min, xd_max, n_initial)
idy_ic = np.random.uniform(yd_min, yd_max, n_initial)
idt_ic = np.full(n_initial, td_0)
X_ic_manual = np.stack((idx_ic, idy_ic, idt_ic), axis=1).astype(NP_DTYPE)
print(f"  Initial points: {len(X_ic_manual)}")

# 3. Boundary Points: Fixed count per edge
# Boundary 1: x_min (Left)
b1_y = np.linspace(yd_min, yd_max, n_bc_edge)
b1_t = np.random.uniform(td_0, td_f, n_bc_edge)
b1_x = np.full(n_bc_edge, xd_min)
X_bc1 = np.stack((b1_x, b1_y, b1_t), axis=1)

# Boundary 2: x_max (Right)
b2_y = np.linspace(yd_min, yd_max, n_bc_edge)
b2_t = np.random.uniform(td_0, td_f, n_bc_edge)
b2_x = np.full(n_bc_edge, xd_max)
X_bc2 = np.stack((b2_x, b2_y, b2_t), axis=1)

# Boundary 3: y_min (Bottom)
b3_x = np.linspace(xd_min, xd_max, n_bc_edge)
b3_t = np.random.uniform(td_0, td_f, n_bc_edge)
b3_y = np.full(n_bc_edge, yd_min)
X_bc3 = np.stack((b3_x, b3_y, b3_t), axis=1)

# Boundary 4: y_max (Top)
b4_x = np.linspace(xd_min, xd_max, n_bc_edge)
b4_t = np.random.uniform(td_0, td_f, n_bc_edge)
b4_y = np.full(n_bc_edge, yd_max)
X_bc4 = np.stack((b4_x, b4_y, b4_t), axis=1)

# Combine all boundary points
X_bc_manual = np.vstack((X_bc1, X_bc2, X_bc3, X_bc4)).astype(NP_DTYPE)
print(f"  Boundary points: {len(X_bc_manual)} ({n_bc_edge} per edge)")

# 使用自定义几何
geomtime = ManualGeometryXTime(geom, timedomain, X_ic_manual, X_bc_manual)


# ===== Initial Conditions (at t=0) =====
ic_p = dde.icbc.IC(geomtime, lambda X: 0.0, boundary_initial, component=0)
ic_u = dde.icbc.IC(geomtime, lambda X: 0.0, boundary_initial, component=1)
ic_v = dde.icbc.IC(geomtime, lambda X: 0.0, boundary_initial, component=2)

# ===== Boundary Conditions for Pressure (p=0 on all boundaries) =====
bc_p = dde.icbc.DirichletBC(geomtime, lambda X: 0.0, boundary_all, component=0)

# ===== Boundary Conditions for Displacement u =====
# du/dx = 0 at x_min and x_max (strain is zero)
bc_u_xmin = dde.icbc.OperatorBC(geomtime,
                                 lambda X, outputs, _: dde.grad.jacobian(outputs, X, i=1, j=0),
                                 boundary_xmin)
bc_u_xmax = dde.icbc.OperatorBC(geomtime,
                                 lambda X, outputs, _: dde.grad.jacobian(outputs, X, i=1, j=0),
                                 boundary_xmax)
# u = 0 at y_min and y_max
bc_u_ymin = dde.icbc.DirichletBC(geomtime, lambda X: 0.0, boundary_ymin, component=1)
bc_u_ymax = dde.icbc.DirichletBC(geomtime, lambda X: 0.0, boundary_ymax, component=1)

# ===== Boundary Conditions for Displacement v =====
# v = 0 at x_min and x_max
bc_v_xmin = dde.icbc.DirichletBC(geomtime, lambda X: 0.0, boundary_xmin, component=2)
bc_v_xmax = dde.icbc.DirichletBC(geomtime, lambda X: 0.0, boundary_xmax, component=2)
# dv/dy = 0 at y_min and y_max (strain is zero)
bc_v_ymin = dde.icbc.OperatorBC(geomtime,
                                 lambda X, outputs, _: dde.grad.jacobian(outputs, X, i=2, j=1),
                                 boundary_ymin)
bc_v_ymax = dde.icbc.OperatorBC(geomtime,
                                 lambda X, outputs, _: dde.grad.jacobian(outputs, X, i=2, j=1),
                                 boundary_ymax)

# ===== Combine all boundary conditions (No Sensors for Forward Problem) =====
bcs = [
    ic_p, ic_u, ic_v,                           # Initial conditions
    bc_p,                                        # Pressure BCs
    bc_u_xmin, bc_u_xmax, bc_u_ymin, bc_u_ymax, # Displacement u BCs
    bc_v_xmin, bc_v_xmax, bc_v_ymin, bc_v_ymax, # Displacement v BCs
]

# ===== Create PDE problem =====
# We use 'anchors' to pass our mixed domain points.
# We set num_domain=0 to prevent DeepXDE from adding extra random points.
# We set num_boundary and num_initial to the counts of our manual points, 
# so DeepXDE will request exactly that many points from our ManualGeometryXTime.
data = dde.data.TimePDE(
    geomtime,
    pde_coupled,
    bcs,
    num_domain=0,  # We provide anchors
    num_boundary=len(X_bc_manual),
    num_initial=len(X_ic_manual),
    anchors=X_anchors,
)

# ===================== Create and Compile Model =====================
model = dde.Model(data, net)

# 自适应权重回调: Gradient Pathology (GP)
class GradientBalanceCallback(dde.callbacks.Callback):
    def __init__(self, period=GP_PERIOD, alpha=GP_ALPHA, sample_size=GP_SAMPLE_SIZE):
        super().__init__()
        self.period = period
        self.alpha = alpha
        self.sample_size = sample_size
        self.shadow_net = DecoupledNet(layer_sizes=[3, 128, 128, 128, 4]).to(device)
        self.shadow_net = self.shadow_net.to(dtype=GP_TORCH_DTYPE)

    def sync_shadow_net(self):
        """
        将主模型的 float64 权重同步到 float32 影子网络。
        """
        shadow_state = build_shadow_state_dict(self.model.net.state_dict(), GP_TORCH_DTYPE)
        self.shadow_net.load_state_dict(shadow_state)
        self.shadow_net.train()

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration
        if iteration % self.period == 0 and iteration > 500:
            try:
                self.update_weights()
            except RuntimeError as exc:
                print(f"\n[GP Warning] Iter {iteration}: GP 权重更新失败，已跳过。本次异常: {exc}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def update_weights(self):
        gp_start_time = time.time()
        iteration = self.model.train_state.iteration

        # 1. 采样一小批点来计算梯度 (节省显存)
        X_train = self.model.data.train_x
        if len(X_train) > self.sample_size:
            idx = np.random.choice(len(X_train), self.sample_size, replace=False)
            X_batch_np = X_train[idx].astype(GP_NP_DTYPE, copy=False)
        else:
            X_batch_np = X_train.astype(GP_NP_DTYPE, copy=False)

        print(
            f"\n[GP Start] Iter {iteration}: sample_size={len(X_batch_np)}, "
            f"dtype={GP_TORCH_DTYPE}, device={device}"
        )

        self.sync_shadow_net()

        X_batch = torch.tensor(X_batch_np, dtype=GP_TORCH_DTYPE, device=device)
        X_batch.requires_grad = True

        # 2. 使用影子网络进行 float32 前向传播
        outputs = self.shadow_net(X_batch)

        # 3. 计算 float32 残差
        residuals = pde_coupled(X_batch, outputs)
        pde_losses = [torch.mean(res ** 2) for res in residuals]

        # 4. 计算梯度范数
        grad_norms = []
        params_to_grad = [
            self.shadow_net.net_p[-1].weight,
            self.shadow_net.net_u[-1].weight,
            self.shadow_net.net_v[-1].weight,
            self.shadow_net.net_Ev[-1].weight,
        ]

        for loss in pde_losses:
            grads = torch.autograd.grad(loss, params_to_grad, retain_graph=True, allow_unused=True)
            total_norm = 0.0
            for g in grads:
                if g is not None:
                    total_norm += torch.norm(g).item()
            grad_norms.append(total_norm)
            
        grad_norms = np.array(grad_norms)
        
        # 5. 计算目标权重
        max_grad = np.max(grad_norms)
        target_weights = max_grad / (grad_norms + 1e-8)
        
        # 6. 更新权重 (混合策略: 只更新 PDE, 固定 BC)
        current_weights = self.model.loss_weights
        new_weights = np.array(current_weights)
        
        # 截断与平滑
        clipped_targets = np.clip(target_weights, 1.0, 1000.0)
        new_pde_weights = self.alpha * new_weights[0:4] + (1 - self.alpha) * clipped_targets

        new_weights[0:4] = new_pde_weights
        # new_weights[4:] 保持不变 (BCs固定)

        self.model.loss_weights = new_weights

        gp_elapsed = time.time() - gp_start_time
        print(f"[GP Done] Iter {iteration}: elapsed={gp_elapsed:.2f}s")
        print(
            f"  New Weights: Mp={new_weights[0]:.1f}, Msx={new_weights[1]:.1f}, "
            f"Msy={new_weights[2]:.1f}, MEv={new_weights[3]:.1f}"
        )

        del X_batch, outputs, residuals, pde_losses
        if device.type == "cuda":
            torch.cuda.empty_cache()

# RAR回调: 残差自适应加密 (Residual-based Adaptive Refinement)
class ResidualRefinementCallback(dde.callbacks.Callback):
    """
    RAR: 轻量级残差自适应加密，使用较小候选集并分块评估残差。
    """
    def __init__(self, period=4000, num_new_points=200, total_candidates=2000, chunk_size=500):
        super().__init__()
        self.period = period              # 每隔多少 epoch 执行一次加密
        self.num_new_points = num_new_points  # 每次新增多少高残差点
        self.total_candidates = total_candidates  # 候选采样点数量
        self.chunk_size = chunk_size      # 候选点评估分块大小
        self.added_count = 0              # 累计新增的点数量

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration
        # 前 5000 个 epoch 不执行，等待网络先收敛到一个初步状态
        if iteration > 5000 and iteration % self.period == 0:
            self.refine_points()

    def refine_points(self):
        """在时空域内寻找高残差点并加入训练集"""
        # 1. 在整个时空域内随机生成候选采样点
        x_cand = np.random.uniform(xd_min, xd_max, self.total_candidates)
        y_cand = np.random.uniform(yd_min, yd_max, self.total_candidates)
        t_cand = np.random.uniform(td_0, td_f, self.total_candidates)
        X_candidates = np.stack((x_cand, y_cand, t_cand), axis=1).astype(NP_DTYPE)

        # 2. 分块评估残差，避免一次性对大量候选点做高阶自动微分
        residual_chunks = []
        for start in range(0, self.total_candidates, self.chunk_size):
            end = min(start + self.chunk_size, self.total_candidates)
            X_chunk = torch.tensor(X_candidates[start:end], dtype=TORCH_DTYPE, device=device)
            X_chunk.requires_grad = True

            outputs = self.model.net(X_chunk)
            residuals = pde_coupled(X_chunk, outputs)

            chunk_residual = torch.zeros(end - start, device=device)
            for res in residuals:
                chunk_residual += torch.mean(res ** 2, dim=1).squeeze()
            residual_chunks.append(chunk_residual.detach().cpu())

        total_residual = torch.cat(residual_chunks, dim=0)

        # 3. 找出残差最大的前 N 个点
        top_k = min(self.num_new_points, len(total_residual))
        _, top_indices = torch.topk(total_residual, top_k)
        new_points = X_candidates[top_indices.cpu().numpy()]

        # 4. 将新点加入到训练 anchors
        # 获取当前 anchors
        current_anchors = self.model.data.train_x
        updated_anchors = np.vstack([current_anchors, new_points])
        
        # 更新 DeepXDE 数据集的 anchors
        self.model.data.train_x = updated_anchors
        self.model.data.num_domain = len(updated_anchors)
        
        self.added_count += len(new_points)
        iteration = self.model.train_state.iteration
        print(f"\n[RAR] Iter {iteration}: Added {len(new_points)} high-residual points "
              f"(Total added: {self.added_count})")
        print(f"  New training size: {len(updated_anchors)}")

# 初始权重配置
# PDE: [Mp, Msx, Msy, MEv] -> 初始设为 1.0
w_pde = [1.0, 1.0, 1.0, 1.0]
# BCs: 固定为 2000.0 (强硬约束，压倒 PDE 上限 1000)
w_bc = [2000.0] * 12
loss_weights = w_pde + w_bc

print("Loss Weights 配置 (正问题 - GP自适应):")
print(f"  PDEs: Adaptive (Init=1.0)")
print(f"  BCs: Fixed at 2000.0 (High Priority)")
print(f"  ENABLE_RAR = {ENABLE_RAR}")
print(f"  GP: period={GP_PERIOD}, sample_size={GP_SAMPLE_SIZE}, shadow_dtype=float32")

# 本次运行的统一时间戳和输出目录；后续图片、数据都放到同一个目录，方便改图。
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = f"barry64_forward_{RUN_TIMESTAMP}"
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
print(f"Run output directory: {RUN_OUTPUT_DIR}")

model.compile(
    "adam",
    lr=initial_lr,
    loss_weights=loss_weights
)

print("\nModel compiled successfully")
print(f"Optimizer: Adam")
print(f"Initial learning rate: {initial_lr}")

# ===================== Callbacks for Monitoring =====================
# For forward problem, we don't need ParameterCallback since parameters are fixed

# Model checkpoint callback
checkpoint_callback = dde.callbacks.ModelCheckpoint(
    "checkpoints/forward_model_best.pt",
    save_better_only=True,
    period=1000
)

# ===================== Training =====================
print("\n" + "="*70)
print("Starting Coupled Forward Training with PyTorch Backend")
print(f"Phase 1: Adam Optimizer (GP=ON, RAR={'ON' if ENABLE_RAR else 'OFF'})")
print("="*70 + "\n")

start_time = time.time()

callbacks = [GradientBalanceCallback(period=GP_PERIOD, alpha=GP_ALPHA, sample_size=GP_SAMPLE_SIZE)]
if ENABLE_RAR:
    callbacks.append(
        ResidualRefinementCallback(
            period=4000,
            num_new_points=200,
            total_candidates=2000,
            chunk_size=500,
        )
    )

# Train the model (Adam)
losshistory, train_state = model.train(
    iterations=EPOCHS,
    batch_size=BATCH_SIZE,
    display_every=500,
    callbacks=callbacks,
)

print("\n" + "="*70)
print("Phase 2: L-BFGS Optimizer (Fine-tuning)")
print("="*70 + "\n")

# L-BFGS Fine-tuning
# 继承最终权重
model.compile(
    "L-BFGS",
    loss_weights=model.loss_weights
)

losshistory_lbfgs, train_state_lbfgs = model.train(
    display_every=100,
)

training_time = time.time() - start_time

print("\n" + "="*70)
print("Training Complete!")
print(f"Total training time: {training_time:.1f}s ({training_time/60:.1f} min)")
print("="*70 + "\n")

# ===================== Save Training Data =====================
training_data_npz = os.path.join(RUN_OUTPUT_DIR, "barry64_forward_training_history_data.npz")
np.savez_compressed(
    training_data_npz,
    run_timestamp=RUN_TIMESTAMP,
    training_time_seconds=np.array(training_time),
    adam_iterations=np.array(EPOCHS),
    batch_size=np.array(BATCH_SIZE),
    n_domain=np.array(n_domain),
    n_bc_edge=np.array(n_bc_edge),
    n_initial=np.array(n_initial),
    initial_lr=np.array(initial_lr),
    loss_weights=np.asarray(loss_weights, dtype=np.float64),
    adam_steps=np.asarray(getattr(losshistory, "steps", [])),
    adam_loss_train=np.asarray(losshistory.loss_train),
    adam_loss_test=np.asarray(losshistory.loss_test),
    lbfgs_steps=np.asarray(getattr(losshistory_lbfgs, "steps", [])),
    lbfgs_loss_train=np.asarray(losshistory_lbfgs.loss_train),
    lbfgs_loss_test=np.asarray(losshistory_lbfgs.loss_test),
)
print(f"Training history data saved: {training_data_npz}")

# ===================== Save Model =====================
model.save(os.path.join(RUN_OUTPUT_DIR, "poromechanics_coupled_forward_pytorch"))
print("Model saved successfully")

# ===================== Analytical Solution Helper =====================
def compute_analytical_field(x_grid, y_grid, t_val):
    """
    Compute analytical solution for p, u, v on a grid at fixed time t_val.
    x_grid, y_grid: flattened arrays or meshgrid arrays of coordinates
    t_val: scalar time value (dimensionless or scaled? User passed t is usually dimensionless 0..td_f)
    Returns: p_true, u_true, v_true (same shape as x_grid)
    """
    # Create flat arrays for calculation
    x_flat = x_grid.flatten()[:, None]
    y_flat = y_grid.flatten()[:, None]
    t_flat = np.full_like(x_flat, t_val)
    
    p_true = np.zeros_like(x_flat)
    u_true = np.zeros_like(x_flat)
    v_true = np.zeros_like(x_flat)
    
    td_Barry = t_flat * t_star * beta
    x0 = 0.25
    y0 = 0.25
    
    # Series expansion
    for n in range(1, 50): # Reduced terms for speed in visualization, usually converges fast
        lambda_n = n * np.pi / Lx
        for q in range(1, 50):
            lambda_q = q * np.pi / Ly
            lambda_nq = lambda_n**2 + lambda_q**2
            
            # Common term
            ps_coeff = - (2.0 * np.sin(lambda_n*x0) * np.sin(lambda_q*y0) / (lambda_nq**2 + 1)) * \
                       (lambda_nq * np.sin(td_Barry) - np.cos(td_Barry) + np.exp(-lambda_nq * td_Barry))
            
            us_coeff = lambda_n * ps_coeff / lambda_nq
            vs_coeff = lambda_q * ps_coeff / lambda_nq
            
            # Superposition
            sin_nx = np.sin(lambda_n * x_flat)
            cos_nx = np.cos(lambda_n * x_flat)
            sin_qy = np.sin(lambda_q * y_flat)
            cos_qy = np.cos(lambda_q * y_flat)
            
            p_true += ps_coeff * sin_nx * sin_qy
            u_true += us_coeff * cos_nx * sin_qy
            v_true += vs_coeff * sin_nx * cos_qy
            
    # Scale to physical units
    p_true = -(Lame1 + 2*Lame2) * p_true * 4 / Lx / Ly / MPa
    u_true = Lx * u_true * 4 / Lx / Ly / m
    v_true = Ly * v_true * 4 / Lx / Ly / m
    
    return p_true.reshape(x_grid.shape), u_true.reshape(x_grid.shape), v_true.reshape(x_grid.shape)

# ===================== Post-Processing and Visualization =====================
print("\nGenerating visualizations...")


def build_timestamped_figure_name(base_name: str) -> str:
    """
    为当前运行生成带时间戳的图片文件名，避免覆盖和文件占用冲突。
    """
    stem, ext = os.path.splitext(base_name)
    return os.path.join(RUN_OUTPUT_DIR, f"{stem}_{RUN_TIMESTAMP}{ext}")


def save_figure_with_fallback(fig: plt.Figure, filename: str) -> str:
    """
    优先保存到目标文件；若文件被外部程序占用，则自动改名后重试。
    对 PNG 输出使用更高 dpi 以提升细节保真度。
    """
    _, ext = os.path.splitext(filename)
    dpi = 600 if ext.lower() == '.png' else 150
    try:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        return filename
    except PermissionError:
        stem, ext = os.path.splitext(filename)
        fallback_name = f"{stem}_retry_{int(time.time())}{ext}"
        print(f"  警告：{filename} 被占用，改存为 {fallback_name}")
        fig.savefig(fallback_name, dpi=dpi, bbox_inches='tight')
        return fallback_name

# ===== Plot 1: Training Loss =====
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Loss history
epochs_plot = np.arange(len(losshistory.loss_train))
ax.semilogy(epochs_plot, losshistory.loss_train, 'b-', linewidth=1.5, label='Train Loss')
if losshistory.loss_test is not None and len(losshistory.loss_test) > 0:
    ax.semilogy(epochs_plot, losshistory.loss_test, 'r--', linewidth=1.5, label='Test Loss')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss History', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
training_loss_png = build_timestamped_figure_name('forward_training_loss.png')
save_figure_with_fallback(fig, training_loss_png)
plt.close()
print(f"  ✓ Training loss plot saved: {training_loss_png}")

# ===== Plot 2: Field Snapshots (Cloud Maps) =====
print("\nGenerating field snapshots...")

def plot_field_snapshots(model, time_snapshots):
    """按每个相位输出 u、v、p 的 3×3 解析/预测/误差云图。"""
    nx, ny = 300, 300
    xs = np.linspace(xd_min, xd_max, nx)
    ys = np.linspace(yd_min, yd_max, ny)
    X_grid, Y_grid = np.meshgrid(xs, ys)
    x_plot = X_grid * x_star / m
    y_plot = Y_grid * x_star / m
    saved_phase_folders = []

    contour_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }

    for t_val in time_snapshots:
        cycle_fraction = t_val / td_f
        phase_pi = cycle_fraction * 2.0
        phase_folder = os.path.join(RUN_OUTPUT_DIR, f"phase_{phase_pi:.2f}pi_{RUN_TIMESTAMP}")
        os.makedirs(phase_folder, exist_ok=True)
        saved_phase_folders.append(phase_folder)
        print(f"  Processing snapshot t = {t_val:.4f} (phase = {phase_pi:.2f} pi)...")

        T_grid = np.full_like(X_grid, t_val)
        X_input = np.hstack(
            (X_grid.reshape(-1, 1), Y_grid.reshape(-1, 1), T_grid.reshape(-1, 1))
        ).astype(NP_DTYPE)

        p_true, u_true, v_true = compute_analytical_field(X_grid, Y_grid, t_val)
        y_pred = model.predict(X_input)
        p_pred = y_pred[:, 0].reshape(X_grid.shape) * p_star / MPa
        u_pred = y_pred[:, 1].reshape(X_grid.shape) * u_star / m
        v_pred = y_pred[:, 2].reshape(X_grid.shape) * u_star / m

        snapshot_data = {
            "run_timestamp": RUN_TIMESTAMP,
            "t_dimensionless": np.array(t_val),
            "t_physical": np.array(t_val * t_star / t_star_Barry),
            "phase_pi": np.array(phase_pi),
            "x": x_plot,
            "y": y_plot,
            "X": X_grid,
            "Y": Y_grid,
            "p_exact": p_true,
            "p_pred": p_pred,
            "p_error": np.abs(p_true - p_pred),
            "u_exact": u_true,
            "u_pred": u_pred,
            "u_error": np.abs(u_true - u_pred),
            "v_exact": v_true,
            "v_pred": v_pred,
            "v_error": np.abs(v_true - v_pred),
        }

        # 参考图列顺序为 u、v、p；行顺序为解析解、预测解和绝对误差。
        variables = [
            ("u", "m", u_true, u_pred),
            ("v", "m", v_true, v_pred),
            ("p", "MPa", p_true, p_pred),
        ]
        with plt.rc_context(contour_rc):
            fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.8))
            for col, (symbol, unit, exact, predicted) in enumerate(variables):
                error = np.abs(exact - predicted)
                field_min = min(float(np.min(exact)), float(np.min(predicted)))
                field_max = max(float(np.max(exact)), float(np.max(predicted)))
                error_max = max(float(np.max(error)), 1e-12)
                entries = [
                    (exact, rf"${symbol}$  ({unit})", "jet_r", field_min, field_max),
                    (predicted, rf"$\widehat{{{symbol}}}$  ({unit})", "jet_r", field_min, field_max),
                    (error, rf"$|{symbol}-\widehat{{{symbol}}}|$  ({unit})", "jet", 0.0, error_max),
                ]
                for row, (field, title, cmap, vmin, vmax) in enumerate(entries):
                    axis = axes[row, col]
                    mesh = axis.pcolormesh(
                        x_plot,
                        y_plot,
                        field,
                        shading="nearest",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        rasterized=True,
                        antialiased=False,
                        linewidth=0,
                    )
                    axis.set_title(title, pad=4)
                    axis.set_xlim(float(np.min(x_plot)), float(np.max(x_plot)))
                    axis.set_ylim(float(np.min(y_plot)), float(np.max(y_plot)))
                    axis.set_aspect("equal", adjustable="box")
                    if col == 0:
                        axis.set_ylabel(r"$y$  (m)")
                    if row == 2:
                        axis.set_xlabel(r"$x$  (m)")
                    axis.tick_params(direction="out", length=2.5, width=0.8)
                    colorbar = fig.colorbar(mesh, ax=axis, fraction=0.046, pad=0.045)
                    colorbar.ax.tick_params(labelsize=8, length=2.5)

            fig.subplots_adjust(
                left=0.060, right=0.965, bottom=0.060, top=0.965,
                wspace=0.32, hspace=0.28,
            )
            figure_stem = os.path.join(
                phase_folder,
                f"barry64_contour_3x3_phase_{phase_pi:.2f}pi_{RUN_TIMESTAMP}",
            )
            fig.savefig(f"{figure_stem}.png", dpi=600)
            fig.savefig(f"{figure_stem}.pdf", dpi=600)
            fig.savefig(f"{figure_stem}.svg")
            plt.close(fig)
            print(f"  Saved 3x3 contour figure: {figure_stem}.[png|pdf|svg]")

        snapshot_npz = os.path.join(phase_folder, "barry64_field_snapshot_data.npz")
        np.savez_compressed(snapshot_npz, **snapshot_data)
        print(f"  Saved snapshot data: {snapshot_npz}")

    return saved_phase_folders


# 一个周期的四个相位：0.5 pi、1.0 pi、1.5 pi 和 2.0 pi。
t_snapshots = [0.25 * td_f, 0.50 * td_f, 0.75 * td_f, 1.00 * td_f]
saved_phase_folders = plot_field_snapshots(model, t_snapshots)


# ===== Plot 3: Predictions at Test Points (Original Line Plots) =====
print("\nEvaluating model at test points (Line Plots)...")

# Predict at all test points
outputs_point1 = model.predict(input_test_point1)
outputs_point2 = model.predict(input_test_point2)
outputs_point3 = model.predict(input_test_point3)

# Extract field variables
pd_point1, ud_point1, vd_point1 = outputs_point1[:, 0:1], outputs_point1[:, 1:2], outputs_point1[:, 2:3]
pd_point2, ud_point2, vd_point2 = outputs_point2[:, 0:1], outputs_point2[:, 1:2], outputs_point2[:, 2:3]
pd_point3, ud_point3, vd_point3 = outputs_point3[:, 0:1], outputs_point3[:, 1:2], outputs_point3[:, 2:3]

# Convert to physical units
p_pred_point1 = pd_point1 * p_star / MPa
p_pred_point2 = pd_point2 * p_star / MPa
p_pred_point3 = pd_point3 * p_star / MPa

u_pred_point1 = ud_point1 * u_star / m
u_pred_point2 = ud_point2 * u_star / m
u_pred_point3 = ud_point3 * u_star / m

v_pred_point1 = vd_point1 * u_star / m
v_pred_point2 = vd_point2 * u_star / m
v_pred_point3 = vd_point3 * u_star / m

# Time for plotting
t_plot1 = input_test_point1[:, 2] * t_star / t_star_Barry
t_plot2 = input_test_point2[:, 2] * t_star / t_star_Barry
t_plot3 = input_test_point3[:, 2] * t_star / t_star_Barry

variables = ['p', 'u', 'v']
var_labels = ['Pressure (MPa)', 'Displacement u (m)', 'Displacement v (m)']
points_location = ['(0.25, 0.75)', '(0.75, 0.75)', '(0.75, 0.25)']
pred_data = [
    [p_pred_point1, p_pred_point2, p_pred_point3],
    [u_pred_point1, u_pred_point2, u_pred_point3],
    [v_pred_point1, v_pred_point2, v_pred_point3]
]

true_data = [
    [p_true_point1, p_true_point2, p_true_point3],
    [u_true_point1, u_true_point2, u_true_point3],
    [v_true_point1, v_true_point2, v_true_point3]
]

t_plots = [t_plot1, t_plot2, t_plot3]

line_data_npz = os.path.join(RUN_OUTPUT_DIR, "barry64_point_line_comparison_data.npz")
np.savez_compressed(
    line_data_npz,
    run_timestamp=RUN_TIMESTAMP,
    point_locations=np.asarray(points_location),
    t_point1=t_plot1,
    t_point2=t_plot2,
    t_point3=t_plot3,
    p_true_point1=p_true_point1,
    p_true_point2=p_true_point2,
    p_true_point3=p_true_point3,
    p_pred_point1=p_pred_point1,
    p_pred_point2=p_pred_point2,
    p_pred_point3=p_pred_point3,
    u_true_point1=u_true_point1,
    u_true_point2=u_true_point2,
    u_true_point3=u_true_point3,
    u_pred_point1=u_pred_point1,
    u_pred_point2=u_pred_point2,
    u_pred_point3=u_pred_point3,
    v_true_point1=v_true_point1,
    v_true_point2=v_true_point2,
    v_true_point3=v_true_point3,
    v_pred_point1=v_pred_point1,
    v_pred_point2=v_pred_point2,
    v_pred_point3=v_pred_point3,
)
print(f"Point line comparison data saved: {line_data_npz}")

pointwise_rc = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

with plt.rc_context(pointwise_rc):
    fig, axes = plt.subplots(3, 3, figsize=(15.2, 10.4), sharex=True)
    ylabels = (r"$p$  (MPa)", r"$u$  (m)", r"$v$  (m)")
    for row, variable in enumerate(variables):
        for col, (x_location, y_location) in enumerate(
            ((0.25, 0.75), (0.75, 0.75), (0.75, 0.25))
        ):
            axis = axes[row, col]
            time_values = np.asarray(t_plots[col]).reshape(-1)
            predicted = np.asarray(pred_data[row][col]).reshape(-1)
            analytical = np.asarray(true_data[row][col]).reshape(-1)
            axis.plot(
                time_values,
                predicted,
                linestyle="none",
                marker="*",
                color="red",
                markersize=8,
                markevery=5,
                label="PINN",
                zorder=3,
            )
            axis.plot(
                time_values,
                analytical,
                color="blue",
                linestyle="--",
                linewidth=2,
                label="Analytical",
                zorder=2,
            )
            axis.set_title(
                rf"${variable}$ at $(x,y)=({x_location:.2f},\ {y_location:.2f})$",
                pad=6,
            )
            if col == 0:
                axis.set_ylabel(ylabels[row])
            if row == 2:
                axis.set_xlabel("Dimensionless Time")
            axis.grid(True, color="#b0b0b0", alpha=0.35, linewidth=0.7)

            combined = np.concatenate((predicted, analytical))
            y_min, y_max = float(np.min(combined)), float(np.max(combined))
            upper_threshold = y_min + 0.70 * (y_max - y_min)
            right_start = int(0.65 * len(time_values))
            right_is_occupied = (
                np.any(predicted[right_start:] >= upper_threshold)
                or np.any(analytical[right_start:] >= upper_threshold)
            )
            axis.legend(loc="upper left" if right_is_occupied else "upper right", frameon=True)
            axis.tick_params(direction="out", length=4, width=0.9)

    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.075, top=0.965,
        wspace=0.20, hspace=0.22,
    )
    pointwise_figure_stem = os.path.join(
        RUN_OUTPUT_DIR, f"barry64_pointwise_3x3_{RUN_TIMESTAMP}"
    )
    fig.savefig(f"{pointwise_figure_stem}.png", dpi=600)
    fig.savefig(f"{pointwise_figure_stem}.pdf")
    fig.savefig(f"{pointwise_figure_stem}.svg")
    plt.close(fig)
    print(f"Pointwise 3x3 figure saved: {pointwise_figure_stem}.[png|pdf|svg]")


# ===== Compute and display errors =====
print("\n" + "="*70)
print("Prediction Errors at Test Points")
print("="*70)

for point_idx, point_name in enumerate(['Point 1 (0.25, 0.75)', 'Point 2 (0.75, 0.75)', 'Point 3 (0.75, 0.25)']):
    print(f"\n{point_name}:")

    p_pred = pred_data[0][point_idx]
    p_true = true_data[0][point_idx]
    u_pred = pred_data[1][point_idx]
    u_true = true_data[1][point_idx]
    v_pred = pred_data[2][point_idx]
    v_true = true_data[2][point_idx]

    # Compute relative L2 errors
    p_error = np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true) * 100
    u_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true) * 100
    v_error = np.linalg.norm(v_pred - v_true) / np.linalg.norm(v_true) * 100

    print(f"  Pressure (p) - Relative L2 error: {p_error:.2f}%")
    print(f"  Displacement (u) - Relative L2 error: {u_error:.2f}%")
    print(f"  Displacement (v) - Relative L2 error: {v_error:.2f}%")

print("\n" + "="*70)
print("All visualizations completed successfully!")
print("="*70 + "\n")

print("\nFiles generated:")
print(f"  - {training_loss_png}")
for folder_name in saved_phase_folders:
    print(f"  - {folder_name}/barry64_contour_3x3_phase_*.[png|pdf|svg]")
print(f"  - {pointwise_figure_stem}.[png|pdf|svg]")
print("  - poromechanics_coupled_forward_pytorch-*.pt (saved model)")
