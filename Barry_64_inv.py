"""
Inverse modeling of nonisothermal multiphase poromechanics using Physics-Informed Neural Networks
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
print(f"Using device: {device}")
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
BATCH_SIZE = 256  # Reduced to improve optimization responsiveness
n_domain = 10000  # Number of collocation points
n_bc_edge = 800  # Points per boundary edge
n_initial = 1000  # Points at t=0

initial_lr = 1e-3
decay_steps = 5000

# ===================== Staged Training Parameters =====================
STAGE0_ITERS = 2000  # 只训练场变量，冻结 k
STAGE1_ITERS = 6000  # 强 k 识别：只解冻 k
STAGE2_ITERS = 4000  # 全耦合联合训练：只反演 k
EPOCHS = (
    STAGE0_ITERS
    + STAGE1_ITERS
    + STAGE2_ITERS
)

STAGE0_LR = 1e-3
STAGE1_LR = 5e-4
STAGE2_LR = 2e-4

ENABLE_STAGED_RAR = False

# ===================== Trainable Parameters =====================
def inverse_softplus_scalar(value: float) -> float:
    """给定正数目标值，返回 softplus 的逆映射，用于稳定初始化正参数。"""
    return float(np.log(np.exp(value) - 1.0))


def positive_parameter(raw_var: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    """用 softplus 保证待反演参数始终为正，避免出现非物理解。"""
    return torch.nn.functional.softplus(raw_var) + min_value


# 使用 raw variable + softplus 映射，避免 k 训练中穿过 0
k_bar_raw = dde.Variable(
    torch.tensor(
        [inverse_softplus_scalar(2.0)],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
)

E_BAR_FIXED = torch.tensor([E_bar_true], dtype=torch.float64, device=device)

print(f"\nTrue k_bar: {k_bar_true:.6f}")
print(f"Initial k_bar guess: {positive_parameter(k_bar_raw).item():.6f}")
print(f"Fixed E_bar: {float(E_BAR_FIXED.item()):.6f}")

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
input_test_point1 = np.hstack([Xdtest[10,30,:].reshape(-1,1), Ydtest[10,30,:].reshape(-1,1), Tdtest[10,30,:].reshape(-1,1)])  # x=0.25, y=0.75
input_test_point2 = np.hstack([Xdtest[30,30,:].reshape(-1,1), Ydtest[30,30,:].reshape(-1,1), Tdtest[30,30,:].reshape(-1,1)])  # x=0.75, y=0.75
input_test_point3 = np.hstack([Xdtest[30,10,:].reshape(-1,1), Ydtest[30,10,:].reshape(-1,1), Tdtest[30,10,:].reshape(-1,1)])  # x=0.75, y=0.25

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
    k_bar_value = positive_parameter(k_bar_raw)
    E_bar_value = E_BAR_FIXED
    Kdr_bar = E_bar_value * E_star / (3*(1-2*nu)) / Kdr_star

    # ===== Compute Svd and its time derivative =====
    Svd = Kdr_bar * Evd - b * pd
    Svd_td = Kdr_bar * Evd_td - b * pd_td

    # ===== PDE 1: Fluid mass balance =====
    injc = injection_source(xd, yd, td)

    Mp = (b**2/Kdr_bar * D_star * pd_td +
          b/Kdr_bar * D_star * Svd_td -
          k_bar_value /mu_bar * (pd_xd_xd + pd_yd_yd) -
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
    解耦多头网络，并显式注入时间相位与耗散特征，降低反演中的时间谱偏差。
    """

    def __init__(self, layer_sizes, activation='tanh'):
        super().__init__()
        self.regularizer = None
        self.input_dim = 10  # x, y, t + sin/cos 周期项 + 衰减记忆项
        hidden_dim = 128
        depth = 3

        self.activation = torch.nn.Tanh()
        self.net_p = self._make_subnet(self.input_dim, hidden_dim, depth)
        self.net_u = self._make_subnet(self.input_dim, hidden_dim, depth)
        self.net_v = self._make_subnet(self.input_dim, hidden_dim, depth)
        self.net_Ev = self._make_subnet(self.input_dim, hidden_dim, depth)

    def _make_subnet(self, in_dim, hidden_dim, depth):
        layers = [torch.nn.Linear(in_dim, hidden_dim), self.activation]
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        layers.append(torch.nn.Linear(hidden_dim, 1))
        return torch.nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = inputs.to(torch.float64)
        td = inputs[:, 2:3]
        tau = td / td_f
        omega = t_star / t_star_Barry

        sin_t = torch.sin(td * omega)
        cos_t = torch.cos(td * omega)
        sin_2t = torch.sin(2.0 * td * omega)
        cos_2t = torch.cos(2.0 * td * omega)
        tau_sq = tau ** 2
        exp_t = torch.exp(-tau)
        exp_4t = torch.exp(-4.0 * tau)

        enhanced_inputs = torch.cat(
            [inputs, sin_t, cos_t, sin_2t, cos_2t, tau_sq, exp_t, exp_4t],
            dim=1,
        )

        pd = self.net_p(enhanced_inputs)
        ud = self.net_u(enhanced_inputs)
        vd = self.net_v(enhanced_inputs)
        Evd = self.net_Ev(enhanced_inputs)
        return torch.cat([pd, ud, vd, Evd], dim=1)


# 创建网络
net = DecoupledNet(layer_sizes=[3, 128, 128, 128, 4])
if torch.cuda.is_available():
    net = net.to(device)

print("\nNeural network created (DecoupledNet, Phase-Enhanced)")
print("Architecture: 4 independent subnets (Input 10 -> 3x128 -> Output 1)")
print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")

# ===================== Setup Geometry and Problem =====================
geom = dde.geometry.Rectangle([xd_min, yd_min], [xd_max, yd_max])
timedomain = dde.geometry.TimeDomain(td_0, td_f)

# === 自定义几何类以支持手动点 ===
class ManualGeometryXTime(dde.geometry.GeometryXTime):
    def __init__(self, geom, timedomain, ic_points, bc_points):
        super().__init__(geom, timedomain)
        self.ic_points = torch.tensor(ic_points, dtype=torch.float64) if isinstance(ic_points, np.ndarray) else ic_points
        self.bc_points = torch.tensor(bc_points, dtype=torch.float64) if isinstance(bc_points, np.ndarray) else bc_points
        # 转换回 numpy，因为 deepxde 内部几何采样通常返回 numpy
        if torch.is_tensor(self.ic_points): self.ic_points = self.ic_points.cpu().detach().numpy()
        if torch.is_tensor(self.bc_points): self.bc_points = self.bc_points.cpu().detach().numpy()

    def random_initial_points(self, n=0, random='pseudo', criteria=None):
        return self.ic_points

    def random_boundary_points(self, n=0, random='pseudo', criteria=None):
        return self.bc_points


def sample_time_phase_balanced(
    total_points: int,
    time_min: float,
    time_max: float,
    n_bins: int,
) -> np.ndarray:
    """
    在整个时间区间上做分桶均衡采样，避免弱相位在反演中长期失声。
    """
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)
    base_count = total_points // n_bins
    remainder = total_points % n_bins
    time_chunks = []

    for idx in range(n_bins):
        current_count = base_count + (1 if idx < remainder else 0)
        if current_count == 0:
            continue
        t_left = bin_edges[idx]
        t_right = bin_edges[idx + 1]
        time_chunks.append(
            np.random.uniform(t_left, t_right, current_count).astype(np.float64)
        )

    return np.concatenate(time_chunks) if time_chunks else np.empty((0,), dtype=np.float64)

# === 混合采样与手动点生成 (Mixed Sampling Strategy) ===
print("\nGenerating manual points for Domain, ICs and BCs...")

N_PHASE_BINS = 8

# 1. Domain Points (Anchors): 40% Uniform + 20% Gaussian + 40% Phase-balanced
n_uni = int(0.4 * n_domain)
n_gauss = int(0.2 * n_domain)
n_phase = n_domain - n_uni - n_gauss

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

# Phase-balanced points over the full cycle
x_phase = np.random.uniform(xd_min, xd_max, n_phase)
y_phase = np.random.uniform(yd_min, yd_max, n_phase)
t_phase = sample_time_phase_balanced(n_phase, td_0, td_f, N_PHASE_BINS)
X_domain_phase = np.stack((x_phase, y_phase, t_phase), axis=1)

# Combined Domain Anchors
X_anchors = np.vstack((X_domain_uni, X_domain_gauss, X_domain_phase)).astype(np.float64)
print(
    f"  Domain anchors: {len(X_anchors)} "
    "(Mixed: 40% Uniform + 20% Gaussian + 40% Phase-balanced)"
)
print(f"  Phase-balanced points: {n_phase} (bins: {N_PHASE_BINS})")

# 2. Initial Points (t=0): Uniform spatial coverage
idx_ic = np.random.uniform(xd_min, xd_max, n_initial)
idy_ic = np.random.uniform(yd_min, yd_max, n_initial)
idt_ic = np.full(n_initial, td_0)
X_ic_manual = np.stack((idx_ic, idy_ic, idt_ic), axis=1).astype(np.float64)
print(f"  Initial points: {len(X_ic_manual)}")

# 3. Boundary Points: Fixed count per edge
# Boundary 1: x_min (Left)
b1_y = np.linspace(yd_min, yd_max, n_bc_edge)
b1_t = sample_time_phase_balanced(n_bc_edge, td_0, td_f, N_PHASE_BINS)
b1_x = np.full(n_bc_edge, xd_min)
X_bc1 = np.stack((b1_x, b1_y, b1_t), axis=1)

# Boundary 2: x_max (Right)
b2_y = np.linspace(yd_min, yd_max, n_bc_edge)
b2_t = sample_time_phase_balanced(n_bc_edge, td_0, td_f, N_PHASE_BINS)
b2_x = np.full(n_bc_edge, xd_max)
X_bc2 = np.stack((b2_x, b2_y, b2_t), axis=1)

# Boundary 3: y_min (Bottom)
b3_x = np.linspace(xd_min, xd_max, n_bc_edge)
b3_t = sample_time_phase_balanced(n_bc_edge, td_0, td_f, N_PHASE_BINS)
b3_y = np.full(n_bc_edge, yd_min)
X_bc3 = np.stack((b3_x, b3_y, b3_t), axis=1)

# Boundary 4: y_max (Top)
b4_x = np.linspace(xd_min, xd_max, n_bc_edge)
b4_t = sample_time_phase_balanced(n_bc_edge, td_0, td_f, N_PHASE_BINS)
b4_y = np.full(n_bc_edge, yd_max)
X_bc4 = np.stack((b4_x, b4_y, b4_t), axis=1)

# Combine all boundary points
X_bc_manual = np.vstack((X_bc1, X_bc2, X_bc3, X_bc4)).astype(np.float64)
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

# ===== Observation Data (Sensors) =====
# Create observation points
observe_x = np.hstack([x_data[x_sensor_ids], y_data[x_sensor_ids], t_data[x_sensor_ids]]).astype(np.float64)
observe_y = np.hstack([x_data[y_sensor_ids], y_data[y_sensor_ids], t_data[y_sensor_ids]]).astype(np.float64)

# Pressure observations
observe_p_x = dde.icbc.PointSetBC(observe_x, pd_true_sensor_Xprescribed.astype(np.float64), component=0)
observe_p_y = dde.icbc.PointSetBC(observe_y, pd_true_sensor_Yprescribed.astype(np.float64), component=0)

# Displacement u observations
observe_u_x = dde.icbc.PointSetBC(observe_x, ud_true_sensor_Xprescribed.astype(np.float64), component=1)
observe_u_y = dde.icbc.PointSetBC(observe_y, ud_true_sensor_Yprescribed.astype(np.float64), component=1)

# Displacement v observations
observe_v_x = dde.icbc.PointSetBC(observe_x, vd_true_sensor_Xprescribed.astype(np.float64), component=2)
observe_v_y = dde.icbc.PointSetBC(observe_y, vd_true_sensor_Yprescribed.astype(np.float64), component=2)

# ===== Combine all boundary conditions =====
bcs = [
    ic_p, ic_u, ic_v,                           # Initial conditions
    bc_p,                                        # Pressure BCs
    bc_u_xmin, bc_u_xmax, bc_u_ymin, bc_u_ymax, # Displacement u BCs
    bc_v_xmin, bc_v_xmax, bc_v_ymin, bc_v_ymax, # Displacement v BCs
    observe_p_x, observe_p_y,                    # Pressure observations
    observe_u_x, observe_u_y,                    # Displacement u observations
    observe_v_x, observe_v_y                     # Displacement v observations
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
    num_test=2000,
    train_distribution='uniform'
)

# ===================== Create and Compile Model =====================
model = dde.Model(data, net)
# ===================== Staged Training Helpers =====================
def get_stage_name(stage: int) -> str:
    stage_names = {
        0: "阶段 0：场变量预热（冻结 k）",
        1: "阶段 1：强 k 识别（只解冻 k）",
        2: "阶段 2：全耦合联合训练（只反演 k）",
    }
    return stage_names[stage]


def get_stage_iterations(stage: int) -> int:
    stage_iters = {
        0: STAGE0_ITERS,
        1: STAGE1_ITERS,
        2: STAGE2_ITERS,
    }
    return stage_iters[stage]


def get_stage_lr(stage: int) -> float:
    stage_lrs = {
        0: STAGE0_LR,
        1: STAGE1_LR,
        2: STAGE2_LR,
    }
    return stage_lrs[stage]


def get_stage_optimizer(stage: int) -> str:
    return "adam"


def get_stage_trainables(stage: int) -> list[dde.Variable]:
    if stage == 0:
        return []
    return [k_bar_raw]


def get_stage_loss_weights(stage: int) -> list[float]:
    if stage == 0:
        w_pde = [1.0, 1.0, 1.0, 1.0]
        w_bc = [50.0] * 12
        w_data = [50.0] * 6
    elif stage == 1:
        w_pde = [20.0, 1.0, 1.0, 1.0]
        w_bc = [50.0] * 12
        w_data = [150.0, 150.0, 20.0, 20.0, 20.0, 20.0]
    else:
        w_pde = [5.0, 5.0, 5.0, 2.0]
        w_bc = [50.0] * 12
        w_data = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
    return w_pde + w_bc + w_data


def stage_uses_gp(stage: int) -> bool:
    return stage == 2


def set_network_trainable_mode(stage: int) -> None:
    """
    单参数版本当前所有阶段均保持网络全可训练。
    """
    for param in model.net.parameters():
        param.requires_grad = True

# ===================== Callbacks for Monitoring =====================
parameter_history = {
    'k_bar': [],
    'epoch': [],
    'loss': [],
    'stage': [],
}

training_state_info = {
    'stage': '未开始',
}


class GradientBalanceCallback(dde.callbacks.Callback):
    """仅对 PDE 项做 GP 自适应，避免反问题中某个方程长期失声。"""

    def __init__(self, period=100, alpha=0.9, power=0.5, min_weight=0.5, max_weight=20.0):
        super().__init__()
        self.period = period
        self.alpha = alpha
        self.power = power
        self.min_weight = min_weight
        self.max_weight = max_weight

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration
        if iteration % self.period == 0 and iteration > 500:
            self.update_weights()

    def update_weights(self):
        X_train = self.model.data.train_x
        if len(X_train) > 2048:
            idx = np.random.choice(len(X_train), 2048, replace=False)
            X_batch = torch.tensor(X_train[idx], dtype=torch.float64, device=device)
        else:
            X_batch = torch.tensor(X_train, dtype=torch.float64, device=device)

        X_batch.requires_grad = True
        outputs = self.model.net(X_batch)
        residuals = pde_coupled(X_batch, outputs)
        pde_losses = [torch.mean(res ** 2) for res in residuals]

        params_to_grad = [
            self.model.net.net_p[-1].weight,
            self.model.net.net_u[-1].weight,
            self.model.net.net_v[-1].weight,
            self.model.net.net_Ev[-1].weight,
        ]

        grad_norms = []
        for loss in pde_losses:
            grads = torch.autograd.grad(loss, params_to_grad, retain_graph=True, allow_unused=True)
            total_norm = 0.0
            for grad in grads:
                if grad is not None:
                    total_norm += torch.norm(grad).item()
            grad_norms.append(total_norm)

        grad_norms = np.array(grad_norms)
        ref_grad = np.median(grad_norms)
        target_weights = (ref_grad / (grad_norms + 1e-8)) ** self.power

        current_weights = np.array(self.model.loss_weights)
        clipped_targets = np.clip(target_weights, self.min_weight, self.max_weight)
        new_pde_weights = self.alpha * current_weights[0:4] + (1 - self.alpha) * clipped_targets
        current_weights[0:4] = new_pde_weights
        self.model.loss_weights = current_weights

        iteration = self.model.train_state.iteration
        if iteration % 1000 == 0:
            print(f"\n[GP Weights] Iter {iteration}:")
            print(
                f"  Grad Norms: Mp={grad_norms[0]:.2e}, Msx={grad_norms[1]:.2e}, "
                f"Msy={grad_norms[2]:.2e}, MEv={grad_norms[3]:.2e}"
            )
            print(
                f"  New PDE Weights: Mp={current_weights[0]:.2f}, Msx={current_weights[1]:.2f}, "
                f"Msy={current_weights[2]:.2f}, MEv={current_weights[3]:.2f}"
            )


class ResidualRefinementCallback(dde.callbacks.Callback):
    """按时间相位分桶执行 RAR，避免新增点再次偏向大值相位。"""

    def __init__(self, period=2000, num_new_points=500, total_candidates=10000, n_phase_bins=8):
        super().__init__()
        self.period = period
        self.num_new_points = num_new_points
        self.total_candidates = total_candidates
        self.n_phase_bins = n_phase_bins
        self.added_count = 0

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration
        if iteration > 5000 and iteration % self.period == 0:
            self.refine_points()

    def refine_points(self):
        x_cand = np.random.uniform(xd_min, xd_max, self.total_candidates)
        y_cand = np.random.uniform(yd_min, yd_max, self.total_candidates)
        t_cand = sample_time_phase_balanced(
            self.total_candidates, td_0, td_f, self.n_phase_bins
        )
        X_candidates = np.stack((x_cand, y_cand, t_cand), axis=1).astype(np.float64)

        X_tensor = torch.tensor(X_candidates, dtype=torch.float64, device=device)
        X_tensor.requires_grad = True
        outputs = self.model.net(X_tensor)
        residuals = pde_coupled(X_tensor, outputs)

        total_residual = torch.zeros(self.total_candidates, device=device)
        for res in residuals:
            total_residual += torch.mean(res ** 2, dim=1).squeeze()

        bin_edges = np.linspace(td_0, td_f, self.n_phase_bins + 1)
        base_count = self.num_new_points // self.n_phase_bins
        remainder = self.num_new_points % self.n_phase_bins
        selected_indices = []
        t_candidates = X_candidates[:, 2]

        for idx in range(self.n_phase_bins):
            current_quota = base_count + (1 if idx < remainder else 0)
            if current_quota == 0:
                continue

            if idx == self.n_phase_bins - 1:
                mask = (t_candidates >= bin_edges[idx]) & (t_candidates <= bin_edges[idx + 1])
            else:
                mask = (t_candidates >= bin_edges[idx]) & (t_candidates < bin_edges[idx + 1])

            bucket_indices = np.where(mask)[0]
            if len(bucket_indices) == 0:
                continue

            bucket_scores = total_residual[bucket_indices]
            top_k = min(current_quota, len(bucket_indices))
            _, local_top = torch.topk(bucket_scores, top_k)
            selected_indices.append(bucket_indices[local_top.cpu().numpy()])

        if not selected_indices:
            print("\n[RAR] 跳过本轮加密：未选出有效相位候选点。")
            return

        chosen_indices = np.concatenate(selected_indices)
        new_points = X_candidates[chosen_indices]

        current_anchors = self.model.data.train_x
        updated_anchors = np.vstack([current_anchors, new_points])
        self.model.data.train_x = updated_anchors
        self.model.data.num_domain = len(updated_anchors)

        self.added_count += len(new_points)
        iteration = self.model.train_state.iteration
        print(
            f"\n[RAR] Iter {iteration}: Added {len(new_points)} phase-balanced high-residual points "
            f"(Total added: {self.added_count})"
        )
        print(f"  New training size: {len(updated_anchors)}")

class ParameterCallback(dde.callbacks.Callback):
    """Callback to track parameter evolution"""

    def __init__(self, period=100):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        # 顺便修复 DeprecationWarning，将 .epoch 改为 .iteration
        iteration = self.model.train_state.iteration
        if iteration % self.period == 0:
            k_val = positive_parameter(k_bar_raw).item()

            # 核心修改：将 loss 数组求和得到总 Loss 标量
            loss_list = self.model.train_state.loss_train
            total_loss = np.sum(loss_list)

            parameter_history['k_bar'].append(k_val)
            parameter_history['epoch'].append(iteration)
            parameter_history['loss'].append(total_loss)  # 存入总 Loss
            parameter_history['stage'].append(training_state_info['stage'])

            k_error = abs(k_val - k_bar_true) / k_bar_true * 100

            print(f"\nIteration {iteration}:")
            print(f"  Stage: {training_state_info['stage']}")
            print(f"  Total Loss: {total_loss:.6e}")
            print(f"  k_bar = {k_val:.6f} (true: {k_bar_true:.6f}, error: {k_error:.2f}%)")

# Model checkpoint callback
checkpoint_callback = dde.callbacks.ModelCheckpoint(
    "checkpoints/model_best.pt",
    save_better_only=True,
    period=1000
)


def train_stage(stage: int):
    stage_name = get_stage_name(stage)
    stage_iters = get_stage_iterations(stage)
    stage_lr = get_stage_lr(stage)
    stage_optimizer = get_stage_optimizer(stage)
    stage_loss_weights = get_stage_loss_weights(stage)
    stage_trainables = get_stage_trainables(stage)
    use_gp = stage_uses_gp(stage)

    training_state_info['stage'] = stage_name

    print("\n" + "=" * 70)
    print(stage_name)
    print("=" * 70)
    print(f"  Iterations: {stage_iters}")
    print(f"  Optimizer: {stage_optimizer}")
    if stage_optimizer == "adam":
        print(f"  Learning rate: {stage_lr:.1e}")
    print(f"  Trainables: {len(stage_trainables)} parameter(s)")
    print(f"  GP: {'ON' if use_gp else 'OFF'}")
    print(f"  RAR: {'ON' if ENABLE_STAGED_RAR else 'OFF'}")

    set_network_trainable_mode(stage)

    compile_kwargs = {"loss_weights": stage_loss_weights}
    if stage_optimizer == "adam":
        compile_kwargs["lr"] = stage_lr
    if stage_trainables:
        compile_kwargs["external_trainable_variables"] = stage_trainables

    model.compile(stage_optimizer, **compile_kwargs)

    callbacks = [ParameterCallback(period=500)]
    if use_gp:
        callbacks.append(
            GradientBalanceCallback(
                period=100,
                alpha=0.9,
                power=0.5,
                min_weight=0.5,
                max_weight=20.0,
            )
        )
    if ENABLE_STAGED_RAR:
        callbacks.append(
            ResidualRefinementCallback(
                period=4000,
                num_new_points=200,
                total_candidates=2000,
                n_phase_bins=N_PHASE_BINS,
            )
        )

    train_kwargs = {
        "iterations": stage_iters,
        "display_every": 500,
        "callbacks": callbacks,
    }
    if stage_optimizer == "adam":
        train_kwargs["batch_size"] = BATCH_SIZE
    return model.train(**train_kwargs)

# ===================== Training =====================
print("\n" + "="*70)
print("Starting Coupled Staged Training with PyTorch Backend")
print("Strategy: Fully Coupled Model + Staged Optimization")
print("="*70 + "\n")

start_time = time.time()

adam_loss_train = []
adam_loss_test = []
train_state = None

for stage in range(3):
    stage_losshistory, train_state = train_stage(stage)
    adam_loss_train.extend(stage_losshistory.loss_train)
    if stage_losshistory.loss_test is not None and len(stage_losshistory.loss_test) > 0:
        adam_loss_test.extend(stage_losshistory.loss_test)

print("\n" + "="*70)
print("Phase 2: L-BFGS Optimizer (Fine-tuning)")
print("="*70 + "\n")

# L-BFGS Fine-tuning
model.compile(
    "L-BFGS",
    external_trainable_variables=[k_bar_raw],
    loss_weights=model.loss_weights
)

losshistory_lbfgs, train_state_lbfgs = model.train(
    display_every=100,
    callbacks=[ParameterCallback(period=100)]
)

# Merge history for plotting (optional, but good for visualization)
# Note: DeepXDE L-BFGS history structure might differ slightly, but for basic plotting strict merging isn't strictly required if we just plot the final result.
# However, parameter_history is updated by callback, so it captures both phases.

training_time = time.time() - start_time

print("\n" + "="*70)
print("Training Complete!")
print(f"Total training time: {training_time:.1f}s ({training_time/60:.1f} min)")
final_k_bar = positive_parameter(k_bar_raw).item()
print(f"Final k_bar: {final_k_bar:.6f} (true: {k_bar_true:.6f})")
k_error_final = abs(final_k_bar - k_bar_true) / k_bar_true * 100
print(f"k_bar error: {k_error_final:.2f}%")
print("="*70 + "\n")

RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")


def build_timestamped_name(base_name: str) -> str:
    """
    为当前运行生成带时间戳的输出文件名，避免覆盖历史结果。
    """
    stem, ext = os.path.splitext(base_name)
    return f"{stem}_{RUN_TIMESTAMP}{ext}"

# ===================== Save Model =====================
model_basename = build_timestamped_name("poromechanics_coupled_pytorch")
model.save(model_basename)
print("Model saved successfully")

# ===================== Post-Processing and Visualization =====================
print("\nGenerating visualizations...")


def save_figure_with_fallback(fig: plt.Figure, filename: str) -> str:
    """
    优先保存到目标文件；若文件被外部程序占用，则自动改名后重试。
    对 PNG 输出使用更高 dpi 以提升细节保真度。
    """
    _, ext = os.path.splitext(filename)
    dpi = 450 if ext.lower() == '.png' else 150
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
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Loss history
epochs_plot = np.arange(len(adam_loss_train))
ax[0].semilogy(epochs_plot, adam_loss_train, 'b-', linewidth=1.5, label='Train Loss')
if len(adam_loss_test) == len(adam_loss_train) and len(adam_loss_test) > 0:
    ax[0].semilogy(epochs_plot, adam_loss_test, 'r--', linewidth=1.5, label='Test Loss')
ax[0].set_xlabel('Iteration', fontsize=12)
ax[0].set_ylabel('Loss', fontsize=12)
ax[0].set_title('Training Loss History', fontsize=13)
ax[0].legend(fontsize=10)
ax[0].grid(True, alpha=0.3)

# Parameter evolution from history
if len(parameter_history['epoch']) > 0:
    ax[1].plot(parameter_history['epoch'], parameter_history['k_bar'], 'b-', linewidth=2, label='k_bar')
    ax[1].axhline(y=k_bar_true, color='b', linestyle='--', linewidth=1.5, alpha=0.7, label='k_bar true')
    ax[1].set_xlabel('Iteration', fontsize=12)
    ax[1].set_ylabel('Parameter Value', fontsize=12)
    ax[1].set_title('k_bar Evolution', fontsize=13)
    ax[1].legend(fontsize=10)
    ax[1].grid(True, alpha=0.3)

plt.tight_layout()
saved_name = save_figure_with_fallback(fig, build_timestamped_name('training_loss.pdf'))
plt.close()
print(f"  ✓ Training loss plot saved: {saved_name}")

# ===== Plot 2: Detailed Parameter Evolution =====
if len(parameter_history['epoch']) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))

    epochs_hist = parameter_history['epoch']

    # Permeability
    k_history = np.array(parameter_history['k_bar']) * k_star
    ax.plot(epochs_hist, k_history, color='red', linewidth=1.6, label='Predicted')
    ax.hlines(
        y=k,
        xmin=epochs_hist[0],
        xmax=epochs_hist[-1],
        color='black',
        linewidth=1.6,
        linestyle='--',
        label='True',
    )
    ax.set_xlabel('epochs')
    ax.set_ylabel(r'$\kappa$ ($\mathrm{m}^2$)')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.xaxis.grid(color='0.65', linestyle='--', linewidth=0.8)

    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.92, wspace=0.3, hspace=0.25)
    parameter_figure_files = []
    for extension in ('png', 'pdf', 'svg'):
        saved_name = save_figure_with_fallback(
            fig,
            build_timestamped_name(f'parameter_evolution.{extension}'),
        )
        parameter_figure_files.append(saved_name)
    plt.close()
    print(f"  ✓ Parameter evolution plots saved: {parameter_figure_files}")

# ===== Plot 3: Predictions at Test Points =====
print("\nEvaluating model at test points...")

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
point_folder_names = [
    f"point_0.25_0.75_{RUN_TIMESTAMP}",
    f"point_0.75_0.75_{RUN_TIMESTAMP}",
    f"point_0.75_0.25_{RUN_TIMESTAMP}",
]

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
saved_point_folders = []

for point_idx, (loc, folder_name, t_plot) in enumerate(
    zip(points_location, point_folder_names, t_plots)
):
    os.makedirs(folder_name, exist_ok=True)
    saved_point_folders.append(folder_name)

    for var_idx, (var, ylabel) in enumerate(zip(variables, var_labels)):
        pred = pred_data[var_idx][point_idx]
        true = true_data[var_idx][point_idx]

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.plot(
            t_plot,
            pred,
            'r*',
            markevery=5,
            markersize=8,
            label='PINN',
            linewidth=1.5,
        )
        ax.plot(t_plot, true, 'b--', linewidth=2, label='Analytical')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()

        saved_name = save_figure_with_fallback(
            fig,
            os.path.join(folder_name, f'{var}_comparison.png'),
        )
        plt.close()
        print(f"  ✓ Saved point comparison plot: {saved_name}")

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
print(f"  - {build_timestamped_name('training_loss.pdf')}")
print(f"  - {build_timestamped_name('parameter_evolution')}.[png|pdf|svg]")
for folder_name in saved_point_folders:
    print(f"  - {folder_name}/[p|u|v]_comparison.png")
print(f"  - {model_basename}-*.pt (saved model)")
