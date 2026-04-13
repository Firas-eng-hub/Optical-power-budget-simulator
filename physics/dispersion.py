import math
import enum
from dataclasses import dataclass
import numpy as np

C_KM_S = 3e5  # speed of light in km/s


class FiberType(enum.Enum):
    STEP_INDEX = "Step-index multimode"
    GRADED_INDEX = "Graded-index multimode"
    SINGLE_MODE = "Single-mode"


@dataclass
class DispersionConfig:
    Dch_ps_nm_km: float
    delta_lambda_nm: float
    L_km: float
    fiber_type: FiberType
    n1: float
    delta: float
    bandwidth_distance_MHz_km: float = 0.0


@dataclass
class DispersionResult:
    delta_tau_ch_ps: float
    delta_tau_modal_ps: float
    delta_tau_total_ps: float
    bandwidth_GHz: float
    bandwidth_model_GHz: float
    bandwidth_vendor_GHz: float


def chromatic_broadening(Dch: float, delta_lambda: float, L: float) -> float:
    return abs(Dch) * delta_lambda * L


def modal_broadening_step_index(n1: float, delta: float, L: float) -> float:
    return (n1 / C_KM_S) * delta * L * 1e12


def modal_broadening_graded_index(n1: float, delta: float, L: float) -> float:
    return (n1 / (8 * C_KM_S)) * (delta ** 2) * L * 1e12


def total_broadening(delta_tau_ch: float, delta_tau_modal: float) -> float:
    return math.sqrt(delta_tau_ch**2 + delta_tau_modal**2)


def bandwidth_from_broadening(delta_tau_total_ps: float) -> float:
    if delta_tau_total_ps <= 0:
        return float('inf')
    return 0.7 / (delta_tau_total_ps * 1e-12) / 1e9


def vendor_bandwidth_GHz(bandwidth_distance_MHz_km: float, L_km: float) -> float:
    if bandwidth_distance_MHz_km <= 0 or L_km <= 0:
        return float('inf')
    return (bandwidth_distance_MHz_km / L_km) / 1000.0


def compute_dispersion(config: DispersionConfig) -> DispersionResult:
    tau_ch = chromatic_broadening(config.Dch_ps_nm_km, config.delta_lambda_nm, config.L_km)

    if config.fiber_type == FiberType.SINGLE_MODE:
        tau_modal = 0.0
    elif config.fiber_type == FiberType.GRADED_INDEX:
        tau_modal = modal_broadening_graded_index(config.n1, config.delta, config.L_km)
    else:
        tau_modal = modal_broadening_step_index(config.n1, config.delta, config.L_km)

    tau_total = total_broadening(tau_ch, tau_modal)
    bp_model = bandwidth_from_broadening(tau_total)
    bp_vendor = vendor_bandwidth_GHz(config.bandwidth_distance_MHz_km, config.L_km)
    bp = min(bp_model, bp_vendor)

    return DispersionResult(
        delta_tau_ch_ps=tau_ch,
        delta_tau_modal_ps=tau_modal,
        delta_tau_total_ps=tau_total,
        bandwidth_GHz=bp,
        bandwidth_model_GHz=bp_model,
        bandwidth_vendor_GHz=bp_vendor,
    )


def dispersion_vs_distance(config: DispersionConfig, L_array: np.ndarray) -> tuple:
    tau_ch = abs(config.Dch_ps_nm_km) * config.delta_lambda_nm * L_array

    if config.fiber_type == FiberType.SINGLE_MODE:
        tau_modal = np.zeros_like(L_array)
    elif config.fiber_type == FiberType.GRADED_INDEX:
        tau_modal = (config.n1 / (8 * C_KM_S)) * (config.delta ** 2) * L_array * 1e12
    else:
        tau_modal = (config.n1 / C_KM_S) * config.delta * L_array * 1e12

    tau_total = np.sqrt(tau_ch**2 + tau_modal**2)
    bw_model = np.where(tau_total > 0, 0.7 / (tau_total * 1e-12) / 1e9, np.inf)

    if config.bandwidth_distance_MHz_km > 0:
        safe_L = np.where(L_array > 0, L_array, np.nan)
        bw_vendor = (config.bandwidth_distance_MHz_km / safe_L) / 1000.0
        bw_vendor = np.where(np.isnan(bw_vendor), np.inf, bw_vendor)
    else:
        bw_vendor = np.full_like(L_array, np.inf, dtype=float)

    bw = np.minimum(bw_model, bw_vendor)
    return tau_total, bw
