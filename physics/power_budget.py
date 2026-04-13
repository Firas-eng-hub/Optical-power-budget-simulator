import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class LinkConfig:
    Pe_dbm: float                       # Emitted power (dBm)
    wavelength_nm: float                # Wavelength (nm)
    L_km: float                         # Link length (km)
    alpha_db_km: float                  # Fiber attenuation coefficient (dB/km)
    Nc: int                             # Number of connectors
    loss_per_connector_db: float        # Loss per connector (dB)
    Ns: int                             # Number of splices
    loss_per_splice_db: float           # Loss per splice (dB)
    coupler_loss_laser_db: float        # Laser-fiber coupler loss (dB)
    coupler_loss_detector_db: float     # Fiber-detector coupler loss (dB)
    other_losses_db: float              # Additional lumped losses (dB)
    NA: float                           # Numerical aperture (display only)
    detector: str                       # 'PIN', 'APD', or 'Manuel'
    bitrate_GHz: float                  # Bitrate (GHz)
    sensitivity_override_dbm: Optional[float] = None  # Fixed sensitivity (overrides formula)


@dataclass
class PowerBudgetResult:
    A_fiber_db: float               # Fiber attenuation (dB)
    A_connectors_db: float          # Total connector loss (dB)
    A_splices_db: float             # Total splice loss (dB)
    coupler_loss_laser_db: float    # Laser-fiber coupler loss (dB)
    coupler_loss_detector_db: float # Fiber-detector coupler loss (dB)
    other_losses_db: float          # Additional lumped losses (dB)
    A_total_db: float               # Sum of all losses (dB)
    Pr_dbm: float                   # Received power (dBm)
    sensitivity_dbm: float          # Detector sensitivity threshold (dBm)
    margin_db: float                # Marge = Pr - sensitivity
    is_feasible: bool               # True if margin >= 0


def fiber_attenuation(alpha: float, L: float) -> float:
    """Returns alpha * L [dB]."""
    return alpha * L


def connector_loss(Nc: int, loss_per_connector_db: float) -> float:
    """Returns Nc * loss_per_connector_db [dB]."""
    return Nc * loss_per_connector_db


def splice_loss(Ns: int, loss_per_splice_db: float) -> float:
    """Returns Ns * loss_per_splice_db [dB]."""
    return Ns * loss_per_splice_db


def receiver_sensitivity(detector: str, bitrate_GHz: float,
                         override_dbm: Optional[float] = None) -> float:
    """
    Minimum detectable power in dBm.
    If override_dbm is set, returns it directly (manual input mode).
    For the exercise presets:
      PIN: -52 dBm
      PIIPN/APD: -64 dBm
    Otherwise falls back to the generic formulas.
    """
    if override_dbm is not None:
        return override_dbm
    detector_upper = detector.upper()
    if detector_upper == 'PIN':
        return -52.0
    if detector_upper in {'PIIPN', 'APD'}:
        return -64.0
    f_MHz = bitrate_GHz * 1000.0
    log_f = 10 * math.log10(max(f_MHz, 1e-9))
    if detector_upper == 'APD':
        return -67.0 + log_f
    return -53.0 + log_f


def compute_power_budget(config: LinkConfig) -> PowerBudgetResult:
    A_fiber = fiber_attenuation(config.alpha_db_km, config.L_km)
    A_conn = connector_loss(config.Nc, config.loss_per_connector_db)
    A_spl = splice_loss(config.Ns, config.loss_per_splice_db)

    A_total = (A_fiber + A_conn + A_spl
               + config.coupler_loss_laser_db + config.coupler_loss_detector_db
               + config.other_losses_db)
    Pr = config.Pe_dbm - A_total

    sens = receiver_sensitivity(config.detector, config.bitrate_GHz,
                                config.sensitivity_override_dbm)
    margin = Pr - sens

    return PowerBudgetResult(
        A_fiber_db=A_fiber,
        A_connectors_db=A_conn,
        A_splices_db=A_spl,
        coupler_loss_laser_db=config.coupler_loss_laser_db,
        coupler_loss_detector_db=config.coupler_loss_detector_db,
        other_losses_db=config.other_losses_db,
        A_total_db=A_total,
        Pr_dbm=Pr,
        sensitivity_dbm=sens,
        margin_db=margin,
        is_feasible=(margin >= 0),
    )


def power_vs_distance(config: LinkConfig, L_array: np.ndarray) -> np.ndarray:
    """Received power (dBm) as a function of distance array (km)."""
    fixed_losses = (config.coupler_loss_laser_db + config.coupler_loss_detector_db
                    + connector_loss(config.Nc, config.loss_per_connector_db)
                    + splice_loss(config.Ns, config.loss_per_splice_db)
                    + config.other_losses_db)
    return config.Pe_dbm - fixed_losses - config.alpha_db_km * L_array


def event_positions_uniform(L_km: float, count: int) -> np.ndarray:
    """Uniformly distributes `count` events inside ]0, L_km[."""
    if count <= 0 or L_km <= 0:
        return np.array([], dtype=float)
    return np.linspace(0.0, float(L_km), int(count) + 2, dtype=float)[1:-1]


def power_vs_distance_stepped(
    config: LinkConfig,
    L_array: np.ndarray,
    connector_positions_km: Optional[np.ndarray] = None,
    splice_positions_km: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Received power profile (dBm) with local drops at connector/splice positions.
    Coupler and other losses remain lumped fixed losses.
    """
    L_vals = np.asarray(L_array, dtype=float)
    conn_pos = (event_positions_uniform(config.L_km, config.Nc)
                if connector_positions_km is None
                else np.sort(np.asarray(connector_positions_km, dtype=float)))
    spl_pos = (event_positions_uniform(config.L_km, config.Ns)
               if splice_positions_km is None
               else np.sort(np.asarray(splice_positions_km, dtype=float)))

    fixed_losses = (
        config.coupler_loss_laser_db
        + config.coupler_loss_detector_db
        + config.other_losses_db
    )

    conn_cum = np.searchsorted(conn_pos, L_vals, side='right') * config.loss_per_connector_db
    spl_cum = np.searchsorted(spl_pos, L_vals, side='right') * config.loss_per_splice_db

    return config.Pe_dbm - fixed_losses - config.alpha_db_km * L_vals - conn_cum - spl_cum
