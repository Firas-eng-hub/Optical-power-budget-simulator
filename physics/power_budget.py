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
    Otherwise uses standard formulas (bitrate in MHz):
      PIN: -53 + 10*log10(f_MHz)
      APD: -67 + 10*log10(f_MHz)
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
               + config.coupler_loss_laser_db + config.coupler_loss_detector_db)
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
                    + splice_loss(config.Ns, config.loss_per_splice_db))
    return config.Pe_dbm - fixed_losses - config.alpha_db_km * L_array
