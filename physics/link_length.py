from dataclasses import dataclass
from .power_budget import LinkConfig, receiver_sensitivity, connector_loss, splice_loss
from .dispersion import DispersionConfig, FiberType


@dataclass
class MaxLengthResult:
    L_power_km: float       # Max length from power budget
    L_disp_km: float        # Max length from dispersion constraint
    L_max_km: float         # min(L_power, L_disp)
    limiting_factor: str    # 'Puissance' or 'Dispersion'


def max_length_from_power(
    Pe_dbm: float,
    fixed_losses_db: float,
    sensitivity_dbm: float,
    alpha_db_km: float,
) -> float:
    """
    L_power = (Pe - fixed_losses - sensitivity) / alpha
    fixed_losses includes: both couplers + connectors + splices (not fiber itself).
    """
    numerator = Pe_dbm - fixed_losses_db - sensitivity_dbm
    if alpha_db_km <= 0:
        return float('inf')
    return max(0.0, numerator / alpha_db_km)


def max_length_from_dispersion(
    BP_target_GHz: float,
    Dch_ps_nm_km: float,
    delta_lambda_nm: float,
) -> float:
    """
    For chromatic-dispersion-limited links:
    L_disp derived from BP_target = 0.7 / (Dch * delta_lambda * L)
    Returns inf if Dch or delta_lambda is 0 (no chromatic limit).
    """
    if abs(Dch_ps_nm_km) <= 0 or delta_lambda_nm <= 0:
        return float('inf')
    tau_target_ps = (0.7 / (BP_target_GHz * 1e9)) * 1e12
    return max(0.0, tau_target_ps / (abs(Dch_ps_nm_km) * delta_lambda_nm))


def compute_max_length(
    link_cfg: LinkConfig,
    disp_cfg: DispersionConfig,
    BP_target_GHz: float,
) -> MaxLengthResult:
    sens = receiver_sensitivity(link_cfg.detector, link_cfg.bitrate_GHz,
                                link_cfg.sensitivity_override_dbm)
    A_conn = connector_loss(link_cfg.Nc, link_cfg.loss_per_connector_db)
    A_spl = splice_loss(link_cfg.Ns, link_cfg.loss_per_splice_db)
    fixed_losses = (link_cfg.coupler_loss_laser_db + link_cfg.coupler_loss_detector_db
                    + A_conn + A_spl)

    L_power = max_length_from_power(link_cfg.Pe_dbm, fixed_losses, sens, link_cfg.alpha_db_km)

    L_disp = max_length_from_dispersion(
        BP_target_GHz,
        disp_cfg.Dch_ps_nm_km,
        disp_cfg.delta_lambda_nm,
    )

    L_max = min(L_power, L_disp)
    limiting = 'Puissance' if L_power <= L_disp else 'Dispersion'

    return MaxLengthResult(
        L_power_km=L_power,
        L_disp_km=L_disp,
        L_max_km=L_max,
        limiting_factor=limiting,
    )
