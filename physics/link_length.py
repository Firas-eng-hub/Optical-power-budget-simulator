from dataclasses import dataclass
from .power_budget import LinkConfig, receiver_sensitivity, connector_loss, splice_loss
from .dispersion import DispersionConfig


@dataclass
class MaxLengthResult:
    L_power_km: float
    L_disp_km: float
    L_max_km: float
    limiting_factor: str


def max_length_from_power(
    Pe_dbm: float,
    fixed_losses_db: float,
    sensitivity_dbm: float,
    alpha_db_km: float,
) -> float:
    numerator = Pe_dbm - fixed_losses_db - sensitivity_dbm
    if alpha_db_km <= 0:
        return float('inf')
    return max(0.0, numerator / alpha_db_km)


def max_length_from_dispersion(
    BP_target_GHz: float,
    Dch_ps_nm_km: float,
    delta_lambda_nm: float,
    bandwidth_distance_MHz_km: float = 0.0,
) -> float:
    limits = []

    if abs(Dch_ps_nm_km) > 0 and delta_lambda_nm > 0 and BP_target_GHz > 0:
        tau_target_ps = (0.7 / (BP_target_GHz * 1e9)) * 1e12
        limits.append(max(0.0, tau_target_ps / (abs(Dch_ps_nm_km) * delta_lambda_nm)))

    if bandwidth_distance_MHz_km > 0 and BP_target_GHz > 0:
        bp_target_MHz = BP_target_GHz * 1000.0
        limits.append(max(0.0, bandwidth_distance_MHz_km / bp_target_MHz))

    return min(limits) if limits else float('inf')


def compute_max_length(
    link_cfg: LinkConfig,
    disp_cfg: DispersionConfig,
    BP_target_GHz: float,
) -> MaxLengthResult:
    sens = receiver_sensitivity(
        link_cfg.detector,
        link_cfg.bitrate_GHz,
        link_cfg.sensitivity_override_dbm,
    )
    A_conn = connector_loss(link_cfg.Nc, link_cfg.loss_per_connector_db)
    A_spl = splice_loss(link_cfg.Ns, link_cfg.loss_per_splice_db)
    fixed_losses = (
        link_cfg.coupler_loss_laser_db
        + link_cfg.coupler_loss_detector_db
        + link_cfg.other_losses_db
        + A_conn
        + A_spl
    )

    L_power = max_length_from_power(link_cfg.Pe_dbm, fixed_losses, sens, link_cfg.alpha_db_km)
    L_disp = max_length_from_dispersion(
        BP_target_GHz,
        disp_cfg.Dch_ps_nm_km,
        disp_cfg.delta_lambda_nm,
        disp_cfg.bandwidth_distance_MHz_km,
    )

    L_max = min(L_power, L_disp)
    limiting = 'Puissance' if L_power <= L_disp else 'Dispersion'

    return MaxLengthResult(
        L_power_km=L_power,
        L_disp_km=L_disp,
        L_max_km=L_max,
        limiting_factor=limiting,
    )
