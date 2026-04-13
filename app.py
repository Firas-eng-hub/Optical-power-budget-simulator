import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from physics.power_budget import LinkConfig, compute_power_budget, power_vs_distance
from physics.dispersion import FiberType, DispersionConfig, compute_dispersion, dispersion_vs_distance
from physics.link_length import compute_max_length

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulateur Liaison Optique",
    page_icon="🔭",
    layout="wide",
)

st.title("Simulateur de Liaison Optique")
st.caption("Bilan de puissance · Dispersion — TP Fibre Optique")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
WAVELENGTH_PRESETS = {
    850:  {"alpha": 2.5, "Dch": 80.0},
    1300: {"alpha": 0.5, "Dch":  0.0},
    1550: {"alpha": 0.2, "Dch": 17.0},
}

SOURCE_PRESETS = {
    "LED": {"Pe_dbm": 0.0, "coupler_db": 17.0, "delta_lambda_nm": 40.0},
    "LASER": {"Pe_dbm": 10.0, "coupler_db": 3.0, "delta_lambda_nm": 2.0},
}

FIBER_CATALOG = {
    "Multimode à saut d'indice": {
        "type": FiberType.STEP_INDEX,
        "alpha": 5.0,
        "bandwidth_distance_MHz_km": 10.0,
    },
    "Multimode à gradient d'indice": {
        "type": FiberType.GRADED_INDEX,
        "alpha": 3.0,
        "bandwidth_distance_MHz_km": 100.0,
    },
    "Monomode": {
        "type": FiberType.SINGLE_MODE,
        "alpha": 0.5,
        "bandwidth_distance_MHz_km": 0.0,
    },
}

FIBER_TYPES = {
    label: data["type"] for label, data in FIBER_CATALOG.items()
}

DEFAULT_N1 = 1.468
DEFAULT_DELTA = 0.01


def fmt(val, decimals=2, unit=""):
    if val == float('inf') or val != val:
        return "∞"
    return f"{val:.{decimals}f} {unit}".strip()


def slider_input(label: str, min_val: float, max_val: float, default: float,
                 step: float, key: str, fmt_str: str = "%.3f"):
    """
    Slider + number_input combo — slide for speed, type for precision.
    Uses st.session_state[key] as the single source of truth.
    """
    state_key = f"_v_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = float(default)

    current = float(st.session_state[state_key])
    # Clamp to bounds in case they changed
    current = max(float(min_val), min(float(max_val), current))

    col_s, col_n = st.columns([3, 1])
    with col_s:
        sv = st.slider(label, float(min_val), float(max_val), current,
                       float(step), key=f"_sl_{key}")
    with col_n:
        nv = st.number_input("", float(min_val), float(max_val), current,
                             float(step), format=fmt_str,
                             label_visibility="collapsed", key=f"_ni_{key}")

    # Whichever widget changed from the stored value wins
    if sv != current:
        st.session_state[state_key] = sv
        return sv
    if nv != current:
        st.session_state[state_key] = nv
        return nv
    return current


# ---------------------------------------------------------------------------
# Sidebar — Inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    use_exercise_catalog = st.checkbox("Utiliser le catalogue de l'exercice", value=True)

    # --- Emetteur ---
    st.subheader("Émetteur optique")
    lambda_choice = st.selectbox("Fenêtre λ (nm)", [850, 1300, 1550, "Personnalisé"])
    if lambda_choice == "Personnalisé":
        lambda_nm = st.number_input("λ personnalisé (nm)", min_value=400.0, max_value=2000.0,
                                    value=1550.0, step=10.0)
        default_alpha = 0.2
        default_Dch = 17.0
    else:
        lambda_nm = float(lambda_choice)
        default_alpha = WAVELENGTH_PRESETS[lambda_choice]["alpha"]
        default_Dch = WAVELENGTH_PRESETS[lambda_choice]["Dch"]

    if use_exercise_catalog:
        source_type = st.radio("Source", ["LED", "LASER"], horizontal=True)
        source_preset = SOURCE_PRESETS[source_type]
        col_pe, col_dl = st.columns(2)
        with col_pe:
            Pe_dbm = st.number_input("Puissance émise Pe (dBm)", min_value=-20.0, max_value=20.0,
                                     value=float(source_preset["Pe_dbm"]), step=0.5, format="%.1f")
        with col_dl:
            delta_lambda = st.number_input("Largeur spectrale Δλ (nm)", min_value=0.1, max_value=100.0,
                                           value=float(source_preset["delta_lambda_nm"]), step=0.1, format="%.1f")
        st.caption(f"Perte de couplage par défaut = {source_preset['coupler_db']:.1f} dB")
    else:
        source_type = "Manuel"
        source_preset = None
        Pe_dbm = slider_input("Puissance émise Pe (dBm)", -20.0, 10.0, -3.0, 0.5,
                              "Pe_dbm", "%.1f")
        delta_lambda = slider_input("Largeur spectrale Δλ (nm)", 0.1, 50.0, 1.0, 0.1,
                                    "delta_lambda", "%.2f")

    # --- Fibre ---
    st.subheader("Fibre optique")
    fiber_label = st.selectbox("Type de fibre", list(FIBER_TYPES.keys()))
    fiber_type = FIBER_TYPES[fiber_label]
    fiber_catalog = FIBER_CATALOG[fiber_label]

    L_km = slider_input("Longueur L (km)", 0.1, 300.0, 15.0, 0.1, "L_km", "%.1f")

    col_a, col_b = st.columns(2)
    with col_a:
        alpha_default = fiber_catalog["alpha"] if use_exercise_catalog else default_alpha
        alpha = st.number_input("α (dB/km)", min_value=0.01, max_value=10.0,
                                value=alpha_default, step=0.05, format="%.3f")
    with col_b:
        Dch = st.number_input("Dch (ps/nm/km)", min_value=-200.0, max_value=200.0,
                              value=default_Dch, step=1.0)
    bandwidth_distance_MHz_km = fiber_catalog["bandwidth_distance_MHz_km"] if use_exercise_catalog else 0.0
    if bandwidth_distance_MHz_km > 0:
        st.caption(f"Produit bande passante-distance = {bandwidth_distance_MHz_km:.1f} MHz·km")

    # --- Coupleurs ---
    st.subheader("Coupleurs")
    col_cl, col_cd = st.columns(2)
    with col_cl:
        coupler_laser_default = source_preset["coupler_db"] if use_exercise_catalog else 2.0
        coupler_laser_db = st.number_input("Perte coupleur source-fibre (dB)",
                                           min_value=0.0, max_value=20.0, value=coupler_laser_default,
                                           step=0.1, format="%.1f")
    with col_cd:
        coupler_detect_db = st.number_input("Perte coupleur fibre-détecteur (dB)",
                                            min_value=0.0, max_value=20.0,
                                            value=1.0 if use_exercise_catalog else 2.5,
                                            step=0.1, format="%.1f")

    # --- Connexions ---
    st.subheader("Connexions")
    col_nc, col_lc = st.columns(2)
    with col_nc:
        Nc = st.number_input("Nb connecteurs", min_value=0, max_value=50,
                             value=2 if use_exercise_catalog else 5, step=1)
    with col_lc:
        loss_per_conn = st.number_input("Perte/connecteur (dB)", min_value=0.0,
                                        max_value=5.0, value=1.0, step=0.1, format="%.2f")

    splice_mode = st.radio("Épissures", ["Manuel (Ns)", "Espacement (m)", "Rouleaux (km)"], horizontal=True)
    if splice_mode == "Manuel (Ns)":
        Ns = st.number_input("Nb épissures", min_value=0, max_value=500,
                             value=11 if use_exercise_catalog else 50, step=1)
    elif splice_mode == "Espacement (m)":
        splice_spacing_m = st.number_input("Espacement épissures (m)",
                                           min_value=10, max_value=5000, value=200, step=10)
        Ns = max(0, int(L_km * 1000 / splice_spacing_m) - 1)
        st.caption(f"→ Ns = {Ns} épissures pour L = {L_km:.1f} km")
    else:
        reel_length_km = st.number_input("Longueur d'un rouleau (km)",
                                         min_value=0.1, max_value=50.0,
                                         value=1.0 if use_exercise_catalog else 2.0,
                                         step=0.1, format="%.1f")
        Ns = max(0, int(np.ceil(L_km / reel_length_km)) - 1)
        st.caption(f"→ Ns = {Ns} épissures pour des rouleaux de {reel_length_km:.1f} km")
    loss_per_splice = st.number_input("Perte/épissure (dB)", min_value=0.0,
                                      max_value=2.0, value=0.3 if use_exercise_catalog else 0.1,
                                      step=0.01, format="%.3f")
    other_losses_db = st.number_input("Autres pertes (dB)", min_value=0.0,
                                      max_value=50.0, value=0.0, step=0.1, format="%.1f")

    # --- Récepteur ---
    st.subheader("Récepteur")
    detector = st.radio("Type de détecteur", ["PIN", "PIIPN", "Manuel"], horizontal=True)
    if detector == "Manuel":
        sensitivity_manual = st.number_input("Sensibilité (dBm)", min_value=-80.0,
                                             max_value=0.0, value=-40.0, step=0.5)
        sensitivity_override = sensitivity_manual
    else:
        sensitivity_override = None
    bitrate_GHz = st.number_input("Débit (GHz)", min_value=0.001, max_value=100.0,
                                  value=0.002 if use_exercise_catalog else 2.488,
                                  step=0.001, format="%.3f")
    BP_target = st.number_input("BP cible (GHz)", min_value=0.001, max_value=100.0,
                                value=0.002 if use_exercise_catalog else 2.488,
                                step=0.001, format="%.3f")

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
link_cfg = LinkConfig(
    Pe_dbm=Pe_dbm,
    wavelength_nm=lambda_nm,
    L_km=L_km,
    alpha_db_km=alpha,
    Nc=int(Nc),
    loss_per_connector_db=float(loss_per_conn),
    Ns=int(Ns),
    loss_per_splice_db=float(loss_per_splice),
    coupler_loss_laser_db=float(coupler_laser_db),
    coupler_loss_detector_db=float(coupler_detect_db),
    other_losses_db=float(other_losses_db),
    NA=0.0,
    detector=detector,
    bitrate_GHz=bitrate_GHz,
    sensitivity_override_dbm=sensitivity_override,
)

disp_cfg = DispersionConfig(
    Dch_ps_nm_km=Dch,
    delta_lambda_nm=delta_lambda,
    L_km=L_km,
    fiber_type=fiber_type,
    n1=DEFAULT_N1,
    delta=DEFAULT_DELTA,
    bandwidth_distance_MHz_km=bandwidth_distance_MHz_km,
)

pb = compute_power_budget(link_cfg)
disp = compute_dispersion(disp_cfg)
lmax = compute_max_length(link_cfg, disp_cfg, BP_target)

# ---------------------------------------------------------------------------
# Row 1 — Key metrics
# ---------------------------------------------------------------------------
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Puissance reçue Pr", f"{pb.Pr_dbm:.2f} dBm",
              delta=f"Sensibilité: {pb.sensitivity_dbm:.1f} dBm",
              delta_color="off")

with c2:
    margin_str = f"{pb.margin_db:+.2f} dB"
    feasibility = "✅ LIAISON OK" if pb.is_feasible else "❌ INSUFFISANT"
    st.metric("Marge", margin_str, delta=feasibility,
              delta_color="normal" if pb.is_feasible else "inverse")

with c3:
    bp_val = disp.bandwidth_GHz
    bp_str = "∞ GHz" if bp_val == float('inf') else f"{bp_val:.3f} GHz"
    bp_delta = f"Δτ total: {disp.delta_tau_total_ps:.1f} ps"
    if disp.bandwidth_vendor_GHz != float('inf'):
        bp_delta = f"Limite catalogue: {disp.bandwidth_vendor_GHz:.3f} GHz"
    st.metric("Bande passante BP", bp_str,
              delta=bp_delta,
              delta_color="off")

with c4:
    lmax_str = "∞ km" if lmax.L_max_km == float('inf') else f"{lmax.L_max_km:.1f} km"
    st.metric("Longueur max", lmax_str,
              delta=f"Limité par: {lmax.limiting_factor}",
              delta_color="off")

# ---------------------------------------------------------------------------
# Row 2 — Detail tables
# ---------------------------------------------------------------------------
st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Bilan de puissance")
    power_data = {
        "Paramètre": [
            "Puissance émise Pe",
            "Coupleur laser-fibre",
            "Atténuation fibre",
            f"Connecteurs ({Nc} × {loss_per_conn:.2f} dB)",
            f"Épissures ({Ns} × {loss_per_splice:.3f} dB)",
            "Coupleur fibre-détecteur",
            "Autres pertes",
            "Perte totale",
            "Puissance reçue Pr",
            "Sensibilité détecteur",
            "Marge",
        ],
        "Valeur": [
            f"{Pe_dbm:.2f} dBm",
            f"{pb.coupler_loss_laser_db:.2f} dB",
            f"{pb.A_fiber_db:.2f} dB",
            f"{pb.A_connectors_db:.2f} dB",
            f"{pb.A_splices_db:.2f} dB",
            f"{pb.coupler_loss_detector_db:.2f} dB",
            f"{pb.other_losses_db:.2f} dB",
            f"{pb.A_total_db:.2f} dB",
            f"{pb.Pr_dbm:.2f} dBm",
            f"{pb.sensitivity_dbm:.2f} dBm",
            f"{pb.margin_db:+.2f} dB",
        ],
    }
    st.table(power_data)

with col_right:
    st.subheader("Dispersion")
    disp_data = {
        "Paramètre": [
            "Δτ chromatique",
            "Δτ modal",
            "Δτ total (RMS)",
            "BP modèle",
            "BP catalogue",
            "Bande passante BP",
        ],
        "Valeur": [
            f"{disp.delta_tau_ch_ps:.2f} ps",
            f"{disp.delta_tau_modal_ps:.2f} ps",
            f"{disp.delta_tau_total_ps:.2f} ps",
            "∞ GHz" if disp.bandwidth_model_GHz == float('inf') else f"{disp.bandwidth_model_GHz:.3f} GHz",
            "∞ GHz" if disp.bandwidth_vendor_GHz == float('inf') else f"{disp.bandwidth_vendor_GHz:.3f} GHz",
            bp_str,
        ],
    }
    st.table(disp_data)

    st.subheader("Longueur maximale")
    lmax_pw = "∞ km" if lmax.L_power_km == float('inf') else f"{lmax.L_power_km:.1f} km"
    lmax_dp = "∞ km" if lmax.L_disp_km == float('inf') else f"{lmax.L_disp_km:.1f} km"
    lmax_data = {
        "Contrainte": ["Puissance", "Dispersion", "L_max = min"],
        "L_max": [lmax_pw, lmax_dp, lmax_str],
    }
    st.table(lmax_data)

# ---------------------------------------------------------------------------
# Row 3 — Plots
# ---------------------------------------------------------------------------
st.markdown("---")
L_plot = np.linspace(0.1, max(200.0, L_km * 2, lmax.L_max_km * 1.3 if lmax.L_max_km != float('inf') else 200.0), 600)
Pr_arr = power_vs_distance(link_cfg, L_plot)
tau_arr, bw_arr = dispersion_vs_distance(disp_cfg, L_plot)

tab0, tab1, tab2 = st.tabs(["🔗 Schéma de liaison", "📶 Puissance vs Distance", "〰️ Dispersion vs Distance"])

# ---- Tab 0: Link schema ----
with tab0:
    is_monomode = (fiber_type == FiberType.SINGLE_MODE)
    if fiber_type == FiberType.GRADED_INDEX:
        fiber_label_disp = "Fibre Multimode à gradient d'indice"
    elif fiber_type == FiberType.STEP_INDEX:
        fiber_label_disp = "Fibre Multimode à saut d'indice"
    else:
        fiber_label_disp = "Fibre Monomode"
    fiber_color = "#4A90D9" if is_monomode else "#E8A838"

    fig_schema = go.Figure()

    # ── Background ──────────────────────────────────────────────────────────
    fig_schema.add_shape(type="rect", x0=0, x1=10, y0=0, y1=10,
                         fillcolor="#F0F4F8", line_width=0, layer="below")

    # ── Emitter block ────────────────────────────────────────────────────────
    fig_schema.add_shape(type="rect", x0=0.4, x1=2.2, y0=3.5, y1=6.5,
                         fillcolor="#1A6B3C", line=dict(color="#0D4A29", width=2))
    fig_schema.add_annotation(x=1.3, y=6.85, text="<b>ÉMETTEUR</b>",
                               showarrow=False, font=dict(size=12, color="#1A6B3C"))
    # Source type label
    src_label = source_type if source_type != "Manuel" else ("Laser (LD)" if is_monomode else "LED")
    fig_schema.add_annotation(x=1.3, y=5.15, text=f"<b>{src_label}</b>",
                               showarrow=False, font=dict(size=11, color="white"))
    # Lambda & Pe labels inside emitter
    fig_schema.add_annotation(x=1.3, y=4.5,
                               text=f"λ = {lambda_nm:.0f} nm",
                               showarrow=False, font=dict(size=10, color="#AAFFCC"))
    fig_schema.add_annotation(x=1.3, y=3.85,
                               text=f"Pe = {Pe_dbm:.1f} dBm",
                               showarrow=False, font=dict(size=10, color="#AAFFCC"))

    # ── Fiber (cable) ────────────────────────────────────────────────────────
    # Outer sheath
    fig_schema.add_shape(type="rect", x0=2.2, x1=7.8, y0=4.55, y1=5.45,
                         fillcolor="#888888", line=dict(color="#555", width=1.5))
    # Cladding
    fig_schema.add_shape(type="rect", x0=2.2, x1=7.8, y0=4.68, y1=5.32,
                         fillcolor="#CCCCEE", line_width=0)
    # Core
    core_h = 0.12 if is_monomode else 0.28
    fig_schema.add_shape(type="rect",
                         x0=2.2, x1=7.8,
                         y0=5.0 - core_h, y1=5.0 + core_h,
                         fillcolor=fiber_color, line_width=0)
    # Fiber label above
    fig_schema.add_annotation(x=5.0, y=5.75,
                               text=f"<b>{fiber_label_disp}</b> — L = {L_km:.1f} km   α = {alpha:.3f} dB/km",
                               showarrow=False, font=dict(size=11, color="#333"))
    fig_schema.add_annotation(x=5.0, y=4.25,
                               text="Chaîne optique de transmission",
                               showarrow=False, font=dict(size=10, color="#555"))

    # Connector markers
    for cx, label in [(2.6, f"C₁"), (7.4, f"C{Nc}")]:
        fig_schema.add_shape(type="rect", x0=cx-0.12, x1=cx+0.12, y0=4.4, y1=5.6,
                             fillcolor="#FFD700", line=dict(color="#AA8800", width=1.5))
        fig_schema.add_annotation(x=cx, y=4.05, text=label,
                                   showarrow=False, font=dict(size=9, color="#AA8800"))

    # Splice markers (small triangles as lines)
    n_splice_shown = min(Ns, 3)
    splice_positions = [3.5 + i * 1.0 for i in range(n_splice_shown)]
    for sx in splice_positions:
        fig_schema.add_shape(type="line", x0=sx, x1=sx, y0=4.55, y1=5.45,
                             line=dict(color="#FF6600", width=2, dash="dot"))
    if n_splice_shown > 0:
        fig_schema.add_annotation(x=splice_positions[0], y=5.75 + 0.45,
                                   text=f"Épissures (×{Ns})",
                                   showarrow=False, font=dict(size=9, color="#FF6600"))

    # ── Light beam arrow ─────────────────────────────────────────────────────
    fig_schema.add_annotation(
        x=2.15, y=5.0, ax=2.85, ay=5.0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowwidth=2, arrowcolor=fiber_color,
        text="", standoff=0,
    )

    # ── Receiver block ───────────────────────────────────────────────────────
    fig_schema.add_shape(type="rect", x0=7.8, x1=9.6, y0=3.5, y1=6.5,
                         fillcolor="#7B2D8B", line=dict(color="#4A1A5A", width=2))
    fig_schema.add_annotation(x=8.7, y=6.85, text="<b>RÉCEPTEUR</b>",
                               showarrow=False, font=dict(size=12, color="#7B2D8B"))
    fig_schema.add_annotation(x=8.7, y=5.15, text=f"<b>{detector}</b>",
                               showarrow=False, font=dict(size=11, color="white"))
    fig_schema.add_annotation(x=8.7, y=4.5,
                               text=f"Pr = {pb.Pr_dbm:.1f} dBm",
                               showarrow=False, font=dict(size=10, color="#DDAAFF"))
    margin_color = "#AAFFCC" if pb.is_feasible else "#FF8888"
    fig_schema.add_annotation(x=8.7, y=3.85,
                               text=f"Marge = {pb.margin_db:+.1f} dB",
                               showarrow=False, font=dict(size=10, color=margin_color))

    # ── Signal power arrow (emitter → fiber) ─────────────────────────────────
    fig_schema.add_annotation(
        x=7.85, y=5.0, ax=7.15, ay=5.0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowwidth=2, arrowcolor=fiber_color,
        text="", standoff=0,
    )

    # ── Feasibility badge ────────────────────────────────────────────────────
    badge_text = "✅  LIAISON OK" if pb.is_feasible else "❌  INSUFFISANT"
    badge_bg = "#D4EDDA" if pb.is_feasible else "#F8D7DA"
    badge_border = "#28A745" if pb.is_feasible else "#DC3545"
    fig_schema.add_shape(type="rect", x0=3.5, x1=6.5, y0=1.2, y1=2.2,
                         fillcolor=badge_bg, line=dict(color=badge_border, width=2))
    fig_schema.add_annotation(x=5.0, y=1.7, text=f"<b>{badge_text}</b>",
                               showarrow=False, font=dict(size=13, color=badge_border))

    # ── Delta-lambda label below emitter ─────────────────────────────────────
    fig_schema.add_annotation(x=1.3, y=2.9,
                               text=f"Δλ = {delta_lambda:.1f} nm",
                               showarrow=False, font=dict(size=10, color="#555"))
    # ── Bitrate label below receiver ─────────────────────────────────────────
    fig_schema.add_annotation(x=8.7, y=2.9,
                               text=f"Débit = {bitrate_GHz:.2f} GHz",
                               showarrow=False, font=dict(size=10, color="#555"))

    fig_schema.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0.5, 7.5]),
        plot_bgcolor="#F0F4F8",
        paper_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_schema, use_container_width=True)

    # Parameter summary below the schema
    col_e, col_f, col_r = st.columns(3)
    with col_e:
        st.markdown("**Émetteur**")
        st.markdown(f"- Source : {src_label}")
        st.markdown(f"- λ = {lambda_nm:.0f} nm")
        st.markdown(f"- Pe = {Pe_dbm:.1f} dBm")
        st.markdown(f"- Δλ = {delta_lambda:.1f} nm")
        st.markdown(f"- Coupleur laser-fibre : {coupler_laser_db:.1f} dB")
    with col_f:
        st.markdown(f"**{fiber_label_disp}**")
        st.markdown(f"- L = {L_km:.1f} km")
        st.markdown(f"- α = {alpha:.3f} dB/km")
        st.markdown(f"- Dch = {Dch:.1f} ps/nm/km")
        if bandwidth_distance_MHz_km > 0:
            st.markdown(f"- B·L = {bandwidth_distance_MHz_km:.1f} MHz·km")
        st.markdown(f"- {Nc} connecteurs × {loss_per_conn:.2f} dB")
        st.markdown(f"- {Ns} épissures × {loss_per_splice:.3f} dB")
        st.markdown(f"- Autres pertes = {other_losses_db:.2f} dB")
    with col_r:
        st.markdown("**Récepteur**")
        st.markdown(f"- Détecteur : {detector}")
        st.markdown(f"- Coupleur fibre-détecteur : {coupler_detect_db:.1f} dB")
        st.markdown(f"- Pr = {pb.Pr_dbm:.2f} dBm")
        st.markdown(f"- Sensibilité = {pb.sensitivity_dbm:.1f} dBm")
        st.markdown(f"- Marge = {pb.margin_db:+.2f} dB")

# ---- Tab 1: Power vs Distance ----
with tab1:
    fig1 = go.Figure()

    # Received power curve
    fig1.add_trace(go.Scatter(
        x=L_plot, y=Pr_arr,
        mode='lines', name='Puissance reçue Pr(L)',
        line=dict(color='royalblue', width=2),
    ))

    # Emitted power horizontal
    fig1.add_hline(y=Pe_dbm, line_dash='dash', line_color='green',
                   annotation_text=f"Pe = {Pe_dbm:.1f} dBm", annotation_position="top right")

    # Sensitivity threshold
    fig1.add_hline(y=pb.sensitivity_dbm, line_dash='dash', line_color='red',
                   annotation_text=f"Sensibilité = {pb.sensitivity_dbm:.1f} dBm",
                   annotation_position="bottom right")

    # Current L marker
    fig1.add_vline(x=L_km, line_dash='dot', line_color='orange',
                   annotation_text=f"L = {L_km} km", annotation_position="top left")

    # Shaded infeasible zone
    infeasible_mask = Pr_arr < pb.sensitivity_dbm
    if infeasible_mask.any():
        L_cross = np.interp(pb.sensitivity_dbm, Pr_arr[::-1], L_plot[::-1])
        fig1.add_vrect(x0=L_cross, x1=L_plot[-1],
                       fillcolor="red", opacity=0.08, layer="below", line_width=0,
                       annotation_text="Zone infaisable", annotation_position="top right")

    fig1.update_layout(
        title="Puissance reçue en fonction de la distance",
        xaxis_title="Distance L (km)",
        yaxis_title="Puissance (dBm)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig1, use_container_width=True)

# ---- Tab 2: Dispersion vs Distance ----
with tab2:
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Broadening on left axis
    fig2.add_trace(go.Scatter(
        x=L_plot, y=tau_arr,
        mode='lines', name='Δτ total (ps)',
        line=dict(color='royalblue', width=2),
    ), secondary_y=False)

    # Bandwidth on right axis — cap inf values for plotting
    bw_plot = np.clip(bw_arr, 0, min(1e4, float(np.nanmax(bw_arr[bw_arr != np.inf])) * 1.2) if np.any(bw_arr != np.inf) else 1e4)
    fig2.add_trace(go.Scatter(
        x=L_plot, y=bw_plot,
        mode='lines', name='Bande passante BP (GHz)',
        line=dict(color='firebrick', width=2),
    ), secondary_y=True)

    # BP target line
    fig2.add_hline(y=BP_target, line_dash='dash', line_color='firebrick',
                   annotation_text=f"BP cible = {BP_target} GHz",
                   annotation_position="bottom right", secondary_y=True)

    # Current L marker
    fig2.add_vline(x=L_km, line_dash='dot', line_color='orange',
                   annotation_text=f"L = {L_km} km", annotation_position="top left")

    fig2.update_yaxes(title_text="Élargissement Δτ (ps)", secondary_y=False, color='royalblue')
    fig2.update_yaxes(title_text="Bande passante BP (GHz)", secondary_y=True, color='firebrick')
    fig2.update_layout(
        title="Dispersion en fonction de la distance",
        xaxis_title="Distance L (km)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer formulas
# ---------------------------------------------------------------------------
with st.expander("📐 Formulaire — Rappel des équations"):
    st.markdown(r"""
**Bilan de puissance**
$$P_r = P_e - L_{couplage} - \alpha L - N_c L_c - N_s L_s - L_{autres} \quad \text{[dBm]}$$
$$\text{Marge} = P_r - P_{sensibilité}$$

**Catalogue exercice** : PIN = $-52$ dBm · PIIPN = $-64$ dBm

---

**Dispersion chromatique** : $\Delta\tau_{ch} = D_{ch} \cdot \Delta\lambda \cdot L$ [ps]

**Dispersion modale (saut d'indice)** : $\Delta\tau_{im} = \frac{n_1}{c} \cdot \Delta \cdot L$ [ps]

**Dispersion modale (gradient d'indice)** : $\Delta\tau_{im} = \frac{n_1}{8c} \cdot \Delta^2 \cdot L$ [ps]

**Dispersion totale** : $\Delta\tau = \sqrt{\Delta\tau_{ch}^2 + \Delta\tau_{im}^2}$

**Bande passante** : $BP = \frac{0.7}{\Delta\tau}$ [GHz]

**Produit bande passante-distance** : $BP \times L = \mathrm{constante}$, donc $BP(L)=\frac{B \cdot L}{L}$
""")
