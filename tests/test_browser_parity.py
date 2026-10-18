"""Compare the deployed browser model with the original Python calculations."""
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics.power_budget import LinkConfig, compute_power_budget
from physics.dispersion import DispersionConfig, FiberType, compute_dispersion
from physics.link_length import compute_max_length

cases, expected = [], []
fibers = {"step": (FiberType.STEP_INDEX, 10), "graded": (FiberType.GRADED_INDEX, 100), "single": (FiberType.SINGLE_MODE, 0)}
for fiber, (fiber_type, product) in fibers.items():
    for source in ("LED", "LASER"):
        for detector in ("PIN", "PIIPN", "manual"):
            for mode in ("manual", "spacing", "reels"):
                for dch in (0, -17, 80):
                    c = dict(fiber=fiber, source=source, detector=detector, spliceMode=mode,
                             length=12.3, alpha=0.5, power=10 if source == "LASER" else 0,
                             width=2 if source == "LASER" else 40, sourceCoupler=3 if source == "LASER" else 17,
                             detectorCoupler=1, otherLoss=0.7, connectors=2, connectorLoss=1,
                             splices=11, spliceLoss=0.3, spacing=200, reel=1, sensitivity=-40,
                             dispersion=dch, target=0.002)
                    ns = c["splices"] if mode == "manual" else max(0, int(c["length"] * 1000 / c["spacing"]) - 1) if mode == "spacing" else max(0, math.ceil(c["length"] / c["reel"]) - 1)
                    link = LinkConfig(c["power"], 1550, c["length"], c["alpha"], c["connectors"], c["connectorLoss"], ns, c["spliceLoss"], c["sourceCoupler"], c["detectorCoupler"], c["otherLoss"], 0, detector, c["target"], -40 if detector == "manual" else None)
                    disp = DispersionConfig(dch, c["width"], c["length"], fiber_type, 1.468, 0.01, product)
                    p, d, m = compute_power_budget(link), compute_dispersion(disp), compute_max_length(link, disp, c["target"])
                    cases.append(c)
                    expected.append(dict(ns=ns, received=p.Pr_dbm, margin=p.margin_db, totalLoss=p.A_total_db, total=d.delta_tau_total_ps, bandwidth=d.bandwidth_GHz, powerLength=m.L_power_km, dispersionLength=m.L_disp_km, maxLength=m.L_max_km))
script = "const {simulate}=require('./site/model.js');let s='';process.stdin.on('data',d=>s+=d);process.stdin.on('end',()=>console.log(JSON.stringify(JSON.parse(s).map(simulate),(_,v)=>v===Infinity?'Infinity':v)));"
result = subprocess.run(["node", "-e", script], input=json.dumps(cases), text=True, capture_output=True, cwd=ROOT, check=True)
for index, (actual, reference) in enumerate(zip(json.loads(result.stdout), expected, strict=True)):
    for key, value in reference.items():
        got = float(actual[key])
        assert math.isclose(got, value, rel_tol=1e-10, abs_tol=1e-10), (index, key, got, value)
print(f"PASS: {len(cases)} scenarios match Python power, dispersion and maximum-length models.")
