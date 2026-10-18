// Browser port of physics/{power_budget,dispersion,link_length}.py.
// Units: km, dB/dBm, ps, nm and GHz.
(function (root) {
  function simulate(c) {
    const ns = c.spliceMode === 'spacing' ? Math.max(0, Math.floor(c.length * 1000 / c.spacing) - 1)
      : c.spliceMode === 'reels' ? Math.max(0, Math.ceil(c.length / c.reel) - 1) : c.splices;
    const sensitivity = c.detector === 'manual' ? c.sensitivity : c.detector === 'PIN' ? -52 : -64;
    const fiberLoss = c.alpha * c.length, connectorLoss = c.connectors * c.connectorLoss, spliceLoss = ns * c.spliceLoss;
    const fixedLoss = connectorLoss + spliceLoss + c.sourceCoupler + c.detectorCoupler + c.otherLoss;
    const totalLoss = fixedLoss + fiberLoss, received = c.power - totalLoss, margin = received - sensitivity;
    const chromatic = Math.abs(c.dispersion) * c.width * c.length;
    const modalPerKm = c.fiber === 'single' ? 0 : c.fiber === 'graded' ? 1.468 / (8 * 3e5) * 0.01 ** 2 * 1e12 : 1.468 / 3e5 * 0.01 * 1e12;
    const modal = modalPerKm * c.length, total = Math.hypot(chromatic, modal);
    const vendorProduct = c.fiber === 'step' ? 10 : c.fiber === 'graded' ? 100 : 0;
    const modelBandwidth = total > 0 ? 700 / total : Infinity;
    const vendorBandwidth = vendorProduct > 0 ? vendorProduct / c.length / 1000 : Infinity;
    const bandwidth = Math.min(modelBandwidth, vendorBandwidth);
    const powerLength = c.alpha > 0 ? Math.max(0, (c.power - fixedLoss - sensitivity) / c.alpha) : Infinity;
    // Match the Python maximum-length estimate: chromatic + catalogue limits.
    const chromaticLength = Math.abs(c.dispersion) * c.width > 0 ? 700 / c.target / (Math.abs(c.dispersion) * c.width) : Infinity;
    const vendorLength = vendorProduct > 0 ? vendorProduct / (c.target * 1000) : Infinity;
    const dispersionLength = Math.min(chromaticLength, vendorLength);
    return {ns,sensitivity,fiberLoss,connectorLoss,spliceLoss,fixedLoss,totalLoss,received,margin,chromatic,modal,total,modelBandwidth,vendorBandwidth,bandwidth,powerLength,dispersionLength,maxLength:Math.min(powerLength,dispersionLength),limiting:powerLength <= dispersionLength ? 'Puissance' : 'Dispersion',feasible:margin >= 0 && bandwidth >= c.target};
  }
  root.OpticalModel = {simulate};
  if (typeof module !== 'undefined') module.exports = {simulate};
})(globalThis);
