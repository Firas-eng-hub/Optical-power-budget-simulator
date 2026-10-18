const $ = id => document.getElementById(id);
const form = $('parameters');
const fmt = (n, digits=2) => Number.isFinite(n) ? n.toFixed(digits) : '∞';
function config() {
  return Object.fromEntries([...form.querySelectorAll('input,select')].map(el => [el.id,el.type === 'number' ? Number(el.value) : el.value]));
}
function rows(id, data) { $(id).innerHTML = data.map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join(''); }
function chart(id, points, threshold, marker, color, unit, title) {
  const w=880,h=260,l=65,t=24,r=20,b=45;
  const maxX=points.at(-1)[0], ys=[...points.map(p=>p[1]),threshold,marker[1]];
  let minY=Math.min(...ys),maxY=Math.max(...ys);const pad=(maxY-minY||1)*0.12;minY-=pad;maxY+=pad;
  const x=v=>l+v/maxX*(w-l-r),y=v=>h-b-(v-minY)/(maxY-minY)*(h-t-b);
  let grid='';for(let i=0;i<=4;i++){const xv=maxX*i/4,yv=minY+(maxY-minY)*i/4;grid+=`<line x1="${l}" y1="${y(yv)}" x2="${w-r}" y2="${y(yv)}" stroke="#e8edf4"/><text x="${l-10}" y="${y(yv)+4}" text-anchor="end">${fmt(yv,unit==='GHz'?3:1)}</text><text x="${x(xv)}" y="${h-23}" text-anchor="middle">${fmt(xv,1)}</text>`;}
  $(id).innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}"><title>${title}</title><g font-family="Segoe UI,Arial" font-size="11" fill="#627187">${grid}<text x="${l}" y="13">${unit}</text><text x="${w-r}" y="${h-4}" text-anchor="end">Distance (km)</text></g><line x1="${l}" x2="${w-r}" y1="${y(threshold)}" y2="${y(threshold)}" stroke="#d45b56" stroke-dasharray="6 5"/><polyline points="${points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5"/><circle cx="${x(marker[0])}" cy="${y(marker[1])}" r="5" fill="${color}" stroke="white" stroke-width="2"/></svg>`;
}
function render() {
  const c=config();
  $('splicesField').hidden=c.spliceMode!=='manual';$('spacingField').hidden=c.spliceMode!=='spacing';$('reelField').hidden=c.spliceMode!=='reels';$('sensitivityField').hidden=c.detector!=='manual';$('wavelength').readOnly=c.wavelengthPreset!=='custom';
  if (!form.checkValidity()) { $('status').className='status bad';$('status').textContent='Veuillez renseigner des paramètres valides pour actualiser les résultats.';return; }
  const v=OpticalModel.simulate(c);
  $('status').className=`status ${v.feasible?'good':'bad'}`;
  $('status').textContent=v.feasible?'● Liaison compatible — puissance et bande passante suffisantes.':`● Liaison insuffisante — ${[v.margin<0?'puissance reçue sous le seuil':null,v.bandwidth<c.target?'bande passante inférieure à la cible':null].filter(Boolean).join(' ; ')}.`;
  $('received').textContent=fmt(v.received)+' dBm';$('threshold').textContent='Sensibilité : '+fmt(v.sensitivity,1)+' dBm';$('margin').textContent=(v.margin>=0?'+':'')+fmt(v.margin)+' dB';$('bandwidth').textContent=fmt(v.bandwidth,3)+' GHz';$('broadening').textContent='Δτ total : '+fmt(v.total,1)+' ps';$('maxLength').textContent=fmt(v.maxLength,1)+' km';$('limiting').textContent='Limite : '+v.limiting;
  $('chainMeta').textContent=c.wavelength+' nm · '+fmt(c.length,1)+' km';$('sourceLabel').textContent=c.source;$('injected').textContent=fmt(c.power-c.sourceCoupler)+' dBm injectés';$('fiberLabel').textContent=$('fiber').selectedOptions[0].textContent;$('events').textContent=c.connectors+' connecteurs · '+v.ns+' épissures';$('detectorLabel').textContent=c.detector==='manual'?'Manuel':c.detector;$('receivedLabel').textContent=fmt(v.received)+' dBm reçus';
  rows('losses',[['Puissance émise',fmt(c.power)+' dBm'],['Couplage source',fmt(c.sourceCoupler)+' dB'],['Atténuation fibre',fmt(v.fiberLoss)+' dB'],['Connecteurs',fmt(v.connectorLoss)+' dB'],['Épissures ('+v.ns+')',fmt(v.spliceLoss)+' dB'],['Couplage détecteur',fmt(c.detectorCoupler)+' dB'],['Autres pertes',fmt(c.otherLoss)+' dB'],['Pertes totales',fmt(v.totalLoss)+' dB'],['Puissance reçue',fmt(v.received)+' dBm']]);
  rows('dispersionTable',[['Δτ chromatique',fmt(v.chromatic)+' ps'],['Δτ modale',fmt(v.modal)+' ps'],['Δτ totale (RMS)',fmt(v.total)+' ps'],['BP modèle',fmt(v.modelBandwidth,4)+' GHz'],['BP catalogue',fmt(v.vendorBandwidth,4)+' GHz'],['BP cible',fmt(c.target,3)+' GHz'],['Limite puissance',fmt(v.powerLength,2)+' km'],['Limite dispersion estimée',fmt(v.dispersionLength,2)+' km']]);
  const end=Math.max(20,c.length*1.6,Math.min(Number.isFinite(v.maxLength)?v.maxLength*1.2:200,600));
  let powerPoints;
  if($('stepped').checked){const events=[];for(const [n,loss] of [[c.connectors,c.connectorLoss],[v.ns,c.spliceLoss]])for(let i=1;i<=n;i++)events.push([c.length*i/(n+1),loss]);events.sort((a,b)=>a[0]-b[0]);let loss=c.sourceCoupler+c.detectorCoupler+c.otherLoss;powerPoints=[[0,c.power-loss]];for(const [d,drop] of events){powerPoints.push([d,c.power-loss-c.alpha*d]);loss+=drop;powerPoints.push([d,c.power-loss-c.alpha*d]);}powerPoints.push([end,c.power-loss-c.alpha*end]);}
  else powerPoints=[[0,c.power-v.fixedLoss],[end,c.power-v.fixedLoss-c.alpha*end]];
  chart('powerChart',powerPoints,v.sensitivity,[c.length,v.received],'#2264db','dBm','Puissance reçue selon la distance');
  const bpPoints=Array.from({length:150},(_,i)=>{const length=0.1+(end-0.1)*i/149;const r=OpticalModel.simulate({...c,length});return[length,Math.min(Number.isFinite(r.bandwidth)?r.bandwidth:1e4,Math.max(c.target*10,Number.isFinite(v.bandwidth)?v.bandwidth*4:1))];});
  chart('dispersionChart',bpPoints,c.target,[c.length,Math.min(v.bandwidth,bpPoints[0][1])],'#8756d0','GHz','Bande passante selon la distance (valeurs élevées plafonnées pour la lisibilité)');
}
form.addEventListener('input',render);
form.addEventListener('change',event=>{const id=event.target.id;if(id==='source'){const laser=$('source').value==='LASER';$('power').value=laser?10:0;$('width').value=laser?2:40;$('sourceCoupler').value=laser?3:17;}if(id==='fiber')$('alpha').value={step:5,graded:3,single:0.5}[$('fiber').value];if(id==='wavelengthPreset'&&$('wavelengthPreset').value!=='custom'){$('wavelength').value=$('wavelengthPreset').value;$('dispersion').value={850:80,1300:0,1550:17}[$('wavelengthPreset').value];}render();});
$('stepped').addEventListener('change',render);$('reset').addEventListener('click',()=>{form.reset();$('stepped').checked=false;render();});form.addEventListener('submit',event=>event.preventDefault());render();
