/**
 * IAEA TRS-398 (Rev. 1, 2024) Photon Reference Dosimetry Application
 * High-Energy Photon Beams Engine & UI Controller
 * Extended for FFF (Flattening Filter-Free) Beams & Saveable Machine/Chamber Profiles
 */

// ==========================================================================
// 1. IAEA TRS-398 Rev. 1 (2024) Physics Database (Table 16, Table 10 & Table 4)
// ==========================================================================

const KQ_PHOTON_TPRS = [
  0.56, 0.59, 0.62, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82
];

// Chamber k_Q values from IAEA TRS-398 Rev. 1 (2024) Table 16 (p. 90-93)
const KQ_PHOTON_CHAMBERS = {
  "PTW 30013 Farmer (0.60 cm³, waterproof)": [
    1.0007, 0.9984, 0.9956, 0.9920, 0.9876, 0.9840, 0.9800, 0.9753, 0.9699, 0.9636, 0.9565, 0.9484
  ],
  "PTW 30010 Farmer (0.60 cm³, classic)": [
    1.0005, 0.9989, 0.9967, 0.9938, 0.9901, 0.9869, 0.9832, 0.9787, 0.9734, 0.9671, 0.9595, 0.9506
  ],
  "PTW 30011 Farmer (0.60 cm³)": [
    1.0005, 0.9989, 0.9969, 0.9942, 0.9906, 0.9875, 0.9838, 0.9793, 0.9739, 0.9674, 0.9595, 0.9501
  ],
  "PTW 30012 Farmer (0.60 cm³)": [
    1.0004, 0.9990, 0.9970, 0.9944, 0.9910, 0.9881, 0.9846, 0.9804, 0.9754, 0.9694, 0.9622, 0.9536
  ],
  "PTW 31010 Semiflex (0.125 cm³)": [
    1.0008, 0.9982, 0.9952, 0.9914, 0.9869, 0.9835, 0.9795, 0.9750, 0.9700, 0.9643, 0.9579, 0.9507
  ],
  "PTW 31013 Semiflex (0.30 cm³)": [
    1.0007, 0.9985, 0.9958, 0.9924, 0.9882, 0.9848, 0.9810, 0.9765, 0.9714, 0.9655, 0.9588, 0.9511
  ],
  "PTW 31021 Semiflex 3D (0.07 cm³)": [
    1.0007, 0.9984, 0.9957, 0.9925, 0.9886, 0.9856, 0.9823, 0.9786, 0.9744, 0.9697, 0.9645, 0.9587
  ],
  "IBA / Wellhöfer FC65-G (0.65 cm³)": [
    1.0004, 0.9990, 0.9972, 0.9946, 0.9912, 0.9882, 0.9846, 0.9802, 0.9748, 0.9683, 0.9603, 0.9507
  ],
  "IBA FC65-P Farmer (0.65 cm³)": [
    1.0005, 0.9988, 0.9966, 0.9936, 0.9897, 0.9865, 0.9826, 0.9780, 0.9725, 0.9660, 0.9583, 0.9491
  ],
  "IBA CC13 (0.13 cm³)": [
    1.0005, 0.9989, 0.9969, 0.9942, 0.9906, 0.9876, 0.9839, 0.9795, 0.9742, 0.9678, 0.9601, 0.9510
  ],
  "NE 2571 Farmer (0.60 cm³)": [
    1.0004, 0.9991, 0.9974, 0.9951, 0.9919, 0.9891, 0.9856, 0.9813, 0.9761, 0.9697, 0.9618, 0.9522
  ],
  "Exradin A12 Farmer (0.64 cm³)": [
    1.0004, 0.9991, 0.9973, 0.9948, 0.9915, 0.9887, 0.9852, 0.9809, 0.9756, 0.9693, 0.9615, 0.9521
  ],
  "Exradin A19 Classic Farmer (0.62 cm³)": [
    1.0005, 0.9989, 0.9968, 0.9940, 0.9904, 0.9873, 0.9836, 0.9792, 0.9738, 0.9675, 0.9599, 0.9509
  ]
};

// Active cavity length in mm for volume averaging (k_vol) evaluation in FFF beams (Table 4)
const CHAMBER_LENGTHS_MM = {
  "PTW 30013": 23.0,
  "PTW 30010": 23.0,
  "PTW 30011": 23.0,
  "PTW 30012": 23.0,
  "PTW 31010": 6.5,
  "PTW 31013": 16.3,
  "PTW 31021": 4.8,
  "FC65-G": 23.0,
  "FC65-P": 23.0,
  "CC13": 5.8,
  "2571": 24.1,
  "A12": 21.6,
  "A19": 21.6
};

// IAEA TRS-398 Rev. 1 Table 10: Coefficients for calculation of k_s for pulsed beams
const KS_TABLE_9_FITS = {
  2.0: { a0: 2.337, a1: -3.636, a2: 2.299 },
  2.5: { a0: 1.474, a1: -1.587, a2: 1.114 },
  3.0: { a0: 1.198, a1: -0.875, a2: 0.677 },
  3.5: { a0: 1.080, a1: -0.542, a2: 0.463 },
  4.0: { a0: 1.022, a1: -0.363, a2: 0.341 },
  5.0: { a0: 0.975, a1: -0.188, a2: 0.214 }
};

// Standard Preloaded Clinical Presets Library
const DEFAULT_PROFILES = [
  {
    id: "user_6mv_preset",
    name: "User Setup (6 MV W, PTW 30013, SSD 100, z=10cm)",
    linacId: "TrueBeam #1 (Clinical Room 1)",
    physicistName: "Medical Physics Specialist",
    nominalEnergy: "6MV",
    isFff: false,
    setupType: "SSD",
    fieldSize: "10 × 10",
    bqiTpr: 0.670,
    zrefDepth: 10.0,
    dmaxDepth: 1.5,
    pddZref: 67.0,
    muDelivered: 100.0,
    targetDosePerMu: 1.000,
    doseRate: 600,
    kvol: 1.0000,
    chamberKey: "PTW 30013 Farmer (0.60 cm³, waterproof)",
    chamberSn: "PTW-30013-10482",
    ndwValue: 53.85,
    ndwUnit: "mGy/nC",
    kelec: 1.0000,
    calLab: "PTW SSDL (2025-06-12)",
    refTemp: 20.0,
    refPress: 101.325,
    tempInput: 20.0,
    tempUnit: "C",
    pressInput: 101.325,
    pressUnit: "kPa",
    mPlus: 12.35,
    mMinus: 12.35,
    v1Voltage: 300,
    m1Reading: 12.35,
    v2Voltage: 100,
    m2Reading: 12.28,
    readings: [12.34, 12.35, 12.35]
  },
  {
    id: "tb_6fff",
    name: "Varian TrueBeam - 6 MV FFF (1400 MU/min)",
    linacId: "Varian TrueBeam #1",
    physicistName: "Medical Physics Specialist",
    nominalEnergy: "6MV",
    isFff: true,
    setupType: "SSD",
    fieldSize: "10 × 10",
    bqiTpr: 0.635,
    zrefDepth: 10.0,
    dmaxDepth: 1.5,
    pddZref: 63.5,
    muDelivered: 100.0,
    targetDosePerMu: 1.000,
    doseRate: 1400,
    kvol: 1.0015,
    chamberKey: "PTW 30013 Farmer (0.60 cm³, waterproof)",
    chamberSn: "PTW-30013-10482",
    ndwValue: 53.85,
    ndwUnit: "mGy/nC",
    kelec: 1.0000,
    calLab: "PTW SSDL (2025-06-12)",
    refTemp: 20.0,
    refPress: 101.325,
    tempInput: 20.0,
    tempUnit: "C",
    pressInput: 101.325,
    pressUnit: "kPa",
    mPlus: 12.45,
    mMinus: 12.45,
    v1Voltage: 300,
    m1Reading: 12.45,
    v2Voltage: 100,
    m2Reading: 12.32,
    readings: [12.44, 12.45, 12.45]
  },
  {
    id: "tb_10w",
    name: "Varian TrueBeam - 10 MV (Flattened)",
    linacId: "Varian TrueBeam #1",
    physicistName: "Medical Physics Specialist",
    nominalEnergy: "10MV",
    isFff: false,
    setupType: "SSD",
    fieldSize: "10 × 10",
    bqiTpr: 0.735,
    zrefDepth: 10.0,
    dmaxDepth: 2.4,
    pddZref: 73.5,
    muDelivered: 100.0,
    targetDosePerMu: 1.000,
    doseRate: 600,
    kvol: 1.0000,
    chamberKey: "PTW 30013 Farmer (0.60 cm³, waterproof)",
    chamberSn: "PTW-30013-10482",
    ndwValue: 53.85,
    ndwUnit: "mGy/nC",
    kelec: 1.0000,
    calLab: "PTW SSDL (2025-06-12)",
    refTemp: 20.0,
    refPress: 101.325,
    tempInput: 20.0,
    tempUnit: "C",
    pressInput: 101.325,
    pressUnit: "kPa",
    mPlus: 13.50,
    mMinus: 13.50,
    v1Voltage: 300,
    m1Reading: 13.50,
    v2Voltage: 100,
    m2Reading: 13.43,
    readings: [13.49, 13.50, 13.50]
  },
  {
    id: "tb_10fff",
    name: "Varian TrueBeam - 10 MV FFF (2400 MU/min)",
    linacId: "Varian TrueBeam #1",
    physicistName: "Medical Physics Specialist",
    nominalEnergy: "10MV",
    isFff: true,
    setupType: "SSD",
    fieldSize: "10 × 10",
    bqiTpr: 0.705,
    zrefDepth: 10.0,
    dmaxDepth: 2.2,
    pddZref: 70.5,
    muDelivered: 100.0,
    targetDosePerMu: 1.000,
    doseRate: 2400,
    kvol: 1.0034,
    chamberKey: "PTW 30013 Farmer (0.60 cm³, waterproof)",
    chamberSn: "PTW-30013-10482",
    ndwValue: 53.85,
    ndwUnit: "mGy/nC",
    kelec: 1.0000,
    calLab: "PTW SSDL (2025-06-12)",
    refTemp: 20.0,
    refPress: 101.325,
    tempInput: 20.0,
    tempUnit: "C",
    pressInput: 101.325,
    pressUnit: "kPa",
    mPlus: 13.60,
    mMinus: 13.60,
    v1Voltage: 300,
    m1Reading: 13.60,
    v2Voltage: 100,
    m2Reading: 13.42,
    readings: [13.59, 13.60, 13.61]
  },
  {
    id: "versa_6fff",
    name: "Elekta Versa HD - 6 MV FFF (1400 MU/min)",
    linacId: "Elekta Versa HD",
    physicistName: "Medical Physics Specialist",
    nominalEnergy: "6MV",
    isFff: true,
    setupType: "SSD",
    fieldSize: "10 × 10",
    bqiTpr: 0.638,
    zrefDepth: 10.0,
    dmaxDepth: 1.5,
    pddZref: 63.8,
    muDelivered: 100.0,
    targetDosePerMu: 1.000,
    doseRate: 1400,
    kvol: 1.0016,
    chamberKey: "PTW 30013 Farmer (0.60 cm³, waterproof)",
    chamberSn: "PTW-30013-10482",
    ndwValue: 53.85,
    ndwUnit: "mGy/nC",
    kelec: 1.0000,
    calLab: "PTW SSDL (2025-06-12)",
    refTemp: 20.0,
    refPress: 101.325,
    tempInput: 20.0,
    tempUnit: "C",
    pressInput: 101.325,
    pressUnit: "kPa",
    mPlus: 12.40,
    mMinus: 12.40,
    v1Voltage: 300,
    m1Reading: 12.40,
    v2Voltage: 100,
    m2Reading: 12.27,
    readings: [12.39, 12.40, 12.40]
  }
];

// Profile storage
let savedProfiles = [];
let activeProfileId = "user_6mv_preset";

// Live state
let state = { ...DEFAULT_PROFILES[0] };

// ==========================================================================
// 2. Physics & Calculation Functions
// ==========================================================================

function getChamberLength(chamberKey) {
  for (const [key, len] of Object.entries(CHAMBER_LENGTHS_MM)) {
    if (chamberKey.includes(key)) return len;
  }
  return 23.0; // Default Farmer length
}

// IAEA TRS-398 Rev. 1 (2024) Section 4.4.6, Eq. (22) & Table 11:
// k_vol = 1 + (0.0062 * TPR20,10 - 0.0036) * (100 / SDD)^2 * L^2
function estimateKvol(chamberKey, isFff, energy) {
  if (!isFff) return 1.0000;
  const lenCm = getChamberLength(chamberKey) / 10.0;
  const tpr = (state && state.bqiTpr) ? state.bqiTpr : (energy && energy.includes("10") ? 0.705 : 0.635);
  const sdd = (state && state.setupType === "SAD") ? 100.0 : (100.0 + ((state && state.zrefDepth) || 10.0));
  const kvol = 1.0 + (0.0062 * tpr - 0.0036) * Math.pow(100.0 / sdd, 2) * Math.pow(lenCm, 2);
  return parseFloat(Math.max(1.0, kvol).toFixed(4));
}

function interpolateKq(chamberKey, tpr) {
  const kqs = KQ_PHOTON_CHAMBERS[chamberKey] || KQ_PHOTON_CHAMBERS["PTW 30013 Farmer (0.60 cm³, waterproof)"];
  const tprs = KQ_PHOTON_TPRS;

  if (tpr <= tprs[0]) return kqs[0];
  if (tpr >= tprs[tprs.length - 1]) return kqs[kqs.length - 1];

  for (let i = 0; i < tprs.length - 1; i++) {
    if (tpr >= tprs[i] && tpr <= tprs[i + 1]) {
      const slope = (kqs[i + 1] - kqs[i]) / (tprs[i + 1] - tprs[i]);
      return kqs[i] + slope * (tpr - tprs[i]);
    }
  }
  return kqs[0];
}

function getTempCelsius(val, unit) {
  if (unit === "F") return (val - 32) * (5 / 9);
  return val;
}

function getPressKpa(val, unit) {
  if (unit === "hPa") return val * 0.1;
  if (unit === "mmHg") return val * (101.325 / 760.0);
  return val;
}

function getNdwGyPerNc(val, unit) {
  if (unit === "mGy/nC") return val / 1000.0;
  if (unit === "cGy/nC") return val / 100.0;
  if (unit === "Gy/rdg") return val;
  return val; // Gy/nC
}

// IAEA TRS-398 Rev. 1 (2024) Section 4.4.3, Eq. (10) (273.15 K)
function calculateKtp(tempC, pressKpa, refTempC, refPressKpa) {
  return ((273.15 + tempC) / (273.15 + refTempC)) * (refPressKpa / pressKpa);
}

function calculateKpol(mUser, mOpp) {
  if (Math.abs(mUser) < 1e-9) return 1.0;
  return (Math.abs(mUser) + Math.abs(mOpp)) / (2.0 * Math.abs(mUser));
}

function calculateKs(v1, v2, m1, m2) {
  if (Math.abs(v2) < 1e-9 || Math.abs(m2) < 1e-9) return { ks: 1.0, method: "Invalid input", ratioV: 1.0, ratioM: 1.0 };
  const ratioV = Math.abs(v1 / v2);
  const ratioM = m1 / m2;

  let bestMatch = null;
  let minDiff = 1e9;
  for (const stdRatio of Object.keys(KS_TABLE_9_FITS).map(Number)) {
    const diff = Math.abs(ratioV - stdRatio);
    if (diff < minDiff) {
      minDiff = diff;
      bestMatch = stdRatio;
    }
  }

  if (minDiff <= 0.08 && bestMatch !== null) {
    const coeffs = KS_TABLE_9_FITS[bestMatch];
    const ks = coeffs.a0 + coeffs.a1 * ratioM + coeffs.a2 * Math.pow(ratioM, 2);
    return {
      ks,
      method: `TRS-398 Rev. 1 Table 10 quadratic fit (V₁/V₂ = ${bestMatch.toFixed(1)}: a₀=${coeffs.a0}, a₁=${coeffs.a1}, a₂=${coeffs.a2})`,
      ratioV,
      ratioM
    };
  } else {
    // Linear approximation
    const denom = ratioV - ratioM;
    const ks = Math.abs(denom) > 1e-6 ? (ratioV - 1.0) / denom : 1.0;
    return {
      ks,
      method: `TRS-398 Rev. 1 Eq. (14) linear formula ((V₁/V₂ - 1) / (V₁/V₂ - M₁/M₂))`,
      ratioV,
      ratioM
    };
  }
}

function calculateDosePerPulse(doseRateMuMin, prfHz = 360) {
  if (prfHz <= 0) return 0;
  // (DoseRate in cGy/min * 10 mGy/cGy) / (60 s/min * PRF Hz)
  return (doseRateMuMin * 10.0) / (60.0 * prfHz);
}

function runFullCalculation() {
  const tempC = getTempCelsius(state.tempInput, state.tempUnit);
  const pressKpa = getPressKpa(state.pressInput, state.pressUnit);
  const refTempC = state.refTemp;
  const refPressKpa = state.refPress;

  // 1. Air density correction k_TP
  const ktp = calculateKtp(tempC, pressKpa, refTempC, refPressKpa);

  // 2. Polarity correction k_pol
  const kpol = calculateKpol(state.mPlus, state.mMinus);

  // 3. Ion recombination k_s
  const ksResult = calculateKs(state.v1Voltage, state.v2Voltage, state.m1Reading, state.m2Reading);
  const ks = ksResult.ks;

  // 4. Electrometer correction k_elec
  const kelec = state.kelec;

  // Total influence correction factor
  const ktot = ktp * kpol * ks * kelec;

  // 5. Electrometer Raw Readings Statistics
  const validReadings = state.readings.filter(r => !isNaN(r) && r > 0);
  const n = validReadings.length;
  const mRawAvg = n > 0 ? validReadings.reduce((a, b) => a + b, 0) / n : 0;
  let stdDev = 0;
  if (n > 1) {
    const sumSq = validReadings.reduce((acc, val) => acc + Math.pow(val - mRawAvg, 2), 0);
    stdDev = Math.sqrt(sumSq / (n - 1));
  }
  const cvPct = mRawAvg > 0 ? (stdDev / mRawAvg) * 100.0 : 0;

  // 6. Corrected reading M_Q
  const mq = mRawAvg * ktot;

  // 7. Beam Quality Correction Factor k_Q
  const kq = interpolateKq(state.chamberKey, state.bqiTpr);

  // 8. Volume Averaging Correction (k_vol) for FFF beams
  const kvol = state.isFff ? (state.kvol || estimateKvol(state.chamberKey, true, state.nominalEnergy)) : 1.0000;

  // 9. Calibration factor N_D,w in Gy/nC
  const ndwGyPerNc = getNdwGyPerNc(state.ndwValue, state.ndwUnit);

  // 10. Absorbed Dose to Water at Reference Depth z_ref = 10 cm (Gy and cGy)
  // TRS-398 with FFF extension: D_w,Q(z_ref) = M_Q * N_D,w,Q0 * k_Q * k_vol
  const dWzrefGy = mq * ndwGyPerNc * kq * kvol;
  const dWzrefCgy = dWzrefGy * 100.0;
  const dosePerMuZrefCgy = state.muDelivered > 0 ? dWzrefCgy / state.muDelivered : 0;

  // 11. Absorbed Dose at d_max using PDD(z_ref)
  // TRS-398 Eq (6.14): D_w,Q(z_max) = D_w,Q(z_ref) * 100 / PDD(z_ref)
  const pdd = state.pddZref;
  const pddFactor = pdd > 0 ? 100.0 / pdd : 1.0;
  const dWzmaxGy = dWzrefGy * pddFactor;
  const dWzmaxCgy = dWzmaxGy * 100.0;
  const dosePerMuDmaxCgy = state.muDelivered > 0 ? dWzmaxCgy / state.muDelivered : 0;

  // 12. Comparison with nominal target
  const targetCgyPerMu = state.targetDosePerMu;
  const diffPct = targetCgyPerMu > 0 ? ((dosePerMuDmaxCgy - targetCgyPerMu) / targetCgyPerMu) * 100.0 : 0;

  // Recommended machine trim/adjustment
  const trimPct = -diffPct / (1.0 + diffPct / 100.0);

  // Dose per pulse (DPP)
  const dppMgy = calculateDosePerPulse(state.doseRate || 600, 360);

  return {
    tempC,
    pressKpa,
    ktp,
    kpol,
    ks,
    ksResult,
    kelec,
    ktot,
    n,
    mRawAvg,
    stdDev,
    cvPct,
    mq,
    kq,
    kvol,
    dppMgy,
    ndwGyPerNc,
    dWzrefGy,
    dWzrefCgy,
    dosePerMuZrefCgy,
    pddFactor,
    dWzmaxGy,
    dWzmaxCgy,
    dosePerMuDmaxCgy,
    targetCgyPerMu,
    diffPct,
    trimPct
  };
}

// ==========================================================================
// 3. UI Update & DOM Rendering
// ==========================================================================

function updateUI() {
  const calc = runFullCalculation();

  // Top Ribbon Badges
  document.getElementById("badgeEnergy").textContent = `${state.nominalEnergy} ${state.isFff ? "FFF" : "Photons"}`;
  
  const fffBadge = document.getElementById("badgeFffMode");
  fffBadge.textContent = state.isFff ? "⚡ Flattening Filter-Free (FFF)" : "Conventional (W)";
  fffBadge.style.color = state.isFff ? "var(--fff-amber)" : "var(--brand-primary)";

  document.getElementById("badgeSetup").innerHTML = `${state.setupType} = 100 cm | z<sub>ref</sub> = ${state.zrefDepth.toFixed(1)} cm`;
  document.getElementById("badgeDmax").innerHTML = `d<sub>max</sub> = ${state.dmaxDepth.toFixed(1)} cm`;
  document.getElementById("badgeBQI").textContent = state.bqiTpr.toFixed(3);
  document.getElementById("badgeChamber").textContent = state.chamberKey.split("(")[0].trim();

  // FFF Header Tag & Panel Visibility
  const fffHeaderBadge = document.getElementById("fffHeaderBadge");
  const fffParamsPanel = document.getElementById("fffParamsPanel");
  const mathStepKvol = document.getElementById("mathStepKvol");
  const btnModeFlattened = document.getElementById("btnModeFlattened");
  const btnModeFFF = document.getElementById("btnModeFFF");
  const resultHeaderMode = document.getElementById("resultHeaderMode");

  if (state.isFff) {
    fffHeaderBadge.textContent = "⚡ FFF MODE ACTIVE";
    fffHeaderBadge.className = "tag tag-amber";
    fffParamsPanel.classList.remove("hidden");
    mathStepKvol.classList.remove("hidden");
    btnModeFlattened.classList.remove("active");
    btnModeFFF.classList.add("active");
    resultHeaderMode.textContent = `IAEA TRS-398 DETERMINATION RESULT (${state.nominalEnergy} FFF)`;
  } else {
    fffHeaderBadge.textContent = "Conventional W";
    fffHeaderBadge.className = "tag tag-gray";
    fffParamsPanel.classList.add("hidden");
    btnModeFlattened.classList.add("active");
    btnModeFFF.classList.remove("active");
    resultHeaderMode.textContent = `IAEA TRS-398 DETERMINATION RESULT (CONVENTIONAL W)`;
  }

  // FFF Dose-per-pulse text & helpers
  document.getElementById("dppValueText").textContent = `${calc.dppMgy.toFixed(3)} mGy/pulse`;
  const chamberLen = getChamberLength(state.chamberKey);
  document.getElementById("kvolHelper").textContent = `${state.chamberKey.split(" ")[1]} (cavity ${chamberLen} mm) ~ ${estimateKvol(state.chamberKey, true, state.nominalEnergy)}`;

  // Recombination high DPP warning badge for FFF
  const fffKsBadge = document.getElementById("fffKsWarningBadge");
  if (state.isFff && calc.ks > 1.015) {
    fffKsBadge.classList.remove("hidden");
    fffKsBadge.textContent = calc.ks > 1.03 ? "High FFF ks (Verify)" : "Normal FFF DPP";
  } else {
    fffKsBadge.classList.add("hidden");
  }

  // Status Badge in Ribbon & Main Result
  const absDiff = Math.abs(calc.diffPct);
  const badgeOutput = document.getElementById("badgeOutputStatus");
  const badgeOutputText = document.getElementById("badgeOutputText");
  const toleranceBadge = document.getElementById("toleranceBadge");
  const toleranceBadgeText = document.getElementById("toleranceBadgeText");

  toleranceBadge.classList.remove("badge-pass", "badge-warn", "badge-fail");

  if (absDiff <= 1.0) {
    badgeOutput.style.backgroundColor = "var(--status-pass-bg)";
    badgeOutput.style.color = "var(--status-pass-text)";
    badgeOutputText.textContent = `Within Tol (Δ = ${calc.diffPct >= 0 ? "+" : ""}${calc.diffPct.toFixed(2)}%)`;
    toleranceBadge.classList.add("badge-pass");
    toleranceBadgeText.textContent = `IN CLINICAL TOLERANCE (PASS)`;
  } else if (absDiff <= 2.0) {
    badgeOutput.style.backgroundColor = "var(--status-warn-bg)";
    badgeOutput.style.color = "var(--status-warn-text)";
    badgeOutputText.textContent = `Action Level (Δ = ${calc.diffPct >= 0 ? "+" : ""}${calc.diffPct.toFixed(2)}%)`;
    toleranceBadge.classList.add("badge-warn");
    toleranceBadgeText.textContent = `ACTION LEVEL WARNING (±1% - ±2%)`;
  } else {
    badgeOutput.style.backgroundColor = "var(--status-fail-bg)";
    badgeOutput.style.color = "var(--status-fail-text)";
    badgeOutputText.textContent = `Out of Tol (Δ = ${calc.diffPct >= 0 ? "+" : ""}${calc.diffPct.toFixed(2)}%)`;
    toleranceBadge.classList.add("badge-fail");
    toleranceBadgeText.textContent = `OUT OF TOLERANCE (RECALIBRATE LINAC)`;
  }

  // N_D,w conversion helper
  document.getElementById("ndwConvertedHelper").textContent = `= ${calc.ndwGyPerNc.toFixed(6)} Gy/nC`;

  // Factor Pills & Formulas
  document.getElementById("pillKtp").textContent = `k_TP = ${calc.ktp.toFixed(4)}`;
  document.getElementById("mathKtp").innerHTML = 
    `k<sub>TP</sub> = ((273.15 + ${calc.tempC.toFixed(1)})/(273.15 + ${state.refTemp.toFixed(1)})) × (${state.refPress.toFixed(3)} / ${calc.pressKpa.toFixed(3)}) = <strong>${calc.ktp.toFixed(4)}</strong>`;

  document.getElementById("pillKpol").textContent = `k_pol = ${calc.kpol.toFixed(4)}`;
  document.getElementById("mathKpol").innerHTML = 
    `k<sub>pol</sub> = (|${state.mPlus.toFixed(2)}| + |${state.mMinus.toFixed(2)}|) / (2 × ${state.mPlus.toFixed(2)}) = <strong>${calc.kpol.toFixed(4)}</strong>`;

  document.getElementById("pillKs").textContent = `k_s = ${calc.ks.toFixed(4)}`;
  document.getElementById("recombMethodText").textContent = `Method: ${calc.ksResult.method}`;
  document.getElementById("recombRatioText").textContent = `M₁/M₂ = ${calc.ksResult.ratioM.toFixed(4)}`;

  // Readings Stats
  document.getElementById("meanRawText").textContent = `${calc.mRawAvg.toFixed(3)} nC`;
  document.getElementById("sdRawText").textContent = `${calc.stdDev.toFixed(4)} nC`;
  const cvEl = document.getElementById("cvRawText");
  cvEl.textContent = `${calc.cvPct.toFixed(2)}%`;
  cvEl.className = calc.cvPct <= 0.2 ? "badge-cv-pass" : "text-deviation";

  // KPIs
  document.getElementById("kpiDoseDmaxCgy").textContent = calc.dWzmaxCgy.toFixed(2);
  document.getElementById("kpiDoseDmaxGy").textContent = `${calc.dWzmaxGy.toFixed(4)} Gy for ${state.muDelivered.toFixed(0)} MU`;
  document.getElementById("kpiDosePerMu").textContent = calc.dosePerMuDmaxCgy.toFixed(4);
  document.getElementById("kpiTargetComparison").textContent = `Target: ${calc.targetCgyPerMu.toFixed(4)} cGy/MU`;

  const diffEl = document.getElementById("kpiDeviation");
  diffEl.textContent = `${calc.diffPct >= 0 ? "+" : ""}${calc.diffPct.toFixed(2)}%`;
  diffEl.style.color = absDiff <= 1.0 ? "var(--status-pass-text)" : (absDiff <= 2.0 ? "var(--status-warn-text)" : "var(--status-fail-text)");

  const adjEl = document.getElementById("kpiAdjustment");
  adjEl.textContent = `${calc.trimPct >= 0 ? "+" : ""}${calc.trimPct.toFixed(2)}%`;
  adjEl.style.color = Math.abs(calc.trimPct) <= 1.0 ? "var(--status-pass-text)" : "var(--brand-primary)";

  // Secondary depth values
  document.getElementById("secDoseZref").textContent = `${calc.dWzrefCgy.toFixed(2)} cGy (${calc.dWzrefGy.toFixed(4)} Gy)`;
  document.getElementById("secDosePerMuZref").textContent = `${calc.dosePerMuZrefCgy.toFixed(4)} cGy/MU`;
  document.getElementById("secPddRatio").textContent = `100 / ${state.pddZref.toFixed(1)} = ${calc.pddFactor.toFixed(4)}`;

  // Step-by-Step Mathematical Derivations
  document.getElementById("calcSubMq").innerHTML = 
    `M<sub>Q</sub> = ${calc.mRawAvg.toFixed(3)} nC × ${calc.ktp.toFixed(4)} × ${calc.kpol.toFixed(4)} × ${calc.ks.toFixed(4)} × ${calc.kelec.toFixed(4)} = <strong>${calc.mq.toFixed(4)} nC</strong> (k<sub>tot</sub> = ${calc.ktot.toFixed(4)})`;

  document.getElementById("calcSubKq").innerHTML = 
    `k<sub>Q</sub> for ${state.chamberKey.split("(")[0].trim()} at TPR<sub>20,10</sub> = ${state.bqiTpr.toFixed(3)} = <strong>${calc.kq.toFixed(4)}</strong>`;

  if (state.isFff) {
    document.getElementById("calcEqDoseZref").innerHTML = `D<sub>w,Q</sub>(z<sub>ref</sub>) = M<sub>Q</sub> × N<sub>D,w,Q0</sub> × k<sub>Q</sub> × k<sub>vol</sub>`;
    document.getElementById("calcSubDoseZref").innerHTML = 
      `D<sub>w,Q</sub>(${state.zrefDepth.toFixed(1)} cm) = ${calc.mq.toFixed(4)} nC × ${calc.ndwGyPerNc.toFixed(6)} Gy/nC × ${calc.kq.toFixed(4)} × ${calc.kvol.toFixed(4)} = <strong>${calc.dWzrefGy.toFixed(4)} Gy (${calc.dWzrefCgy.toFixed(2)} cGy)</strong>`;
  } else {
    document.getElementById("calcEqDoseZref").innerHTML = `D<sub>w,Q</sub>(z<sub>ref</sub>) = M<sub>Q</sub> × N<sub>D,w,Q0</sub> × k<sub>Q</sub>`;
    document.getElementById("calcSubDoseZref").innerHTML = 
      `D<sub>w,Q</sub>(${state.zrefDepth.toFixed(1)} cm) = ${calc.mq.toFixed(4)} nC × ${calc.ndwGyPerNc.toFixed(6)} Gy/nC × ${calc.kq.toFixed(4)} = <strong>${calc.dWzrefGy.toFixed(4)} Gy (${calc.dWzrefCgy.toFixed(2)} cGy)</strong>`;
  }

  document.getElementById("calcSubDoseDmax").innerHTML = 
    `D<sub>w,Q</sub>(${state.dmaxDepth.toFixed(1)} cm) = ${calc.dWzrefGy.toFixed(4)} Gy × (100 / ${state.pddZref.toFixed(1)}) = <strong>${calc.dWzmaxGy.toFixed(4)} Gy (${calc.dWzmaxCgy.toFixed(2)} cGy)</strong>`;

  // Update SVG charts & schematic
  renderSchematic(state.isFff);
  renderKqChart(calc.kq);
  updateTableHighlight();
}

// ==========================================================================
// 4. Physical Geometry SVG Schematic (Flattened vs Peaked FFF)
// ==========================================================================

function renderSchematic(isFff) {
  const svg = document.getElementById("schematicSvg");
  const beamFill = isFff ? "url(#beamGradientFff)" : "url(#beamGradient)";
  const strokeColor = isFff ? "#f59e0b" : "#38bdf8";

  let html = `
    <defs>
      <linearGradient id="beamGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
        <stop offset="35%" stop-color="#38bdf8" stop-opacity="0.20"/>
        <stop offset="100%" stop-color="#0284c7" stop-opacity="0.08"/>
      </linearGradient>
      <linearGradient id="beamGradientFff" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.45"/>
        <stop offset="35%" stop-color="#f59e0b" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#d97706" stop-opacity="0.10"/>
      </linearGradient>
      <linearGradient id="waterGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#0369a1" stop-opacity="0.45"/>
      </linearGradient>
    </defs>

    <!-- Radiation Source -->
    <circle cx="270" cy="20" r="6" fill="${strokeColor}"/>
    <text x="270" y="12" text-anchor="middle" class="svg-text-small" fill="${strokeColor}" font-weight="bold">
      ${isFff ? "⚡ Target (No Flattening Filter)" : "Radiation Target + Flattening Filter"}
    </text>

    <!-- Radiation Beam Cone -->
    <polygon points="270,22 170,120 370,120" fill="${beamFill}" stroke="${strokeColor}" stroke-width="1.5" stroke-dasharray="4,2"/>
    <polygon points="170,120 120,265 420,265 370,120" fill="${beamFill}" stroke="${strokeColor}" stroke-width="1.5" stroke-dasharray="4,2"/>

    <!-- SSD 100 Dimension Line -->
    <line x1="430" y1="20" x2="430" y2="120" stroke="#94a3b8" stroke-width="1.5"/>
    <line x1="424" y1="20" x2="436" y2="20" stroke="#94a3b8" stroke-width="1.5"/>
    <line x1="424" y1="120" x2="436" y2="120" stroke="#94a3b8" stroke-width="1.5"/>
    <text x="445" y="75" class="svg-text" font-weight="bold">SSD = 100 cm</text>

    <!-- Water Phantom Tank -->
    <rect x="90" y="120" width="360" height="150" fill="url(#waterGradient)" stroke="#0284c7" stroke-width="2" rx="4"/>
    
    <!-- Water Surface Line -->
    <line x1="90" y1="120" x2="450" y2="120" stroke="#0284c7" stroke-width="2.5"/>
    <text x="100" y="114" class="svg-text-small" fill="#0284c7" font-weight="bold">Water Surface (10 × 10 cm²)</text>

    <!-- Depth marker to z_ref -->
    <line x1="68" y1="120" x2="68" y2="215" stroke="#94a3b8" stroke-width="1.5"/>
    <line x1="62" y1="120" x2="74" y2="120" stroke="#94a3b8" stroke-width="1.5"/>
    <line x1="62" y1="215" x2="74" y2="215" stroke="#94a3b8" stroke-width="1.5"/>
    <text x="24" y="172" class="svg-text" font-weight="bold">z_ref = 10 cm</text>

    <!-- d_max line at 1.5 cm -->
    <line x1="150" y1="135" x2="390" y2="135" stroke="#ef4444" stroke-width="2" stroke-dasharray="3,3"/>
    <text x="395" y="138" class="svg-text-small" fill="#ef4444" font-weight="bold">d_max = ${state.dmaxDepth.toFixed(1)} cm</text>

    <!-- z_ref line at 10 cm -->
    <line x1="120" y1="215" x2="420" y2="215" stroke="#10b981" stroke-width="1.5" stroke-dasharray="4,2"/>
    <text x="395" y="212" class="svg-text-small" fill="#10b981" font-weight="bold">z_ref = 10 cm (${state.pddZref.toFixed(1)}% PDD)</text>
  `;

  // Draw Lateral Beam Profile at Water Surface (Flat vs Peaked FFF)
  if (isFff) {
    // Peaked conical FFF profile
    html += `
      <path d="M 170 120 Q 270 96 370 120" fill="none" stroke="#f59e0b" stroke-width="3"/>
      <text x="270" y="90" text-anchor="middle" class="svg-text-small" fill="#d97706" font-weight="bold">⚡ Forward-Peaked FFF Lateral Profile</text>
    `;
  } else {
    // Uniform flat profile
    html += `
      <line x1="170" y1="120" x2="370" y2="120" stroke="#0284c7" stroke-width="3"/>
      <text x="270" y="106" text-anchor="middle" class="svg-text-small" fill="#0284c7" font-weight="bold">Flat Beam Profile</text>
    `;
  }

  // PTW Farmer Chamber Drawing at z_ref
  html += `
    <g transform="translate(270, 215)">
      <!-- Chamber Body Thimble -->
      <rect x="-35" y="-6" width="70" height="12" rx="6" fill="#f8fafc" stroke="#1e293b" stroke-width="1.5"/>
      <!-- Central Aluminum Electrode -->
      <line x1="-30" y1="0" x2="15" y2="0" stroke="#64748b" stroke-width="2"/>
      <!-- Reference Point crosshair -->
      <circle cx="0" cy="0" r="2.5" fill="#ef4444"/>
      <line x1="-5" y1="0" x2="5" y2="0" stroke="#ef4444" stroke-width="1"/>
      <line x1="0" y1="-5" x2="0" y2="5" stroke="#ef4444" stroke-width="1"/>
      <!-- Chamber Stem and Cable -->
      <rect x="35" y="-4" width="25" height="8" fill="#475569"/>
      <path d="M60 0 Q80 0 100 20" stroke="#334155" stroke-width="3" fill="none"/>
      <text x="0" y="24" text-anchor="middle" class="svg-text-small" fill="#1e293b" font-weight="bold">
        ${state.chamberKey.split(" ")[0]} ${state.chamberKey.split(" ")[1]} (Ref Point at Axis)
      </text>
    </g>
  `;

  svg.innerHTML = html;
}

// ==========================================================================
// 5. Readings Table Controller
// ==========================================================================

function renderReadingsTable() {
  const tbody = document.getElementById("readingsBody");
  tbody.innerHTML = "";

  const mean = state.readings.length > 0 ? state.readings.reduce((a, b) => a + b, 0) / state.readings.length : 0;

  state.readings.forEach((reading, index) => {
    const tr = document.createElement("tr");

    const devPct = mean > 0 ? ((reading - mean) / mean) * 100.0 : 0;
    const devClass = Math.abs(devPct) <= 0.2 ? "style='color:#10b981;font-weight:600;'" : "style='color:#ef4444;font-weight:600;'";

    tr.innerHTML = `
      <td><strong>${index + 1}</strong></td>
      <td>${state.muDelivered} MU</td>
      <td>
        <input type="number" step="0.01" class="reading-input" data-index="${index}" value="${reading.toFixed(2)}">
      </td>
      <td ${devClass}>${devPct >= 0 ? "+" : ""}${devPct.toFixed(2)}%</td>
      <td>
        <button class="btn-del-reading" data-index="${index}" title="Remove this reading">✕</button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  tbody.querySelectorAll(".reading-input").forEach(input => {
    input.addEventListener("input", (e) => {
      const idx = parseInt(e.target.dataset.index, 10);
      const val = parseFloat(e.target.value);
      if (!isNaN(val)) {
        state.readings[idx] = val;
        updateUI();
      }
    });
  });

  tbody.querySelectorAll(".btn-del-reading").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.target.dataset.index, 10);
      if (state.readings.length > 1) {
        state.readings.splice(idx, 1);
        renderReadingsTable();
        updateUI();
      } else {
        alert("At least one reading must remain.");
      }
    });
  });
}

// ==========================================================================
// 6. Interactive SVG Beam Quality Curve (k_Q vs TPR_20,10)
// ==========================================================================

function renderKqChart(activeKq) {
  const svg = document.getElementById("kqChartSvg");
  const width = 540;
  const height = 220;
  const padLeft = 55;
  const padRight = 30;
  const padTop = 25;
  const padBottom = 35;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const minTpr = 0.50;
  const maxTpr = 0.84;
  const minKq = 0.93;
  const maxKq = 1.01;

  function scaleX(tpr) {
    return padLeft + ((tpr - minTpr) / (maxTpr - minTpr)) * chartW;
  }
  function scaleY(kq) {
    return padTop + ((maxKq - kq) / (maxKq - minKq)) * chartH;
  }

  let html = `<g stroke="currentColor" opacity="0.15">`;

  for (let k = 0.94; k <= 1.01; k += 0.01) {
    const y = scaleY(k);
    html += `<line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke-dasharray="2,2"/>`;
  }
  for (let t = 0.50; t <= 0.84; t += 0.05) {
    const x = scaleX(t);
    html += `<line x1="${x}" y1="${padTop}" x2="${x}" y2="${height - padBottom}" stroke-dasharray="2,2"/>`;
  }
  html += `</g>`;

  html += `
    <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${height - padBottom}" stroke="currentColor" opacity="0.5" stroke-width="1.5"/>
    <line x1="${padLeft}" y1="${height - padBottom}" x2="${width - padRight}" y2="${height - padBottom}" stroke="currentColor" opacity="0.5" stroke-width="1.5"/>
  `;

  for (let k = 0.94; k <= 1.00; k += 0.02) {
    html += `<text x="${padLeft - 8}" y="${scaleY(k) + 4}" text-anchor="end" class="svg-text-small" fill="currentColor">${k.toFixed(2)}</text>`;
  }
  for (let t = 0.50; t <= 0.84; t += 0.05) {
    html += `<text x="${scaleX(t)}" y="${height - padBottom + 16}" text-anchor="middle" class="svg-text-small" fill="currentColor">${t.toFixed(2)}</text>`;
  }

  html += `
    <text x="${width / 2}" y="${height - 6}" text-anchor="middle" class="svg-text-small" font-weight="bold" fill="currentColor">Beam Quality Specifier TPR₂₀,₁₀</text>
    <text x="16" y="${height / 2}" text-anchor="middle" transform="rotate(-90 16 ${height / 2})" class="svg-text-small" font-weight="bold" fill="currentColor">k_Q,Q₀ Factor</text>
  `;

  Object.keys(KQ_PHOTON_CHAMBERS).forEach(chKey => {
    if (chKey !== state.chamberKey) {
      const kqs = KQ_PHOTON_CHAMBERS[chKey];
      let pathD = "";
      KQ_PHOTON_TPRS.forEach((tpr, i) => {
        pathD += (i === 0 ? "M" : "L") + `${scaleX(tpr)},${scaleY(kqs[i])} `;
      });
      html += `<path d="${pathD}" fill="none" stroke="#94a3b8" stroke-width="1" opacity="0.35"/>`;
    }
  });

  const activeKqs = KQ_PHOTON_CHAMBERS[state.chamberKey] || KQ_PHOTON_CHAMBERS["PTW 30013 Farmer (0.60 cm³, waterproof)"];
  let activePathD = "";
  KQ_PHOTON_TPRS.forEach((tpr, i) => {
    activePathD += (i === 0 ? "M" : "L") + `${scaleX(tpr)},${scaleY(activeKqs[i])} `;
  });
  const strokeLineColor = state.isFff ? "#f59e0b" : "#0284c7";
  html += `<path d="${activePathD}" fill="none" stroke="${strokeLineColor}" stroke-width="2.5"/>`;

  const currentX = scaleX(state.bqiTpr);
  const currentY = scaleY(activeKq);

  html += `
    <line x1="${padLeft}" y1="${currentY}" x2="${currentX}" y2="${currentY}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3,3"/>
    <line x1="${currentX}" y1="${currentY}" x2="${currentX}" y2="${height - padBottom}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3,3"/>

    <circle cx="${currentX}" cy="${currentY}" r="7" fill="#ef4444" stroke="#ffffff" stroke-width="2"/>

    <g transform="translate(${Math.min(currentX + 10, width - 180)}, ${Math.max(currentY - 35, padTop + 5)})">
      <rect x="0" y="0" width="165" height="30" rx="4" fill="#1e293b" opacity="0.95"/>
      <text x="8" y="14" fill="#ffffff" font-size="10" font-weight="bold">TPR₂₀,₁₀ = ${state.bqiTpr.toFixed(3)} ${state.isFff ? "(FFF)" : ""}</text>
      <text x="8" y="24" fill="${state.isFff ? "#fde047" : "#38bdf8"}" font-size="10" font-weight="bold">k_Q = ${activeKq.toFixed(4)}</text>
    </g>
  `;

  svg.innerHTML = html;
}

// ==========================================================================
// 7. Table 16 Inspector (TRS-398 Rev. 1, 2024)
// ==========================================================================

function buildTable63() {
  const table = document.getElementById("tblTrs63");
  let html = `<thead><tr><th>Chamber Model</th>`;
  KQ_PHOTON_TPRS.forEach(tpr => {
    html += `<th>${tpr.toFixed(2)}</th>`;
  });
  html += `</tr></thead><tbody>`;

  Object.keys(KQ_PHOTON_CHAMBERS).forEach(chKey => {
    html += `<tr data-chamber="${chKey}"><td><strong>${chKey}</strong></td>`;
    const rowVals = KQ_PHOTON_CHAMBERS[chKey];
    rowVals.forEach((val, i) => {
      html += `<td data-tpr="${KQ_PHOTON_TPRS[i]}">${val.toFixed(4)}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody>`;
  table.innerHTML = html;
}

function updateTableHighlight() {
  const table = document.getElementById("tblTrs63");
  if (!table) return;

  table.querySelectorAll("tr").forEach(tr => tr.classList.remove("active-chamber-row"));
  table.querySelectorAll("td").forEach(td => td.classList.remove("active-cell"));

  const targetRow = table.querySelector(`tr[data-chamber="${state.chamberKey}"]`);
  if (targetRow) {
    targetRow.classList.add("active-chamber-row");
    let closestIndex = 0;
    let minDiff = 1e9;
    KQ_PHOTON_TPRS.forEach((tpr, i) => {
      const diff = Math.abs(state.bqiTpr - tpr);
      if (diff < minDiff) {
        minDiff = diff;
        closestIndex = i;
      }
    });
    const cells = targetRow.querySelectorAll("td");
    if (cells[closestIndex + 1]) {
      cells[closestIndex + 1].classList.add("active-cell");
    }
  }
}

// ==========================================================================
// 8. Profile Management (Save, Load, Export, Import)
// ==========================================================================

function initProfiles() {
  const raw = localStorage.getItem("trs398_saved_profiles");
  if (raw) {
    try {
      savedProfiles = JSON.parse(raw);
    } catch (e) {
      savedProfiles = [...DEFAULT_PROFILES];
    }
  } else {
    savedProfiles = [...DEFAULT_PROFILES];
    localStorage.setItem("trs398_saved_profiles", JSON.stringify(savedProfiles));
  }
  populateProfileDropdown();
}

function populateProfileDropdown() {
  const select = document.getElementById("profileSelect");
  select.innerHTML = "";

  savedProfiles.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name;
    if (p.id === activeProfileId) opt.selected = true;
    select.appendChild(opt);
  });
}

function loadProfile(profileId) {
  const found = savedProfiles.find(p => p.id === profileId);
  if (!found) return;

  activeProfileId = profileId;
  state = { ...found };

  // Sync inputs
  const getEl = id => document.getElementById(id);
  getEl("linacId").value = state.linacId;
  getEl("physicistName").value = state.physicistName;
  getEl("nominalEnergy").value = state.nominalEnergy;
  getEl("setupType").value = state.setupType;
  getEl("bqiTpr").value = state.bqiTpr;
  getEl("zrefDepth").value = state.zrefDepth;
  getEl("dmaxDepth").value = state.dmaxDepth;
  getEl("pddZref").value = state.pddZref;
  getEl("muDelivered").value = state.muDelivered;
  getEl("targetDosePerMu").value = state.targetDosePerMu;

  getEl("chamberSelect").value = state.chamberKey;
  getEl("chamberSn").value = state.chamberSn;
  getEl("ndwValue").value = state.ndwValue;
  getEl("ndwUnit").value = state.ndwUnit;
  getEl("kelec").value = state.kelec;
  getEl("calLab").value = state.calLab;
  getEl("refTemp").value = state.refTemp;
  getEl("refPress").value = state.refPress;

  getEl("tempInput").value = state.tempInput;
  getEl("tempUnit").value = state.tempUnit;
  getEl("pressInput").value = state.pressInput;
  getEl("pressUnit").value = state.pressUnit;

  getEl("mPlus").value = state.mPlus;
  getEl("mMinus").value = state.mMinus;
  getEl("v1Voltage").value = state.v1Voltage;
  getEl("m1Reading").value = state.m1Reading;
  getEl("v2Voltage").value = state.v2Voltage;
  getEl("m2Reading").value = state.m2Reading;

  // FFF params
  getEl("kvolInput").value = state.kvol || estimateKvol(state.chamberKey, state.isFff, state.nominalEnergy);
  getEl("doseRateInput").value = state.doseRate || (state.isFff ? 1400 : 600);

  renderReadingsTable();
  updateUI();
}

function saveNewProfile(name) {
  if (!name || name.trim() === "") return;
  const newId = "prof_" + Date.now();
  const newProfile = {
    ...state,
    id: newId,
    name: name.trim()
  };
  savedProfiles.push(newProfile);
  localStorage.setItem("trs398_saved_profiles", JSON.stringify(savedProfiles));
  activeProfileId = newId;
  populateProfileDropdown();
  alert(`Profile "${name.trim()}" successfully saved!`);
}

function updateActiveProfile() {
  const idx = savedProfiles.findIndex(p => p.id === activeProfileId);
  if (idx === -1) return;

  savedProfiles[idx] = {
    ...state,
    id: activeProfileId,
    name: savedProfiles[idx].name
  };
  localStorage.setItem("trs398_saved_profiles", JSON.stringify(savedProfiles));
  alert(`Profile "${savedProfiles[idx].name}" updated with current settings!`);
}

function deleteActiveProfile() {
  if (savedProfiles.length <= 1) {
    alert("Cannot delete the only remaining profile.");
    return;
  }
  const curr = savedProfiles.find(p => p.id === activeProfileId);
  if (confirm(`Are you sure you want to delete profile "${curr.name}"?`)) {
    savedProfiles = savedProfiles.filter(p => p.id !== activeProfileId);
    activeProfileId = savedProfiles[0].id;
    localStorage.setItem("trs398_saved_profiles", JSON.stringify(savedProfiles));
    populateProfileDropdown();
    loadProfile(activeProfileId);
  }
}

function exportProfiles() {
  const data = JSON.stringify(savedProfiles, null, 2);
  const blob = new Blob([data], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `TRS398_Linac_Profiles_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function importProfiles(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const parsed = JSON.parse(e.target.result);
      if (Array.isArray(parsed) && parsed.length > 0) {
        savedProfiles = parsed;
        localStorage.setItem("trs398_saved_profiles", JSON.stringify(savedProfiles));
        activeProfileId = savedProfiles[0].id;
        populateProfileDropdown();
        loadProfile(activeProfileId);
        alert(`Successfully imported ${savedProfiles.length} profiles!`);
      } else {
        alert("Invalid profiles JSON file format.");
      }
    } catch (err) {
      alert("Error reading profiles file: " + err.message);
    }
  };
  reader.readAsText(file);
}

// ==========================================================================
// 9. Clinical Report Generator (IAEA TRS-398 Worksheet)
// ==========================================================================

function populateReport() {
  const calc = runFullCalculation();
  const now = new Date();

  document.getElementById("rptLinac").textContent = document.getElementById("linacId").value;
  document.getElementById("rptEnergy").textContent = `${state.nominalEnergy} ${state.isFff ? "FFF" : "Photons"}`;
  document.getElementById("rptBeamMode").textContent = state.isFff ? "⚡ Flattening Filter-Free (FFF)" : "Conventional Flattened (W)";
  document.getElementById("rptPhysicist").textContent = document.getElementById("physicistName").value;
  document.getElementById("rptDateTime").textContent = now.toLocaleDateString() + " " + now.toLocaleTimeString();

  document.getElementById("rptChamber").textContent = state.chamberKey;
  document.getElementById("rptChamberSn").textContent = state.chamberSn;
  document.getElementById("rptNdw").textContent = `${state.ndwValue} ${state.ndwUnit} (${calc.ndwGyPerNc.toFixed(6)} Gy/nC)`;
  document.getElementById("rptKelec").textContent = state.kelec.toFixed(4);
  document.getElementById("rptCalLab").textContent = state.calLab;

  document.getElementById("rptDmaxDepth").textContent = `${state.dmaxDepth.toFixed(1)} cm`;
  document.getElementById("rptDmaxCol").textContent = `${state.dmaxDepth.toFixed(1)} cm`;
  document.getElementById("rptBQI").textContent = `${state.bqiTpr.toFixed(3)} (TPR₂₀,₁₀)`;
  document.getElementById("rptPdd").textContent = `${state.pddZref.toFixed(1)}%`;
  document.getElementById("rptKq").textContent = `${calc.kq.toFixed(4)} (IAEA TRS-398 Rev. 1 Table 16 / Eq. 34)`;

  const rptKvolRow = document.getElementById("rptKvolRow");
  if (state.isFff) {
    rptKvolRow.style.display = "";
    document.getElementById("rptKvol").textContent = `${calc.kvol.toFixed(4)} (FFF Profile Average, DPP: ${calc.dppMgy.toFixed(3)} mGy/pulse)`;
  } else {
    rptKvolRow.style.display = "none";
  }

  document.getElementById("rptTP").textContent = `T = ${calc.tempC.toFixed(1)} °C, P = ${calc.pressKpa.toFixed(2)} kPa`;
  document.getElementById("rptKtp").textContent = calc.ktp.toFixed(4);
  document.getElementById("rptKpol").textContent = calc.kpol.toFixed(4);
  document.getElementById("rptKs").textContent = `${calc.ks.toFixed(4)} (${calc.ksResult.method.split("(")[0]})`;
  document.getElementById("rptKtot").textContent = calc.ktot.toFixed(4);
  document.getElementById("rptMeanRaw").textContent = `${calc.mRawAvg.toFixed(3)} nC (σ = ${calc.stdDev.toFixed(4)}, CV = ${calc.cvPct.toFixed(2)}%)`;
  document.getElementById("rptMq").textContent = `${calc.mq.toFixed(4)} nC`;

  document.getElementById("rptMu1").textContent = `${state.muDelivered.toFixed(0)} MU`;
  document.getElementById("rptDoseZrefGy").textContent = calc.dWzrefGy.toFixed(4);
  document.getElementById("rptDoseZrefCgy").textContent = calc.dWzrefCgy.toFixed(2);
  document.getElementById("rptDosePerMuZref").textContent = `${calc.dosePerMuZrefCgy.toFixed(4)} cGy/MU`;

  document.getElementById("rptMu2").textContent = `${state.muDelivered.toFixed(0)} MU`;
  document.getElementById("rptDoseDmaxGy").textContent = calc.dWzmaxGy.toFixed(4);
  document.getElementById("rptDoseDmaxCgy").textContent = calc.dWzmaxCgy.toFixed(2);
  document.getElementById("rptDosePerMuDmax").textContent = `${calc.dosePerMuDmaxCgy.toFixed(4)} cGy/MU`;

  document.getElementById("rptTargetOutput").textContent = `${calc.targetCgyPerMu.toFixed(3)} cGy/MU at d_max`;
  document.getElementById("rptMeasuredOutput").textContent = `${calc.dosePerMuDmaxCgy.toFixed(4)} cGy/MU`;
  
  const diffSign = calc.diffPct >= 0 ? "+" : "";
  document.getElementById("rptDiffPct").textContent = `${diffSign}${calc.diffPct.toFixed(2)}%`;

  const compEl = document.getElementById("rptCompliance");
  const absDiff = Math.abs(calc.diffPct);
  if (absDiff <= 1.0) {
    compEl.innerHTML = `<span style="color:#15803d;font-weight:700;">PASS (Within clinical tolerance ±1.0%)</span>`;
  } else if (absDiff <= 2.0) {
    compEl.innerHTML = `<span style="color:#a16207;font-weight:700;">ACTION REQUIRED (Within ±1.0% to ±2.0%)</span>`;
  } else {
    compEl.innerHTML = `<span style="color:#b91c1c;font-weight:700;">OUT OF TOLERANCE (Exceeds ±2.0% - Recalibrate Linac)</span>`;
  }

  const trimSign = calc.trimPct >= 0 ? "+" : "";
  document.getElementById("rptAdjustment").textContent = `${trimSign}${calc.trimPct.toFixed(2)}% MU trim`;
}

// ==========================================================================
// 10. Event Listeners & Binding
// ==========================================================================

function bindInputs() {
  const getEl = id => document.getElementById(id);

  // Profile management
  getEl("profileSelect").addEventListener("change", e => loadProfile(e.target.value));
  getEl("saveProfileBtn").addEventListener("click", () => {
    getEl("newProfileName").value = `${getEl("linacId").value} - ${state.nominalEnergy} ${state.isFff ? "FFF" : "W"}`;
    getEl("saveProfileModal").classList.remove("hidden");
  });
  getEl("closeSaveProfileModal").addEventListener("click", () => getEl("saveProfileModal").classList.add("hidden"));
  getEl("cancelSaveProfileBtn").addEventListener("click", () => getEl("saveProfileModal").classList.add("hidden"));
  getEl("confirmSaveProfileBtn").addEventListener("click", () => {
    const name = getEl("newProfileName").value;
    saveNewProfile(name);
    getEl("saveProfileModal").classList.add("hidden");
  });
  getEl("updateProfileBtn").addEventListener("click", updateActiveProfile);
  getEl("deleteProfileBtn").addEventListener("click", deleteActiveProfile);
  getEl("exportProfilesBtn").addEventListener("click", exportProfiles);
  getEl("importProfilesBtn").addEventListener("click", () => getEl("importFileInput").click());
  getEl("importFileInput").addEventListener("change", (e) => {
    if (e.target.files.length > 0) importProfiles(e.target.files[0]);
  });

  // FFF Mode switches
  getEl("btnModeFlattened").addEventListener("click", () => {
    state.isFff = false;
    state.kvol = 1.0000;
    if (state.bqiTpr === 0.635) state.bqiTpr = 0.670;
    if (state.pddZref === 63.5) state.pddZref = 67.0;
    getEl("bqiTpr").value = state.bqiTpr;
    getEl("pddZref").value = state.pddZref;
    getEl("doseRateInput").value = 600;
    state.doseRate = 600;
    updateUI();
  });

  getEl("btnModeFFF").addEventListener("click", () => {
    state.isFff = true;
    state.kvol = estimateKvol(state.chamberKey, true, state.nominalEnergy);
    if (state.bqiTpr === 0.670) state.bqiTpr = 0.635;
    if (state.pddZref === 67.0) state.pddZref = 63.5;
    getEl("bqiTpr").value = state.bqiTpr;
    getEl("pddZref").value = state.pddZref;
    getEl("kvolInput").value = state.kvol;
    getEl("doseRateInput").value = 1400;
    state.doseRate = 1400;
    updateUI();
  });

  getEl("kvolInput").addEventListener("input", e => { state.kvol = parseFloat(e.target.value) || 1.0; updateUI(); });
  getEl("doseRateInput").addEventListener("input", e => { state.doseRate = parseFloat(e.target.value) || 600; updateUI(); });

  // Energy & Setup
  getEl("nominalEnergy").addEventListener("change", e => { 
    state.nominalEnergy = e.target.value; 
    if (e.target.value === "10MV") {
      state.dmaxDepth = state.isFff ? 2.2 : 2.4;
      state.bqiTpr = state.isFff ? 0.705 : 0.735;
      state.pddZref = state.isFff ? 70.5 : 73.5;
    } else if (e.target.value === "6MV") {
      state.dmaxDepth = 1.5;
      state.bqiTpr = state.isFff ? 0.635 : 0.670;
      state.pddZref = state.isFff ? 63.5 : 67.0;
    }
    getEl("dmaxDepth").value = state.dmaxDepth;
    getEl("bqiTpr").value = state.bqiTpr;
    getEl("pddZref").value = state.pddZref;
    state.kvol = estimateKvol(state.chamberKey, state.isFff, state.nominalEnergy);
    getEl("kvolInput").value = state.kvol;
    updateUI(); 
  });

  getEl("setupType").addEventListener("change", e => { state.setupType = e.target.value; updateUI(); });
  getEl("bqiTpr").addEventListener("input", e => { state.bqiTpr = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("zrefDepth").addEventListener("input", e => { state.zrefDepth = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("dmaxDepth").addEventListener("input", e => { state.dmaxDepth = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("pddZref").addEventListener("input", e => { state.pddZref = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("muDelivered").addEventListener("input", e => { state.muDelivered = parseFloat(e.target.value) || 0; renderReadingsTable(); updateUI(); });
  getEl("targetDosePerMu").addEventListener("input", e => { state.targetDosePerMu = parseFloat(e.target.value) || 0; updateUI(); });

  // Chamber & Cal
  getEl("chamberSelect").addEventListener("change", e => { 
    state.chamberKey = e.target.value; 
    state.kvol = estimateKvol(state.chamberKey, state.isFff, state.nominalEnergy);
    getEl("kvolInput").value = state.kvol;
    updateUI(); 
  });
  getEl("chamberSn").addEventListener("input", e => { state.chamberSn = e.target.value; });
  getEl("ndwValue").addEventListener("input", e => { state.ndwValue = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("ndwUnit").addEventListener("change", e => { state.ndwUnit = e.target.value; updateUI(); });
  getEl("kelec").addEventListener("input", e => { state.kelec = parseFloat(e.target.value) || 1.0; updateUI(); });
  getEl("calLab").addEventListener("input", e => { state.calLab = e.target.value; });
  getEl("refTemp").addEventListener("input", e => { state.refTemp = parseFloat(e.target.value) || 20.0; updateUI(); });
  getEl("refPress").addEventListener("input", e => { state.refPress = parseFloat(e.target.value) || 101.325; updateUI(); });

  // Environmental
  getEl("tempInput").addEventListener("input", e => { state.tempInput = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("tempUnit").addEventListener("change", e => { state.tempUnit = e.target.value; updateUI(); });
  getEl("pressInput").addEventListener("input", e => { state.pressInput = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("pressUnit").addEventListener("change", e => { state.pressUnit = e.target.value; updateUI(); });

  // Polarity
  getEl("mPlus").addEventListener("input", e => { state.mPlus = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("mMinus").addEventListener("input", e => { state.mMinus = parseFloat(e.target.value) || 0; updateUI(); });

  // Recombination
  getEl("v1Voltage").addEventListener("input", e => { state.v1Voltage = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("m1Reading").addEventListener("input", e => { state.m1Reading = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("v2Voltage").addEventListener("input", e => { state.v2Voltage = parseFloat(e.target.value) || 0; updateUI(); });
  getEl("m2Reading").addEventListener("input", e => { state.m2Reading = parseFloat(e.target.value) || 0; updateUI(); });

  // Readings toolbar
  getEl("addReadingBtn").addEventListener("click", () => {
    const lastReading = state.readings.length > 0 ? state.readings[state.readings.length - 1] : 12.35;
    state.readings.push(lastReading);
    renderReadingsTable();
    updateUI();
  });

  getEl("quickFillBtn").addEventListener("click", () => {
    state.readings = [12.34, 12.35, 12.35];
    renderReadingsTable();
    updateUI();
  });

  // Table toggle
  getEl("toggleTableBtn").addEventListener("click", () => {
    const tblContainer = getEl("tableContainer");
    const isHidden = tblContainer.classList.contains("hidden");
    if (isHidden) {
      tblContainer.classList.remove("hidden");
      getEl("toggleTableBtn").textContent = "Hide Table 16";
    } else {
      tblContainer.classList.add("hidden");
      getEl("toggleTableBtn").textContent = "View Table 16";
    }
  });

  // Reset User Setup
  getEl("presetUserPromptBtn").addEventListener("click", () => {
    loadProfile("user_6mv_preset");
  });

  // Report modal
  getEl("printReportBtn").addEventListener("click", () => {
    populateReport();
    getEl("reportModal").classList.remove("hidden");
  });

  getEl("closeReportBtn").addEventListener("click", () => {
    getEl("reportModal").classList.add("hidden");
  });

  getEl("doPrintBtn").addEventListener("click", () => {
    window.print();
  });

  // CSV Export
  getEl("exportCsvBtn").addEventListener("click", exportCsv);

  // JSON Export
  getEl("exportJsonBtn").addEventListener("click", exportJson);

  // Copy Summary
  getEl("copySummaryBtn").addEventListener("click", copySummary);

  // Theme Toggle
  getEl("themeToggleBtn").addEventListener("click", () => {
    const isDark = document.body.classList.contains("theme-dark");
    if (isDark) {
      document.body.classList.remove("theme-dark");
      document.body.classList.add("theme-light");
      getEl("themeIcon").textContent = "🌙";
      getEl("themeText").textContent = "Dark";
      localStorage.setItem("trs398_theme", "light");
    } else {
      document.body.classList.remove("theme-light");
      document.body.classList.add("theme-dark");
      getEl("themeIcon").textContent = "☀️";
      getEl("themeText").textContent = "Light";
      localStorage.setItem("trs398_theme", "dark");
    }
    const calc = runFullCalculation();
    renderKqChart(calc.kq);
  });
}

// ==========================================================================
// 11. Export Utilities (CSV, JSON, Clipboard)
// ==========================================================================

function exportCsv() {
  const calc = runFullCalculation();
  const rows = [
    ["Parameter", "Value", "Unit"],
    ["Protocol", "IAEA TRS-398", ""],
    ["Linac ID", document.getElementById("linacId").value, ""],
    ["Nominal Energy", state.nominalEnergy, ""],
    ["Beam Mode", state.isFff ? "Flattening Filter-Free (FFF)" : "Conventional Flattened (W)", ""],
    ["Setup", state.setupType, "SSD = 100 cm"],
    ["Measurement Depth (z_ref)", state.zrefDepth, "cm"],
    ["Depth of Max Dose (d_max)", state.dmaxDepth, "cm"],
    ["Beam Quality Index (TPR_20,10)", state.bqiTpr, ""],
    ["PDD at z_ref", state.pddZref, "%"],
    ["Ionization Chamber", state.chamberKey, ""],
    ["Chamber S/N", state.chamberSn, ""],
    ["Calibration Factor N_D,w", state.ndwValue, state.ndwUnit],
    ["Volume Averaging (k_vol)", calc.kvol.toFixed(4), ""],
    ["Linac Dose Rate", state.doseRate || 600, "MU/min"],
    ["Dose Per Pulse", calc.dppMgy.toFixed(3), "mGy/pulse"],
    ["k_TP", calc.ktp.toFixed(5), ""],
    ["k_pol", calc.kpol.toFixed(5), ""],
    ["k_s", calc.ks.toFixed(5), ""],
    ["Total Correction Factor k_tot", calc.ktot.toFixed(5), ""],
    ["Mean Raw Reading M_raw", calc.mRawAvg.toFixed(4), "nC"],
    ["Corrected Reading M_Q", calc.mq.toFixed(4), "nC"],
    ["Beam Quality Factor k_Q", calc.kq.toFixed(4), ""],
    ["Absorbed Dose at z_ref (10 cm)", calc.dWzrefCgy.toFixed(3), "cGy"],
    ["Absorbed Dose at d_max (1.5 cm)", calc.dWzmaxCgy.toFixed(3), "cGy"],
    ["Delivered Monitor Units", state.muDelivered, "MU"],
    ["Dose Output at d_max", calc.dosePerMuDmaxCgy.toFixed(4), "cGy/MU"],
    ["Nominal Target", calc.targetCgyPerMu.toFixed(4), "cGy/MU"],
    ["Output Deviation (Δ)", calc.diffPct.toFixed(2), "%"],
    ["Recommended MU Trim", calc.trimPct.toFixed(2), "%"]
  ];

  const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.map(cell => `"${cell}"`).join(",")).join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `TRS398_${state.nominalEnergy}_${state.isFff ? "FFF_" : ""}${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function exportJson() {
  const calc = runFullCalculation();
  const session = {
    protocol: "IAEA TRS-398",
    version: "2.0 (Flattened & FFF Beams)",
    timestamp: new Date().toISOString(),
    profileId: activeProfileId,
    state,
    calculation: calc
  };

  const str = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(session, null, 2));
  const link = document.createElement("a");
  link.setAttribute("href", str);
  link.setAttribute("download", `TRS398_Session_${state.nominalEnergy}_${state.isFff ? "FFF_" : ""}${new Date().toISOString().slice(0, 10)}.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function copySummary() {
  const calc = runFullCalculation();
  const summary = `=== IAEA TRS-398 REFERENCE DOSIMETRY SUMMARY ===
Linac: ${document.getElementById("linacId").value}
Energy: ${state.nominalEnergy} ${state.isFff ? "⚡ FFF (Flattening Filter-Free)" : "Flattened (W)"}
Setup: ${state.setupType} = 100 cm | Field: 10×10 cm²
Measurement Depth: z_ref = ${state.zrefDepth} cm | d_max = ${state.dmaxDepth} cm
Beam Quality: TPR₂₀,₁₀ = ${state.bqiTpr.toFixed(3)} | PDD(10) = ${state.pddZref.toFixed(1)}%
Chamber: ${state.chamberKey} (S/N: ${state.chamberSn})
k_Q: ${calc.kq.toFixed(4)}${state.isFff ? ` | k_vol: ${calc.kvol.toFixed(4)}` : ""}
k_TP: ${calc.ktp.toFixed(4)} | k_pol: ${calc.kpol.toFixed(4)} | k_s: ${calc.ks.toFixed(4)}
${state.isFff ? `Dose Rate: ${state.doseRate || 1400} MU/min | DPP: ${calc.dppMgy.toFixed(3)} mGy/pulse\n` : ""}Mean Reading: ${calc.mRawAvg.toFixed(3)} nC for ${state.muDelivered} MU (CV: ${calc.cvPct.toFixed(2)}%)
Dose at z_ref (10 cm): ${calc.dWzrefCgy.toFixed(2)} cGy (${calc.dosePerMuZrefCgy.toFixed(4)} cGy/MU)
Dose at d_max (${state.dmaxDepth} cm): ${calc.dWzmaxCgy.toFixed(2)} cGy (${calc.dosePerMuDmaxCgy.toFixed(4)} cGy/MU)
Target: ${calc.targetCgyPerMu.toFixed(4)} cGy/MU
Output Deviation: ${calc.diffPct >= 0 ? "+" : ""}${calc.diffPct.toFixed(2)}% (${Math.abs(calc.diffPct) <= 1.0 ? "PASS" : "ACTION REQUIRED"})
Recommended Linac Adjustment: ${calc.trimPct >= 0 ? "+" : ""}${calc.trimPct.toFixed(2)}% MU trim`;

  navigator.clipboard.writeText(summary).then(() => {
    alert("TRS-398 calculation summary copied to clipboard!");
  }).catch(() => {
    alert("Summary copied to console.");
    console.log(summary);
  });
}

// ==========================================================================
// 12. Initialization
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  const savedTheme = localStorage.getItem("trs398_theme");
  if (savedTheme === "dark") {
    document.body.classList.remove("theme-light");
    document.body.classList.add("theme-dark");
    document.getElementById("themeIcon").textContent = "☀️";
    document.getElementById("themeText").textContent = "Light";
  }

  initProfiles();
  buildTable63();
  renderReadingsTable();
  bindInputs();
  loadProfile(activeProfileId);
});
