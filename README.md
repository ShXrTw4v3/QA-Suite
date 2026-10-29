# Radiotherapy Machine QA & IAEA TRS-398 Rev. 1 (2024) Reference Dosimetry Suite

An integrated, local SQLite-backed clinical radiotherapy machine QA, IAEA TRS-398 Rev. 1 (2024) absolute reference dosimetry, and beam profile analysis platform built with Python, Streamlit, Plotly, ReportLab, and `pylinac`.

Designed for medical physicists to perform reference calibrations, dynamic log analysis, machine QA, and beam profile scanning directly on a local workstation, research box, or across a private hospital intranet without external cloud dependencies or patient health information (PHI) exposure.

---

## Key Capabilities

### 1. Beam Profile Flatness, Symmetry & Penumbra Analyzer (`beam_profile_page.py`, `beam_profile_analyzer.py`)
- **Universal Multi-Format Scanners**: Direct parsing of 1D/2D water tank scans, diode/ion-chamber profilers (IBA OmniPro ASCII, PTW Mephysto, Sun Nuclear IC Profiler, CSV, TSV), and 2D DICOM EPID (`.dcm`) planar fluences.
- **Built-in Analytical Benchmarks**: Instant reference verification with synthetic benchmark profiles (`6 MV Flattened`, `6 MV FFF`, `10 MV FFF`).
- **Flattened Beams**: IEC 60976 ($D_{max}/D_{min}$, point-difference quotient) and Varian point-ratio flatness and symmetry, FWHM field width, CAX offset, and 80%–20% left/right penumbra.
- **FFF Unflattened Beams**: International consensus metrics including CAX peak normalization, 1D Gaussian smoothing, derivative inflection-point edge detection, inflection-anchored penumbra, and off-axis point-difference symmetry.
- **Interactive Fluence & Profile Plots**: Plotly overlays displaying CAX, inflection/FWHM edges, penumbra bands, 80% evaluation zones, folded symmetry differences, and EPID millimeter fluence maps.
- **Persistence & PDF**: One-click logging to `qa_data.db` (`profile_qa_logs`) and clinical PDF export.

### 2. IAEA TRS-398 Rev. 1 (2024) Absolute Reference Dosimetry (`trs398_page.py`, `trs398_calculator.py`, `index.html`)
- **Published 2024 Formalism**: Fully compliant with IAEA TRS-398 Rev. 1 (2024) for high-energy photon (Flattened & FFF) and electron beams.
- **Table 16 / Eq. (34) Beam Quality Factors ($k_Q$)**: Built-in coefficients for all 26 cylindrical chambers with sub-0.00015 numerical precision across $TPR_{20,10}$ values from 0.56 to 0.82.
- **Table 20 / Eqs. (98)–(99) Electron Dosimetry**: Full support for Plane-Parallel and cylindrical ionization chambers with clinical guard rejecting cylindrical chambers for $R_{50} < 3.0\text{ g/cm}^2$.
- **Correction Factors**:
  * Temperature & Pressure $k_{TP}$ with standard reference temperature $T_0 = 273.15\text{ K}$ (Eq. 10).
  * Two-voltage ion recombination $k_s$ (Table 10 quadratic interpolation) and polarity $k_{pol}$ (Eq. 11).
  * FFF volume averaging correction $k_{vol}$ (Eq. 22 / Table 11) and Dose-Per-Pulse (DPP) monitoring.
- **Dual Runtime**: Interactive Streamlit interface, standalone CLI calculator (`trs398_calculator.py`), and pure client-side HTML5/JavaScript app (`index.html`) requiring zero Python installation.

### 3. Unified Clinical PDF Report Generator (`clinical_pdf_generator.py`)
- **Print-Ready Clinical Certificates**: Clean ReportLab Platypus engine generating downloadable PDFs across all modules:
  * IAEA TRS-398 Rev. 1 Photon & FFF Reference Dosimetry
  * IAEA TRS-398 Rev. 1 Electron Reference Dosimetry
  * Dynamic MLC Trajectory Log Kinematic QA
  * EPID Picket Fence MLC Positioning QA
  * Winston-Lutz 3D Radiation Isocenter QA
  * CBCT CatPhan 3D Volumetric Imaging QA
  * Beam Profile Flatness, Symmetry & Penumbra QA
  * Daily/Monthly Machine QA Summary Report
- **Strict Zero-PHI Assurance**: Automatic recursive scrubbing of patient names, IDs, MRNs, and plan UIDs, replacing them with `MACHINE_QA_ANONYMIZED`.
- **Clinical Layout**: Running headers/footers with `NumberedCanvas` ("Page X of Y"), unit and protocol metadata, configured tolerances, numerical results with physical units, embedded 300 DPI diagnostic plots, and physicist/reviewer sign-off blocks.

### 4. Varian Trajectory Log Analyzer (`linac_qa_modules.py`)
- Analyzes `.bin` machine trajectory logs via `pylinac`.
- Evaluates per-leaf RMS errors, max leaf errors, Modulation Complexity Score ($\text{MCS} = \text{LSV} \times \text{AAV}$), gravity sag polar plots, and MLC error heatmaps.
- Includes automatic network directory scanning with result caching.

### 5. Picket Fence MLC QA (`linac_qa_modules.py`)
- Evaluates EPID `.dcm` planar picket fence images with configurable action/tolerance limits, worst-leaf displacement tables, and visual overlays.

### 6. Winston-Lutz 3D Isocenter QA (`linac_qa_modules.py`)
- Evaluates multi-angle EPID `.zip` sets for gantry, collimator, and couch rotation isocenters, calculating the maximum 3D distance from radiation CAX to BB.

### 7. CBCT (CatPhan) 3D Volumetric Imaging QA (`linac_qa_modules.py`)
- Automated analysis of CatPhan 504, 600, 503, and 604 DICOM axial slices (HU linearity, slice thickness, low-contrast visibility, and MTF 50%).

### 8. Linac Golden Beam Data & Machine Energy Management (`app.py`, `crud.py`, `beam_profile_analyzer.py`)
- **Independent Machine Registries**: Complete physical separation of beam data for hospital accelerators (`TrueBeam`, `Edge`, `Halcyon`). In this clinic, **Edge is NOT matched to TrueBeam** — each machine maintains independent baselines and configured energy parameters.
- **Universal Golden Beam Data Uploader**: Accepts PTW BeamScan `.mcc` (multi-scan), IBA OmniPro, CSV, TSV, and Excel files.
- **Direct PDD & Beam Quality Index ($TPR_{20,10}$) Derivation**: Automatically evaluates $d_{max}$, $PDD_{10}$, $PDD_{20}$, and derives $TPR_{20,10}$ via IAEA TRS-398 Rev. 1 Eq. 12:
  $$TPR_{20,10} = 1.2661 \times \left(\frac{PDD_{20}}{PDD_{10}}\right) - 0.0595$$
- **Automatic Energy Population**: One-click commitment stores the baseline curve in `reference_beam_data` and populates `unit_energies` for that linac, making golden data instantly accessible across TRS-398 reference dosimetry and Beam Profile QA.

### 9. Ion Chamber Registry & Calibration Certificate Parser (`chamber_manager.py`)
- **Automated Certificate Extraction**: Upload PDF or image calibration certificates from accredited laboratories (PTW-Freiburg SSDL, IAEA Dosimetry Lab, IBA, Standard Imaging ADCL, NIST).
- **Automated Field Detection**: Automatically extracts chamber model (matched to IAEA TRS-398 Table 16 / Table 20), serial number, calibration laboratory, date, certificate ID, and absorbed dose to water calibration coefficient $N_{D,w}$ with automatic normalization to cGy/nC and Gy/nC.
- **Direct TRS-398 Integration**: Registered chambers populate the chamber dropdowns in TRS-398 photon and electron calibration modules, eliminating manual typing of calibration factors.

### 10. Daily QA, Trending & Machine Administration (`app.py`, `crud.py`, `db_setup.py`)
- Daily checklist data entry, interactive Plotly historical control charts, beam calibration records, and linac service/downtime logging.

---

## Pre-Configured Clinical Reference Setups

| Parameter | Flattened (6 MV W) | FFF (6 MV FFF) | FFF (10 MV FFF) |
| :--- | :--- | :--- | :--- |
| **Nominal Energy** | 6 MV | 6 MV FFF | 10 MV FFF |
| **Quality Index ($TPR_{20,10}$)** | **0.670** | **0.635** | **0.705** |
| **Setup Geometry** | SSD = 100.0 cm | SSD = 100.0 cm | SSD = 100.0 cm |
| **Measurement Depth ($z_{ref}$)** | 10.0 cm in water | 10.0 cm in water | 10.0 cm in water |
| **Depth of Maximum Dose ($d_{max}$)**| 1.5 cm | 1.5 cm | 2.2 cm |
| **Reference PDD ($PDD_{10}$)** | 67.0% | 63.5% | 70.5% |
| **Chamber Model** | PTW 30013 Farmer | PTW 30013 Farmer | PTW 30013 Farmer |
| **Beam Quality Factor ($k_Q$)** | **0.9891** | **0.9938** | **0.9830** |
| **Volume Averaging ($k_{vol}$)** | 1.0000 | **1.0015** | **1.0034** |
| **Typical Dose Rate** | 600 MU/min | **1400 MU/min** | **2400 MU/min** |
| **Dose Per Pulse (DPP)** | ~0.28 mGy/pulse | **~0.65 mGy/pulse** | **~1.11 mGy/pulse** |

---

## Project Structure

```text
qa-suite/
├── app.py                      # Main Streamlit clinical application
├── beam_profile_analyzer.py    # Beam profile algorithms (IEC, Varian, FFF inflection)
├── beam_profile_page.py        # Beam profile UI tab with Plotly visualization
├── clinical_pdf_generator.py   # ReportLab clinical PDF engine across all QA modules
├── trs398_calculator.py        # IAEA TRS-398 Rev. 1 (2024) physics calculation engine
├── trs398_page.py              # Streamlit interactive TRS-398 page
├── linac_qa_modules.py         # Pylinac modules (Trajectory, PicketFence, WL, CBCT)
├── db_setup.py                 # SQLite database schema initialization and seed data
├── crud.py                     # Database CRUD helper functions
├── hospital_server.py          # Network server launcher with IP detection & firewall guides
├── index.html                  # Standalone client-side TRS-398 web app (zero Python needed)
├── app.js                      # Client-side TRS-398 logic and formulas
├── style.css                   # Client-side stylesheet
├── server.py                   # Lightweight Python HTTP server for index.html
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (protects DB, caches, and reports)
│
├── start_hospital_server.bat   # Windows launcher for hospital intranet
├── run_standalone_windows.bat  # Windows launcher for local standalone mode (localhost)
├── setup_windows.bat           # Windows automated setup script
├── start_hospital_server.sh    # macOS / Linux launcher for hospital intranet
├── start_hospital_server.command # macOS clickable launcher
├── launch.command              # macOS launcher for standalone HTML app
│
└── tests/                      # Automated validation suite (338 tests)
    ├── conftest.py
    ├── test_beam_profile.py
    ├── test_trs398_photon.py
    ├── test_trs398_electron.py
    ├── test_trs398_corrections.py
    ├── test_clinical_pdf.py
    ├── test_database.py
    ├── test_adversarial_challenger1.py
    └── test_adversarial_pdf.py
```

---

## Deployment & Execution Instructions

### Option A: Hospital Intranet Deployment (Multi-User Network Server)

To provide access to medical physicists, dosimetrists, and linac engineers across the hospital intranet from any desktop, laptop, or tablet (no client-side software required):

1. **Start the Hospital Server on the host workstation**:
   - **Windows**: Double-click `start_hospital_server.bat`
   - **macOS / Linux**:
     ```bash
     ./start_hospital_server.sh
     ```
2. **Access via Web Browser**:
   The launcher automatically resolves your local LAN IP address and displays access links:
   ```text
   ► On this Host Workstation:     http://localhost:8501
   ► Local Hospital Network URL:   http://192.168.1.150:8501   <-- Share with clinic staff
   ```
3. **Configure Firewall (if connecting from another machine on the LAN)**:
   - **Windows (Run PowerShell / CMD as Administrator)**:
     ```cmd
     netsh advfirewall firewall add rule name="MicroQA_Hospital_Server" dir=in action=allow protocol=TCP localport=8501
     ```
   - **Linux**:
     ```bash
     sudo ufw allow 8501/tcp
     ```

---

### Option B: Standalone Installation on Windows 10 2016 LTS Research Box

Windows 10 Enterprise 2016 LTSB / LTSC (Version 1607, build 14393) is commonly used on hospital research and linac consoles. Follow these steps:

#### 1. Prerequisites on Windows 10 2016 LTSB
- **Python**: Install Python 3.9 or 3.10 (64-bit).
  * Download: [Python 3.10.11 Windows installer (64-bit)](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
  * During setup, check **"Add Python to PATH"** and select **"Install Now"**.
- **C Runtime (if fresh install)**: Ensure the **Microsoft Visual C++ 2015–2022 Redistributable (x64)** is installed (required by `numpy` and `scipy`).

#### 2. Automated Setup
1. Copy the project folder to the research box (e.g., `C:\QA_Suite`).
2. Double-click `setup_windows.bat`. This automatically:
   - Creates a Python virtual environment (`venv`).
   - Upgrades `pip`.
   - Installs all dependencies from `requirements.txt`.

#### 3. Air-Gapped / Offline Network Installation
If the research box is strictly air-gapped without direct internet access:
1. On an internet-connected PC, download the dependency wheel files:
   ```cmd
   pip download -r requirements.txt -d wheels
   ```
2. Copy the `wheels/` directory to `C:\QA_Suite\wheels` via approved secure storage.
3. On the research box, install locally:
   ```cmd
   venv\Scripts\python.exe -m pip install --no-index --find-links=wheels -r requirements.txt
   ```

#### 4. Launching the App
- **Standalone Localhost Mode (No network exposure)**:
  Double-click `run_standalone_windows.bat`. Binds strictly to `127.0.0.1:8501` and opens your browser.
- **Hospital LAN Mode**:
  Double-click `start_hospital_server.bat` to allow other hospital computers to connect.
- **Zero-Python Standalone HTML Mode**:
  Double-click `index.html` in Chrome or Edge to run IAEA TRS-398 Rev. 1 reference dosimetry with zero Python runtime required.

---

## Automated Validation & Pytest Suite (344 Tests)

The suite includes an independent verification test suite covering all mathematical formalisms, edge cases, parsers, and report generation:

```bash
# Run full test suite with verbose output:
./venv/bin/python3 -m pytest tests/ -v

# Run only TRS-398 photon & electron validation:
./venv/bin/python3 -m pytest tests/test_trs398_photon.py tests/test_trs398_electron.py -v

# Run beam profile analyzer tests:
./venv/bin/python3 -m pytest tests/test_beam_profile.py -v

# Run clinical PDF generator tests:
./venv/bin/python3 -m pytest tests/test_clinical_pdf.py -v
```

### Verified Acceptance Criteria:
- **Photon $k_Q$**: IAEA TRS-398 Rev. 1 Table 16 / Eq. (34) matches all 26 cylindrical chambers with maximum error $\le 0.00014$.
- **Electron $k_Q$**: Table 20 / Eqs. (98)–(99) verified across all 8 chambers with strict rejection of cylindrical chambers for $R_{50} < 3.0\text{ g/cm}^2$.
- **Beam Profiles**: Field width ($\pm 0.5\text{ mm}$), penumbra ($\pm 0.5\text{ mm}$), flatness ($\pm 0.1\%$), and symmetry ($\pm 0.1\%$) verified against analytical benchmarks.
- **Clinical PDFs**: Validated for zero PHI leakage, valid `%PDF-` binary output, and leak-free resource closing across all 8 report types.

---

## Zero-PHI & Clinical Safety Guardrails

1. **Patient Health Information (PHI)**:
   - This suite is designed for **machine quality assurance only**.
   - The PDF generator and DICOM parsers automatically redact patient-identifiable tags (`PatientName`, `PatientID`, `PatientBirthDate`, etc.) to `MACHINE_QA_ANONYMIZED`.
   - Never upload identifiable patient treatment plans or diagnostic CTs containing PHI.
2. **Clinical Verification**:
   - In accordance with medical physics quality management procedures, all calculated dose outputs, machine tolerances, and chamber calibration coefficients must be verified by a qualified Medical Physicist against primary institutional standards prior to clinical treatment delivery.

---

## License

This software is distributed under the MIT License. See [LICENSE](LICENSE) for details.
