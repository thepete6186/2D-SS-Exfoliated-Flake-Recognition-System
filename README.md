# Automated Optical Microscopy Pipeline for 2D Material Identification

An automated hardware-software framework designed to scan optical microscopy samples, isolate substrate and flake signatures in HSV color space, execute raster-scan image acquisition, and automatically locate high-priority 2D material flakes for downstream inspection or transfer.

---

## 🔬 System Workflow Architecture

1. **Substrate Calibration (Initialization):**
   - The operator places a sample under the microscope and clicks a few representative points on the bare substrate.
   - The system extracts baseline HSV optical signatures ($H_{\text{sub}}, S_{\text{sub}}, V_{\text{sub}}$) using the core `HSVPipeline`.

2. **Automated Raster Scan Initialization:**
   - The motorized stage homes/navigates to the origin starting position (e.g., top-left / bottom-left corner of the target grid).
   - The camera initiates a grid scan (e.g., a $50 \times 50$ grid yielding $2,500$ frames).

3. **Frame-by-Frame Processing:**
   For every acquired frame, the pipeline automatically:
   - Computes local channel histograms ($H, S, V$).
   - Runs the `HSVPipeline` to generate probabilistic mask maps ($P_{\text{map}}$).
   - Identifies candidate flake locations via contour detection.
   - Calculates real physical/pixel surface area for candidate flakes.

4. **Target Queueing & Re-centering:**
   - Filters candidate flakes against an area threshold ($A \ge A_{\text{min}}$).
   - Logs spatial coordinates of qualifying targets relative to the stage grid.
   - Automatically directs the motorized stage back to the highest-priority target coordinates for high-magnification verification or manipulation.

---

## 🛠️ Hardware Integration Specs

- **Optical Detector / Camera:** Carl Zeiss Axiocam 208 Color
- **Motorized Stage:** Precise XY5050 Stage
- **Stage Controller:** Zolix ZC300 Motion Controller

---

## 🚀 Near-Term Development Roadmap

### Current Focus (Immediate Step):
- [x] **Interactive Calibration Engine:** Live camera stream view with click-to-calibrate for bare substrate selection (annotator toggle/Shift-click → points JSON → `--substrate-json` CLI flag → pipeline override).
- [ ] **Hardware Driver Integration:** Build Python wrappers for:
  - Zeiss Axiocam 208 capture pipeline.
  - Zolix ZC300 stage controller over serial/USB communications (step movement, home, position querying).
- [ ] **Automated Raster-Scan Loop:** Implement coordinated frame acquisition and stage stepping for systematic grid coverage.

### Future Roadmap (Post-Scanning):
- [ ] Stitching individual frame heatmaps into a full-wafer mosaic.
- [ ] Flake classification by layer thickness (monolayer vs. bilayer vs. bulk).
- [ ] Automated micro-manipulator pickup path generation.

---

## 📦 Software Components

| Component | Location | Description |
|-----------|----------|-------------|
| **hsv-pipeline-semi** | `hsv-pipeline-semi/` | **Main experimental pipeline** — Semi-supervised flake detection with click-based calibration |
| hsv-pipeline | `hsv-pipeline/` | Unsupervised CSSPN pipeline (reference implementation) |
| hsv-pipeline-backup | `hsv-pipeline-backup/` | Original unsupervised pipeline (untouched backup) |
| transfer-stage-control | `transfer-stage-control/` | Stage controller GUI and hardware drivers (in development) |
| baseline | `baseline/` | Baseline CNN-based flake detection (reference) |
| edge-det | `edge-det./` | Edge-aware denoising experiments |

---

## 🔬 HSV Pipeline-Semi (Primary Detection Engine)

The semi-supervised pipeline combines automatic substrate detection with minimal user input to learn flake signatures:

### Key Features
- **Global substrate detection** via histogram mode analysis
- **Click-based calibration** — operator clicks 2-3 flake regions to define signature
- **Direction-aware matching** — only pixels deviating in the same direction as clicked flakes are flagged
- **Calibrate mode** — click once on one image, apply signature to entire sample
- **Per-image substrate adaptation** — each frame gets automatic substrate detection

### Usage
```bash
# Interactive single-image mode
python hsv-pipeline-semi/run_semi_supervised.py --image ../dataset/sample4/ws2-251104161126416.jpg

# Calibrate mode (click on first image, apply to all in sample)
python hsv-pipeline-semi/run_semi_supervised.py --sample sample4 --calibrate

# Use specific calibration image
python hsv-pipeline-semi/run_semi_supervised.py --sample sample2 --calibrate --calibration-image tmd_sample_3.jpg
```

### Experimental Results
- **Sample2**: 19/19 images matched (100% success) with tmd_sample_3 calibration
- **Sample3**: 2/2 images matched (100% success)
- **Sample4**: 3/4 images matched (75% success) — one image had different substrate hue

See `hsv-pipeline-semi/README.md` for full documentation.

---

## 🎯 Project Goals

1. **Automation**: Minimize operator intervention through intelligent substrate detection and flake signature learning
2. **Precision**: Locate flakes with sub-pixel accuracy for downstream transfer/stamping operations
3. **Throughput**: Scan large-area samples (50×50 grids) in minutes, not hours
4. **Adaptability**: Handle varying substrate materials, flake types, and imaging conditions across samples

---

## 📋 System Requirements

- Python 3.7+
- numpy, opencv-python, matplotlib, scipy
- Zeiss Axiocam 208 SDK (pending integration)
- Zolix ZC300 stage control (pyserial, Modbus RTU — see `stage/`)
- Motorized microscope stage with XY5050 controller

---

## 🎛️ Stage Control (`stage/`)

Driver package for the Zolix ZC300 motion controller (Modbus RTU over a
USB virtual COM port, fixed 115200 8N1). Works in integer pulses;
physical-unit conversion stays in `camera/coordinate_mapper.py`.

```python
from stage import ZolixZC300, SimulatedStage

stage = ZolixZC300(port="COM3")        # SimulatedStage() for dry runs
stage.connect()
stage.move_relative("x", 1000)          # blocks until the axis stops
stage.home("all")
print(stage.get_position())             # {'x': ..., 'y': ..., 'r': ...}
stage.disconnect()
```

Quick CLI for scripting stage moves without the GUI:

```
python move_stage.py --simulate status        # simulated stage, no hardware
python move_stage.py --port COM3 status
python move_stage.py --port COM3 rel x 1000   # relative move
python move_stage.py --port COM3 abs y 25000  # absolute move
python move_stage.py --port COM3 home all
python move_stage.py --port COM3 stop
python move_stage.py --port COM3 speed x 2000
```

Smoke test on the lab PC (validates the absolute-move and home opcodes,
which are implemented from the register map but not yet exercised on
hardware):

```
python -m stage.zc300_smoke --list-ports
python -m stage.zc300_smoke --port COM3 --identify --status
python -m stage.zc300_smoke --port COM3 --jog x 1000
```

Adapted from [transfer-stage-control](https://github.com/Lewbert/transfer-stage-control)
(MIT), with absolute moves, homing, and blocking waits added for
raster scanning.

---

## 🧪 Testing

Install test dependencies and run the suite:

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v
```

Coverage:

```bash
python -m pytest tests/ --cov=stage --cov=camera --cov-report=term-missing
```

The test suite covers:
- **MODBUS RTU layer** — frame builders, parsers, CRC-16 golden vectors
- **ZolixZC300 driver** — connect handshake, motion sequences, error mapping (via fake serial port)
- **SimulatedStage** — position math, soft limits, rate-modeled motion, typed errors
- **move_stage.py CLI** — all subcommands via `--simulate`
- **Camera decode** — Bayer/YUY2/GenTL format detection and demosaicing
- **Annotator payload** — JSON roundtrip and calibration fallback
- **HSV pipeline** — flake signatures, substrate calibration, CLI override resolution

---

## 🔄 Pipeline Status

| Module | Status | Notes |
|--------|--------|-------|
| HSV Detection Algorithm | ✅ Complete | Semi-supervised pipeline operational |
| Substrate Calibration | ✅ Complete | Click-to-calibrate (annotator) + global histogram fallback |
| Flake Signature Learning | ✅ Complete | Click-based extraction with direction-aware matching |
| Camera Driver | ⏳ Pending | Zeiss Axiocam 208 integration required |
| Stage Controller | 🧪 Implemented | Zolix ZC300 Modbus RTU driver (`stage/`) — pending hardware validation |
| Raster-Scan Loop | ⏳ Pending | Coordinated acquisition + stage stepping |
| Target Queueing | ⏳ Pending | Contour detection + area filtering + re-centering |
| Wafer Stitching | 🔮 Future | Full-sample mosaic generation |

---

## 📄 License

[Add license information here]

---

## 👥 Contact

[Add contact information here]