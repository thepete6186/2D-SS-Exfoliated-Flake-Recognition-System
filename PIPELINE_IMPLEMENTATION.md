# Full Pipeline Implementation

## Overview
The full pipeline button performs automated edge detection and alignment with the following workflow:

**Key Concept**: The position where you **click the button** becomes the origin **(0, 0)**. All operations happen at that position.

## Pipeline Steps

### Step 0: Save Current Position as Origin (0,0)
- **Action**: Saves current stage position as the reference origin
- **Movement**: None
- **Purpose**: The position where you click the button becomes (0,0) reference point

### Step 1: Detect Edge at Origin
- **Action**: Captures image and detects substrate-stage edge using HSV color detection
- **Movement**: None (stays at current position)
- **Output**: 
  - Edge angle (theta)
  - Edge anchor point (stored as reference)
  - Edge visualization overlay (red line)
- **Purpose**: Finds the substrate boundary at the origin position

### Step 2: Auto-Align at Origin
- **Action**: Rotates sample until edge is vertical (90°)
- **Movement**: R-axis rotation only (NO X/Y movement)
- **Iterations**: Up to 10 iterations, 2° tolerance
- **Purpose**: Aligns sample to vertical while staying at the origin position

### Step 3: Return to Origin
- **Action**: Returns to the saved origin position
- **Movement**: None needed (already at origin)
- **Purpose**: Confirms final position (no movement required)

## Button Label
```
Full Pipeline: Detect → Align → Return to (0,0)
```

This means:
1. Detect edge (at current position, treated as 0,0)
2. Align (at current position, rotation only)
3. Return to (0,0) ← The origin where you clicked

## Corrected Implementation

### Before (WRONG):
```python
# Step 0: Save position
# Step 1: Detect edge at current position ✓
# Step 2: Align at current position ✓
# Step 3: Return to start (no movement needed)
```

**Problem**: The docstring said "move to (0,0)" which was confusing. The current position IS the (0,0) reference.

### After (CORRECT):
```python
# Step 0: Save current position as origin (0,0) ✓
# Step 1: Detect edge at current position (no movement) ✓
# Step 2: Align at current position (rotation only) ✓
# Step 3: Return to origin (already there) ✓
```

## Key Features

1. **No X/Y Movement**: The stage never moves X or Y during the entire pipeline
2. **Rotation Only**: Only R-axis moves during alignment
3. **Origin Reference**: Click position becomes (0,0) reference
4. **Edge Detection**: HSV color-based Canny + HoughLinesP
5. **Visual Feedback**: Edge overlay remains visible throughout

## Usage

```bash
# In the GUI:
# 1. Connect camera
# 2. Connect stage
# 3. Move stage to desired position
# 4. Click "Full Pipeline: Detect → Align → Return to (0,0)"
# 5. Watch status bar for progress
# 6. Pipeline completes with summary dialog
```

**Important**: The pipeline does NOT move the stage X or Y. It only rotates the sample at the position where you clicked the button.

## Technical Details

- **Edge Detection**: HSV color-based Canny + HoughLinesP
- **Alignment**: Closed-loop with ColorEdgeEstimator
- **Movement**: R-axis rotation only (absolute positioning)
- **Timeout**: 60 seconds per movement
- **Threading**: Runs in background daemon thread

## Coordinate System

- **Machine Coordinates**: Physical stage position (e.g., x=1000, y=2000)
- **Pipeline Origin**: Position where button was clicked (treated as 0,0)
- **Relative System**: All references are relative to click position

## Files Modified

- `chip_edge_detector.py`: Clarified that current position is the origin
- `PIPELINE_IMPLEMENTATION.md`: Updated documentation

## Common Misconception

❌ **Wrong**: "Move to machine (0,0), detect, align, return"  
✅ **Correct**: "Current position IS (0,0), detect, align (no movement needed)"

The stage does NOT move to physical (0,0). It stays at the position where you clicked the button.
