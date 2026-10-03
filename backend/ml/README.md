# ML workspace

No model is trained or used yet. Reference-card detection and perspective calibration
are deterministic OpenCV processing in `app/calibration.py`.

The next milestone will add:

1. loose-lens and framed-rim contour extraction selected by `capture_mode`;
2. the clearly labeled approximate 1.5 mm inward offset for framed mode;
3. A, B and perimeter from the calibrated scale;
4. a versioned training dataset and evaluation script with sources and licenses.

The web interface already sends the two inputs needed to route that logic:

- `capture_mode`: `loose` or `framed`;
- `eye`: `left` or `right`.
