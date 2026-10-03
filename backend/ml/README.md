# ML workspace

The measurement model is deliberately not implemented in capture milestone 1.

The next milestone will add:

1. reference-card corner detection and perspective rectification;
2. loose-lens and framed-rim contour extraction selected by `capture_mode`;
3. conversion from pixels to millimetres;
4. a versioned training dataset and evaluation script.

The web interface already sends the two inputs needed to route that logic:

- `capture_mode`: `loose` or `framed`;
- `eye`: `left` or `right`.

