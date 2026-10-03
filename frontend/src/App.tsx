import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  Check,
  ChevronRight,
  CircleHelp,
  Glasses,
  ImagePlus,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import {
  CaptureMode,
  CaptureReceipt,
  Eye,
  submitCapture,
} from "./api";

const MAX_FILE_SIZE = 4 * 1024 * 1024;

function App() {
  const [captureMode, setCaptureMode] = useState<CaptureMode>("loose");
  const [eye, setEye] = useState<Eye>("left");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [receipt, setReceipt] = useState<CaptureReceipt | null>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const galleryInput = useRef<HTMLInputElement>(null);
  const calibration = receipt?.calibration;
  const measurement = receipt?.measurement;

  const previewUrl = useMemo(
    () => (imageFile ? URL.createObjectURL(imageFile) : ""),
    [imageFile],
  );

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"].includes(file.type)) {
      setError("Choose an original JPG, PNG, HEIC, HEIF, or WebP image.");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError("This image is larger than 4 MB. Retake it at a smaller resolution or choose a smaller original.");
      return;
    }

    setError("");
    setReceipt(null);
    setImageFile(file);
  }

  function resetCapture() {
    setImageFile(null);
    setReceipt(null);
    setError("");
  }

  async function continueCapture() {
    if (!imageFile) {
      setError("Take or choose a photo first.");
      return;
    }

    setIsSubmitting(true);
    setError("");
    setReceipt(null);
    try {
      const response = await submitCapture(imageFile, captureMode, eye);
      setReceipt(response);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "We could not validate the image. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="OptiFrame home">
          <span className="brand-mark"><Glasses size={25} strokeWidth={2.25} /></span>
          <span>OptiFrame</span>
        </a>
        <a className="help-button" href="#photo-guide" aria-label="Jump to capture guide">
          <CircleHelp size={20} />
        </a>
      </header>

      <main id="top">
        <section className="progress-wrap" aria-label="Progress">
          <div className="progress-copy">
            <span>Step 2 of 4</span>
            <span>Measure</span>
          </div>
          <div className="progress-track"><span /></div>
          <ol className="progress-labels">
            <li>Capture</li>
            <li className="active">Measure</li>
            <li>Design</li>
            <li>Export</li>
          </ol>
        </section>

        <section className="hero">
          <div className="eyebrow"><Sparkles size={14} /> Lens measurement</div>
          <h1>Measure your lens</h1>
          <p>Photograph one target lens beside a standard-size card to estimate its width, height and perimeter.</p>
        </section>

        <section className="capture-card">
          <fieldset className="field-group" disabled={isSubmitting}>
            <legend>How is the lens positioned?</legend>
            <div className="choice-grid">
              <button
                className={`choice-card ${captureMode === "loose" ? "selected" : ""}`}
                type="button"
                onClick={() => { setCaptureMode("loose"); setReceipt(null); }}
                aria-pressed={captureMode === "loose"}
              >
                <span className="radio-dot" />
                <ScanLine size={26} />
                <strong>Loose lens</strong>
                <small>Measures the detected lens edge directly</small>
              </button>
              <button
                className={`choice-card ${captureMode === "framed" ? "selected" : ""}`}
                type="button"
                onClick={() => { setCaptureMode("framed"); setReceipt(null); }}
                aria-pressed={captureMode === "framed"}
              >
                <span className="radio-dot" />
                <Glasses size={26} />
                <strong>Inside a frame</strong>
                <small>Approximate 1.5 mm inward rim offset</small>
              </button>
            </div>
          </fieldset>

          <fieldset className="field-group compact" disabled={isSubmitting}>
            <legend>Which eye is this for?</legend>
            <div className="segmented-control">
              <button
                type="button"
                className={eye === "left" ? "active" : ""}
                aria-pressed={eye === "left"}
                onClick={() => { setEye("left"); setReceipt(null); }}
              >Left eye</button>
              <button
                type="button"
                className={eye === "right" ? "active" : ""}
                aria-pressed={eye === "right"}
                onClick={() => { setEye("right"); setReceipt(null); }}
              >Right eye</button>
            </div>
            <p className="orientation-note">Choose from the wearer’s point of view.</p>
          </fieldset>

          <div className="field-group" id="photo-guide">
            <div className="section-heading">
              <h2>Take the photo</h2>
              <span>Original images only</span>
            </div>

            {!imageFile ? (
              <div className="capture-stage">
                <div className="guide-scene" aria-hidden="true">
                  <div className="guide-card">
                    <span>CARD</span>
                    <small>85.60 × 53.98 mm</small>
                  </div>
                  <div className={`guide-lens ${captureMode === "framed" ? "with-frame" : ""}`}>
                    {captureMode === "framed" && <span className="bridge-line" />}
                  </div>
                  <span className="corner top-left" />
                  <span className="corner top-right" />
                  <span className="corner bottom-left" />
                  <span className="corner bottom-right" />
                </div>
                <p>Keep one target {captureMode === "framed" ? "rim" : "lens"} and all four card corners visible.</p>
                <div className="capture-actions">
                  <button className="primary-button" type="button" onClick={() => cameraInput.current?.click()}>
                    <Camera size={20} /> Open camera
                  </button>
                  <button className="secondary-button" type="button" onClick={() => galleryInput.current?.click()}>
                    <ImagePlus size={20} /> Choose original
                  </button>
                </div>
              </div>
            ) : (
              <div className="preview-stage">
                {imageFile.type === "image/heic" || imageFile.type === "image/heif" ? (
                  <p className="preview-placeholder">HEIC/HEIF selected. The calibrated preview will appear after processing.</p>
                ) : <img src={previewUrl} alt="Selected lens and reference card" />}
                <button className="replace-button" type="button" disabled={isSubmitting} onClick={resetCapture}>
                  <RotateCcw size={17} /> Replace
                </button>
                <div className="file-summary">
                  <span>{imageFile.name}</span>
                  <span>{(imageFile.size / 1024 / 1024).toFixed(1)} MB</span>
                </div>
              </div>
            )}

            <input ref={cameraInput} hidden disabled={isSubmitting} type="file" accept="image/*" capture="environment" onChange={chooseImage} />
            <input ref={galleryInput} hidden disabled={isSubmitting} type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif" onChange={chooseImage} />
          </div>

          <ul className="checklist" aria-label="Photo checklist">
            <li><Check size={16} /> One blank 85.60 × 53.98 mm card, with space around all four corners</li>
            <li><Check size={16} /> Card and lens share a flat, plain contrasting surface; camera nearly overhead</li>
            <li><Check size={16} /> Align the lens top edge with the card’s long edge for correct boxing dimensions</li>
            <li><Check size={16} /> One complete target {captureMode === "framed" ? "rim" : "lens edge"} is visible and separate from the card</li>
            <li><Check size={16} /> The image is sharp and has no strong glare</li>
          </ul>

          {captureMode === "framed" && (
            <div className="notice">
              <ShieldCheck size={19} />
              <p><strong>Approximation mode</strong>The app detects the outside of one rim and offsets that polygon inward by 1.5 mm. The reported lens dimensions are approximate.</p>
            </div>
          )}

          {error && <p className="error-message" role="alert">{error}</p>}

          {calibration?.status === "retry" && (
            <div className="error-message" role="alert">
              <strong>Retake photo — calibration incomplete</strong>
              <p>{calibration.message}</p>
              <button className="secondary-button" type="button" onClick={resetCapture}>Choose a new photo</button>
            </div>
          )}

          {measurement?.status === "retry" && (
            <div className="error-message" role="alert">
              <strong>Retake photo — lens measurement incomplete</strong>
              <p>{measurement.message}</p>
              <button className="secondary-button" type="button" onClick={resetCapture}>Choose a new photo</button>
            </div>
          )}

          {calibration?.status === "calibrated" && receipt ? (
            <section className="calibration-result" aria-label="Calibration result">
              <div className="success-card" role="status">
                <span><Check size={20} /></span>
                <div>
                  <strong>{measurement?.status === "measured" ? "Lens measured" : "Reference card calibrated"}</strong>
                  <p>{receipt.eye} eye · {receipt.capture_mode === "framed" ? "inside a frame (approximation mode)" : "loose lens"}</p>
                </div>
              </div>
              <figure className="rectified-preview">
                <div className="rectified-image">
                  <img src={calibration.preview_data_url} alt="Top-down rectified photo with detected reference card and lens overlays" />
                  <svg viewBox={`0 0 ${calibration.preview_width_px} ${calibration.preview_height_px}`} aria-hidden="true">
                    <polygon className="card-outline" points={calibration.rectified_card_corners_px.map(point => point.join(",")).join(" ")} />
                    {calibration.rectified_card_corners_px.map(([x, y], index) => (
                      <circle className="card-corner" key={index} cx={x} cy={y} r={calibration.preview_width_px / 130} />
                    ))}
                    {measurement?.status === "measured" && measurement.detected_outer_contour_px && (
                      <polygon className="outer-contour" points={measurement.detected_outer_contour_px.map(point => point.join(",")).join(" ")} />
                    )}
                    {measurement?.status === "measured" && (
                      <polygon className="lens-contour" points={measurement.contour_px.map(point => point.join(",")).join(" ")} />
                    )}
                  </svg>
                </div>
                <figcaption>
                  Top-down preview · green marks the card · {measurement?.status === "measured" ? "orange marks the measured lens contour" : "no reliable lens contour was found"}
                </figcaption>
              </figure>
              {measurement?.status === "measured" && (
                <dl className="measurement-grid" aria-label="Lens measurements">
                  <div><dt>Width A</dt><dd>{measurement.width_a_mm.toFixed(2)} <small>mm</small></dd></div>
                  <div><dt>Height B</dt><dd>{measurement.height_b_mm.toFixed(2)} <small>mm</small></dd></div>
                  <div><dt>Perimeter</dt><dd>{measurement.perimeter_mm.toFixed(2)} <small>mm</small></dd></div>
                </dl>
              )}
              <dl className="calibration-details">
                <div><dt>Reference card</dt><dd>85.60 × 53.98 mm</dd></div>
                <div><dt>Preview scale</dt><dd>{calibration.millimetres_per_pixel.toFixed(4)} mm / pixel</dd></div>
                {measurement?.status === "measured" && <div><dt>Method</dt><dd>{measurement.method === "ml_lens_mask" ? "ML lens mask" : measurement.method === "ml_rim_inset" ? "ML rim mask − 1.5 mm (approximate)" : measurement.approximate ? "Outer rim − 1.5 mm (approximate)" : "Direct detected contour"}</dd></div>}
              </dl>
              <p className="calibration-note">
                {measurement?.status === "measured"
                  ? measurement.approximate
                    ? "Approximate framed result. Confirm the orange inset follows the expected lens edge. Real-photo accuracy has not been validated yet."
                    : "Confirm the orange contour follows the lens edge. Real-photo accuracy has not been validated yet."
                  : "Card calibration succeeded, but the lens contour was not reliable enough to report dimensions."}
              </p>
              <button className="secondary-button" type="button" onClick={resetCapture}><RotateCcw size={17} /> Take another photo</button>
            </section>
          ) : (
            <button
              className="continue-button"
              type="button"
              disabled={!imageFile || isSubmitting}
              onClick={continueCapture}
            >
              {isSubmitting ? "Calibrating and measuring…" : receipt?.status === "retry" ? "Try measurement again" : "Measure lens"}
              {!isSubmitting && <ChevronRight size={20} />}
            </button>
          )}
        </section>

        <p className="privacy-note"><ShieldCheck size={15} /> Photos are processed in memory and are not saved.</p>
      </main>
    </div>
  );
}

export default App;
