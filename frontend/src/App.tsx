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

    if (!file.type.startsWith("image/")) {
      setError("Choose an original JPG, PNG, HEIC, or WebP image.");
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
            <span>Step 1 of 4</span>
            <span>Capture</span>
          </div>
          <div className="progress-track"><span /></div>
          <ol className="progress-labels">
            <li className="active">Capture</li>
            <li>Measure</li>
            <li>Design</li>
            <li>Export</li>
          </ol>
        </section>

        <section className="hero">
          <div className="eyebrow"><Sparkles size={14} /> Lens measurement</div>
          <h1>Photograph your lens</h1>
          <p>Place it beside a standard-size card so we can turn image pixels into real millimetres.</p>
        </section>

        <section className="capture-card">
          <fieldset className="field-group">
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
                <small>Measures the visible edge directly</small>
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
                <small>Uses an approximate 1.5 mm inset</small>
              </button>
            </div>
          </fieldset>

          <fieldset className="field-group compact">
            <legend>Which eye is this for?</legend>
            <div className="segmented-control">
              <button
                type="button"
                className={eye === "left" ? "active" : ""}
                onClick={() => { setEye("left"); setReceipt(null); }}
              >Left eye</button>
              <button
                type="button"
                className={eye === "right" ? "active" : ""}
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
                <p>Keep the full {captureMode === "framed" ? "frame" : "lens"} and all four card corners visible.</p>
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
                <img src={previewUrl} alt="Selected lens and reference card" />
                <button className="replace-button" type="button" onClick={resetCapture}>
                  <RotateCcw size={17} /> Replace
                </button>
                <div className="file-summary">
                  <span>{imageFile.name}</span>
                  <span>{(imageFile.size / 1024 / 1024).toFixed(1)} MB</span>
                </div>
              </div>
            )}

            <input ref={cameraInput} hidden type="file" accept="image/*" capture="environment" onChange={chooseImage} />
            <input ref={galleryInput} hidden type="file" accept="image/*" onChange={chooseImage} />
          </div>

          <ul className="checklist" aria-label="Photo checklist">
            <li><Check size={16} /> The card is flat and all four corners are visible</li>
            <li><Check size={16} /> The full {captureMode === "framed" ? "frame" : "lens edge"} is inside the photo</li>
            <li><Check size={16} /> The image is sharp and has no strong glare</li>
          </ul>

          {captureMode === "framed" && (
            <div className="notice">
              <ShieldCheck size={19} />
              <p><strong>Approximation mode</strong>The next step will move the detected rim inward by 1.5 mm.</p>
            </div>
          )}

          {error && <p className="error-message" role="alert">{error}</p>}

          {receipt ? (
            <div className="success-card" role="status">
              <span><Check size={20} /></span>
              <div>
                <strong>Photo ready for measurement</strong>
                <p>{receipt.width_px} × {receipt.height_px} px · {receipt.eye} eye · {receipt.capture_mode} mode</p>
              </div>
            </div>
          ) : (
            <button
              className="continue-button"
              type="button"
              disabled={!imageFile || isSubmitting}
              onClick={continueCapture}
            >
              {isSubmitting ? "Checking image…" : "Continue to measurement"}
              {!isSubmitting && <ChevronRight size={20} />}
            </button>
          )}
        </section>

        <p className="privacy-note"><ShieldCheck size={15} /> This prototype validates your image without saving it.</p>
      </main>
    </div>
  );
}

export default App;
