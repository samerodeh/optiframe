export type CaptureMode = "loose" | "framed";
export type Eye = "left" | "right";

export interface CalibrationReady {
  status: "calibrated";
  card_size_mm: [number, number];
  corners_px: [number, number][];
  rectified_card_corners_px: [number, number][];
  source_to_rectified: number[][];
  source_to_card_mm: number[][];
  millimetres_per_pixel: number;
  confidence: number;
  preview_width_px: number;
  preview_height_px: number;
  preview_data_url: string;
}

export interface CalibrationRetry {
  status: "retry";
  reason: "card_missing" | "card_cropped" | "image_blurry" | "low_confidence";
  message: string;
}

export interface CaptureReceipt {
  capture_id: string;
  capture_mode: CaptureMode;
  eye: Eye;
  filename: string;
  width_px: number;
  height_px: number;
  size_bytes: number;
  status: "calibrated" | "retry";
  next_step: "lens_segmentation" | "retry_capture";
  calibration: CalibrationReady | CalibrationRetry;
}

export async function submitCapture(
  image: File,
  captureMode: CaptureMode,
  eye: Eye,
): Promise<CaptureReceipt> {
  const body = new FormData();
  body.append("image", image);
  body.append("capture_mode", captureMode);
  body.append("eye", eye);

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch("/api/captures", {
      method: "POST",
      body,
      signal: controller.signal,
    });

    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as
        | { detail?: unknown }
        | null;
      throw new Error(typeof payload?.detail === "string" ? payload.detail : "The image could not be calibrated. Please try again.");
    }

    return await response.json() as CaptureReceipt;
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error("Calibration took too long. Check your connection and try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}
