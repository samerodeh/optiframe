export type CaptureMode = "loose" | "framed";
export type Eye = "left" | "right";

export interface CaptureReceipt {
  capture_id: string;
  capture_mode: CaptureMode;
  eye: Eye;
  filename: string;
  width_px: number;
  height_px: number;
  size_bytes: number;
  status: "received";
  next_step: "reference_detection";
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

  const response = await fetch("/api/captures", {
    method: "POST",
    body,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { detail?: string }
      | null;
    throw new Error(payload?.detail ?? "The image could not be uploaded.");
  }

  return response.json() as Promise<CaptureReceipt>;
}
