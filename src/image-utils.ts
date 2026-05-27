/** Image helpers: client-side downscaling and dimension probing. */

export interface Dimensions {
  width: number;
  height: number;
}

export async function getDimensions(blob: Blob): Promise<Dimensions> {
  const bitmap = await createImageBitmap(blob);
  const dims = { width: bitmap.width, height: bitmap.height };
  bitmap.close();
  return dims;
}

/**
 * Downscale an image so its longest edge is at most `maxEdge`, preserving aspect ratio.
 * Returns the original blob unchanged if it's already within bounds. Used before sending
 * to GPU backends to save compute and avoid VRAM blowups, the remote engine then
 * upscales 4×, still yielding a high-resolution result.
 */
export async function downscaleIfNeeded(blob: Blob, maxEdge: number): Promise<Blob> {
  const bitmap = await createImageBitmap(blob);
  const { width, height } = bitmap;
  const longest = Math.max(width, height);
  if (longest <= maxEdge) {
    bitmap.close();
    return blob;
  }

  const ratio = maxEdge / longest;
  const w = Math.round(width * ratio);
  const h = Math.round(height * ratio);
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap.close();
    return blob;
  }
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  return canvas.convertToBlob({ type: "image/png" });
}
