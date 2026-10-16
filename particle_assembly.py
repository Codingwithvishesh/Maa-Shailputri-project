import cv2
import numpy as np
import os
import sys

# ---- Timing & Configuration -------------------------------------------
TOTAL_FRAMES = 350    # particle flight duration (~7 seconds at 50 FPS)
BLEND_FRAMES = 50     # fade into the final picture (~1.0 second)
FRAME_MS = 20         # delay per frame in milliseconds (display mode: ~50 FPS)
MAX_DELAY = 0.5       # wider departure stagger (0 to 1) for extended flight
PARTICLE_STEP = 3     # sample every N pixels (lower = sharper picture)
CANVAS_BG = (20, 6, 10)  # Deep violet temple background (BGR)


def run_flying_particles_assembly(image_path="shailputri.jpg", save_video_path=None):
    if not os.path.exists(image_path):
        fallback = os.path.join("public", "character.jpg")
        if os.path.exists(fallback):
            print(f"Notice: '{image_path}' not found. Using fallback '{fallback}'.")
            image_path = fallback
        else:
            print(f"Error: Could not find image '{image_path}'.")
            return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read '{image_path}'. Ensure it is a valid image.")
        return

    # Resize image to fit comfortably on screen
    orig_H, orig_W = img.shape[:2]
    max_display_h = 840
    if orig_H > max_display_h:
        scale = max_display_h / float(orig_H)
        img = cv2.resize(img, (int(orig_W * scale), max_display_h), interpolation=cv2.INTER_AREA)

    H, W, _ = img.shape
    print(f"Loaded image: {image_path} ({W}x{H})")

    # Interactive GUI Window
    window_name = "Navratri Day 1 - Shailputri Dot Reveal"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Video Writer (optional export)
    writer = None
    if save_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(1000 / FRAME_MS)
        writer = cv2.VideoWriter(save_video_path, fourcc, fps, (W, H))
        print(f"Recording animation to '{save_video_path}' at {fps} FPS...")

    # Grid sampling
    step = PARTICLE_STEP
    y_coords, x_coords = np.mgrid[0:H:step, 0:W:step]
    targets = np.column_stack((x_coords.ravel(), y_coords.ravel())).astype(np.float32)
    colors = img[y_coords.ravel(), x_coords.ravel()]  # shape (N, 3), uint8 (BGR)
    num_particles = len(targets)
    print(f"Assembling {num_particles:,} particles with vortex curves...")

    # Spawn points from outer perimeter with circular swirl angles
    angles = np.random.uniform(0, 2 * np.pi, size=num_particles)
    dists = np.maximum(W, H) * np.random.uniform(0.7, 1.4, size=num_particles)
    starts = np.column_stack((
        W / 2.0 + np.cos(angles) * dists,
        H / 2.0 + np.sin(angles) * dists
    )).astype(np.float32)

    # Perpendicular vectors for curved spiral trajectories
    dx = targets[:, 0] - starts[:, 0]
    dy = targets[:, 1] - starts[:, 1]
    lengths = np.sqrt(dx**2 + dy**2) + 1e-5
    px = -dy / lengths
    py = dx / lengths
    swirl_mags = np.random.uniform(-90.0, 90.0, size=num_particles).astype(np.float32)

    # Saffron-pink color for early energy phase (BGR: 171, 143, 255)
    gold_bgr = np.array([171, 143, 255], dtype=np.float32)

    # Organic delay so points fly in waves
    delays = np.random.uniform(0.0, MAX_DELAY, size=num_particles).astype(np.float32)

    print("Playing advanced cosmic vortex assembly... Press ESC inside the window to exit.")

    # Flight phase (Vectorized curved vortex dynamics)
    for frame_idx in range(TOTAL_FRAMES):
        canvas = np.full((H, W, 3), CANVAS_BG, dtype=np.uint8)

        t_global = frame_idx / float(TOTAL_FRAMES - 1)
        t_local = (t_global - delays) / (1.0 - delays)
        t_local = np.clip(t_local, 0.0, 1.0)

        # Smooth quartic ease-out curve
        ease_progress = 1.0 - np.power(1.0 - t_local, 4)

        # Base interpolated coordinates
        cur_x = starts[:, 0] + (targets[:, 0] - starts[:, 0]) * ease_progress
        cur_y = starts[:, 1] + (targets[:, 1] - starts[:, 1]) * ease_progress

        # Add harmonic vortex wave displacement
        wave = np.sin(t_local * np.pi) * (1.0 - t_local)
        cur_x += px * swirl_mags * wave
        cur_y += py * swirl_mags * wave

        cur_xi = np.clip(cur_x.astype(np.int32), 1, W - 2)
        cur_yi = np.clip(cur_y.astype(np.int32), 1, H - 2)

        # Dynamic color transition: Golden energy sparks -> Target pixel colors
        gold_factor = np.clip((0.55 - t_local) / 0.55, 0.0, 1.0)[:, np.newaxis]
        frame_colors = (colors * (1.0 - gold_factor) + gold_bgr * gold_factor).astype(np.uint8)

        # Vectorized crisp particle drawing with golden halo
        canvas[cur_yi, cur_xi] = frame_colors
        canvas[cur_yi + 1, cur_xi] = frame_colors
        canvas[cur_yi - 1, cur_xi] = frame_colors
        canvas[cur_yi, cur_xi + 1] = frame_colors
        canvas[cur_yi, cur_xi - 1] = frame_colors

        if writer:
            writer.write(canvas)

        cv2.imshow(window_name, canvas)
        if cv2.waitKey(FRAME_MS) & 0xFF == 27:
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            return

    # Seamless crossfade into final high-res picture
    last_canvas = canvas.copy()
    for alpha in np.linspace(0.0, 1.0, BLEND_FRAMES):
        blended = cv2.addWeighted(img, float(alpha), last_canvas, 1.0 - float(alpha), 0)
        if writer:
            writer.write(blended)
        cv2.imshow(window_name, blended)
        if cv2.waitKey(FRAME_MS) & 0xFF == 27:
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            return

    if writer:
        for _ in range(int(1000 / FRAME_MS)):
            writer.write(img)
        writer.release()
        print(f"Video saved successfully to '{save_video_path}'!")

    cv2.imshow(window_name, img)
    print("Cosmic Assembly complete! Press any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    target_img = sys.argv[1] if len(sys.argv) > 1 else "shailputri.jpg"
    video_out = sys.argv[2] if len(sys.argv) > 2 else None
    run_flying_particles_assembly(target_img, video_out)
