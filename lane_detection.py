import cv2
import numpy as np
import os
from glob import glob

# Bright (white) detection threshold in HLS
LIGHTNESS_MIN = 180

# Edge detection thresholds
EDGE_LOW  = 50
EDGE_HIGH = 150

# Region of interest (trapezoid proportions)
ROI_HEIGHT_RATIO = 0.50
ROI_LEFT_TOP     = 0.35
ROI_RIGHT_TOP    = 0.60
ROI_LEFT_BOTTOM  = 0.15
ROI_RIGHT_BOTTOM = 1

# Hough transform settings
HOUGH_PARAMS = {
    "rho": 1,
    "theta": np.pi / 180,
    "threshold": 15,
    "min_len": 20,
    "max_gap": 30
}

# Slope constraints
LEFT_RANGE  = (-1.5, -0.35)
RIGHT_RANGE = (0.20, 0.80)
DASH_MIN    = 1.0
DASH_REGION = 0.60

# Drawing styles
LINE_COLOR  = (128, 0, 128)
LINE_WIDTH  = 3
DOT_COLOR   = (255, 192, 203)
DOT_SIZE    = 8


def extract_white_regions(image):
    """Create mask of bright pixels based on HLS lightness channel."""
    hls_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    light_channel = hls_img[:, :, 1]
    return cv2.inRange(light_channel, LIGHTNESS_MIN, 255)


def apply_roi_mask(edge_img):
    """Keep only trapezoidal region corresponding to road area."""
    height, width = edge_img.shape[:2]

    polygon = np.array([[
        (int(ROI_LEFT_BOTTOM * width), height),
        (int(ROI_LEFT_TOP * width), int(ROI_HEIGHT_RATIO * height)),
        (int(ROI_RIGHT_TOP * width), int(ROI_HEIGHT_RATIO * height)),
        (int(ROI_RIGHT_BOTTOM * width), height)
    ]], dtype=np.int32)

    mask = np.zeros_like(edge_img)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edge_img, mask)


def split_line_groups(hough_lines, width):
    """
    Separate detected line segments into left and right lane candidates.
    Falls back to dashed center lines if left edge is missing.
    """
    left_candidates = []
    right_candidates = []
    dashed_candidates = []

    if hough_lines is None:
        return left_candidates, right_candidates

    midpoint = width / 2

    for line in hough_lines:
        x1, y1, x2, y2 = line[0]

        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        center_x = (x1 + x2) / 2

        if abs(slope) < 0.15:
            continue

        if LEFT_RANGE[0] <= slope <= LEFT_RANGE[1] and center_x < midpoint:
            left_candidates.append([x1, y1, x2, y2])

        elif RIGHT_RANGE[0] <= slope <= RIGHT_RANGE[1] and center_x >= midpoint:
            right_candidates.append([x1, y1, x2, y2])

        elif slope >= DASH_MIN and center_x <= DASH_REGION * width:
            dashed_candidates.append([x1, y1, x2, y2])

    # fallback logic
    if not left_candidates:
        left_candidates = dashed_candidates

    return left_candidates, right_candidates


def approximate_lane_line(segments, y_bottom, y_top):
    """Fit a single line from multiple segments."""
    if not segments:
        return None

    points = np.array(segments).reshape(-1, 2)
    x_vals = points[:, 0].astype(float)
    y_vals = points[:, 1].astype(float)

    coeff = np.polyfit(y_vals, x_vals, 1)
    poly  = np.poly1d(coeff)

    return (
        int(poly(y_bottom)), int(y_bottom),
        int(poly(y_top)),    int(y_top)
    )


def process_single_image(path, save_dir):
    image = cv2.imread(path)

    if image is None:
        print(f"[WARN] Cannot load {path}")
        return

    h, w = image.shape[:2]
    top_y = int(ROI_HEIGHT_RATIO * h)
    bottom_y = h - 1

    # Step 1: isolate white
    mask = extract_white_regions(image)

    # Step 2: detect edges
    edges = cv2.Canny(mask, EDGE_LOW, EDGE_HIGH)

    # Step 3: apply ROI
    roi_edges = apply_roi_mask(edges)

    # Step 4: detect line segments
    lines = cv2.HoughLinesP(
        roi_edges,
        HOUGH_PARAMS["rho"],
        HOUGH_PARAMS["theta"],
        HOUGH_PARAMS["threshold"],
        minLineLength=HOUGH_PARAMS["min_len"],
        maxLineGap=HOUGH_PARAMS["max_gap"]
    )

    # Step 5: group + fit lines
    left_group, right_group = split_line_groups(lines, w)
    left_lane  = approximate_lane_line(left_group, bottom_y, top_y)
    right_lane = approximate_lane_line(right_group, bottom_y, top_y)

    # Step 6: draw results
    output = image.copy()

    if left_lane:
        cv2.line(output, left_lane[:2], left_lane[2:], LINE_COLOR, LINE_WIDTH)

    if right_lane:
        cv2.line(output, right_lane[:2], right_lane[2:], LINE_COLOR, LINE_WIDTH)

    # Step 7: draw center markers
    if left_lane and right_lane:
        bottom_center = (
            (left_lane[0] + right_lane[0]) // 2,
            bottom_y
        )
        top_center = (
            (left_lane[2] + right_lane[2]) // 2,
            top_y
        )

        cv2.circle(output, bottom_center, DOT_SIZE, DOT_COLOR, -1)
        cv2.circle(output, top_center, DOT_SIZE, DOT_COLOR, -1)

    # Step 8: save output
    filename = os.path.basename(path)
    output_path = os.path.join(save_dir, filename)
    cv2.imwrite(output_path, output)

    detected = []
    if left_lane: detected.append("left")
    if right_lane: detected.append("right")

    label = ", ".join(detected) if detected else "no lanes"
    print(f"{filename} -> ({label})")


def run_pipeline():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    input_dir  = os.path.join(base_dir, "Images")
    output_dir = os.path.join(base_dir, "Output")

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, "*.jpg")))

    if not files:
        print("No input images found.")
        return

    print(f"Processing {len(files)} images...\n")

    for img_path in files:
        process_single_image(img_path, output_dir)

    print(f"\nFinished. Results saved in: {output_dir}")


if __name__ == "__main__":
    run_pipeline()