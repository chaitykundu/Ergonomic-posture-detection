import cv2
import numpy as np

# ---------------------------------------------
# Clean text with black outline + white fill
# ---------------------------------------------
def draw_text(img, text, pos, color=(255, 255, 255)):
    x, y = pos
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)


# ---------------------------------------------
# Clean small arrow
# ---------------------------------------------
def draw_arrow(img, start, end, color):
    cv2.arrowedLine(
        img, start, end,
        color, 2,
        tipLength=0.25,
        line_type=cv2.LINE_AA
    )


# ---------------------------------------------
# MAIN: Apply posture + workstation corrections
# ---------------------------------------------
def apply_correction_arrows(frame, posture_report, workstation_report, lm, W, H):

    # Helper: safe landmark read
    def L(index):
        return (int(lm[index].x * W), int(lm[index].y * H))

    # -------------------------------------------------
    # Prevent overlap by tracking last drawn label area
    # -------------------------------------------------
    used_positions = []

    def place_label(base_x, base_y):
        """
        Places text without overlapping previous texts.
        """
        y = base_y
        while any(abs(y - uy) < 25 for ux, uy in used_positions):
            y += 25  # shift downward until free
        used_positions.append((base_x, y))
        return base_x, y

    # =================================================
    # POSTURE CORRECTIONS (with proper spacing)
    # =================================================

    # ---------------- NECK ----------------
    if posture_report["neck_flexion"]["severity"] == "red":
        nose = L(0)
        end = (nose[0], nose[1] - 60)
        draw_arrow(frame, nose, end, (0, 0, 255))

        tx, ty = place_label(end[0] - 80, end[1] - 10)
        draw_text(frame, "Straighten Neck ↑", (tx, ty))


    # ---------------- ELBOW (shift upward) ----------------
    if posture_report["elbow_angle"]["severity"] == "red":
        elbow = L(13)
        end = (elbow[0] + 60, elbow[1] + 20)
        draw_arrow(frame, elbow, end, (0, 200, 255))

        tx, ty = place_label(end[0] - 40, end[1] - 20)
        draw_text(frame, "Reduce Elbow Extension", (tx, ty))


    # ---------------- WRIST (shift downward) ----------------
    if posture_report["wrist_deviation"]["severity"] == "red":
        wrist = L(15)
        end = (wrist[0] - 60, wrist[1])
        draw_arrow(frame, wrist, end, (255, 120, 0))

        tx, ty = place_label(end[0] - 80, end[1] + 20)
        draw_text(frame, "Neutral Wrist →", (tx, ty))


    # =================================================
    # WORKSTATION CORRECTIONS
    # =================================================

    # ---------------- MONITOR ----------------
    if "monitor" in workstation_report:
        for rule_id, rule in workstation_report["monitor"].items():
            if rule["severity"] == "red" and "bbox" in rule:
                x1, y1, x2, y2 = rule["bbox"]
                mid = ((x1+x2)//2, y1)
                end = (mid[0], mid[1] - 70)

                draw_arrow(frame, mid, end, (0, 255, 255))
                tx, ty = place_label(end[0] - 50, end[1] - 10)
                draw_text(frame, "Lower Monitor ↓", (tx, ty))


    # ---------------- DESK HEIGHT ----------------
    if "worksurface" in workstation_report:
        for rule_id, rule in workstation_report["worksurface"].items():
            if rule["severity"] == "red" and "bbox" in rule:
                x1, y1, x2, y2 = rule["bbox"]
                mid = ((x1+x2)//2, y2)
                end = (mid[0], mid[1] + 70)

                draw_arrow(frame, mid, end, (255, 0, 255))

                tx, ty = place_label(end[0] - 50, end[1] + 20)
                draw_text(frame, "Raise Desk ↑", (tx, ty))


    return frame
