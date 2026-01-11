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
    # Track occupied regions to prevent overlap
    # -------------------------------------------------
    occupied_regions = []

    def is_region_free(x, y, text_width=250, text_height=30):
        """Check if a text region is free from other labels."""
        for (ox, oy, ow, oh) in occupied_regions:
            # Check if regions overlap
            if not (x + text_width < ox or x > ox + ow or 
                    y + text_height < oy or y > oy + oh):
                return False
        return True

    def find_free_position(preferred_x, preferred_y, text_width=250, text_height=30, max_attempts=15):
        """Find a free position near the preferred location."""
        x, y = preferred_x, preferred_y
        
        # Try vertical offsets first
        for offset in range(0, max_attempts * 35, 35):
            # Try below
            if is_region_free(x, y + offset, text_width, text_height):
                occupied_regions.append((x, y + offset, text_width, text_height))
                return x, y + offset
            
            # Try above
            if offset > 0 and is_region_free(x, y - offset, text_width, text_height):
                occupied_regions.append((x, y - offset, text_width, text_height))
                return x, y - offset
        
        # If vertical doesn't work, try horizontal shifts
        for x_offset in [50, -50, 100, -100]:
            if is_region_free(x + x_offset, y, text_width, text_height):
                occupied_regions.append((x + x_offset, y, text_width, text_height))
                return x + x_offset, y
        
        # Last resort: use original position
        occupied_regions.append((x, y, text_width, text_height))
        return x, y

    # =================================================
    # POSTURE CORRECTIONS
    # =================================================

    # ---------------- NECK ----------------
    if posture_report.get("neck_flexion", {}).get("severity") == "red":
        nose = L(0)
        end = (nose[0], nose[1] - 60)
        draw_arrow(frame, nose, end, (0, 0, 255))

        tx, ty = find_free_position(end[0] - 80, end[1] - 10)
        draw_text(frame, "Straighten Neck", (tx, ty))

    # ---------------- ELBOW ----------------
    if posture_report.get("elbow_angle", {}).get("severity") == "red":
        elbow = L(13)
        end = (elbow[0] + 60, elbow[1] + 20)
        draw_arrow(frame, elbow, end, (0, 200, 255))

        tx, ty = find_free_position(end[0] - 40, end[1] - 20)
        draw_text(frame, "Reduce Elbow Extension", (tx, ty))

    # ---------------- WRIST ----------------
    if posture_report.get("wrist_deviation", {}).get("severity") == "red":
        wrist = L(15)
        end = (wrist[0] - 60, wrist[1])
        draw_arrow(frame, wrist, end, (255, 120, 0))

        tx, ty = find_free_position(end[0] - 80, end[1] + 20)
        draw_text(frame, "Neutral Wrist", (tx, ty))

    # =================================================
    # WORKSTATION CORRECTIONS
    # =================================================

    # ---------------- MONITOR ----------------
    if "monitor" in workstation_report:
        for rule_id, rule in workstation_report["monitor"].items():
            if rule.get("severity") == "red" and "bbox" in rule:
                x1, y1, x2, y2 = rule["bbox"]
                mid = ((x1+x2)//2, y1)
                end = (mid[0], mid[1] - 70)

                draw_arrow(frame, mid, end, (0, 255, 255))
                tx, ty = find_free_position(end[0] - 50, end[1] - 10)
                draw_text(frame, "Lower Monitor", (tx, ty))

    # ---------------- DESK HEIGHT ----------------
    if "worksurface" in workstation_report:
        for rule_id, rule in workstation_report["worksurface"].items():
            if rule.get("severity") == "red" and "bbox" in rule:
                x1, y1, x2, y2 = rule["bbox"]
                mid = ((x1+x2)//2, y2)
                end = (mid[0], mid[1] + 70)

                draw_arrow(frame, mid, end, (255, 0, 255))
                tx, ty = find_free_position(end[0] - 50, end[1] + 20)
                draw_text(frame, "Raise Desk", (tx, ty))

    return frame