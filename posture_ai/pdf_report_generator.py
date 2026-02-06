# pdf_report_generator.py
import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from PIL import Image


def draw_wrapped_text(c, text, x_left=50, x_right=550, y_start=750,
                      line_spacing=14, font_name="Helvetica", font_size=10):
    """Draw text with perfect word wrapping. Returns new y-position."""
    c.setFont(font_name, font_size)
    max_width = x_right - x_left
    y = y_start

    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]

    for i, line in enumerate(lines):
        if i > 0:
            y -= 8  # space between paragraphs

        words = line.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            if c.stringWidth(test_line, font_name, font_size) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    c.drawString(x_left, y, current_line)
                    y -= line_spacing
                current_line = word

                if y < 80:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = 750

        if current_line:
            c.drawString(x_left, y, current_line)
            y -= line_spacing

    return y - 10  # padding after section


def add_section(c, title, content, y_position):
    """Add section with bold title + wrapped content."""
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y_position, title)
    c.setStrokeColor(colors.lightgrey)
    c.line(50, y_position - 4, 280, y_position - 4)
    y_position -= 30

    if isinstance(content, list):
        content = "\n".join(content)

    return draw_wrapped_text(c, content, y_start=y_position)


def generate_pdf_report(final_iso_report, ai_report, image_path="output/annotated_1.jpg"):
    """
    Generate a beautiful, unique, multi-page ergonomics report.
    Saved as: output/ergonomics_report_2025-04-05_14-30-22.pdf
    """
    # Create output folder if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Unique filename with timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_pdf = f"output/ergonomics_report_{timestamp}.pdf"

    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    # === HEADER ===
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.HexColor("#1a5fb4"))
    c.drawCentredString(width / 2, 770, "POSTURA")
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, 745, "Ergonomics Evaluation Report")
    y_pos = 710

    # === POSTURE ANALYSIS ===
    posture_lines = []
    for joint, data in final_iso_report.get("posture", {}).items():
        sev = data.get("severity", "unknown").capitalize()
        ang = data.get("angle", "N/A")
        joint_name = joint.replace("_", " ").title()
        posture_lines.append(f"• {joint_name}: {sev}")
    y_pos = add_section(c, "Posture Analysis", "\n".join(posture_lines), y_pos)

    # === WORKSTATION ANALYSIS ===
    ws_lines = []
    for component, rules in final_iso_report.get("workstation", {}).items():
        comp_name = component.replace("_", " ").title()

        for rule_id, rule in rules.items():
            sev = rule.get("severity", "unknown").capitalize()
            rule_name = rule_id.replace("_", " ")

            ws_lines.append(
                f"• {comp_name}: {rule_name}: {sev}"
            )
    y_pos = add_section(c, "Workstation Analysis", "\n".join(ws_lines), y_pos)

    # === RISK SUMMARY ===
    # risk_summary = ai_report.get("risk_summary", "No risk summary available.")
    # y_pos = add_section(c, "Risk Summary", risk_summary, y_pos)
    
    # === PAIN SUMMARY ===
    pain_summary_lines = []

    avg_pain = ai_report.get("exercise_recommendations", {}).get("average_pain_vas")
    if avg_pain:
        pain_summary_lines.append(f"Average Pain Level: {avg_pain}")

    main_region = (
        ai_report.get("exercise_recommendations", {}).get("main_pain_region")
        or ai_report.get("exercise_recommendations", {}).get("main_pain_regions")
    )

    if main_region:
        if "region" in main_region:
            region_text = main_region["region"]
        elif "regions" in main_region:
            region_text = ", ".join(main_region["regions"])
        else:
            region_text = "Unknown"

        pain_summary_lines.append(
            f"Main Pain Region(s): {region_text} (VAS {main_region.get('vas', 'N/A')})"
        )


    future_risk = ai_report.get("exercise_recommendations", {}).get("future_risk")
    if future_risk:
        pain_summary_lines.append(f"Future Risk: {future_risk['label']}")

    if pain_summary_lines:
        y_pos = add_section(
            c,
            "Pain & Risk Summary",
            "\n".join(pain_summary_lines),
            y_pos
        )

    # === AI POSTURE & ERGONOMIC CORRECTIONS ===
    corrections = ai_report.get("corrections", [])

    if corrections:
        correction_lines = []

        for item in corrections:
            title = item.get("title")
            desc = item.get("description")

            if title and desc:
                correction_lines.append(
                    f"• {title}: {desc}"
                )
            elif title:
                correction_lines.append(f"• {title}")
            elif desc:
                correction_lines.append(f"• {desc}")

        y_pos = add_section(
            c,
            "Posture & Ergonomic Corrections",
            "\n".join(correction_lines),
            y_pos
        )


    # === EXERCISE RECOMMENDATIONS ===
    # exercises = ai_report.get("exercise_recommendations", [])
    # if not exercises:
    #     exercises = ["No specific exercises recommended at this time."]
    # y_pos = add_section(c, "Exercise Recommendations", "\n\n".join(exercises), y_pos)

    # === FINAL ADVICE ===
    final_advice = ai_report.get("final_advice", "No final advice available.")
    y_pos = add_section(c, "Final Advice", final_advice, y_pos)

    # === LARGE BEAUTIFUL ANNOTATED IMAGE ===
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            img_ratio = img_width / img_height

            max_image_width = 520
            max_image_height = 620
            available_height = y_pos - 100

            # Try to fit large image on current page
            display_width = max_image_width
            display_height = display_width / img_ratio

            if display_height > available_height or display_height > max_image_height:
                display_height = min(available_height, max_image_height)
                display_width = display_height * img_ratio
                if display_width > max_image_width:
                    display_width = max_image_width
                    display_height = display_width / img_ratio

            # If not enough space → new page (image gets full glory)
            if available_height < display_height + 80:
                c.showPage()
                display_width = max_image_width
                display_height = min(max_image_height, display_width / img_ratio)
                y_pos = 780

            img_x = (width - display_width) / 2
            img_y = y_pos - display_height - 40

            c.drawImage(image_path, img_x, img_y,
                        width=display_width, height=display_height,
                        preserveAspectRatio=True, mask='auto')

            # Caption + subtle border
            c.setFont("Helvetica-Oblique", 11)
            c.setFillColor(colors.grey)
            c.drawCentredString(width / 2, img_y - 25, "Annotated posture analysis from your uploaded photo")

            c.setStrokeColor(colors.lightgrey)
            c.setLineWidth(0.5)
            c.rect(img_x - 5, img_y - 5, display_width + 10, display_height + 10)

        except Exception as e:
            print(f"Warning: Could not insert image: {e}")
    else:
        print(f"Warning: Image not found: {image_path}")

    c.save()
    print(f"PDF report generated: {output_pdf}")
    return output_pdf