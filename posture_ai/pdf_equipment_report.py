import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas


def generate_equipment_pdf(equipment_list):
    """
    Generate a standalone PDF report containing ONLY
    ergonomic equipment recommendations.
    """

    # Ensure output folder exists
    os.makedirs("output", exist_ok=True)

    # Create unique filename
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_pdf = f"output/equipment_report_{timestamp}.pdf"

    # Create PDF canvas
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    y = 760

    # =========================
    # HEADER
    # =========================
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.HexColor("#1a5fb4"))
    c.drawCentredString(width / 2, y, "POSTURA")
    y -= 28

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, y, "Equipment Recommendation Report")
    y -= 40

    # =========================
    # EMPTY STATE
    # =========================
    if not equipment_list:
        c.setFont("Helvetica", 12)
        c.drawString(50, y, "No equipment recommendations were generated.")
        c.save()
        return output_pdf

    # =========================
    # EQUIPMENT LIST
    # =========================
    for idx, item in enumerate(equipment_list, start=1):

        # Page break protection
        if y < 140:
            c.showPage()
            y = 760

        name = item.get("name", "Equipment")
        priority = item.get("priority", "medium").upper()
        description = item.get("description", "")
        why = item.get("why_recommended", "")
        improvement = item.get("improvement_percentage", "")
        price = item.get("price_range", "")
        source = item.get("source", "")
        target_issue = item.get("target_issue", "")

        # Priority color
        priority_color = {
            "HIGH": colors.red,
            "MEDIUM": colors.orange,
            "LOW": colors.green,
        }.get(priority, colors.black)

        # ---- Equipment Title ----
        c.setFont("Helvetica-Bold", 13)
        c.setFillColor(colors.black)
        c.drawString(50, y, f"{idx}. {name}")
        y -= 16

        # ---- Priority ----
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(priority_color)
        c.drawString(50, y, f"Priority: {priority}")
        y -= 14

        # ---- Improvement ----
        if improvement:
            c.setFont("Helvetica", 11)
            c.setFillColor(colors.black)
            c.drawString(50, y, f"Expected Improvement: {improvement}")
            y -= 14

        # ---- Target Issue ----
        if target_issue:
            c.drawString(50, y, f"Target Issue: {target_issue}")
            y -= 14

        # ---- Price ----
        if price:
            c.drawString(50, y, f"Estimated Price Range: {price}")
            y -= 14

        # ---- Description ----
        if description:
            c.drawString(50, y, f"Description: {description}")
            y -= 14

        # ---- Why Recommended ----
        if why:
            c.drawString(50, y, f"Why this is recommended: {why}")
            y -= 14

        # ---- Source ----
        if source:
            c.setFont("Helvetica-Oblique", 10)
            c.setFillColor(colors.grey)
            c.drawString(50, y, f"Source: {source}")
            y -= 18

        # Separator line
        c.setStrokeColor(colors.lightgrey)
        c.line(50, y, 550, y)
        y -= 18

    # =========================
    # FOOTER DISCLAIMER
    # =========================
    if y < 120:
        c.showPage()
        y = 760

    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.grey)
    c.drawString(
        50,
        y,
        "This equipment list is generated based on ISO 9241-5 ergonomic analysis."
    )
    y -= 12
    c.drawString(
        50,
        y,
        "These recommendations are provided for general ergonomic guidance and wellness support."
    )

    # Save PDF
    c.save()

    return output_pdf