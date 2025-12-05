from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from PIL import Image
import os
import json

# Helper function to add a section to the PDF
def add_section(c, title, content, y_position):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, title)
    c.setFont("Helvetica", 10)
    y_position -= 20  # Added more space between title and content
    for line in content.split("\n"):
        c.drawString(50, y_position, line)
        y_position -= 12
    return y_position

# Function to generate the PDF report
def generate_pdf_report(final_iso_report, ai_report, image_path="output/annotated_1.jpg"):
    # Create the PDF canvas
    output_pdf = "ergonomics_report.pdf"
    c = canvas.Canvas(output_pdf, pagesize=letter)

    # Title (Centered)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(300, 750, "POSTURA - Ergonomics Evaluation Report")

    # Add Posture Analysis Section
    posture_content = ""
    for key, value in final_iso_report['posture'].items():
        posture_content += f"{key}: {value['severity']} (Angle: {value['angle']})\n"
    y_position = 700
    y_position = add_section(c, "Posture Analysis", posture_content, y_position)

    # Add Workstation Analysis Section
    workstation_content = ""
    for comp, rules in final_iso_report['workstation'].items():
        workstation_content += f"{comp}:\n"
        for rule_id, rule_data in rules.items():
            workstation_content += f"{rule_id}: {rule_data['severity']} (Delta: {rule_data['delta']})\n"
    y_position = add_section(c, "Workstation Analysis", workstation_content, y_position)

    # Add Risk Summary Section (from AI)
    risk_summary = ai_report.get("risk_summary", "No risk summary available.")
    y_position = add_section(c, "Risk Summary", risk_summary, y_position)

    # Add Exercise Recommendations Section (from AI)
    exercise_recommendations = "\n".join(ai_report.get("exercise_recommendations", ["No exercises recommended."]))
    y_position = add_section(c, "Exercise Recommendations", exercise_recommendations, y_position)

    # Add Final Advice Section (from AI)
    final_advice = ai_report.get("final_advice", "No final advice available.")
    y_position = add_section(c, "Final Advice", final_advice, y_position)

    # Add annotated image to PDF
    if os.path.exists(image_path):
        # Dynamically adjust the image placement after the text
        image_height = 300  # Image height
        available_space = y_position - 50  # Space left before the bottom
        image_y_position = available_space - image_height  # Place image just above the bottom

        c.drawImage(image_path, 50, image_y_position, width=500, height=image_height)
    else:
        print("No annotated image found.")

    # Save PDF
    c.save()

    print(f"✅ Ergonomics PDF report saved as: {output_pdf}")
    return output_pdf
