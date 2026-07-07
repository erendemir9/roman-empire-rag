from fpdf import FPDF

txt_path = "../data/roman_empire.txt"
pdf_path = "../data/roman_empire.pdf"


def is_heading(line):
    # section headers look like "1. Introduction and Timeline"
    return len(line) > 2 and line[0].isdigit() and ". " in line[:4]


f = open(txt_path, encoding="utf-8")
text = f.read()
f.close()

lines = text.split("\n")

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# first line is the title
title = lines[0]
pdf.set_font("Helvetica", "B", 16)
pdf.multi_cell(0, 10, title)
pdf.ln(4)

for line in lines[1:]:
    line = line.strip()
    if line == "":
        continue
    if is_heading(line):
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, line)
        pdf.ln(2)
    else:
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, line)
        pdf.ln(3)

pdf.output(pdf_path)
print("Done, created ../data/roman_empire.pdf")
