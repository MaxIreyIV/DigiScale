import cv2
import pytesseract
import numpy as np
import os
import re

OUTPUT_DIR = "ocr_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regular‑expression patterns to capture the four fields
NAME_RE = re.compile(r"Name\s*:?\s*(.+)", re.I)
#
# Sex may be "M" or "F" optionally surrounded by a circle/bullet,
# or as a single Unicode circled letter Ⓜ / Ⓕ.
SEX_RE = re.compile(
    r"(?:[○●\(\[\{]?\s*(?P<sex>[MF])\s*[)○●\]\}]?|(?P<circled>Ⓜ|Ⓕ))",
    re.I,
)
 # Accept DD.MM.YY, DD/MM/YY, DD‑MM‑YYYY, etc.
DOB_RE = re.compile(
    r"(?:Date\s*of\s*Birth|DOB)\s*:?\s*([0-9]{2}[./-][0-9]{2}[./-][0-9]{2,4})",
    re.I,
)
VILLAGE_RE = re.compile(r"Village\s*:?\s*(.+)", re.I)

def parse_info(text: str):
    info = {"name": None, "sex": None, "dob": None, "village": None}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if info["name"] is None:
            m = NAME_RE.search(line)
            if m:
                info["name"] = m.group(1).strip()
                continue
        if info["sex"] is None:
            m = SEX_RE.search(line)
            if m:
                if m.group("sex"):
                    info["sex"] = m.group("sex").upper()
                else:
                    info["sex"] = "M" if "Ⓜ" in m.group(0) else "F"
                continue
        if info["dob"] is None:
            m = DOB_RE.search(line)
            if m:
                info["dob"] = m.group(1).strip()
                continue
        if info["village"] is None:
            m = VILLAGE_RE.search(line)
            if m:
                info["village"] = m.group(1).strip()
    return info

def save_info(info: dict):
    """
    Save each non‑None field into its own text file inside OUTPUT_DIR.
    """
    for key, value in info.items():
        if value:
            with open(os.path.join(OUTPUT_DIR, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(value + "\n")

def process_frame(img_bgr, draw_boxes: bool = False):
    """
    Run Tesseract OCR on a BGR image held in memory and extract
    Name / Sex / DOB / Village.

    Args
    ----
    img_bgr : np.ndarray
        OpenCV BGR image.
    draw_boxes : bool, optional
        If True, draw OCR bounding boxes and return the annotated frame.

    Returns
    -------
    tuple
        (info_dict, raw_text)  when draw_boxes is False
        (info_dict, raw_text, annotated_img)  when draw_boxes is True
    """
    # OCR
    text = pytesseract.image_to_string(img_bgr)
    info = parse_info(text)
    save_info(info)

    if draw_boxes:
        d = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT)
        for x, y, w, h, conf in zip(d["left"], d["top"], d["width"],
                                    d["height"], d["conf"]):
            if int(conf) > 0:
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return info, text, img_bgr

    return info, text