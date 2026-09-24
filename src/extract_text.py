from pathlib import Path
import json, pymupdf    # pymupdf for reading pdfs...

input_dir = Path("data/pdf_files")
output_dir = Path("data/extracted_data")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

for file in input_dir.glob("*.pdf"):
    data = []

    pdf = pymupdf.open(file)   # open every file...

    # read each page in each file....
    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text")

        data.append({
            "document": file.stem,
            "page": page_number,
            "text": text
        })
    pdf.close()


    # specify the output path and file name...
    output_path = output_dir / f"{file.stem}.json"

    # encoding='utf-8' → supports different characters and languages
    # ensure_ascii=False → keeps non-ASCII characters readable
    # indent=4 → formats the JSON with 4-space indentation
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"Extracted: {file.stem} | "
        f"Saved to: {output_dir}"
        )