
import json, re            # Regular Expression (re) for cleaning..
from pathlib import Path

input_dir = Path("data/extracted_data")
output_dir = Path("data/cleaned_data")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# read all files....

for file in input_dir.glob("*.json"):

    clean_data = []   
    
    with open(file, 'r', encoding='utf-8') as f:
        context = json.load(f)

    # read each page in each file....

    for page in context:

        text = page["text"].strip()
        text = re.sub(r"[🔹•]", "", text)

        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        text = "\n".join(lines)

        clean_data.append({
            "document": page["document"],
            "page": page["page"],
            "text": text
        })

    # specifying the output path....
    
    output_path = output_dir / f"{file.stem}_clean.json"

    # encoding='utf-8' → supports different characters and languages
    # ensure_ascii=False → keeps non-ASCII characters readable
    # indent=4 → formats the JSON with 4-space indentation
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=3)

    print(
        f"Cleaned: {file.stem} | "
        f"Saved to: {output_path}"
        )