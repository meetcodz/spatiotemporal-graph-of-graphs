import os
import json

folder = r"d:\first intern(maybe)\stgogcompleteworkfromscratch"
output_file = r"d:\first intern(maybe)\notebooks_analysis.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        out.write(f"\n{'='*60}\n")
        out.write(f"FILE: {fname} (Size: {os.path.getsize(fpath)} bytes)\n")
        out.write(f"{'='*60}\n")
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                nb = json.load(fp)
            cells = nb.get("cells", [])
            out.write(f"Total cells: {len(cells)}\n")
            
            # Print outlines of code cells
            for idx, c in enumerate(cells):
                ctype = c.get("cell_type", "")
                src = "".join(c.get("source", []))
                lines = [l.strip() for l in src.splitlines() if l.strip()]
                if not lines:
                    continue
                first_few = [l for l in lines if l.startswith(("class ", "def ", "import ", "from ", "#", "model =", "class_name"))]
                out.write(f"Cell {idx:02d} [{ctype}]: {lines[0][:90]}\n")
                for key_line in first_few[:5]:
                    if key_line != lines[0]:
                        out.write(f"     -> {key_line[:90]}\n")
                
                # Check for outputs
                outputs = c.get("outputs", [])
                for o in outputs:
                    if "text" in o:
                        otext = "".join(o["text"]).strip()
                        if otext:
                            last_l = otext.splitlines()[-1]
                            out.write(f"     [OUTPUT]: {last_l[:100]}\n")
                    elif "data" in o and "text/plain" in o["data"]:
                        otext = "".join(o["data"]["text/plain"]).strip()
                        if otext:
                            out.write(f"     [OUTPUT]: {otext[:100]}\n")
        except Exception as e:
            out.write(f"Error processing {fname}: {e}\n")

print("Analysis script finished writing to notebooks_analysis.txt")
