from pathlib import Path

def load_txt_folder(folder: str) -> list[tuple[str, str]]:
    """Return list of (path_str, text) for all .txt files in folder (non-recursive)."""
    p = Path(folder)
    files = sorted([f for f in p.iterdir() if f.suffix.lower() == ".txt"])
    docs = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            # Normalize newlines & collapse whitespace
            text = " ".join(text.split())
            docs.append((str(f), text))
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    return docs
