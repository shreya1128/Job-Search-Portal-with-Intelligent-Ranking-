"""Package the complete JobIntel project into a clean production ZIP archive."""
import os
import zipfile
from pathlib import Path

ROOT = Path("C:/Users/shrey/OneDrive/Documents/Desktop/CIE/jobintel").resolve()
OUTPUT_PATHS = [
    Path("C:/Users/shrey/.gemini/antigravity/brain/7996b32d-374d-43be-a757-fbacb5a89b3a/jobintel_complete_phase8_production.zip"),
    Path("C:/Users/shrey/OneDrive/Documents/Desktop/CIE/jobintel_complete_phase8_production.zip"),
]

EXCLUDE_DIRS = {
    ".venv", "venv", "env",
    "__pycache__", ".pytest_cache",
    ".git", ".idea", ".vscode",
    "_backups", "_session_archive",
}

EXCLUDE_EXTS = {
    ".pyc", ".pyo", ".pyd"
}

EXCLUDE_FILES = {
    ".env", "desktop.ini"
}

def should_exclude(rel_path: Path) -> bool:
    parts = rel_path.parts
    for p in parts:
        if p in EXCLUDE_DIRS:
            return True
    if rel_path.name in EXCLUDE_FILES:
        return True
    if rel_path.suffix in EXCLUDE_EXTS:
        return True
    return False

def make_zip(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total_size = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(ROOT)
                if should_exclude(rel):
                    continue
                zf.write(full, arcname=f"jobintel/{rel.as_posix()}")
                count += 1
                total_size += full.stat().st_size
    print(f"Created {out_path}: {count} files, {total_size / (1024*1024):.1f} MB uncompressed, {out_path.stat().st_size / (1024*1024):.1f} MB compressed")

if __name__ == "__main__":
    for out in OUTPUT_PATHS:
        make_zip(out)

    # Verification: check that the zip opens and contains key files
    test_zip = OUTPUT_PATHS[0]
    with zipfile.ZipFile(test_zip, "r") as zf:
        names = zf.namelist()
        print(f"\nVerification of {test_zip.name}:")
        print(f"Total entries: {len(names)}")
        key_files = [
            "jobintel/FINAL_HANDOFF.md",
            "jobintel/requirements.txt",
            "jobintel/LICENSE",
            "jobintel/wsgi.py",
            "jobintel/jobintel/web/app.py",
            "jobintel/jobintel/rec/upload_guard.py",
            "jobintel/jobintel/rec/documents.py",
            "jobintel/data/index/job_index.pkl",
            "jobintel/data/processed/skill_lexicon.json",
            "jobintel/jobintel.db",
            "jobintel/docs/PHASE8_REPORT.md",
            "jobintel/docs/SYLLABUS_MAPPING.md",
            "jobintel/docs/TESTING_AND_VERIFICATION.md",
            "jobintel/docs/DEPLOYMENT_AND_SECURITY.md",
            "jobintel/docs/LICENSES_AND_DEPENDENCIES.md",
        ]
        all_present = True
        for k in key_files:
            present = k in names
            print(f"  [{'OK' if present else 'MISSING'}] {k}")
            if not present:
                all_present = False
        assert all_present, "Some key files were missing from the ZIP!"
        print("\nAll key files verified successfully!")
