def test_project_files_exist():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "app.py").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "data/knowledge_base").exists()
