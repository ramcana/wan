"""
Test suite for duplicate detection functionality
"""

import os
import tempfile
import shutil
from pathlib import Path
import json

from duplicate_detector import DuplicateDetector


def test_duplicate_detector():
    """Test basic duplicate detection functionality"""
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        # Exact duplicates
        file1 = temp_path / "file1.py"
        file2 = temp_path / "file2.py"
        file1.write_text("print('Hello World')\n")
        file2.write_text("print('Hello World')\n")
        
        # Similar files
        file3 = temp_path / "file3.py"
        file4 = temp_path / "file4.py"
        file3.write_text("print('Hello World')\n# Comment 1\n")
        file4.write_text("print('Hello World')\n# Comment 2\n")
        
        # Unique file
        file5 = temp_path / "file5.py"
        file5.write_text("print('Different content')\n")
        
        # Initialize detector
        detector = DuplicateDetector(str(temp_path))
        
        # Run scan
        report = detector.scan_for_duplicates()
        
        # Verify results
        print(f"Files scanned: {report.total_files_scanned}")
        print(f"Duplicate files found: {len(report.duplicate_files)}")
        print(f"Duplicate groups: {len(report.duplicate_groups)}")
        
        # Should find exact duplicates
        assert len(report.duplicate_files) >= 2, "Should find at least 2 duplicate files"
        assert len(report.duplicate_groups) >= 1, "Should find at least 1 duplicate group"
        
        # Test safe removal
        if report.duplicate_groups:
            results = detector.safe_remove_duplicates(report.duplicate_groups, auto_remove_exact=True)
            print("Removal results:", results)
            
            # Verify backup was created
            assert 'backup' in results, "Should create backup"
            
            # Verify some files were removed
            if 'removal' in results:
                print("Files successfully removed")
        
        print("✓ Duplicate detector test passed")


def test_code_similarity():
    """Test code similarity detection"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create similar Python files
        file1 = temp_path / "similar1.py"
        file2 = temp_path / "similar2.py"
        
        code1 = """
def hello_world():
    print("Hello World")
    return True

if __name__ == "__main__":
    hello_world()
"""
        
        code2 = """
def hello_world():
    print("Hello World")  # Different comment
    return True

if __name__ == "__main__":
    hello_world()
"""
        
        file1.write_text(code1)
        file2.write_text(code2)
        
        detector = DuplicateDetector(str(temp_path))
        similarity = detector._calculate_similarity(file1, file2)
        
        print(f"Code similarity: {similarity:.2f}")
        assert similarity > 0.8, f"Similar code should have high similarity score, got {similarity}"
        
        print("✓ Code similarity test passed")


def test_backup_and_rollback():
    """Test backup and rollback functionality"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        file1 = temp_path / "test1.txt"
        file2 = temp_path / "test2.txt"
        content = "Test content for backup"
        
        file1.write_text(content)
        file2.write_text(content)
        
        detector = DuplicateDetector(str(temp_path))
        
        # Create backup
        files_to_backup = [str(file1), str(file2)]
        backup_path = detector.create_backup(files_to_backup)
        
        print(f"Backup created at: {backup_path}")
        assert Path(backup_path).exists(), "Backup directory should exist"
        
        # Verify manifest exists
        manifest_path = Path(backup_path) / "manifest.json"
        assert manifest_path.exists(), "Backup manifest should exist"
        
        # Remove original files
        file1.unlink()
        file2.unlink()
        
        assert not file1.exists(), "Original file should be removed"
        assert not file2.exists(), "Original file should be removed"
        
        # Test rollback
        success = detector.rollback_removal(backup_path)
        assert success, "Rollback should succeed"
        
        # Verify files are restored
        assert file1.exists(), "File should be restored"
        assert file2.exists(), "File should be restored"
        assert file1.read_text() == content, "File content should be restored"
        
        print("✓ Backup and rollback test passed")


if __name__ == "__main__":
    print("Running duplicate detector tests...")
    
    try:
        test_duplicate_detector()
        test_code_similarity()
        test_backup_and_rollback()
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise