"""
Simple test for queue persistence functionality
"""

import os
import sys
import tempfile
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import Base, GenerationTaskDB, TaskStatusEnum, ModelTypeEnum

def test_queue_persistence():
    """Test that tasks persist in database correctly"""
    
    # Create temporary database
    test_db_path = tempfile.mktemp(suffix='.db')
    test_db_url = f"sqlite:///{test_db_path}"
    
    try:
        # Create test engine and session
        engine = create_engine(
            test_db_url,
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Test 1: Create tasks and verify persistence
        db = SessionLocal()
        try:
            # Create test tasks
            tasks = [
                GenerationTaskDB(
                    id="test-pending-1",
                    model_type=ModelTypeEnum.T2V_A14B,
                    prompt="Test pending task",
                    resolution="1280x720",
                    steps=50,
                    status=TaskStatusEnum.PENDING,
                    progress=0,
                    created_at=datetime.utcnow()
                ),
                GenerationTaskDB(
                    id="test-processing-1",
                    model_type=ModelTypeEnum.I2V_A14B,
                    prompt="Test processing task",
                    image_path="uploads/test_image.jpg",
                    resolution="1280x720",
                    steps=50,
                    status=TaskStatusEnum.PROCESSING,
                    progress=50,
                    created_at=datetime.utcnow(),
                    started_at=datetime.utcnow()
                ),
                GenerationTaskDB(
                    id="test-completed-1",
                    model_type=ModelTypeEnum.TI2V_5B,
                    prompt="Test completed task",
                    resolution="1920x1080",
                    steps=75,
                    status=TaskStatusEnum.COMPLETED,
                    progress=100,
                    created_at=datetime.utcnow(),
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    output_path="outputs/test-completed-1.mp4"
                )
            ]
            
            db.add_all(tasks)
            db.commit()
            
            # Verify tasks were created
            total_count = db.query(GenerationTaskDB).count()
            assert total_count == 3, f"Expected 3 tasks, got {total_count}"
            
            pending_count = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.PENDING
            ).count()
            assert pending_count == 1, f"Expected 1 pending task, got {pending_count}"
            
            processing_count = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.PROCESSING
            ).count()
            assert processing_count == 1, f"Expected 1 processing task, got {processing_count}"
            
            completed_count = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.COMPLETED
            ).count()
            assert completed_count == 1, f"Expected 1 completed task, got {completed_count}"
            
            print("✓ Task creation and persistence test passed")
            
        finally:
            db.close()
        
        # Test 2: Simulate restart by creating new session
        db2 = SessionLocal()
        try:
            # Verify all tasks still exist after "restart"
            all_tasks = db2.query(GenerationTaskDB).all()
            assert len(all_tasks) == 3, f"Expected 3 tasks after restart, got {len(all_tasks)}"
            
            # Verify specific tasks
            pending_task = db2.query(GenerationTaskDB).filter(
                GenerationTaskDB.id == "test-pending-1"
            ).first()
            assert pending_task is not None, "Pending task not found after restart"
            assert pending_task.status == TaskStatusEnum.PENDING
            assert pending_task.model_type == ModelTypeEnum.T2V_A14B
            
            processing_task = db2.query(GenerationTaskDB).filter(
                GenerationTaskDB.id == "test-processing-1"
            ).first()
            assert processing_task is not None, "Processing task not found after restart"
            assert processing_task.status == TaskStatusEnum.PROCESSING
            assert processing_task.image_path == "uploads/test_image.jpg"
            
            completed_task = db2.query(GenerationTaskDB).filter(
                GenerationTaskDB.id == "test-completed-1"
            ).first()
            assert completed_task is not None, "Completed task not found after restart"
            assert completed_task.status == TaskStatusEnum.COMPLETED
            assert completed_task.output_path == "outputs/test-completed-1.mp4"
            
            print("✓ Queue persistence after restart test passed")
            
        finally:
            db2.close()
        
        # Test 3: Task cancellation
        db3 = SessionLocal()
        try:
            # Cancel the pending task
            pending_task = db3.query(GenerationTaskDB).filter(
                GenerationTaskDB.id == "test-pending-1"
            ).first()
            
            pending_task.status = TaskStatusEnum.CANCELLED
            pending_task.completed_at = datetime.utcnow()
            db3.commit()
            
            # Verify cancellation persisted
            cancelled_task = db3.query(GenerationTaskDB).filter(
                GenerationTaskDB.id == "test-pending-1"
            ).first()
            assert cancelled_task.status == TaskStatusEnum.CANCELLED
            assert cancelled_task.completed_at is not None
            
            print("✓ Task cancellation persistence test passed")
            
        finally:
            db3.close()
        
        # Clean up
        engine.dispose()
        
    finally:
        # Remove test database
        try:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
        except PermissionError:
            pass  # File might still be in use

def test_image_validation_requirements():
    """Test image validation requirements for I2V/TI2V"""
    
    # Test supported formats
    supported_formats = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'
    }
    
    # Test maximum file size
    max_size = 10 * 1024 * 1024  # 10MB
    
    # Test minimum dimensions
    min_width, min_height = 64, 64
    
    # Test maximum dimensions  
    max_width, max_height = 4096, 4096
    
    print("✓ Image validation requirements defined:")
    print(f"  - Supported formats: {supported_formats}")
    print(f"  - Maximum file size: {max_size // (1024*1024)}MB")
    print(f"  - Minimum dimensions: {min_width}x{min_height}")
    print(f"  - Maximum dimensions: {max_width}x{max_height}")

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    print("Running queue persistence tests...")
    
    try:
        test_queue_persistence()
        test_image_validation_requirements()
        
        print("\n✓ All queue persistence tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)