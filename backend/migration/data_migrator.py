"""
Data migration utilities for migrating existing Gradio outputs to new SQLite system.
"""

import os
import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from PIL import Image
import cv2

from ..models.schemas import GenerationTask, TaskStatus, ModelType
from ..database import get_db_connection

logger = logging.getLogger(__name__)

class DataMigrator:
    """Handles migration of existing Gradio outputs to new SQLite system."""
    
    def __init__(self, 
                 gradio_outputs_dir: str = "outputs",
                 new_outputs_dir: str = "backend/outputs",
                 backup_dir: str = "migration_backup"):
        self.gradio_outputs_dir = Path(gradio_outputs_dir)
        self.new_outputs_dir = Path(new_outputs_dir)
        self.backup_dir = Path(backup_dir)
        self.migration_log = []
        
    def create_backup(self) -> bool:
        """Create backup of existing outputs before migration."""
        try:
            if self.gradio_outputs_dir.exists():
                logger.info(f"Creating backup at {self.backup_dir}")
                shutil.copytree(self.gradio_outputs_dir, self.backup_dir, dirs_exist_ok=True)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def scan_gradio_outputs(self) -> List[Dict]:
        """Scan existing Gradio outputs directory for video files and metadata."""
        outputs = []
        
        if not self.gradio_outputs_dir.exists():
            logger.warning(f"Gradio outputs directory {self.gradio_outputs_dir} not found")
            return outputs
            
        # Look for video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        for file_path in self.gradio_outputs_dir.rglob('*'):
            if file_path.suffix.lower() in video_extensions:
                metadata = self._extract_metadata(file_path)
                if metadata:
                    outputs.append(metadata)
                    
        logger.info(f"Found {len(outputs)} video files to migrate")
        return outputs
    
    def _extract_metadata(self, video_path: Path) -> Optional[Dict]:
        """Extract metadata from video file and associated files."""
        try:
            # Basic file info
            stat = video_path.stat()
            metadata = {
                'original_path': str(video_path),
                'filename': video_path.name,
                'created_at': datetime.fromtimestamp(stat.st_ctime),
                'file_size': stat.st_size,
                'model_type': self._infer_model_type(video_path),
                'prompt': self._extract_prompt(video_path),
                'resolution': self._get_video_resolution(video_path),
                'duration': self._get_video_duration(video_path)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {video_path}: {e}")
            return None
    
    def _infer_model_type(self, video_path: Path) -> str:
        """Infer model type from file path or name."""
        path_str = str(video_path).lower()
        
        if 't2v' in path_str or 'text2video' in path_str:
            return ModelType.T2V_A14B
        elif 'i2v' in path_str or 'image2video' in path_str:
            return ModelType.I2V_A14B
        elif 'ti2v' in path_str or 'textimage2video' in path_str:
            return ModelType.TI2V_5B
        else:
            # Default to T2V if can't determine
            return ModelType.T2V_A14B
    
    def _extract_prompt(self, video_path: Path) -> str:
        """Try to extract prompt from associated metadata files or filename."""
        # Look for associated metadata files
        metadata_files = [
            video_path.with_suffix('.json'),
            video_path.with_suffix('.txt'),
            video_path.parent / f"{video_path.stem}_metadata.json"
        ]
        
        for metadata_file in metadata_files:
            if metadata_file.exists():
                try:
                    if metadata_file.suffix == '.json':
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'prompt' in data:
                                return data['prompt']
                            elif 'text' in data:
                                return data['text']
                    elif metadata_file.suffix == '.txt':
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            return f.read().strip()
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        # Fallback: try to extract from filename
        filename = video_path.stem
        # Remove common prefixes/suffixes
        for prefix in ['output_', 'generated_', 'video_']:
            if filename.startswith(prefix):
                filename = filename[len(prefix):]
        
        return filename.replace('_', ' ') if filename else "Migrated video"
    
    def _get_video_resolution(self, video_path: Path) -> str:
        """Get video resolution using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return f"{width}x{height}"
        except Exception as e:
            logger.warning(f"Failed to get resolution for {video_path}: {e}")
            return "1280x720"  # Default
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frame_count / fps if fps > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to get duration for {video_path}: {e}")
            return 0.0
    
    def migrate_to_sqlite(self, outputs: List[Dict]) -> Tuple[int, int]:
        """Migrate video metadata to SQLite database."""
        success_count = 0
        error_count = 0
        
        # Ensure new outputs directory exists
        self.new_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            for output in outputs:
                try:
                    # Copy video file to new location
                    old_path = Path(output['original_path'])
                    new_filename = f"migrated_{output['filename']}"
                    new_path = self.new_outputs_dir / new_filename
                    
                    shutil.copy2(old_path, new_path)
                    
                    # Generate thumbnail
                    thumbnail_path = self._generate_thumbnail(new_path)
                    
                    # Create database entry
                    task_data = {
                        'id': f"migrated_{old_path.stem}_{int(output['created_at'].timestamp())}",
                        'model_type': output['model_type'],
                        'prompt': output['prompt'],
                        'resolution': output['resolution'],
                        'status': TaskStatus.COMPLETED,
                        'progress': 100,
                        'created_at': output['created_at'],
                        'completed_at': output['created_at'],
                        'output_path': str(new_path),
                        'thumbnail_path': thumbnail_path,
                        'metadata': json.dumps({
                            'duration': output['duration'],
                            'file_size': output['file_size'],
                            'migrated_from': str(old_path),
                            'migration_date': datetime.now().isoformat()
                        })
                    }
                    
                    # Insert into database
                    cursor.execute("""
                        INSERT INTO generation_tasks 
                        (id, model_type, prompt, resolution, status, progress, 
                         created_at, completed_at, output_path, thumbnail_path, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_data['id'], task_data['model_type'], task_data['prompt'],
                        task_data['resolution'], task_data['status'], task_data['progress'],
                        task_data['created_at'], task_data['completed_at'],
                        task_data['output_path'], task_data['thumbnail_path'],
                        task_data['metadata']
                    ))
                    
                    success_count += 1
                    self.migration_log.append(f"✓ Migrated: {old_path.name}")
                    
                except Exception as e:
                    error_count += 1
                    error_msg = f"✗ Failed to migrate {output['original_path']}: {e}"
                    logger.error(error_msg)
                    self.migration_log.append(error_msg)
            
            conn.commit()
        
        return success_count, error_count
    
    def _generate_thumbnail(self, video_path: Path) -> Optional[str]:
        """Generate thumbnail for migrated video."""
        try:
            thumbnail_dir = self.new_outputs_dir / "thumbnails"
            thumbnail_dir.mkdir(exist_ok=True)
            
            thumbnail_path = thumbnail_dir / f"{video_path.stem}_thumb.jpg"
            
            # Extract first frame as thumbnail
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Resize to thumbnail size
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                
                if aspect_ratio > 1:  # Landscape
                    new_width = 320
                    new_height = int(320 / aspect_ratio)
                else:  # Portrait or square
                    new_height = 240
                    new_width = int(240 * aspect_ratio)
                
                resized = cv2.resize(frame, (new_width, new_height))
                cv2.imwrite(str(thumbnail_path), resized)
                
                return str(thumbnail_path)
                
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for {video_path}: {e}")
        
        return None
    
    def run_migration(self) -> Dict:
        """Run complete migration process."""
        logger.info("Starting data migration process")
        
        # Create backup
        backup_created = self.create_backup()
        
        # Scan existing outputs
        outputs = self.scan_gradio_outputs()
        
        if not outputs:
            return {
                'success': True,
                'message': 'No outputs found to migrate',
                'backup_created': backup_created,
                'migrated_count': 0,
                'error_count': 0,
                'log': self.migration_log
            }
        
        # Migrate to SQLite
        success_count, error_count = self.migrate_to_sqlite(outputs)
        
        # Generate migration report
        report = {
            'success': error_count == 0,
            'message': f'Migration completed: {success_count} successful, {error_count} errors',
            'backup_created': backup_created,
            'migrated_count': success_count,
            'error_count': error_count,
            'log': self.migration_log
        }
        
        # Save migration report
        report_path = Path("migration_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Migration completed. Report saved to {report_path}")
        return report

def run_migration_cli():
    """CLI entry point for migration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate Gradio outputs to new SQLite system')
    parser.add_argument('--gradio-dir', default='outputs', help='Gradio outputs directory')
    parser.add_argument('--new-dir', default='backend/outputs', help='New outputs directory')
    parser.add_argument('--backup-dir', default='migration_backup', help='Backup directory')
    parser.add_argument('--dry-run', action='store_true', help='Scan only, do not migrate')
    
    args = parser.parse_args()
    
    migrator = DataMigrator(args.gradio_dir, args.new_dir, args.backup_dir)
    
    if args.dry_run:
        outputs = migrator.scan_gradio_outputs()
        print(f"Found {len(outputs)} files to migrate:")
        for output in outputs:
            print(f"  - {output['filename']} ({output['model_type']})")
    else:
        report = migrator.run_migration()
        print(f"Migration {report['message']}")
        if report['log']:
            print("\nDetailed log:")
            for entry in report['log']:
                print(f"  {entry}")

if __name__ == "__main__":
    run_migration_cli()