---
title: migration.data_migrator
category: api
tags: [api, migration]
---

# migration.data_migrator

Data migration utilities for migrating existing Gradio outputs to new SQLite system.

## Classes

### DataMigrator

Handles migration of existing Gradio outputs to new SQLite system.

#### Methods

##### __init__(self: Any, gradio_outputs_dir: str, new_outputs_dir: str, backup_dir: str)



##### create_backup(self: Any) -> bool

Create backup of existing outputs before migration.

##### scan_gradio_outputs(self: Any) -> <ast.Subscript object at 0x000001942F8A4970>

Scan existing Gradio outputs directory for video files and metadata.

##### _extract_metadata(self: Any, video_path: Path) -> <ast.Subscript object at 0x000001942F8A5750>

Extract metadata from video file and associated files.

##### _infer_model_type(self: Any, video_path: Path) -> str

Infer model type from file path or name.

##### _extract_prompt(self: Any, video_path: Path) -> str

Try to extract prompt from associated metadata files or filename.

##### _get_video_resolution(self: Any, video_path: Path) -> str

Get video resolution using OpenCV.

##### _get_video_duration(self: Any, video_path: Path) -> float

Get video duration in seconds using OpenCV.

##### migrate_to_sqlite(self: Any, outputs: <ast.Subscript object at 0x0000019430100D90>) -> <ast.Subscript object at 0x0000019433D54A00>

Migrate video metadata to SQLite database.

##### _generate_thumbnail(self: Any, video_path: Path) -> <ast.Subscript object at 0x0000019433D57760>

Generate thumbnail for migrated video.

##### run_migration(self: Any) -> Dict

Run complete migration process.

