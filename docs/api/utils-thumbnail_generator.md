---
title: utils.thumbnail_generator
category: api
tags: [api, utils]
---

# utils.thumbnail_generator

Thumbnail generation utilities for video files

## Classes

### ThumbnailGenerator

Generate thumbnails for video files using ffmpeg

#### Methods

##### __init__(self: Any, thumbnails_dir: str)



##### generate_thumbnail(self: Any, video_path: str, thumbnail_name: <ast.Subscript object at 0x000001942A1BF2B0>, timestamp: str, width: int, height: int) -> <ast.Subscript object at 0x0000019428D7C1C0>

Generate a thumbnail for a video file

Args:
    video_path: Path to the video file
    thumbnail_name: Name for the thumbnail file (without extension)
    timestamp: Timestamp to capture (format: HH:MM:SS)
    width: Thumbnail width in pixels
    height: Thumbnail height in pixels
    
Returns:
    Path to the generated thumbnail or None if failed

##### generate_thumbnail_async(self: Any, video_path: str, thumbnail_name: <ast.Subscript object at 0x0000019428D7FDF0>) -> <ast.Subscript object at 0x0000019428D7C5B0>

Generate thumbnail asynchronously (placeholder for future async implementation)
Currently just calls the sync version

##### delete_thumbnail(self: Any, thumbnail_path: str) -> bool

Delete a thumbnail file

Args:
    thumbnail_path: Path to the thumbnail file
    
Returns:
    True if deleted successfully, False otherwise

##### _check_ffmpeg(self: Any) -> bool

Check if ffmpeg is available in the system

