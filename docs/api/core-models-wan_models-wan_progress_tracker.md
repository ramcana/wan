---
title: core.models.wan_models.wan_progress_tracker
category: api
tags: [api, core]
---

# core.models.wan_models.wan_progress_tracker



## Classes

### ProgressUpdate



### WANProgressTracker

Minimal, in-memory progress tracker.
Replace with a NVML/WS-backed implementation later if needed.

#### Methods

##### __init__(self: Any)



##### on_update(self: Any, cb: <ast.Subscript object at 0x000001942FCDB250>) -> None



##### _emit(self: Any, pu: ProgressUpdate) -> None



##### start(self: Any, task_id: str, message: <ast.Subscript object at 0x00000194319C4AF0>) -> ProgressUpdate



##### update(self: Any, task_id: str, pct: <ast.Subscript object at 0x00000194319C41F0>, message: <ast.Subscript object at 0x00000194319C4130>, eta_seconds: <ast.Subscript object at 0x00000194319C4070>) -> ProgressUpdate



##### complete(self: Any, task_id: str, message: <ast.Subscript object at 0x0000019431972F80>) -> ProgressUpdate



##### fail(self: Any, task_id: str, error: str) -> ProgressUpdate



##### get(self: Any, task_id: str) -> <ast.Subscript object at 0x0000019431972200>



