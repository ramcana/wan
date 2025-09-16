---
title: tools.onboarding.developer_checklist
category: api
tags: [api, tools]
---

# tools.onboarding.developer_checklist

Developer Checklist and Progress Tracking

This module provides a comprehensive checklist for new developers
and tracks their onboarding progress.

## Classes

### ChecklistItem

Individual checklist item

### ChecklistProgress

Overall checklist progress

### DeveloperChecklist

Manages developer onboarding checklist and progress tracking

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x00000194341CE140>, developer_name: <ast.Subscript object at 0x00000194341CE080>)



##### _create_checklist_items(self: Any) -> <ast.Subscript object at 0x0000019434334550>

Create the comprehensive developer checklist

##### _load_progress(self: Any) -> ChecklistProgress

Load existing progress or create new progress tracking

##### _save_progress(self: Any)

Save current progress to file

##### _update_progress_metrics(self: Any, progress: ChecklistProgress)

Update progress metrics

##### _parse_time_estimate(self: Any, time_str: str) -> int

Parse time estimate string to minutes

##### complete_item(self: Any, item_id: str, notes: <ast.Subscript object at 0x0000019434541FC0>) -> bool

Mark an item as completed

##### uncomplete_item(self: Any, item_id: str) -> bool

Mark an item as not completed

##### validate_item(self: Any, item_id: str) -> bool

Validate an item using its validation command

##### get_next_items(self: Any) -> <ast.Subscript object at 0x0000019431A06740>

Get next items that can be completed (prerequisites met)

##### get_status_summary(self: Any) -> <ast.Subscript object at 0x0000019431A053C0>

Get comprehensive status summary

##### generate_report(self: Any) -> str

Generate a comprehensive progress report

