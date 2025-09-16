---
title: scripts.migrate_to_real_generation
category: api
tags: [api, scripts]
---

# scripts.migrate_to_real_generation

Migration script to transition from mock to real AI generation mode.
This script handles configuration updates, database migrations, and validation.

## Classes

### RealGenerationMigrator

Handles migration from mock to real AI generation mode.

#### Methods

##### __init__(self: Any)



##### backup_current_config(self: Any) -> bool

Create backup of current configuration.

##### update_configuration_for_real_generation(self: Any) -> bool

Update configuration to enable real generation mode.

##### rollback_configuration(self: Any) -> bool

Rollback configuration to backup.

