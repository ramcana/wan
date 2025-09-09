"""initial tables

Revision ID: 0001
Revises: 
Create Date: 2025-02-14
"""

from alembic import op
import sqlalchemy as sa
from backend.repositories.database import TaskStatusEnum, ModelTypeEnum

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'generation_tasks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('model_type', sa.Enum(ModelTypeEnum), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('image_path', sa.String(), nullable=True),
        sa.Column('end_image_path', sa.String(), nullable=True),
        sa.Column('resolution', sa.String(), nullable=False, default='1280x720'),
        sa.Column('steps', sa.Integer(), nullable=False, default=50),
        sa.Column('lora_path', sa.String(), nullable=True),
        sa.Column('lora_strength', sa.Float(), nullable=False, default=1.0),
        sa.Column('status', sa.Enum(TaskStatusEnum), nullable=False, default=TaskStatusEnum.PENDING),
        sa.Column('progress', sa.Integer(), nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('output_path', sa.String(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('estimated_time_minutes', sa.Integer(), nullable=True),
        sa.Column('generation_time_minutes', sa.Float(), nullable=True),
    )

    op.create_table(
        'system_stats',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('cpu_percent', sa.Float(), nullable=False),
        sa.Column('ram_used_gb', sa.Float(), nullable=False),
        sa.Column('ram_total_gb', sa.Float(), nullable=False),
        sa.Column('ram_percent', sa.Float(), nullable=False),
        sa.Column('gpu_percent', sa.Float(), nullable=False),
        sa.Column('vram_used_mb', sa.Float(), nullable=False),
        sa.Column('vram_total_mb', sa.Float(), nullable=False),
        sa.Column('vram_percent', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
    )

def downgrade() -> None:
    op.drop_table('system_stats')
    op.drop_table('generation_tasks')
