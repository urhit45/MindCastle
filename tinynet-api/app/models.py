"""
SQLAlchemy models for TinyNet
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import Base
import uuid


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    nodes = relationship("Node", back_populates="user")
    todos = relationship("Todo", back_populates="user")


class Node(Base):
    """Node model for mind mapping"""
    __tablename__ = "nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    is_hub = Column(Boolean, default=False)
    status = Column(String, nullable=False)  # Using the states from config
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="nodes")
    source_links = relationship("Link", foreign_keys="Link.source_id", back_populates="source_node")
    target_links = relationship("Link", foreign_keys="Link.target_id", back_populates="target_node")
    progress_logs = relationship("ProgressLog", back_populates="node")
    todos = relationship("Todo", back_populates="node")


class Link(Base):
    """Link model for connecting nodes"""
    __tablename__ = "links"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    target_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    strength = Column(Float, default=1.0)
    
    # Relationships
    source_node = relationship("Node", foreign_keys=[source_id], back_populates="source_links")
    target_node = relationship("Node", foreign_keys=[target_id], back_populates="target_links")


class ProgressLog(Base):
    """Progress log for tracking node progress"""
    __tablename__ = "progress_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    text = Column(Text, nullable=False)
    next_step = Column(String, nullable=True)  # Using templates from config
    state = Column(String, nullable=False)  # Using states from config
    
    # Relationships
    node = relationship("Node", back_populates="progress_logs")


class Todo(Base):
    """Todo model for task management"""
    __tablename__ = "todos"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    status = Column(String, nullable=False)
    due_at = Column(DateTime(timezone=True), nullable=True)
    node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    node = relationship("Node", back_populates="todos")
    user = relationship("User", back_populates="todos")


# Create indexes
Index("idx_nodes_title", Node.title)
Index("idx_progress_logs_node_created", ProgressLog.node_id, ProgressLog.created_at.desc())
Index("idx_todos_status_due", Todo.status, Todo.due_at)
