"""
SQLAlchemy models for TinyNet
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text, Index, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import Base
import uuid


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    nodes = relationship("Node", back_populates="user")
    todos = relationship("Todo", back_populates="user")


class Node(Base):
    """Node model for mind mapping"""
    __tablename__ = "nodes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    is_hub = Column(Boolean, default=False)
    status = Column(String, nullable=False)  # Using the states from config
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
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
    source_id = Column(String(36), ForeignKey("nodes.id"), nullable=False)
    target_id = Column(String(36), ForeignKey("nodes.id"), nullable=False)
    strength = Column(Float, default=1.0)
    
    # Relationships
    source_node = relationship("Node", foreign_keys=[source_id], back_populates="source_links")
    target_node = relationship("Node", foreign_keys=[target_id], back_populates="target_links")


class ProgressLog(Base):
    """Progress log for tracking node progress"""
    __tablename__ = "progress_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(36), ForeignKey("nodes.id"), nullable=False)
    created_at = Column(DateTime, default=func.now())
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
    due_at = Column(DateTime, nullable=True)
    node_id = Column(String(36), ForeignKey("nodes.id"), nullable=True)
    created_at = Column(DateTime, default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    node = relationship("Node", back_populates="todos")
    user = relationship("User", back_populates="todos")


class GraphEdge(Base):
    """Learned graph relationship between two engines (node-level proxy for now)."""
    __tablename__ = "graph_edges"

    id = Column(Integer, primary_key=True, index=True)
    source_engine_id = Column(String(36), nullable=False, index=True)
    target_engine_id = Column(String(36), nullable=False, index=True)
    weight = Column(Float, nullable=False, default=0.0)
    sample_count = Column(Integer, nullable=False, default=0)
    signal_breakdown = Column(JSON, nullable=False, default=dict)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


# Create indexes
Index("idx_nodes_title", Node.title)
Index("idx_progress_logs_node_created", ProgressLog.node_id, ProgressLog.created_at.desc())
Index("idx_todos_status_due", Todo.status, Todo.due_at)
Index("idx_graph_edges_pair", GraphEdge.source_engine_id, GraphEdge.target_engine_id, unique=True)
