"""
Pydantic schemas for TinyNet API
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


# Base schemas
class UserBase(BaseModel):
    email: str = Field(..., description="User email address")


class NodeBase(BaseModel):
    title: str = Field(..., description="Node title")
    is_hub: bool = Field(default=False, description="Whether this is a hub node")
    status: str = Field(..., description="Node status")


class LinkBase(BaseModel):
    source_id: UUID = Field(..., description="Source node ID")
    target_id: UUID = Field(..., description="Target node ID")
    strength: float = Field(default=1.0, ge=0.0, le=10.0, description="Link strength")


class ProgressLogBase(BaseModel):
    text: str = Field(..., description="Progress log text")
    next_step: Optional[str] = Field(None, description="Next step template")
    state: str = Field(..., description="Progress state")


class TodoBase(BaseModel):
    title: str = Field(..., description="Todo title")
    status: str = Field(..., description="Todo status")
    due_at: Optional[datetime] = Field(None, description="Due date")
    node_id: Optional[UUID] = Field(None, description="Associated node ID")


# Create schemas
class UserCreate(UserBase):
    pass


class NodeCreate(NodeBase):
    pass


class LinkCreate(LinkBase):
    pass


class ProgressLogCreate(ProgressLogBase):
    node_id: UUID = Field(..., description="Node ID")


class TodoCreate(TodoBase):
    pass


# Read schemas
class User(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class Node(NodeBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    user_id: int
    
    class Config:
        from_attributes = True


class Link(LinkBase):
    id: int
    
    class Config:
        from_attributes = True


class ProgressLog(ProgressLogBase):
    id: int
    node_id: UUID
    created_at: datetime
    
    class Config:
        from_attributes = True


class Todo(TodoBase):
    id: int
    created_at: datetime
    user_id: int
    
    class Config:
        from_attributes = True


# Update schemas
class UserUpdate(BaseModel):
    email: Optional[str] = None


class NodeUpdate(BaseModel):
    title: Optional[str] = None
    is_hub: Optional[bool] = None
    status: Optional[str] = None


class LinkUpdate(BaseModel):
    strength: Optional[float] = None


class ProgressLogUpdate(BaseModel):
    text: Optional[str] = None
    next_step: Optional[str] = None
    state: Optional[str] = None


class TodoUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    due_at: Optional[datetime] = None
    node_id: Optional[UUID] = None


# Response schemas
class NodeWithRelations(Node):
    source_links: List[Link] = []
    target_links: List[Link] = []
    progress_logs: List[ProgressLog] = []
    todos: List[Todo] = []


class UserWithRelations(User):
    nodes: List[Node] = []
    todos: List[Todo] = []
