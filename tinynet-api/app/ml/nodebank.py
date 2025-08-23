"""
Node Bank for storing and retrieving node embeddings
Provides similarity search for link hints in TinyNet
"""

import sqlite3
import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import json


class NodeBank:
    """
    Node Bank for storing and retrieving node embeddings.
    
    Stores up to N=200 recent node embeddings (32-d from TinyNet trunk)
    with (node_id, title, vec, updated_at) in SQLite.
    """
    
    def __init__(self, db_path: str = "tinynet.db", max_nodes: int = 200):
        """
        Initialize NodeBank.
        
        Args:
            db_path: Path to SQLite database
            max_nodes: Maximum number of nodes to store
        """
        self.db_path = db_path
        self.max_nodes = max_nodes
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create node_embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                vec TEXT NOT NULL,  -- JSON array of 32 floats
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for similarity search
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_updated_at 
            ON node_embeddings(updated_at DESC)
        ''')
        
        conn.commit()
        conn.close()
        
        logging.info(f"NodeBank initialized with max_nodes={self.max_nodes}")
    
    def upsert_node_embedding(self, node_id: str, title: str, vec: np.ndarray):
        """
        Insert or update a node embedding.
        
        Args:
            node_id: Unique node identifier
            title: Node title
            vec: 32-dimensional embedding vector
        """
        if vec.shape != (32,):
            raise ValueError(f"Expected vector shape (32,), got {vec.shape}")
        
        # Convert vector to JSON string
        vec_json = json.dumps(vec.tolist())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if we need to remove old nodes
            cursor.execute('SELECT COUNT(*) FROM node_embeddings')
            count = cursor.fetchone()[0]
            
            if count >= self.max_nodes:
                # Remove oldest node
                cursor.execute('''
                    DELETE FROM node_embeddings 
                    WHERE node_id = (
                        SELECT node_id FROM node_embeddings 
                        ORDER BY updated_at ASC 
                        LIMIT 1
                    )
                ''')
                logging.info(f"Removed oldest node to maintain max_nodes={self.max_nodes}")
            
            # Insert or update
            cursor.execute('''
                INSERT OR REPLACE INTO node_embeddings (node_id, title, vec, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (node_id, title, vec_json, datetime.now().isoformat()))
            
            conn.commit()
            logging.info(f"Upserted node embedding for {node_id}: {title}")
            
        except Exception as e:
            logging.error(f"Error upserting node embedding: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def topk_similar(self, vec: np.ndarray, k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Find top-k most similar nodes to the given vector.
        
        Args:
            vec: Query vector (32-dimensional)
            k: Number of similar nodes to return
            
        Returns:
            List of (node_id, title, similarity) tuples, sorted by similarity
        """
        if vec.shape != (32,):
            raise ValueError(f"Expected vector shape (32,), got {vec.shape}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all stored embeddings
            cursor.execute('SELECT node_id, title, vec FROM node_embeddings')
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            similarities = []
            for node_id, title, vec_json in rows:
                stored_vec = np.array(json.loads(vec_json))
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(vec, stored_vec)
                similarities.append((node_id, title, similarity))
            
            # Sort by similarity (descending) and return top-k
            similarities.sort(key=lambda x: x[2], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            logging.error(f"Error computing similarities: {e}")
            return []
        finally:
            conn.close()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_node_count(self) -> int:
        """Get the current number of stored nodes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM node_embeddings')
            count = cursor.fetchone()[0]
            return count
        finally:
            conn.close()
    
    def clear_all(self):
        """Clear all stored node embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM node_embeddings')
            conn.commit()
            logging.info("Cleared all node embeddings")
        finally:
            conn.close()
    
    def get_recent_nodes(self, limit: int = 10) -> List[Tuple[str, str, str]]:
        """
        Get recent nodes for debugging.
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            List of (node_id, title, updated_at) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT node_id, title, updated_at 
                FROM node_embeddings 
                ORDER BY updated_at DESC 
                LIMIT ?
            ''', (limit,))
            
            return cursor.fetchall()
        finally:
            conn.close()
    
    def add_sample_nodes(self):
        """Add some sample nodes for testing."""
        sample_nodes = [
            ("node_001", "Running Progress", np.random.randn(32)),
            ("node_002", "Guitar Practice", np.random.randn(32)),
            ("node_003", "Learning AI Basics", np.random.randn(32)),
            ("node_004", "Fitness Goals", np.random.randn(32)),
            ("node_005", "Work Projects", np.random.randn(32))
        ]
        
        for node_id, title, vec in sample_nodes:
            # Normalize the vector
            vec = vec / np.linalg.norm(vec)
            self.upsert_node_embedding(node_id, title, vec)
        
        logging.info(f"Added {len(sample_nodes)} sample nodes")
