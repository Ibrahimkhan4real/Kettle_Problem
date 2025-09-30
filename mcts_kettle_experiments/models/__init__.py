# models/__init__.py
"""
MCTS algorithm implementations for the kettle environment.

This package contains various MCTS algorithm implementations designed
for energy-efficient control tasks.
"""

from .base_mcts import BaseMCTS, MCTSNode, EpisodeResult, MCTSStats
from .mcts_rave import MCTSRAVE, RAVENode
from .mcts_uct import UCT, UCTNode

__all__ = [
    'BaseMCTS',
    'MCTSNode', 
    'EpisodeResult',
    'MCTSStats',
    'MCTSRAVE',
    'RAVENode',
    'UCT',
    'UCTNode'
]

# Algorithm registry for easy access
ALGORITHM_REGISTRY = {
    'MCTS-RAVE': MCTSRAVE,
    'UCT': UCT,
    'MCTS-NoRAVE': MCTSRAVE,  # Uses RAVE class with RAVE disabled
}

def get_algorithm_class(algorithm_name: str):
    """Get algorithm class by name"""
    if algorithm_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                        f"Available: {list(ALGORITHM_REGISTRY.keys())}")
    
    return ALGORITHM_REGISTRY[algorithm_name]

def list_algorithms():
    """List all available algorithms"""
    return list(ALGORITHM_REGISTRY.keys())