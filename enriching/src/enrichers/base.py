"""Base class for enrichers."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models import EnrichedParallel


class BaseEnricher(ABC):
    """
    Base class for all enrichers.
    
    Enrichers add new fields to parallel matches based on custom logic.
    Each enricher should implement the enrich() method.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the enricher.
        
        Args:
            params: Configuration parameters for this enricher.
        """
        self.params = params or {}
    
    @abstractmethod
    def enrich(self, parallel: EnrichedParallel) -> EnrichedParallel:
        """
        Enrich a parallel match with additional fields.
        
        Args:
            parallel: The parallel match to enrich.
            
        Returns:
            The enriched parallel match (can modify in-place or return new instance).
        """
        pass
    
    @property
    @abstractmethod
    def field_names(self) -> list:
        """
        Return the names of fields this enricher adds.
        
        Returns:
            List of field names that will be added to the output.
        """
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(params={self.params})"
