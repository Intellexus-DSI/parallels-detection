"""Text normalization utilities for Tibetan text."""

import re
import logging

logger = logging.getLogger(__name__)


class TibetanTextNormalizer:
    """Normalize Tibetan text by cleaning spaces and formatting."""
    
    # Tibetan Unicode ranges
    TIBETAN_RANGE = r'[\u0F00-\u0FFF]'
    
    # Tibetan punctuation
    TSHEG = '\u0F0B'          # ་ (syllable separator)
    SHAD = '\u0F0D'           # །
    DOUBLE_SHAD = '\u0F0E'    # ༎
    GTER_TSHEG = '\u0F14'     # ༔
    
    @classmethod
    def remove_tibetan_spaces(cls, text: str) -> str:
        """
        Remove all spaces from Tibetan text while preserving structure.
        
        This removes:
        - Spaces between Tibetan characters
        - Spaces before tsheg (་)
        - Spaces before shad (།)
        - Multiple consecutive spaces
        
        Args:
            text: Input text with potential spacing issues
            
        Returns:
            Cleaned text with proper Tibetan spacing
        """
        if not text:
            return text
        
        # Remove spaces between Tibetan characters
        # Pattern: Tibetan char + space + Tibetan char
        text = re.sub(
            f'({cls.TIBETAN_RANGE})\\s+({cls.TIBETAN_RANGE})',
            r'\1\2',
            text
        )
        
        # Remove space before tsheg (་)
        text = re.sub(f'\\s+{cls.TSHEG}', cls.TSHEG, text)
        
        # Remove space before shad (།)
        text = re.sub(f'\\s+{cls.SHAD}', cls.SHAD, text)
        
        # Remove space before double shad (༎)
        text = re.sub(f'\\s+{cls.DOUBLE_SHAD}', cls.DOUBLE_SHAD, text)
        
        # Remove space before gter tsheg (༔)
        text = re.sub(f'\\s+{cls.GTER_TSHEG}', cls.GTER_TSHEG, text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'  +', ' ', text)
        
        return text.strip()
    
    @classmethod
    def normalize_spaces(cls, text: str) -> str:
        """
        Comprehensive space normalization for Tibetan text.
        
        This is a more aggressive version that removes all inappropriate spaces
        from Tibetan text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # Apply Tibetan-specific space removal
        text = cls.remove_tibetan_spaces(text)
        
        # Additional cleanup: remove spaces around underscores (common in rKTs)
        text = re.sub(r'\s*_\s*', '_', text)
        
        # Remove spaces around slashes (common in rKTs)
        text = re.sub(r'\s*/\s*', '/', text)
        
        # Final cleanup: multiple spaces to single
        text = re.sub(r'  +', ' ', text)
        
        return text.strip()
    
    @classmethod
    def normalize_text(cls, text: str, remove_spaces: bool = True) -> str:
        """
        Main normalization function.
        
        Args:
            text: Input text
            remove_spaces: Whether to remove spaces from Tibetan text
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        if remove_spaces:
            text = cls.normalize_spaces(text)
        
        return text


def normalize_tibetan_text(text: str, remove_spaces: bool = True) -> str:
    """
    Convenience function for normalizing Tibetan text.
    
    Args:
        text: Input text
        remove_spaces: Whether to remove inappropriate spaces
        
    Returns:
        Normalized text
    """
    return TibetanTextNormalizer.normalize_text(text, remove_spaces=remove_spaces)
