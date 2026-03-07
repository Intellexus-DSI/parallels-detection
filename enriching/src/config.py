"""Configuration for the enriching pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal


@dataclass
class EnricherConfig:
    """Configuration for a specific enricher."""
    
    name: str
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class InputConfig:
    """Configuration for input files."""
    
    input_path: Path
    format: Literal["csv", "parquet", "json"] = "csv"


@dataclass
class OutputConfig:
    """Configuration for output files."""
    
    output_path: Path
    format: Literal["csv", "parquet", "json"] = "csv"
    max_lines_per_file: int = 0  # 0 = unlimited
    encoding: str = "utf-16-le"  # CSV encoding (utf-16-le, utf-8, etc.)


@dataclass
class EnrichingConfig:
    """Main configuration for the enriching pipeline."""
    
    input: InputConfig
    output: OutputConfig
    enrichers: List[EnricherConfig] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> "EnrichingConfig":
        """Create configuration from dictionary."""
        input_config = InputConfig(
            input_path=Path(data["input"]["path"]),
            format=data["input"].get("format", "csv"),
        )
        
        output_config = OutputConfig(
            output_path=Path(data["output"]["path"]),
            format=data["output"].get("format", "csv"),
            max_lines_per_file=data["output"].get("max_lines_per_file", 0),
            encoding=data["output"].get("encoding", "utf-16-le"),
        )
        
        enrichers = [
            EnricherConfig(
                name=e["name"],
                enabled=e.get("enabled", True),
                params=e.get("params", {}),
            )
            for e in data.get("enrichers", [])
        ]
        
        return cls(
            input=input_config,
            output=output_config,
            enrichers=enrichers,
        )
