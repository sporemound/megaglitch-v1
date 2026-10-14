"""Real-time audio and audiovisual modulation support."""

from .fm_engine import FMEngine
from .modulation_bus import ModulationBus, ModulationMapping

__all__ = ["FMEngine", "ModulationBus", "ModulationMapping"]
