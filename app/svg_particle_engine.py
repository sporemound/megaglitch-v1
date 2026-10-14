"""
Gravitational Particle 2D Physics Engine
Comprehensive particle simulation with gravity, attractors, collisions, and audio reactivity
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import time


@dataclass
class Particle:
    """Represents a single particle in the simulation"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    mass: float = 1.0
    radius: float = 5.0
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA
    shape_type: str = 'circle'  # 'circle', 'polygon', 'star', 'curve'
    lifetime: int = -1  # -1 = infinite, >0 = frames remaining
    age: int = 0
    trail: List[Tuple[float, float]] = field(default_factory=list)
    active: bool = True


@dataclass
class GravityWell:
    """Represents a gravity attractor/repulsor point"""
    x: float
    y: float
    strength: float  # Positive = attract, Negative = repel
    radius: float = 100.0  # Maximum effective distance


class ParticlePhysicsEngine:
    """
    Core 2D particle physics simulation engine with multiple force types
    """
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        max_particles: int = 1000,
        gravity: Tuple[float, float] = (0.0, 0.5),
        damping: float = 0.99
    ):
        self.width = width
        self.height = height
        self.max_particles = max_particles
        self.gravity = np.array(gravity, dtype=np.float32)
        self.damping = damping
        
        self.particles: List[Particle] = []
        self.gravity_wells: List[GravityWell] = []
        
        # Physics parameters
        self.particle_interaction_enabled = False
        self.particle_interaction_strength = 0.1
        self.collision_enabled = False
        self.boundary_mode = 'wrap'  # 'wrap', 'bounce', 'destroy'
        
        # Vector field (for noise-based forces)
        self.vector_field_enabled = False
        self.vector_field_scale = 50.0
        self.vector_field_intensity = 1.0
        self.vector_field_time = 0.0
        
        # Audio reactivity
        self.audio_reactive = False
        self.audio_data: Optional[Dict[str, Any]] = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_update_time = time.time()
        
    def set_dimensions(self, width: int, height: int):
        """Update canvas dimensions"""
        self.width = width
        self.height = height
        
    def add_particle(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        mass: float = 1.0,
        radius: float = 5.0,
        color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        shape_type: str = 'circle',
        lifetime: int = -1
    ) -> Optional[Particle]:
        """Add a new particle to the simulation"""
        if len(self.particles) >= self.max_particles:
            # Remove oldest inactive or expired particle
            self._cleanup_particles()
            if len(self.particles) >= self.max_particles:
                return None
                
        particle = Particle(
            x=x, y=y, vx=vx, vy=vy,
            mass=mass, radius=radius,
            color=color, shape_type=shape_type,
            lifetime=lifetime
        )
        self.particles.append(particle)
        return particle
        
    def spawn_particles_batch(
        self,
        count: int,
        spawn_area: Tuple[float, float, float, float] = None,  # (x, y, width, height)
        velocity_range: Tuple[float, float] = (-5.0, 5.0),
        size_range: Tuple[float, float] = (3.0, 8.0),
        color_palette: List[Tuple[int, int, int, int]] = None,
        shape_types: List[str] = None,
        lifetime_range: Tuple[int, int] = (-1, -1)
    ):
        """Spawn multiple particles at once"""
        if spawn_area is None:
            spawn_area = (0, 0, self.width, self.height)
            
        if color_palette is None:
            color_palette = [(255, 255, 255, 255)]
            
        if shape_types is None:
            shape_types = ['circle']
            
        sx, sy, sw, sh = spawn_area
        
        for _ in range(count):
            x = np.random.uniform(sx, sx + sw)
            y = np.random.uniform(sy, sy + sh)
            vx = np.random.uniform(*velocity_range)
            vy = np.random.uniform(*velocity_range)
            radius = np.random.uniform(*size_range)
            color = color_palette[np.random.randint(0, len(color_palette))]
            shape = shape_types[np.random.randint(0, len(shape_types))]
            
            if lifetime_range[0] == -1 and lifetime_range[1] == -1:
                lifetime = -1
            else:
                lifetime = np.random.randint(lifetime_range[0], lifetime_range[1] + 1)
                
            self.add_particle(x, y, vx, vy, radius=radius, color=color, 
                            shape_type=shape, lifetime=lifetime)
    
    def add_gravity_well(self, x: float, y: float, strength: float, radius: float = 100.0):
        """Add a gravity well attractor/repulsor"""
        well = GravityWell(x=x, y=y, strength=strength, radius=radius)
        self.gravity_wells.append(well)
        return well
        
    def clear_gravity_wells(self):
        """Remove all gravity wells"""
        self.gravity_wells.clear()
        
    def update_audio_data(self, audio_data: Dict[str, Any]):
        """Update audio data for reactivity"""
        self.audio_data = audio_data
        
    def update(self, dt: float = 1.0):
        """
        Main physics update loop
        dt: delta time (typically 1.0 for per-frame updates)
        """
        self.frame_count += 1
        self.vector_field_time += dt * 0.016  # Approximate frame time
        
        # Audio-reactive particle spawning
        if self.audio_reactive and self.audio_data:
            self._handle_audio_reactivity()
        
        # Update each particle
        for particle in self.particles:
            if not particle.active:
                continue
                
            # Age and lifetime management
            particle.age += 1
            if particle.lifetime > 0:
                particle.lifetime -= 1
                if particle.lifetime <= 0:
                    particle.active = False
                    continue
            
            # Reset acceleration
            particle.ax = 0.0
            particle.ay = 0.0
            
            # Apply global gravity
            particle.ax += self.gravity[0]
            particle.ay += self.gravity[1]
            
            # Apply gravity wells
            for well in self.gravity_wells:
                self._apply_gravity_well(particle, well)
            
            # Apply vector field forces
            if self.vector_field_enabled:
                self._apply_vector_field(particle)
            
            # Apply particle-to-particle interactions
            if self.particle_interaction_enabled:
                self._apply_particle_interactions(particle)
            
            # Integrate physics (Euler integration)
            particle.vx += particle.ax * dt
            particle.vy += particle.ay * dt
            
            # Apply damping
            particle.vx *= self.damping
            particle.vy *= self.damping
            
            # Update position
            old_x, old_y = particle.x, particle.y
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            
            # Update trail
            if len(particle.trail) == 0 or \
               (particle.x - old_x)**2 + (particle.y - old_y)**2 > 1.0:
                particle.trail.append((old_x, old_y))
                # Limit trail length
                max_trail = 100
                if len(particle.trail) > max_trail:
                    particle.trail.pop(0)
            
            # Handle boundaries
            self._handle_boundary(particle)
            
        # Handle collisions if enabled
        if self.collision_enabled:
            self._handle_collisions()
            
        # Cleanup dead particles periodically
        if self.frame_count % 60 == 0:
            self._cleanup_particles()
    
    def _apply_gravity_well(self, particle: Particle, well: GravityWell):
        """Apply force from a gravity well to a particle"""
        dx = well.x - particle.x
        dy = well.y - particle.y
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq)
        
        if dist < 1.0 or dist > well.radius:
            return
            
        # Gravitational force: F = G * m1 * m2 / r^2
        # Simplified: F = strength / dist^2
        force_magnitude = well.strength / dist_sq
        
        # Normalize direction and apply force
        particle.ax += (dx / dist) * force_magnitude
        particle.ay += (dy / dist) * force_magnitude
    
    def _apply_vector_field(self, particle: Particle):
        """Apply noise-based vector field force"""
        # Use Perlin-like noise approximation with sine waves
        scale = self.vector_field_scale
        t = self.vector_field_time
        
        # Multi-octave noise
        fx = 0.0
        fy = 0.0
        
        for octave in range(3):
            freq = (2 ** octave) / scale
            amp = 1.0 / (2 ** octave)
            
            fx += amp * np.sin(particle.x * freq + t + octave * 1.5)
            fy += amp * np.cos(particle.y * freq + t * 0.7 + octave * 2.1)
        
        # Apply force
        intensity = self.vector_field_intensity
        particle.ax += fx * intensity
        particle.ay += fy * intensity
    
    def _apply_particle_interactions(self, particle: Particle):
        """Apply forces from nearby particles"""
        interaction_radius = 50.0
        
        for other in self.particles:
            if other is particle or not other.active:
                continue
                
            dx = other.x - particle.x
            dy = other.y - particle.y
            dist_sq = dx * dx + dy * dy
            
            if dist_sq > interaction_radius ** 2 or dist_sq < 1.0:
                continue
                
            dist = np.sqrt(dist_sq)
            
            # Repulsion force (particles push each other away)
            force = -self.particle_interaction_strength / dist_sq
            
            particle.ax += (dx / dist) * force
            particle.ay += (dy / dist) * force
    
    def _handle_boundary(self, particle: Particle):
        """Handle particle boundary conditions"""
        if self.boundary_mode == 'wrap':
            # Wrap around screen edges
            if particle.x < 0:
                particle.x += self.width
            elif particle.x > self.width:
                particle.x -= self.width
                
            if particle.y < 0:
                particle.y += self.height
            elif particle.y > self.height:
                particle.y -= self.height
                
        elif self.boundary_mode == 'bounce':
            # Bounce off edges
            if particle.x < 0 or particle.x > self.width:
                particle.vx *= -0.8
                particle.x = np.clip(particle.x, 0, self.width)
                
            if particle.y < 0 or particle.y > self.height:
                particle.vy *= -0.8
                particle.y = np.clip(particle.y, 0, self.height)
                
        elif self.boundary_mode == 'destroy':
            # Destroy particles outside bounds
            if (particle.x < -100 or particle.x > self.width + 100 or
                particle.y < -100 or particle.y > self.height + 100):
                particle.active = False
    
    def _handle_collisions(self):
        """Handle particle-to-particle collisions (simple elastic collision)"""
        active_particles = [p for p in self.particles if p.active]
        
        for i, p1 in enumerate(active_particles):
            for p2 in active_particles[i+1:]:
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dist_sq = dx * dx + dy * dy
                min_dist = p1.radius + p2.radius
                
                if dist_sq < min_dist ** 2 and dist_sq > 0:
                    dist = np.sqrt(dist_sq)
                    
                    # Separate particles
                    overlap = min_dist - dist
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    
                    p1.x -= dx_norm * overlap * 0.5
                    p1.y -= dy_norm * overlap * 0.5
                    p2.x += dx_norm * overlap * 0.5
                    p2.y += dy_norm * overlap * 0.5
                    
                    # Elastic collision response
                    dvx = p2.vx - p1.vx
                    dvy = p2.vy - p1.vy
                    
                    dot_product = dvx * dx_norm + dvy * dy_norm
                    
                    if dot_product < 0:  # Moving towards each other
                        mass_sum = p1.mass + p2.mass
                        p1.vx += 2 * p2.mass / mass_sum * dot_product * dx_norm
                        p1.vy += 2 * p2.mass / mass_sum * dot_product * dy_norm
                        p2.vx -= 2 * p1.mass / mass_sum * dot_product * dx_norm
                        p2.vy -= 2 * p1.mass / mass_sum * dot_product * dy_norm
    
    def _handle_audio_reactivity(self):
        """Spawn particles or modify behavior based on audio input"""
        if not self.audio_data:
            return
            
        amplitude = self.audio_data.get('amplitude', 0.0)
        bass = self.audio_data.get('bass', 0.0)
        mid = self.audio_data.get('mid', 0.0)
        high = self.audio_data.get('high', 0.0)
        beat_detected = self.audio_data.get('beat', False)
        
        # Spawn particles based on amplitude
        spawn_rate = int(amplitude * 5)
        if spawn_rate > 0:
            # Color based on frequency bands
            r = int(np.clip(bass * 255, 0, 255))
            g = int(np.clip(mid * 255, 0, 255))
            b = int(np.clip(high * 255, 0, 255))
            
            for _ in range(min(spawn_rate, 10)):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height * 0.3)  # Spawn in top region
                
                # Velocity based on audio
                speed = 5.0 + amplitude * 10.0
                angle = np.random.uniform(0, 2 * np.pi)
                vx = np.cos(angle) * speed
                vy = np.sin(angle) * speed
                
                radius = 3.0 + amplitude * 5.0
                
                self.add_particle(
                    x, y, vx, vy,
                    radius=radius,
                    color=(r, g, b, 200),
                    lifetime=60 + int(amplitude * 120)
                )
        
        # Beat detection: create gravity pulse
        if beat_detected:
            # Add temporary gravity well at center
            self.add_gravity_well(
                self.width / 2,
                self.height / 2,
                strength=bass * 50.0,
                radius=min(self.width, self.height) * 0.5
            )
    
    def _cleanup_particles(self):
        """Remove inactive particles"""
        self.particles = [p for p in self.particles if p.active]
    
    def get_active_particles(self) -> List[Particle]:
        """Return list of all active particles"""
        return [p for p in self.particles if p.active]
    
    def clear_all_particles(self):
        """Remove all particles"""
        self.particles.clear()
    
    def reset(self):
        """Reset the entire simulation"""
        self.particles.clear()
        self.gravity_wells.clear()
        self.frame_count = 0
        self.vector_field_time = 0.0










