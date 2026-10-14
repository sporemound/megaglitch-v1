"""
SVG Generator and Renderer
Converts particle states to SVG vector graphics and rasterizes for display
"""

import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import io
import logging
import os
from xml.etree import ElementTree as ET

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False
    logging.warning("cairosvg not available - SVG rasterization will use PIL fallback")

from svg_particle_engine import Particle


class SVGGenerator:
    """
    Generates SVG graphics from particle simulation data
    """
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.background_color = None  # None = transparent
        
    def set_dimensions(self, width: int, height: int):
        """Update canvas dimensions"""
        self.width = width
        self.height = height
    
    def generate_svg(
        self,
        particles: List[Particle],
        include_trails: bool = True,
        trail_fade: bool = True,
        use_gradients: bool = True
    ) -> str:
        """
        Generate SVG XML string from particle list
        
        Args:
            particles: List of particles to render
            include_trails: Whether to render particle trails
            trail_fade: Whether trails should fade out
            use_gradients: Whether to use gradient fills
            
        Returns:
            SVG XML as string
        """
        # Create SVG root element
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': str(self.width),
            'height': str(self.height),
            'viewBox': f'0 0 {self.width} {self.height}'
        })
        
        # Add definitions for gradients and filters
        defs = ET.SubElement(svg, 'defs')
        
        if use_gradients:
            self._add_gradient_definitions(defs, particles)
        
        # Add background if specified
        if self.background_color:
            bg = ET.SubElement(svg, 'rect', {
                'width': str(self.width),
                'height': str(self.height),
                'fill': self._color_to_hex(self.background_color)
            })
        
        # Create layer for trails
        if include_trails:
            trail_group = ET.SubElement(svg, 'g', {'id': 'trails'})
            for particle in particles:
                if len(particle.trail) > 1:
                    self._add_trail(trail_group, particle, trail_fade)
        
        # Create layer for particles
        particle_group = ET.SubElement(svg, 'g', {'id': 'particles'})
        for idx, particle in enumerate(particles):
            self._add_particle_shape(particle_group, particle, idx, use_gradients)
        
        # Convert to string
        svg_string = ET.tostring(svg, encoding='unicode', method='xml')
        return svg_string
    
    def _add_gradient_definitions(self, defs: ET.Element, particles: List[Particle]):
        """Add radial gradient definitions for particles"""
        # Create a few gradient variants
        for i, particle in enumerate(particles[:20]):  # Limit to avoid bloat
            grad_id = f'grad_{i}'
            grad = ET.SubElement(defs, 'radialGradient', {'id': grad_id})
            
            r, g, b, a = particle.color
            
            # Bright center
            stop1 = ET.SubElement(grad, 'stop', {
                'offset': '0%',
                'stop-color': f'rgb({min(255, r+50)}, {min(255, g+50)}, {min(255, b+50)})',
                'stop-opacity': str(a / 255.0)
            })
            
            # Fade to edge
            stop2 = ET.SubElement(grad, 'stop', {
                'offset': '100%',
                'stop-color': f'rgb({r}, {g}, {b})',
                'stop-opacity': str((a / 255.0) * 0.3)
            })
    
    def _add_trail(self, parent: ET.Element, particle: Particle, fade: bool = True):
        """Add particle trail as SVG path"""
        if len(particle.trail) < 2:
            return
            
        # Build path data
        trail_points = particle.trail + [(particle.x, particle.y)]
        
        # Create smooth path using quadratic bezier curves
        path_data = f'M {trail_points[0][0]:.2f},{trail_points[0][1]:.2f}'
        
        for i in range(1, len(trail_points)):
            x, y = trail_points[i]
            
            if i < len(trail_points) - 1:
                # Use quadratic curves for smoothness
                next_x, next_y = trail_points[i + 1]
                cx = (x + next_x) / 2
                cy = (y + next_y) / 2
                path_data += f' Q {x:.2f},{y:.2f} {cx:.2f},{cy:.2f}'
            else:
                path_data += f' L {x:.2f},{y:.2f}'
        
        r, g, b, a = particle.color
        
        # Calculate opacity based on trail fade
        if fade:
            opacity = (a / 255.0) * 0.5
        else:
            opacity = a / 255.0
        
        # Create path element
        path = ET.SubElement(parent, 'path', {
            'd': path_data,
            'stroke': f'rgb({r}, {g}, {b})',
            'stroke-width': str(max(1.0, particle.radius * 0.3)),
            'stroke-opacity': str(opacity),
            'fill': 'none',
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round'
        })
    
    def _add_particle_shape(
        self,
        parent: ET.Element,
        particle: Particle,
        idx: int,
        use_gradients: bool
    ):
        """Add particle shape to SVG"""
        r, g, b, a = particle.color
        fill_color = f'rgb({r}, {g}, {b})'
        opacity = a / 255.0
        
        # Use gradient if available
        if use_gradients and idx < 20:
            fill = f'url(#grad_{idx})'
            opacity_attr = None
        else:
            fill = fill_color
            opacity_attr = str(opacity)
        
        if particle.shape_type == 'circle':
            attrs = {
                'cx': str(particle.x),
                'cy': str(particle.y),
                'r': str(particle.radius),
                'fill': fill
            }
            if opacity_attr:
                attrs['fill-opacity'] = opacity_attr
            ET.SubElement(parent, 'circle', attrs)
            
        elif particle.shape_type == 'polygon':
            # Create regular polygon
            sides = 6
            points = self._generate_polygon_points(
                particle.x, particle.y, particle.radius, sides
            )
            attrs = {
                'points': points,
                'fill': fill
            }
            if opacity_attr:
                attrs['fill-opacity'] = opacity_attr
            ET.SubElement(parent, 'polygon', attrs)
            
        elif particle.shape_type == 'star':
            # Create star shape
            points = self._generate_star_points(
                particle.x, particle.y, particle.radius, 5
            )
            attrs = {
                'points': points,
                'fill': fill
            }
            if opacity_attr:
                attrs['fill-opacity'] = opacity_attr
            ET.SubElement(parent, 'polygon', attrs)
            
        elif particle.shape_type == 'curve':
            # Create abstract curve shape
            path_data = self._generate_curve_path(
                particle.x, particle.y, particle.radius
            )
            attrs = {
                'd': path_data,
                'fill': fill
            }
            if opacity_attr:
                attrs['fill-opacity'] = opacity_attr
            ET.SubElement(parent, 'path', attrs)
    
    def _generate_polygon_points(
        self,
        cx: float,
        cy: float,
        radius: float,
        sides: int
    ) -> str:
        """Generate points for a regular polygon"""
        points = []
        for i in range(sides):
            angle = 2 * np.pi * i / sides - np.pi / 2
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            points.append(f'{x:.2f},{y:.2f}')
        return ' '.join(points)
    
    def _generate_star_points(
        self,
        cx: float,
        cy: float,
        radius: float,
        points: int
    ) -> str:
        """Generate points for a star shape"""
        inner_radius = radius * 0.4
        vertices = []
        
        for i in range(points * 2):
            angle = np.pi * i / points - np.pi / 2
            r = radius if i % 2 == 0 else inner_radius
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            vertices.append(f'{x:.2f},{y:.2f}')
        
        return ' '.join(vertices)
    
    def _generate_curve_path(
        self,
        cx: float,
        cy: float,
        radius: float
    ) -> str:
        """Generate an abstract curved path"""
        # Create a blob-like shape using cubic bezier curves
        r = radius
        path = f'M {cx},{cy - r}'
        
        # Use control points to create organic curves
        ctrl_offset = r * 0.552  # Magic number for circular curves
        
        path += f' C {cx + ctrl_offset},{cy - r} {cx + r},{cy - ctrl_offset} {cx + r},{cy}'
        path += f' C {cx + r},{cy + ctrl_offset} {cx + ctrl_offset},{cy + r} {cx},{cy + r}'
        path += f' C {cx - ctrl_offset},{cy + r} {cx - r},{cy + ctrl_offset} {cx - r},{cy}'
        path += f' C {cx - r},{cy - ctrl_offset} {cx - ctrl_offset},{cy - r} {cx},{cy - r}'
        path += ' Z'
        
        return path
    
    def _color_to_hex(self, color: Tuple[int, int, int, int]) -> str:
        """Convert RGBA tuple to hex color"""
        r, g, b, a = color
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def save_svg(self, svg_string: str, filepath: str):
        """Save SVG string to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(svg_string)
            logging.debug(f"Saved SVG to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save SVG: {e}")
    
    def rasterize_svg(
        self,
        svg_string: str,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Convert SVG string to PIL Image
        
        Args:
            svg_string: SVG XML as string
            output_size: Optional (width, height) for output. Defaults to canvas size
            
        Returns:
            PIL Image in RGBA mode
        """
        if output_size is None:
            output_size = (self.width, self.height)
        
        width, height = output_size
        
        if CAIROSVG_AVAILABLE:
            try:
                # Use cairosvg for high-quality rendering
                png_data = cairosvg.svg2png(
                    bytestring=svg_string.encode('utf-8'),
                    output_width=width,
                    output_height=height
                )
                img = Image.open(io.BytesIO(png_data))
                return img.convert('RGBA')
            except Exception as e:
                logging.error(f"cairosvg rendering failed: {e}, using fallback")
        
        # Fallback: create blank image with warning
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        logging.warning("SVG rasterization not available - install cairosvg for proper rendering")
        return img
    
    def render_particles_to_image(
        self,
        particles: List[Particle],
        include_trails: bool = True,
        trail_fade: bool = True,
        use_gradients: bool = True,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Complete pipeline: generate SVG from particles and rasterize
        
        Returns:
            PIL Image in RGBA mode
        """
        svg_string = self.generate_svg(
            particles,
            include_trails=include_trails,
            trail_fade=trail_fade,
            use_gradients=use_gradients
        )
        
        return self.rasterize_svg(svg_string, output_size)


def composite_images(background: Image.Image, overlay: Image.Image, blend_mode: str = 'normal') -> Image.Image:
    """
    Composite two images together
    
    Args:
        background: Base image
        overlay: Image to overlay (should be RGBA)
        blend_mode: 'normal', 'add', 'multiply', 'screen'
        
    Returns:
        Composited PIL Image
    """
    # Ensure both images are RGBA
    bg = background.convert('RGBA')
    ov = overlay.convert('RGBA')
    
    # Resize overlay to match background if needed
    if bg.size != ov.size:
        ov = ov.resize(bg.size, Image.Resampling.LANCZOS)
    
    if blend_mode == 'normal':
        # Alpha compositing
        result = Image.alpha_composite(bg, ov)
        
    elif blend_mode == 'add':
        # Additive blending
        bg_array = np.array(bg, dtype=np.float32)
        ov_array = np.array(ov, dtype=np.float32)
        
        # Extract alpha
        ov_alpha = ov_array[:, :, 3:4] / 255.0
        
        # Add RGB channels weighted by overlay alpha
        result_array = np.clip(bg_array[:, :, :3] + ov_array[:, :, :3] * ov_alpha, 0, 255)
        
        # Keep background alpha
        result_array = np.dstack([result_array, bg_array[:, :, 3]])
        result = Image.fromarray(result_array.astype(np.uint8), 'RGBA')
        
    elif blend_mode == 'multiply':
        # Multiplicative blending
        bg_array = np.array(bg, dtype=np.float32) / 255.0
        ov_array = np.array(ov, dtype=np.float32) / 255.0
        
        ov_alpha = ov_array[:, :, 3:4]
        
        # Multiply RGB
        result_rgb = bg_array[:, :, :3] * (1 - ov_alpha + ov_alpha * ov_array[:, :, :3])
        result_array = np.clip(result_rgb * 255, 0, 255)
        result_array = np.dstack([result_array, bg_array[:, :, 3] * 255])
        result = Image.fromarray(result_array.astype(np.uint8), 'RGBA')
        
    elif blend_mode == 'screen':
        # Screen blending
        bg_array = np.array(bg, dtype=np.float32) / 255.0
        ov_array = np.array(ov, dtype=np.float32) / 255.0
        
        ov_alpha = ov_array[:, :, 3:4]
        
        # Screen formula: 1 - (1-a)*(1-b)
        result_rgb = 1 - (1 - bg_array[:, :, :3]) * (1 - ov_alpha * ov_array[:, :, :3])
        result_array = np.clip(result_rgb * 255, 0, 255)
        result_array = np.dstack([result_array, bg_array[:, :, 3] * 255])
        result = Image.fromarray(result_array.astype(np.uint8), 'RGBA')
    
    else:
        result = Image.alpha_composite(bg, ov)
    
    return result










