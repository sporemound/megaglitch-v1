import dearpygui.dearpygui as dpg
from pathlib import Path
import os
import logging


DEFAULT_UI_FONT_SIZE = 20


def _find_standard_ui_font():
    """Return a conventional system sans-serif font when one is available."""
    if os.name == "nt":
        candidates = (
            Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf",
            Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        )
    elif os.name == "posix":
        candidates = (
            # DejaVu Sans is the broadly available traditional Linux UI default.
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("/System/Library/Fonts/Helvetica.ttc"),
        )
    else:
        candidates = ()

    return next((path for path in candidates if path.is_file()), None)


def apply_modern_theme():
    """Apply a modern aesthetic to DearPyGui.
    Theme: Fluorescent mycelium in a pygmy forest at night.
    """
    
    # 1. Setup a conventional, readable system UI font. On Linux this resolves
    # to DejaVu Sans on a standard installation. The 20 px baseline favors
    # comfortable reading on typical desktop displays.
    font_path = _find_standard_ui_font()
    if font_path is not None:
        try:
            with dpg.font_registry():
                default_font = dpg.add_font(str(font_path), DEFAULT_UI_FONT_SIZE)
            dpg.bind_font(default_font)
        except Exception as e:
            logging.warning(f"Could not load system UI font {font_path}: {e}")
    else:
        logging.warning("No standard system UI font found; using Dear PyGui's default font")

    # 2. Setup Theme
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Styling: Glassmorphism and rounded corners
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)

            # Colors: Pygmy forest at night with fluorescent mycelium accents
            # Backgrounds - deep, dark mossy forest
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (8, 12, 10, 250))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (12, 18, 14, 180))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (15, 22, 18, 240))
            
            # Frames (Inputs, Sliders, Checkboxes)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (20, 30, 24, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (35, 55, 45, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (0, 180, 150, 150)) # Cyan glow
            
            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, (25, 40, 32, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 180, 150, 200)) # Fluorescent cyan
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 255, 204, 255)) # Bright cyan
            
            # Headers & Title Bars
            dpg.add_theme_color(dpg.mvThemeCol_Header, (30, 50, 40, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (0, 180, 150, 180))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (0, 255, 204, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (12, 20, 16, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (20, 45, 35, 255))
            
            # Sliders/Grabs (Mycelium glow)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (140, 255, 50, 200)) # Neon green
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (180, 255, 100, 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 255, 204, 255))
            
            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (150, 150, 150, 255))

        # Node specific theme
        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 8)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (25, 30, 28, 240)) # Slightly lighter background
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (35, 45, 40, 250))
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (40, 55, 48, 255))
            
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255)) # Ensure node text is white
            
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (22, 40, 32, 255))
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (35, 65, 50, 255))
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (0, 150, 120, 255)) # Cyan selected
            
            dpg.add_theme_color(dpg.mvNodeCol_Link, (140, 255, 50, 200)) # Neon green mycelium threads
            dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (200, 255, 100, 255))
            dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (0, 255, 204, 255)) # Bright cyan
            
            dpg.add_theme_color(dpg.mvNodeCol_Pin, (0, 255, 204, 255))
            dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (140, 255, 50, 255))
            
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelector, (0, 255, 204, 50))
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelectorOutline, (0, 255, 204, 150))
            
    dpg.bind_theme(global_theme)
