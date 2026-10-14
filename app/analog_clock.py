import math
import time
import random
import dearpygui.dearpygui as dpg

class AnalogClock:
    def __init__(self, parent, radius=50, pos=(10, 10)):
        self.tag = dpg.generate_uuid()
        self.drawlist_tag = dpg.generate_uuid()
        self.radius = radius
        self.center = (radius, radius)
        
        self.state = {
            "phase": random.uniform(0, math.pi * 2),
            "smooth_time": 0.0,
            "last_base_time": 0.0,
            "local_time": 0.0,
            "clock_offset": random.uniform(-0.2, 0.2),
            "stress": 0.0,
        }
        
        with dpg.drawlist(width=radius*2, height=radius*2, parent=parent, tag=self.drawlist_tag, pos=pos):
            # Clock face
            dpg.draw_circle(self.center, radius - 2, color=(180, 180, 180, 255), thickness=2)
            
            # Ticks
            for i in range(12):
                angle = i * math.pi / 6
                start_r = radius - 8 if i % 3 == 0 else radius - 5
                p1 = (self.center[0] + math.sin(angle) * start_r, self.center[1] - math.cos(angle) * start_r)
                p2 = (self.center[0] + math.sin(angle) * (radius - 2), self.center[1] - math.cos(angle) * (radius - 2))
                dpg.draw_line(p1, p2, color=(180, 180, 180, 255), thickness=2 if i % 3 == 0 else 1)
                
            # Hands
            self.hour_hand = dpg.draw_line(self.center, self.center, color=(255, 255, 255, 255), thickness=3)
            self.minute_hand = dpg.draw_line(self.center, self.center, color=(200, 200, 255, 255), thickness=2)
            self.second_hand = dpg.draw_line(self.center, self.center, color=(255, 100, 100, 255), thickness=1)
            dpg.draw_circle(self.center, 3, color=(200, 200, 200, 255), fill=(200, 200, 200, 255))
            
    def update(self, megaglitch_time=0.0, time_macros=None):
        if not dpg.does_item_exist(self.drawlist_tag):
            return False
            
        if time_macros is None:
            time_macros = {}
            
        warp = float(time_macros.get("local_time_warp", 0.0))
        quant = float(time_macros.get("clock_quantization", 0.0))
        sync = float(time_macros.get("sync_desync", 0.0))
        hazard = float(time_macros.get("event_hazard", 0.0))
        hysteresis = float(time_macros.get("hysteresis_speed", 0.0))
        phase_reset = float(time_macros.get("phase_reset_chance", 0.0))
        threshold = float(max(0.05, time_macros.get("rupture_threshold", 0.5)))
        reversal = float(max(0.0, min(1.0, float(time_macros.get("time_reversal", 0.0)))))
        coupling = 0.5  # representative coupling
        
        # Calculate local time delta for reversal
        if self.state["last_base_time"] == 0.0:
            self.state["last_base_time"] = megaglitch_time
            self.state["local_time"] = megaglitch_time
            
        delta_t = megaglitch_time - self.state["last_base_time"]
        self.state["last_base_time"] = megaglitch_time
        
        flow_rate = 1.0 - 2.0 * reversal
        self.state["local_time"] += delta_t * flow_rate
        local_base_time = self.state["local_time"]
        
        self.state["phase"] += 0.031 + 0.17 * hazard + 0.05 * abs(warp) + 0.06 * coupling
        
        if sync >= 0.0:
            self.state["clock_offset"] *= max(0.0, 1.0 - (0.02 + 0.16 * sync) * (0.35 + hysteresis))
        else:
            self.state["clock_offset"] += random.gauss(0.0, 0.012 + abs(sync) * 0.06)
            
        stress_drive = (
            abs(warp) * 0.035 + abs(sync) * 0.018 +
            max(0.0, hazard - 0.2) * 0.10 + max(0.0, phase_reset - 0.1) * 0.18 + coupling * 0.04
        )
        self.state["stress"] = max(0.0, self.state["stress"] * 0.9 + stress_drive)
        
        fault_event = False
        if self.state["stress"] > max(0.12, threshold):
            fault_event = True
        elif random.random() < (hazard * 0.018 + phase_reset * 0.08) * (1.0 + coupling):
            fault_event = True
            
        if fault_event:
            self.state["phase"] += random.uniform(-math.pi, math.pi) * (0.45 + hazard + coupling * 0.5)
            self.state["clock_offset"] += random.gauss(0.0, 0.12 + hazard * 0.35)
            self.state["stress"] *= max(0.12, 0.45 - coupling * 0.2)
            
        target_time = (
            local_base_time * (1.0 + warp * (0.38 + 0.18 * coupling) * math.sin(self.state["phase"] * 0.7 + local_base_time * 0.9)) +
            self.state["clock_offset"]
        )
        
        if quant > 1e-6:
            step = max(1.0 / 60.0, 0.02 + quant * 1.8)
            target_time = round(target_time / step) * step
            
        smooth = 0.02 + 0.42 * hysteresis
        self.state["smooth_time"] += (target_time - self.state["smooth_time"]) * smooth
        
        display_time = self.state["smooth_time"]
            
        # megaglitch_time is in seconds
        # Calculate angles (0 is top, clockwise)
        # 1 second = 2pi / 60 radians
        s_angle = (display_time % 60.0) * math.pi / 30
        m_angle = ((display_time / 60.0) % 60.0) * math.pi / 30
        h_angle = ((display_time / 3600.0) % 12.0) * math.pi / 6
        
        # Calculate endpoints
        h_end = (self.center[0] + math.sin(h_angle) * (self.radius * 0.5), 
                 self.center[1] - math.cos(h_angle) * (self.radius * 0.5))
        m_end = (self.center[0] + math.sin(m_angle) * (self.radius * 0.7), 
                 self.center[1] - math.cos(m_angle) * (self.radius * 0.7))
        s_end = (self.center[0] + math.sin(s_angle) * (self.radius * 0.8), 
                 self.center[1] - math.cos(s_angle) * (self.radius * 0.8))
                 
        dpg.configure_item(self.hour_hand, p2=h_end)
        dpg.configure_item(self.minute_hand, p2=m_end)
        dpg.configure_item(self.second_hand, p2=s_end)
        return True

_clocks = []

def create_clock(parent, radius=50, pos=(10, 10)):
    clock = AnalogClock(parent, radius, pos)
    _clocks.append(clock)
    return clock

def update_clocks(megaglitch_time=0.0, time_macros=None):
    global _clocks
    active_clocks = []
    for clock in _clocks:
        if clock.update(megaglitch_time, time_macros):
            active_clocks.append(clock)
    _clocks = active_clocks
