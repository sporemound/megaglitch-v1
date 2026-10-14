use eframe::egui::{self, Align2, Color32, FontId, Response, Sense, Stroke, Ui, Vec2, Widget};

pub const BG: Color32 = Color32::from_rgb(7, 9, 12);
pub const PANEL: Color32 = Color32::from_rgb(12, 16, 20);
pub const PANEL_RAISED: Color32 = Color32::from_rgb(17, 22, 27);
pub const HAIRLINE: Color32 = Color32::from_rgb(39, 49, 58);
pub const TEXT: Color32 = Color32::from_rgb(222, 230, 234);
pub const MUTED: Color32 = Color32::from_rgb(112, 127, 136);
pub const ACCENT: Color32 = Color32::from_rgb(72, 211, 202);
pub const SIGNAL: Color32 = Color32::from_rgb(151, 235, 111);
pub const WARNING: Color32 = Color32::from_rgb(244, 169, 76);

pub fn install(ctx: &egui::Context) {
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    style.spacing.item_spacing = Vec2::new(8.0, 7.0);
    style.spacing.button_padding = Vec2::new(10.0, 6.0);
    style.spacing.interact_size = Vec2::new(34.0, 28.0);
    style.visuals.dark_mode = true;
    style.visuals.panel_fill = PANEL;
    style.visuals.window_fill = PANEL;
    style.visuals.extreme_bg_color = BG;
    style.visuals.faint_bg_color = PANEL_RAISED;
    style.visuals.override_text_color = Some(TEXT);
    style.visuals.selection.bg_fill = ACCENT.gamma_multiply(0.35);
    style.visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    style.visuals.widgets.noninteractive.bg_fill = PANEL;
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, HAIRLINE);
    style.visuals.widgets.inactive.bg_fill = PANEL_RAISED;
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, HAIRLINE);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(23, 33, 39);
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.7));
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(25, 42, 45);
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    style.visuals.window_corner_radius = 2.0.into();
    style.visuals.widgets.noninteractive.corner_radius = 2.0.into();
    style.visuals.widgets.inactive.corner_radius = 2.0.into();
    style.visuals.widgets.hovered.corner_radius = 2.0.into();
    style.visuals.widgets.active.corner_radius = 2.0.into();
    ctx.set_style_of(egui::Theme::Dark, style);
}

pub fn section(ui: &mut Ui, eyebrow: &str, title: &str) {
    ui.label(
        egui::RichText::new(eyebrow.to_uppercase())
            .size(9.0)
            .color(ACCENT)
            .strong(),
    );
    ui.label(egui::RichText::new(title).size(16.0).color(TEXT).strong());
    ui.add_space(3.0);
    let rect = ui.available_rect_before_wrap();
    ui.painter().line_segment(
        [rect.left_top(), rect.right_top()],
        Stroke::new(1.0, HAIRLINE),
    );
    ui.add_space(5.0);
}

pub fn status_lamp(ui: &mut Ui, label: &str, active: bool, color: Color32) -> Response {
    let desired = Vec2::new(ui.spacing().interact_size.x + 66.0, 24.0);
    let (rect, response) = ui.allocate_exact_size(desired, Sense::hover());
    let lamp = if active {
        color
    } else {
        MUTED.gamma_multiply(0.35)
    };
    ui.painter()
        .circle_filled(rect.left_center() + Vec2::new(7.0, 0.0), 3.5, lamp);
    ui.painter().text(
        rect.left_center() + Vec2::new(17.0, 0.0),
        Align2::LEFT_CENTER,
        label,
        FontId::monospace(10.0),
        if active { TEXT } else { MUTED },
    );
    response
}

pub fn meter(ui: &mut Ui, label: &str, value: f32, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .monospace()
                .size(9.0)
                .color(MUTED),
        );
        let width = ui.available_width().max(24.0);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 6.0), Sense::hover());
        ui.painter().rect_filled(rect, 1.0, BG);
        let fill = egui::Rect::from_min_size(
            rect.min,
            Vec2::new(rect.width() * value.clamp(0.0, 1.0), rect.height()),
        );
        ui.painter().rect_filled(fill, 1.0, color);
    });
}

pub struct Knob<'a> {
    value: &'a mut f32,
    label: &'a str,
    range: std::ops::RangeInclusive<f32>,
}

impl<'a> Knob<'a> {
    pub fn new(value: &'a mut f32, label: &'a str, range: std::ops::RangeInclusive<f32>) -> Self {
        Self {
            value,
            label,
            range,
        }
    }
}

impl Widget for Knob<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let desired = Vec2::new(58.0, 72.0);
        let (rect, mut response) = ui.allocate_exact_size(desired, Sense::click_and_drag());
        let start = *self.range.start();
        let end = *self.range.end();
        if response.dragged() {
            *self.value =
                (*self.value - response.drag_delta().y * (end - start) / 180.0).clamp(start, end);
            response.mark_changed();
        }
        let t = ((*self.value - start) / (end - start)).clamp(0.0, 1.0);
        let center = rect.center_top() + Vec2::new(0.0, 26.0);
        let radius = 18.0;
        ui.painter().circle_filled(center, radius, BG);
        ui.painter()
            .circle_stroke(center, radius, Stroke::new(1.0, HAIRLINE));
        let angle = std::f32::consts::PI * (0.75 + 1.5 * t);
        let tip = center + Vec2::angled(angle) * (radius - 5.0);
        ui.painter()
            .line_segment([center, tip], Stroke::new(2.0, ACCENT));
        ui.painter().text(
            rect.center_bottom() - Vec2::new(0.0, 13.0),
            Align2::CENTER_CENTER,
            self.label,
            FontId::monospace(9.0),
            MUTED,
        );
        ui.painter().text(
            rect.center_bottom(),
            Align2::CENTER_BOTTOM,
            format!("{:.2}", *self.value),
            FontId::monospace(9.0),
            TEXT,
        );
        response
    }
}
