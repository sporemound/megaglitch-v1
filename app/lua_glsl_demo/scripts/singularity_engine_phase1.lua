-- SingularityEngine
-- Lua/OpenCV CUDA realtime accretion-glitch compositor.
--
-- This version is not a vortex/whirlpool effect. The image is built from
-- masks, delayed-frame orbital fragments, ring layers, rim corruption, and
-- only mild outer lensing.

local params = {
    input = 0,                  -- 0 for webcam, or a video file path string
    window_name = "SingularityEngine",
    quit_key = string.byte("q"),
    escape_key = 27,
    cuda_device = 0,
    print_every_frames = 30,
    wait_key_ms = 1,

    event_horizon_radius = 0.145,
    photon_ring_radius = 0.172,
    photon_ring_width = 0.018,
    accretion_band_count = 7,
    arc_fragment_count = 8,
    arc_thickness = 0.028,
    arc_length_variation = 0.55,
    arc_rotation_speed = 0.72,
    arc_stutter_amount = 0.34,
    temporal_echo_strength = 0.62,
    frame_buffer_length = 42,
    rgb_split_amount = 6.0,
    rim_corruption_amount = 0.78,
    outer_lensing_strength = 0.055,
    debris_opacity = 0.82,
    void_opacity = 1.0,

    center = { 0.5, 0.5 },
    center_drift = { 0.018, 0.014 },
    time_scale = 1.0,
    remap_interpolation = nil, -- initialized after cv is loaded
    remap_border = nil,        -- initialized after cv is loaded

    bootstrap_paths = {
        enabled = true,
        opencv_bin = [[C:\src\opencv-cuda\build_py312\bin\Release]],
        cuda_bin = [[C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin]],
    },
}

local function prepend_package_cpath(path)
    if not path or path == "" then
        return
    end
    local module_pattern = path .. [[\?.dll]]
    if not package.cpath:lower():find(module_pattern:lower(), 1, true) then
        package.cpath = module_pattern .. ";" .. package.cpath
    end
end

if params.bootstrap_paths.enabled then
    prepend_package_cpath(params.bootstrap_paths.opencv_bin)
    prepend_package_cpath(params.bootstrap_paths.cuda_bin)
end

local cv = require("cv")
require("cv.core")
require("cv.videoio")
require("cv.highgui")
require("cv.imgproc")
require("cv.cuda")

params.remap_interpolation = params.remap_interpolation or cv.INTER_LINEAR
params.remap_border = params.remap_border or cv.BORDER_REFLECT_101

local function now_seconds()
    if cv.getTickCount and cv.getTickFrequency then
        return cv.getTickCount() / cv.getTickFrequency()
    end
    return os.clock()
end

local function clamp(value, lo, hi)
    if value < lo then return lo end
    if value > hi then return hi end
    return value
end

local function smoothstep(edge0, edge1, value)
    local denom = math.max(1.0e-6, edge1 - edge0)
    local t = clamp((value - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
end

local function hash01(a, b, c)
    local v = math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return v - math.floor(v)
end

local function atan2(y, x)
    if math.atan2 then
        return math.atan2(y, x)
    end
    local ok, value = pcall(function() return math.atan(y, x) end)
    if ok and value ~= nil then
        return value
    end
    if x > 0 then
        return math.atan(y / x)
    elseif x < 0 and y >= 0 then
        return math.atan(y / x) + math.pi
    elseif x < 0 and y < 0 then
        return math.atan(y / x) - math.pi
    elseif y > 0 then
        return math.pi * 0.5
    elseif y < 0 then
        return -math.pi * 0.5
    end
    return 0.0
end

local function is_file_input(input)
    return type(input) == "string" and input ~= ""
end

local function open_capture(input)
    if is_file_input(input) then
        return cv.VideoCapture({ input })
    end
    return cv.VideoCapture({ tonumber(input) or 0 })
end

local function read_frame(capture)
    local ok, frame = capture:read()
    if type(ok) == "userdata" then
        frame = ok
        ok = frame and not frame:empty()
    end
    return ok, frame
end

local function mat_width(mat)
    if type(mat.cols) == "function" then return mat:cols() end
    return mat.cols
end

local function mat_height(mat)
    if type(mat.rows) == "function" then return mat:rows() end
    return mat.rows
end

local function make_gpu_mat()
    if cv.cuda_GpuMat then return cv.cuda_GpuMat() end
    return cv.cuda.GpuMat()
end

local function make_mat(height, width, typ)
    local ok, mat = pcall(function()
        return cv.Mat.zeros({ height, width }, typ)
    end)
    if ok then return mat end

    ok, mat = pcall(function()
        return cv.Mat({ height, width }, typ)
    end)
    if ok then return mat end

    return cv.Mat(height, width, typ)
end

local function set_mat_value(mat, y, x, value)
    local attempts = {
        function() mat:set(y, x, value) end,
        function() mat:set({ y, x }, value) end,
        function() mat:set(y + 1, x + 1, value) end,
        function() mat:set({ y + 1, x + 1 }, value) end,
        function() mat:at(y, x)[0] = value end,
        function() mat:at({ y, x })[1] = value end,
        function() mat:at(y + 1, x + 1)[1] = value end,
        function() mat:at({ y + 1, x + 1 })[1] = value end,
    }
    for _, attempt in ipairs(attempts) do
        local ok = pcall(attempt)
        if ok then return end
    end
end

local function upload_mat(mat)
    local gpu = make_gpu_mat()
    local ok = pcall(function() gpu:upload(mat) end)
    if ok then return gpu end
    gpu:upload({ mat })
    return gpu
end

local function upload_into(gpu, mat)
    local ok = pcall(function() gpu:upload(mat) end)
    if ok then return gpu end
    gpu:upload({ mat })
    return gpu
end

local function download_mat(gpu)
    local ok, result = pcall(function() return gpu:download() end)
    if ok then return result end
    return gpu:download({})
end

local function clone_gpu(gpu)
    local ok, out = pcall(function() return gpu:clone() end)
    if ok then return out end
    return upload_mat(download_mat(gpu))
end

local function clone_mat(mat)
    local ok, out = pcall(function() return mat:clone() end)
    if ok then return out end
    return download_mat(upload_mat(mat))
end

local function add_weighted_gpu(a, aw, b, bw, gamma)
    local ok, out = pcall(function()
        return cv.cuda.addWeighted(a, aw, b, bw, gamma or 0.0)
    end)
    if ok then return out end
    return cv.cuda.addWeighted({ src1 = a, alpha = aw, src2 = b, beta = bw, gamma = gamma or 0.0 })
end

local function remap_gpu(src, map_x, map_y, interpolation, border)
    local ok, out = pcall(function()
        return cv.cuda.remap(src, map_x, map_y, interpolation, nil, border)
    end)
    if ok then return out end
    ok, out = pcall(function()
        return cv.cuda.remap(src, map_x, map_y, interpolation)
    end)
    if ok then return out end
    return cv.cuda.remap({ src = src, xmap = map_x, ymap = map_y, interpolation = interpolation, borderMode = border })
end

local function resize_gpu(src, width, height, interpolation)
    local ok, out = pcall(function()
        return cv.cuda.resize(src, { width, height }, nil, 0.0, 0.0, interpolation)
    end)
    if ok then return out end
    ok, out = pcall(function()
        return cv.cuda.resize(src, { width, height }, interpolation)
    end)
    if ok then return out end
    return cv.cuda.resize({ src = src, dsize = { width, height }, interpolation = interpolation })
end

local function copy_to_with_mask(src_gpu, dst_gpu, mask_gpu)
    local ok = pcall(function() src_gpu:copyTo(dst_gpu, mask_gpu) end)
    if ok then return dst_gpu end
    ok = pcall(function() src_gpu:copyTo({ dst_gpu, mask_gpu }) end)
    if ok then return dst_gpu end
    ok = pcall(function() cv.cuda.copyTo(src_gpu, dst_gpu, mask_gpu) end)
    if ok then return dst_gpu end
    return dst_gpu
end

local function make_black_like(frame)
    return make_mat(mat_height(frame), mat_width(frame), frame:type())
end

local function make_bright_like(frame)
    local h, w = mat_height(frame), mat_width(frame)
    local out = make_mat(h, w, frame:type())
    for y = 0, h - 1 do
        for x = 0, w - 1 do
            set_mat_value(out, y, x, { 255, 255, 255 })
        end
    end
    return out
end

local SingularityEngine = {
    params = params,
    runtime = {
        frame_index = 0,
        fps = 0.0,
        last_report_time = now_seconds(),
        start_time = now_seconds(),
    },
    frame_buffer = {},
    gpu = {
        input = nil,
        black = nil,
    },
    cache = {
        width = 0,
        height = 0,
        masks = {},
        maps = {},
        arcs = {},
    },
}

function SingularityEngine:update_black_hole(frame)
    local t = (now_seconds() - self.runtime.start_time) * self.params.time_scale
    local width, height = mat_width(frame), mat_height(frame)
    local scale = math.min(width, height)
    local cx = (self.params.center[1] + self.params.center_drift[1] * math.sin(t * 0.31)) * (width - 1)
    local cy = (self.params.center[2] + self.params.center_drift[2] * math.cos(t * 0.27)) * (height - 1)

    self.black_hole = {
        time = t,
        cx = cx,
        cy = cy,
        scale = scale,
        event_radius = self.params.event_horizon_radius * scale,
        photon_radius = self.params.photon_ring_radius * scale,
        photon_width = math.max(1.0, self.params.photon_ring_width * scale),
        rim_width = math.max(2.0, self.params.photon_ring_width * scale * 1.6),
    }
    return self.black_hole
end

function SingularityEngine:generate_masks(frame, bh)
    local width, height = mat_width(frame), mat_height(frame)
    local masks = {
        event = make_mat(height, width, cv.CV_8UC1),
        photon = make_mat(height, width, cv.CV_8UC1),
        rim = make_mat(height, width, cv.CV_8UC1),
        collapse = make_mat(height, width, cv.CV_8UC1),
        disk = make_mat(height, width, cv.CV_8UC1),
    }

    local event_outer = bh.event_radius + bh.rim_width
    local photon_inner = bh.photon_radius - bh.photon_width * 0.5
    local photon_outer = bh.photon_radius + bh.photon_width * 0.5
    local disk_inner = bh.photon_radius + bh.photon_width * 0.8
    local disk_outer = bh.scale * 0.46

    for y = 0, height - 1 do
        for x = 0, width - 1 do
            local dx, dy = x - bh.cx, y - bh.cy
            local r = math.sqrt(dx * dx + dy * dy)
            local angle = atan2(dy, dx)
            local noise = hash01(math.floor(angle * 19.0), math.floor(r * 0.12), math.floor(bh.time * 18.0))
            local rim_jitter = (noise - 0.5) * bh.rim_width * 1.2
            local unstable_event = bh.event_radius + rim_jitter

            if r <= unstable_event then
                set_mat_value(masks.event, y, x, math.floor(255 * self.params.void_opacity))
            end
            if r > bh.event_radius - bh.rim_width and r < event_outer then
                local rim_alpha = 1.0 - math.abs(r - bh.event_radius) / bh.rim_width
                set_mat_value(masks.rim, y, x, math.floor(255 * clamp(rim_alpha + noise * 0.35, 0.0, 1.0)))
            end
            if r >= photon_inner and r <= photon_outer then
                local broken = hash01(math.floor(angle * 48.0), 3, math.floor(bh.time * 9.0))
                if broken > 0.26 then
                    local a = 1.0 - math.abs(r - bh.photon_radius) / (bh.photon_width * 0.5)
                    set_mat_value(masks.photon, y, x, math.floor(255 * clamp(a * (0.72 + broken * 0.6), 0.0, 1.0)))
                end
            end
            if r >= disk_inner and r <= disk_outer then
                set_mat_value(masks.disk, y, x, 255)
            end
            if r >= bh.event_radius and r <= bh.event_radius + bh.rim_width * 1.8 then
                set_mat_value(masks.collapse, y, x, 255)
            end
        end
    end

    self.cache.masks = masks
    self.gpu.mask_event = upload_mat(masks.event)
    self.gpu.mask_photon = upload_mat(masks.photon)
    self.gpu.mask_rim = upload_mat(masks.rim)
    self.gpu.mask_collapse = upload_mat(masks.collapse)
    self.gpu.mask_disk = upload_mat(masks.disk)
    return masks
end

function SingularityEngine:generate_orbital_arcs(frame, bh)
    local width, height = mat_width(frame), mat_height(frame)
    local arcs = {}
    local band_count = math.max(1, math.floor(self.params.accretion_band_count))
    local fragments = math.max(1, math.floor(self.params.arc_fragment_count))
    local base_inner = bh.photon_radius + bh.photon_width * 1.4
    local base_outer = bh.scale * 0.47
    local span = math.max(1.0, base_outer - base_inner)

    for i = 1, band_count do
        local frac = (i - 0.5) / band_count
        local radius = base_inner + span * frac
        local thickness = math.max(1.0, self.params.arc_thickness * bh.scale * (0.75 + 0.4 * hash01(i, 1, 0)))
        local direction = (i % 2 == 0) and -1.0 or 1.0
        local speed = self.params.arc_rotation_speed * direction * (0.45 + 1.3 * hash01(i, 2, 0))
        local stutter = math.floor((bh.time * (2.0 + i * 0.37)) % 2.0) * self.params.arc_stutter_amount
        local phase = bh.time * speed * (1.0 - stutter) + i * 1.771
        local delay = 1 + math.floor((i / band_count) * math.max(1, #self.frame_buffer - 1))
        local source_offset = {
            math.sin(i * 2.1 + bh.time * 0.17) * 18.0,
            math.cos(i * 1.7 - bh.time * 0.13) * 14.0,
        }
        local rgb_offset = self.params.rgb_split_amount * (0.35 + hash01(i, 7, 0))

        local mask = make_mat(height, width, cv.CV_8UC1)
        local map_x = make_mat(height, width, cv.CV_32FC1)
        local map_y = make_mat(height, width, cv.CV_32FC1)

        for y = 0, height - 1 do
            for x = 0, width - 1 do
                local dx, dy = x - bh.cx, y - bh.cy
                local r = math.sqrt(dx * dx + dy * dy)
                local radial = 1.0 - smoothstep(thickness * 0.35, thickness, math.abs(r - radius))
                local angle = atan2(dy, dx)
                local local_angle = (angle + phase + math.pi * 4.0) % (math.pi * 2.0)
                local sector = math.floor(local_angle / (math.pi * 2.0 / fragments))
                local slot = (local_angle / (math.pi * 2.0 / fragments)) - sector
                local gate = hash01(i, sector, math.floor(bh.time * (1.0 + i * 0.13)))
                local arc_len = 0.22 + (1.0 - self.params.arc_length_variation) * 0.55 + gate * self.params.arc_length_variation * 0.55
                local dropout = hash01(sector, i * 3, math.floor(bh.time * 6.0))

                if radial > 0.02 and slot < arc_len and dropout > 0.22 then
                    local alpha = clamp(radial * self.params.debris_opacity * (0.55 + gate * 0.55), 0.0, 1.0)
                    set_mat_value(mask, y, x, math.floor(alpha * 255))
                end

                local sample_angle = angle - direction * 0.12 * (1.0 + frac) + math.sin(bh.time + i) * 0.03
                local sx = bh.cx + math.cos(sample_angle) * r + source_offset[1]
                local sy = bh.cy + math.sin(sample_angle) * r + source_offset[2]
                set_mat_value(map_x, y, x, sx)
                set_mat_value(map_y, y, x, sy)
            end
        end

        arcs[i] = {
            mask = mask,
            map_x = map_x,
            map_y = map_y,
            gpu_mask = upload_mat(mask),
            gpu_map_x = upload_mat(map_x),
            gpu_map_y = upload_mat(map_y),
            delay = delay,
            rgb_offset = rgb_offset,
            opacity = self.params.debris_opacity,
        }
    end

    self.cache.arcs = arcs
    return arcs
end

function SingularityEngine:apply_outer_lensing(src_gpu, frame, bh)
    local width, height = mat_width(frame), mat_height(frame)
    local map_x = make_mat(height, width, cv.CV_32FC1)
    local map_y = make_mat(height, width, cv.CV_32FC1)
    local outer_start = bh.photon_radius + bh.photon_width * 2.0
    local outer_end = bh.scale * 0.58

    for y = 0, height - 1 do
        for x = 0, width - 1 do
            local dx, dy = x - bh.cx, y - bh.cy
            local r = math.sqrt(dx * dx + dy * dy)
            local influence = (1.0 - smoothstep(outer_start, outer_end, r)) * smoothstep(bh.event_radius, outer_start, r)
            local pull = self.params.outer_lensing_strength * influence * bh.scale
            local inv_r = 1.0 / math.max(r, 1.0)
            set_mat_value(map_x, y, x, x - dx * inv_r * pull)
            set_mat_value(map_y, y, x, y - dy * inv_r * pull)
        end
    end

    return remap_gpu(src_gpu, upload_mat(map_x), upload_mat(map_y), self.params.remap_interpolation, self.params.remap_border)
end

function SingularityEngine:apply_event_horizon(base_gpu, frame, bh)
    if not self.gpu.black then
        self.gpu.black = upload_mat(make_black_like(frame))
    end
    local out = clone_gpu(base_gpu)
    copy_to_with_mask(self.gpu.black, out, self.gpu.mask_event)

    if self.params.temporal_echo_strength > 0.0 and #self.frame_buffer > 2 then
        local ghost = self.frame_buffer[math.max(1, #self.frame_buffer - math.floor(#self.frame_buffer * 0.65))]
        local ghost_gpu = upload_mat(ghost)
        local ghost_mix = add_weighted_gpu(out, 1.0 - self.params.temporal_echo_strength * 0.18, ghost_gpu, self.params.temporal_echo_strength * 0.18, 0.0)
        copy_to_with_mask(ghost_mix, out, self.gpu.mask_rim)
    end
    return out
end

function SingularityEngine:apply_photon_ring(base_gpu, frame, bh)
    local delay_index = math.max(1, #self.frame_buffer - math.floor(#self.frame_buffer * 0.25))
    local source = self.frame_buffer[delay_index] or frame
    local source_gpu = upload_mat(source)
    local bright = add_weighted_gpu(source_gpu, 1.45, source_gpu, 0.0, 48.0)

    local shifted_r = remap_gpu(bright, self.cache.maps.rgb_r_x or self.gpu.identity_map_x, self.cache.maps.rgb_r_y or self.gpu.identity_map_y, self.params.remap_interpolation, self.params.remap_border)
    local ring = add_weighted_gpu(bright, 0.55, shifted_r, 0.65, 18.0)
    local out = clone_gpu(base_gpu)
    copy_to_with_mask(ring, out, self.gpu.mask_photon)
    return out
end

function SingularityEngine:apply_accretion_bands(base_gpu, frame, bh)
    local out = clone_gpu(base_gpu)
    for _, arc in ipairs(self.cache.arcs or {}) do
        local src_frame = self.frame_buffer[math.max(1, #self.frame_buffer - arc.delay)] or frame
        local src_gpu = upload_mat(src_frame)
        local debris = remap_gpu(src_gpu, arc.gpu_map_x, arc.gpu_map_y, self.params.remap_interpolation, self.params.remap_border)
        local mixed = add_weighted_gpu(out, 1.0 - arc.opacity * 0.28, debris, arc.opacity, 0.0)
        copy_to_with_mask(mixed, out, arc.gpu_mask)
    end
    return out
end

function SingularityEngine:apply_temporal_debris(base_gpu, frame, bh)
    if #self.frame_buffer < 3 then
        return base_gpu
    end
    local out = clone_gpu(base_gpu)
    local echo_count = math.min(4, #self.frame_buffer - 1)
    for i = 1, echo_count do
        local idx = math.max(1, #self.frame_buffer - i * math.floor(#self.frame_buffer / (echo_count + 1)))
        local echo_gpu = upload_mat(self.frame_buffer[idx])
        local opacity = self.params.temporal_echo_strength * (0.10 / i)
        local mixed = add_weighted_gpu(out, 1.0 - opacity, echo_gpu, opacity, 0.0)
        copy_to_with_mask(mixed, out, self.gpu.mask_disk)
    end
    return out
end

function SingularityEngine:apply_data_collapse(base_gpu, frame, bh)
    local width, height = mat_width(frame), mat_height(frame)
    local small_w = math.max(8, math.floor(width * (1.0 - self.params.rim_corruption_amount * 0.55)))
    local small_h = math.max(8, math.floor(height * (1.0 - self.params.rim_corruption_amount * 0.55)))

    local ok, crushed = pcall(function()
        local small = resize_gpu(base_gpu, small_w, small_h, cv.INTER_NEAREST)
        return resize_gpu(small, width, height, cv.INTER_NEAREST)
    end)
    if not ok then
        return base_gpu
    end

    local out = clone_gpu(base_gpu)
    local mixed = add_weighted_gpu(out, 1.0 - self.params.rim_corruption_amount * 0.45, crushed, self.params.rim_corruption_amount * 0.75, -18.0)
    copy_to_with_mask(mixed, out, self.gpu.mask_collapse)
    return out
end

function SingularityEngine:composite_layers(frame)
    upload_into(self.gpu.input, frame)
    local bh = self:update_black_hole(frame)
    self:generate_masks(frame, bh)
    self:generate_rgb_split_maps(frame, bh)
    self:generate_orbital_arcs(frame, bh)

    local out = self:apply_outer_lensing(self.gpu.input, frame, bh)
    out = self:apply_temporal_debris(out, frame, bh)
    out = self:apply_accretion_bands(out, frame, bh)
    out = self:apply_photon_ring(out, frame, bh)
    out = self:apply_data_collapse(out, frame, bh)
    out = self:apply_event_horizon(out, frame, bh)
    return out
end

function SingularityEngine:generate_rgb_split_maps(frame, bh)
    local width, height = mat_width(frame), mat_height(frame)
    local map_rx = make_mat(height, width, cv.CV_32FC1)
    local map_ry = make_mat(height, width, cv.CV_32FC1)
    local amount = self.params.rgb_split_amount
    for y = 0, height - 1 do
        for x = 0, width - 1 do
            local dx, dy = x - bh.cx, y - bh.cy
            local r = math.sqrt(dx * dx + dy * dy)
            local inv_r = 1.0 / math.max(r, 1.0)
            set_mat_value(map_rx, y, x, x + dx * inv_r * amount)
            set_mat_value(map_ry, y, x, y + dy * inv_r * amount)
        end
    end
    self.cache.maps.rgb_r_x = upload_mat(map_rx)
    self.cache.maps.rgb_r_y = upload_mat(map_ry)
end

function SingularityEngine:push_frame_buffer(frame)
    table.insert(self.frame_buffer, clone_mat(frame))
    while #self.frame_buffer > math.max(1, math.floor(self.params.frame_buffer_length)) do
        table.remove(self.frame_buffer, 1)
    end
end

function SingularityEngine:process_frame(frame)
    self:push_frame_buffer(frame)
    local out_gpu = self:composite_layers(frame)
    return download_mat(out_gpu)
end

function SingularityEngine:init()
    local device_count = cv.cuda.getCudaEnabledDeviceCount()
    assert(device_count > 0, "OpenCV CUDA reports no CUDA-enabled devices")
    cv.cuda.setDevice(self.params.cuda_device)
    self.gpu.input = make_gpu_mat()
    self.capture = open_capture(self.params.input)
    assert(self.capture:isOpened(), "Could not open video input: " .. tostring(self.params.input))
    cv.namedWindow(self.params.window_name, cv.WINDOW_NORMAL)
end

function SingularityEngine:handle_display_and_fps(output)
    cv.imshow(self.params.window_name, output)
    self.runtime.frame_index = self.runtime.frame_index + 1
    if self.runtime.frame_index % self.params.print_every_frames == 0 then
        local now = now_seconds()
        local elapsed = now - self.runtime.last_report_time
        if elapsed > 0 then
            self.runtime.fps = self.params.print_every_frames / elapsed
            print(string.format("[SingularityEngine] %.2f FPS", self.runtime.fps))
        end
        self.runtime.last_report_time = now
    end
    return cv.waitKey(self.params.wait_key_ms)
end

function SingularityEngine:run()
    self:init()
    while true do
        local ok, frame = read_frame(self.capture)
        if not ok or not frame or frame:empty() then
            if is_file_input(self.params.input) then
                self.capture:set(cv.CAP_PROP_POS_FRAMES, 0)
            else
                break
            end
        else
            local output = self:process_frame(frame)
            local key = self:handle_display_and_fps(output)
            if key == self.params.quit_key or key == self.params.escape_key then
                break
            end
        end
    end
    if self.capture then
        self.capture:release()
    end
    cv.destroyWindow(self.params.window_name)
end

SingularityEngine:run()
