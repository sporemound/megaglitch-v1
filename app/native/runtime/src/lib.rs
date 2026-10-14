use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub const MG_OK: i32 = 0;
pub const MG_EMPTY: i32 = 1;
pub const MG_QUEUE_FULL: i32 = 2;
pub const MG_INVALID_ARGUMENT: i32 = -1;
pub const MG_INVALID_STATE: i32 = -2;
pub const MG_BUFFER_TOO_SMALL: i32 = -3;
pub const MG_INTERNAL_ERROR: i32 = -4;

pub const MODE_PASSTHROUGH: u32 = 0;
pub const MODE_TEMPORAL_MIX: u32 = 1;
pub const MODE_INVERT: u32 = 2;
pub const GRAPH_VERSION_V1: u32 = 1;
pub const OP_INPUT: u32 = 1;
pub const OP_INVERT: u32 = 2;
pub const OP_BRIGHTNESS_CONTRAST: u32 = 3;
pub const OP_TEMPORAL_MIX: u32 = 4;
pub const OP_BLEND: u32 = 5;
pub const OP_OUTPUT: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RuntimeGraphOp {
    pub opcode: u32,
    pub flags: u32,
    pub arg0: f32,
    pub arg1: f32,
    pub arg2: u32,
    pub arg3: u32,
}

#[derive(Clone)]
struct GraphProgram {
    operations: Vec<RuntimeGraphOp>,
}

impl GraphProgram {
    fn validate(
        version: u32,
        operations: &[RuntimeGraphOp],
        history_capacity: usize,
    ) -> Result<Self, String> {
        if version != GRAPH_VERSION_V1 {
            return Err(format!("unsupported graph IR version {version}"));
        }
        if operations.len() < 2 || operations.len() > 64 {
            return Err("graph must contain between 2 and 64 operations".into());
        }
        if operations.first().map(|op| op.opcode) != Some(OP_INPUT)
            || operations.last().map(|op| op.opcode) != Some(OP_OUTPUT)
        {
            return Err("graph must begin with Input and end with Output".into());
        }
        for (index, operation) in operations.iter().enumerate() {
            match operation.opcode {
                OP_INPUT if index == 0 && operation.arg2 == 0 => {}
                OP_INPUT => return Err("graph v1 supports one primary Input at operation 0".into()),
                OP_OUTPUT if index + 1 == operations.len() => {}
                OP_OUTPUT => return Err("Output must be the final operation".into()),
                OP_INVERT => {}
                OP_BRIGHTNESS_CONTRAST => {
                    if !operation.arg0.is_finite() || !operation.arg1.is_finite() {
                        return Err("brightness/contrast parameters must be finite".into());
                    }
                }
                OP_TEMPORAL_MIX | OP_BLEND => {
                    if !operation.arg0.is_finite() || !(0.0..=1.0).contains(&operation.arg0) {
                        return Err("temporal/blend mix must be between 0 and 1".into());
                    }
                    if operation.arg2 as usize >= history_capacity {
                        return Err(
                            "temporal/blend delay must be smaller than history capacity".into()
                        );
                    }
                }
                opcode => return Err(format!("unsupported graph opcode {opcode}")),
            }
        }
        Ok(Self {
            operations: operations.to_vec(),
        })
    }

    fn passthrough() -> Self {
        Self {
            operations: vec![
                RuntimeGraphOp {
                    opcode: OP_INPUT,
                    flags: 0,
                    arg0: 0.0,
                    arg1: 0.0,
                    arg2: 0,
                    arg3: 0,
                },
                RuntimeGraphOp {
                    opcode: OP_OUTPUT,
                    flags: 0,
                    arg0: 0.0,
                    arg1: 0.0,
                    arg2: 0,
                    arg3: 0,
                },
            ],
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineState {
    Initializing = 1,
    WarmingUp = 2,
    Ready = 3,
    Running = 4,
    Stopping = 5,
    Stopped = 6,
    Failed = 7,
}

#[repr(C)]
#[derive(Default)]
pub struct RuntimeStats {
    pub submitted: u64,
    pub processed: u64,
    pub delivered: u64,
    pub input_dropped: u64,
    pub output_dropped: u64,
    pub commands_applied: u64,
    pub frames_in_history: u64,
    pub allocated_bytes: u64,
    pub sequence: u64,
    pub state: u32,
}

#[derive(Default)]
struct AtomicStats {
    submitted: AtomicU64,
    processed: AtomicU64,
    delivered: AtomicU64,
    input_dropped: AtomicU64,
    output_dropped: AtomicU64,
    commands_applied: AtomicU64,
    frames_in_history: AtomicU64,
    sequence: AtomicU64,
}

#[derive(Clone, Copy)]
struct Parameters {
    delay: usize,
    mix: f32,
    frozen: bool,
}

enum Command {
    Parameters(Parameters),
    Graph(Arc<GraphProgram>),
    Clear,
    Stop,
}

struct PendingWorker {
    input_rx: Receiver<usize>,
    output_tx: SyncSender<usize>,
    free_tx: SyncSender<usize>,
    command_rx: Receiver<Command>,
}

pub struct Engine {
    frame_bytes: usize,
    history_capacity: usize,
    queue_capacity: usize,
    mode: u32,
    state: Arc<AtomicU8>,
    stopping: Arc<AtomicBool>,
    stats: Arc<AtomicStats>,
    buffers: Arc<Vec<Mutex<Vec<u8>>>>,
    buffer_sequences: Arc<Vec<AtomicU64>>,
    input_tx: SyncSender<usize>,
    output_rx: Receiver<usize>,
    free_rx: Receiver<usize>,
    recycle_tx: SyncSender<usize>,
    command_tx: SyncSender<Command>,
    pending_worker: Option<PendingWorker>,
    worker: Option<JoinHandle<()>>,
    parameters: Parameters,
    graph: Arc<GraphProgram>,
}

impl Engine {
    fn new(
        width: usize,
        height: usize,
        channels: usize,
        history_capacity: usize,
        queue_capacity: usize,
        mode: u32,
    ) -> Result<Self, String> {
        if width == 0
            || height == 0
            || channels == 0
            || history_capacity == 0
            || queue_capacity == 0
        {
            return Err("dimensions, history capacity, and queue capacity must be positive".into());
        }
        if mode != MODE_PASSTHROUGH && mode != MODE_TEMPORAL_MIX && mode != MODE_INVERT {
            return Err(format!("unsupported native graph mode {mode}"));
        }
        let graph = match mode {
            MODE_PASSTHROUGH => GraphProgram::passthrough(),
            MODE_INVERT => GraphProgram {
                operations: vec![
                    RuntimeGraphOp {
                        opcode: OP_INPUT,
                        flags: 0,
                        arg0: 0.0,
                        arg1: 0.0,
                        arg2: 0,
                        arg3: 0,
                    },
                    RuntimeGraphOp {
                        opcode: OP_INVERT,
                        flags: 0,
                        arg0: 0.0,
                        arg1: 0.0,
                        arg2: 0,
                        arg3: 0,
                    },
                    RuntimeGraphOp {
                        opcode: OP_OUTPUT,
                        flags: 0,
                        arg0: 0.0,
                        arg1: 0.0,
                        arg2: 0,
                        arg3: 0,
                    },
                ],
            },
            MODE_TEMPORAL_MIX => GraphProgram {
                operations: vec![
                    RuntimeGraphOp {
                        opcode: OP_INPUT,
                        flags: 0,
                        arg0: 0.0,
                        arg1: 0.0,
                        arg2: 0,
                        arg3: 0,
                    },
                    RuntimeGraphOp {
                        opcode: OP_TEMPORAL_MIX,
                        flags: 2,
                        arg0: 0.5,
                        arg1: 0.0,
                        arg2: 0,
                        arg3: 0,
                    },
                    RuntimeGraphOp {
                        opcode: OP_OUTPUT,
                        flags: 0,
                        arg0: 0.0,
                        arg1: 0.0,
                        arg2: 0,
                        arg3: 0,
                    },
                ],
            },
            _ => unreachable!(),
        };
        Self::new_with_graph(
            width,
            height,
            channels,
            history_capacity,
            queue_capacity,
            mode,
            graph,
        )
    }

    fn new_with_graph(
        width: usize,
        height: usize,
        channels: usize,
        history_capacity: usize,
        queue_capacity: usize,
        mode: u32,
        graph: GraphProgram,
    ) -> Result<Self, String> {
        if width == 0
            || height == 0
            || channels == 0
            || history_capacity == 0
            || queue_capacity == 0
        {
            return Err("dimensions, history capacity, and queue capacity must be positive".into());
        }
        let frame_bytes = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(channels))
            .ok_or_else(|| "frame dimensions overflow addressable memory".to_string())?;
        let _history_bytes = frame_bytes
            .checked_mul(history_capacity)
            .ok_or_else(|| "history allocation overflows addressable memory".to_string())?;
        let _queue_bytes = frame_bytes
            .checked_mul(queue_capacity)
            .ok_or_else(|| "queue allocation overflows addressable memory".to_string())?;

        let state = Arc::new(AtomicU8::new(EngineState::Initializing as u8));
        let buffers = Arc::new(
            (0..queue_capacity)
                .map(|_| Mutex::new(vec![0; frame_bytes]))
                .collect::<Vec<_>>(),
        );
        let buffer_sequences = Arc::new(
            (0..queue_capacity)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>(),
        );
        let (input_tx, input_rx) = mpsc::sync_channel(queue_capacity);
        let (output_tx, output_rx) = mpsc::sync_channel(queue_capacity);
        let (free_tx, free_rx) = mpsc::sync_channel(queue_capacity);
        for index in 0..queue_capacity {
            free_tx
                .send(index)
                .map_err(|_| "failed to initialize frame pool".to_string())?;
        }
        let (command_tx, command_rx) = mpsc::sync_channel(32);
        state.store(EngineState::WarmingUp as u8, Ordering::Release);
        // Allocation itself is the warm-up for this dependency-free vertical slice.
        // Synthetic content is never inserted into temporal state.
        state.store(EngineState::Ready as u8, Ordering::Release);
        Ok(Self {
            frame_bytes,
            history_capacity,
            queue_capacity,
            mode,
            state,
            stopping: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(AtomicStats::default()),
            buffers,
            buffer_sequences,
            input_tx,
            output_rx,
            free_rx,
            recycle_tx: free_tx.clone(),
            command_tx,
            pending_worker: Some(PendingWorker {
                input_rx,
                output_tx,
                free_tx,
                command_rx,
            }),
            worker: None,
            parameters: Parameters {
                delay: 0,
                mix: 0.5,
                frozen: false,
            },
            graph: Arc::new(graph),
        })
    }

    fn state(&self) -> EngineState {
        match self.state.load(Ordering::Acquire) {
            1 => EngineState::Initializing,
            2 => EngineState::WarmingUp,
            3 => EngineState::Ready,
            4 => EngineState::Running,
            5 => EngineState::Stopping,
            6 => EngineState::Stopped,
            _ => EngineState::Failed,
        }
    }

    fn start(&mut self) -> Result<(), String> {
        if self.state() != EngineState::Ready {
            return Err(format!("engine cannot start from {:?}", self.state()));
        }
        let channels = self
            .pending_worker
            .take()
            .ok_or_else(|| "worker channels unavailable".to_string())?;
        let state = Arc::clone(&self.state);
        let stopping = Arc::clone(&self.stopping);
        let stats = Arc::clone(&self.stats);
        let buffers = Arc::clone(&self.buffers);
        let buffer_sequences = Arc::clone(&self.buffer_sequences);
        let frame_bytes = self.frame_bytes;
        let history_capacity = self.history_capacity;
        let mode = self.mode;
        let initial_parameters = self.parameters;
        let initial_graph = Arc::clone(&self.graph);
        self.stopping.store(false, Ordering::Release);
        self.state
            .store(EngineState::Running as u8, Ordering::Release);
        self.worker = Some(thread::spawn(move || {
            let worker_result = catch_unwind(AssertUnwindSafe(|| {
                worker_loop(
                    channels,
                    buffers,
                    buffer_sequences,
                    frame_bytes,
                    history_capacity,
                    mode,
                    initial_parameters,
                    initial_graph,
                    Arc::clone(&stopping),
                    Arc::clone(&stats),
                )
            }));
            if worker_result.is_err() {
                state.store(EngineState::Failed as u8, Ordering::Release);
            } else if state.load(Ordering::Acquire) != EngineState::Failed as u8 {
                state.store(EngineState::Stopped as u8, Ordering::Release);
            }
        }));
        Ok(())
    }

    fn submit(&mut self, bytes: &[u8]) -> Result<bool, String> {
        if self.state() != EngineState::Running {
            return Err("engine is not running".into());
        }
        if bytes.len() != self.frame_bytes {
            return Err(format!(
                "expected {} frame bytes, got {}",
                self.frame_bytes,
                bytes.len()
            ));
        }
        let index = match self.free_rx.try_recv() {
            Ok(index) => index,
            Err(TryRecvError::Empty) => {
                self.stats.input_dropped.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }
            Err(TryRecvError::Disconnected) => return Err("frame pool disconnected".into()),
        };
        {
            let mut buffer = self.buffers[index]
                .lock()
                .map_err(|_| "frame buffer lock poisoned".to_string())?;
            buffer.copy_from_slice(bytes);
        }
        match self.input_tx.try_send(index) {
            Ok(()) => {
                self.stats.submitted.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(TrySendError::Full(index)) => {
                self.stats.input_dropped.fetch_add(1, Ordering::Relaxed);
                let _ = self.recycle_tx.try_send(index);
                Ok(false)
            }
            Err(TrySendError::Disconnected(_)) => Err("input queue disconnected".into()),
        }
    }

    fn poll(&mut self, output: &mut [u8]) -> Result<Option<u64>, String> {
        if output.len() < self.frame_bytes {
            return Err(format!(
                "output requires at least {} bytes",
                self.frame_bytes
            ));
        }
        let index = match self.output_rx.try_recv() {
            Ok(index) => index,
            Err(TryRecvError::Empty) => return Ok(None),
            Err(TryRecvError::Disconnected) => return Err("output queue disconnected".into()),
        };
        {
            let buffer = self.buffers[index]
                .lock()
                .map_err(|_| "frame buffer lock poisoned".to_string())?;
            output[..self.frame_bytes].copy_from_slice(&buffer);
        }
        self.stats.delivered.fetch_add(1, Ordering::Relaxed);
        let sequence = self.buffer_sequences[index].load(Ordering::Acquire);
        self.recycle_slot(index)?;
        Ok(Some(sequence))
    }

    fn wait(&mut self, output: &mut [u8], timeout_ms: u64) -> Result<Option<u64>, String> {
        if output.len() < self.frame_bytes {
            return Err(format!(
                "output requires at least {} bytes",
                self.frame_bytes
            ));
        }
        let index = match self
            .output_rx
            .recv_timeout(Duration::from_millis(timeout_ms))
        {
            Ok(index) => index,
            Err(mpsc::RecvTimeoutError::Timeout) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("output queue disconnected".into());
            }
        };
        {
            let buffer = self.buffers[index]
                .lock()
                .map_err(|_| "frame buffer lock poisoned".to_string())?;
            output[..self.frame_bytes].copy_from_slice(&buffer);
        }
        self.stats.delivered.fetch_add(1, Ordering::Relaxed);
        let sequence = self.buffer_sequences[index].load(Ordering::Acquire);
        self.recycle_slot(index)?;
        Ok(Some(sequence))
    }

    fn recycle_slot(&self, index: usize) -> Result<(), String> {
        self.recycle_tx
            .try_send(index)
            .map_err(|_| "failed to recycle output slot".to_string())
    }

    fn update_parameters(&mut self, delay: usize, mix: f32, frozen: bool) -> Result<(), String> {
        if !(0.0..=1.0).contains(&mix) {
            return Err("mix must be between 0 and 1".into());
        }
        if delay >= self.history_capacity {
            return Err("delay must be smaller than history capacity".into());
        }
        self.parameters = Parameters { delay, mix, frozen };
        if self.state() == EngineState::Running {
            self.command_tx
                .try_send(Command::Parameters(self.parameters))
                .map_err(|_| "command queue is full or disconnected".to_string())?;
        }
        Ok(())
    }

    fn update_graph(&mut self, graph: GraphProgram) -> Result<(), String> {
        if self.state() != EngineState::Running {
            return Err("engine is not running".into());
        }
        let graph = Arc::new(graph);
        self.command_tx
            .try_send(Command::Graph(Arc::clone(&graph)))
            .map_err(|_| "command queue is full or disconnected".to_string())?;
        self.graph = graph;
        Ok(())
    }

    fn clear(&self) -> Result<(), String> {
        if self.state() != EngineState::Running {
            return Err("engine is not running".into());
        }
        self.command_tx
            .try_send(Command::Clear)
            .map_err(|_| "command queue is full or disconnected".to_string())
    }

    fn shutdown(&mut self) {
        match self.state() {
            EngineState::Stopped | EngineState::Failed => return,
            EngineState::Ready => {
                self.state
                    .store(EngineState::Stopped as u8, Ordering::Release);
                return;
            }
            _ => {}
        }
        self.state
            .store(EngineState::Stopping as u8, Ordering::Release);
        self.stopping.store(true, Ordering::Release);
        let _ = self.command_tx.try_send(Command::Stop);
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                self.state
                    .store(EngineState::Failed as u8, Ordering::Release);
                return;
            }
        }
        self.state
            .store(EngineState::Stopped as u8, Ordering::Release);
    }
}

fn worker_loop(
    channels: PendingWorker,
    buffers: Arc<Vec<Mutex<Vec<u8>>>>,
    buffer_sequences: Arc<Vec<AtomicU64>>,
    frame_bytes: usize,
    history_capacity: usize,
    _mode: u32,
    mut parameters: Parameters,
    mut graph: Arc<GraphProgram>,
    stopping: Arc<AtomicBool>,
    stats: Arc<AtomicStats>,
) {
    let mut history = vec![0_u8; frame_bytes * history_capacity];
    let mut history_count = 0_usize;
    let mut write_index = 0_usize;
    while !stopping.load(Ordering::Acquire) {
        loop {
            match channels.command_rx.try_recv() {
                Ok(Command::Parameters(snapshot)) => {
                    parameters = snapshot;
                    stats.commands_applied.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Command::Graph(snapshot)) => {
                    graph = snapshot;
                    stats.commands_applied.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Command::Clear) => {
                    history.fill(0);
                    history_count = 0;
                    write_index = 0;
                    stats.frames_in_history.store(0, Ordering::Release);
                    stats.commands_applied.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Command::Stop) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        let index = match channels.input_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(index) => index,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        };
        // Commands sent while the worker was blocked waiting for input must be
        // visible to this frame, not one frame later.
        loop {
            match channels.command_rx.try_recv() {
                Ok(Command::Parameters(snapshot)) => {
                    parameters = snapshot;
                    stats.commands_applied.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Command::Graph(snapshot)) => {
                    graph = snapshot;
                    stats.commands_applied.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Command::Clear) => {
                    history.fill(0);
                    history_count = 0;
                    write_index = 0;
                    stats.frames_in_history.store(0, Ordering::Release);
                    stats.commands_applied.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Command::Stop) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        let mut buffer = match buffers[index].lock() {
            Ok(buffer) => buffer,
            Err(_) => return,
        };
        let temporal_operations = graph.operations.iter().filter(|operation| {
            operation.opcode == OP_TEMPORAL_MIX || operation.opcode == OP_BLEND
        });
        let mut has_temporal = false;
        let mut history_frozen = false;
        for operation in temporal_operations {
            has_temporal = true;
            history_frozen |= if operation.flags & 2 != 0 {
                parameters.frozen
            } else {
                operation.flags & 1 != 0
            };
        }
        if has_temporal && !history_frozen {
            let offset = write_index * frame_bytes;
            history[offset..offset + frame_bytes].copy_from_slice(&buffer);
            write_index = (write_index + 1) % history_capacity;
            history_count = history_count.saturating_add(1).min(history_capacity);
            stats
                .frames_in_history
                .store(history_count as u64, Ordering::Release);
        }
        for operation in &graph.operations {
            match operation.opcode {
                OP_INPUT | OP_OUTPUT => {}
                OP_INVERT => {
                    for output in buffer.iter_mut() {
                        *output = 255_u8 - *output;
                    }
                }
                OP_BRIGHTNESS_CONTRAST => {
                    let alpha = 1.0_f32 + operation.arg1 / 100.0_f32;
                    let beta = operation.arg0;
                    for output in buffer.iter_mut() {
                        *output = ((*output as f32 * alpha + beta).clamp(0.0, 255.0)) as u8;
                    }
                }
                OP_TEMPORAL_MIX | OP_BLEND if history_count > 0 => {
                    let (delay, mix) = if operation.flags & 2 != 0 {
                        (parameters.delay, parameters.mix)
                    } else {
                        (operation.arg2 as usize, operation.arg0)
                    };
                    let delay = delay.min(history_count - 1);
                    let slot = (write_index + history_capacity - 1 - delay) % history_capacity;
                    let historical = &history[slot * frame_bytes..(slot + 1) * frame_bytes];
                    let alpha = (mix * 65_536.0 + 0.5) as u32;
                    let inverse = 65_536_u32 - alpha;
                    for (output, old) in buffer.iter_mut().zip(historical.iter()) {
                        *output = (((*output as u32 * inverse + *old as u32 * alpha) + 32_768)
                            >> 16) as u8;
                    }
                }
                OP_TEMPORAL_MIX | OP_BLEND => {}
                _ => return,
            }
        }
        drop(buffer);
        stats.processed.fetch_add(1, Ordering::Relaxed);
        let sequence = stats.sequence.fetch_add(1, Ordering::AcqRel) + 1;
        buffer_sequences[index].store(sequence, Ordering::Release);
        match channels.output_tx.try_send(index) {
            Ok(()) => {}
            Err(TrySendError::Full(index)) | Err(TrySendError::Disconnected(index)) => {
                stats.output_dropped.fetch_add(1, Ordering::Relaxed);
                let _ = channels.free_tx.try_send(index);
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn write_error(message: &str, error: *mut c_char, error_len: usize) {
    if error.is_null() || error_len == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let count = bytes.len().min(error_len - 1);
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), error.cast::<u8>(), count);
        *error.add(count) = 0;
    }
}

unsafe fn engine_mut<'a>(handle: *mut Engine) -> Result<&'a mut Engine, String> {
    if handle.is_null() {
        Err("engine handle is null".into())
    } else {
        Ok(unsafe { &mut *handle })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_create(
    width: usize,
    height: usize,
    channels: usize,
    history_capacity: usize,
    queue_capacity: usize,
    mode: u32,
    error: *mut c_char,
    error_len: usize,
) -> *mut Engine {
    match catch_unwind(AssertUnwindSafe(|| {
        Engine::new(
            width,
            height,
            channels,
            history_capacity,
            queue_capacity,
            mode,
        )
    })) {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)),
        Ok(Err(message)) => {
            write_error(&message, error, error_len);
            ptr::null_mut()
        }
        Err(_) => {
            write_error("panic while creating native engine", error, error_len);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_create_graph(
    width: usize,
    height: usize,
    channels: usize,
    history_capacity: usize,
    queue_capacity: usize,
    graph_version: u32,
    operations: *const RuntimeGraphOp,
    operation_count: usize,
    error: *mut c_char,
    error_len: usize,
) -> *mut Engine {
    if operations.is_null() && operation_count != 0 {
        write_error("graph operations pointer is null", error, error_len);
        return ptr::null_mut();
    }
    match catch_unwind(AssertUnwindSafe(|| {
        let operation_slice = if operation_count == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(operations, operation_count) }
        };
        let graph = GraphProgram::validate(graph_version, operation_slice, history_capacity)?;
        Engine::new_with_graph(
            width,
            height,
            channels,
            history_capacity,
            queue_capacity,
            u32::MAX,
            graph,
        )
    })) {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)),
        Ok(Err(message)) => {
            write_error(&message, error, error_len);
            ptr::null_mut()
        }
        Err(_) => {
            write_error("panic while creating native graph engine", error, error_len);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_start(
    handle: *mut Engine,
    error: *mut c_char,
    error_len: usize,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| unsafe { engine_mut(handle)?.start() })) {
        Ok(Ok(())) => MG_OK,
        Ok(Err(message)) => {
            write_error(&message, error, error_len);
            MG_INVALID_STATE
        }
        Err(_) => {
            write_error("panic while starting native engine", error, error_len);
            MG_INTERNAL_ERROR
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_state(handle: *mut Engine) -> u32 {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        engine_mut(handle).map(|engine| engine.state() as u32)
    })) {
        Ok(Ok(state)) => state,
        _ => EngineState::Failed as u32,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_frame_bytes(handle: *mut Engine) -> usize {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        engine_mut(handle).map(|engine| engine.frame_bytes)
    })) {
        Ok(Ok(size)) => size,
        _ => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_submit(
    handle: *mut Engine,
    data: *const u8,
    length: usize,
    error: *mut c_char,
    error_len: usize,
) -> i32 {
    if data.is_null() {
        write_error("input frame pointer is null", error, error_len);
        return MG_INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        let bytes = slice::from_raw_parts(data, length);
        engine_mut(handle)?.submit(bytes)
    })) {
        Ok(Ok(true)) => MG_OK,
        Ok(Ok(false)) => MG_QUEUE_FULL,
        Ok(Err(message)) => {
            write_error(&message, error, error_len);
            MG_INVALID_ARGUMENT
        }
        Err(_) => {
            write_error("panic while submitting frame", error, error_len);
            MG_INTERNAL_ERROR
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_poll(
    handle: *mut Engine,
    output: *mut u8,
    output_len: usize,
    sequence: *mut u64,
    error: *mut c_char,
    error_len: usize,
) -> i32 {
    if output.is_null() {
        write_error("output frame pointer is null", error, error_len);
        return MG_INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        let output_slice = slice::from_raw_parts_mut(output, output_len);
        engine_mut(handle)?.poll(output_slice)
    })) {
        Ok(Ok(Some(value))) => {
            if !sequence.is_null() {
                unsafe { *sequence = value };
            }
            MG_OK
        }
        Ok(Ok(None)) => MG_EMPTY,
        Ok(Err(message)) => {
            let code = if message.contains("at least") {
                MG_BUFFER_TOO_SMALL
            } else {
                MG_INVALID_STATE
            };
            write_error(&message, error, error_len);
            code
        }
        Err(_) => {
            write_error("panic while polling frame", error, error_len);
            MG_INTERNAL_ERROR
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_wait(
    handle: *mut Engine,
    output: *mut u8,
    output_len: usize,
    timeout_ms: u64,
    sequence: *mut u64,
    error: *mut c_char,
    error_len: usize,
) -> i32 {
    if output.is_null() {
        write_error("output frame pointer is null", error, error_len);
        return MG_INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        let output_slice = slice::from_raw_parts_mut(output, output_len);
        engine_mut(handle)?.wait(output_slice, timeout_ms)
    })) {
        Ok(Ok(Some(value))) => {
            if !sequence.is_null() {
                unsafe { *sequence = value };
            }
            MG_OK
        }
        Ok(Ok(None)) => MG_EMPTY,
        Ok(Err(message)) => {
            let code = if message.contains("at least") {
                MG_BUFFER_TOO_SMALL
            } else {
                MG_INVALID_STATE
            };
            write_error(&message, error, error_len);
            code
        }
        Err(_) => {
            write_error("panic while waiting for frame", error, error_len);
            MG_INTERNAL_ERROR
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_set_parameters(
    handle: *mut Engine,
    delay: usize,
    mix: f32,
    frozen: bool,
    error: *mut c_char,
    error_len: usize,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        engine_mut(handle)?.update_parameters(delay, mix, frozen)
    })) {
        Ok(Ok(())) => MG_OK,
        Ok(Err(message)) => {
            write_error(&message, error, error_len);
            MG_INVALID_ARGUMENT
        }
        Err(_) => {
            write_error("panic while updating parameters", error, error_len);
            MG_INTERNAL_ERROR
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_set_graph(
    handle: *mut Engine,
    graph_version: u32,
    operations: *const RuntimeGraphOp,
    operation_count: usize,
    error: *mut c_char,
    error_len: usize,
) -> i32 {
    if operations.is_null() && operation_count != 0 {
        write_error("graph operations pointer is null", error, error_len);
        return MG_INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        let engine = engine_mut(handle)?;
        let operation_slice = if operation_count == 0 {
            &[]
        } else {
            slice::from_raw_parts(operations, operation_count)
        };
        let graph =
            GraphProgram::validate(graph_version, operation_slice, engine.history_capacity)?;
        engine.update_graph(graph)
    })) {
        Ok(Ok(())) => MG_OK,
        Ok(Err(message)) => {
            write_error(&message, error, error_len);
            MG_INVALID_ARGUMENT
        }
        Err(_) => {
            write_error("panic while updating native graph", error, error_len);
            MG_INTERNAL_ERROR
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_clear(
    handle: *mut Engine,
    error: *mut c_char,
    error_len: usize,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| unsafe { engine_mut(handle)?.clear() })) {
        Ok(Ok(())) => MG_OK,
        Ok(Err(message)) => {
            write_error(&message, error, error_len);
            MG_INVALID_STATE
        }
        Err(_) => {
            write_error("panic while clearing native engine", error, error_len);
            MG_INTERNAL_ERROR
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_get_statistics(handle: *mut Engine, output: *mut RuntimeStats) -> i32 {
    if output.is_null() {
        return MG_INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        let engine = engine_mut(handle)?;
        let history_bytes = engine.frame_bytes.saturating_mul(engine.history_capacity);
        let queue_bytes = engine.frame_bytes.saturating_mul(engine.queue_capacity);
        *output = RuntimeStats {
            submitted: engine.stats.submitted.load(Ordering::Relaxed),
            processed: engine.stats.processed.load(Ordering::Relaxed),
            delivered: engine.stats.delivered.load(Ordering::Relaxed),
            input_dropped: engine.stats.input_dropped.load(Ordering::Relaxed),
            output_dropped: engine.stats.output_dropped.load(Ordering::Relaxed),
            commands_applied: engine.stats.commands_applied.load(Ordering::Relaxed),
            frames_in_history: engine.stats.frames_in_history.load(Ordering::Relaxed),
            allocated_bytes: history_bytes.saturating_add(queue_bytes) as u64,
            sequence: engine.stats.sequence.load(Ordering::Relaxed),
            state: engine.state() as u32,
        };
        Ok::<(), String>(())
    })) {
        Ok(Ok(())) => MG_OK,
        _ => MG_INTERNAL_ERROR,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_shutdown(handle: *mut Engine) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        let engine = engine_mut(handle)?;
        engine.shutdown();
        Ok::<(), String>(())
    })) {
        Ok(Ok(())) => MG_OK,
        _ => MG_INTERNAL_ERROR,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mg_engine_destroy(handle: *mut Engine) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(handle));
    }));
}
