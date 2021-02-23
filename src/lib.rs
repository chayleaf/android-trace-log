pub use chrono;
use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take, take_while},
    character::complete::{digit1, hex_digit1, tab},
    combinator::{map, map_res, opt, value},
    error::ParseError,
    multi::many0,
    number::complete::{le_u16, le_u32, le_u64, le_u8},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    error, fmt,
    io::{self, Write},
    num::{NonZeroU64, ParseIntError},
    slice, str,
    time::Duration,
};

/// Clock used for the trace log
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub enum Clock {
    /// Global time (gettimeofday)
    Global,
    /// Thread-local CPU time
    Cpu,
    /// Wall time
    Wall,
    /// Both CPU time and wall time
    Dual,
}

/// An event time offset since the trace start.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub enum Time {
    /// Global time (gettimeofday)
    Global(Duration),
    /// Monotonic time. At least one member is guaranteed to be Some.
    Monotonic {
        /// Thread-local CPU time
        cpu: Option<Duration>,
        /// Wall time
        wall: Option<Duration>,
    },
}

/// The process VM
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub enum Vm {
    /// Dalvik
    Dalvik,
    /// Android Runtime
    Art,
}

/// Garbage collector tracing info
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct GcTrace {
    /// Number of allocated objects
    pub alloc_count: u64,
    /// Size of allocated objects in bytes
    pub alloc_size: u64,
    /// Number of completed garbage collections
    pub gc_count: u64,
}

/// A traced thread
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Thread {
    /// Thread ID
    pub id: u16,
    /// Thread name
    pub name: String,
}

/// A traced method
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Method {
    /// Method ID
    pub id: u32,
    /// Method name
    pub name: String,
    /// Class name
    pub class_name: String,
    /// The method's [type signature](https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html#wp276)
    pub signature: String,
    /// The method's source file
    pub source_file: String,
    /// The method's source line
    pub source_line: Option<u32>,
}

/// Trace event action
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub enum Action {
    /// Recorded when entering a method
    Enter,
    /// Recorded when exiting a method
    Exit,
    /// Recorded when unwinding from a method because of an exception
    Unwind,
}

/// Trace event
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Event {
    /// Event action
    pub action: Action,
    /// Event thread ID
    pub thread_id: u16,
    /// Event method ID
    pub method_id: u32,
    /// Event time
    pub time: Time,
}

/// A view into an event with convenience methods for accessing the associated thread and method. Use [`AndroidTraceLog::iter`] to get an iterator over these.
#[derive(Copy, Clone, Debug)]
pub struct EventView<'a> {
    action: Action,
    thread_id: usize,
    threads: &'a Vec<Thread>,
    method_id: usize,
    methods: &'a Vec<Method>,
    time: Time,
}

impl<'a> EventView<'a> {
    /// Event action
    pub fn action(&self) -> Action {
        self.action
    }
    /// Event thread
    pub fn thread(&self) -> &Thread {
        self.threads.get(self.thread_id).unwrap()
    }
    /// Event method
    pub fn method(&self) -> &Method {
        self.methods.get(self.method_id).unwrap()
    }
    /// Event time
    pub fn time(&self) -> Time {
        self.time
    }
}

/// Android trace log
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct AndroidTraceLog {
    /// Whether the trace has overflowed the trace buffer and is missing events
    pub data_file_overflow: bool,
    /// The clock used for the events
    pub clock: Clock,
    /// Total recording duration
    pub elapsed_time: Duration,
    /// Total number of calls registered
    pub total_method_calls: u64,
    /// The approximate trace overhead per call
    pub clock_call_overhead: Duration,
    /// The process VM used
    pub vm: Vm,
    /// Trace start time
    #[cfg_attr(feature = "serde", serde(with = "chrono::serde::ts_nanoseconds"))]
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// PID of the traced process
    pub pid: Option<u32>,
    /// If GC is being tracked, contains the GC trace info
    pub gc_trace: Option<GcTrace>,
    /// All threads in the trace log
    pub threads: Vec<Thread>,
    /// All methods in the trace log
    pub methods: Vec<Method>,
    /// All events in the trace log
    pub events: Vec<Event>,
}

impl AndroidTraceLog {
    /// Verifies the object can be serialized and iterated over
    pub fn valid(&self) -> bool {
        let method_ids = self.methods.iter().map(|x| x.id).collect::<HashSet<_>>();
        let thread_ids = self.threads.iter().map(|x| x.id).collect::<HashSet<_>>();
        for event in self.events.iter() {
            if !method_ids.contains(&event.method_id) || !thread_ids.contains(&event.thread_id) {
                return false;
            }
            match (self.clock, event.time) {
                (Clock::Global, Time::Global(_)) => {
                    if event.thread_id > 0xFF {
                        return false;
                    }
                }
                (
                    Clock::Cpu,
                    Time::Monotonic {
                        cpu: Some(_),
                        wall: None,
                    },
                ) => {}
                (
                    Clock::Wall,
                    Time::Monotonic {
                        cpu: None,
                        wall: Some(_),
                    },
                ) => {}
                (
                    Clock::Dual,
                    Time::Monotonic {
                        cpu: Some(_),
                        wall: Some(_),
                    },
                ) => {}
                _ => return false,
            }
            if event.method_id > (0xFFFFFFFF >> 2) {
                return false;
            }
        }
        true
    }

    /// Serialize the trace into a Write object
    ///
    /// Panics if using global clock (only used in file version 1) with 2-byte thread IDs (only used in file version 2+)
    /// or if some event's time doesn't match the trace's `clock` value. Will not panic if [`valid`](AndroidTraceLog::valid) returned `true`.
    pub fn serialize_into<W: Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(w, "*version")?;
        let version: u16 = match self.clock {
            Clock::Global => 1,
            Clock::Cpu => 2,
            Clock::Wall => 2,
            Clock::Dual => 3,
        };
        writeln!(w, "{}", version)?;
        writeln!(w, "data-file-overflow={}", self.data_file_overflow)?;
        writeln!(
            w,
            "clock={}",
            match self.clock {
                Clock::Global => "global",
                Clock::Cpu => "thread-cpu",
                Clock::Wall => "wall",
                Clock::Dual => "dual",
            }
        )?;
        writeln!(w, "elapsed-time-usec={}", self.elapsed_time.as_micros())?;
        writeln!(w, "num-method-calls={}", self.total_method_calls)?;
        writeln!(
            w,
            "clock-call-overhead-nsec={}",
            self.clock_call_overhead.as_nanos()
        )?;
        writeln!(
            w,
            "vm={}",
            match self.vm {
                Vm::Dalvik => "dalvik",
                Vm::Art => "art",
            }
        )?;
        if let Some(pid) = self.pid {
            writeln!(w, "pid={}", pid)?;
        }
        if let Some(ref gc) = self.gc_trace {
            writeln!(w, "alloc-count={}", gc.alloc_count)?;
            writeln!(w, "alloc-size={}", gc.alloc_size)?;
            writeln!(w, "gc-count={}", gc.gc_count)?;
        }
        writeln!(w, "*threads")?;
        for thread in self.threads.iter() {
            writeln!(w, "{}\t{}", thread.id, thread.name)?;
        }
        writeln!(w, "*methods")?;
        for method in self.methods.iter() {
            if version <= 1 || method.id > 0 {
                write!(w, "{:#x}", method.id << 2)
            } else {
                write!(w, "{}", method.id << 2)
            }?;
            write!(
                w,
                "\t{}\t{}\t{}\t{}",
                method.class_name, method.name, method.signature, method.source_file
            )?;
            if let Some(line) = method.source_line {
                write!(w, "\t{}", line)?;
            }
            writeln!(w)?;
        }
        writeln!(w, "*end")?;
        w.write_all(b"SLOW")?;
        w.write_all(&version.to_le_bytes())?;
        // Header size
        w.write_all(&32u16.to_le_bytes())?;
        let start_time = self.start_time.naive_utc();
        w.write_all(
            &(start_time.timestamp() as u64 * 1000000
                + (start_time.timestamp_nanos() as u64 / 1000) % 1000000)
                .to_le_bytes(),
        )?;
        if version >= 3 {
            // Entry size
            w.write_all(
                &match self.clock {
                    Clock::Dual => 14u16,
                    _ => 10u16,
                }
                .to_le_bytes(),
            )?;
            // Header padding
            w.write_all(b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0")?;
        } else {
            // Header padding
            w.write_all(b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")?;
        };
        for event in self.events.iter() {
            if version == 1 {
                w.write_all(
                    &u8::try_from(event.thread_id)
                    .expect("Thread ID too big for given log version (guessed from global clock type)")
                    .to_le_bytes()
                )?;
            } else {
                w.write_all(&event.thread_id.to_le_bytes())?;
            }
            let id = (event.method_id << 2)
                | match event.action {
                    Action::Enter => 0,
                    Action::Exit => 1,
                    Action::Unwind => 2,
                };
            w.write_all(&id.to_le_bytes())?;
            let (time_a, time_b) = match (self.clock, event.time) {
                (Clock::Global, Time::Global(time)) => (time, None),
                (
                    Clock::Cpu,
                    Time::Monotonic {
                        cpu: Some(time),
                        wall: None,
                    },
                ) => (time, None),
                (
                    Clock::Wall,
                    Time::Monotonic {
                        wall: Some(time),
                        cpu: None,
                    },
                ) => (time, None),
                (
                    Clock::Dual,
                    Time::Monotonic {
                        cpu: Some(cpu),
                        wall: Some(wall),
                    },
                ) => (cpu, Some(wall)),
                _ => panic!("Mismatch between clock type and recorded event time"),
            };
            w.write_all(&(time_a.as_micros() as u32).to_le_bytes())?;
            if let Some(time_b) = time_b {
                w.write_all(&(time_b.as_micros() as u32).to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Serialize the trace into a byte array. Can panic if the log is invalid; see [`serialize_into`](AndroidTraceLog::serialize_into)
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut ret = Vec::new();
        self.serialize_into(&mut ret)?;
        Ok(ret)
    }

    /// Iterate over all events while preparing an event view that allows to quickly access an event's thread and method.
    /// Has some startup time due to needing to prepare method/thread lookup tables.
    ///
    /// Panics if an event has an invalid method/thread ID. Will not panic if [`valid`](AndroidTraceLog::valid) returned `true`.
    pub fn iter(&self) -> impl Iterator<Item = EventView<'_>> {
        let method_keys = self
            .methods
            .iter()
            .enumerate()
            .map(|(i, method)| (method.id, i))
            .collect::<HashMap<_, _>>();
        let thread_keys = self
            .threads
            .iter()
            .enumerate()
            .map(|(i, thread)| (thread.id, i))
            .collect::<HashMap<_, _>>();
        self.events.iter().map(move |event| EventView {
            action: event.action,
            time: event.time,
            methods: &self.methods,
            threads: &self.threads,
            method_id: method_keys[&event.method_id],
            thread_id: thread_keys[&event.thread_id],
        })
    }
}

trait Num: Sized {
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError>;
    /// Panics if slice isn't valid utf8
    fn from_slice_radix(s: &[u8], radix: u32) -> Result<Self, ParseIntError> {
        Self::from_str_radix(str::from_utf8(s).unwrap(), radix)
    }
}

macro_rules! impl_num {
    [$($type:ty),+] => {
        $(impl Num for $type {
            fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError> {
                Self::from_str_radix(s, radix)
            }
        })+
    };
}

impl_num![usize, u64, u32, u16, u8, isize, i64, i32, i16, i8];

fn parse_hex<T: Num>(input: &[u8]) -> IResult<&[u8], T> {
    preceded(
        tag("0x"),
        map_res(hex_digit1, |s| T::from_slice_radix(s, 16)),
    )(input)
}

fn parse_dec<T: Num>(input: &[u8]) -> IResult<&[u8], T> {
    map_res(digit1, |s| T::from_slice_radix(s, 10))(input)
}

fn parse_num<T: Num>(input: &[u8]) -> IResult<&[u8], T> {
    alt((parse_hex, parse_dec))(input)
}

fn parse_bool(input: &[u8]) -> IResult<&[u8], bool> {
    alt((value(false, tag(b"false")), value(true, tag(b"true"))))(input)
}

fn parse_clock(input: &[u8]) -> IResult<&[u8], Clock> {
    alt((
        value(Clock::Global, tag(b"global")),
        value(Clock::Cpu, tag(b"thread-cpu")),
        value(Clock::Wall, tag(b"wall")),
        value(Clock::Dual, tag(b"dual")),
    ))(input)
}

fn parse_vm(input: &[u8]) -> IResult<&[u8], Vm> {
    alt((
        value(Vm::Dalvik, tag(b"dalvik")),
        value(Vm::Art, tag(b"art")),
    ))(input)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum HeaderLine {
    DataFileOverflow(bool),
    Clock(Clock),
    ElapsedTime(Duration),
    NumMethodCalls(u64),
    ClockCallOverhead(Duration),
    Vm(Vm),
    Pid(u32),
    AllocCount(u64),
    AllocSize(u64),
    GcCount(u64),
}

fn parse_header_line(input: &[u8]) -> IResult<&[u8], HeaderLine> {
    fn header_line<'a, T, F, E>(
        tag_name: &'a [u8],
        parser: F,
    ) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], T, E> + 'a
    where
        T: 'a,
        F: FnMut(&'a [u8]) -> IResult<&'a [u8], T, E> + 'a,
        E: ParseError<&'a [u8]> + 'a,
    {
        delimited(pair(tag(tag_name), tag(b"=")), parser, tag(b"\n"))
    }

    alt((
        header_line(
            b"data-file-overflow",
            map(parse_bool, HeaderLine::DataFileOverflow),
        ),
        header_line(b"clock", map(parse_clock, HeaderLine::Clock)),
        header_line(
            b"elapsed-time-usec",
            map(parse_num, |x| {
                HeaderLine::ElapsedTime(Duration::from_micros(x))
            }),
        ),
        header_line(
            b"num-method-calls",
            map(parse_num, HeaderLine::NumMethodCalls),
        ),
        header_line(
            b"clock-call-overhead-nsec",
            map(parse_num, |x| {
                HeaderLine::ClockCallOverhead(Duration::from_nanos(x))
            }),
        ),
        header_line(b"vm", map(parse_vm, HeaderLine::Vm)),
        header_line(b"pid", map(parse_num, HeaderLine::Pid)),
        header_line(b"alloc-count", map(parse_num, HeaderLine::AllocCount)),
        header_line(b"alloc-size", map(parse_num, HeaderLine::AllocSize)),
        header_line(b"gc-count", map(parse_num, HeaderLine::GcCount)),
    ))(input)
}

fn terminated_with<'a>(until: &'a u8) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], &'a str> + 'a {
    terminated(
        map_res(is_not(slice::from_ref(until)), str::from_utf8),
        tag(slice::from_ref(until)),
    )
}

fn parse_thread(input: &[u8]) -> IResult<&[u8], Thread> {
    map(
        pair(terminated(parse_num, tab), terminated_with(&b'\n')),
        |(id, name)| Thread {
            id,
            name: name.into(),
        },
    )(input)
}

fn parse_method(input: &[u8]) -> IResult<&[u8], Method> {
    map(
        tuple((
            terminated(parse_num::<u32>, tab),
            terminated_with(&b'\t'),
            terminated_with(&b'\t'),
            terminated_with(&b'\t'),
            map_res(take_while(|x| x != b'\n' && x != b'\t'), str::from_utf8),
            alt((
                value(None, tag(b"\n")),
                delimited(tag(b"\t"), opt(parse_num), tag(b"\n")),
            )),
        )),
        |(id, class_name, name, signature, source_file, source_line)| Method {
            id: id >> 2,
            class_name: class_name.into(),
            signature: signature.into(),
            name: name.into(),
            source_file: source_file.into(),
            source_line,
        },
    )(input)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum Error {
    InvalidEventAction,
    InvalidFormat,
    UnsupportedVersion,
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEventAction => write!(f, "invalid event action"),
            Self::InvalidFormat => write!(f, "invalid file format"),
            Self::UnsupportedVersion => write!(f, "unsupported version"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Version {
    One,
    Two,
    Three(Option<NonZeroU64>),
}

fn event_parser(version: Version, clock: Clock) -> impl Fn(&[u8]) -> IResult<&[u8], Event> {
    let clock = match (version, clock) {
        (Version::One, _) => Clock::Global,
        (Version::Two, Clock::Dual) => Clock::Cpu, // Should be an error but I'm too lazy
        (_, clock) => clock,
    };
    move |input: &[u8]| -> IResult<&[u8], Event> {
        let (input, thread_id) = match version {
            Version::One => map(le_u8, |thread_id| thread_id.into())(input)?,
            _ => le_u16(input)?,
        };
        let (input, (action, method_id)) = map_res(le_u32, |x| {
            let action = match x & 0b11 {
                0 => Action::Enter,
                1 => Action::Exit,
                2 => Action::Unwind,
                _ => return Err(Error::InvalidEventAction),
            };
            Ok((action, x >> 2))
        })(input)?;
        let (input, time_a) = le_u32(input)?;
        let time_a = Duration::from_micros(time_a as _);
        let (mut input, time) = match clock {
            Clock::Dual => map(le_u32, |time_b| Time::Monotonic {
                cpu: Some(time_a),
                wall: Some(Duration::from_micros(time_b as _)),
            })(input)?,
            Clock::Cpu => (
                input,
                Time::Monotonic {
                    cpu: Some(time_a),
                    wall: None,
                },
            ),
            Clock::Wall => (
                input,
                Time::Monotonic {
                    wall: Some(time_a),
                    cpu: None,
                },
            ),
            Clock::Global => (input, Time::Global(time_a)),
        };

        if let Version::Three(Some(event_size)) = version {
            let bytes_read = match version { // Thread ID
                Version::One => 1,
                _ => 2,
            }
                + 4 // Action + method ID
                + 4 // Time A
                + match clock { // Time B
                    Clock::Dual => 4,
                    _ => 0,
                };
            let event_size = event_size.get();
            if bytes_read < event_size {
                input = take(event_size - bytes_read)(input)?.0;
            }
        }

        Ok((
            input,
            Event {
                action,
                thread_id,
                method_id,
                time,
            },
        ))
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Section {
    Version,
    Threads,
    Methods,
    End,
}

fn parse_section(input: &[u8]) -> IResult<&[u8], Section> {
    delimited(
        tag(b"*"),
        alt((
            value(Section::Version, tag(b"version")),
            value(Section::Threads, tag(b"threads")),
            value(Section::Methods, tag(b"methods")),
            value(Section::End, tag(b"end")),
        )),
        tag(b"\n"),
    )(input)
}

fn parse_android_trace(input: &[u8]) -> IResult<&[u8], AndroidTraceLog> {
    use chrono::TimeZone;
    let mut ret = AndroidTraceLog {
        data_file_overflow: Default::default(),
        clock: Clock::Global,
        elapsed_time: Default::default(),
        total_method_calls: Default::default(),
        clock_call_overhead: Default::default(),
        vm: Vm::Dalvik,
        start_time: chrono::Utc.ymd(1970, 1, 1).and_hms(0, 0, 0),
        pid: Default::default(),
        gc_trace: Default::default(),
        threads: Default::default(),
        methods: Default::default(),
        events: Default::default(),
    };
    let (input, _) = tag(b"*version\n")(input)?;
    let mut section = Section::Version;
    let (input, version) = map_res(parse_num, |x| match x {
        1u32 => Ok(Version::One),
        2u32 => Ok(Version::Two),
        3u32 => Ok(Version::Three(None)),
        _ => Err(Error::UnsupportedVersion),
    })(input)?;
    let (mut input, _) = tag(b"\n")(input)?;
    loop {
        match section {
            Section::Version => {
                enum Line {
                    NewSection(Section),
                    Info(HeaderLine),
                }
                let (inp, line) = alt((
                    map(parse_section, Line::NewSection),
                    map(parse_header_line, Line::Info),
                ))(input)?;
                input = inp;
                use HeaderLine::*;
                fn empty_trace() -> GcTrace {
                    GcTrace {
                        alloc_count: 0,
                        alloc_size: 0,
                        gc_count: 0,
                    }
                }
                match line {
                    Line::NewSection(sect) => section = sect,
                    Line::Info(DataFileOverflow(v)) => ret.data_file_overflow = v,
                    Line::Info(Clock(v)) => ret.clock = v,
                    Line::Info(ElapsedTime(v)) => ret.elapsed_time = v,
                    Line::Info(NumMethodCalls(v)) => ret.total_method_calls = v,
                    Line::Info(ClockCallOverhead(v)) => ret.clock_call_overhead = v,
                    Line::Info(Vm(v)) => ret.vm = v,
                    Line::Info(Pid(v)) => ret.pid = Some(v),
                    Line::Info(AllocCount(v)) => {
                        ret.gc_trace.get_or_insert_with(empty_trace).alloc_count = v
                    }
                    Line::Info(AllocSize(v)) => {
                        ret.gc_trace.get_or_insert_with(empty_trace).alloc_size = v
                    }
                    Line::Info(GcCount(v)) => {
                        ret.gc_trace.get_or_insert_with(empty_trace).gc_count = v
                    }
                }
            }
            Section::Threads => {
                enum Line {
                    NewSection(Section),
                    Thread(Thread),
                }
                let (inp, line) = alt((
                    map(parse_section, Line::NewSection),
                    map(parse_thread, Line::Thread),
                ))(input)?;
                input = inp;
                match line {
                    Line::NewSection(sect) => section = sect,
                    Line::Thread(thread) => ret.threads.push(thread),
                }
            }
            Section::Methods => {
                enum Line {
                    NewSection(Section),
                    Method(Method),
                }
                let (inp, line) = alt((
                    map(parse_section, Line::NewSection),
                    map(parse_method, Line::Method),
                ))(input)?;
                input = inp;
                match line {
                    Line::NewSection(sect) => section = sect,
                    Line::Method(method) => ret.methods.push(method),
                }
            }
            Section::End => break,
        }
    }
    let (input, _) = tag(b"SLOW")(input)?;

    let (input, bin_version) = map_res(le_u16, |x| match x {
        1 => Ok(Version::One),
        2 => Ok(Version::Two),
        3 => Ok(Version::Three(None)),
        _ => Err(Error::UnsupportedVersion),
    })(input)?;

    if version != bin_version {
        log::warn!(
            "Text and binary version mismatch: {:?}/{:?}",
            version,
            bin_version
        );
    }

    let (input, data_offset) = le_u16(input)?;

    let (input, start_time) = map(le_u64, |start_time| {
        chrono::DateTime::from_utc(
            chrono::NaiveDateTime::from_timestamp(
                (start_time / 1000000) as _,
                ((start_time % 1000000) * 1000) as _,
            ),
            chrono::Utc,
        )
    })(input)?;

    ret.start_time = start_time;

    let ((input, bin_version), header_len) = match bin_version {
        Version::Three(_) => (
            map(le_u16, |event_size| {
                Version::Three(NonZeroU64::new(event_size as _))
            })(input)?,
            4 + 2 + 2 + 8 + 2,
        ),
        bin_version => ((input, bin_version), 4 + 2 + 2 + 8),
    };

    let input = if header_len < data_offset {
        take(data_offset - header_len)(input)?.0
    } else {
        input
    };

    let parser = event_parser(bin_version, ret.clock);
    let (input, events) = many0(parser)(input)?;
    ret.events = events;
    Ok((input, ret))
}

/// Parse a byte array into an android trace
/// # Example
/// ```
/// # use std::{error, fs, io::{self, prelude::*}};
/// # struct Error;
/// # impl From<io::Error> for Error {
/// #     fn from(_: io::Error) -> Self {
/// #         Self
/// #     }
/// # }
/// # impl<'a> From<Box<dyn error::Error + 'a>> for Error {
/// #     fn from(_: Box<dyn error::Error + 'a>) -> Self {
/// #         Self
/// #     }
/// # }
/// # fn _unused() -> Result<(), Error> {
/// let mut trace_bytes = Vec::new();
/// fs::File::open("dmtrace.trace")?.read_to_end(&mut trace_bytes);
/// let trace = android_trace_log::parse(&trace_bytes[..])?;
/// # let _ = trace;
/// # Ok(())
/// # }
/// ```
pub fn parse(bytes: &[u8]) -> Result<AndroidTraceLog, Box<dyn error::Error + '_>> {
    Ok(parse_android_trace(bytes)?.1)
}

#[cfg(test)]
mod tests {
    use {crate::*, chrono::DurationRound};

    #[test]
    fn section_names() {
        assert_eq!(
            parse_section(b"*version\n"),
            Ok((b"" as _, Section::Version))
        );
        assert_eq!(
            parse_section(b"*methods\n"),
            Ok((b"" as _, Section::Methods))
        );
        assert_eq!(
            parse_section(b"*threads\n"),
            Ok((b"" as _, Section::Threads))
        );
        assert_eq!(parse_section(b"*end\n"), Ok((b"" as _, Section::End)));
        assert!(parse_section(b"*whatever\n").is_err());
    }

    #[test]
    fn header_lines() {
        assert_eq!(
            parse_header_line(b"data-file-overflow=false\n"),
            Ok((b"" as _, HeaderLine::DataFileOverflow(false)))
        );
        assert_eq!(
            parse_header_line(b"data-file-overflow=true\n"),
            Ok((b"" as _, HeaderLine::DataFileOverflow(true)))
        );
        assert_eq!(
            parse_header_line(b"clock=global\n"),
            Ok((b"" as _, HeaderLine::Clock(Clock::Global)))
        );
        assert_eq!(
            parse_header_line(b"clock=thread-cpu\n"),
            Ok((b"" as _, HeaderLine::Clock(Clock::Cpu)))
        );
        assert_eq!(
            parse_header_line(b"clock=wall\n"),
            Ok((b"" as _, HeaderLine::Clock(Clock::Wall)))
        );
        assert_eq!(
            parse_header_line(b"clock=dual\n"),
            Ok((b"" as _, HeaderLine::Clock(Clock::Dual)))
        );
        assert_eq!(
            parse_header_line(b"elapsed-time-usec=123456\n"),
            Ok((
                b"" as _,
                HeaderLine::ElapsedTime(Duration::from_micros(123456))
            ))
        );
        assert_eq!(
            parse_header_line(b"num-method-calls=654321\n"),
            Ok((b"" as _, HeaderLine::NumMethodCalls(654321)))
        );
        assert_eq!(
            parse_header_line(b"clock-call-overhead-nsec=1337\n"),
            Ok((
                b"" as _,
                HeaderLine::ClockCallOverhead(Duration::from_nanos(1337))
            ))
        );
        assert_eq!(
            parse_header_line(b"vm=dalvik\n"),
            Ok((b"" as _, HeaderLine::Vm(Vm::Dalvik)))
        );
        assert_eq!(
            parse_header_line(b"vm=art\n"),
            Ok((b"" as _, HeaderLine::Vm(Vm::Art)))
        );
        assert_eq!(
            parse_header_line(b"pid=666\n"),
            Ok((b"" as _, HeaderLine::Pid(666)))
        );
        assert_eq!(
            parse_header_line(b"alloc-size=123\n"),
            Ok((b"" as _, HeaderLine::AllocSize(123)))
        );
        assert_eq!(
            parse_header_line(b"alloc-count=456\n"),
            Ok((b"" as _, HeaderLine::AllocCount(456)))
        );
        assert_eq!(
            parse_header_line(b"gc-count=420\n"),
            Ok((b"" as _, HeaderLine::GcCount(420)))
        );
    }
    #[test]
    fn threads() {
        assert_eq!(
            parse_thread(b"13337\tA thread\n"),
            Ok((
                b"" as _,
                Thread {
                    id: 13337,
                    name: "A thread".into()
                }
            ))
        );
    }
    #[test]
    fn methods() {
        assert_eq!(
            parse_method(b"0xeb0\tclass\tname\tsig\tfile\t123\n"),
            Ok((
                b"" as _,
                Method {
                    id: 0xEB0 / 4,
                    class_name: "class".into(),
                    name: "name".into(),
                    signature: "sig".into(),
                    source_file: "file".into(),
                    source_line: Some(123),
                }
            ))
        );
        assert_eq!(
            parse_method(b"0x664\tclazz\tname\tsig\t\n"),
            Ok((
                b"" as _,
                Method {
                    id: 0x664 / 4,
                    class_name: "clazz".into(),
                    name: "name".into(),
                    signature: "sig".into(),
                    source_file: "".into(),
                    source_line: None,
                }
            ))
        );
        assert_eq!(
            parse_method(b"0\tclass\tname\tsig\t\t\n"),
            Ok((
                b"" as _,
                Method {
                    id: 0,
                    class_name: "class".into(),
                    name: "name".into(),
                    signature: "sig".into(),
                    source_file: "".into(),
                    source_line: None,
                }
            ))
        );
    }
    #[test]
    fn events() {
        let parser1 = event_parser(Version::One, Clock::Global);
        assert_eq!(
            parser1(b"\x45\x04\x00\x00\x00\x01\x00\x00\x00"),
            Ok((
                b"" as _,
                Event {
                    thread_id: 69,
                    method_id: 1,
                    action: Action::Enter,
                    time: Time::Global(Duration::from_micros(1)),
                }
            ))
        );
        assert!(parser1(b"\x45\x03\x00\x00\x00\x01\x00\x00\x00").is_err());
        let parser2 = event_parser(Version::Two, Clock::Wall);
        assert_eq!(
            parser2(b"\x39\x05\x09\x00\x00\x00\x01\x00\x00\x00"),
            Ok((
                b"" as _,
                Event {
                    thread_id: 1337,
                    method_id: 2,
                    action: Action::Exit,
                    time: Time::Monotonic {
                        cpu: None,
                        wall: Some(Duration::from_micros(1)),
                    },
                }
            ))
        );
        let parser3 = event_parser(Version::Three(NonZeroU64::new(20)), Clock::Dual);
        assert_eq!(
            parser3(
                b"\xFF\xFF\x02\x00\x00\x00\x10\x00\x00\x00\x20\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            ),
            Ok((
                b"" as _,
                Event {
                    thread_id: 0xFFFF,
                    method_id: 0,
                    action: Action::Unwind,
                    time: Time::Monotonic {
                        cpu: Some(Duration::from_micros(0x10)),
                        wall: Some(Duration::from_micros(0x20)),
                    }
                }
            ))
        );
        assert!(parser3(b"\xFF\xFF\x02\x00\x00\x00\x10\x00\x00\x00\x20\x00\x00\x00").is_err());
    }

    #[test]
    fn trace_and_event_view() {
        let now = chrono::Utc::now();
        let trace = AndroidTraceLog {
            data_file_overflow: true,
            clock: Clock::Dual,
            elapsed_time: Duration::from_secs(10),
            total_method_calls: 2,
            clock_call_overhead: Duration::from_nanos(100),
            vm: Vm::Art,
            start_time: now
                .duration_trunc(chrono::Duration::from_std(Duration::from_micros(1)).unwrap())
                .unwrap(),
            pid: Some(256),
            gc_trace: Some(GcTrace {
                alloc_count: 1,
                gc_count: 2,
                alloc_size: 3,
            }),
            threads: vec![
                Thread {
                    id: 1337,
                    name: "Thread 1".into(),
                },
                Thread {
                    id: 13337,
                    name: "Thread 2".into(),
                },
            ],
            methods: vec![
                Method {
                    id: 0,
                    name: "meth0".into(),
                    class_name: "class".into(),
                    signature: "()V".into(),
                    source_file: "class.java".into(),
                    source_line: Some(5),
                },
                Method {
                    id: 1,
                    name: "meth1".into(),
                    class_name: "class".into(),
                    signature: "(I)I".into(),
                    source_file: "".into(),
                    source_line: None,
                },
            ],
            events: vec![
                Event {
                    method_id: 0,
                    thread_id: 1337,
                    time: Time::Monotonic {
                        cpu: Some(Duration::from_millis(1)),
                        wall: Some(Duration::from_millis(1)),
                    },
                    action: Action::Enter,
                },
                Event {
                    method_id: 1,
                    thread_id: 13337,
                    time: Time::Monotonic {
                        cpu: Some(Duration::from_millis(2)),
                        wall: Some(Duration::from_millis(2)),
                    },
                    action: Action::Enter,
                },
                Event {
                    method_id: 0,
                    thread_id: 1337,
                    time: Time::Monotonic {
                        cpu: Some(Duration::from_millis(2)),
                        wall: Some(Duration::from_millis(2)),
                    },
                    action: Action::Exit,
                },
                Event {
                    method_id: 1,
                    thread_id: 13337,
                    time: Time::Monotonic {
                        cpu: Some(Duration::from_millis(3)),
                        wall: Some(Duration::from_millis(3)),
                    },
                    action: Action::Unwind,
                },
            ],
        };

        let mut iter = trace.iter();
        let view = iter.next().unwrap();
        assert_eq!(
            (view.action(), view.time()),
            (trace.events[0].action, trace.events[0].time)
        );
        assert_eq!(
            (view.method(), view.thread()),
            (&trace.methods[0], &trace.threads[0])
        );
        let view = iter.next().unwrap();
        assert_eq!(
            (view.action(), view.time()),
            (trace.events[1].action, trace.events[1].time)
        );
        assert_eq!(
            (view.method(), view.thread()),
            (&trace.methods[1], &trace.threads[1])
        );
        let view = iter.next().unwrap();
        assert_eq!(
            (view.action(), view.time()),
            (trace.events[2].action, trace.events[2].time)
        );
        assert_eq!(
            (view.method(), view.thread()),
            (&trace.methods[0], &trace.threads[0])
        );
        let view = iter.next().unwrap();
        assert_eq!(
            (view.action(), view.time()),
            (trace.events[3].action, trace.events[3].time)
        );
        assert_eq!(
            (view.method(), view.thread()),
            (&trace.methods[1], &trace.threads[1])
        );
        assert!(iter.next().is_none());
        assert!(trace.valid());

        let trace2 = parse(&trace.serialize().unwrap()).unwrap();
        assert_eq!(trace, trace2);
    }
}
