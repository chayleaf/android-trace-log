pub use chrono;
use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_till, take_until, take_while1},
    combinator::map_res,
    error::ErrorKind,
    AsChar, IResult,
};
use serde::{Deserialize, Serialize};
use std::{
    array::TryFromSliceError,
    convert::{TryFrom, TryInto},
    error, fmt,
    io::{self, Write},
    mem,
    num::{NonZeroU64, ParseIntError},
    str,
    time::Duration,
};

/// Clock used for the trace log
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl Default for Clock {
    fn default() -> Self {
        Self::Global
    }
}

/// The process VM
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Vm {
    /// Dalvik
    Dalvik,
    /// Android Runtime
    Art,
}

impl Default for Vm {
    fn default() -> Self {
        Self::Dalvik
    }
}

/// Garbage collector tracing info
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GcTrace {
    /// Number of allocated objects
    pub alloc_count: u64,
    /// Size of allocated objects in bytes
    pub alloc_size: u64,
    /// Number of completed garbage collections
    pub gc_count: u64,
}

/// A traced thread
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Thread {
    /// Thread ID
    pub id: u16,
    /// Thread name
    pub name: String,
}

/// A traced method
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// Recorded when entering a method
    Enter,
    /// Recorded when exiting a method
    Exit,
    /// Recorded when unwinding from a method because of an exception
    Unwind,
}

/// Trace event
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Android trace log
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AndroidTrace {
    /// Whether the trace has overflowed the trace buffer and is missing events
    pub data_file_overflow: bool,
    /// The clock used for the events
    pub clock: Clock,
    /// Total recording duration
    pub elapsed_time: Duration,
    /// Total number of calls registered
    pub num_method_calls: u64,
    /// The approximate trace overhead per call
    pub clock_call_overhead: Duration,
    /// The process VM used
    pub vm: Vm,
    /// Trace start time
    #[serde(with = "chrono::serde::ts_nanoseconds")]
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

impl Default for AndroidTrace {
    fn default() -> Self {
        use chrono::TimeZone;
        Self {
            data_file_overflow: Default::default(),
            clock: Default::default(),
            elapsed_time: Default::default(),
            num_method_calls: Default::default(),
            clock_call_overhead: Default::default(),
            vm: Default::default(),
            start_time: chrono::Utc.ymd(1970, 1, 1).and_hms(0, 0, 0),
            pid: Default::default(),
            gc_trace: Default::default(),
            threads: Default::default(),
            methods: Default::default(),
            events: Default::default(),
        }
    }
}

impl AndroidTrace {
    /// Serialize the trace into a Write object
    /// 
    /// Panics if using global clock (only used in file version 1) with 2-byte thread IDs (only used in file version 2+) or if some event's time doesn't match the trace's `clock` value.
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
        writeln!(w, "num-method-calls={}", self.num_method_calls)?;
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
                write!(w, "{:#x}", method.id)
            } else {
                write!(w, "{}", method.id)
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
                w.write_all(&u8::try_from(event.thread_id).expect("Thread ID too big for given log version (guessed from global clock type)").to_le_bytes())?;
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

    /// Serialize the trace into a byte array
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut ret = Vec::new();
        self.serialize_into(&mut ret)?;
        Ok(ret)
    }
}

trait Num: Sized {
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError>;
    /// Panics if slice isn't valid utf8
    fn from_slice_radix(s: &[u8], radix: u32) -> Result<Self, ParseIntError> {
        Self::from_str_radix(str::from_utf8(s).unwrap(), radix)
    }

    fn from_slice_bin(s: &[u8]) -> Result<Self, TryFromSliceError>;

    fn parse_bin(input: &[u8]) -> IResult<&[u8], Self> {
        map_res(take(mem::size_of::<Self>()), Self::from_slice_bin)(input)
    }
}

impl Num for usize {
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError> {
        Self::from_str_radix(s, radix)
    }

    fn from_slice_bin(s: &[u8]) -> Result<Self, TryFromSliceError> {
        Ok(Self::from_le_bytes(s.try_into()?))
    }
}

impl Num for u64 {
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError> {
        Self::from_str_radix(s, radix)
    }

    fn from_slice_bin(s: &[u8]) -> Result<Self, TryFromSliceError> {
        Ok(Self::from_le_bytes(s.try_into()?))
    }
}

impl Num for u32 {
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError> {
        Self::from_str_radix(s, radix)
    }

    fn from_slice_bin(s: &[u8]) -> Result<Self, TryFromSliceError> {
        Ok(Self::from_le_bytes(s.try_into()?))
    }
}

impl Num for u16 {
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError> {
        Self::from_str_radix(s, radix)
    }

    fn from_slice_bin(s: &[u8]) -> Result<Self, TryFromSliceError> {
        Ok(Self::from_le_bytes(s.try_into()?))
    }
}

impl Num for u8 {
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError> {
        Self::from_str_radix(s, radix)
    }

    fn from_slice_bin(s: &[u8]) -> Result<Self, TryFromSliceError> {
        Ok(Self::from_le_bytes(s.try_into()?))
    }
}

fn hex_primary<T: Num>(input: &[u8]) -> IResult<&[u8], T> {
    let (input, _) = tag("0x")(input)?;
    map_res(take_while1(|c: u8| c.is_hex_digit()), |s| {
        Num::from_slice_radix(s, 16)
    })(input)
}

fn dec_primary<T: Num + fmt::Debug>(input: &[u8]) -> IResult<&[u8], T> {
    map_res(take_while1(|c: u8| c.is_dec_digit()), |s| {
        Num::from_slice_radix(s, 10)
    })(input)
}

fn parse_num<T: Num + fmt::Debug>(input: &[u8]) -> IResult<&[u8], T> {
    alt((hex_primary, dec_primary))(input)
}

/// Returns everything before the tag (Consumes the tag)
fn take_str_until<'a>(tag: &'a [u8]) -> impl Fn(&[u8]) -> IResult<&[u8], &str> + 'a {
    move |input| {
        let (input, ret) = map_res(take_until(tag), str::from_utf8)(input)?;
        let (input, _) = nom::bytes::complete::tag(tag)(input)?;
        Ok((input, ret))
    }
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
    let (input, k) = alt((
        tag(b"data-file-overflow="),
        tag(b"clock="),
        tag(b"elapsed-time-usec="),
        tag(b"num-method-calls="),
        tag(b"clock-call-overhead-nsec="),
        tag(b"vm="),
        tag(b"pid="),
        tag(b"alloc-count="),
        tag(b"alloc-size="),
        tag(b"gc-count="),
    ))(input)?;
    Ok(match k {
        b"data-file-overflow=" => {
            let (input, v) = match alt((tag(b"true\n"), tag(b"false\n")))(input)? {
                (input, b"true\n") => (input, true),
                (input, b"false\n") => (input, false),
                _ => unreachable!(),
            };
            (input, HeaderLine::DataFileOverflow(v))
        }
        b"clock=" => {
            let (input, v) = match alt((
                tag(b"global\n"),
                tag(b"thread-cpu\n"),
                tag(b"wall\n"),
                tag(b"dual\n"),
            ))(input)?
            {
                (input, b"global\n") => (input, Clock::Global),
                (input, b"thread-cpu\n") => (input, Clock::Cpu),
                (input, b"wall\n") => (input, Clock::Wall),
                (input, b"dual\n") => (input, Clock::Dual),
                _ => unreachable!(),
            };
            (input, HeaderLine::Clock(v))
        }
        b"elapsed-time-usec=" => {
            let (input, v) = parse_num(input)?;
            let (input, _) = tag(b"\n")(input)?;
            (input, HeaderLine::ElapsedTime(Duration::from_micros(v)))
        }
        b"num-method-calls=" => {
            let (input, v) = parse_num(input)?;
            let (input, _) = tag(b"\n")(input)?;
            (input, HeaderLine::NumMethodCalls(v))
        }
        b"clock-call-overhead-nsec=" => {
            let (input, v) = parse_num(input)?;
            let (input, _) = tag(b"\n")(input)?;
            (
                input,
                HeaderLine::ClockCallOverhead(Duration::from_nanos(v)),
            )
        }
        b"vm=" => {
            let (input, v) = match alt((tag(b"dalvik\n"), tag(b"art\n")))(input)? {
                (input, b"dalvik\n") => (input, Vm::Dalvik),
                (input, b"art\n") => (input, Vm::Art),
                _ => unreachable!(),
            };
            (input, HeaderLine::Vm(v))
        }
        b"pid=" => {
            let (input, v) = parse_num(input)?;
            let (input, _) = tag(b"\n")(input)?;
            (input, HeaderLine::Pid(v))
        }
        b"alloc-count=" => {
            let (input, v) = parse_num(input)?;
            let (input, _) = tag(b"\n")(input)?;
            (input, HeaderLine::AllocCount(v))
        }
        b"alloc-size=" => {
            let (input, v) = parse_num(input)?;
            let (input, _) = tag(b"\n")(input)?;
            (input, HeaderLine::AllocSize(v))
        }
        b"gc-count=" => {
            let (input, v) = parse_num(input)?;
            let (input, _) = tag(b"\n")(input)?;
            (input, HeaderLine::GcCount(v))
        }
        _ => unreachable!(),
    })
}

fn parse_thread(input: &[u8]) -> IResult<&[u8], Thread> {
    let (input, id) = parse_num::<u16>(input)?;
    let (input, _) = tag(b"\t")(input)?;
    let (input, name) = take_str_until(b"\n")(input)?;
    Ok((
        input,
        Thread {
            id,
            name: name.into(),
        },
    ))
}

fn parse_method(input: &[u8]) -> IResult<&[u8], Method> {
    let (input, id) = parse_num(input)?;
    let (input, _) = tag(b"\t")(input)?;
    let (input, class_name) = take_str_until(b"\t")(input)?;
    let (input, name) = take_str_until(b"\t")(input)?;
    let (input, signature) = take_str_until(b"\t")(input)?;
    let (input, source_file) =
        map_res(take_till(|c| c == b'\t' || c == b'\n'), str::from_utf8)(input)?;

    let (input, source_line) =
        if let Ok((input, _)) = tag::<_, _, (&[u8], ErrorKind)>(b"\t\n")(input) {
            (input, None)
        } else if let Ok((input, _)) = tag::<_, _, (&[u8], ErrorKind)>(b"\t")(input) {
            let (input, num) = parse_num(input)?;
            (input, Some(num))
        } else {
            (input, None)
        };
    let (input, _) = tag(b"\n")(input)?;
    Ok((
        input,
        Method {
            id,
            class_name: class_name.into(),
            signature: signature.into(),
            name: name.into(),
            source_file: source_file.into(),
            source_line,
        },
    ))
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
            Version::One => {
                let (input, thread_id) = u8::parse_bin(input)?;
                (input, thread_id as _)
            }
            _ => u16::parse_bin(input)?,
        };
        let (input, (action, method_id)) = map_res(u32::parse_bin, |x| {
            let action = match x & 0b11 {
                0 => Action::Enter,
                1 => Action::Exit,
                2 => Action::Unwind,
                _ => return Err(Error::InvalidEventAction),
            };
            Ok((action, x >> 2))
        })(input)?;
        let (input, time_a) = u32::parse_bin(input)?;
        let time_a = Duration::from_micros(time_a as _);
        let (mut input, time) = match clock {
            Clock::Dual => {
                let (input, time_b) = u32::parse_bin(input)?;
                (
                    input,
                    Time::Monotonic {
                        cpu: Some(time_a),
                        wall: Some(Duration::from_micros(time_b as _)),
                    },
                )
            }
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
            let bytes_read = match version {
                Version::One => 1,
                _ => 2,
            } + 4
                + 4
                + match clock {
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum Section {
    Version,
    Threads,
    Methods,
    End,
}

fn parse_section(input: &[u8]) -> IResult<&[u8], Section> {
    let (input, _) = tag(b"*")(input)?;
    match alt((
        tag(b"version\n"),
        tag(b"threads\n"),
        tag(b"methods\n"),
        tag(b"end\n"),
    ))(input)
    {
        Ok((input, b"version\n")) => Ok((input, Section::Version)),
        Ok((input, b"threads\n")) => Ok((input, Section::Threads)),
        Ok((input, b"methods\n")) => Ok((input, Section::Methods)),
        Ok((input, b"end\n")) => Ok((input, Section::End)),
        Err(err) => Err(err),
        _ => unreachable!(),
    }
}

fn parse_android_trace<'a>(input: &'a [u8]) -> IResult<&'a [u8], AndroidTrace> {
    let mut ret = AndroidTrace::default();
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
                    |input| {
                        let (input, section) = parse_section(input)?;
                        Ok((input, Line::NewSection(section)))
                    },
                    |input| {
                        let (input, line) = parse_header_line(input)?;
                        Ok((input, Line::Info(line)))
                    },
                ))(input)?;
                input = inp;
                use HeaderLine::*;
                match line {
                    Line::NewSection(sect) => section = sect,
                    Line::Info(DataFileOverflow(v)) => ret.data_file_overflow = v,
                    Line::Info(Clock(v)) => ret.clock = v,
                    Line::Info(ElapsedTime(v)) => ret.elapsed_time = v,
                    Line::Info(NumMethodCalls(v)) => ret.num_method_calls = v,
                    Line::Info(ClockCallOverhead(v)) => ret.clock_call_overhead = v,
                    Line::Info(Vm(v)) => ret.vm = v,
                    Line::Info(Pid(v)) => ret.pid = Some(v),
                    Line::Info(AllocCount(v)) => {
                        ret.gc_trace
                            .get_or_insert_with(GcTrace::default)
                            .alloc_count = v
                    }
                    Line::Info(AllocSize(v)) => {
                        ret.gc_trace.get_or_insert_with(GcTrace::default).alloc_size = v
                    }
                    Line::Info(GcCount(v)) => {
                        ret.gc_trace.get_or_insert_with(GcTrace::default).gc_count = v
                    }
                }
            }
            Section::Threads => {
                enum Line {
                    NewSection(Section),
                    Thread(Thread),
                }
                let (inp, line) = alt((
                    |input| {
                        let (input, section) = parse_section(input)?;
                        Ok((input, Line::NewSection(section)))
                    },
                    |input| {
                        let (input, thread) = parse_thread(input)?;
                        Ok((input, Line::Thread(thread)))
                    },
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
                    |input| {
                        let (input, section) = parse_section(input)?;
                        Ok((input, Line::NewSection(section)))
                    },
                    |input| {
                        let (input, method) = parse_method(input)?;
                        Ok((input, Line::Method(method)))
                    },
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

    let (input, bin_version) = map_res(u16::parse_bin, |x| match x {
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

    let (input, data_offset) = u16::parse_bin(input)?;

    let (input, start_time) = u64::parse_bin(input)?;

    ret.start_time = chrono::DateTime::from_utc(
        chrono::NaiveDateTime::from_timestamp(
            (start_time / 1000000) as _,
            ((start_time % 1000000) * 1000) as _,
        ),
        chrono::Utc,
    );

    let (input, bin_version, header_len) = match bin_version {
        Version::Three(_) => {
            let (input, event_size) = u16::parse_bin(input)?;
            (
                input,
                Version::Three(NonZeroU64::new(event_size as _)),
                4 + 2 + 2 + 8 + 2,
            )
        }
        bin_version => (input, bin_version, 4 + 2 + 2 + 8),
    };

    let mut input = if header_len < data_offset {
        take(data_offset - header_len)(input)?.0
    } else {
        input
    };

    let entry_len = match bin_version {
        Version::Three(Some(len)) => len.get(),
        _ => 9,
    };

    let parser = event_parser(bin_version, ret.clock);
    ret.events
        .reserve(((input.len() as u64) / entry_len) as usize);
    while let Ok((inp, event)) = parser(input) {
        input = inp;
        ret.events.push(event);
    }

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
/// let trace = android_trace_parser::parse(&trace_bytes[..])?;
/// # let _ = trace;
/// # Ok(())
/// # }
/// ```
pub fn parse(bytes: &[u8]) -> Result<AndroidTrace, Box<dyn error::Error + '_>> {
    match parse_android_trace(bytes) {
        Ok((_, trace)) => Ok(trace),
        Err(err) => Err(Box::new(err)),
    }
}

// TODO tests