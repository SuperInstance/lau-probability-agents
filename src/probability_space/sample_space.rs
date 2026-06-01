//! Sample space Ω and events

use serde::{Serialize, Deserialize};
use std::fmt;

/// A sample point ω ∈ Ω
pub type SamplePoint = f64;

/// An event is a subset of the sample space, represented as a set of sample points
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    /// For discrete: explicit points. For continuous: interval [low, high]
    pub points: Vec<SamplePoint>,
    /// If true, treat as continuous interval [low, high]
    pub is_interval: bool,
    pub low: f64,
    pub high: f64,
}

impl Event {
    /// Create a discrete event from explicit points
    pub fn discrete(points: Vec<SamplePoint>) -> Self {
        Event { points, is_interval: false, low: 0.0, high: 0.0 }
    }

    /// Create a continuous interval event [low, high]
    pub fn interval(low: f64, high: f64) -> Self {
        Event { points: vec![], is_interval: true, low, high }
    }

    /// The empty event ∅
    pub fn empty() -> Self {
        Event::discrete(vec![])
    }

    /// The sure event Ω (entire real line)
    pub fn sure() -> Self {
        Event::interval(f64::NEG_INFINITY, f64::INFINITY)
    }

    /// Union of two events
    pub fn union(&self, other: &Event) -> Event {
        if self.is_interval && other.is_interval {
            Event::interval(
                self.low.min(other.low),
                self.high.max(other.high),
            )
        } else {
            let mut pts: Vec<SamplePoint> = self.points.iter().chain(other.points.iter()).copied().collect();
            pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            pts.dedup();
            Event::discrete(pts)
        }
    }

    /// Intersection of two events
    pub fn intersection(&self, other: &Event) -> Event {
        if self.is_interval && other.is_interval {
            Event::interval(
                self.low.max(other.low),
                self.high.min(other.high),
            )
        } else {
            let other_set: std::collections::HashSet<SamplePoint> = other.points.iter().copied().collect();
            let pts: Vec<SamplePoint> = self.points.iter()
                .filter(|p| other_set.contains(p))
                .copied()
                .collect();
            Event::discrete(pts)
        }
    }

    /// Complement (against the sure event)
    pub fn complement(&self) -> Event {
        if self.is_interval {
            // Complement of [a,b] is (-∞, a) ∪ (b, ∞), simplified as two intervals
            // We represent as "not in [a,b]" which we handle specially
            Event::interval(self.high, f64::INFINITY) // approximate
        } else {
            // For discrete, we can't compute complement without knowing Ω
            Event::empty()
        }
    }

    /// Check if this event contains a sample point
    pub fn contains(&self, point: SamplePoint) -> bool {
        if self.is_interval {
            point >= self.low && point <= self.high
        } else {
            self.points.iter().any(|p| (*p - point).abs() < 1e-12)
        }
    }

    /// Is the event empty?
    pub fn is_empty(&self) -> bool {
        if self.is_interval {
            self.low > self.high
        } else {
            self.points.is_empty()
        }
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_interval {
            write!(f, "[{}, {}]", self.low, self.high)
        } else if self.points.is_empty() {
            write!(f, "∅")
        } else {
            write!(f, "{{{}}}", self.points.iter()
                .map(|p| format!("{:.4}", p))
                .collect::<Vec<_>>()
                .join(", "))
        }
    }
}

/// Sample space representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampleSpace {
    /// Finite sample space {ω₁, ..., ωₙ}
    Finite { outcomes: Vec<SamplePoint> },
    /// Countably infinite
    Countable { outcomes: Vec<SamplePoint> },
    /// Continuous (interval [a, b] or ℝ)
    Continuous { low: f64, high: f64 },
}

impl SampleSpace {
    /// Finite sample space
    pub fn finite(outcomes: Vec<SamplePoint>) -> Self {
        SampleSpace::Finite { outcomes }
    }

    /// Continuous uniform sample space [a, b]
    pub fn continuous(low: f64, high: f64) -> Self {
        SampleSpace::Continuous { low, high }
    }

    /// The entire real line
    pub fn real_line() -> Self {
        SampleSpace::Continuous { low: f64::NEG_INFINITY, high: f64::INFINITY }
    }

    /// Number of outcomes (None for continuous)
    pub fn cardinality(&self) -> Option<usize> {
        match self {
            SampleSpace::Finite { outcomes } => Some(outcomes.len()),
            SampleSpace::Countable { outcomes } => Some(outcomes.len()),
            SampleSpace::Continuous { .. } => None,
        }
    }
}
