//! 看门狗模块
//! 
//! 【功能】
//! - 心跳检测
//! - 故障恢复
//! - 超时告警

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct Watchdog {
    timeout_ms: u64,
    last_feed: Arc<AtomicU64>,
    feed_count: Arc<AtomicU64>,
}

impl Watchdog {
    pub fn new(timeout_ms: u64) -> Self {
        let now = Utc::now().timestamp_millis() as u64;
        
        Self {
            timeout_ms,
            last_feed: Arc::new(AtomicU64::new(now)),
            feed_count: Arc::new(AtomicU64::new(0)),
        }
    }
    
    pub fn feed(&self) {
        let now = Utc::now().timestamp_millis() as u64;
        self.last_feed.store(now, Ordering::SeqCst);
        self.feed_count.fetch_add(1, Ordering::SeqCst);
    }
    
    pub fn is_alive(&self) -> bool {
        let now = Utc::now().timestamp_millis() as u64;
        let last = self.last_feed.load(Ordering::SeqCst);
        
        (now - last) < self.timeout_ms
    }
    
    pub fn get_status(&self) -> WatchdogStatus {
        let now = Utc::now().timestamp_millis() as u64;
        let last = self.last_feed.load(Ordering::SeqCst);
        let count = self.feed_count.load(Ordering::SeqCst);
        
        WatchdogStatus {
            is_alive: self.is_alive(),
            time_since_last_feed_ms: now.saturating_sub(last),
            timeout_ms: self.timeout_ms,
            feed_count: count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WatchdogStatus {
    pub is_alive: bool,
    pub time_since_last_feed_ms: u64,
    pub timeout_ms: u64,
    pub feed_count: u64,
}

impl std::fmt::Display for WatchdogStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Watchdog[alive={}, since_last={}ms, timeout={}ms, feeds={}]",
            self.is_alive, self.time_since_last_feed_ms, self.timeout_ms, self.feed_count)
    }
}
