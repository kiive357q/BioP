//! 影子模式模块
//! 
//! 【功能】
//! - 影子模式运行: AI 控制 vs 人工控制对比
//! - 安全验证
//! - 故障注入测试

use crate::{SafetyThresholds, ControlAction, SensorData};
use chrono::Utc;

pub struct ShadowMode {
    thresholds: SafetyThresholds,
    safety_violations: u64,
    total_predictions: u64,
}

impl ShadowMode {
    pub fn new(thresholds: SafetyThresholds) -> Self {
        Self {
            thresholds,
            safety_violations: 0,
            total_predictions: 0,
        }
    }
    
    pub fn validate_action(
        &mut self,
        action: &ControlAction,
        sensor_data: &SensorData
    ) -> (bool, Vec<String>) {
        self.total_predictions += 1;
        
        let mut violations = Vec::new();
        let mut is_safe = true;
        
        if sensor_data.dissolved_oxygen < self.thresholds.do_min_mg_l {
            violations.push(format!(
                "DO 低于安全阈值: {:.2} < {:.2}",
                sensor_data.dissolved_oxygen,
                self.thresholds.do_min_mg_l
            ));
            is_safe = false;
            self.safety_violations += 1;
        }
        
        if sensor_data.dissolved_oxygen > self.thresholds.do_max_mg_l {
            violations.push(format!(
                "DO 高于安全阈值: {:.2} > {:.2}",
                sensor_data.dissolved_oxygen,
                self.thresholds.do_max_mg_l
            ));
        }
        
        if sensor_data.total_phosphorus > self.thresholds.tp_max_mg_l {
            violations.push(format!(
                "TP 超标: {:.2} > {:.2}",
                sensor_data.total_phosphorus,
                self.thresholds.tp_max_mg_l
            ));
            self.safety_violations += 1;
        }
        
        if !violations.is_empty() {
            log::warn!("[SHADOW] 安全违规检测: {:?}", violations);
        }
        
        (is_safe, violations)
    }
    
    pub fn get_safety_stats(&self) -> SafetyStats {
        SafetyStats {
            total_predictions: self.total_predictions,
            safety_violations: self.safety_violations,
            violation_rate: if self.total_predictions > 0 {
                self.safety_violations as f64 / self.total_predictions as f64
            } else {
                0.0
            },
        }
    }
    
    pub fn reset_stats(&mut self) {
        self.total_predictions = 0;
        self.safety_violations = 0;
    }
}

#[derive(Debug, Clone)]
pub struct SafetyStats {
    pub total_predictions: u64,
    pub safety_violations: u64,
    pub violation_rate: f64,
}
