# Module VI: Observability and Monitoring

## Learning Objectives

By the end of this module, you will be able to:

- Implement comprehensive observability for agentic AI systems
- Design monitoring strategies for multi-agent applications
- Build alerting and anomaly detection systems
- Create performance dashboards and metrics collection
- Implement distributed tracing for agent interactions
- Design logging strategies for debugging and analysis
- Build health checks and system reliability monitoring

## 1. Introduction to Agentic AI Observability

Observability in agentic AI systems goes beyond traditional application monitoring. It requires understanding agent behavior, decision-making processes, inter-agent communications, and the complex emergent behaviors that arise from multi-agent interactions.

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Agentic AI Observability Architecture</text>
  
  <!-- Agent Layer -->
  <rect x="50" y="80" width="700" height="120" fill="#e3f2fd" stroke="#1976d2" stroke-width="2" rx="10"/>
  <text x="400" y="100" text-anchor="middle" font-size="14" font-weight="bold" fill="#1976d2">Agent Layer</text>
  
  <!-- Individual Agents -->
  <circle cx="150" cy="140" r="25" fill="#42a5f5" stroke="#1976d2" stroke-width="2"/>
  <text x="150" y="145" text-anchor="middle" font-size="10" fill="white">Agent A</text>
  
  <circle cx="300" cy="140" r="25" fill="#42a5f5" stroke="#1976d2" stroke-width="2"/>
  <text x="300" y="145" text-anchor="middle" font-size="10" fill="white">Agent B</text>
  
  <circle cx="450" cy="140" r="25" fill="#42a5f5" stroke="#1976d2" stroke-width="2"/>
  <text x="450" y="145" text-anchor="middle" font-size="10" fill="white">Agent C</text>
  
  <circle cx="600" cy="140" r="25" fill="#42a5f5" stroke="#1976d2" stroke-width="2"/>
  <text x="600" y="145" text-anchor="middle" font-size="10" fill="white">Agent D</text>
  
  <!-- Communication Lines -->
  <line x1="175" y1="140" x2="275" y2="140" stroke="#666" stroke-width="2" stroke-dasharray="5,5"/>
  <line x1="325" y1="140" x2="425" y2="140" stroke="#666" stroke-width="2" stroke-dasharray="5,5"/>
  <line x1="475" y1="140" x2="575" y2="140" stroke="#666" stroke-width="2" stroke-dasharray="5,5"/>
  
  <!-- Observability Layer -->
  <rect x="50" y="230" width="700" height="280" fill="#fff3e0" stroke="#f57c00" stroke-width="2" rx="10"/>
  <text x="400" y="250" text-anchor="middle" font-size="14" font-weight="bold" fill="#f57c00">Observability Layer</text>
  
  <!-- Metrics Collection -->
  <rect x="80" y="270" width="150" height="80" fill="#ffcc02" stroke="#f57c00" stroke-width="1" rx="5"/>
  <text x="155" y="290" text-anchor="middle" font-size="12" font-weight="bold" fill="#e65100">Metrics</text>
  <text x="155" y="305" text-anchor="middle" font-size="10" fill="#e65100">• Performance</text>
  <text x="155" y="318" text-anchor="middle" font-size="10" fill="#e65100">• Resource Usage</text>
  <text x="155" y="331" text-anchor="middle" font-size="10" fill="#e65100">• Success Rates</text>
  
  <!-- Logging -->
  <rect x="250" y="270" width="150" height="80" fill="#ffcc02" stroke="#f57c00" stroke-width="1" rx="5"/>
  <text x="325" y="290" text-anchor="middle" font-size="12" font-weight="bold" fill="#e65100">Logging</text>
  <text x="325" y="305" text-anchor="middle" font-size="10" fill="#e65100">• Decision Logs</text>
  <text x="325" y="318" text-anchor="middle" font-size="10" fill="#e65100">• Error Tracking</text>
  <text x="325" y="331" text-anchor="middle" font-size="10" fill="#e65100">• Audit Trails</text>
  
  <!-- Tracing -->
  <rect x="420" y="270" width="150" height="80" fill="#ffcc02" stroke="#f57c00" stroke-width="1" rx="5"/>
  <text x="495" y="290" text-anchor="middle" font-size="12" font-weight="bold" fill="#e65100">Tracing</text>
  <text x="495" y="305" text-anchor="middle" font-size="10" fill="#e65100">• Request Flow</text>
  <text x="495" y="318" text-anchor="middle" font-size="10" fill="#e65100">• Dependencies</text>
  <text x="495" y="331" text-anchor="middle" font-size="10" fill="#e65100">• Latency</text>
  
  <!-- Health Checks -->
  <rect x="590" y="270" width="130" height="80" fill="#ffcc02" stroke="#f57c00" stroke-width="1" rx="5"/>
  <text x="655" y="290" text-anchor="middle" font-size="12" font-weight="bold" fill="#e65100">Health</text>
  <text x="655" y="305" text-anchor="middle" font-size="10" fill="#e65100">• Liveness</text>
  <text x="655" y="318" text-anchor="middle" font-size="10" fill="#e65100">• Readiness</text>
  <text x="655" y="331" text-anchor="middle" font-size="10" fill="#e65100">• Dependencies</text>
  
  <!-- Analysis Layer -->
  <rect x="80" y="370" width="200" height="80" fill="#c8e6c9" stroke="#388e3c" stroke-width="1" rx="5"/>
  <text x="180" y="390" text-anchor="middle" font-size="12" font-weight="bold" fill="#1b5e20">Analytics</text>
  <text x="180" y="405" text-anchor="middle" font-size="10" fill="#1b5e20">• Pattern Detection</text>
  <text x="180" y="418" text-anchor="middle" font-size="10" fill="#1b5e20">• Anomaly Detection</text>
  <text x="180" y="431" text-anchor="middle" font-size="10" fill="#1b5e20">• Trend Analysis</text>
  
  <!-- Alerting -->
  <rect x="300" y="370" width="200" height="80" fill="#ffcdd2" stroke="#d32f2f" stroke-width="1" rx="5"/>
  <text x="400" y="390" text-anchor="middle" font-size="12" font-weight="bold" fill="#b71c1c">Alerting</text>
  <text x="400" y="405" text-anchor="middle" font-size="10" fill="#b71c1c">• Threshold Alerts</text>
  <text x="400" y="418" text-anchor="middle" font-size="10" fill="#b71c1c">• Anomaly Alerts</text>
  <text x="400" y="431" text-anchor="middle" font-size="10" fill="#b71c1c">• Escalation</text>
  
  <!-- Dashboards -->
  <rect x="520" y="370" width="200" height="80" fill="#e1bee7" stroke="#7b1fa2" stroke-width="1" rx="5"/>
  <text x="620" y="390" text-anchor="middle" font-size="12" font-weight="bold" fill="#4a148c">Dashboards</text>
  <text x="620" y="405" text-anchor="middle" font-size="10" fill="#4a148c">• Real-time Views</text>
  <text x="620" y="418" text-anchor="middle" font-size="10" fill="#4a148c">• Historical Data</text>
  <text x="620" y="431" text-anchor="middle" font-size="10" fill="#4a148c">• Custom Views</text>
  
  <!-- Data Flow Arrows -->
  <path d="M 150 165 L 155 280" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 300 165 L 325 280" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 450 165 L 495 280" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 600 165 L 655 280" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <path d="M 155 350 L 180 370" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 325 350 L 400 370" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 495 350 L 620 370" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Arrow marker definition -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
  
  <!-- Key Features -->
  <text x="50" y="540" font-size="12" font-weight="bold" fill="#2c3e50">Key Features:</text>
  <text x="50" y="555" font-size="10" fill="#2c3e50">• Multi-dimensional monitoring across agents, workflows, and infrastructure</text>
  <text x="50" y="570" font-size="10" fill="#2c3e50">• Real-time observability with historical analysis capabilities</text>
  <text x="50" y="585" font-size="10" fill="#2c3e50">• Automated anomaly detection and intelligent alerting</text>
</svg>
```

### 1.1 Observability Pillars for Agentic AI

The three pillars of observability (metrics, logs, and traces) take on special meaning in agentic AI systems:

1. **Metrics**: Quantitative measurements of agent performance, resource usage, and system health
2. **Logs**: Detailed records of agent decisions, actions, and state changes
3. **Traces**: End-to-end tracking of requests and workflows across multiple agents

### 1.2 Unique Challenges in Agentic AI Observability

- **Non-deterministic Behavior**: Agents may make different decisions given the same inputs
- **Emergent Behaviors**: Complex behaviors arising from agent interactions
- **Dynamic Topologies**: Agent networks that change over time
- **Context Dependency**: Agent behavior heavily dependent on context and history
- **Multi-modal Interactions**: Agents communicating through various channels and protocols

## 2. Metrics Collection and Monitoring

### 2.1 Core Metrics Framework

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import time
import threading
import statistics
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, deque

class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"      # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"      # Duration measurements
    RATE = "rate"        # Events per unit time

class MetricLevel(Enum):
    """Importance levels for metrics"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricValue:
    """Represents a single metric measurement"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    level: MetricLevel = MetricLevel.INFO
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricSummary:
    """Summary statistics for a metric over time"""
    name: str
    count: int
    min_value: float
    max_value: float
    mean: float
    median: float
    std_dev: float
    percentiles: Dict[str, float]  # e.g., {"p95": 0.95, "p99": 0.99}
    time_window: timedelta
    last_updated: datetime

class MetricCollector(ABC):
    """Abstract base class for metric collectors"""
    
    @abstractmethod
    def collect(self, metric: MetricValue):
        """Collect a metric value"""
        pass
    
    @abstractmethod
    def get_metrics(self, name_pattern: str = None, 
                   time_range: tuple = None) -> List[MetricValue]:
        """Retrieve collected metrics"""
        pass
    
    @abstractmethod
    def get_summary(self, name: str, time_window: timedelta) -> Optional[MetricSummary]:
        """Get summary statistics for a metric"""
        pass

class InMemoryMetricCollector(MetricCollector):
    """In-memory metric collector with time-based retention"""
    
    def __init__(self, retention_period: timedelta = timedelta(hours=24)):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque())
        self.retention_period = retention_period
        self.lock = threading.RLock()
        self.logger = logging.getLogger("MetricCollector")
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def collect(self, metric: MetricValue):
        """Collect a metric value"""
        with self.lock:
            self.metrics[metric.name].append(metric)
            self.logger.debug(f"Collected metric: {metric.name} = {metric.value}")
    
    def get_metrics(self, name_pattern: str = None, 
                   time_range: tuple = None) -> List[MetricValue]:
        """Retrieve collected metrics"""
        with self.lock:
            results = []
            
            for name, metric_deque in self.metrics.items():
                if name_pattern and name_pattern not in name:
                    continue
                
                for metric in metric_deque:
                    if time_range:
                        start_time, end_time = time_range
                        if not (start_time <= metric.timestamp <= end_time):
                            continue
                    
                    results.append(metric)
            
            return sorted(results, key=lambda m: m.timestamp)
    
    def get_summary(self, name: str, time_window: timedelta) -> Optional[MetricSummary]:
        """Get summary statistics for a metric"""
        with self.lock:
            if name not in self.metrics:
                return None
            
            cutoff_time = datetime.now() - time_window
            values = [m.value for m in self.metrics[name] 
                     if m.timestamp >= cutoff_time]
            
            if not values:
                return None
            
            # Calculate statistics
            count = len(values)
            min_value = min(values)
            max_value = max(values)
            mean = statistics.mean(values)
            median = statistics.median(values)
            std_dev = statistics.stdev(values) if count > 1 else 0.0
            
            # Calculate percentiles
            percentiles = {}
            if count > 0:
                sorted_values = sorted(values)
                percentiles = {
                    "p50": sorted_values[int(0.5 * count)],
                    "p90": sorted_values[int(0.9 * count)],
                    "p95": sorted_values[int(0.95 * count)],
                    "p99": sorted_values[int(0.99 * count)] if count > 1 else sorted_values[0]
                }
            
            return MetricSummary(
                name=name,
                count=count,
                min_value=min_value,
                max_value=max_value,
                mean=mean,
                median=median,
                std_dev=std_dev,
                percentiles=percentiles,
                time_window=time_window,
                last_updated=datetime.now()
            )
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        with self.lock:
            cutoff_time = datetime.now() - self.retention_period
            
            for name, metric_deque in self.metrics.items():
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()
    
    def _start_cleanup_thread(self):
        """Start background thread for metric cleanup"""
        def cleanup_worker():
            while True:
                time.sleep(300)  # Cleanup every 5 minutes
                self._cleanup_old_metrics()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

class AgentMetrics:
    """Metrics collection specifically for agentic AI systems"""
    
    def __init__(self, collector: MetricCollector, agent_id: str):
        self.collector = collector
        self.agent_id = agent_id
        self.start_time = datetime.now()
        
        # Performance counters
        self.task_counter = 0
        self.success_counter = 0
        self.error_counter = 0
        
        # Timing measurements
        self.active_timers: Dict[str, float] = {}
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        metric = MetricValue(
            name=f"agent.{self.agent_id}.{name}",
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.COUNTER
        )
        self.collector.collect(metric)
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Set a gauge metric"""
        metric = MetricValue(
            name=f"agent.{self.agent_id}.{name}",
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.GAUGE
        )
        self.collector.collect(metric)
    
    def start_timer(self, name: str) -> str:
        """Start a timer for measuring duration"""
        timer_id = f"{name}_{int(time.time() * 1000)}"
        self.active_timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str, tags: Dict[str, str] = None) -> float:
        """End a timer and record the duration"""
        if timer_id not in self.active_timers:
            return 0.0
        
        duration = time.time() - self.active_timers[timer_id]
        del self.active_timers[timer_id]
        
        # Extract metric name from timer_id
        name = timer_id.rsplit('_', 1)[0]
        
        metric = MetricValue(
            name=f"agent.{self.agent_id}.{name}.duration",
            value=duration,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.TIMER
        )
        self.collector.collect(metric)
        
        return duration
    
    def record_task_execution(self, task_name: str, success: bool, 
                            duration: float, metadata: Dict[str, Any] = None):
        """Record metrics for task execution"""
        self.task_counter += 1
        
        if success:
            self.success_counter += 1
        else:
            self.error_counter += 1
        
        # Record task metrics
        self.increment_counter("tasks.total")
        self.increment_counter(f"tasks.{task_name}.total")
        
        if success:
            self.increment_counter("tasks.success")
            self.increment_counter(f"tasks.{task_name}.success")
        else:
            self.increment_counter("tasks.error")
            self.increment_counter(f"tasks.{task_name}.error")
        
        # Record duration
        self.set_gauge(f"tasks.{task_name}.duration", duration)
        
        # Record success rate
        success_rate = self.success_counter / self.task_counter if self.task_counter > 0 else 0
        self.set_gauge("tasks.success_rate", success_rate)
    
    def record_resource_usage(self, cpu_percent: float, memory_mb: float, 
                            network_bytes: int = 0):
        """Record resource usage metrics"""
        self.set_gauge("resources.cpu_percent", cpu_percent)
        self.set_gauge("resources.memory_mb", memory_mb)
        if network_bytes > 0:
            self.set_gauge("resources.network_bytes", network_bytes)
    
    def record_decision_metrics(self, decision_type: str, confidence: float, 
                              options_considered: int, decision_time: float):
        """Record metrics about agent decision-making"""
        self.set_gauge(f"decisions.{decision_type}.confidence", confidence)
        self.set_gauge(f"decisions.{decision_type}.options_considered", options_considered)
        self.set_gauge(f"decisions.{decision_type}.decision_time", decision_time)
        self.increment_counter(f"decisions.{decision_type}.total")
    
    def record_communication_metrics(self, target_agent: str, message_type: str, 
                                   message_size: int, success: bool):
        """Record metrics about inter-agent communication"""
        tags = {"target_agent": target_agent, "message_type": message_type}
        
        self.increment_counter("communication.messages.total", tags=tags)
        self.set_gauge("communication.message_size", message_size, tags=tags)
        
        if success:
            self.increment_counter("communication.messages.success", tags=tags)
        else:
            self.increment_counter("communication.messages.error", tags=tags)
    
    def get_agent_summary(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get summary of agent metrics"""
        summaries = {}
        
        # Get summaries for key metrics
        key_metrics = [
            "tasks.total", "tasks.success", "tasks.error", "tasks.success_rate",
            "resources.cpu_percent", "resources.memory_mb",
            "communication.messages.total"
        ]
        
        for metric_name in key_metrics:
            full_name = f"agent.{self.agent_id}.{metric_name}"
            summary = self.collector.get_summary(full_name, time_window)
            if summary:
                summaries[metric_name] = summary
        
        return {
            "agent_id": self.agent_id,
            "uptime": datetime.now() - self.start_time,
            "metrics": summaries,
            "counters": {
                "total_tasks": self.task_counter,
                "successful_tasks": self.success_counter,
                "failed_tasks": self.error_counter
            }
        }
```

### 2.2 System-wide Metrics Aggregation

```python
class SystemMetricsAggregator:
    """Aggregates metrics across multiple agents and system components"""
    
    def __init__(self, collector: MetricCollector):
        self.collector = collector
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.system_start_time = datetime.now()
    
    def register_agent(self, agent_id: str) -> AgentMetrics:
        """Register a new agent for metrics collection"""
        agent_metrics = AgentMetrics(self.collector, agent_id)
        self.agent_metrics[agent_id] = agent_metrics
        return agent_metrics
    
    def get_system_overview(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get system-wide metrics overview"""
        overview = {
            "system_uptime": datetime.now() - self.system_start_time,
            "total_agents": len(self.agent_metrics),
            "time_window": str(time_window),
            "agents": {},
            "system_totals": {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "total_messages": 0
            }
        }
        
        # Collect metrics from all agents
        for agent_id, agent_metrics in self.agent_metrics.items():
            agent_summary = agent_metrics.get_agent_summary(time_window)
            overview["agents"][agent_id] = agent_summary
            
            # Aggregate system totals
            overview["system_totals"]["total_tasks"] += agent_metrics.task_counter
            overview["system_totals"]["successful_tasks"] += agent_metrics.success_counter
            overview["system_totals"]["failed_tasks"] += agent_metrics.error_counter
        
        # Calculate system-wide success rate
        total_tasks = overview["system_totals"]["total_tasks"]
        if total_tasks > 0:
            overview["system_totals"]["success_rate"] = (
                overview["system_totals"]["successful_tasks"] / total_tasks
            )
        else:
            overview["system_totals"]["success_rate"] = 0.0
        
        return overview
    
    def get_performance_trends(self, metric_name: str, 
                             time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Analyze performance trends for a specific metric"""
        end_time = datetime.now()
        start_time = end_time - time_window
        
        # Get all metrics for the specified name pattern
        metrics = self.collector.get_metrics(
            name_pattern=metric_name,
            time_range=(start_time, end_time)
        )
        
        if not metrics:
            return {"error": f"No metrics found for pattern: {metric_name}"}
        
        # Group metrics by hour for trend analysis
        hourly_data = defaultdict(list)
        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[hour_key].append(metric.value)
        
        # Calculate hourly averages
        trend_data = []
        for hour, values in sorted(hourly_data.items()):
            trend_data.append({
                "timestamp": hour,
                "average": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            })
        
        # Calculate overall trend
        if len(trend_data) >= 2:
            first_avg = trend_data[0]["average"]
            last_avg = trend_data[-1]["average"]
            trend_direction = "increasing" if last_avg > first_avg else "decreasing"
            trend_magnitude = abs(last_avg - first_avg) / first_avg if first_avg != 0 else 0
        else:
            trend_direction = "stable"
            trend_magnitude = 0
        
        return {
            "metric_name": metric_name,
            "time_window": str(time_window),
            "data_points": len(metrics),
            "trend_direction": trend_direction,
            "trend_magnitude": trend_magnitude,
            "hourly_data": trend_data
        }
```

## 3. Logging and Audit Trails

### 3.1 Structured Logging for Agents

```python
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

class LogLevel(Enum):
    """Log levels for agent activities"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventType(Enum):
    """Types of events that can be logged"""
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"
    DECISION_MADE = "decision_made"
    COMMUNICATION_SENT = "communication_sent"
    COMMUNICATION_RECEIVED = "communication_received"
    STATE_CHANGE = "state_change"
    RESOURCE_ACCESS = "resource_access"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_ALERT = "performance_alert"

@dataclass
class AgentLogEntry:
    """Structured log entry for agent activities"""
    timestamp: datetime
    agent_id: str
    event_type: EventType
    level: LogLevel
    message: str
    context: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        return data
    
    def to_json(self) -> str:
        """Convert log entry to JSON string"""
        return json.dumps(self.to_dict())

class AgentLogger:
    """Specialized logger for agentic AI systems"""
    
    def __init__(self, agent_id: str, logger_name: str = None):
        self.agent_id = agent_id
        self.logger = logging.getLogger(logger_name or f"agent.{agent_id}")
        self.session_id: Optional[str] = None
        self.trace_id: Optional[str] = None
        self.span_id: Optional[str] = None
        
        # Configure structured logging
        self._setup_structured_logging()
    
    def _setup_structured_logging(self):
        """Setup structured logging format"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def set_context(self, session_id: str = None, trace_id: str = None, 
                   span_id: str = None):
        """Set logging context for correlation"""
        if session_id:
            self.session_id = session_id
        if trace_id:
            self.trace_id = trace_id
        if span_id:
            self.span_id = span_id
    
    def log_event(self, event_type: EventType, message: str, 
                 level: LogLevel = LogLevel.INFO, 
                 context: Dict[str, Any] = None,
                 user_id: str = None):
        """Log a structured event"""
        log_entry = AgentLogEntry(
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            event_type=event_type,
            level=level,
            message=message,
            context=context or {},
            trace_id=self.trace_id,
            span_id=self.span_id,
            session_id=self.session_id,
            user_id=user_id
        )
        
        # Log using standard logger
        log_level = getattr(logging, level.value)
        self.logger.log(log_level, log_entry.to_json())
        
        return log_entry
    
    def log_task_start(self, task_name: str, task_params: Dict[str, Any] = None):
        """Log task start event"""
        context = {
            "task_name": task_name,
            "task_params": task_params or {}
        }
        return self.log_event(
            EventType.TASK_START,
            f"Starting task: {task_name}",
            LogLevel.INFO,
            context
        )
    
    def log_task_complete(self, task_name: str, result: Any = None, 
                         duration: float = None):
        """Log task completion event"""
        context = {
            "task_name": task_name,
            "result": str(result) if result is not None else None,
            "duration_seconds": duration
        }
        return self.log_event(
            EventType.TASK_COMPLETE,
            f"Completed task: {task_name}",
            LogLevel.INFO,
            context
        )
    
    def log_task_error(self, task_name: str, error: Exception, 
                      context_data: Dict[str, Any] = None):
        """Log task error event"""
        context = {
            "task_name": task_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "additional_context": context_data or {}
        }
        return self.log_event(
            EventType.TASK_ERROR,
            f"Task failed: {task_name} - {str(error)}",
            LogLevel.ERROR,
            context
        )
    
    def log_decision(self, decision_type: str, decision: str, 
                    confidence: float, reasoning: str = None,
                    alternatives: List[str] = None):
        """Log agent decision-making event"""
        context = {
            "decision_type": decision_type,
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "alternatives_considered": alternatives or []
        }
        return self.log_event(
            EventType.DECISION_MADE,
            f"Decision made: {decision_type} -> {decision}",
            LogLevel.INFO,
            context
        )
    
    def log_communication(self, direction: str, target_agent: str, 
                         message_type: str, message_content: Any = None,
                         success: bool = True):
        """Log inter-agent communication event"""
        event_type = (EventType.COMMUNICATION_SENT if direction == "sent" 
                     else EventType.COMMUNICATION_RECEIVED)
        
        context = {
            "direction": direction,
            "target_agent": target_agent,
            "message_type": message_type,
            "message_content": str(message_content) if message_content else None,
            "success": success
        }
        
        level = LogLevel.INFO if success else LogLevel.WARNING
        message = f"Communication {direction}: {message_type} {'to' if direction == 'sent' else 'from'} {target_agent}"
        
        return self.log_event(event_type, message, level, context)
    
    def log_state_change(self, old_state: str, new_state: str, 
                        trigger: str = None, metadata: Dict[str, Any] = None):
        """Log agent state change event"""
        context = {
            "old_state": old_state,
            "new_state": new_state,
            "trigger": trigger,
            "metadata": metadata or {}
        }
        return self.log_event(
            EventType.STATE_CHANGE,
            f"State changed: {old_state} -> {new_state}",
            LogLevel.INFO,
            context
        )
    
    def log_security_event(self, event_description: str, severity: str,
                          details: Dict[str, Any] = None):
        """Log security-related event"""
        context = {
            "security_event": event_description,
            "severity": severity,
            "details": details or {}
        }
        
        level = LogLevel.CRITICAL if severity == "high" else LogLevel.WARNING
        return self.log_event(
            EventType.SECURITY_EVENT,
            f"Security event: {event_description}",
            level,
            context
        )
```

### 3.2 Audit Trail System

```python
from typing import List, Iterator
from dataclasses import dataclass
import sqlite3
import threading
from contextlib import contextmanager

@dataclass
class AuditQuery:
    """Query parameters for audit trail search"""
    agent_ids: List[str] = None
    event_types: List[EventType] = None
    start_time: datetime = None
    end_time: datetime = None
    user_ids: List[str] = None
    session_ids: List[str] = None
    trace_ids: List[str] = None
    log_levels: List[LogLevel] = None
    limit: int = 1000
    offset: int = 0

class AuditTrailStore:
    """Persistent storage for audit trails"""
    
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the audit trail database"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context TEXT,
                    trace_id TEXT,
                    span_id TEXT,
                    parent_span_id TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON audit_logs(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_logs(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_id ON audit_logs(trace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON audit_logs(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_logs(user_id)")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def store_log_entry(self, log_entry: AgentLogEntry):
        """Store a log entry in the audit trail"""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO audit_logs (
                        timestamp, agent_id, event_type, level, message, context,
                        trace_id, span_id, parent_span_id, session_id, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_entry.timestamp.isoformat(),
                    log_entry.agent_id,
                    log_entry.event_type.value,
                    log_entry.level.value,
                    log_entry.message,
                    json.dumps(log_entry.context),
                    log_entry.trace_id,
                    log_entry.span_id,
                    log_entry.parent_span_id,
                    log_entry.session_id,
                    log_entry.user_id
                ))
    
    def query_audit_trail(self, query: AuditQuery) -> Iterator[AgentLogEntry]:
        """Query the audit trail with filtering"""
        with self.lock:
            with self._get_connection() as conn:
                # Build dynamic query
                where_clauses = []
                params = []
                
                if query.agent_ids:
                    placeholders = ','.join('?' * len(query.agent_ids))
                    where_clauses.append(f"agent_id IN ({placeholders})")
                    params.extend(query.agent_ids)
                
                if query.event_types:
                    placeholders = ','.join('?' * len(query.event_types))
                    where_clauses.append(f"event_type IN ({placeholders})")
                    params.extend([et.value for et in query.event_types])
                
                if query.start_time:
                    where_clauses.append("timestamp >= ?")
                    params.append(query.start_time.isoformat())
                
                if query.end_time:
                    where_clauses.append("timestamp <= ?")
                    params.append(query.end_time.isoformat())
                
                if query.user_ids:
                    placeholders = ','.join('?' * len(query.user_ids))
                    where_clauses.append(f"user_id IN ({placeholders})")
                    params.extend(query.user_ids)
                
                if query.session_ids:
                    placeholders = ','.join('?' * len(query.session_ids))
                    where_clauses.append(f"session_id IN ({placeholders})")
                    params.extend(query.session_ids)
                
                if query.trace_ids:
                    placeholders = ','.join('?' * len(query.trace_ids))
                    where_clauses.append(f"trace_id IN ({placeholders})")
                    params.extend(query.trace_ids)
                
                if query.log_levels:
                    placeholders = ','.join('?' * len(query.log_levels))
                    where_clauses.append(f"level IN ({placeholders})")
                    params.extend([ll.value for ll in query.log_levels])
                
                # Construct final query
                base_query = "SELECT * FROM audit_logs"
                if where_clauses:
                    base_query += " WHERE " + " AND ".join(where_clauses)
                
                base_query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([query.limit, query.offset])
                
                # Execute query
                cursor = conn.execute(base_query, params)
                
                for row in cursor:
                    yield AgentLogEntry(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        agent_id=row['agent_id'],
                        event_type=EventType(row['event_type']),
                        level=LogLevel(row['level']),
                        message=row['message'],
                        context=json.loads(row['context']) if row['context'] else {},
                        trace_id=row['trace_id'],
                        span_id=row['span_id'],
                        parent_span_id=row['parent_span_id'],
                        session_id=row['session_id'],
                        user_id=row['user_id']
                    )
    
    def get_audit_summary(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Get summary statistics for audit trail"""
        end_time = datetime.now()
        start_time = end_time - time_window
        
        with self._get_connection() as conn:
            # Total events
            cursor = conn.execute(
                "SELECT COUNT(*) as total FROM audit_logs WHERE timestamp >= ? AND timestamp <= ?",
                (start_time.isoformat(), end_time.isoformat())
            )
            total_events = cursor.fetchone()['total']
            
            # Events by type
            cursor = conn.execute("""
                SELECT event_type, COUNT(*) as count 
                FROM audit_logs 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY event_type
                ORDER BY count DESC
            """, (start_time.isoformat(), end_time.isoformat()))
            events_by_type = {row['event_type']: row['count'] for row in cursor}
            
            # Events by agent
            cursor = conn.execute("""
                SELECT agent_id, COUNT(*) as count 
                FROM audit_logs 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY agent_id
                ORDER BY count DESC
            """, (start_time.isoformat(), end_time.isoformat()))
            events_by_agent = {row['agent_id']: row['count'] for row in cursor}
            
            # Events by level
            cursor = conn.execute("""
                SELECT level, COUNT(*) as count 
                FROM audit_logs 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY level
                ORDER BY count DESC
            """, (start_time.isoformat(), end_time.isoformat()))
            events_by_level = {row['level']: row['count'] for row in cursor}
            
            return {
                "time_window": str(time_window),
                "total_events": total_events,
                "events_by_type": events_by_type,
                "events_by_agent": events_by_agent,
                "events_by_level": events_by_level,
                "summary_generated_at": datetime.now().isoformat()
            }
```

## 4. Distributed Tracing

### 4.1 Tracing Framework for Multi-Agent Systems

```python
import uuid
import time
from typing import Dict, List, Optional, Any, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from enum import Enum

class SpanKind(Enum):
    """Types of spans in distributed tracing"""
    INTERNAL = "internal"  # Internal operation
    SERVER = "server"      # Server-side operation
    CLIENT = "client"      # Client-side operation
    PRODUCER = "producer"  # Message producer
    CONSUMER = "consumer"  # Message consumer

class SpanStatus(Enum):
    """Status of a span"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class SpanContext:
    """Context information for a span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage
        }

@dataclass
class Span:
    """Represents a single operation in a distributed trace"""
    context: SpanContext
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    kind: SpanKind = SpanKind.INTERNAL
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    agent_id: Optional[str] = None
    
    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
            self.status = status
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the span"""
        self.tags[key] = value
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            "level": level,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization"""
        return {
            "context": self.context.to_dict(),
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status.value,
            "kind": self.kind.value,
            "tags": self.tags,
            "logs": self.logs,
            "agent_id": self.agent_id
        }

class TracingContext:
    """Thread-local tracing context"""
    
    def __init__(self):
        self._local = threading.local()
    
    @property
    def current_span(self) -> Optional[Span]:
        """Get the current active span"""
        return getattr(self._local, 'current_span', None)
    
    @current_span.setter
    def current_span(self, span: Optional[Span]):
        """Set the current active span"""
        self._local.current_span = span
    
    @property
    def trace_id(self) -> Optional[str]:
        """Get the current trace ID"""
        span = self.current_span
        return span.context.trace_id if span else None
    
    def create_child_context(self, operation_name: str) -> SpanContext:
        """Create a child span context"""
        current = self.current_span
        if current:
            return SpanContext(
                trace_id=current.context.trace_id,
                span_id=str(uuid.uuid4()),
                parent_span_id=current.context.span_id,
                baggage=current.context.baggage.copy()
            )
        else:
            return SpanContext(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4())
            )

class Tracer:
    """Distributed tracer for agentic AI systems"""
    
    def __init__(self, service_name: str, agent_id: str = None):
        self.service_name = service_name
        self.agent_id = agent_id
        self.context = TracingContext()
        self.spans: List[Span] = []
        self.span_processors: List[Callable[[Span], None]] = []
        self.lock = threading.RLock()
    
    def add_span_processor(self, processor: Callable[[Span], None]):
        """Add a span processor for handling completed spans"""
        self.span_processors.append(processor)
    
    def start_span(self, operation_name: str, 
                  parent_context: SpanContext = None,
                  kind: SpanKind = SpanKind.INTERNAL,
                  tags: Dict[str, Any] = None) -> Span:
        """Start a new span"""
        if parent_context:
            context = SpanContext(
                trace_id=parent_context.trace_id,
                span_id=str(uuid.uuid4()),
                parent_span_id=parent_context.span_id,
                baggage=parent_context.baggage.copy()
            )
        else:
            context = self.context.create_child_context(operation_name)
        
        span = Span(
            context=context,
            operation_name=operation_name,
            start_time=time.time(),
            kind=kind,
            agent_id=self.agent_id
        )
        
        # Set default tags
        span.set_tag("service.name", self.service_name)
        if self.agent_id:
            span.set_tag("agent.id", self.agent_id)
        
        # Set additional tags
        if tags:
            for key, value in tags.items():
                span.set_tag(key, value)
        
        with self.lock:
            self.spans.append(span)
        
        return span
    
    @contextmanager
    def span(self, operation_name: str, 
            parent_context: SpanContext = None,
            kind: SpanKind = SpanKind.INTERNAL,
            tags: Dict[str, Any] = None) -> ContextManager[Span]:
        """Context manager for automatic span lifecycle management"""
        span = self.start_span(operation_name, parent_context, kind, tags)
        old_span = self.context.current_span
        self.context.current_span = span
        
        try:
            yield span
            span.finish(SpanStatus.OK)
        except Exception as e:
            span.set_tag("error", True)
            span.set_tag("error.message", str(e))
            span.set_tag("error.type", type(e).__name__)
            span.log(f"Exception occurred: {str(e)}", level="error")
            span.finish(SpanStatus.ERROR)
            raise
        finally:
            self.context.current_span = old_span
            
            # Process completed span
            for processor in self.span_processors:
                try:
                    processor(span)
                except Exception as e:
                    # Don't let span processor errors affect the main operation
                    logging.error(f"Span processor error: {e}")
    
    def inject_context(self, span_context: SpanContext) -> Dict[str, str]:
        """Inject span context into headers for propagation"""
        return {
            "trace-id": span_context.trace_id,
            "span-id": span_context.span_id,
            "parent-span-id": span_context.parent_span_id or "",
            "baggage": json.dumps(span_context.baggage)
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from headers"""
        trace_id = headers.get("trace-id")
        span_id = headers.get("span-id")
        parent_span_id = headers.get("parent-span-id")
        baggage_str = headers.get("baggage", "{}")
        
        if not trace_id or not span_id:
            return None
        
        try:
            baggage = json.loads(baggage_str)
        except (json.JSONDecodeError, TypeError):
            baggage = {}
        
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id if parent_span_id else None,
            baggage=baggage
        )
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a specific trace"""
        with self.lock:
            return [span for span in self.spans if span.context.trace_id == trace_id]
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary information for a trace"""
        trace_spans = self.get_trace(trace_id)
        
        if not trace_spans:
            return {"error": f"No spans found for trace {trace_id}"}
        
        # Calculate trace duration
        start_time = min(span.start_time for span in trace_spans)
        end_time = max(span.end_time or span.start_time for span in trace_spans)
        total_duration = end_time - start_time
        
        # Count spans by status
        status_counts = {}
        for span in trace_spans:
            status = span.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get involved agents
        agents = set(span.agent_id for span in trace_spans if span.agent_id)
        
        # Get operations
        operations = [span.operation_name for span in trace_spans]
        
        return {
            "trace_id": trace_id,
            "total_spans": len(trace_spans),
            "total_duration": total_duration,
            "status_counts": status_counts,
            "involved_agents": list(agents),
            "operations": operations,
            "start_time": start_time,
            "end_time": end_time
        }

class AgentTracer(Tracer):
    """Specialized tracer for individual agents"""
    
    def __init__(self, agent_id: str, service_name: str = "agentic-ai"):
        super().__init__(service_name, agent_id)
        self.task_spans: Dict[str, Span] = {}
    
    def start_task_trace(self, task_name: str, task_params: Dict[str, Any] = None) -> Span:
        """Start tracing a task execution"""
        tags = {
            "task.name": task_name,
            "task.type": "agent_task"
        }
        
        if task_params:
            tags["task.params"] = json.dumps(task_params)
        
        span = self.start_span(
            f"task:{task_name}",
            kind=SpanKind.INTERNAL,
            tags=tags
        )
        
        self.task_spans[task_name] = span
        return span
    
    def trace_decision(self, decision_type: str, decision: str, 
                      confidence: float, reasoning: str = None) -> Span:
        """Trace an agent decision"""
        tags = {
            "decision.type": decision_type,
            "decision.result": decision,
            "decision.confidence": confidence
        }
        
        if reasoning:
            tags["decision.reasoning"] = reasoning
        
        with self.span(f"decision:{decision_type}", tags=tags) as span:
            span.log(f"Decision made: {decision} (confidence: {confidence})")
            return span
    
    def trace_communication(self, target_agent: str, message_type: str, 
                          direction: str = "outbound") -> Span:
        """Trace inter-agent communication"""
        kind = SpanKind.CLIENT if direction == "outbound" else SpanKind.SERVER
        
        tags = {
            "communication.target_agent": target_agent,
            "communication.message_type": message_type,
            "communication.direction": direction
        }
        
        operation_name = f"comm:{direction}:{message_type}"
        return self.start_span(operation_name, kind=kind, tags=tags)

class TraceCollector:
    """Collects and stores distributed traces"""
    
    def __init__(self):
        self.traces: Dict[str, List[Span]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def collect_span(self, span: Span):
        """Collect a completed span"""
        with self.lock:
            self.traces[span.context.trace_id].append(span)
    
    def get_trace_timeline(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get chronological timeline of events in a trace"""
        with self.lock:
            spans = self.traces.get(trace_id, [])
            
            # Create timeline events
            events = []
            
            for span in spans:
                # Start event
                events.append({
                    "timestamp": span.start_time,
                    "type": "span_start",
                    "span_id": span.context.span_id,
                    "operation": span.operation_name,
                    "agent_id": span.agent_id,
                    "tags": span.tags
                })
                
                # Log events
                for log_entry in span.logs:
                    events.append({
                        "timestamp": log_entry["timestamp"],
                        "type": "log",
                        "span_id": span.context.span_id,
                        "message": log_entry["message"],
                        "level": log_entry["level"],
                        "agent_id": span.agent_id
                    })
                
                # End event
                if span.end_time:
                    events.append({
                        "timestamp": span.end_time,
                        "type": "span_end",
                        "span_id": span.context.span_id,
                        "operation": span.operation_name,
                        "duration": span.duration,
                        "status": span.status.value,
                        "agent_id": span.agent_id
                    })
            
            # Sort by timestamp
            return sorted(events, key=lambda e: e["timestamp"])
```

### 4.2 Cross-Agent Trace Correlation

```python
class CrossAgentTraceCorrelator:
    """Correlates traces across multiple agents"""
    
    def __init__(self, trace_collector: TraceCollector):
        self.trace_collector = trace_collector
        self.agent_tracers: Dict[str, AgentTracer] = {}
    
    def register_agent_tracer(self, agent_id: str, tracer: AgentTracer):
        """Register a tracer for an agent"""
        self.agent_tracers[agent_id] = tracer
        tracer.add_span_processor(self.trace_collector.collect_span)
    
    def create_cross_agent_trace(self, initiating_agent: str, 
                                target_agents: List[str],
                                operation_name: str) -> str:
        """Create a new trace that spans multiple agents"""
        trace_id = str(uuid.uuid4())
        
        # Start root span in initiating agent
        if initiating_agent in self.agent_tracers:
            tracer = self.agent_tracers[initiating_agent]
            root_context = SpanContext(trace_id=trace_id, span_id=str(uuid.uuid4()))
            
            with tracer.span(operation_name, parent_context=root_context) as root_span:
                root_span.set_tag("trace.type", "cross_agent")
                root_span.set_tag("trace.initiating_agent", initiating_agent)
                root_span.set_tag("trace.target_agents", ",".join(target_agents))
        
        return trace_id
    
    def get_cross_agent_analysis(self, trace_id: str) -> Dict[str, Any]:
        """Analyze cross-agent interactions in a trace"""
        timeline = self.trace_collector.get_trace_timeline(trace_id)
        
        if not timeline:
            return {"error": f"No timeline found for trace {trace_id}"}
        
        # Analyze agent interactions
        agent_interactions = defaultdict(list)
        agent_activity = defaultdict(lambda: {"start": None, "end": None, "operations": []})
        
        for event in timeline:
            agent_id = event.get("agent_id")
            if not agent_id:
                continue
            
            if event["type"] == "span_start":
                if agent_activity[agent_id]["start"] is None:
                    agent_activity[agent_id]["start"] = event["timestamp"]
                agent_activity[agent_id]["operations"].append(event["operation"])
            
            elif event["type"] == "span_end":
                agent_activity[agent_id]["end"] = event["timestamp"]
        
        # Calculate interaction patterns
        agents = list(agent_activity.keys())
        interaction_matrix = {}
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    # Look for communication patterns
                    comm_events = [
                        event for event in timeline
                        if (event.get("agent_id") == agent1 and 
                            agent2 in str(event.get("tags", {})).lower())
                    ]
                    interaction_matrix[f"{agent1}->{agent2}"] = len(comm_events)
        
        return {
             "trace_id": trace_id,
             "total_events": len(timeline),
             "involved_agents": agents,
             "agent_activity": dict(agent_activity),
             "interaction_matrix": interaction_matrix,
             "trace_duration": timeline[-1]["timestamp"] - timeline[0]["timestamp"] if timeline else 0
         }
```

## 5. Alerting and Anomaly Detection

### 5.1 Intelligent Alerting System

```python
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """Represents an alert"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str  # Agent ID or system component
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "source": self.source,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "tags": self.tags,
            "metadata": self.metadata
        }

class AlertRule:
    """Defines conditions for triggering alerts"""
    
    def __init__(self, name: str, metric_pattern: str, 
                 condition: Callable[[float], bool],
                 severity: AlertSeverity,
                 description: str = "",
                 cooldown_minutes: int = 5):
        self.name = name
        self.metric_pattern = metric_pattern
        self.condition = condition
        self.severity = severity
        self.description = description
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered: Optional[datetime] = None
    
    def should_trigger(self, metric_name: str, value: float) -> bool:
        """Check if this rule should trigger for the given metric"""
        # Check if metric matches pattern
        if self.metric_pattern not in metric_name:
            return False
        
        # Check cooldown period
        if self.last_triggered:
            time_since_last = datetime.now() - self.last_triggered
            if time_since_last.total_seconds() < (self.cooldown_minutes * 60):
                return False
        
        # Check condition
        return self.condition(value)
    
    def trigger(self) -> None:
        """Mark this rule as triggered"""
        self.last_triggered = datetime.now()

class AlertChannel(ABC):
    """Abstract base class for alert notification channels"""
    
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert notification"""
        pass

class EmailAlertChannel(AlertChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int,
                 username: str, password: str,
                 from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            
            Title: {alert.title}
            Description: {alert.description}
            Severity: {alert.severity.value}
            Source: {alert.source}
            Metric: {alert.metric_name}
            Current Value: {alert.current_value}
            Threshold: {alert.threshold_value}
            Timestamp: {alert.timestamp}
            
            Tags: {json.dumps(alert.tags, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
            return False

class SlackAlertChannel(AlertChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        try:
            color_map = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            payload = {
                "text": f"Alert: {alert.title}",
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": str(alert.current_value), "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ],
                    "ts": alert.timestamp.timestamp()
                }]
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            response = requests.post(self.webhook_url, json=payload)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Failed to send Slack alert: {e}")
            return False

class AlertManager:
    """Manages alerts and notifications for agentic AI systems"""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.rules: List[AlertRule] = []
        self.channels: List[AlertChannel] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.lock = threading.RLock()
        
        # Start monitoring thread
        self._start_monitoring()
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        with self.lock:
            self.rules.append(rule)
    
    def add_channel(self, channel: AlertChannel):
        """Add a notification channel"""
        with self.lock:
            self.channels.append(channel)
    
    def create_standard_rules(self, agent_id: str = None):
        """Create standard alerting rules for agentic AI systems"""
        prefix = f"agent.{agent_id}." if agent_id else ""
        
        # High error rate
        self.add_rule(AlertRule(
            name="High Error Rate",
            metric_pattern=f"{prefix}tasks.error",
            condition=lambda x: x > 0.1,  # More than 10% error rate
            severity=AlertSeverity.HIGH,
            description="Agent error rate is above acceptable threshold"
        ))
        
        # Low success rate
        self.add_rule(AlertRule(
            name="Low Success Rate",
            metric_pattern=f"{prefix}tasks.success_rate",
            condition=lambda x: x < 0.8,  # Less than 80% success rate
            severity=AlertSeverity.MEDIUM,
            description="Agent success rate is below expected threshold"
        ))
        
        # High CPU usage
        self.add_rule(AlertRule(
            name="High CPU Usage",
            metric_pattern=f"{prefix}resources.cpu_percent",
            condition=lambda x: x > 80,  # More than 80% CPU
            severity=AlertSeverity.MEDIUM,
            description="Agent CPU usage is high"
        ))
        
        # High memory usage
        self.add_rule(AlertRule(
            name="High Memory Usage",
            metric_pattern=f"{prefix}resources.memory_mb",
            condition=lambda x: x > 1000,  # More than 1GB
            severity=AlertSeverity.MEDIUM,
            description="Agent memory usage is high"
        ))
        
        # Slow response time
        self.add_rule(AlertRule(
            name="Slow Response Time",
            metric_pattern=f"{prefix}tasks.duration",
            condition=lambda x: x > 30,  # More than 30 seconds
            severity=AlertSeverity.LOW,
            description="Agent response time is slower than expected"
        ))
    
    def check_metrics(self):
        """Check current metrics against alert rules"""
        with self.lock:
            # Get recent metrics (last 5 minutes)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
            
            recent_metrics = self.metric_collector.get_metrics(
                time_range=(start_time, end_time)
            )
            
            # Group metrics by name and get latest values
            latest_metrics = {}
            for metric in recent_metrics:
                if (metric.name not in latest_metrics or 
                    metric.timestamp > latest_metrics[metric.name].timestamp):
                    latest_metrics[metric.name] = metric
            
            # Check each metric against rules
            for metric_name, metric in latest_metrics.items():
                for rule in self.rules:
                    if rule.should_trigger(metric_name, metric.value):
                        self._create_alert(rule, metric)
                        rule.trigger()
    
    def _create_alert(self, rule: AlertRule, metric: MetricValue):
        """Create and send an alert"""
        alert_id = f"{rule.name}_{metric.name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            title=rule.name,
            description=rule.description,
            severity=rule.severity,
            source=metric.tags.get('agent_id', 'system'),
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=0,  # Would need to extract from condition
            timestamp=metric.timestamp,
            tags=metric.tags,
            metadata=metric.metadata
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        for channel in self.channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                logging.error(f"Failed to send alert via channel: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str = None) -> bool:
        """Acknowledge an active alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
                if user:
                    self.active_alerts[alert_id].metadata['acknowledged_by'] = user
                    self.active_alerts[alert_id].metadata['acknowledged_at'] = datetime.now().isoformat()
                return True
            return False
    
    def resolve_alert(self, alert_id: str, user: str = None) -> bool:
        """Resolve an active alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].status = AlertStatus.RESOLVED
                if user:
                    self.active_alerts[alert_id].metadata['resolved_by'] = user
                    self.active_alerts[alert_id].metadata['resolved_at'] = datetime.now().isoformat()
                del self.active_alerts[alert_id]
                return True
            return False
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get currently active alerts"""
        with self.lock:
            alerts = list(self.active_alerts.values())
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Get summary of alerts in time window"""
        end_time = datetime.now()
        start_time = end_time - time_window
        
        relevant_alerts = [
            alert for alert in self.alert_history
            if start_time <= alert.timestamp <= end_time
        ]
        
        # Count by severity
        severity_counts = {}
        for alert in relevant_alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by source
        source_counts = {}
        for alert in relevant_alerts:
            source = alert.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "time_window": str(time_window),
            "total_alerts": len(relevant_alerts),
            "active_alerts": len(self.active_alerts),
            "severity_breakdown": severity_counts,
            "source_breakdown": source_counts,
            "summary_generated_at": datetime.now().isoformat()
        }
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor_worker():
            while True:
                try:
                    self.check_metrics()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logging.error(f"Error in alert monitoring: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        monitor_thread.start()
```

### 5.2 Anomaly Detection

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import deque
import pickle

class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection"""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Train the anomaly detector"""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)"""
        pass
    
    @abstractmethod
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        pass

class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detection using Z-score"""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
    
    def fit(self, data: np.ndarray) -> None:
        """Fit statistical parameters"""
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies using Z-score"""
        z_scores = np.abs((data - self.mean_) / (self.std_ + 1e-8))
        return np.where(np.any(z_scores > self.threshold, axis=1), -1, 1)
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return Z-scores as anomaly scores"""
        z_scores = np.abs((data - self.mean_) / (self.std_ + 1e-8))
        return np.max(z_scores, axis=1)

class MLAnomalyDetector(AnomalyDetector):
    """Machine learning-based anomaly detection using Isolation Forest"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """Train the isolation forest model"""
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)
        self.fitted = True
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        scaled_data = self.scaler.transform(data)
        return self.model.predict(scaled_data)
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring")
        
        scaled_data = self.scaler.transform(data)
        # Convert decision function scores to positive anomaly scores
        scores = -self.model.decision_function(scaled_data)
        return scores

class AgentAnomalyMonitor:
    """Monitors agent behavior for anomalies"""
    
    def __init__(self, agent_id: str, 
                 metric_collector: MetricCollector,
                 detector: AnomalyDetector,
                 window_size: int = 100,
                 retrain_interval: int = 1000):
        self.agent_id = agent_id
        self.metric_collector = metric_collector
        self.detector = detector
        self.window_size = window_size
        self.retrain_interval = retrain_interval
        
        # Data storage
        self.metric_history = deque(maxlen=window_size * 10)  # Keep more for retraining
        self.anomaly_history: List[Dict[str, Any]] = []
        
        # Tracking
        self.samples_processed = 0
        self.last_retrain = 0
        self.is_trained = False
        
        # Feature extraction configuration
        self.feature_names = [
            'task_duration', 'cpu_percent', 'memory_mb',
            'success_rate', 'error_rate', 'queue_size'
        ]
    
    def extract_features(self, metrics: List[MetricValue]) -> Optional[np.ndarray]:
        """Extract feature vector from metrics"""
        # Group metrics by name
        metric_dict = {}
        for metric in metrics:
            if metric.tags.get('agent_id') == self.agent_id:
                metric_dict[metric.name.split('.')[-1]] = metric.value
        
        # Extract features in consistent order
        features = []
        for feature_name in self.feature_names:
            if feature_name in metric_dict:
                features.append(metric_dict[feature_name])
            else:
                features.append(0.0)  # Default value for missing metrics
        
        return np.array(features) if features else None
    
    def update(self, metrics: List[MetricValue]) -> Optional[Dict[str, Any]]:
        """Update with new metrics and check for anomalies"""
        features = self.extract_features(metrics)
        if features is None:
            return None
        
        # Store in history
        self.metric_history.append({
            'timestamp': datetime.now(),
            'features': features,
            'metrics': {m.name: m.value for m in metrics if m.tags.get('agent_id') == self.agent_id}
        })
        
        self.samples_processed += 1
        
        # Initial training
        if not self.is_trained and len(self.metric_history) >= self.window_size:
            self._train_detector()
        
        # Periodic retraining
        elif (self.is_trained and 
              self.samples_processed - self.last_retrain >= self.retrain_interval):
            self._retrain_detector()
        
        # Anomaly detection
        if self.is_trained:
            return self._detect_anomaly(features)
        
        return None
    
    def _train_detector(self):
        """Initial training of the detector"""
        if len(self.metric_history) < self.window_size:
            return
        
        # Prepare training data
        training_data = np.array([
            entry['features'] for entry in list(self.metric_history)[-self.window_size:]
        ])
        
        # Train detector
        self.detector.fit(training_data)
        self.is_trained = True
        self.last_retrain = self.samples_processed
        
        logging.info(f"Anomaly detector trained for agent {self.agent_id} with {len(training_data)} samples")
    
    def _retrain_detector(self):
        """Retrain the detector with recent data"""
        # Use all available history for retraining
        training_data = np.array([
            entry['features'] for entry in self.metric_history
        ])
        
        # Retrain detector
        self.detector.fit(training_data)
        self.last_retrain = self.samples_processed
        
        logging.info(f"Anomaly detector retrained for agent {self.agent_id} with {len(training_data)} samples")
    
    def _detect_anomaly(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect if current features represent an anomaly"""
        features_2d = features.reshape(1, -1)
        
        # Get prediction and score
        prediction = self.detector.predict(features_2d)[0]
        score = self.detector.score(features_2d)[0]
        
        if prediction == -1:  # Anomaly detected
            anomaly_info = {
                'agent_id': self.agent_id,
                'timestamp': datetime.now(),
                'anomaly_score': float(score),
                'features': features.tolist(),
                'feature_names': self.feature_names,
                'severity': self._calculate_severity(score)
            }
            
            self.anomaly_history.append(anomaly_info)
            return anomaly_info
        
        return None
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate anomaly severity based on score"""
        if score > 0.8:
            return "critical"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"
    
    def get_anomaly_summary(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        end_time = datetime.now()
        start_time = end_time - time_window
        
        recent_anomalies = [
            anomaly for anomaly in self.anomaly_history
            if start_time <= anomaly['timestamp'] <= end_time
        ]
        
        # Calculate statistics
        severity_counts = {}
        for anomaly in recent_anomalies:
            severity = anomaly['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        avg_score = np.mean([a['anomaly_score'] for a in recent_anomalies]) if recent_anomalies else 0
        
        return {
            'agent_id': self.agent_id,
            'time_window': str(time_window),
            'total_anomalies': len(recent_anomalies),
            'severity_breakdown': severity_counts,
            'average_anomaly_score': float(avg_score),
            'is_trained': self.is_trained,
            'samples_processed': self.samples_processed,
            'last_retrain': self.last_retrain
        }
```

## 6. Practical Implementation Examples

### 6.1 Complete Observability Setup

```python
# Example: Setting up complete observability for a multi-agent system

def setup_observability_stack(agent_ids: List[str]) -> Dict[str, Any]:
    """Set up complete observability stack for multiple agents"""
    
    # 1. Initialize metric collection
    metric_collector = InMemoryMetricCollector()
    
    # 2. Set up logging
    logger = AgentLogger(log_level=LogLevel.INFO)
    audit_store = AuditTrailStore()
    
    # 3. Initialize tracing
    tracer = Tracer()
    agent_tracers = {}
    for agent_id in agent_ids:
        agent_tracers[agent_id] = AgentTracer(agent_id, tracer)
    
    trace_correlator = CrossAgentTraceCorrelator()
    for agent_tracer in agent_tracers.values():
        trace_correlator.register_agent_tracer(agent_tracer)
    
    # 4. Set up alerting
    alert_manager = AlertManager(metric_collector)
    
    # Add notification channels
    email_channel = EmailAlertChannel(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="alerts@company.com",
        password="app_password",
        from_email="alerts@company.com",
        to_emails=["ops@company.com"]
    )
    alert_manager.add_channel(email_channel)
    
    # Create standard alert rules for each agent
    for agent_id in agent_ids:
        alert_manager.create_standard_rules(agent_id)
    
    # 5. Set up anomaly detection
    anomaly_monitors = {}
    for agent_id in agent_ids:
        detector = MLAnomalyDetector(contamination=0.05)
        anomaly_monitors[agent_id] = AgentAnomalyMonitor(
            agent_id=agent_id,
            metric_collector=metric_collector,
            detector=detector
        )
    
    return {
        'metric_collector': metric_collector,
        'logger': logger,
        'audit_store': audit_store,
        'tracer': tracer,
        'agent_tracers': agent_tracers,
        'trace_correlator': trace_correlator,
        'alert_manager': alert_manager,
        'anomaly_monitors': anomaly_monitors
    }

# Usage example
observability_stack = setup_observability_stack(['agent_1', 'agent_2', 'agent_3'])

# Simulate agent activity
def simulate_agent_activity(agent_id: str, observability_stack: Dict[str, Any]):
    """Simulate agent activity with observability"""
    
    metric_collector = observability_stack['metric_collector']
    logger = observability_stack['logger']
    agent_tracer = observability_stack['agent_tracers'][agent_id]
    anomaly_monitor = observability_stack['anomaly_monitors'][agent_id]
    
    # Start a task trace
    with agent_tracer.trace_task("process_request", {"request_id": "req_123"}):
        
        # Log task start
        logger.log(
            level=LogLevel.INFO,
            event_type=EventType.TASK_START,
            agent_id=agent_id,
            message="Starting request processing",
            metadata={"request_id": "req_123"}
        )
        
        # Simulate some work and collect metrics
        import random
        import time
        
        # Simulate task duration
        task_duration = random.uniform(1, 5)
        time.sleep(task_duration)
        
        # Collect metrics
        metrics = [
            MetricValue(
                name=f"agent.{agent_id}.task_duration",
                value=task_duration,
                timestamp=datetime.now(),
                tags={"agent_id": agent_id, "task_type": "process_request"}
            ),
            MetricValue(
                name=f"agent.{agent_id}.cpu_percent",
                value=random.uniform(10, 90),
                timestamp=datetime.now(),
                tags={"agent_id": agent_id}
            ),
            MetricValue(
                name=f"agent.{agent_id}.memory_mb",
                value=random.uniform(100, 800),
                timestamp=datetime.now(),
                tags={"agent_id": agent_id}
            )
        ]
        
        # Record metrics
        for metric in metrics:
            metric_collector.record_metric(metric)
        
        # Check for anomalies
        anomaly_result = anomaly_monitor.update(metrics)
        if anomaly_result:
            logger.log(
                level=LogLevel.WARNING,
                event_type=EventType.ANOMALY_DETECTED,
                agent_id=agent_id,
                message=f"Anomaly detected with score {anomaly_result['anomaly_score']:.3f}",
                metadata=anomaly_result
            )
        
        # Simulate success/failure
        success = random.random() > 0.1  # 90% success rate
        
        if success:
            logger.log(
                level=LogLevel.INFO,
                event_type=EventType.TASK_COMPLETE,
                agent_id=agent_id,
                message="Request processed successfully",
                metadata={"request_id": "req_123", "duration": task_duration}
            )
        else:
            logger.log(
                level=LogLevel.ERROR,
                event_type=EventType.TASK_ERROR,
                agent_id=agent_id,
                message="Request processing failed",
                metadata={"request_id": "req_123", "error": "Simulated error"}
            )

# Run simulation
for i in range(10):
    for agent_id in ['agent_1', 'agent_2', 'agent_3']:
        simulate_agent_activity(agent_id, observability_stack)
    time.sleep(1)

# Get observability insights
print("=== Observability Summary ===")

# Alert summary
alert_summary = observability_stack['alert_manager'].get_alert_summary()
print(f"Alerts in last 24h: {alert_summary['total_alerts']}")
print(f"Active alerts: {alert_summary['active_alerts']}")

# Anomaly summary for each agent
for agent_id in ['agent_1', 'agent_2', 'agent_3']:
    anomaly_summary = observability_stack['anomaly_monitors'][agent_id].get_anomaly_summary()
    print(f"Agent {agent_id} anomalies: {anomaly_summary['total_anomalies']}")

# Trace analysis
trace_summary = observability_stack['trace_correlator'].analyze_cross_agent_interactions()
print(f"Cross-agent interactions analyzed: {len(trace_summary)}")
```

## 7. Hands-on Exercises

### Exercise 1: Custom Metric Collection

```python
class CustomBusinessMetrics(MetricCollector):
    """Custom metrics for business-specific KPIs"""
    
    def __init__(self):
        super().__init__()
        self.business_metrics = {
            'customer_satisfaction': [],
            'revenue_per_task': [],
            'sla_compliance': [],
            'cost_per_operation': []
        }
    
    def record_business_metric(self, metric_name: str, value: float, 
                              context: Dict[str, Any] = None):
        """Record business-specific metrics"""
        if metric_name in self.business_metrics:
            metric = MetricValue(
                name=f"business.{metric_name}",
                value=value,
                timestamp=datetime.now(),
                tags=context or {},
                metadata={'category': 'business'}
            )
            self.record_metric(metric)
            self.business_metrics[metric_name].append(metric)
    
    def get_business_dashboard(self) -> Dict[str, Any]:
        """Generate business metrics dashboard"""
        dashboard = {}
        
        for metric_name, metrics in self.business_metrics.items():
            if metrics:
                values = [m.value for m in metrics[-100:]]  # Last 100 values
                dashboard[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values),
                    'trend': 'up' if len(values) > 1 and values[-1] > values[-2] else 'down',
                    'count': len(values)
                }
        
        return dashboard

# TODO: Implement CustomBusinessMetrics and integrate with your agent system
```

### Exercise 2: Advanced Alert Rules

```python
class SmartAlertRule(AlertRule):
    """Advanced alert rule with machine learning-based thresholds"""
    
    def __init__(self, name: str, metric_pattern: str, 
                 severity: AlertSeverity, description: str = "",
                 adaptive_threshold: bool = True):
        # Initialize with a placeholder condition
        super().__init__(name, metric_pattern, lambda x: False, severity, description)
        
        self.adaptive_threshold = adaptive_threshold
        self.metric_history = deque(maxlen=1000)
        self.threshold_percentile = 95  # Alert on top 5% of values
        self.min_samples = 50
    
    def update_threshold(self, value: float):
        """Update adaptive threshold based on historical data"""
        self.metric_history.append(value)
        
        if len(self.metric_history) >= self.min_samples:
            threshold = np.percentile(list(self.metric_history), self.threshold_percentile)
            self.condition = lambda x: x > threshold
    
    def should_trigger(self, metric_name: str, value: float) -> bool:
        """Enhanced trigger logic with adaptive thresholds"""
        if self.adaptive_threshold:
            self.update_threshold(value)
        
        return super().should_trigger(metric_name, value)

# TODO: Create SmartAlertRule instances and test with your metrics
```

### Exercise 3: Distributed Tracing Dashboard

```python
class TracingDashboard:
    """Web dashboard for visualizing distributed traces"""
    
    def __init__(self, trace_correlator: CrossAgentTraceCorrelator):
        self.trace_correlator = trace_correlator
    
    def generate_trace_visualization(self, trace_id: str) -> Dict[str, Any]:
        """Generate data for trace visualization"""
        timeline = self.trace_correlator.get_trace_timeline(trace_id)
        
        # Convert to visualization format
        vis_data = {
            'nodes': [],
            'edges': [],
            'timeline': []
        }
        
        agent_positions = {}
        for i, event in enumerate(timeline):
            agent_id = event.get('agent_id', 'unknown')
            
            # Add node for agent if not exists
            if agent_id not in agent_positions:
                agent_positions[agent_id] = len(vis_data['nodes'])
                vis_data['nodes'].append({
                    'id': agent_id,
                    'label': agent_id,
                    'type': 'agent'
                })
            
            # Add timeline event
            vis_data['timeline'].append({
                'timestamp': event['timestamp'].isoformat(),
                'agent': agent_id,
                'event': event.get('event_type', 'unknown'),
                'duration': event.get('duration', 0),
                'status': event.get('status', 'unknown')
            })
        
        return vis_data
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health from traces"""
        # TODO: Implement system health analysis
        return {
            'total_traces': 0,
            'average_duration': 0,
            'error_rate': 0,
            'bottlenecks': [],
            'recommendations': []
        }

# TODO: Build a web interface using Flask/FastAPI to display the dashboard
```

## 8. Module Summary

### Key Concepts Learned

1. **Observability Fundamentals**
   - Three pillars: Metrics, Logs, and Traces
   - Unique challenges in agentic AI systems
   - Importance of correlation across agents

2. **Metrics Collection and Monitoring**
   - Designing effective metric schemas
   - Real-time and batch collection strategies
   - System-level and agent-specific metrics

3. **Logging and Audit Trails**
   - Structured logging for agent activities
   - Audit trail design for compliance
   - Log correlation and analysis

4. **Distributed Tracing**
   - Tracing agent interactions and workflows
   - Cross-agent trace correlation
   - Performance bottleneck identification

5. **Alerting and Anomaly Detection**
   - Intelligent alerting systems
   - Machine learning-based anomaly detection
   - Multi-channel notification strategies

### Practical Skills Developed

- **Implementing comprehensive observability stacks**
- **Designing custom metrics for agentic systems**
- **Building intelligent alerting systems**
- **Creating anomaly detection pipelines**
- **Developing distributed tracing solutions**
- **Correlating data across multiple observability dimensions**

### Real-world Applications

- **Production monitoring** of multi-agent systems
- **Performance optimization** through observability insights
- **Incident response** and root cause analysis
- **Compliance and audit** trail maintenance
- **Capacity planning** based on usage patterns
- **Quality assurance** through continuous monitoring

### Next Steps

In the next module, we'll explore **Interoperability of Agents**, covering:
- Agent communication protocols
- Standard interfaces and APIs
- Integration with external systems
- Cross-platform agent deployment
- Ecosystem integration patterns

The observability patterns learned here will be essential for monitoring these complex interoperable systems.

---

**Continue to Module VII: Interoperability of Agents** →