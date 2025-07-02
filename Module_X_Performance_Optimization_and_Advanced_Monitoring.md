# Module X: Performance Optimization and Advanced Monitoring

## Learning Objectives

By the end of this module, you will be able to:
1. **Profile and optimize** agent system performance using advanced techniques
2. **Implement comprehensive monitoring** with distributed tracing and observability
3. **Design cost-effective** resource allocation and optimization strategies
4. **Apply security hardening** measures for production agent systems
5. **Establish governance frameworks** for compliance and operational excellence

## Introduction

Performance optimization and advanced monitoring are critical for maintaining efficient, reliable, and cost-effective agent systems in production. This module covers sophisticated techniques for profiling, optimizing, and monitoring multi-agent systems at scale.

### Performance Optimization Architecture

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Performance Optimization & Monitoring Architecture</text>
  
  <!-- Performance Profiling Layer -->
  <rect x="50" y="60" width="700" height="100" fill="#e8f4fd" stroke="#3498db" stroke-width="2" rx="10"/>
  <text x="400" y="85" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Performance Profiling Layer</text>
  <rect x="70" y="95" width="120" height="50" fill="#3498db" rx="5"/>
  <text x="130" y="125" text-anchor="middle" font-size="10" fill="white">CPU Profiler</text>
  <rect x="210" y="95" width="120" height="50" fill="#3498db" rx="5"/>
  <text x="270" y="125" text-anchor="middle" font-size="10" fill="white">Memory Profiler</text>
  <rect x="350" y="95" width="120" height="50" fill="#3498db" rx="5"/>
  <text x="410" y="125" text-anchor="middle" font-size="10" fill="white">I/O Profiler</text>
  <rect x="490" y="95" width="120" height="50" fill="#3498db" rx="5"/>
  <text x="550" y="125" text-anchor="middle" font-size="10" fill="white">Network Profiler</text>
  <rect x="630" y="95" width="100" height="50" fill="#3498db" rx="5"/>
  <text x="680" y="125" text-anchor="middle" font-size="10" fill="white">Trace Profiler</text>
  
  <!-- Distributed Tracing Layer -->
  <rect x="50" y="180" width="700" height="100" fill="#fff3cd" stroke="#f39c12" stroke-width="2" rx="10"/>
  <text x="400" y="205" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Distributed Tracing & Observability</text>
  <rect x="70" y="215" width="140" height="50" fill="#f39c12" rx="5"/>
  <text x="140" y="245" text-anchor="middle" font-size="10" fill="white">Trace Collection</text>
  <rect x="230" y="215" width="140" height="50" fill="#f39c12" rx="5"/>
  <text x="300" y="245" text-anchor="middle" font-size="10" fill="white">Span Analysis</text>
  <rect x="390" y="215" width="140" height="50" fill="#f39c12" rx="5"/>
  <text x="460" y="245" text-anchor="middle" font-size="10" fill="white">Correlation</text>
  <rect x="550" y="215" width="140" height="50" fill="#f39c12" rx="5"/>
  <text x="620" y="245" text-anchor="middle" font-size="10" fill="white">Visualization</text>
  
  <!-- Optimization Engine Layer -->
  <rect x="50" y="300" width="700" height="100" fill="#d4edda" stroke="#27ae60" stroke-width="2" rx="10"/>
  <text x="400" y="325" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Optimization Engine</text>
  <rect x="70" y="335" width="130" height="50" fill="#27ae60" rx="5"/>
  <text x="135" y="365" text-anchor="middle" font-size="10" fill="white">Resource Optimizer</text>
  <rect x="220" y="335" width="130" height="50" fill="#27ae60" rx="5"/>
  <text x="285" y="365" text-anchor="middle" font-size="10" fill="white">Cost Optimizer</text>
  <rect x="370" y="335" width="130" height="50" fill="#27ae60" rx="5"/>
  <text x="435" y="365" text-anchor="middle" font-size="10" fill="white">Performance Tuner</text>
  <rect x="520" y="335" width="130" height="50" fill="#27ae60" rx="5"/>
  <text x="585" y="365" text-anchor="middle" font-size="10" fill="white">Auto-Tuner</text>
  
  <!-- Security & Compliance Layer -->
  <rect x="50" y="420" width="700" height="100" fill="#f8d7da" stroke="#e74c3c" stroke-width="2" rx="10"/>
  <text x="400" y="445" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Security & Compliance</text>
  <rect x="70" y="455" width="120" height="50" fill="#e74c3c" rx="5"/>
  <text x="130" y="485" text-anchor="middle" font-size="10" fill="white">Security Scanner</text>
  <rect x="210" y="455" width="120" height="50" fill="#e74c3c" rx="5"/>
  <text x="270" y="485" text-anchor="middle" font-size="10" fill="white">Audit Logger</text>
  <rect x="350" y="455" width="120" height="50" fill="#e74c3c" rx="5"/>
  <text x="410" y="485" text-anchor="middle" font-size="10" fill="white">Compliance Check</text>
  <rect x="490" y="455" width="120" height="50" fill="#e74c3c" rx="5"/>
  <text x="550" y="485" text-anchor="middle" font-size="10" fill="white">Policy Engine</text>
  <rect x="630" y="455" width="100" height="50" fill="#e74c3c" rx="5"/>
  <text x="680" y="485" text-anchor="middle" font-size="10" fill="white">Governance</text>
  
  <!-- Governance Dashboard -->
  <rect x="250" y="540" width="300" height="50" fill="#6c757d" stroke="#495057" stroke-width="2" rx="10"/>
  <text x="400" y="570" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Unified Governance Dashboard</text>
  
  <!-- Arrows -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Vertical arrows -->
  <line x1="400" y1="160" x2="400" y2="180" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="280" x2="400" y2="300" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="400" x2="400" y2="420" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="520" x2="400" y2="540" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
</svg>
```

## 1. Performance Profiling and Analysis

### 1.1 Comprehensive Performance Profiler

```python
import asyncio
import time
import psutil
import threading
import tracemalloc
import cProfile
import pstats
import io
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import json
import logging
from collections import defaultdict, deque
import statistics
import gc
import sys

class ProfilerType(Enum):
    """Types of profilers available"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    CUSTOM = "custom"

@dataclass
class ProfileMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class ProfileSession:
    """Profiling session information"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    profiler_types: List[ProfilerType] = field(default_factory=list)
    metrics: List[ProfileMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceProfiler:
    """Comprehensive performance profiler for agent systems"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.active_sessions: Dict[str, ProfileSession] = {}
        self.profilers: Dict[ProfilerType, Callable] = {}
        self.background_tasks: Dict[str, asyncio.Task] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.cpu_profiler: Optional[cProfile.Profile] = None
        self.memory_tracker_active = False
        
        # Initialize built-in profilers
        self._setup_builtin_profilers()
        
    def _setup_builtin_profilers(self):
        """Set up built-in profiler functions"""
        self.profilers[ProfilerType.CPU] = self._profile_cpu
        self.profilers[ProfilerType.MEMORY] = self._profile_memory
        self.profilers[ProfilerType.IO] = self._profile_io
        self.profilers[ProfilerType.NETWORK] = self._profile_network
    
    async def start_profiling(self, session_id: str, 
                            profiler_types: List[ProfilerType],
                            metadata: Optional[Dict[str, Any]] = None) -> ProfileSession:
        """Start a profiling session"""
        if session_id in self.active_sessions:
            raise ValueError(f"Profiling session '{session_id}' already active")
        
        session = ProfileSession(
            session_id=session_id,
            start_time=time.time(),
            profiler_types=profiler_types,
            metadata=metadata or {}
        )
        
        self.active_sessions[session_id] = session
        
        # Start CPU profiling if requested
        if ProfilerType.CPU in profiler_types:
            self.cpu_profiler = cProfile.Profile()
            self.cpu_profiler.enable()
        
        # Start memory tracking if requested
        if ProfilerType.MEMORY in profiler_types and not self.memory_tracker_active:
            tracemalloc.start()
            self.memory_tracker_active = True
        
        # Start background sampling for other profilers
        sampling_task = asyncio.create_task(
            self._background_sampling(session_id, profiler_types)
        )
        self.background_tasks[session_id] = sampling_task
        
        logging.info(f"Started profiling session '{session_id}' with types: {profiler_types}")
        return session
    
    async def stop_profiling(self, session_id: str) -> ProfileSession:
        """Stop a profiling session and return results"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Profiling session '{session_id}' not found")
        
        session = self.active_sessions[session_id]
        session.end_time = time.time()
        
        # Stop CPU profiling
        if ProfilerType.CPU in session.profiler_types and self.cpu_profiler:
            self.cpu_profiler.disable()
            
            # Capture CPU profiling results
            s = io.StringIO()
            ps = pstats.Stats(self.cpu_profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            cpu_stats = s.getvalue()
            session.metadata['cpu_profile'] = cpu_stats
            self.cpu_profiler = None
        
        # Stop memory tracking
        if ProfilerType.MEMORY in session.profiler_types and self.memory_tracker_active:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.memory_tracker_active = False
            
            session.metadata['memory_peak'] = peak
            session.metadata['memory_current'] = current
        
        # Stop background sampling
        if session_id in self.background_tasks:
            self.background_tasks[session_id].cancel()
            try:
                await self.background_tasks[session_id]
            except asyncio.CancelledError:
                pass
            del self.background_tasks[session_id]
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logging.info(f"Stopped profiling session '{session_id}'")
        return session
    
    async def _background_sampling(self, session_id: str, profiler_types: List[ProfilerType]):
        """Background sampling for continuous profiling"""
        session = self.active_sessions[session_id]
        
        try:
            while session_id in self.active_sessions:
                timestamp = time.time()
                
                # Sample each requested profiler type
                for profiler_type in profiler_types:
                    if profiler_type in self.profilers:
                        try:
                            metrics = await self.profilers[profiler_type]()
                            for metric in metrics:
                                metric.timestamp = timestamp
                                session.metrics.append(metric)
                                self.metrics_buffer.append(metric)
                        except Exception as e:
                            logging.warning(f"Profiler {profiler_type} failed: {e}")
                
                await asyncio.sleep(self.sampling_interval)
                
        except asyncio.CancelledError:
            pass
    
    async def _profile_cpu(self) -> List[ProfileMetric]:
        """Profile CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        return [
            ProfileMetric("cpu_percent", cpu_percent, "%", time.time()),
            ProfileMetric("cpu_count", cpu_count, "cores", time.time()),
            ProfileMetric("load_avg_1m", load_avg[0], "load", time.time()),
            ProfileMetric("load_avg_5m", load_avg[1], "load", time.time()),
            ProfileMetric("load_avg_15m", load_avg[2], "load", time.time()),
        ]
    
    async def _profile_memory(self) -> List[ProfileMetric]:
        """Profile memory usage"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Garbage collection stats
        gc_stats = gc.get_stats()
        gc_counts = gc.get_count()
        
        metrics = [
            ProfileMetric("memory_total", memory.total, "bytes", time.time()),
            ProfileMetric("memory_available", memory.available, "bytes", time.time()),
            ProfileMetric("memory_used", memory.used, "bytes", time.time()),
            ProfileMetric("memory_percent", memory.percent, "%", time.time()),
            ProfileMetric("swap_total", swap.total, "bytes", time.time()),
            ProfileMetric("swap_used", swap.used, "bytes", time.time()),
            ProfileMetric("process_rss", process_memory.rss, "bytes", time.time()),
            ProfileMetric("process_vms", process_memory.vms, "bytes", time.time()),
            ProfileMetric("gc_gen0", gc_counts[0], "objects", time.time()),
            ProfileMetric("gc_gen1", gc_counts[1], "objects", time.time()),
            ProfileMetric("gc_gen2", gc_counts[2], "objects", time.time()),
        ]
        
        return metrics
    
    async def _profile_io(self) -> List[ProfileMetric]:
        """Profile I/O operations"""
        io_counters = psutil.disk_io_counters()
        process = psutil.Process()
        process_io = process.io_counters()
        
        if io_counters:
            return [
                ProfileMetric("disk_read_bytes", io_counters.read_bytes, "bytes", time.time()),
                ProfileMetric("disk_write_bytes", io_counters.write_bytes, "bytes", time.time()),
                ProfileMetric("disk_read_count", io_counters.read_count, "operations", time.time()),
                ProfileMetric("disk_write_count", io_counters.write_count, "operations", time.time()),
                ProfileMetric("process_read_bytes", process_io.read_bytes, "bytes", time.time()),
                ProfileMetric("process_write_bytes", process_io.write_bytes, "bytes", time.time()),
            ]
        return []
    
    async def _profile_network(self) -> List[ProfileMetric]:
        """Profile network usage"""
        net_io = psutil.net_io_counters()
        connections = len(psutil.net_connections())
        
        if net_io:
            return [
                ProfileMetric("net_bytes_sent", net_io.bytes_sent, "bytes", time.time()),
                ProfileMetric("net_bytes_recv", net_io.bytes_recv, "bytes", time.time()),
                ProfileMetric("net_packets_sent", net_io.packets_sent, "packets", time.time()),
                ProfileMetric("net_packets_recv", net_io.packets_recv, "packets", time.time()),
                ProfileMetric("net_connections", connections, "connections", time.time()),
            ]
        return []
    
    def register_custom_profiler(self, profiler_type: ProfilerType, profiler_func: Callable):
        """Register a custom profiler function"""
        self.profilers[profiler_type] = profiler_func
        logging.info(f"Registered custom profiler: {profiler_type}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a profiling session"""
        # Find session in active or completed sessions
        session = self.active_sessions.get(session_id)
        if not session:
            # Look for completed session in metrics buffer
            session_metrics = [m for m in self.metrics_buffer 
                             if m.context.get('session_id') == session_id]
            if not session_metrics:
                raise ValueError(f"Session '{session_id}' not found")
        else:
            session_metrics = session.metrics
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in session_metrics:
            metric_groups[metric.name].append(metric.value)
        
        # Calculate statistics for each metric
        summary = {}
        for metric_name, values in metric_groups.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return {
            "session_id": session_id,
            "duration": (session.end_time or time.time()) - session.start_time if session else 0,
            "metric_count": len(session_metrics),
            "metrics": summary
        }
    
    @contextmanager
    def profile_context(self, name: str, profiler_types: List[ProfilerType] = None):
        """Context manager for profiling code blocks"""
        if profiler_types is None:
            profiler_types = [ProfilerType.CPU, ProfilerType.MEMORY]
        
        session_id = f"context_{name}_{int(time.time())}"
        
        try:
            # Start profiling (synchronous version)
            loop = asyncio.get_event_loop()
            session = loop.run_until_complete(
                self.start_profiling(session_id, profiler_types)
            )
            yield session
        finally:
            # Stop profiling
            loop.run_until_complete(self.stop_profiling(session_id))

### 1.2 Performance Bottleneck Detector

class BottleneckType(Enum):
    """Types of performance bottlenecks"""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    LOCK_CONTENTION = "lock_contention"
    ALGORITHM_INEFFICIENCY = "algorithm_inefficiency"

@dataclass
class PerformanceBottleneck:
    """Detected performance bottleneck"""
    type: BottleneckType
    severity: str  # low, medium, high, critical
    description: str
    location: str  # function, module, or component
    impact_score: float  # 0-100
    recommendations: List[str]
    metrics: Dict[str, float]
    timestamp: float

class BottleneckDetector:
    """Detects performance bottlenecks in agent systems"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.detection_rules: Dict[BottleneckType, Callable] = {}
        self.thresholds: Dict[str, float] = {
            "cpu_high": 80.0,
            "memory_high": 85.0,
            "io_wait_high": 50.0,
            "network_latency_high": 1000.0,  # ms
            "gc_frequency_high": 10.0,  # per second
        }
        
        self._setup_detection_rules()
    
    def _setup_detection_rules(self):
        """Set up bottleneck detection rules"""
        self.detection_rules[BottleneckType.CPU_BOUND] = self._detect_cpu_bottleneck
        self.detection_rules[BottleneckType.MEMORY_BOUND] = self._detect_memory_bottleneck
        self.detection_rules[BottleneckType.IO_BOUND] = self._detect_io_bottleneck
        self.detection_rules[BottleneckType.NETWORK_BOUND] = self._detect_network_bottleneck
    
    async def analyze_session(self, session_id: str) -> List[PerformanceBottleneck]:
        """Analyze a profiling session for bottlenecks"""
        session_summary = self.profiler.get_session_summary(session_id)
        bottlenecks = []
        
        # Run each detection rule
        for bottleneck_type, detection_func in self.detection_rules.items():
            try:
                detected = await detection_func(session_summary)
                if detected:
                    bottlenecks.extend(detected)
            except Exception as e:
                logging.warning(f"Bottleneck detection failed for {bottleneck_type}: {e}")
        
        # Sort by impact score (highest first)
        bottlenecks.sort(key=lambda x: x.impact_score, reverse=True)
        
        return bottlenecks
```

## 3. Cost Optimization and Resource Management

### 3.1 Cost-Aware Resource Optimizer

```python
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import defaultdict

class ResourceType(Enum):
    """Types of resources that can be optimized"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"

class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    BALANCE_COST_PERFORMANCE = "balance_cost_performance"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"

@dataclass
class ResourceCost:
    """Cost information for a resource"""
    resource_type: ResourceType
    unit_cost: float  # Cost per unit per hour
    currency: str = "USD"
    billing_model: str = "hourly"  # hourly, monthly, pay-per-use
    minimum_charge: float = 0.0
    
@dataclass
class ResourceUsage:
    """Resource usage information"""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    allocated: float
    utilization_rate: float  # current_usage / allocated
    timestamp: float

@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation"""
    resource_type: ResourceType
    current_cost: float
    optimized_cost: float
    savings: float
    savings_percentage: float
    action: str
    description: str
    risk_level: str  # low, medium, high
    implementation_effort: str  # low, medium, high
    estimated_impact: Dict[str, float]

class CostOptimizer:
    """Intelligent cost optimization system"""
    
    def __init__(self):
        self.resource_costs: Dict[ResourceType, ResourceCost] = {}
        self.usage_history: Dict[ResourceType, List[ResourceUsage]] = defaultdict(list)
        self.optimization_rules: List[Callable] = []
        self.cost_thresholds: Dict[str, float] = {
            "high_cost_alert": 1000.0,  # USD per month
            "waste_threshold": 0.3,  # 30% utilization threshold
            "overprovisioning_threshold": 0.8  # 80% peak utilization
        }
        
        self._setup_default_optimization_rules()
    
    def register_resource_cost(self, resource_cost: ResourceCost):
        """Register cost information for a resource type"""
        self.resource_costs[resource_cost.resource_type] = resource_cost
    
    def record_usage(self, usage: ResourceUsage):
        """Record resource usage data"""
        self.usage_history[usage.resource_type].append(usage)
        
        # Keep only recent history (last 30 days)
        cutoff_time = time.time() - (30 * 24 * 3600)
        self.usage_history[usage.resource_type] = [
            u for u in self.usage_history[usage.resource_type]
            if u.timestamp > cutoff_time
        ]
    
    async def analyze_costs(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """Analyze current costs and usage patterns"""
        analysis = {
            "total_cost": 0.0,
            "resource_costs": {},
            "cost_trends": {},
            "utilization_analysis": {},
            "waste_analysis": {}
        }
        
        cutoff_time = time.time() - (time_period_hours * 3600)
        
        for resource_type, usage_list in self.usage_history.items():
            recent_usage = [u for u in usage_list if u.timestamp > cutoff_time]
            
            if not recent_usage or resource_type not in self.resource_costs:
                continue
            
            cost_info = self.resource_costs[resource_type]
            
            # Calculate costs
            total_usage = sum(u.allocated * (time_period_hours / len(recent_usage)) 
                            for u in recent_usage)
            resource_cost = total_usage * cost_info.unit_cost
            
            analysis["resource_costs"][resource_type.value] = {
                "cost": resource_cost,
                "usage": total_usage,
                "unit_cost": cost_info.unit_cost
            }
            analysis["total_cost"] += resource_cost
            
            # Utilization analysis
            avg_utilization = statistics.mean([u.utilization_rate for u in recent_usage])
            peak_utilization = max([u.utilization_rate for u in recent_usage])
            
            analysis["utilization_analysis"][resource_type.value] = {
                "average": avg_utilization,
                "peak": peak_utilization,
                "efficiency": "good" if avg_utilization > 0.7 else "poor"
            }
            
            # Waste analysis
            if avg_utilization < self.cost_thresholds["waste_threshold"]:
                wasted_capacity = sum(u.allocated * (1 - u.utilization_rate) 
                                    for u in recent_usage) / len(recent_usage)
                wasted_cost = wasted_capacity * cost_info.unit_cost * time_period_hours
                
                analysis["waste_analysis"][resource_type.value] = {
                    "wasted_capacity": wasted_capacity,
                    "wasted_cost": wasted_cost,
                    "waste_percentage": (1 - avg_utilization) * 100
                }
        
        return analysis
    
    async def generate_optimization_recommendations(self, 
                                                  strategy: OptimizationStrategy = OptimizationStrategy.BALANCE_COST_PERFORMANCE) -> List[CostOptimizationRecommendation]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        for rule in self.optimization_rules:
            try:
                rule_recommendations = await rule(strategy)
                recommendations.extend(rule_recommendations)
            except Exception as e:
                logging.warning(f"Optimization rule failed: {e}")
        
        # Sort by potential savings
        recommendations.sort(key=lambda x: x.savings, reverse=True)
        
        return recommendations
    
    def _setup_default_optimization_rules(self):
        """Set up default optimization rules"""
        self.optimization_rules = [
            self._rule_rightsizing,
            self._rule_reserved_instances,
            self._rule_spot_instances,
            self._rule_auto_scaling,
            self._rule_storage_optimization
        ]
    
    async def _rule_rightsizing(self, strategy: OptimizationStrategy) -> List[CostOptimizationRecommendation]:
        """Rule for rightsizing resources based on actual usage"""
        recommendations = []
        
        for resource_type, usage_list in self.usage_history.items():
            if not usage_list or resource_type not in self.resource_costs:
                continue
            
            recent_usage = usage_list[-168:]  # Last week
            if len(recent_usage) < 24:  # Need at least 24 hours of data
                continue
            
            avg_usage = statistics.mean([u.current_usage for u in recent_usage])
            peak_usage = max([u.current_usage for u in recent_usage])
            current_allocated = recent_usage[-1].allocated
            
            # Determine optimal allocation based on strategy
            if strategy == OptimizationStrategy.MINIMIZE_COST:
                optimal_allocation = avg_usage * 1.2  # 20% buffer
            elif strategy == OptimizationStrategy.MAXIMIZE_PERFORMANCE:
                optimal_allocation = peak_usage * 1.5  # 50% buffer
            else:  # BALANCE_COST_PERFORMANCE
                optimal_allocation = max(avg_usage * 1.3, peak_usage * 1.1)
            
            if abs(optimal_allocation - current_allocated) / current_allocated > 0.1:  # 10% threshold
                cost_info = self.resource_costs[resource_type]
                current_cost = current_allocated * cost_info.unit_cost * 24 * 30  # Monthly
                optimized_cost = optimal_allocation * cost_info.unit_cost * 24 * 30
                savings = current_cost - optimized_cost
                
                if savings > 0:
                    action = "downsize" if optimal_allocation < current_allocated else "upsize"
                    risk_level = "low" if action == "downsize" else "medium"
                    
                    recommendations.append(CostOptimizationRecommendation(
                        resource_type=resource_type,
                        current_cost=current_cost,
                        optimized_cost=optimized_cost,
                        savings=savings,
                        savings_percentage=(savings / current_cost) * 100,
                        action=f"{action}_resource",
                        description=f"{action.capitalize()} {resource_type.value} from {current_allocated:.2f} to {optimal_allocation:.2f} units",
                        risk_level=risk_level,
                        implementation_effort="low",
                        estimated_impact={
                            "cost_reduction": savings,
                            "performance_impact": 0 if action == "downsize" else 10
                        }
                    ))
        
        return recommendations
    
    async def _rule_reserved_instances(self, strategy: OptimizationStrategy) -> List[CostOptimizationRecommendation]:
        """Rule for recommending reserved instances for stable workloads"""
        recommendations = []
        
        # Analyze usage stability
        for resource_type, usage_list in self.usage_history.items():
            if len(usage_list) < 168:  # Need at least a week of data
                continue
            
            # Calculate usage stability (coefficient of variation)
            allocations = [u.allocated for u in usage_list]
            if not allocations:
                continue
            
            mean_allocation = statistics.mean(allocations)
            std_allocation = statistics.stdev(allocations) if len(allocations) > 1 else 0
            cv = std_allocation / mean_allocation if mean_allocation > 0 else float('inf')
            
            # If usage is stable (low coefficient of variation), recommend reserved instances
            if cv < 0.2 and mean_allocation > 0:  # Stable usage
                cost_info = self.resource_costs.get(resource_type)
                if not cost_info:
                    continue
                
                # Assume 30% savings with reserved instances
                current_monthly_cost = mean_allocation * cost_info.unit_cost * 24 * 30
                reserved_monthly_cost = current_monthly_cost * 0.7
                monthly_savings = current_monthly_cost - reserved_monthly_cost
                
                if monthly_savings > 50:  # Minimum savings threshold
                    recommendations.append(CostOptimizationRecommendation(
                        resource_type=resource_type,
                        current_cost=current_monthly_cost,
                        optimized_cost=reserved_monthly_cost,
                        savings=monthly_savings,
                        savings_percentage=30.0,
                        action="purchase_reserved_instances",
                        description=f"Purchase reserved instances for stable {resource_type.value} usage",
                        risk_level="low",
                        implementation_effort="medium",
                        estimated_impact={
                            "cost_reduction": monthly_savings,
                            "commitment_period": 12  # months
                        }
                    ))
        
        return recommendations
    
    async def _rule_spot_instances(self, strategy: OptimizationStrategy) -> List[CostOptimizationRecommendation]:
        """Rule for recommending spot instances for fault-tolerant workloads"""
        recommendations = []
        
        # This would analyze workload characteristics and recommend spot instances
        # for appropriate use cases (batch processing, development environments, etc.)
        
        return recommendations
    
    async def _rule_auto_scaling(self, strategy: OptimizationStrategy) -> List[CostOptimizationRecommendation]:
        """Rule for recommending auto-scaling configurations"""
        recommendations = []
        
        for resource_type, usage_list in self.usage_history.items():
            if len(usage_list) < 48:  # Need at least 2 days of data
                continue
            
            # Analyze usage patterns for scaling opportunities
            hourly_usage = defaultdict(list)
            for usage in usage_list:
                hour = int((usage.timestamp % 86400) / 3600)  # Hour of day
                hourly_usage[hour].append(usage.utilization_rate)
            
            # Check for significant usage variation
            hourly_averages = [statistics.mean(usage_list) for usage_list in hourly_usage.values()]
            if hourly_averages:
                usage_variation = max(hourly_averages) - min(hourly_averages)
                
                if usage_variation > 0.3:  # 30% variation
                    cost_info = self.resource_costs.get(resource_type)
                    if cost_info:
                        # Estimate savings from auto-scaling
                        current_allocation = usage_list[-1].allocated
                        avg_needed = statistics.mean([max(hourly_usage[h]) for h in hourly_usage]) * current_allocation
                        
                        current_monthly_cost = current_allocation * cost_info.unit_cost * 24 * 30
                        optimized_monthly_cost = avg_needed * cost_info.unit_cost * 24 * 30
                        monthly_savings = current_monthly_cost - optimized_monthly_cost
                        
                        if monthly_savings > 20:  # Minimum savings threshold
                            recommendations.append(CostOptimizationRecommendation(
                                resource_type=resource_type,
                                current_cost=current_monthly_cost,
                                optimized_cost=optimized_monthly_cost,
                                savings=monthly_savings,
                                savings_percentage=(monthly_savings / current_monthly_cost) * 100,
                                action="implement_auto_scaling",
                                description=f"Implement auto-scaling for {resource_type.value} based on usage patterns",
                                risk_level="medium",
                                implementation_effort="high",
                                estimated_impact={
                                    "cost_reduction": monthly_savings,
                                    "usage_variation": usage_variation
                                }
                            ))
        
        return recommendations
    
    async def _rule_storage_optimization(self, strategy: OptimizationStrategy) -> List[CostOptimizationRecommendation]:
        """Rule for storage optimization (tiering, compression, etc.)"""
        recommendations = []
        
        # This would analyze storage access patterns and recommend
        # appropriate storage tiers, compression, or cleanup
        
        return recommendations
    
    def add_optimization_rule(self, rule_func: Callable):
        """Add a custom optimization rule"""
        self.optimization_rules.append(rule_func)
    
    async def simulate_optimization(self, recommendations: List[CostOptimizationRecommendation]) -> Dict[str, Any]:
        """Simulate the impact of applying optimization recommendations"""
        total_current_cost = sum(rec.current_cost for rec in recommendations)
        total_optimized_cost = sum(rec.optimized_cost for rec in recommendations)
        total_savings = total_current_cost - total_optimized_cost
        
        # Risk assessment
        risk_distribution = defaultdict(int)
        for rec in recommendations:
            risk_distribution[rec.risk_level] += 1
        
        # Implementation effort assessment
        effort_distribution = defaultdict(int)
        for rec in recommendations:
            effort_distribution[rec.implementation_effort] += 1
        
        return {
            "total_current_cost": total_current_cost,
            "total_optimized_cost": total_optimized_cost,
            "total_savings": total_savings,
            "savings_percentage": (total_savings / total_current_cost * 100) if total_current_cost > 0 else 0,
            "recommendation_count": len(recommendations),
            "risk_distribution": dict(risk_distribution),
            "effort_distribution": dict(effort_distribution),
            "implementation_timeline": self._estimate_implementation_timeline(recommendations)
        }
    
    def _estimate_implementation_timeline(self, recommendations: List[CostOptimizationRecommendation]) -> Dict[str, int]:
        """Estimate implementation timeline for recommendations"""
        effort_to_days = {"low": 1, "medium": 7, "high": 30}
        
        timeline = {"immediate": 0, "short_term": 0, "long_term": 0}
        
        for rec in recommendations:
            days = effort_to_days.get(rec.implementation_effort, 7)
            if days <= 1:
                timeline["immediate"] += 1
            elif days <= 7:
                timeline["short_term"] += 1
            else:
                timeline["long_term"] += 1
        
        return timeline

### 3.2 Intelligent Resource Scheduler

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class ScheduledTask:
    """Task to be scheduled"""
    task_id: str
    priority: TaskPriority
    resource_requirements: Dict[ResourceType, float]
    estimated_duration: float
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    cost_budget: Optional[float] = None
    preferred_resources: List[str] = field(default_factory=list)
    
class IntelligentResourceScheduler:
    """AI-powered resource scheduler for cost optimization"""
    
    def __init__(self, cost_optimizer: CostOptimizer):
        self.cost_optimizer = cost_optimizer
        self.available_resources: Dict[str, Dict[ResourceType, float]] = {}
        self.resource_costs: Dict[str, Dict[ResourceType, float]] = {}
        self.task_queue: List[ScheduledTask] = []
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: List[ScheduledTask] = []
        
    def register_resource_pool(self, pool_id: str, 
                             resources: Dict[ResourceType, float],
                             costs: Dict[ResourceType, float]):
        """Register an available resource pool"""
        self.available_resources[pool_id] = resources
        self.resource_costs[pool_id] = costs
    
    async def schedule_task(self, task: ScheduledTask) -> Dict[str, Any]:
        """Schedule a task with cost optimization"""
        # Find optimal resource allocation
        allocation = await self._find_optimal_allocation(task)
        
        if allocation:
            # Reserve resources
            self._reserve_resources(allocation["pool_id"], task.resource_requirements)
            
            # Add to running tasks
            self.running_tasks[task.task_id] = task
            
            return {
                "status": "scheduled",
                "allocation": allocation,
                "estimated_cost": allocation["estimated_cost"],
                "start_time": time.time()
            }
        else:
            # Add to queue
            self.task_queue.append(task)
            return {
                "status": "queued",
                "position": len(self.task_queue),
                "estimated_wait_time": self._estimate_wait_time(task)
            }
    
    async def _find_optimal_allocation(self, task: ScheduledTask) -> Optional[Dict[str, Any]]:
        """Find optimal resource allocation for a task"""
        best_allocation = None
        best_score = float('inf')
        
        for pool_id, available in self.available_resources.items():
            # Check if pool has sufficient resources
            if self._can_accommodate(available, task.resource_requirements):
                # Calculate cost
                pool_costs = self.resource_costs[pool_id]
                estimated_cost = sum(
                    task.resource_requirements[res_type] * pool_costs.get(res_type, 0) * task.estimated_duration
                    for res_type in task.resource_requirements
                )
                
                # Check budget constraint
                if task.cost_budget and estimated_cost > task.cost_budget:
                    continue
                
                # Calculate optimization score (lower is better)
                score = self._calculate_allocation_score(task, pool_id, estimated_cost)
                
                if score < best_score:
                    best_score = score
                    best_allocation = {
                        "pool_id": pool_id,
                        "estimated_cost": estimated_cost,
                        "score": score
                    }
        
        return best_allocation
    
    def _can_accommodate(self, available: Dict[ResourceType, float], 
                       required: Dict[ResourceType, float]) -> bool:
        """Check if available resources can accommodate requirements"""
        for res_type, amount in required.items():
            if available.get(res_type, 0) < amount:
                return False
        return True
    
    def _calculate_allocation_score(self, task: ScheduledTask, pool_id: str, cost: float) -> float:
        """Calculate allocation score for optimization"""
        score = cost  # Base score is cost
        
        # Priority adjustment
        priority_weights = {
            TaskPriority.CRITICAL: 0.1,
            TaskPriority.HIGH: 0.5,
            TaskPriority.MEDIUM: 1.0,
            TaskPriority.LOW: 2.0,
            TaskPriority.BACKGROUND: 5.0
        }
        score *= priority_weights[task.priority]
        
        # Deadline pressure
        if task.deadline:
            time_to_deadline = task.deadline - time.time()
            if time_to_deadline < task.estimated_duration * 2:
                score *= 0.5  # Prioritize urgent tasks
        
        # Preferred resource bonus
        if pool_id in task.preferred_resources:
            score *= 0.8
        
        return score
    
    def _reserve_resources(self, pool_id: str, requirements: Dict[ResourceType, float]):
        """Reserve resources in a pool"""
        for res_type, amount in requirements.items():
            self.available_resources[pool_id][res_type] -= amount
    
    def _release_resources(self, pool_id: str, requirements: Dict[ResourceType, float]):
        """Release resources back to a pool"""
        for res_type, amount in requirements.items():
            self.available_resources[pool_id][res_type] += amount
    
    async def complete_task(self, task_id: str, actual_duration: float, actual_cost: float):
        """Mark a task as completed and update metrics"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            
            # Release resources
            # (This would need to track which pool was used)
            
            # Move to completed
            del self.running_tasks[task_id]
            self.completed_tasks.append(task)
            
            # Update cost tracking
            # (Implementation would update cost optimizer with actual usage)
            
            # Process queue
            await self._process_task_queue()
    
    async def _process_task_queue(self):
        """Process queued tasks"""
        scheduled_tasks = []
        
        for task in self.task_queue:
            allocation = await self._find_optimal_allocation(task)
            if allocation:
                await self.schedule_task(task)
                scheduled_tasks.append(task)
        
        # Remove scheduled tasks from queue
        for task in scheduled_tasks:
            self.task_queue.remove(task)
    
    def _estimate_wait_time(self, task: ScheduledTask) -> float:
        """Estimate wait time for a queued task"""
        # Simple estimation based on running tasks
        running_durations = [t.estimated_duration for t in self.running_tasks.values()]
        return statistics.mean(running_durations) if running_durations else 0
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            "running_tasks": len(self.running_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "resource_utilization": self._calculate_resource_utilization(),
            "average_wait_time": self._calculate_average_wait_time()
        }
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization"""
        utilization = {}
        
        for pool_id, resources in self.available_resources.items():
            pool_utilization = {}
            for res_type, available in resources.items():
                # This would need to track total capacity
                total_capacity = available + sum(
                    task.resource_requirements.get(res_type, 0)
                    for task in self.running_tasks.values()
                )
                if total_capacity > 0:
                    used = total_capacity - available
                    pool_utilization[res_type.value] = (used / total_capacity) * 100
            
            utilization[pool_id] = pool_utilization
        
        return utilization
    
    def _calculate_average_wait_time(self) -> float:
        """Calculate average wait time for completed tasks"""
        # This would track actual wait times
        return 0.0  # Placeholder
```

## 4. Security Hardening and Compliance

### 4.1 Security Scanner and Vulnerability Assessment

```python
import hashlib
import re
import subprocess
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from pathlib import Path

class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    INJECTION = "injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_EXPOSURE = "data_exposure"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    CRYPTOGRAPHY = "cryptography"
    NETWORK = "network"
    LOGGING = "logging"

class SeverityLevel(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SecurityVulnerability:
    """Security vulnerability information"""
    vulnerability_id: str
    type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    location: str  # file, function, or component
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    detected_at: float = field(default_factory=time.time)

@dataclass
class SecurityScanResult:
    """Security scan results"""
    scan_id: str
    scan_type: str
    start_time: float
    end_time: float
    target: str
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SecurityScanner:
    """Comprehensive security scanner for agent systems"""
    
    def __init__(self):
        self.scan_rules: Dict[VulnerabilityType, List[Callable]] = {}
        self.vulnerability_database: Dict[str, SecurityVulnerability] = {}
        self.scan_history: List[SecurityScanResult] = []
        self.whitelist: Set[str] = set()
        
        self._setup_default_scan_rules()
    
    def _setup_default_scan_rules(self):
        """Set up default security scan rules"""
        self.scan_rules[VulnerabilityType.INJECTION] = [self._scan_sql_injection, self._scan_command_injection]
        self.scan_rules[VulnerabilityType.AUTHENTICATION] = [self._scan_weak_authentication]
        self.scan_rules[VulnerabilityType.AUTHORIZATION] = [self._scan_authorization_bypass]
        self.scan_rules[VulnerabilityType.DATA_EXPOSURE] = [self._scan_data_exposure]
        self.scan_rules[VulnerabilityType.CONFIGURATION] = [self._scan_misconfigurations]
        self.scan_rules[VulnerabilityType.DEPENDENCY] = [self._scan_vulnerable_dependencies]
        self.scan_rules[VulnerabilityType.CRYPTOGRAPHY] = [self._scan_crypto_issues]
    
    async def scan_codebase(self, target_path: str, scan_types: List[VulnerabilityType] = None) -> SecurityScanResult:
        """Scan codebase for security vulnerabilities"""
        scan_id = f"scan_{int(time.time())}"
        start_time = time.time()
        
        if scan_types is None:
            scan_types = list(VulnerabilityType)
        
        vulnerabilities = []
        
        # Run each scan type
        for vuln_type in scan_types:
            if vuln_type in self.scan_rules:
                for scan_rule in self.scan_rules[vuln_type]:
                    try:
                        found_vulns = await scan_rule(target_path)
                        vulnerabilities.extend(found_vulns)
                    except Exception as e:
                        logging.warning(f"Security scan rule {scan_rule.__name__} failed: {e}")
        
        # Filter whitelisted vulnerabilities
        vulnerabilities = [v for v in vulnerabilities if v.vulnerability_id not in self.whitelist]
        
        # Create summary
        summary = {severity.value: 0 for severity in SeverityLevel}
        for vuln in vulnerabilities:
            summary[vuln.severity.value] += 1
        
        end_time = time.time()
        
        result = SecurityScanResult(
            scan_id=scan_id,
            scan_type="codebase",
            start_time=start_time,
            end_time=end_time,
            target=target_path,
            vulnerabilities=vulnerabilities,
            summary=summary,
            metadata={"scan_duration": end_time - start_time}
        )
        
        self.scan_history.append(result)
        return result
    
    async def _scan_sql_injection(self, target_path: str) -> List[SecurityVulnerability]:
        """Scan for SQL injection vulnerabilities"""
        vulnerabilities = []
        
        # Patterns that might indicate SQL injection vulnerabilities
        sql_patterns = [
            r'execute\s*\(.*\+.*\)',  # String concatenation in execute
            r'query\s*\(.*\+.*\)',    # String concatenation in query
            r'SELECT.*\+.*FROM',      # Direct string concatenation in SQL
            r'INSERT.*\+.*VALUES',    # String concatenation in INSERT
        ]
        
        try:
            for file_path in Path(target_path).rglob("*.py"):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in sql_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln_id = hashlib.md5(f"{file_path}:{line_num}:{pattern}".encode()).hexdigest()[:8]
                            
                            vulnerabilities.append(SecurityVulnerability(
                                vulnerability_id=vuln_id,
                                type=VulnerabilityType.INJECTION,
                                severity=SeverityLevel.HIGH,
                                title="Potential SQL Injection",
                                description=f"Potential SQL injection vulnerability detected in {file_path}:{line_num}",
                                location=f"{file_path}:{line_num}",
                                remediation=[
                                    "Use parameterized queries or prepared statements",
                                    "Validate and sanitize all user inputs",
                                    "Use ORM frameworks with built-in protection"
                                ]
                            ))
        except Exception as e:
            logging.warning(f"SQL injection scan failed: {e}")
        
        return vulnerabilities
    
    async def _scan_command_injection(self, target_path: str) -> List[SecurityVulnerability]:
        """Scan for command injection vulnerabilities"""
        vulnerabilities = []
        
        # Patterns that might indicate command injection
        command_patterns = [
            r'subprocess\.call\s*\(.*\+.*\)',
            r'os\.system\s*\(.*\+.*\)',
            r'subprocess\.run\s*\(.*\+.*\)',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        try:
            for file_path in Path(target_path).rglob("*.py"):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in command_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln_id = hashlib.md5(f"{file_path}:{line_num}:{pattern}".encode()).hexdigest()[:8]
                            
                            severity = SeverityLevel.CRITICAL if 'eval' in pattern or 'exec' in pattern else SeverityLevel.HIGH
                            
                            vulnerabilities.append(SecurityVulnerability(
                                vulnerability_id=vuln_id,
                                type=VulnerabilityType.INJECTION,
                                severity=severity,
                                title="Potential Command Injection",
                                description=f"Potential command injection vulnerability detected in {file_path}:{line_num}",
                                location=f"{file_path}:{line_num}",
                                remediation=[
                                    "Use subprocess with shell=False and argument lists",
                                    "Validate and sanitize all user inputs",
                                    "Avoid eval() and exec() functions",
                                    "Use allowlists for permitted commands"
                                ]
                            ))
        except Exception as e:
            logging.warning(f"Command injection scan failed: {e}")
        
        return vulnerabilities
    
    async def _scan_weak_authentication(self, target_path: str) -> List[SecurityVulnerability]:
        """Scan for weak authentication mechanisms"""
        vulnerabilities = []
        
        # Patterns indicating weak authentication
        auth_patterns = [
            r'password\s*==\s*["\'][^"\']["\']',  # Hardcoded passwords
            r'api_key\s*=\s*["\'][^"\']["\']',     # Hardcoded API keys
            r'secret\s*=\s*["\'][^"\']["\']',      # Hardcoded secrets
            r'md5\s*\(',                          # Weak hashing
            r'sha1\s*\(',                         # Weak hashing
        ]
        
        try:
            for file_path in Path(target_path).rglob("*.py"):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in auth_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln_id = hashlib.md5(f"{file_path}:{line_num}:{pattern}".encode()).hexdigest()[:8]
                            
                            if 'password' in pattern or 'api_key' in pattern or 'secret' in pattern:
                                title = "Hardcoded Credentials"
                                severity = SeverityLevel.CRITICAL
                                remediation = [
                                    "Use environment variables for secrets",
                                    "Implement secure credential management",
                                    "Use key management services"
                                ]
                            else:
                                title = "Weak Cryptographic Hash"
                                severity = SeverityLevel.MEDIUM
                                remediation = [
                                    "Use SHA-256 or stronger hashing algorithms",
                                    "Implement proper salt for password hashing",
                                    "Consider using bcrypt or Argon2"
                                ]
                            
                            vulnerabilities.append(SecurityVulnerability(
                                vulnerability_id=vuln_id,
                                type=VulnerabilityType.AUTHENTICATION,
                                severity=severity,
                                title=title,
                                description=f"Weak authentication mechanism detected in {file_path}:{line_num}",
                                location=f"{file_path}:{line_num}",
                                remediation=remediation
                            ))
        except Exception as e:
            logging.warning(f"Authentication scan failed: {e}")
        
        return vulnerabilities
    
    async def _scan_authorization_bypass(self, target_path: str) -> List[SecurityVulnerability]:
        """Scan for authorization bypass vulnerabilities"""
        vulnerabilities = []
        # Implementation would check for missing authorization checks
        return vulnerabilities
    
    async def _scan_data_exposure(self, target_path: str) -> List[SecurityVulnerability]:
        """Scan for data exposure vulnerabilities"""
        vulnerabilities = []
        
        # Patterns indicating potential data exposure
        exposure_patterns = [
            r'print\s*\(.*password.*\)',
            r'logging\..*\(.*password.*\)',
            r'console\.log\s*\(.*password.*\)',
            r'debug\s*=\s*True',
        ]
        
        try:
            for file_path in Path(target_path).rglob("*.py"):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in exposure_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln_id = hashlib.md5(f"{file_path}:{line_num}:{pattern}".encode()).hexdigest()[:8]
                            
                            vulnerabilities.append(SecurityVulnerability(
                                vulnerability_id=vuln_id,
                                type=VulnerabilityType.DATA_EXPOSURE,
                                severity=SeverityLevel.MEDIUM,
                                title="Potential Data Exposure",
                                description=f"Potential sensitive data exposure in {file_path}:{line_num}",
                                location=f"{file_path}:{line_num}",
                                remediation=[
                                    "Remove or mask sensitive data from logs",
                                    "Disable debug mode in production",
                                    "Implement proper logging practices"
                                ]
                            ))
        except Exception as e:
            logging.warning(f"Data exposure scan failed: {e}")
        
        return vulnerabilities
    
    async def _scan_misconfigurations(self, target_path: str) -> List[SecurityVulnerability]:
        """Scan for security misconfigurations"""
        vulnerabilities = []
        # Implementation would check for insecure configurations
        return vulnerabilities
    
    async def _scan_vulnerable_dependencies(self, target_path: str) -> List[SecurityVulnerability]:
        """Scan for vulnerable dependencies"""
        vulnerabilities = []
        
        # Check requirements.txt for known vulnerable packages
        requirements_file = Path(target_path) / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    requirements = f.read()
                
                # This would integrate with vulnerability databases
                # For demo purposes, check for some known vulnerable patterns
                vulnerable_patterns = [
                    r'django==1\.',  # Old Django versions
                    r'flask==0\.',   # Old Flask versions
                    r'requests==2\.1[0-9]\.',  # Old requests versions
                ]
                
                for line_num, line in enumerate(requirements.split('\n'), 1):
                    for pattern in vulnerable_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln_id = hashlib.md5(f"requirements.txt:{line_num}:{pattern}".encode()).hexdigest()[:8]
                            
                            vulnerabilities.append(SecurityVulnerability(
                                vulnerability_id=vuln_id,
                                type=VulnerabilityType.DEPENDENCY,
                                severity=SeverityLevel.HIGH,
                                title="Vulnerable Dependency",
                                description=f"Potentially vulnerable dependency in requirements.txt:{line_num}",
                                location=f"requirements.txt:{line_num}",
                                remediation=[
                                    "Update to the latest secure version",
                                    "Review security advisories",
                                    "Use dependency scanning tools"
                                ]
                            ))
            except Exception as e:
                logging.warning(f"Dependency scan failed: {e}")
        
        return vulnerabilities
    
    async def _scan_crypto_issues(self, target_path: str) -> List[SecurityVulnerability]:
        """Scan for cryptographic issues"""
        vulnerabilities = []
        
        # Patterns indicating cryptographic issues
        crypto_patterns = [
            r'random\.random\s*\(',  # Weak random number generation
            r'DES\s*\(',             # Weak encryption
            r'RC4\s*\(',             # Weak encryption
            r'ssl_verify\s*=\s*False',  # SSL verification disabled
        ]
        
        try:
            for file_path in Path(target_path).rglob("*.py"):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in crypto_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln_id = hashlib.md5(f"{file_path}:{line_num}:{pattern}".encode()).hexdigest()[:8]
                            
                            vulnerabilities.append(SecurityVulnerability(
                                vulnerability_id=vuln_id,
                                type=VulnerabilityType.CRYPTOGRAPHY,
                                severity=SeverityLevel.HIGH,
                                title="Cryptographic Weakness",
                                description=f"Cryptographic weakness detected in {file_path}:{line_num}",
                                location=f"{file_path}:{line_num}",
                                remediation=[
                                    "Use cryptographically secure random number generators",
                                    "Use strong encryption algorithms (AES-256)",
                                    "Enable SSL/TLS verification",
                                    "Follow cryptographic best practices"
                                ]
                            ))
        except Exception as e:
            logging.warning(f"Cryptography scan failed: {e}")
        
        return vulnerabilities
    
    def add_to_whitelist(self, vulnerability_id: str):
        """Add vulnerability to whitelist (mark as false positive)"""
        self.whitelist.add(vulnerability_id)
    
    def get_scan_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get security scan summary for the last N days"""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_scans = [scan for scan in self.scan_history if scan.start_time > cutoff_time]
        
        if not recent_scans:
            return {"total_scans": 0}
        
        # Aggregate statistics
        total_vulnerabilities = sum(len(scan.vulnerabilities) for scan in recent_scans)
        severity_counts = {severity.value: 0 for severity in SeverityLevel}
        type_counts = {vuln_type.value: 0 for vuln_type in VulnerabilityType}
        
        for scan in recent_scans:
            for vuln in scan.vulnerabilities:
                severity_counts[vuln.severity.value] += 1
                type_counts[vuln.type.value] += 1
        
        return {
            "total_scans": len(recent_scans),
            "total_vulnerabilities": total_vulnerabilities,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "average_vulnerabilities_per_scan": total_vulnerabilities / len(recent_scans),
            "scan_frequency": len(recent_scans) / days
        }

### 4.2 Compliance and Governance Framework

class ComplianceStandard(Enum):
    """Compliance standards"""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"
    CUSTOM = "custom"

class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"

@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    standard: ComplianceStandard
    category: str
    title: str
    description: str
    requirement: str
    check_function: Callable
    severity: SeverityLevel
    remediation_guidance: List[str] = field(default_factory=list)

@dataclass
class ComplianceCheckResult:
    """Result of a compliance check"""
    rule_id: str
    status: ComplianceStatus
    score: float  # 0-100
    findings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    standard: ComplianceStandard
    generated_at: float
    overall_score: float
    status: ComplianceStatus
    check_results: List[ComplianceCheckResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class ComplianceManager:
    """Compliance and governance management system"""
    
    def __init__(self):
        self.compliance_rules: Dict[ComplianceStandard, List[ComplianceRule]] = {}
        self.compliance_history: List[ComplianceReport] = []
        self.policy_documents: Dict[str, Dict[str, Any]] = {}
        self.audit_logs: List[Dict[str, Any]] = []
        
        self._setup_default_compliance_rules()
    
    def _setup_default_compliance_rules(self):
        """Set up default compliance rules for common standards"""
        # SOC 2 Type II rules
        soc2_rules = [
            ComplianceRule(
                rule_id="soc2_access_control",
                standard=ComplianceStandard.SOC2,
                category="Security",
                title="Access Control",
                description="Logical and physical access controls restrict access to system resources",
                requirement="CC6.1 - Access Control",
                check_function=self._check_access_control,
                severity=SeverityLevel.HIGH,
                remediation_guidance=[
                    "Implement role-based access control (RBAC)",
                    "Regular access reviews and certifications",
                    "Multi-factor authentication for privileged accounts"
                ]
            ),
            ComplianceRule(
                rule_id="soc2_encryption",
                standard=ComplianceStandard.SOC2,
                category="Security",
                title="Data Encryption",
                description="Data is encrypted in transit and at rest",
                requirement="CC6.7 - Encryption",
                check_function=self._check_encryption,
                severity=SeverityLevel.HIGH,
                remediation_guidance=[
                    "Implement TLS 1.2+ for data in transit",
                    "Use AES-256 for data at rest",
                    "Proper key management practices"
                ]
            )
        ]
        
        # GDPR rules
        gdpr_rules = [
            ComplianceRule(
                rule_id="gdpr_data_protection",
                standard=ComplianceStandard.GDPR,
                category="Privacy",
                title="Data Protection by Design",
                description="Data protection measures are implemented by design and by default",
                requirement="Article 25 - Data Protection by Design",
                check_function=self._check_data_protection,
                severity=SeverityLevel.CRITICAL,
                remediation_guidance=[
                    "Implement privacy by design principles",
                    "Data minimization practices",
                    "Regular privacy impact assessments"
                ]
            )
        ]
        
        self.compliance_rules[ComplianceStandard.SOC2] = soc2_rules
        self.compliance_rules[ComplianceStandard.GDPR] = gdpr_rules
    
    async def run_compliance_check(self, standard: ComplianceStandard, 
                                 target_system: Dict[str, Any]) -> ComplianceReport:
        """Run compliance check for a specific standard"""
        report_id = f"compliance_{standard.value}_{int(time.time())}"
        check_results = []
        
        rules = self.compliance_rules.get(standard, [])
        
        for rule in rules:
            try:
                result = await rule.check_function(target_system)
                result.rule_id = rule.rule_id
                check_results.append(result)
            except Exception as e:
                logging.error(f"Compliance check {rule.rule_id} failed: {e}")
                check_results.append(ComplianceCheckResult(
                    rule_id=rule.rule_id,
                    status=ComplianceStatus.UNKNOWN,
                    score=0,
                    findings=[f"Check failed: {e}"]
                ))
        
        # Calculate overall score and status
        if check_results:
            overall_score = sum(result.score for result in check_results) / len(check_results)
            
            if overall_score >= 90:
                overall_status = ComplianceStatus.COMPLIANT
            elif overall_score >= 70:
                overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                overall_status = ComplianceStatus.NON_COMPLIANT
        else:
            overall_score = 0
            overall_status = ComplianceStatus.UNKNOWN
        
        # Generate recommendations
        recommendations = self._generate_recommendations(check_results, standard)
        
        # Create summary
        summary = self._create_compliance_summary(check_results)
        
        report = ComplianceReport(
            report_id=report_id,
            standard=standard,
            generated_at=time.time(),
            overall_score=overall_score,
            status=overall_status,
            check_results=check_results,
            summary=summary,
            recommendations=recommendations
        )
        
        self.compliance_history.append(report)
        
        # Log audit event
        self._log_audit_event("compliance_check", {
            "standard": standard.value,
            "report_id": report_id,
            "score": overall_score,
            "status": overall_status.value
        })
        
        return report
    
    async def _check_access_control(self, target_system: Dict[str, Any]) -> ComplianceCheckResult:
        """Check access control implementation"""
        findings = []
        score = 100
        
        # Check for RBAC implementation
        if not target_system.get("rbac_enabled", False):
            findings.append("Role-based access control not implemented")
            score -= 30
        
        # Check for MFA
        if not target_system.get("mfa_enabled", False):
            findings.append("Multi-factor authentication not enabled")
            score -= 25
        
        # Check for access logging
        if not target_system.get("access_logging", False):
            findings.append("Access logging not implemented")
            score -= 20
        
        # Check for regular access reviews
        last_review = target_system.get("last_access_review")
        if not last_review or (time.time() - last_review) > (90 * 24 * 3600):  # 90 days
            findings.append("Access review overdue (>90 days)")
            score -= 25
        
        status = ComplianceStatus.COMPLIANT if score >= 80 else ComplianceStatus.NON_COMPLIANT
        
        return ComplianceCheckResult(
            rule_id="",  # Will be set by caller
            status=status,
            score=max(0, score),
            findings=findings,
            evidence={
                "rbac_enabled": target_system.get("rbac_enabled", False),
                "mfa_enabled": target_system.get("mfa_enabled", False),
                "access_logging": target_system.get("access_logging", False),
                "last_access_review": target_system.get("last_access_review")
            }
        )
    
    async def _check_encryption(self, target_system: Dict[str, Any]) -> ComplianceCheckResult:
        """Check encryption implementation"""
        findings = []
        score = 100
        
        # Check encryption in transit
        tls_version = target_system.get("tls_version")
        if not tls_version or float(tls_version) < 1.2:
            findings.append("TLS version below 1.2")
            score -= 40
        
        # Check encryption at rest
        if not target_system.get("encryption_at_rest", False):
            findings.append("Data encryption at rest not implemented")
            score -= 40
        
        # Check key management
        if not target_system.get("key_management", False):
            findings.append("Proper key management not implemented")
            score -= 20
        
        status = ComplianceStatus.COMPLIANT if score >= 80 else ComplianceStatus.NON_COMPLIANT
        
        return ComplianceCheckResult(
            rule_id="",
            status=status,
            score=max(0, score),
            findings=findings,
            evidence={
                "tls_version": tls_version,
                "encryption_at_rest": target_system.get("encryption_at_rest", False),
                "key_management": target_system.get("key_management", False)
            }
        )
    
    async def _check_data_protection(self, target_system: Dict[str, Any]) -> ComplianceCheckResult:
        """Check GDPR data protection implementation"""
        findings = []
        score = 100
        
        # Check data minimization
        if not target_system.get("data_minimization", False):
            findings.append("Data minimization principles not implemented")
            score -= 25
        
        # Check consent management
        if not target_system.get("consent_management", False):
            findings.append("Consent management system not implemented")
            score -= 25
        
        # Check data subject rights
        if not target_system.get("data_subject_rights", False):
            findings.append("Data subject rights not implemented")
            score -= 25
        
        # Check privacy impact assessment
        if not target_system.get("privacy_impact_assessment", False):
            findings.append("Privacy impact assessment not conducted")
            score -= 25
        
        status = ComplianceStatus.COMPLIANT if score >= 80 else ComplianceStatus.NON_COMPLIANT
        
        return ComplianceCheckResult(
            rule_id="",
            status=status,
            score=max(0, score),
            findings=findings,
            evidence={
                "data_minimization": target_system.get("data_minimization", False),
                "consent_management": target_system.get("consent_management", False),
                "data_subject_rights": target_system.get("data_subject_rights", False),
                "privacy_impact_assessment": target_system.get("privacy_impact_assessment", False)
            }
        )
    
    def _generate_recommendations(self, check_results: List[ComplianceCheckResult], 
                                standard: ComplianceStandard) -> List[str]:
        """Generate compliance recommendations based on check results"""
        recommendations = []
        
        # Find rules with low scores
        failing_rules = [result for result in check_results if result.score < 80]
        
        if failing_rules:
            recommendations.append(f"Address {len(failing_rules)} failing compliance checks")
            
            # Get specific recommendations from rule definitions
            rules = self.compliance_rules.get(standard, [])
            for result in failing_rules:
                rule = next((r for r in rules if r.rule_id == result.rule_id), None)
                if rule:
                    recommendations.extend(rule.remediation_guidance)
        
        return recommendations
    
    def _create_compliance_summary(self, check_results: List[ComplianceCheckResult]) -> Dict[str, Any]:
        """Create compliance summary statistics"""
        if not check_results:
            return {}
        
        status_counts = {status.value: 0 for status in ComplianceStatus}
        for result in check_results:
            status_counts[result.status.value] += 1
        
        return {
            "total_checks": len(check_results),
            "status_distribution": status_counts,
            "average_score": sum(result.score for result in check_results) / len(check_results),
            "passing_checks": len([r for r in check_results if r.score >= 80]),
            "failing_checks": len([r for r in check_results if r.score < 80])
        }
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event"""
        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
            "user": "system"  # Would be actual user in real implementation
        }
        self.audit_logs.append(audit_entry)
    
    def add_custom_rule(self, rule: ComplianceRule):
        """Add a custom compliance rule"""
        if rule.standard not in self.compliance_rules:
            self.compliance_rules[rule.standard] = []
        
        self.compliance_rules[rule.standard].append(rule)
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        if not self.compliance_history:
            return {"message": "No compliance reports available"}
        
        latest_reports = {}
        for report in self.compliance_history:
            if (report.standard not in latest_reports or 
                report.generated_at > latest_reports[report.standard].generated_at):
                latest_reports[report.standard] = report
        
        dashboard = {
            "standards_tracked": len(latest_reports),
            "overall_compliance_score": 0,
            "standards_status": {},
            "recent_trends": {},
            "critical_findings": []
        }
        
        if latest_reports:
            # Calculate overall compliance score
            total_score = sum(report.overall_score for report in latest_reports.values())
            dashboard["overall_compliance_score"] = total_score / len(latest_reports)
            
            # Standards status
            for standard, report in latest_reports.items():
                dashboard["standards_status"][standard.value] = {
                    "score": report.overall_score,
                    "status": report.status.value,
                    "last_check": report.generated_at
                }
            
            # Critical findings
            for report in latest_reports.values():
                critical_results = [r for r in report.check_results if r.score < 50]
                for result in critical_results:
                    dashboard["critical_findings"].extend(result.findings)
        
        return dashboard
```

## 5. Practical Implementation Examples

### 5.1 Complete Performance Optimization System

```python
async def setup_performance_optimization_system():
    """Set up a complete performance optimization and monitoring system"""
    
    # Initialize all components
    profiler = PerformanceProfiler()
    bottleneck_detector = BottleneckDetector()
    tracer = DistributedTracer()
    observability_dashboard = ObservabilityDashboard(tracer)
    cost_optimizer = CostOptimizer()
    resource_scheduler = IntelligentResourceScheduler()
    security_scanner = SecurityScanner()
    compliance_manager = ComplianceManager()
    
    print("🚀 Performance Optimization System Setup Complete")
    print("=" * 50)
    
    # Start performance profiling
    session_id = await profiler.start_profiling_session(
        "production_system",
        [ProfilerType.CPU, ProfilerType.MEMORY, ProfilerType.IO, ProfilerType.NETWORK]
    )
    print(f"📊 Started profiling session: {session_id}")
    
    # Simulate some workload
    await simulate_agent_workload()
    
    # Stop profiling and analyze
    session_summary = await profiler.stop_profiling_session(session_id)
    print(f"⏱️  Profiling completed. Duration: {session_summary['duration']:.2f}s")
    
    # Detect bottlenecks
    bottlenecks = await bottleneck_detector.detect_bottlenecks(session_summary)
    if bottlenecks:
        print(f"⚠️  Detected {len(bottlenecks)} performance bottlenecks:")
        for bottleneck in bottlenecks:
            print(f"   - {bottleneck.type.value}: {bottleneck.description}")
    else:
        print("✅ No performance bottlenecks detected")
    
    # Start distributed tracing
    trace = tracer.start_trace("agent_system_operation")
    
    # Simulate distributed operations
    with tracer.start_span("data_processing", SpanKind.INTERNAL) as span:
        span.set_tag("component", "data_processor")
        await asyncio.sleep(0.1)  # Simulate processing
        
        with tracer.start_span("database_query", SpanKind.CLIENT) as db_span:
            db_span.set_tag("db.type", "postgresql")
            db_span.set_tag("db.statement", "SELECT * FROM agents")
            await asyncio.sleep(0.05)  # Simulate DB query
        
        with tracer.start_span("external_api_call", SpanKind.CLIENT) as api_span:
            api_span.set_tag("http.method", "POST")
            api_span.set_tag("http.url", "https://api.example.com/process")
            await asyncio.sleep(0.08)  # Simulate API call
    
    tracer.finish_trace(trace)
    
    # Get observability insights
    dashboard_data = observability_dashboard.get_system_overview()
    print(f"📈 System Overview:")
    print(f"   - Active traces: {dashboard_data['active_traces']}")
    print(f"   - Total spans: {dashboard_data['total_spans']}")
    print(f"   - Error rate: {dashboard_data['error_rate']:.2%}")
    
    # Run cost optimization
    current_costs = {
        "compute": {"type": "m5.large", "count": 10, "hourly_rate": 0.096},
        "storage": {"type": "gp2", "size_gb": 1000, "monthly_rate": 0.10},
        "network": {"data_transfer_gb": 500, "rate_per_gb": 0.09}
    }
    
    recommendations = await cost_optimizer.analyze_costs(current_costs)
    print(f"💰 Cost Optimization Recommendations:")
    for rec in recommendations:
        print(f"   - {rec.title}: ${rec.estimated_savings:.2f}/month savings")
    
    # Run security scan
    scan_result = await security_scanner.scan_codebase("/path/to/agent/codebase")
    print(f"🔒 Security Scan Results:")
    print(f"   - Vulnerabilities found: {len(scan_result.vulnerabilities)}")
    if scan_result.vulnerabilities:
        critical_vulns = [v for v in scan_result.vulnerabilities if v.severity == SeverityLevel.CRITICAL]
        high_vulns = [v for v in scan_result.vulnerabilities if v.severity == SeverityLevel.HIGH]
        print(f"   - Critical: {len(critical_vulns)}, High: {len(high_vulns)}")
    
    # Run compliance check
    target_system = {
        "rbac_enabled": True,
        "mfa_enabled": False,
        "access_logging": True,
        "last_access_review": time.time() - (30 * 24 * 3600),  # 30 days ago
        "tls_version": "1.3",
        "encryption_at_rest": True,
        "key_management": True
    }
    
    compliance_report = await compliance_manager.run_compliance_check(
        ComplianceStandard.SOC2, target_system
    )
    print(f"📋 SOC 2 Compliance Report:")
    print(f"   - Overall score: {compliance_report.overall_score:.1f}/100")
    print(f"   - Status: {compliance_report.status.value}")
    print(f"   - Recommendations: {len(compliance_report.recommendations)}")
    
    return {
        "profiler": profiler,
        "tracer": tracer,
        "cost_optimizer": cost_optimizer,
        "security_scanner": security_scanner,
        "compliance_manager": compliance_manager
    }

async def simulate_agent_workload():
    """Simulate agent system workload for demonstration"""
    # Simulate CPU-intensive task
    start_time = time.time()
    result = sum(i * i for i in range(100000))
    
    # Simulate memory allocation
    data = [random.random() for _ in range(50000)]
    
    # Simulate I/O operation
    await asyncio.sleep(0.05)
    
    # Simulate network operation
    await asyncio.sleep(0.03)
    
    return result

async def demonstrate_advanced_monitoring():
    """Demonstrate advanced monitoring capabilities"""
    print("🔍 Advanced Monitoring Demonstration")
    print("=" * 40)
    
    # Set up monitoring system
    system = await setup_performance_optimization_system()
    
    # Demonstrate real-time monitoring
    print("\n📊 Real-time Performance Metrics:")
    
    # Simulate continuous monitoring
    for i in range(5):
        print(f"\n--- Monitoring Cycle {i+1} ---")
        
        # Start a new trace for this cycle
        tracer = system["tracer"]
        trace = tracer.start_trace(f"monitoring_cycle_{i+1}")
        
        # Simulate various operations
        with tracer.start_span("agent_processing", SpanKind.INTERNAL) as span:
            span.set_tag("cycle", i+1)
            span.set_tag("operation", "data_analysis")
            
            # Simulate processing time
            processing_time = random.uniform(0.05, 0.15)
            await asyncio.sleep(processing_time)
            
            # Add some metrics
            span.set_tag("processing_time_ms", processing_time * 1000)
            span.set_tag("records_processed", random.randint(100, 1000))
            
            # Simulate potential errors
            if random.random() < 0.1:  # 10% chance of error
                span.set_tag("error", True)
                span.log("Simulated processing error")
                span.set_status(SpanStatus.ERROR)
        
        tracer.finish_trace(trace)
        
        # Get current system metrics
        dashboard = system["tracer"]
        overview = ObservabilityDashboard(dashboard).get_system_overview()
        
        print(f"   Traces: {overview['active_traces']}, "
              f"Spans: {overview['total_spans']}, "
              f"Errors: {overview['error_rate']:.1%}")
        
        await asyncio.sleep(1)  # Wait between cycles
    
    print("\n✅ Advanced monitoring demonstration completed")

### 5.2 Performance Optimization Workflow

def create_optimization_workflow():
    """Create a comprehensive performance optimization workflow"""
    
    class OptimizationWorkflow:
        def __init__(self):
            self.profiler = PerformanceProfiler()
            self.bottleneck_detector = BottleneckDetector()
            self.cost_optimizer = CostOptimizer()
            self.security_scanner = SecurityScanner()
            self.compliance_manager = ComplianceManager()
            
            self.optimization_history = []
        
        async def run_full_optimization(self, target_system: str) -> Dict[str, Any]:
            """Run complete optimization workflow"""
            workflow_id = f"optimization_{int(time.time())}"
            start_time = time.time()
            
            results = {
                "workflow_id": workflow_id,
                "target_system": target_system,
                "start_time": start_time,
                "phases": {}
            }
            
            try:
                # Phase 1: Performance Profiling
                print("🔍 Phase 1: Performance Profiling")
                session_id = await self.profiler.start_profiling_session(
                    target_system, [ProfilerType.CPU, ProfilerType.MEMORY, ProfilerType.IO]
                )
                
                # Simulate workload
                await simulate_agent_workload()
                
                session_summary = await self.profiler.stop_profiling_session(session_id)
                results["phases"]["profiling"] = {
                    "session_id": session_id,
                    "duration": session_summary["duration"],
                    "metrics": session_summary["metrics"]
                }
                
                # Phase 2: Bottleneck Detection
                print("⚠️  Phase 2: Bottleneck Detection")
                bottlenecks = await self.bottleneck_detector.detect_bottlenecks(session_summary)
                results["phases"]["bottleneck_detection"] = {
                    "bottlenecks_found": len(bottlenecks),
                    "bottlenecks": [{
                        "type": b.type.value,
                        "severity": b.severity.value,
                        "description": b.description
                    } for b in bottlenecks]
                }
                
                # Phase 3: Cost Analysis
                print("💰 Phase 3: Cost Analysis")
                current_costs = {
                    "compute": {"type": "m5.large", "count": 5, "hourly_rate": 0.096},
                    "storage": {"type": "gp2", "size_gb": 500, "monthly_rate": 0.10}
                }
                
                cost_recommendations = await self.cost_optimizer.analyze_costs(current_costs)
                total_savings = sum(rec.estimated_savings for rec in cost_recommendations)
                
                results["phases"]["cost_analysis"] = {
                    "recommendations": len(cost_recommendations),
                    "estimated_monthly_savings": total_savings,
                    "details": [{
                        "title": rec.title,
                        "savings": rec.estimated_savings,
                        "confidence": rec.confidence
                    } for rec in cost_recommendations]
                }
                
                # Phase 4: Security Assessment
                print("🔒 Phase 4: Security Assessment")
                scan_result = await self.security_scanner.scan_codebase("/path/to/system")
                
                results["phases"]["security_assessment"] = {
                    "vulnerabilities_found": len(scan_result.vulnerabilities),
                    "severity_breakdown": scan_result.summary,
                    "scan_duration": scan_result.end_time - scan_result.start_time
                }
                
                # Phase 5: Compliance Check
                print("📋 Phase 5: Compliance Check")
                system_config = {
                    "rbac_enabled": True,
                    "mfa_enabled": True,
                    "access_logging": True,
                    "last_access_review": time.time() - (15 * 24 * 3600),
                    "tls_version": "1.3",
                    "encryption_at_rest": True,
                    "key_management": True
                }
                
                compliance_report = await self.compliance_manager.run_compliance_check(
                    ComplianceStandard.SOC2, system_config
                )
                
                results["phases"]["compliance_check"] = {
                    "overall_score": compliance_report.overall_score,
                    "status": compliance_report.status.value,
                    "recommendations": len(compliance_report.recommendations)
                }
                
                # Calculate overall optimization score
                performance_score = 100 - (len(bottlenecks) * 10)  # Deduct for bottlenecks
                cost_score = min(100, total_savings * 10)  # Scale savings to score
                security_score = max(0, 100 - len(scan_result.vulnerabilities) * 5)
                compliance_score = compliance_report.overall_score
                
                overall_score = (performance_score + cost_score + security_score + compliance_score) / 4
                
                results["overall_score"] = overall_score
                results["end_time"] = time.time()
                results["total_duration"] = results["end_time"] - start_time
                
                # Store in history
                self.optimization_history.append(results)
                
                print(f"\n✅ Optimization workflow completed!")
                print(f"   Overall Score: {overall_score:.1f}/100")
                print(f"   Duration: {results['total_duration']:.2f}s")
                
                return results
                
            except Exception as e:
                print(f"❌ Optimization workflow failed: {e}")
                results["error"] = str(e)
                results["end_time"] = time.time()
                return results
        
        def get_optimization_trends(self, days: int = 30) -> Dict[str, Any]:
            """Get optimization trends over time"""
            cutoff_time = time.time() - (days * 24 * 3600)
            recent_optimizations = [
                opt for opt in self.optimization_history 
                if opt["start_time"] > cutoff_time
            ]
            
            if not recent_optimizations:
                return {"message": "No recent optimizations found"}
            
            # Calculate trends
            scores = [opt["overall_score"] for opt in recent_optimizations if "overall_score" in opt]
            
            return {
                "total_optimizations": len(recent_optimizations),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "score_trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
                "latest_score": scores[-1] if scores else 0,
                "optimization_frequency": len(recent_optimizations) / days
            }
    
    return OptimizationWorkflow()
```

## 6. Hands-on Exercises

### Exercise 1: Custom Performance Profiler

```python
class CustomPerformanceProfiler:
    """Custom performance profiler with advanced features"""
    
    def __init__(self):
        self.custom_metrics = {}
        self.profiling_rules = []
        self.alert_thresholds = {}
    
    def add_custom_metric(self, name: str, collection_function: Callable):
        """Add a custom metric to track"""
        self.custom_metrics[name] = collection_function
    
    def add_profiling_rule(self, condition: Callable, action: Callable):
        """Add a rule that triggers actions based on profiling data"""
        self.profiling_rules.append((condition, action))
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold for a metric"""
        self.alert_thresholds[metric] = threshold
    
    async def profile_with_rules(self, duration: float) -> Dict[str, Any]:
        """Profile system and apply rules"""
        # Implementation for custom profiling with rules
        results = {"custom_metrics": {}, "alerts": [], "rule_actions": []}
        
        # Collect custom metrics
        for name, func in self.custom_metrics.items():
            try:
                value = await func() if asyncio.iscoroutinefunction(func) else func()
                results["custom_metrics"][name] = value
                
                # Check alert thresholds
                if name in self.alert_thresholds and value > self.alert_thresholds[name]:
                    results["alerts"].append(f"{name} exceeded threshold: {value} > {self.alert_thresholds[name]}")
            except Exception as e:
                results["custom_metrics"][name] = f"Error: {e}"
        
        # Apply profiling rules
        for condition, action in self.profiling_rules:
            try:
                if condition(results):
                    action_result = await action(results) if asyncio.iscoroutinefunction(action) else action(results)
                    results["rule_actions"].append(action_result)
            except Exception as e:
                results["rule_actions"].append(f"Rule action failed: {e}")
        
        return results

# Exercise: Implement custom metrics and rules
# TODO: Add memory leak detection
# TODO: Add database connection pool monitoring
# TODO: Add custom alerting rules
```

### Exercise 2: Advanced Cost Optimization Strategy

```python
class PredictiveCostOptimizer:
    """Advanced cost optimizer with predictive capabilities"""
    
    def __init__(self):
        self.usage_history = []
        self.cost_models = {}
        self.optimization_strategies = []
    
    def add_usage_data(self, timestamp: float, usage_metrics: Dict[str, float]):
        """Add historical usage data for prediction"""
        self.usage_history.append({
            "timestamp": timestamp,
            "metrics": usage_metrics
        })
    
    def predict_future_usage(self, days_ahead: int) -> Dict[str, float]:
        """Predict future resource usage based on historical data"""
        if len(self.usage_history) < 7:  # Need at least a week of data
            return {"error": "Insufficient historical data"}
        
        # Simple trend analysis (in practice, use more sophisticated ML models)
        recent_data = self.usage_history[-7:]  # Last 7 data points
        
        predictions = {}
        for metric in recent_data[0]["metrics"].keys():
            values = [data["metrics"][metric] for data in recent_data]
            
            # Calculate trend
            if len(values) > 1:
                trend = (values[-1] - values[0]) / len(values)
                predicted_value = values[-1] + (trend * days_ahead)
                predictions[metric] = max(0, predicted_value)  # Ensure non-negative
            else:
                predictions[metric] = values[0]
        
        return predictions
    
    async def optimize_with_prediction(self, current_resources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize costs using predictive analysis"""
        recommendations = []
        
        # Predict usage for next 30 days
        predicted_usage = self.predict_future_usage(30)
        
        if "error" not in predicted_usage:
            # Generate recommendations based on predictions
            for resource, predicted_value in predicted_usage.items():
                current_value = current_resources.get(resource, {}).get("current_usage", 0)
                
                if predicted_value > current_value * 1.2:  # 20% increase predicted
                    recommendations.append({
                        "type": "scale_up",
                        "resource": resource,
                        "current": current_value,
                        "predicted": predicted_value,
                        "recommendation": f"Consider scaling up {resource} by {((predicted_value/current_value - 1) * 100):.1f}%",
                        "confidence": 0.8
                    })
                elif predicted_value < current_value * 0.8:  # 20% decrease predicted
                    recommendations.append({
                        "type": "scale_down",
                        "resource": resource,
                        "current": current_value,
                        "predicted": predicted_value,
                        "recommendation": f"Consider scaling down {resource} by {((1 - predicted_value/current_value) * 100):.1f}%",
                        "confidence": 0.8
                    })
        
        return recommendations

# Exercise: Implement machine learning-based cost prediction
# TODO: Add seasonal pattern detection
# TODO: Implement anomaly detection for cost spikes
# TODO: Add multi-cloud cost optimization
```

### Exercise 3: Comprehensive Security and Compliance Framework

```python
class AdvancedSecurityFramework:
    """Advanced security framework with automated remediation"""
    
    def __init__(self):
        self.security_policies = {}
        self.remediation_actions = {}
        self.compliance_mappings = {}
        self.risk_assessments = []
    
    def define_security_policy(self, policy_id: str, policy_definition: Dict[str, Any]):
        """Define a security policy"""
        self.security_policies[policy_id] = policy_definition
    
    def add_remediation_action(self, vulnerability_type: str, action: Callable):
        """Add automated remediation action for vulnerability type"""
        if vulnerability_type not in self.remediation_actions:
            self.remediation_actions[vulnerability_type] = []
        self.remediation_actions[vulnerability_type].append(action)
    
    async def automated_security_assessment(self, target_system: str) -> Dict[str, Any]:
        """Perform automated security assessment with remediation"""
        assessment_results = {
            "assessment_id": f"security_assessment_{int(time.time())}",
            "target_system": target_system,
            "vulnerabilities": [],
            "policy_violations": [],
            "remediation_actions_taken": [],
            "risk_score": 0
        }
        
        # Simulate vulnerability scanning
        # In practice, integrate with actual security scanning tools
        simulated_vulnerabilities = [
            {"type": "injection", "severity": "high", "location": "api_endpoint.py:42"},
            {"type": "authentication", "severity": "medium", "location": "auth_module.py:15"},
            {"type": "data_exposure", "severity": "low", "location": "logging.py:8"}
        ]
        
        for vuln in simulated_vulnerabilities:
            assessment_results["vulnerabilities"].append(vuln)
            
            # Attempt automated remediation
            vuln_type = vuln["type"]
            if vuln_type in self.remediation_actions:
                for action in self.remediation_actions[vuln_type]:
                    try:
                        result = await action(vuln) if asyncio.iscoroutinefunction(action) else action(vuln)
                        assessment_results["remediation_actions_taken"].append({
                            "vulnerability": vuln,
                            "action_result": result
                        })
                    except Exception as e:
                        assessment_results["remediation_actions_taken"].append({
                            "vulnerability": vuln,
                            "action_result": f"Failed: {e}"
                        })
        
        # Calculate risk score
        severity_weights = {"critical": 10, "high": 7, "medium": 4, "low": 1}
        total_risk = sum(severity_weights.get(vuln["severity"], 0) for vuln in simulated_vulnerabilities)
        assessment_results["risk_score"] = min(100, total_risk)  # Cap at 100
        
        return assessment_results
    
    def generate_compliance_report(self, standards: List[str]) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        # Implementation for multi-standard compliance reporting
        return {
            "standards_assessed": standards,
            "overall_compliance_score": 85.5,
            "recommendations": [
                "Implement additional access controls",
                "Enhance data encryption practices",
                "Improve audit logging coverage"
            ]
        }

# Exercise: Implement advanced security features
# TODO: Add threat modeling capabilities
# TODO: Implement security orchestration and automated response (SOAR)
# TODO: Add integration with external threat intelligence feeds
```

## Module Summary

In this module, you've learned to implement comprehensive performance optimization and advanced monitoring for agent systems:

### Key Concepts Learned:
1. **Performance Profiling and Analysis**
   - CPU, memory, I/O, and network profiling
   - Bottleneck detection and analysis
   - Custom profiling strategies

2. **Distributed Tracing and Observability**
   - Span-based tracing architecture
   - Service dependency mapping
   - Performance correlation analysis

3. **Cost Optimization and Resource Management**
   - Cost-aware resource optimization
   - Intelligent resource scheduling
   - Predictive cost analysis

4. **Security Hardening and Compliance**
   - Automated vulnerability scanning
   - Compliance framework implementation
   - Security policy enforcement

### Practical Skills Developed:
1. **Performance Engineering**
   - System profiling and optimization
   - Bottleneck identification and resolution
   - Performance monitoring strategies

2. **Observability Implementation**
   - Distributed tracing setup
   - Metrics collection and analysis
   - Dashboard and alerting systems

3. **Cost Management**
   - Resource optimization strategies
   - Cost prediction and planning
   - Multi-cloud cost optimization

4. **Security and Governance**
   - Security scanning and assessment
   - Compliance monitoring and reporting
   - Automated remediation workflows

### Real-world Applications:
1. **Enterprise Systems**
   - Large-scale agent deployments
   - Multi-tenant performance optimization
   - Enterprise compliance requirements

2. **Cloud-Native Applications**
   - Microservices performance monitoring
   - Container and serverless optimization
   - Cloud cost management

3. **High-Performance Computing**
   - Scientific computing workloads
   - Real-time processing systems
   - Resource-intensive AI/ML applications

4. **Regulated Industries**
   - Financial services compliance
   - Healthcare data protection
   - Government security requirements

### Next Steps:
You're now ready to move on to the final module covering **Future Trends and Research Directions** in agent systems, where you'll explore emerging technologies, research frontiers, and prepare for the evolving landscape of intelligent agents.
    
    async def _detect_cpu_bottleneck(self, session_summary: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """Detect CPU-bound bottlenecks"""
        bottlenecks = []
        metrics = session_summary.get("metrics", {})
        
        cpu_percent = metrics.get("cpu_percent", {})
        if cpu_percent and cpu_percent.get("mean", 0) > self.thresholds["cpu_high"]:
            severity = "critical" if cpu_percent["mean"] > 95 else "high"
            impact_score = min(cpu_percent["mean"], 100)
            
            bottleneck = PerformanceBottleneck(
                type=BottleneckType.CPU_BOUND,
                severity=severity,
                description=f"High CPU utilization: {cpu_percent['mean']:.1f}% average",
                location="system",
                impact_score=impact_score,
                recommendations=[
                    "Profile CPU-intensive functions",
                    "Consider algorithm optimization",
                    "Implement parallel processing",
                    "Scale horizontally if possible"
                ],
                metrics={"cpu_percent_mean": cpu_percent["mean"]},
                timestamp=time.time()
            )
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _detect_memory_bottleneck(self, session_summary: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """Detect memory-bound bottlenecks"""
        bottlenecks = []
        metrics = session_summary.get("metrics", {})
        
        memory_percent = metrics.get("memory_percent", {})
        if memory_percent and memory_percent.get("mean", 0) > self.thresholds["memory_high"]:
            severity = "critical" if memory_percent["mean"] > 95 else "high"
            impact_score = min(memory_percent["mean"], 100)
            
            bottleneck = PerformanceBottleneck(
                type=BottleneckType.MEMORY_BOUND,
                severity=severity,
                description=f"High memory utilization: {memory_percent['mean']:.1f}% average",
                location="system",
                impact_score=impact_score,
                recommendations=[
                    "Profile memory usage patterns",
                    "Implement memory pooling",
                    "Optimize data structures",
                    "Consider memory-mapped files for large datasets"
                ],
                metrics={"memory_percent_mean": memory_percent["mean"]},
                timestamp=time.time()
            )
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _detect_io_bottleneck(self, session_summary: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """Detect I/O-bound bottlenecks"""
        bottlenecks = []
        # Implementation would analyze I/O patterns and wait times
        return bottlenecks
    
    async def _detect_network_bottleneck(self, session_summary: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """Detect network-bound bottlenecks"""
        bottlenecks = []
        # Implementation would analyze network latency and throughput
        return bottlenecks
    
    def set_threshold(self, metric: str, value: float):
        """Set detection threshold for a metric"""
        self.thresholds[metric] = value
    
    def add_custom_detection_rule(self, bottleneck_type: BottleneckType, 
                                detection_func: Callable):
        """Add a custom bottleneck detection rule"""
        self.detection_rules[bottleneck_type] = detection_func
```

## 2. Distributed Tracing and Observability

### 2.1 Distributed Tracing System

```python
import uuid
import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import time
import threading
from collections import defaultdict

class SpanKind(Enum):
    """Types of spans in distributed tracing"""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"

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
    
@dataclass
class Span:
    """Distributed tracing span"""
    context: SpanContext
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span"""
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
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)

@dataclass
class Trace:
    """Complete distributed trace"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    service_map: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def add_span(self, span: Span):
        """Add a span to the trace"""
        self.spans.append(span)
        
        # Update trace timing
        if self.start_time is None or span.start_time < self.start_time:
            self.start_time = span.start_time
        
        if span.end_time:
            if self.end_time is None or span.end_time > self.end_time:
                self.end_time = span.end_time
        
        if self.start_time and self.end_time:
            self.duration = self.end_time - self.start_time
    
    def get_root_spans(self) -> List[Span]:
        """Get root spans (spans without parents)"""
        return [span for span in self.spans if span.context.parent_span_id is None]
    
    def get_children(self, parent_span_id: str) -> List[Span]:
        """Get child spans of a parent"""
        return [span for span in self.spans 
                if span.context.parent_span_id == parent_span_id]
    
    def build_span_tree(self) -> Dict[str, Any]:
        """Build a hierarchical tree of spans"""
        def build_node(span: Span) -> Dict[str, Any]:
            children = self.get_children(span.context.span_id)
            return {
                "span": span,
                "children": [build_node(child) for child in children]
            }
        
        root_spans = self.get_root_spans()
        return {
            "trace_id": self.trace_id,
            "duration": self.duration,
            "span_count": len(self.spans),
            "roots": [build_node(root) for root in root_spans]
        }

class DistributedTracer:
    """Distributed tracing system for agent communications"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans: Dict[str, Span] = {}  # thread_id -> span
        self.completed_traces: Dict[str, Trace] = {}
        self.span_processors: List[Callable] = []
        self.sampling_rate = 1.0  # Sample 100% by default
        self._local = threading.local()
        
    def start_span(self, operation_name: str, 
                  parent_context: Optional[SpanContext] = None,
                  kind: SpanKind = SpanKind.INTERNAL,
                  tags: Optional[Dict[str, Any]] = None) -> Span:
        """Start a new span"""
        
        # Generate IDs
        span_id = str(uuid.uuid4())
        
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        
        # Create span context
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        
        # Create span
        span = Span(
            context=context,
            operation_name=operation_name,
            start_time=time.time(),
            kind=kind,
            tags=tags or {}
        )
        
        # Set default tags
        span.set_tag("service.name", self.service_name)
        span.set_tag("span.kind", kind.value)
        
        # Store as active span for current thread
        thread_id = threading.get_ident()
        self.active_spans[thread_id] = span
        
        return span
    
    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK):
        """Finish a span and process it"""
        span.finish(status)
        
        # Remove from active spans
        thread_id = threading.get_ident()
        if thread_id in self.active_spans and self.active_spans[thread_id] == span:
            del self.active_spans[thread_id]
        
        # Process the span
        self._process_span(span)
    
    def get_active_span(self) -> Optional[Span]:
        """Get the currently active span for this thread"""
        thread_id = threading.get_ident()
        return self.active_spans.get(thread_id)
    
    def _process_span(self, span: Span):
        """Process a completed span"""
        # Add to trace
        trace_id = span.context.trace_id
        if trace_id not in self.completed_traces:
            self.completed_traces[trace_id] = Trace(trace_id=trace_id)
        
        self.completed_traces[trace_id].add_span(span)
        
        # Run span processors
        for processor in self.span_processors:
            try:
                processor(span)
            except Exception as e:
                logging.warning(f"Span processor failed: {e}")
    
    def add_span_processor(self, processor: Callable):
        """Add a span processor"""
        self.span_processors.append(processor)
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a completed trace"""
        return self.completed_traces.get(trace_id)
    
    def get_all_traces(self) -> List[Trace]:
        """Get all completed traces"""
        return list(self.completed_traces.values())
    
    @asynccontextmanager
    async def trace(self, operation_name: str, 
                   parent_context: Optional[SpanContext] = None,
                   kind: SpanKind = SpanKind.INTERNAL,
                   tags: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations"""
        span = self.start_span(operation_name, parent_context, kind, tags)
        try:
            yield span
            self.finish_span(span, SpanStatus.OK)
        except Exception as e:
            span.log(f"Exception: {e}", level="error")
            span.set_tag("error", True)
            span.set_tag("error.message", str(e))
            self.finish_span(span, SpanStatus.ERROR)
            raise
    
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
        
        if not trace_id or not span_id:
            return None
        
        parent_span_id = headers.get("parent-span-id") or None
        baggage_str = headers.get("baggage", "{}")
        
        try:
            baggage = json.loads(baggage_str)
        except json.JSONDecodeError:
            baggage = {}
        
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )

### 2.2 Observability Dashboard

class ObservabilityDashboard:
    """Comprehensive observability dashboard"""
    
    def __init__(self, tracer: DistributedTracer, 
                 profiler: PerformanceProfiler,
                 bottleneck_detector: BottleneckDetector):
        self.tracer = tracer
        self.profiler = profiler
        self.bottleneck_detector = bottleneck_detector
        self.dashboard_data: Dict[str, Any] = {}
        
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        traces = self.tracer.get_all_traces()
        
        # Trace statistics
        trace_stats = self._calculate_trace_statistics(traces)
        
        # Service map
        service_map = self._build_service_map(traces)
        
        # Error analysis
        error_analysis = self._analyze_errors(traces)
        
        # Performance metrics
        performance_metrics = await self._get_performance_metrics()
        
        return {
            "timestamp": time.time(),
            "trace_statistics": trace_stats,
            "service_map": service_map,
            "error_analysis": error_analysis,
            "performance_metrics": performance_metrics,
            "system_health": self._calculate_system_health(trace_stats, error_analysis)
        }
    
    def _calculate_trace_statistics(self, traces: List[Trace]) -> Dict[str, Any]:
        """Calculate trace statistics"""
        if not traces:
            return {"total_traces": 0}
        
        durations = [trace.duration for trace in traces if trace.duration]
        span_counts = [len(trace.spans) for trace in traces]
        
        return {
            "total_traces": len(traces),
            "avg_duration": statistics.mean(durations) if durations else 0,
            "p95_duration": self._percentile(durations, 95) if durations else 0,
            "p99_duration": self._percentile(durations, 99) if durations else 0,
            "avg_span_count": statistics.mean(span_counts) if span_counts else 0,
            "total_spans": sum(span_counts)
        }
    
    def _build_service_map(self, traces: List[Trace]) -> Dict[str, Any]:
        """Build service dependency map"""
        services = set()
        dependencies = defaultdict(set)
        
        for trace in traces:
            for span in trace.spans:
                service_name = span.tags.get("service.name", "unknown")
                services.add(service_name)
                
                # Find dependencies based on parent-child relationships
                if span.context.parent_span_id:
                    parent_span = next(
                        (s for s in trace.spans 
                         if s.context.span_id == span.context.parent_span_id),
                        None
                    )
                    if parent_span:
                        parent_service = parent_span.tags.get("service.name", "unknown")
                        if parent_service != service_name:
                            dependencies[parent_service].add(service_name)
        
        return {
            "services": list(services),
            "dependencies": {k: list(v) for k, v in dependencies.items()},
            "service_count": len(services)
        }
    
    def _analyze_errors(self, traces: List[Trace]) -> Dict[str, Any]:
        """Analyze errors in traces"""
        total_spans = 0
        error_spans = 0
        error_types = defaultdict(int)
        error_services = defaultdict(int)
        
        for trace in traces:
            for span in trace.spans:
                total_spans += 1
                
                if span.status == SpanStatus.ERROR or span.tags.get("error"):
                    error_spans += 1
                    
                    # Categorize error
                    error_message = span.tags.get("error.message", "unknown")
                    error_types[error_message] += 1
                    
                    service_name = span.tags.get("service.name", "unknown")
                    error_services[service_name] += 1
        
        error_rate = (error_spans / total_spans * 100) if total_spans > 0 else 0
        
        return {
            "total_spans": total_spans,
            "error_spans": error_spans,
            "error_rate": error_rate,
            "error_types": dict(error_types),
            "error_services": dict(error_services)
        }
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        # Get latest metrics from profiler buffer
        recent_metrics = list(self.profiler.metrics_buffer)[-100:]  # Last 100 metrics
        
        if not recent_metrics:
            return {}
        
        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)
        
        # Calculate current values
        current_metrics = {}
        for metric_name, values in metric_groups.items():
            if values:
                current_metrics[metric_name] = {
                    "current": values[-1],
                    "avg": statistics.mean(values),
                    "trend": "up" if len(values) > 1 and values[-1] > values[0] else "down"
                }
        
        return current_metrics
    
    def _calculate_system_health(self, trace_stats: Dict[str, Any], 
                               error_analysis: Dict[str, Any]) -> str:
        """Calculate overall system health"""
        error_rate = error_analysis.get("error_rate", 0)
        
        if error_rate > 10:
            return "critical"
        elif error_rate > 5:
            return "warning"
        elif error_rate > 1:
            return "degraded"
        else:
            return "healthy"
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def get_trace_details(self, trace_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific trace"""
        trace = self.tracer.get_trace(trace_id)
        if not trace:
            return {"error": "Trace not found"}
        
        span_tree = trace.build_span_tree()
        
        # Calculate additional metrics
        critical_path = self._find_critical_path(trace)
        bottlenecks = await self._analyze_trace_bottlenecks(trace)
        
        return {
            "trace": {
                "trace_id": trace.trace_id,
                "duration": trace.duration,
                "span_count": len(trace.spans),
                "start_time": trace.start_time,
                "end_time": trace.end_time
            },
            "span_tree": span_tree,
            "critical_path": critical_path,
            "bottlenecks": bottlenecks
        }
    
    def _find_critical_path(self, trace: Trace) -> List[Dict[str, Any]]:
        """Find the critical path (longest duration path) in the trace"""
        # Implementation would find the path with longest cumulative duration
        # This is a simplified version
        longest_spans = sorted(trace.spans, key=lambda s: s.duration or 0, reverse=True)[:5]
        
        return [{
            "span_id": span.context.span_id,
            "operation_name": span.operation_name,
            "duration": span.duration,
            "service": span.tags.get("service.name", "unknown")
        } for span in longest_spans]
    
    async def _analyze_trace_bottlenecks(self, trace: Trace) -> List[Dict[str, Any]]:
        """Analyze bottlenecks in a specific trace"""
        bottlenecks = []
        
        # Find spans with unusually long durations
        durations = [span.duration for span in trace.spans if span.duration]
        if durations:
            avg_duration = statistics.mean(durations)
            threshold = avg_duration * 3  # 3x average
            
            for span in trace.spans:
                if span.duration and span.duration > threshold:
                    bottlenecks.append({
                        "span_id": span.context.span_id,
                        "operation_name": span.operation_name,
                        "duration": span.duration,
                        "severity": "high" if span.duration > threshold * 2 else "medium",
                        "type": "slow_operation"
                    })
        
        return bottlenecks
```