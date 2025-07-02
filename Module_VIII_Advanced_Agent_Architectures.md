# Module VIII: Advanced Agent Architectures

## Learning Objectives

By the end of this module, you will be able to:

1. **Design Multi-Agent Systems** - Architect complex systems with multiple specialized agents
2. **Implement Hierarchical Agent Structures** - Build agent hierarchies with delegation and coordination
3. **Create Swarm Intelligence Systems** - Develop emergent behavior through agent swarms
4. **Build Hybrid Agent Architectures** - Combine different agent paradigms for optimal performance
5. **Design Adaptive Agent Networks** - Create self-organizing and evolving agent systems
6. **Implement Agent Specialization Patterns** - Build domain-specific agent architectures

## Introduction to Advanced Agent Architectures

Advanced agent architectures go beyond single-agent systems to create sophisticated, multi-agent ecosystems that can solve complex problems through collaboration, specialization, and emergent behavior. These architectures leverage the collective intelligence of multiple agents working together.

### Architecture Overview

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Advanced Agent Architecture Patterns</text>
  
  <!-- Hierarchical Architecture -->
  <g transform="translate(50, 60)">
    <text x="100" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#34495e">Hierarchical</text>
    <!-- Coordinator -->
    <rect x="75" y="30" width="50" height="30" fill="#3498db" stroke="#2980b9" stroke-width="2" rx="5"/>
    <text x="100" y="50" text-anchor="middle" font-size="10" fill="white">Coordinator</text>
    <!-- Managers -->
    <rect x="25" y="80" width="40" height="25" fill="#e74c3c" stroke="#c0392b" stroke-width="1" rx="3"/>
    <text x="45" y="97" text-anchor="middle" font-size="8" fill="white">Manager</text>
    <rect x="85" y="80" width="40" height="25" fill="#e74c3c" stroke="#c0392b" stroke-width="1" rx="3"/>
    <text x="105" y="97" text-anchor="middle" font-size="8" fill="white">Manager</text>
    <rect x="135" y="80" width="40" height="25" fill="#e74c3c" stroke="#c0392b" stroke-width="1" rx="3"/>
    <text x="155" y="97" text-anchor="middle" font-size="8" fill="white">Manager</text>
    <!-- Workers -->
    <rect x="15" y="120" width="25" height="20" fill="#f39c12" stroke="#e67e22" stroke-width="1" rx="2"/>
    <text x="27" y="133" text-anchor="middle" font-size="7" fill="white">Worker</text>
    <rect x="45" y="120" width="25" height="20" fill="#f39c12" stroke="#e67e22" stroke-width="1" rx="2"/>
    <text x="57" y="133" text-anchor="middle" font-size="7" fill="white">Worker</text>
    <rect x="85" y="120" width="25" height="20" fill="#f39c12" stroke="#e67e22" stroke-width="1" rx="2"/>
    <text x="97" y="133" text-anchor="middle" font-size="7" fill="white">Worker</text>
    <rect x="115" y="120" width="25" height="20" fill="#f39c12" stroke="#e67e22" stroke-width="1" rx="2"/>
    <text x="127" y="133" text-anchor="middle" font-size="7" fill="white">Worker</text>
    <rect x="145" y="120" width="25" height="20" fill="#f39c12" stroke="#e67e22" stroke-width="1" rx="2"/>
    <text x="157" y="133" text-anchor="middle" font-size="7" fill="white">Worker</text>
    <rect x="175" y="120" width="25" height="20" fill="#f39c12" stroke="#e67e22" stroke-width="1" rx="2"/>
    <text x="187" y="133" text-anchor="middle" font-size="7" fill="white">Worker</text>
    <!-- Connections -->
    <line x1="100" y1="60" x2="45" y2="80" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="100" y1="60" x2="105" y2="80" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="100" y1="60" x2="155" y2="80" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="45" y1="105" x2="27" y2="120" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="45" y1="105" x2="57" y2="120" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="105" y1="105" x2="97" y2="120" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="105" y1="105" x2="127" y2="120" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="155" y1="105" x2="157" y2="120" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="155" y1="105" x2="187" y2="120" stroke="#7f8c8d" stroke-width="1"/>
  </g>
  
  <!-- Swarm Architecture -->
  <g transform="translate(300, 60)">
    <text x="100" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#34495e">Swarm Intelligence</text>
    <!-- Swarm agents -->
    <circle cx="50" cy="50" r="12" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
    <text x="50" y="55" text-anchor="middle" font-size="8" fill="white">A1</text>
    <circle cx="100" cy="40" r="12" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
    <text x="100" y="45" text-anchor="middle" font-size="8" fill="white">A2</text>
    <circle cx="150" cy="55" r="12" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
    <text x="150" y="60" text-anchor="middle" font-size="8" fill="white">A3</text>
    <circle cx="75" cy="90" r="12" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
    <text x="75" y="95" text-anchor="middle" font-size="8" fill="white">A4</text>
    <circle cx="125" cy="85" r="12" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
    <text x="125" y="90" text-anchor="middle" font-size="8" fill="white">A5</text>
    <circle cx="90" cy="120" r="12" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
    <text x="90" y="125" text-anchor="middle" font-size="8" fill="white">A6</text>
    <!-- Swarm connections -->
    <line x1="50" y1="50" x2="100" y2="40" stroke="#9b59b6" stroke-width="1" stroke-dasharray="3,3"/>
    <line x1="100" y1="40" x2="150" y2="55" stroke="#9b59b6" stroke-width="1" stroke-dasharray="3,3"/>
    <line x1="50" y1="50" x2="75" y2="90" stroke="#9b59b6" stroke-width="1" stroke-dasharray="3,3"/>
    <line x1="75" y1="90" x2="125" y2="85" stroke="#9b59b6" stroke-width="1" stroke-dasharray="3,3"/>
    <line x1="125" y1="85" x2="150" y2="55" stroke="#9b59b6" stroke-width="1" stroke-dasharray="3,3"/>
    <line x1="75" y1="90" x2="90" y2="120" stroke="#9b59b6" stroke-width="1" stroke-dasharray="3,3"/>
    <line x1="125" y1="85" x2="90" y2="120" stroke="#9b59b6" stroke-width="1" stroke-dasharray="3,3"/>
  </g>
  
  <!-- Hybrid Architecture -->
  <g transform="translate(550, 60)">
    <text x="100" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#34495e">Hybrid</text>
    <!-- Central coordinator -->
    <rect x="75" y="30" width="50" height="30" fill="#1abc9c" stroke="#16a085" stroke-width="2" rx="5"/>
    <text x="100" y="50" text-anchor="middle" font-size="10" fill="white">Hub</text>
    <!-- Specialized agents -->
    <rect x="20" y="80" width="35" height="25" fill="#e67e22" stroke="#d35400" stroke-width="1" rx="3"/>
    <text x="37" y="97" text-anchor="middle" font-size="8" fill="white">NLP</text>
    <rect x="65" y="80" width="35" height="25" fill="#e67e22" stroke="#d35400" stroke-width="1" rx="3"/>
    <text x="82" y="97" text-anchor="middle" font-size="8" fill="white">Vision</text>
    <rect x="110" y="80" width="35" height="25" fill="#e67e22" stroke="#d35400" stroke-width="1" rx="3"/>
    <text x="127" y="97" text-anchor="middle" font-size="8" fill="white">Logic</text>
    <rect x="155" y="80" width="35" height="25" fill="#e67e22" stroke="#d35400" stroke-width="1" rx="3"/>
    <text x="172" y="97" text-anchor="middle" font-size="8" fill="white">Data</text>
    <!-- Swarm cluster -->
    <circle cx="50" cy="130" r="8" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
    <text x="50" y="134" text-anchor="middle" font-size="6" fill="white">S1</text>
    <circle cx="70" cy="125" r="8" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
    <text x="70" y="129" text-anchor="middle" font-size="6" fill="white">S2</text>
    <circle cx="60" cy="145" r="8" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
    <text x="60" y="149" text-anchor="middle" font-size="6" fill="white">S3</text>
    <!-- Connections -->
    <line x1="100" y1="60" x2="37" y2="80" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="100" y1="60" x2="82" y2="80" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="100" y1="60" x2="127" y2="80" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="100" y1="60" x2="172" y2="80" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="82" y1="105" x2="60" y2="125" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="50" y1="130" x2="70" y2="125" stroke="#9b59b6" stroke-width="1" stroke-dasharray="2,2"/>
    <line x1="70" y1="125" x2="60" y2="145" stroke="#9b59b6" stroke-width="1" stroke-dasharray="2,2"/>
    <line x1="60" y1="145" x2="50" y2="130" stroke="#9b59b6" stroke-width="1" stroke-dasharray="2,2"/>
  </g>
  
  <!-- Federated Architecture -->
  <g transform="translate(50, 220)">
    <text x="150" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#34495e">Federated Networks</text>
    <!-- Federation nodes -->
    <rect x="50" y="40" width="60" height="40" fill="#2ecc71" stroke="#27ae60" stroke-width="2" rx="5"/>
    <text x="80" y="65" text-anchor="middle" font-size="10" fill="white">Domain A</text>
    <rect x="140" y="40" width="60" height="40" fill="#2ecc71" stroke="#27ae60" stroke-width="2" rx="5"/>
    <text x="170" y="65" text-anchor="middle" font-size="10" fill="white">Domain B</text>
    <rect x="230" y="40" width="60" height="40" fill="#2ecc71" stroke="#27ae60" stroke-width="2" rx="5"/>
    <text x="260" y="65" text-anchor="middle" font-size="10" fill="white">Domain C</text>
    <!-- Sub-agents -->
    <circle cx="65" cy="100" r="8" fill="#f39c12" stroke="#e67e22" stroke-width="1"/>
    <circle cx="95" cy="100" r="8" fill="#f39c12" stroke="#e67e22" stroke-width="1"/>
    <circle cx="155" cy="100" r="8" fill="#f39c12" stroke="#e67e22" stroke-width="1"/>
    <circle cx="185" cy="100" r="8" fill="#f39c12" stroke="#e67e22" stroke-width="1"/>
    <circle cx="245" cy="100" r="8" fill="#f39c12" stroke="#e67e22" stroke-width="1"/>
    <circle cx="275" cy="100" r="8" fill="#f39c12" stroke="#e67e22" stroke-width="1"/>
    <!-- Inter-domain connections -->
    <line x1="110" y1="60" x2="140" y2="60" stroke="#34495e" stroke-width="2"/>
    <line x1="200" y1="60" x2="230" y2="60" stroke="#34495e" stroke-width="2"/>
    <!-- Intra-domain connections -->
    <line x1="80" y1="80" x2="65" y2="92" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="80" y1="80" x2="95" y2="92" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="170" y1="80" x2="155" y2="92" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="170" y1="80" x2="185" y2="92" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="260" y1="80" x2="245" y2="92" stroke="#7f8c8d" stroke-width="1"/>
    <line x1="260" y1="80" x2="275" y2="92" stroke="#7f8c8d" stroke-width="1"/>
  </g>
  
  <!-- Adaptive Architecture -->
  <g transform="translate(450, 220)">
    <text x="150" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#34495e">Adaptive Networks</text>
    <!-- Core adaptive system -->
    <ellipse cx="150" cy="70" rx="40" ry="25" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
    <text x="150" y="75" text-anchor="middle" font-size="10" fill="white">Adaptive Core</text>
    <!-- Dynamic agents -->
    <rect x="80" y="40" width="30" height="20" fill="#3498db" stroke="#2980b9" stroke-width="1" rx="3"/>
    <text x="95" y="53" text-anchor="middle" font-size="8" fill="white">Dyn1</text>
    <rect x="220" y="50" width="30" height="20" fill="#3498db" stroke="#2980b9" stroke-width="1" rx="3"/>
    <text x="235" y="63" text-anchor="middle" font-size="8" fill="white">Dyn2</text>
    <rect x="90" y="100" width="30" height="20" fill="#3498db" stroke="#2980b9" stroke-width="1" rx="3"/>
    <text x="105" y="113" text-anchor="middle" font-size="8" fill="white">Dyn3</text>
    <rect x="210" y="95" width="30" height="20" fill="#3498db" stroke="#2980b9" stroke-width="1" rx="3"/>
    <text x="225" y="108" text-anchor="middle" font-size="8" fill="white">Dyn4</text>
    <!-- Adaptive connections -->
    <line x1="110" y1="50" x2="130" y2="60" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5"/>
    <line x1="220" y1="60" x2="180" y2="65" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5"/>
    <line x1="120" y1="110" x2="135" y2="85" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5"/>
    <line x1="210" y1="105" x2="175" y2="80" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5"/>
    <!-- Evolution indicator -->
    <path d="M 150 110 Q 160 120 170 110 Q 180 100 190 110" stroke="#f39c12" stroke-width="2" fill="none"/>
    <text x="170" y="135" text-anchor="middle" font-size="8" fill="#f39c12">Evolution</text>
  </g>
  
  <!-- Legend -->
  <g transform="translate(50, 400)">
    <text x="0" y="15" font-size="14" font-weight="bold" fill="#2c3e50">Architecture Patterns:</text>
    <rect x="0" y="25" width="15" height="10" fill="#3498db"/>
    <text x="20" y="34" font-size="12" fill="#34495e">Coordinator/Manager Agents</text>
    <rect x="200" y="25" width="15" height="10" fill="#e74c3c"/>
    <text x="220" y="34" font-size="12" fill="#34495e">Specialized Agents</text>
    <circle cx="7" cy="50" r="7" fill="#9b59b6"/>
    <text x="20" y="55" font-size="12" fill="#34495e">Swarm Agents</text>
    <rect x="200" y="45" width="15" height="10" fill="#f39c12"/>
    <text x="220" y="54" font-size="12" fill="#34495e">Worker/Execution Agents</text>
    <rect x="0" y="65" width="15" height="10" fill="#2ecc71"/>
    <text x="20" y="74" font-size="12" fill="#34495e">Domain/Federation Nodes</text>
    <rect x="200" y="65" width="15" height="10" fill="#1abc9c"/>
    <text x="220" y="74" font-size="12" fill="#34495e">Hub/Gateway Agents</text>
  </g>
  
  <!-- Key Features -->
  <g transform="translate(450, 400)">
    <text x="0" y="15" font-size="14" font-weight="bold" fill="#2c3e50">Key Features:</text>
    <text x="0" y="35" font-size="12" fill="#34495e">• Hierarchical delegation and control</text>
    <text x="0" y="50" font-size="12" fill="#34495e">• Emergent swarm intelligence</text>
    <text x="0" y="65" font-size="12" fill="#34495e">• Specialized domain expertise</text>
    <text x="0" y="80" font-size="12" fill="#34495e">• Adaptive network topology</text>
    <text x="0" y="95" font-size="12" fill="#34495e">• Cross-domain federation</text>
    <text x="0" y="110" font-size="12" fill="#34495e">• Dynamic agent composition</text>
  </g>
</svg>
```

## 1. Multi-Agent System Foundations

### 1.1 Agent Coordination Patterns

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import asyncio
import uuid
from datetime import datetime, timedelta
import json
import logging

class CoordinationPattern(Enum):
    """Different coordination patterns for multi-agent systems"""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    MARKET_BASED = "market_based"
    CONTRACT_NET = "contract_net"
    BLACKBOARD = "blackboard"

class AgentRole(Enum):
    """Roles that agents can play in the system"""
    COORDINATOR = "coordinator"
    MANAGER = "manager"
    WORKER = "worker"
    SPECIALIST = "specialist"
    BROKER = "broker"
    MONITOR = "monitor"

@dataclass
class AgentCapability:
    """Represents a capability that an agent possesses"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
    reliability: float = 1.0

@dataclass
class Task:
    """Represents a task in the multi-agent system"""
    id: str
    name: str
    description: str
    required_capabilities: List[str]
    input_data: Dict[str, Any]
    priority: int = 1
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[timedelta] = None
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""
    
    def __init__(self, agent_id: str, name: str, role: AgentRole):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.capabilities: List[AgentCapability] = []
        self.current_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[str] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.is_active = True
        self.load_factor = 0.0
        self.trust_score = 1.0
        
    def add_capability(self, capability: AgentCapability):
        """Add a capability to this agent"""
        self.capabilities.append(capability)
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(cap.name == capability_name for cap in self.capabilities)
    
    def get_capability(self, capability_name: str) -> Optional[AgentCapability]:
        """Get a specific capability"""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap
        return None
    
    def calculate_load_factor(self) -> float:
        """Calculate current load factor based on active tasks"""
        if not self.current_tasks:
            self.load_factor = 0.0
        else:
            # Simple load calculation based on number of tasks
            self.load_factor = min(len(self.current_tasks) / 5.0, 1.0)
        return self.load_factor
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task and return the result"""
        pass
    
    @abstractmethod
    async def can_handle_task(self, task: Task) -> bool:
        """Check if this agent can handle the given task"""
        pass
    
    async def assign_task(self, task: Task) -> bool:
        """Assign a task to this agent"""
        if await self.can_handle_task(task) and len(self.current_tasks) < 5:
            task.assigned_agent = self.agent_id
            task.status = "assigned"
            self.current_tasks[task.id] = task
            return True
        return False
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as completed"""
        if task_id in self.current_tasks:
            task = self.current_tasks[task_id]
            task.status = "completed"
            task.result = result
            self.completed_tasks.append(task_id)
            del self.current_tasks[task_id]
            
            # Update performance history
            self.performance_history.append({
                "task_id": task_id,
                "completion_time": datetime.now(),
                "duration": datetime.now() - task.created_at,
                "success": True
            })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this agent"""
        if not self.performance_history:
            return {"success_rate": 1.0, "avg_duration": 0.0, "task_count": 0}
        
        successful_tasks = [h for h in self.performance_history if h["success"]]
        success_rate = len(successful_tasks) / len(self.performance_history)
        
        if successful_tasks:
            avg_duration = sum(h["duration"].total_seconds() for h in successful_tasks) / len(successful_tasks)
        else:
            avg_duration = 0.0
        
        return {
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "task_count": len(self.performance_history),
            "current_load": self.calculate_load_factor()
        }

class CoordinatorAgent(BaseAgent):
    """Coordinator agent that manages task distribution and system coordination"""
    
    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, AgentRole.COORDINATOR)
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []
        self.coordination_strategy = CoordinationPattern.CENTRALIZED
        
        # Add coordination capabilities
        self.add_capability(AgentCapability(
            name="task_coordination",
            description="Coordinate task distribution among agents",
            input_schema={"tasks": "array", "agents": "array"},
            output_schema={"assignments": "object"}
        ))
        
        self.add_capability(AgentCapability(
            name="resource_management",
            description="Manage system resources and agent allocation",
            input_schema={"resources": "object"},
            output_schema={"allocation": "object"}
        ))
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent under this coordinator"""
        self.managed_agents[agent.agent_id] = agent
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.managed_agents:
            del self.managed_agents[agent_id]
    
    async def can_handle_task(self, task: Task) -> bool:
        """Coordinator can handle coordination tasks"""
        return any(cap.name in ["task_coordination", "resource_management"] 
                  for cap in self.capabilities)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute coordination tasks"""
        if task.name == "coordinate_tasks":
            return await self.coordinate_tasks(task.input_data.get("tasks", []))
        elif task.name == "manage_resources":
            return await self.manage_resources(task.input_data.get("resources", {}))
        else:
            raise ValueError(f"Unknown coordination task: {task.name}")
    
    async def coordinate_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate the distribution of tasks among managed agents"""
        assignments = {}
        
        for task_data in tasks:
            task = Task(
                id=str(uuid.uuid4()),
                name=task_data["name"],
                description=task_data.get("description", ""),
                required_capabilities=task_data.get("required_capabilities", []),
                input_data=task_data.get("input_data", {}),
                priority=task_data.get("priority", 1)
            )
            
            # Find best agent for this task
            best_agent = await self.find_best_agent(task)
            if best_agent:
                success = await best_agent.assign_task(task)
                if success:
                    assignments[task.id] = {
                        "agent_id": best_agent.agent_id,
                        "agent_name": best_agent.name,
                        "task_name": task.name
                    }
                else:
                    # Add to queue if assignment failed
                    self.task_queue.append(task)
            else:
                self.task_queue.append(task)
        
        return {
            "assignments": assignments,
            "queued_tasks": len(self.task_queue),
            "total_tasks": len(tasks)
        }
    
    async def find_best_agent(self, task: Task) -> Optional[BaseAgent]:
        """Find the best agent to handle a specific task"""
        suitable_agents = []
        
        for agent in self.managed_agents.values():
            if not agent.is_active:
                continue
                
            # Check if agent has required capabilities
            has_all_capabilities = all(
                agent.has_capability(cap) for cap in task.required_capabilities
            )
            
            if has_all_capabilities and await agent.can_handle_task(task):
                # Calculate agent score based on performance and load
                metrics = agent.get_performance_metrics()
                score = (
                    metrics["success_rate"] * 0.4 +
                    (1 - metrics["current_load"]) * 0.3 +
                    agent.trust_score * 0.3
                )
                suitable_agents.append((agent, score))
        
        if suitable_agents:
            # Sort by score and return best agent
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]
        
        return None
    
    async def manage_resources(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Manage system resources and agent allocation"""
        resource_allocation = {}
        
        # Calculate total system load
        total_load = sum(agent.calculate_load_factor() 
                        for agent in self.managed_agents.values())
        avg_load = total_load / len(self.managed_agents) if self.managed_agents else 0
        
        # Identify overloaded and underloaded agents
        overloaded_agents = [agent for agent in self.managed_agents.values() 
                           if agent.calculate_load_factor() > 0.8]
        underloaded_agents = [agent for agent in self.managed_agents.values() 
                            if agent.calculate_load_factor() < 0.3]
        
        resource_allocation = {
            "total_agents": len(self.managed_agents),
            "active_agents": len([a for a in self.managed_agents.values() if a.is_active]),
            "average_load": avg_load,
            "overloaded_agents": len(overloaded_agents),
            "underloaded_agents": len(underloaded_agents),
            "queued_tasks": len(self.task_queue)
        }
        
        return resource_allocation
    
    async def rebalance_load(self):
        """Rebalance load across agents by redistributing tasks"""
        # This is a simplified load balancing strategy
        overloaded_agents = [agent for agent in self.managed_agents.values() 
                           if agent.calculate_load_factor() > 0.8]
        underloaded_agents = [agent for agent in self.managed_agents.values() 
                            if agent.calculate_load_factor() < 0.3]
        
        for overloaded_agent in overloaded_agents:
            if not underloaded_agents:
                break
                
            # Move some tasks from overloaded to underloaded agents
            tasks_to_move = list(overloaded_agent.current_tasks.values())[:2]
            
            for task in tasks_to_move:
                for underloaded_agent in underloaded_agents:
                    if await underloaded_agent.can_handle_task(task):
                        # Move task
                        del overloaded_agent.current_tasks[task.id]
                        task.assigned_agent = underloaded_agent.agent_id
                        underloaded_agent.current_tasks[task.id] = task
                        break
```

### 1.2 Specialized Agent Types

```python
class SpecialistAgent(BaseAgent):
    """Specialist agent with domain-specific expertise"""
    
    def __init__(self, agent_id: str, name: str, domain: str):
        super().__init__(agent_id, name, AgentRole.SPECIALIST)
        self.domain = domain
        self.expertise_level = 1.0
        self.learning_rate = 0.1
    
    async def can_handle_task(self, task: Task) -> bool:
        """Check if this specialist can handle the task"""
        # Check if task requires domain expertise
        domain_match = (
            self.domain.lower() in task.description.lower() or
            any(self.domain.lower() in cap.lower() for cap in task.required_capabilities)
        )
        
        # Check capability match
        capability_match = any(
            self.has_capability(cap) for cap in task.required_capabilities
        )
        
        return domain_match and capability_match and len(self.current_tasks) < 3
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute task with domain expertise"""
        # Simulate domain-specific processing
        await asyncio.sleep(1)  # Simulate processing time
        
        # Apply domain expertise
        expertise_bonus = self.expertise_level * 0.2
        quality_score = min(0.8 + expertise_bonus, 1.0)
        
        result = {
            "task_id": task.id,
            "agent_id": self.agent_id,
            "domain": self.domain,
            "quality_score": quality_score,
            "expertise_applied": self.expertise_level,
            "output": f"Domain-specific result for {task.name} in {self.domain}"
        }
        
        # Learn from task execution
        self.expertise_level = min(self.expertise_level + self.learning_rate * 0.1, 2.0)
        
        return result

class WorkerAgent(BaseAgent):
    """Worker agent for general task execution"""
    
    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, AgentRole.WORKER)
        self.max_concurrent_tasks = 5
        
        # Add general capabilities
        self.add_capability(AgentCapability(
            name="data_processing",
            description="Process and transform data",
            input_schema={"data": "object"},
            output_schema={"processed_data": "object"}
        ))
        
        self.add_capability(AgentCapability(
            name="computation",
            description="Perform computational tasks",
            input_schema={"operation": "string", "parameters": "object"},
            output_schema={"result": "object"}
        ))
    
    async def can_handle_task(self, task: Task) -> bool:
        """Check if worker can handle the task"""
        # Workers can handle tasks if they have the required capabilities
        # and are not overloaded
        has_capabilities = all(
            self.has_capability(cap) for cap in task.required_capabilities
        )
        
        not_overloaded = len(self.current_tasks) < self.max_concurrent_tasks
        
        return has_capabilities and not_overloaded
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute general tasks"""
        # Simulate task processing
        processing_time = task.input_data.get("complexity", 1) * 0.5
        await asyncio.sleep(processing_time)
        
        result = {
            "task_id": task.id,
            "agent_id": self.agent_id,
            "processing_time": processing_time,
            "output": f"Processed {task.name} successfully",
            "metadata": {
                "worker_load": self.calculate_load_factor(),
                "capabilities_used": task.required_capabilities
            }
        }
        
        return result

class BrokerAgent(BaseAgent):
    """Broker agent for facilitating agent interactions and negotiations"""
    
    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, AgentRole.BROKER)
        self.active_negotiations: Dict[str, Dict[str, Any]] = {}
        self.market_prices: Dict[str, float] = {}
        
        # Add broker capabilities
        self.add_capability(AgentCapability(
            name="negotiation",
            description="Facilitate negotiations between agents",
            input_schema={"parties": "array", "terms": "object"},
            output_schema={"agreement": "object"}
        ))
        
        self.add_capability(AgentCapability(
            name="market_making",
            description="Provide market-making services for agent capabilities",
            input_schema={"capability": "string", "demand": "number"},
            output_schema={"price": "number"}
        ))
    
    async def can_handle_task(self, task: Task) -> bool:
        """Check if broker can handle the task"""
        broker_tasks = ["negotiation", "market_making", "resource_allocation"]
        return any(task_type in task.name.lower() for task_type in broker_tasks)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute broker tasks"""
        if "negotiation" in task.name.lower():
            return await self.facilitate_negotiation(task.input_data)
        elif "market" in task.name.lower():
            return await self.provide_market_service(task.input_data)
        else:
            return {"error": "Unknown broker task"}
    
    async def facilitate_negotiation(self, negotiation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Facilitate negotiation between agents"""
        negotiation_id = str(uuid.uuid4())
        parties = negotiation_data.get("parties", [])
        terms = negotiation_data.get("terms", {})
        
        # Simulate negotiation process
        await asyncio.sleep(0.5)
        
        # Simple negotiation logic
        agreement = {
            "negotiation_id": negotiation_id,
            "parties": parties,
            "agreed_terms": terms,
            "broker_fee": 0.05,
            "status": "agreed",
            "timestamp": datetime.now().isoformat()
        }
        
        self.active_negotiations[negotiation_id] = agreement
        
        return agreement
    
    async def provide_market_service(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide market-making services"""
        capability = market_data.get("capability")
        demand = market_data.get("demand", 1.0)
        
        # Simple pricing model based on demand
        base_price = 1.0
        price = base_price * (1 + demand * 0.1)
        
        self.market_prices[capability] = price
        
        return {
            "capability": capability,
            "current_price": price,
            "demand_factor": demand,
            "market_depth": len(self.market_prices)
        }
```

## 2. Hierarchical Agent Structures

### 2.1 Multi-Level Hierarchy Implementation

```python
class HierarchicalLevel(Enum):
    """Levels in the hierarchical agent structure"""
    EXECUTIVE = 1
    MANAGEMENT = 2
    OPERATIONAL = 3
    WORKER = 4

@dataclass
class HierarchicalNode:
    """Node in the hierarchical structure"""
    agent: BaseAgent
    level: HierarchicalLevel
    parent: Optional['HierarchicalNode'] = None
    children: List['HierarchicalNode'] = field(default_factory=list)
    authority_scope: Set[str] = field(default_factory=set)
    delegation_rules: Dict[str, Any] = field(default_factory=dict)

class HierarchicalAgentSystem:
    """Manages hierarchical agent structures with delegation and coordination"""
    
    def __init__(self, name: str):
        self.name = name
        self.root_node: Optional[HierarchicalNode] = None
        self.nodes: Dict[str, HierarchicalNode] = {}
        self.delegation_policies: Dict[str, Callable] = {}
        self.escalation_rules: List[Dict[str, Any]] = []
    
    def set_root_agent(self, agent: BaseAgent, authority_scope: Set[str]):
        """Set the root (executive) agent"""
        self.root_node = HierarchicalNode(
            agent=agent,
            level=HierarchicalLevel.EXECUTIVE,
            authority_scope=authority_scope
        )
        self.nodes[agent.agent_id] = self.root_node
    
    def add_agent(self, agent: BaseAgent, parent_id: str, 
                  level: HierarchicalLevel, authority_scope: Set[str]) -> bool:
        """Add an agent to the hierarchy under a parent"""
        if parent_id not in self.nodes:
            return False
        
        parent_node = self.nodes[parent_id]
        
        # Validate hierarchy level constraints
        if level.value <= parent_node.level.value:
            return False
        
        new_node = HierarchicalNode(
            agent=agent,
            level=level,
            parent=parent_node,
            authority_scope=authority_scope
        )
        
        parent_node.children.append(new_node)
        self.nodes[agent.agent_id] = new_node
        
        return True
    
    def set_delegation_policy(self, policy_name: str, policy_func: Callable):
        """Set a delegation policy function"""
        self.delegation_policies[policy_name] = policy_func
    
    def add_escalation_rule(self, condition: str, target_level: HierarchicalLevel):
        """Add an escalation rule"""
        self.escalation_rules.append({
            "condition": condition,
            "target_level": target_level
        })
    
    async def delegate_task(self, task: Task, delegator_id: str) -> Optional[str]:
        """Delegate a task down the hierarchy"""
        if delegator_id not in self.nodes:
            return None
        
        delegator_node = self.nodes[delegator_id]
        
        # Check if delegator has authority for this task
        if not self._has_authority(delegator_node, task):
            return None
        
        # Find best subordinate for delegation
        best_subordinate = await self._find_best_subordinate(delegator_node, task)
        
        if best_subordinate:
            success = await best_subordinate.agent.assign_task(task)
            if success:
                return best_subordinate.agent.agent_id
        
        return None
    
    async def escalate_task(self, task: Task, escalator_id: str) -> Optional[str]:
        """Escalate a task up the hierarchy"""
        if escalator_id not in self.nodes:
            return None
        
        escalator_node = self.nodes[escalator_id]
        
        # Check escalation rules
        target_level = self._determine_escalation_level(task)
        
        # Find appropriate superior
        current_node = escalator_node.parent
        while current_node and current_node.level.value > target_level.value:
            current_node = current_node.parent
        
        if current_node:
            success = await current_node.agent.assign_task(task)
            if success:
                return current_node.agent.agent_id
        
        return None
    
    def _has_authority(self, node: HierarchicalNode, task: Task) -> bool:
        """Check if a node has authority to handle/delegate a task"""
        # Check if any required capability is in authority scope
        return any(cap in node.authority_scope for cap in task.required_capabilities)
    
    async def _find_best_subordinate(self, parent_node: HierarchicalNode, 
                                   task: Task) -> Optional[HierarchicalNode]:
        """Find the best subordinate to delegate a task to"""
        suitable_subordinates = []
        
        for child_node in parent_node.children:
            if (await child_node.agent.can_handle_task(task) and 
                self._has_authority(child_node, task)):
                
                # Calculate suitability score
                metrics = child_node.agent.get_performance_metrics()
                score = (
                    metrics["success_rate"] * 0.5 +
                    (1 - metrics["current_load"]) * 0.3 +
                    child_node.agent.trust_score * 0.2
                )
                suitable_subordinates.append((child_node, score))
        
        if suitable_subordinates:
            suitable_subordinates.sort(key=lambda x: x[1], reverse=True)
            return suitable_subordinates[0][0]
        
        return None
    
    def _determine_escalation_level(self, task: Task) -> HierarchicalLevel:
        """Determine the appropriate escalation level for a task"""
        # Apply escalation rules
        for rule in self.escalation_rules:
            if self._evaluate_escalation_condition(rule["condition"], task):
                return rule["target_level"]
        
        # Default escalation to management level
        return HierarchicalLevel.MANAGEMENT
    
    def _evaluate_escalation_condition(self, condition: str, task: Task) -> bool:
        """Evaluate an escalation condition"""
        # Simple condition evaluation (can be extended)
        if condition == "high_priority" and task.priority >= 8:
            return True
        elif condition == "complex_task" and len(task.required_capabilities) > 3:
            return True
        elif condition == "urgent_deadline" and task.deadline:
            time_remaining = task.deadline - datetime.now()
            return time_remaining.total_seconds() < 3600  # Less than 1 hour
        
        return False
    
    def get_hierarchy_structure(self) -> Dict[str, Any]:
        """Get the current hierarchy structure"""
        if not self.root_node:
            return {}
        
        def build_structure(node: HierarchicalNode) -> Dict[str, Any]:
            return {
                "agent_id": node.agent.agent_id,
                "agent_name": node.agent.name,
                "level": node.level.name,
                "authority_scope": list(node.authority_scope),
                "current_load": node.agent.calculate_load_factor(),
                "children": [build_structure(child) for child in node.children]
            }
        
        return build_structure(self.root_node)

class ExecutiveAgent(CoordinatorAgent):
    """Executive-level agent with high-level decision making"""
    
    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name)
        self.role = AgentRole.COORDINATOR
        self.strategic_goals: List[Dict[str, Any]] = []
        self.policy_framework: Dict[str, Any] = {}
        
        # Add executive capabilities
        self.add_capability(AgentCapability(
            name="strategic_planning",
            description="Develop and execute strategic plans",
            input_schema={"objectives": "array", "constraints": "object"},
            output_schema={"strategy": "object"}
        ))
        
        self.add_capability(AgentCapability(
            name="policy_making",
            description="Create and enforce organizational policies",
            input_schema={"domain": "string", "requirements": "object"},
            output_schema={"policy": "object"}
        ))
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute executive-level tasks"""
        if "strategic" in task.name.lower():
            return await self.develop_strategy(task.input_data)
        elif "policy" in task.name.lower():
            return await self.create_policy(task.input_data)
        else:
            return await super().execute_task(task)
    
    async def develop_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop strategic plans"""
        objectives = strategy_data.get("objectives", [])
        constraints = strategy_data.get("constraints", {})
        
        # Simulate strategic planning
        await asyncio.sleep(2)  # Strategic thinking takes time
        
        strategy = {
            "strategy_id": str(uuid.uuid4()),
            "objectives": objectives,
            "constraints": constraints,
            "action_items": [
                f"Implement objective: {obj}" for obj in objectives
            ],
            "timeline": "6 months",
            "success_metrics": ["efficiency", "quality", "cost_reduction"],
            "created_by": self.agent_id,
            "created_at": datetime.now().isoformat()
        }
        
        self.strategic_goals.append(strategy)
        return strategy
    
    async def create_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create organizational policies"""
        domain = policy_data.get("domain")
        requirements = policy_data.get("requirements", {})
        
        policy = {
            "policy_id": str(uuid.uuid4()),
            "domain": domain,
            "requirements": requirements,
            "rules": [
                f"All agents must comply with {req}" 
                for req in requirements.keys()
            ],
            "enforcement_level": "mandatory",
            "created_by": self.agent_id,
            "effective_date": datetime.now().isoformat()
        }
        
        self.policy_framework[domain] = policy
        return policy

class ManagerAgent(CoordinatorAgent):
    """Manager-level agent for operational coordination"""
    
    def __init__(self, agent_id: str, name: str, department: str):
        super().__init__(agent_id, name)
        self.role = AgentRole.MANAGER
        self.department = department
        self.team_performance: Dict[str, Any] = {}
        self.resource_budget: Dict[str, float] = {}
        
        # Add management capabilities
        self.add_capability(AgentCapability(
            name="team_management",
            description="Manage team performance and coordination",
            input_schema={"team_members": "array", "objectives": "object"},
            output_schema={"performance_report": "object"}
        ))
        
        self.add_capability(AgentCapability(
            name="resource_allocation",
            description="Allocate resources within department",
            input_schema={"resources": "object", "priorities": "array"},
            output_schema={"allocation_plan": "object"}
        ))
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute management tasks"""
        if "team" in task.name.lower():
            return await self.manage_team(task.input_data)
        elif "resource" in task.name.lower():
            return await self.allocate_resources(task.input_data)
        else:
            return await super().execute_task(task)
    
    async def manage_team(self, team_data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage team performance"""
        team_members = team_data.get("team_members", [])
        objectives = team_data.get("objectives", {})
        
        # Analyze team performance
        team_metrics = {}
        for member_id in team_members:
            if member_id in self.managed_agents:
                agent = self.managed_agents[member_id]
                team_metrics[member_id] = agent.get_performance_metrics()
        
        # Calculate team performance
        if team_metrics:
            avg_success_rate = sum(m["success_rate"] for m in team_metrics.values()) / len(team_metrics)
            avg_load = sum(m["current_load"] for m in team_metrics.values()) / len(team_metrics)
        else:
            avg_success_rate = avg_load = 0.0
        
        performance_report = {
            "department": self.department,
            "team_size": len(team_members),
            "average_success_rate": avg_success_rate,
            "average_load": avg_load,
            "objectives_status": "on_track" if avg_success_rate > 0.8 else "needs_attention",
            "recommendations": self._generate_recommendations(avg_success_rate, avg_load),
            "report_date": datetime.now().isoformat()
        }
        
        self.team_performance[datetime.now().date().isoformat()] = performance_report
        return performance_report
    
    async def allocate_resources(self, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate departmental resources"""
        available_resources = resource_data.get("resources", {})
        priorities = resource_data.get("priorities", [])
        
        allocation_plan = {
            "department": self.department,
            "total_budget": sum(available_resources.values()),
            "allocations": {},
            "priority_funding": {}
        }
        
        # Allocate based on priorities
        total_budget = sum(available_resources.values())
        for i, priority in enumerate(priorities):
            # Higher priority gets more resources
            allocation_percentage = 0.4 / (i + 1)  # Decreasing allocation
            allocation_plan["allocations"][priority] = total_budget * allocation_percentage
        
        return allocation_plan
    
    def _generate_recommendations(self, success_rate: float, load: float) -> List[str]:
        """Generate management recommendations based on team metrics"""
        recommendations = []
        
        if success_rate < 0.7:
            recommendations.append("Consider additional training for team members")
            recommendations.append("Review task complexity and agent capabilities")
        
        if load > 0.8:
            recommendations.append("Team is overloaded - consider hiring or redistribution")
        elif load < 0.3:
            recommendations.append("Team has capacity for additional tasks")
        
        if success_rate > 0.9 and load < 0.6:
            recommendations.append("Team performing excellently - consider expansion")
        
        return recommendations
```

## 3. Swarm Intelligence Systems

### 3.1 Swarm Agent Implementation

```python
class SwarmBehavior(Enum):
    """Different swarm behaviors"""
    FLOCKING = "flocking"
    FORAGING = "foraging"
    CONSENSUS = "consensus"
    EXPLORATION = "exploration"
    OPTIMIZATION = "optimization"

@dataclass
class SwarmState:
    """State information for swarm agents"""
    position: List[float]
    velocity: List[float]
    fitness: float
    local_best: List[float]
    local_best_fitness: float
    neighbors: Set[str] = field(default_factory=set)
    pheromone_trails: Dict[str, float] = field(default_factory=dict)

class SwarmAgent(BaseAgent):
    """Agent that exhibits swarm intelligence behaviors"""
    
    def __init__(self, agent_id: str, name: str, swarm_id: str):
        super().__init__(agent_id, name, AgentRole.WORKER)
        self.swarm_id = swarm_id
        self.swarm_state = SwarmState(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            fitness=0.0,
            local_best=[0.0, 0.0],
            local_best_fitness=0.0
        )
        self.behavior_mode = SwarmBehavior.EXPLORATION
        self.communication_range = 5.0
        self.learning_rate = 0.1
        
        # Add swarm capabilities
        self.add_capability(AgentCapability(
            name="swarm_coordination",
            description="Coordinate with other swarm agents",
            input_schema={"neighbors": "array", "objective": "string"},
            output_schema={"coordination_result": "object"}
        ))
        
        self.add_capability(AgentCapability(
            name="collective_optimization",
            description="Participate in collective optimization",
            input_schema={"problem_space": "object", "constraints": "object"},
            output_schema={"optimization_contribution": "object"}
        ))
    
    async def can_handle_task(self, task: Task) -> bool:
        """Swarm agents can handle distributed and optimization tasks"""
        swarm_tasks = ["optimization", "search", "exploration", "distributed"]
        return any(task_type in task.name.lower() for task_type in swarm_tasks)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute swarm-based tasks"""
        if "optimization" in task.name.lower():
            return await self.participate_in_optimization(task.input_data)
        elif "search" in task.name.lower():
            return await self.participate_in_search(task.input_data)
        elif "consensus" in task.name.lower():
            return await self.participate_in_consensus(task.input_data)
        else:
            return await self.execute_swarm_behavior(task)
    
    async def participate_in_optimization(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Participate in swarm optimization (PSO-like behavior)"""
        objective_function = optimization_data.get("objective_function")
        search_space = optimization_data.get("search_space", [-10, 10])
        
        # Update position based on swarm dynamics
        await self.update_position_pso(search_space)
        
        # Evaluate fitness at current position
        current_fitness = await self.evaluate_fitness(objective_function)
        
        # Update personal best
        if current_fitness > self.swarm_state.local_best_fitness:
            self.swarm_state.local_best = self.swarm_state.position.copy()
            self.swarm_state.local_best_fitness = current_fitness
        
        return {
            "agent_id": self.agent_id,
            "position": self.swarm_state.position,
            "fitness": current_fitness,
            "local_best": self.swarm_state.local_best,
            "local_best_fitness": self.swarm_state.local_best_fitness
        }
    
    async def participate_in_search(self, search_data: Dict[str, Any]) -> Dict[str, Any]:
        """Participate in distributed search (ant colony-like behavior)"""
        search_target = search_data.get("target")
        search_area = search_data.get("area", [0, 100])
        
        # Simulate ant colony foraging behavior
        await self.update_position_aco(search_area)
        
        # Check if target found
        target_found = await self.check_target_proximity(search_target)
        
        if target_found:
            # Lay pheromone trail
            await self.lay_pheromone_trail()
        
        return {
            "agent_id": self.agent_id,
            "position": self.swarm_state.position,
            "target_found": target_found,
            "pheromone_strength": sum(self.swarm_state.pheromone_trails.values())
        }
    
    async def participate_in_consensus(self, consensus_data: Dict[str, Any]) -> Dict[str, Any]:
        """Participate in swarm consensus building"""
        proposal = consensus_data.get("proposal")
        voting_options = consensus_data.get("options", [])
        
        # Simple consensus mechanism
        my_vote = await self.evaluate_proposal(proposal, voting_options)
        
        # Influence from neighbors (simplified)
        neighbor_influence = await self.get_neighbor_influence()
        
        # Adjust vote based on neighbor influence
        final_vote = self.adjust_vote_with_influence(my_vote, neighbor_influence)
        
        return {
            "agent_id": self.agent_id,
            "initial_vote": my_vote,
            "final_vote": final_vote,
            "neighbor_influence": neighbor_influence
        }
    
    async def update_position_pso(self, search_space: List[float]):
        """Update position using Particle Swarm Optimization dynamics"""
        # PSO velocity update (simplified)
        inertia = 0.7
        cognitive = 1.5
        social = 1.5
        
        # Random factors
        r1, r2 = 0.5, 0.5  # Simplified random values
        
        # Assume global best is available (would come from swarm coordinator)
        global_best = [5.0, 5.0]  # Placeholder
        
        for i in range(len(self.swarm_state.velocity)):
            self.swarm_state.velocity[i] = (
                inertia * self.swarm_state.velocity[i] +
                cognitive * r1 * (self.swarm_state.local_best[i] - self.swarm_state.position[i]) +
                social * r2 * (global_best[i] - self.swarm_state.position[i])
            )
            
            # Update position
            self.swarm_state.position[i] += self.swarm_state.velocity[i]
            
            # Boundary constraints
            self.swarm_state.position[i] = max(search_space[0], 
                                             min(search_space[1], self.swarm_state.position[i]))
    
    async def update_position_aco(self, search_area: List[float]):
        """Update position using Ant Colony Optimization dynamics"""
        # Follow pheromone trails with some randomness
        if self.swarm_state.pheromone_trails:
            # Move towards strongest pheromone trail
            strongest_trail = max(self.swarm_state.pheromone_trails.items(), 
                                key=lambda x: x[1])
            # Simplified movement towards trail
            direction = [0.1, 0.1]  # Placeholder direction
        else:
            # Random exploration
            direction = [0.2 * (0.5 - 0.5), 0.2 * (0.5 - 0.5)]  # Random direction
        
        # Update position
        for i in range(len(self.swarm_state.position)):
            self.swarm_state.position[i] += direction[i]
            # Keep within search area
            self.swarm_state.position[i] = max(search_area[0], 
                                             min(search_area[1], self.swarm_state.position[i]))
    
    async def evaluate_fitness(self, objective_function: str) -> float:
        """Evaluate fitness at current position"""
        # Simplified fitness evaluation
        x, y = self.swarm_state.position[0], self.swarm_state.position[1]
        
        if objective_function == "sphere":
            fitness = -(x**2 + y**2)  # Negative because we want to minimize
        elif objective_function == "rastrigin":
            fitness = -(20 + x**2 - 10*1 + y**2 - 10*1)  # Simplified Rastrigin
        else:
            fitness = -(abs(x) + abs(y))  # Default: minimize distance from origin
        
        self.swarm_state.fitness = fitness
        return fitness
    
    async def check_target_proximity(self, target: Dict[str, Any]) -> bool:
        """Check if agent is near the search target"""
        if not target:
            return False
        
        target_pos = target.get("position", [0, 0])
        threshold = target.get("threshold", 1.0)
        
        # Calculate distance to target
        distance = sum((self.swarm_state.position[i] - target_pos[i])**2 
                      for i in range(len(target_pos)))**0.5
        
        return distance <= threshold
    
    async def lay_pheromone_trail(self):
        """Lay pheromone trail at current position"""
        position_key = f"{self.swarm_state.position[0]:.1f},{self.swarm_state.position[1]:.1f}"
        current_strength = self.swarm_state.pheromone_trails.get(position_key, 0.0)
        self.swarm_state.pheromone_trails[position_key] = min(current_strength + 1.0, 10.0)
    
    async def evaluate_proposal(self, proposal: Dict[str, Any], 
                              options: List[str]) -> str:
        """Evaluate a proposal and return vote"""
        # Simplified proposal evaluation
        if not options:
            return "abstain"
        
        # Random choice for demonstration (would be more sophisticated)
        import random
        return random.choice(options)
    
    async def get_neighbor_influence(self) -> Dict[str, Any]:
        """Get influence from neighboring agents"""
        # Simplified neighbor influence
        return {
            "neighbor_count": len(self.swarm_state.neighbors),
            "average_opinion": 0.5,  # Placeholder
            "consensus_strength": 0.7
        }
    
    def adjust_vote_with_influence(self, my_vote: str, 
                                 neighbor_influence: Dict[str, Any]) -> str:
        """Adjust vote based on neighbor influence"""
        consensus_strength = neighbor_influence.get("consensus_strength", 0.0)
        
        # If strong consensus among neighbors, consider changing vote
        if consensus_strength > 0.8:
            # Simplified influence mechanism
            return my_vote  # For now, stick with original vote
        
        return my_vote
    
    async def execute_swarm_behavior(self, task: Task) -> Dict[str, Any]:
        """Execute general swarm behavior"""
        behavior_result = {
            "agent_id": self.agent_id,
            "swarm_id": self.swarm_id,
            "behavior_mode": self.behavior_mode.value,
            "position": self.swarm_state.position,
            "fitness": self.swarm_state.fitness,
            "neighbors": list(self.swarm_state.neighbors)
        }
        
        return behavior_result

class SwarmCoordinator:
    """Coordinates swarm agents and manages collective behavior"""
    
    def __init__(self, swarm_id: str, swarm_size: int):
        self.swarm_id = swarm_id
        self.swarm_size = swarm_size
        self.agents: Dict[str, SwarmAgent] = {}
        self.global_best_position: List[float] = [0.0, 0.0]
        self.global_best_fitness: float = float('-inf')
        self.convergence_threshold = 0.01
        self.max_iterations = 1000
        self.current_iteration = 0
    
    def add_agent(self, agent: SwarmAgent):
        """Add an agent to the swarm"""
        self.agents[agent.agent_id] = agent
        self.update_neighbor_relationships()
    
    def update_neighbor_relationships(self):
        """Update neighbor relationships based on communication range"""
        agent_list = list(self.agents.values())
        
        for i, agent1 in enumerate(agent_list):
            agent1.swarm_state.neighbors.clear()
            
            for j, agent2 in enumerate(agent_list):
                if i != j:
                    distance = self.calculate_distance(
                        agent1.swarm_state.position,
                        agent2.swarm_state.position
                    )
                    
                    if distance <= agent1.communication_range:
                        agent1.swarm_state.neighbors.add(agent2.agent_id)
    
    def calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return sum((pos1[i] - pos2[i])**2 for i in range(len(pos1)))**0.5
    
    async def coordinate_optimization(self, objective_function: str, 
                                   search_space: List[float]) -> Dict[str, Any]:
        """Coordinate swarm optimization process"""
        optimization_data = {
            "objective_function": objective_function,
            "search_space": search_space
        }
        
        iteration_results = []
        
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            
            # Execute optimization step for all agents
            agent_results = []
            for agent in self.agents.values():
                result = await agent.participate_in_optimization(optimization_data)
                agent_results.append(result)
                
                # Update global best
                if result["fitness"] > self.global_best_fitness:
                    self.global_best_fitness = result["fitness"]
                    self.global_best_position = result["position"].copy()
            
            # Update neighbor relationships
            self.update_neighbor_relationships()
            
            # Check convergence
            if self.check_convergence():
                break
            
            iteration_results.append({
                "iteration": iteration,
                "global_best_fitness": self.global_best_fitness,
                "global_best_position": self.global_best_position.copy(),
                "agent_results": agent_results
            })
        
        return {
            "swarm_id": self.swarm_id,
            "final_best_position": self.global_best_position,
            "final_best_fitness": self.global_best_fitness,
            "iterations_completed": self.current_iteration + 1,
            "converged": self.check_convergence(),
            "iteration_history": iteration_results[-10:]  # Last 10 iterations
        }
    
    def check_convergence(self) -> bool:
        """Check if swarm has converged"""
        if len(self.agents) < 2:
            return True
        
        # Calculate diversity (spread of positions)
        positions = [agent.swarm_state.position for agent in self.agents.values()]
        
        # Calculate average position
        avg_position = [sum(pos[i] for pos in positions) / len(positions) 
                       for i in range(len(positions[0]))]
        
        # Calculate average distance from center
        avg_distance = sum(self.calculate_distance(pos, avg_position) 
                          for pos in positions) / len(positions)
        
        return avg_distance < self.convergence_threshold
    
    async def coordinate_consensus(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate consensus building among swarm agents"""
        consensus_data = {
            "proposal": proposal,
            "options": proposal.get("voting_options", ["yes", "no"])
        }
        
        # Collect votes from all agents
        votes = []
        for agent in self.agents.values():
            vote_result = await agent.participate_in_consensus(consensus_data)
            votes.append(vote_result)
        
        # Analyze consensus
        final_votes = [vote["final_vote"] for vote in votes]
        vote_counts = {}
        for vote in final_votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Determine consensus
        total_votes = len(final_votes)
        consensus_threshold = 0.7  # 70% agreement
        
        consensus_reached = False
        consensus_decision = None
        
        for option, count in vote_counts.items():
            if count / total_votes >= consensus_threshold:
                consensus_reached = True
                consensus_decision = option
                break
        
        return {
            "swarm_id": self.swarm_id,
            "proposal": proposal,
            "vote_counts": vote_counts,
            "total_votes": total_votes,
            "consensus_reached": consensus_reached,
            "consensus_decision": consensus_decision,
            "consensus_strength": max(vote_counts.values()) / total_votes if vote_counts else 0
         }
```

## 4. Hybrid and Adaptive Architectures

### 4.1 Hybrid Agent Systems

```python
class ArchitectureType(Enum):
    """Types of agent architectures"""
    HIERARCHICAL = "hierarchical"
    SWARM = "swarm"
    FEDERATED = "federated"
    MARKET = "market"
    HYBRID = "hybrid"

@dataclass
class ArchitectureComponent:
    """Component of a hybrid architecture"""
    component_type: ArchitectureType
    agents: Set[str]
    coordinator: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class HybridAgentSystem:
    """Manages hybrid agent architectures that combine multiple patterns"""
    
    def __init__(self, system_id: str, name: str):
        self.system_id = system_id
        self.name = name
        self.components: Dict[str, ArchitectureComponent] = {}
        self.agents: Dict[str, BaseAgent] = {}
        self.adaptation_rules: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.current_configuration: Dict[str, Any] = {}
        
        # Architecture coordinators
        self.hierarchical_system: Optional[HierarchicalAgentSystem] = None
        self.swarm_coordinators: Dict[str, SwarmCoordinator] = {}
        self.market_system: Optional[AgentMarketplace] = None
    
    def add_component(self, component_id: str, component: ArchitectureComponent):
        """Add an architecture component to the hybrid system"""
        self.components[component_id] = component
        
        # Initialize appropriate coordinator
        if component.component_type == ArchitectureType.HIERARCHICAL:
            if not self.hierarchical_system:
                self.hierarchical_system = HierarchicalAgentSystem(f"{self.name}_hierarchy")
        
        elif component.component_type == ArchitectureType.SWARM:
            swarm_coordinator = SwarmCoordinator(component_id, len(component.agents))
            self.swarm_coordinators[component_id] = swarm_coordinator
        
        elif component.component_type == ArchitectureType.MARKET:
            if not self.market_system:
                self.market_system = AgentMarketplace()
    
    def add_agent(self, agent: BaseAgent, component_id: str):
        """Add an agent to a specific component"""
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not found")
        
        self.agents[agent.agent_id] = agent
        self.components[component_id].agents.add(agent.agent_id)
        
        # Register agent with appropriate coordinator
        component = self.components[component_id]
        
        if component.component_type == ArchitectureType.HIERARCHICAL and self.hierarchical_system:
            # Add to hierarchy (simplified - would need proper parent assignment)
            if not self.hierarchical_system.root_node:
                self.hierarchical_system.set_root_agent(agent, {"all"})
        
        elif component.component_type == ArchitectureType.SWARM:
            if isinstance(agent, SwarmAgent) and component_id in self.swarm_coordinators:
                self.swarm_coordinators[component_id].add_agent(agent)
        
        elif component.component_type == ArchitectureType.MARKET and self.market_system:
            self.market_system.register_agent(agent)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task using the most appropriate architecture component"""
        # Determine best component for the task
        best_component = await self.select_optimal_component(task)
        
        if not best_component:
            return {"error": "No suitable component found for task"}
        
        # Execute task using the selected component
        result = await self.execute_with_component(task, best_component)
        
        # Record performance for adaptation
        await self.record_performance(task, best_component, result)
        
        return result
    
    async def select_optimal_component(self, task: Task) -> Optional[str]:
        """Select the optimal architecture component for a task"""
        component_scores = {}
        
        for component_id, component in self.components.items():
            score = await self.calculate_component_suitability(task, component)
            component_scores[component_id] = score
        
        if not component_scores:
            return None
        
        # Return component with highest score
        best_component = max(component_scores.items(), key=lambda x: x[1])
        return best_component[0] if best_component[1] > 0 else None
    
    async def calculate_component_suitability(self, task: Task, 
                                            component: ArchitectureComponent) -> float:
        """Calculate how suitable a component is for a task"""
        base_score = 0.0
        
        # Task type suitability
        if "optimization" in task.name.lower() and component.component_type == ArchitectureType.SWARM:
            base_score += 0.8
        elif "strategic" in task.name.lower() and component.component_type == ArchitectureType.HIERARCHICAL:
            base_score += 0.8
        elif "negotiation" in task.name.lower() and component.component_type == ArchitectureType.MARKET:
            base_score += 0.8
        else:
            base_score += 0.3  # Default suitability
        
        # Agent availability
        available_agents = 0
        for agent_id in component.agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if await agent.can_handle_task(task):
                    available_agents += 1
        
        availability_score = min(available_agents / max(len(task.required_capabilities), 1), 1.0)
        
        # Performance history
        performance_score = component.performance_metrics.get("average_success_rate", 0.5)
        
        # Load factor
        total_load = sum(self.agents[agent_id].calculate_load_factor() 
                        for agent_id in component.agents if agent_id in self.agents)
        avg_load = total_load / len(component.agents) if component.agents else 1.0
        load_score = 1.0 - avg_load
        
        # Weighted combination
        final_score = (
            base_score * 0.4 +
            availability_score * 0.3 +
            performance_score * 0.2 +
            load_score * 0.1
        )
        
        return final_score
    
    async def execute_with_component(self, task: Task, component_id: str) -> Dict[str, Any]:
        """Execute a task using a specific component"""
        component = self.components[component_id]
        
        if component.component_type == ArchitectureType.HIERARCHICAL:
            return await self.execute_hierarchical(task, component)
        elif component.component_type == ArchitectureType.SWARM:
            return await self.execute_swarm(task, component_id)
        elif component.component_type == ArchitectureType.MARKET:
            return await self.execute_market(task, component)
        else:
            return await self.execute_direct(task, component)
    
    async def execute_hierarchical(self, task: Task, component: ArchitectureComponent) -> Dict[str, Any]:
        """Execute task using hierarchical coordination"""
        if not self.hierarchical_system or not self.hierarchical_system.root_node:
            return {"error": "Hierarchical system not properly initialized"}
        
        # Delegate task through hierarchy
        assigned_agent_id = await self.hierarchical_system.delegate_task(
            task, self.hierarchical_system.root_node.agent.agent_id
        )
        
        if assigned_agent_id and assigned_agent_id in self.agents:
            result = await self.agents[assigned_agent_id].execute_task(task)
            result["execution_mode"] = "hierarchical"
            result["assigned_agent"] = assigned_agent_id
            return result
        
        return {"error": "Failed to delegate task in hierarchy"}
    
    async def execute_swarm(self, task: Task, component_id: str) -> Dict[str, Any]:
        """Execute task using swarm coordination"""
        if component_id not in self.swarm_coordinators:
            return {"error": "Swarm coordinator not found"}
        
        coordinator = self.swarm_coordinators[component_id]
        
        if "optimization" in task.name.lower():
            result = await coordinator.coordinate_optimization(
                task.input_data.get("objective_function", "sphere"),
                task.input_data.get("search_space", [-10, 10])
            )
        elif "consensus" in task.name.lower():
            result = await coordinator.coordinate_consensus(task.input_data)
        else:
            # Execute with individual swarm agents
            agent_results = []
            for agent_id in coordinator.agents:
                if agent_id in self.agents:
                    agent_result = await self.agents[agent_id].execute_task(task)
                    agent_results.append(agent_result)
            
            result = {
                "execution_mode": "swarm",
                "agent_results": agent_results,
                "swarm_id": component_id
            }
        
        result["execution_mode"] = "swarm"
        return result
    
    async def execute_market(self, task: Task, component: ArchitectureComponent) -> Dict[str, Any]:
        """Execute task using market-based coordination"""
        if not self.market_system:
            return {"error": "Market system not initialized"}
        
        # Create auction for the task
        auction_result = await self.market_system.create_auction(
            task.name,
            task.required_capabilities,
            {"budget": 100.0, "deadline": task.deadline}
        )
        
        if auction_result.get("winner"):
            winner_id = auction_result["winner"]
            if winner_id in self.agents:
                result = await self.agents[winner_id].execute_task(task)
                result["execution_mode"] = "market"
                result["auction_info"] = auction_result
                return result
        
        return {"error": "No winner in market auction"}
    
    async def execute_direct(self, task: Task, component: ArchitectureComponent) -> Dict[str, Any]:
        """Execute task using direct agent assignment"""
        # Find best agent in component
        best_agent = None
        best_score = 0.0
        
        for agent_id in component.agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if await agent.can_handle_task(task):
                    metrics = agent.get_performance_metrics()
                    score = metrics["success_rate"] * (1 - metrics["current_load"])
                    if score > best_score:
                        best_score = score
                        best_agent = agent
        
        if best_agent:
            result = await best_agent.execute_task(task)
            result["execution_mode"] = "direct"
            result["selected_agent"] = best_agent.agent_id
            return result
        
        return {"error": "No suitable agent found in component"}
    
    async def record_performance(self, task: Task, component_id: str, result: Dict[str, Any]):
        """Record performance metrics for adaptation"""
        success = "error" not in result
        execution_time = result.get("execution_time", 0.0)
        
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task.name,
            "component_id": component_id,
            "success": success,
            "execution_time": execution_time,
            "task_complexity": len(task.required_capabilities)
        }
        
        self.performance_history.append(performance_record)
        
        # Update component metrics
        component = self.components[component_id]
        recent_records = [r for r in self.performance_history[-100:] 
                         if r["component_id"] == component_id]
        
        if recent_records:
            success_rate = sum(1 for r in recent_records if r["success"]) / len(recent_records)
            avg_time = sum(r["execution_time"] for r in recent_records) / len(recent_records)
            
            component.performance_metrics.update({
                "average_success_rate": success_rate,
                "average_execution_time": avg_time,
                "total_tasks": len(recent_records)
            })
        
        # Trigger adaptation if needed
        await self.check_adaptation_triggers()
    
    async def check_adaptation_triggers(self):
        """Check if system adaptation is needed"""
        for rule in self.adaptation_rules:
            if await self.evaluate_adaptation_rule(rule):
                await self.apply_adaptation(rule)
    
    async def evaluate_adaptation_rule(self, rule: Dict[str, Any]) -> bool:
        """Evaluate if an adaptation rule should trigger"""
        condition = rule.get("condition")
        threshold = rule.get("threshold", 0.5)
        
        if condition == "low_success_rate":
            recent_records = self.performance_history[-50:]  # Last 50 tasks
            if recent_records:
                success_rate = sum(1 for r in recent_records if r["success"]) / len(recent_records)
                return success_rate < threshold
        
        elif condition == "high_load":
            total_load = sum(agent.calculate_load_factor() for agent in self.agents.values())
            avg_load = total_load / len(self.agents) if self.agents else 0
            return avg_load > threshold
        
        elif condition == "component_imbalance":
            component_usage = {}
            for record in self.performance_history[-100:]:
                comp_id = record["component_id"]
                component_usage[comp_id] = component_usage.get(comp_id, 0) + 1
            
            if len(component_usage) > 1:
                max_usage = max(component_usage.values())
                min_usage = min(component_usage.values())
                imbalance_ratio = max_usage / max(min_usage, 1)
                return imbalance_ratio > threshold
        
        return False
    
    async def apply_adaptation(self, rule: Dict[str, Any]):
        """Apply an adaptation based on a triggered rule"""
        adaptation_type = rule.get("adaptation")
        
        if adaptation_type == "rebalance_components":
            await self.rebalance_components()
        elif adaptation_type == "adjust_thresholds":
            await self.adjust_selection_thresholds()
        elif adaptation_type == "add_agents":
            await self.suggest_agent_addition()
    
    async def rebalance_components(self):
        """Rebalance agents across components"""
        # Simplified rebalancing logic
        print(f"Rebalancing components in {self.name}")
        
        # Calculate component loads
        component_loads = {}
        for comp_id, component in self.components.items():
            total_load = sum(self.agents[agent_id].calculate_load_factor() 
                           for agent_id in component.agents if agent_id in self.agents)
            component_loads[comp_id] = total_load / len(component.agents) if component.agents else 0
        
        # Move agents from overloaded to underloaded components
        # (Implementation would depend on specific requirements)
    
    async def adjust_selection_thresholds(self):
        """Adjust component selection thresholds based on performance"""
        print(f"Adjusting selection thresholds in {self.name}")
        
        # Analyze performance patterns and adjust selection criteria
        # (Implementation would involve machine learning or heuristic adjustments)
    
    async def suggest_agent_addition(self):
        """Suggest adding new agents to improve performance"""
        print(f"Suggesting agent addition for {self.name}")
        
        # Analyze bottlenecks and suggest new agent types
        # (Implementation would involve capacity planning algorithms)
    
    def add_adaptation_rule(self, condition: str, threshold: float, adaptation: str):
        """Add an adaptation rule to the system"""
        rule = {
            "condition": condition,
            "threshold": threshold,
            "adaptation": adaptation
        }
        self.adaptation_rules.append(rule)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        component_status = {}
        for comp_id, component in self.components.items():
            component_status[comp_id] = {
                "type": component.component_type.value,
                "agent_count": len(component.agents),
                "performance_metrics": component.performance_metrics,
                "configuration": component.configuration
            }
        
        recent_performance = self.performance_history[-20:] if self.performance_history else []
        overall_success_rate = (sum(1 for r in recent_performance if r["success"]) / 
                              len(recent_performance)) if recent_performance else 0.0
        
        return {
            "system_id": self.system_id,
            "name": self.name,
            "total_agents": len(self.agents),
            "components": component_status,
            "overall_success_rate": overall_success_rate,
            "adaptation_rules": len(self.adaptation_rules),
            "performance_history_size": len(self.performance_history)
        }

class AdaptiveAgent(BaseAgent):
    """Agent that can adapt its behavior and capabilities"""
    
    def __init__(self, agent_id: str, name: str, role: AgentRole):
        super().__init__(agent_id, name, role)
        self.adaptation_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.3
        self.behavior_patterns: Dict[str, Any] = {}
        self.capability_weights: Dict[str, float] = {}
        
        # Add adaptive capabilities
        self.add_capability(AgentCapability(
            name="self_adaptation",
            description="Adapt behavior based on performance feedback",
            input_schema={"feedback": "object", "context": "object"},
            output_schema={"adaptation_result": "object"}
        ))
        
        self.add_capability(AgentCapability(
            name="capability_learning",
            description="Learn and improve capabilities over time",
            input_schema={"experience": "object", "outcomes": "array"},
            output_schema={"learning_result": "object"}
        ))
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute task with adaptive behavior"""
        # Record task attempt
        start_time = datetime.now()
        
        # Apply current behavioral adaptations
        adapted_approach = await self.adapt_approach_for_task(task)
        
        # Execute with adapted approach
        try:
            if adapted_approach.get("use_collaboration"):
                result = await self.execute_collaborative_task(task)
            elif adapted_approach.get("use_decomposition"):
                result = await self.execute_decomposed_task(task)
            else:
                result = await super().execute_task(task)
            
            success = True
        except Exception as e:
            result = {"error": str(e)}
            success = False
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Record performance for learning
        performance_record = {
            "task_type": task.name,
            "approach": adapted_approach,
            "success": success,
            "execution_time": execution_time,
            "timestamp": start_time.isoformat()
        }
        
        await self.learn_from_performance(performance_record)
        
        result["execution_time"] = execution_time
        result["adaptive_approach"] = adapted_approach
        
        return result
    
    async def adapt_approach_for_task(self, task: Task) -> Dict[str, Any]:
        """Adapt approach based on task characteristics and past performance"""
        task_signature = self.get_task_signature(task)
        
        # Check if we have experience with similar tasks
        similar_experiences = [record for record in self.adaptation_history 
                             if record.get("task_signature") == task_signature]
        
        if similar_experiences:
            # Use successful patterns from past experiences
            successful_approaches = [record["approach"] for record in similar_experiences 
                                   if record["success"]]
            
            if successful_approaches:
                # Use the most recent successful approach
                return successful_approaches[-1]
        
        # Default adaptive approach based on task characteristics
        approach = {
            "use_collaboration": len(task.required_capabilities) > 2,
            "use_decomposition": task.priority > 7,
            "timeout_multiplier": 1.0,
            "retry_count": 1
        }
        
        return approach
    
    def get_task_signature(self, task: Task) -> str:
        """Generate a signature for task type classification"""
        capabilities_str = ",".join(sorted(task.required_capabilities))
        priority_level = "high" if task.priority > 7 else "medium" if task.priority > 4 else "low"
        return f"{task.name}:{capabilities_str}:{priority_level}"
    
    async def execute_collaborative_task(self, task: Task) -> Dict[str, Any]:
        """Execute task using collaborative approach"""
        # Simplified collaborative execution
        # In practice, this would involve finding and coordinating with other agents
        
        collaboration_result = {
            "approach": "collaborative",
            "collaborators": [],  # Would be populated with actual collaborators
            "coordination_overhead": 0.1,
            "result": f"Collaborative execution of {task.name}"
        }
        
        return collaboration_result
    
    async def execute_decomposed_task(self, task: Task) -> Dict[str, Any]:
        """Execute task using decomposition approach"""
        # Simplified task decomposition
        subtasks = self.decompose_task(task)
        
        subtask_results = []
        for subtask in subtasks:
            # Execute each subtask
            subtask_result = await super().execute_task(subtask)
            subtask_results.append(subtask_result)
        
        # Combine results
        combined_result = {
            "approach": "decomposed",
            "subtask_count": len(subtasks),
            "subtask_results": subtask_results,
            "result": f"Decomposed execution of {task.name}"
        }
        
        return combined_result
    
    def decompose_task(self, task: Task) -> List[Task]:
        """Decompose a complex task into simpler subtasks"""
        subtasks = []
        
        # Simple decomposition based on required capabilities
        for i, capability in enumerate(task.required_capabilities):
            subtask = Task(
                task_id=f"{task.task_id}_sub_{i}",
                name=f"{task.name}_subtask_{i}",
                description=f"Subtask for {capability}",
                required_capabilities=[capability],
                input_data=task.input_data,
                priority=task.priority,
                deadline=task.deadline
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def learn_from_performance(self, performance_record: Dict[str, Any]):
        """Learn from task execution performance"""
        task_signature = performance_record.get("task_signature", 
                                               self.get_task_signature_from_record(performance_record))
        
        # Add task signature to record
        performance_record["task_signature"] = task_signature
        
        # Store in adaptation history
        self.adaptation_history.append(performance_record)
        
        # Update behavior patterns
        await self.update_behavior_patterns(performance_record)
        
        # Update capability weights
        await self.update_capability_weights(performance_record)
        
        # Trigger adaptation if performance is below threshold
        if not performance_record["success"]:
            await self.trigger_adaptation(performance_record)
    
    def get_task_signature_from_record(self, record: Dict[str, Any]) -> str:
        """Extract task signature from performance record"""
        # Simplified signature extraction
        return record.get("task_type", "unknown")
    
    async def update_behavior_patterns(self, performance_record: Dict[str, Any]):
        """Update learned behavior patterns"""
        task_signature = performance_record["task_signature"]
        approach = performance_record["approach"]
        success = performance_record["success"]
        
        if task_signature not in self.behavior_patterns:
            self.behavior_patterns[task_signature] = {
                "successful_approaches": [],
                "failed_approaches": [],
                "success_rate": 0.0,
                "total_attempts": 0
            }
        
        pattern = self.behavior_patterns[task_signature]
        pattern["total_attempts"] += 1
        
        if success:
            pattern["successful_approaches"].append(approach)
        else:
            pattern["failed_approaches"].append(approach)
        
        # Update success rate
        successful_attempts = len(pattern["successful_approaches"])
        pattern["success_rate"] = successful_attempts / pattern["total_attempts"]
    
    async def update_capability_weights(self, performance_record: Dict[str, Any]):
        """Update weights for different capabilities based on performance"""
        # Simplified capability weight updating
        # In practice, this would involve more sophisticated learning algorithms
        
        task_type = performance_record["task_type"]
        success = performance_record["success"]
        
        if task_type not in self.capability_weights:
            self.capability_weights[task_type] = 1.0
        
        # Adjust weight based on success/failure
        if success:
            self.capability_weights[task_type] += self.learning_rate * 0.1
        else:
            self.capability_weights[task_type] -= self.learning_rate * 0.1
        
        # Keep weights in reasonable bounds
        self.capability_weights[task_type] = max(0.1, min(2.0, self.capability_weights[task_type]))
    
    async def trigger_adaptation(self, performance_record: Dict[str, Any]):
        """Trigger behavioral adaptation after failure"""
        task_signature = performance_record["task_signature"]
        failed_approach = performance_record["approach"]
        
        # Analyze what went wrong and adapt
        adaptation = {
            "timestamp": datetime.now().isoformat(),
            "trigger": "task_failure",
            "task_signature": task_signature,
            "failed_approach": failed_approach,
            "adaptation_actions": []
        }
        
        # Determine adaptation actions
        if failed_approach.get("use_collaboration"):
            adaptation["adaptation_actions"].append("reduce_collaboration_threshold")
        
        if failed_approach.get("use_decomposition"):
            adaptation["adaptation_actions"].append("improve_decomposition_strategy")
        
        adaptation["adaptation_actions"].append("increase_timeout_multiplier")
        
        # Apply adaptations
        await self.apply_adaptations(adaptation)
    
    async def apply_adaptations(self, adaptation: Dict[str, Any]):
        """Apply behavioral adaptations"""
        for action in adaptation["adaptation_actions"]:
            if action == "reduce_collaboration_threshold":
                # Reduce tendency to collaborate for this task type
                pass
            elif action == "improve_decomposition_strategy":
                # Improve task decomposition approach
                pass
            elif action == "increase_timeout_multiplier":
                # Increase timeout for similar tasks
                pass
        
        print(f"Applied adaptations: {adaptation['adaptation_actions']}")
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of agent's adaptations and learning"""
        total_tasks = len(self.adaptation_history)
        successful_tasks = sum(1 for record in self.adaptation_history if record["success"])
        
        return {
            "agent_id": self.agent_id,
            "total_tasks_executed": total_tasks,
            "overall_success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "learned_patterns": len(self.behavior_patterns),
            "capability_weights": self.capability_weights.copy(),
            "recent_adaptations": self.adaptation_history[-5:] if self.adaptation_history else []
         }
```

### 4.2 Market-Based Agent Systems

```python
class BidType(Enum):
    """Types of bids in agent marketplace"""
    FIXED_PRICE = "fixed_price"
    AUCTION = "auction"
    NEGOTIATION = "negotiation"
    SUBSCRIPTION = "subscription"

@dataclass
class AgentBid:
    """Represents a bid from an agent for a task"""
    agent_id: str
    task_id: str
    bid_amount: float
    estimated_completion_time: float
    confidence_score: float
    bid_type: BidType
    conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class MarketTransaction:
    """Represents a completed transaction in the marketplace"""
    transaction_id: str
    task_id: str
    buyer_id: str
    seller_id: str
    agreed_price: float
    completion_time: float
    quality_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class AgentMarketplace:
    """Market-based coordination system for agents"""
    
    def __init__(self, marketplace_id: str = "default_market"):
        self.marketplace_id = marketplace_id
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.active_auctions: Dict[str, Dict[str, Any]] = {}
        self.pending_bids: Dict[str, List[AgentBid]] = {}
        self.completed_transactions: List[MarketTransaction] = []
        self.agent_ratings: Dict[str, Dict[str, float]] = {}
        self.market_fees = 0.05  # 5% transaction fee
        
        # Market statistics
        self.total_volume = 0.0
        self.average_completion_time = 0.0
        self.market_efficiency = 0.0
    
    def register_agent(self, agent: BaseAgent, initial_balance: float = 1000.0):
        """Register an agent in the marketplace"""
        self.registered_agents[agent.agent_id] = agent
        
        # Initialize agent rating and balance
        self.agent_ratings[agent.agent_id] = {
            "reliability": 5.0,  # Out of 10
            "quality": 5.0,
            "speed": 5.0,
            "communication": 5.0,
            "total_transactions": 0,
            "balance": initial_balance
        }
    
    async def create_auction(self, task_name: str, required_capabilities: List[str], 
                           auction_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an auction for a task"""
        auction_id = f"auction_{len(self.active_auctions)}_{int(datetime.now().timestamp())}"
        
        auction = {
            "auction_id": auction_id,
            "task_name": task_name,
            "required_capabilities": required_capabilities,
            "budget": auction_params.get("budget", 100.0),
            "deadline": auction_params.get("deadline", datetime.now() + timedelta(hours=24)),
            "auction_type": auction_params.get("type", "sealed_bid"),
            "minimum_rating": auction_params.get("minimum_rating", 3.0),
            "created_at": datetime.now(),
            "status": "active",
            "bids": []
        }
        
        self.active_auctions[auction_id] = auction
        self.pending_bids[auction_id] = []
        
        # Notify eligible agents
        await self.notify_eligible_agents(auction)
        
        return {"auction_id": auction_id, "status": "created"}
    
    async def notify_eligible_agents(self, auction: Dict[str, Any]):
        """Notify agents that are eligible for an auction"""
        required_caps = set(auction["required_capabilities"])
        minimum_rating = auction["minimum_rating"]
        
        for agent_id, agent in self.registered_agents.items():
            # Check if agent has required capabilities
            agent_caps = set(cap.name for cap in agent.capabilities)
            if required_caps.issubset(agent_caps):
                # Check agent rating
                rating = self.agent_ratings[agent_id]
                avg_rating = (rating["reliability"] + rating["quality"] + 
                            rating["speed"] + rating["communication"]) / 4
                
                if avg_rating >= minimum_rating:
                    # Notify agent about auction (simplified)
                    print(f"Notifying agent {agent_id} about auction {auction['auction_id']}")
    
    async def submit_bid(self, agent_id: str, auction_id: str, bid: AgentBid) -> Dict[str, Any]:
        """Submit a bid for an auction"""
        if auction_id not in self.active_auctions:
            return {"error": "Auction not found"}
        
        if agent_id not in self.registered_agents:
            return {"error": "Agent not registered"}
        
        auction = self.active_auctions[auction_id]
        
        # Validate bid
        if bid.bid_amount > auction["budget"]:
            return {"error": "Bid exceeds budget"}
        
        if datetime.now() > auction["deadline"]:
            return {"error": "Auction has expired"}
        
        # Add bid to auction
        self.pending_bids[auction_id].append(bid)
        auction["bids"].append({
            "agent_id": agent_id,
            "amount": bid.bid_amount,
            "completion_time": bid.estimated_completion_time,
            "confidence": bid.confidence_score
        })
        
        return {"status": "bid_submitted", "bid_id": f"{auction_id}_{agent_id}"}
    
    async def evaluate_bids(self, auction_id: str) -> Dict[str, Any]:
        """Evaluate bids and select winner"""
        if auction_id not in self.active_auctions:
            return {"error": "Auction not found"}
        
        auction = self.active_auctions[auction_id]
        bids = self.pending_bids[auction_id]
        
        if not bids:
            return {"error": "No bids received"}
        
        # Score each bid based on multiple criteria
        scored_bids = []
        for bid in bids:
            agent_rating = self.agent_ratings[bid.agent_id]
            
            # Calculate composite score
            price_score = (auction["budget"] - bid.bid_amount) / auction["budget"]  # Lower price is better
            time_score = max(0, 1 - bid.estimated_completion_time / 24)  # Faster is better
            quality_score = agent_rating["quality"] / 10
            reliability_score = agent_rating["reliability"] / 10
            confidence_score = bid.confidence_score
            
            composite_score = (
                price_score * 0.3 +
                time_score * 0.2 +
                quality_score * 0.2 +
                reliability_score * 0.2 +
                confidence_score * 0.1
            )
            
            scored_bids.append({
                "bid": bid,
                "score": composite_score,
                "breakdown": {
                    "price_score": price_score,
                    "time_score": time_score,
                    "quality_score": quality_score,
                    "reliability_score": reliability_score,
                    "confidence_score": confidence_score
                }
            })
        
        # Select winner (highest score)
        winner = max(scored_bids, key=lambda x: x["score"])
        
        # Update auction status
        auction["status"] = "completed"
        auction["winner"] = winner["bid"].agent_id
        auction["winning_bid"] = winner["bid"].bid_amount
        
        return {
            "winner": winner["bid"].agent_id,
            "winning_bid": winner["bid"].bid_amount,
            "score": winner["score"],
            "total_bids": len(bids)
        }
    
    async def execute_transaction(self, auction_id: str) -> Dict[str, Any]:
        """Execute the transaction for a completed auction"""
        if auction_id not in self.active_auctions:
            return {"error": "Auction not found"}
        
        auction = self.active_auctions[auction_id]
        
        if auction["status"] != "completed" or "winner" not in auction:
            return {"error": "Auction not ready for transaction"}
        
        winner_id = auction["winner"]
        winning_bid = auction["winning_bid"]
        
        # Create task for winner
        task = Task(
            task_id=f"market_task_{auction_id}",
            name=auction["task_name"],
            description=f"Market task from auction {auction_id}",
            required_capabilities=auction["required_capabilities"],
            input_data={"auction_id": auction_id, "budget": winning_bid},
            priority=5,
            deadline=auction["deadline"]
        )
        
        # Execute task
        start_time = datetime.now()
        try:
            result = await self.registered_agents[winner_id].execute_task(task)
            success = "error" not in result
        except Exception as e:
            result = {"error": str(e)}
            success = False
        
        completion_time = (datetime.now() - start_time).total_seconds() / 3600  # Hours
        
        # Calculate quality score (simplified)
        quality_score = 8.0 if success else 2.0
        
        # Process payment
        fee = winning_bid * self.market_fees
        net_payment = winning_bid - fee
        
        # Update agent balance
        self.agent_ratings[winner_id]["balance"] += net_payment
        
        # Record transaction
        transaction = MarketTransaction(
            transaction_id=f"txn_{auction_id}",
            task_id=task.task_id,
            buyer_id="marketplace",  # Simplified
            seller_id=winner_id,
            agreed_price=winning_bid,
            completion_time=completion_time,
            quality_score=quality_score
        )
        
        self.completed_transactions.append(transaction)
        
        # Update agent ratings
        await self.update_agent_rating(winner_id, transaction)
        
        # Update market statistics
        self.total_volume += winning_bid
        self.update_market_statistics()
        
        return {
            "transaction_id": transaction.transaction_id,
            "success": success,
            "payment": net_payment,
            "fee": fee,
            "quality_score": quality_score,
            "completion_time": completion_time
        }
    
    async def update_agent_rating(self, agent_id: str, transaction: MarketTransaction):
        """Update agent rating based on transaction performance"""
        rating = self.agent_ratings[agent_id]
        
        # Update reliability (based on successful completion)
        reliability_update = 0.5 if transaction.quality_score > 5 else -0.5
        rating["reliability"] = max(0, min(10, rating["reliability"] + reliability_update))
        
        # Update quality (based on quality score)
        quality_update = (transaction.quality_score - rating["quality"]) * 0.1
        rating["quality"] = max(0, min(10, rating["quality"] + quality_update))
        
        # Update speed (based on completion time vs estimate)
        # This would require storing the estimated time from the bid
        speed_update = 0.1 if transaction.completion_time < 2 else -0.1
        rating["speed"] = max(0, min(10, rating["speed"] + speed_update))
        
        # Increment transaction count
        rating["total_transactions"] += 1
    
    def update_market_statistics(self):
        """Update overall market statistics"""
        if self.completed_transactions:
            total_time = sum(t.completion_time for t in self.completed_transactions)
            self.average_completion_time = total_time / len(self.completed_transactions)
            
            successful_transactions = sum(1 for t in self.completed_transactions if t.quality_score > 5)
            self.market_efficiency = successful_transactions / len(self.completed_transactions)
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get comprehensive market status"""
        active_auctions = len([a for a in self.active_auctions.values() if a["status"] == "active"])
        
        return {
            "marketplace_id": self.marketplace_id,
            "registered_agents": len(self.registered_agents),
            "active_auctions": active_auctions,
            "completed_transactions": len(self.completed_transactions),
            "total_volume": self.total_volume,
            "average_completion_time": self.average_completion_time,
            "market_efficiency": self.market_efficiency,
            "top_agents": self.get_top_agents(5)
        }
    
    def get_top_agents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top-rated agents in the marketplace"""
        agent_scores = []
        
        for agent_id, rating in self.agent_ratings.items():
            if rating["total_transactions"] > 0:
                avg_rating = (rating["reliability"] + rating["quality"] + 
                            rating["speed"] + rating["communication"]) / 4
                agent_scores.append({
                    "agent_id": agent_id,
                    "average_rating": avg_rating,
                    "total_transactions": rating["total_transactions"],
                    "balance": rating["balance"]
                })
        
        # Sort by average rating and return top agents
        agent_scores.sort(key=lambda x: x["average_rating"], reverse=True)
        return agent_scores[:limit]
```

## 5. Practical Implementation Examples

### 5.1 Complete Advanced Architecture Setup

```python
async def setup_advanced_architecture_system():
    """Set up a comprehensive advanced agent architecture system"""
    
    # Initialize hybrid system
    hybrid_system = HybridAgentSystem("enterprise_system", "Enterprise Multi-Agent System")
    
    # Create architecture components
    hierarchical_component = ArchitectureComponent(
        component_type=ArchitectureType.HIERARCHICAL,
        agents=set(),
        configuration={"max_depth": 4, "span_of_control": 5}
    )
    
    swarm_component = ArchitectureComponent(
        component_type=ArchitectureType.SWARM,
        agents=set(),
        configuration={"swarm_size": 20, "optimization_algorithm": "PSO"}
    )
    
    market_component = ArchitectureComponent(
        component_type=ArchitectureType.MARKET,
        agents=set(),
        configuration={"auction_duration": 3600, "minimum_rating": 4.0}
    )
    
    # Add components to hybrid system
    hybrid_system.add_component("hierarchy", hierarchical_component)
    hybrid_system.add_component("swarm", swarm_component)
    hybrid_system.add_component("market", market_component)
    
    # Create agents
    agents = []
    
    # Hierarchical agents
    executive = ExecutiveAgent("exec_001", "Chief Executive Agent", AgentRole.COORDINATOR)
    manager1 = ManagerAgent("mgr_001", "Operations Manager", AgentRole.COORDINATOR)
    manager2 = ManagerAgent("mgr_002", "Strategy Manager", AgentRole.COORDINATOR)
    
    # Specialist agents
    data_specialist = SpecialistAgent("spec_001", "Data Analyst", AgentRole.SPECIALIST)
    ml_specialist = SpecialistAgent("spec_002", "ML Engineer", AgentRole.SPECIALIST)
    
    # Worker agents
    workers = [WorkerAgent(f"worker_{i:03d}", f"Worker {i}", AgentRole.WORKER) 
              for i in range(1, 11)]
    
    # Swarm agents
    swarm_agents = [SwarmAgent(f"swarm_{i:03d}", f"Swarm Agent {i}", AgentRole.WORKER)
                   for i in range(1, 21)]
    
    # Adaptive agents
    adaptive_agents = [AdaptiveAgent(f"adaptive_{i:03d}", f"Adaptive Agent {i}", AgentRole.SPECIALIST)
                      for i in range(1, 6)]
    
    agents.extend([executive, manager1, manager2, data_specialist, ml_specialist])
    agents.extend(workers)
    agents.extend(swarm_agents)
    agents.extend(adaptive_agents)
    
    # Add agents to appropriate components
    for agent in [executive, manager1, manager2, data_specialist, ml_specialist] + workers[:5]:
        hybrid_system.add_agent(agent, "hierarchy")
    
    for agent in swarm_agents:
        hybrid_system.add_agent(agent, "swarm")
    
    for agent in workers[5:] + adaptive_agents:
        hybrid_system.add_agent(agent, "market")
    
    # Add adaptation rules
    hybrid_system.add_adaptation_rule("low_success_rate", 0.7, "rebalance_components")
    hybrid_system.add_adaptation_rule("high_load", 0.8, "add_agents")
    hybrid_system.add_adaptation_rule("component_imbalance", 3.0, "adjust_thresholds")
    
    return hybrid_system, agents

async def demonstrate_advanced_architectures():
    """Demonstrate advanced agent architectures in action"""
    
    print("Setting up advanced architecture system...")
    hybrid_system, agents = await setup_advanced_architecture_system()
    
    # Create diverse tasks
    tasks = [
        Task(
            task_id="strategic_001",
            name="strategic_planning",
            description="Develop quarterly strategic plan",
            required_capabilities=["strategic_thinking", "data_analysis"],
            input_data={"quarter": "Q1", "budget": 1000000},
            priority=9,
            deadline=datetime.now() + timedelta(days=7)
        ),
        Task(
            task_id="optimization_001",
            name="optimization_problem",
            description="Optimize resource allocation",
            required_capabilities=["optimization", "mathematical_modeling"],
            input_data={"objective_function": "minimize_cost", "constraints": ["budget", "time"]},
            priority=7,
            deadline=datetime.now() + timedelta(hours=12)
        ),
        Task(
            task_id="negotiation_001",
            name="negotiation_task",
            description="Negotiate service contract",
            required_capabilities=["negotiation", "contract_analysis"],
            input_data={"contract_value": 500000, "terms": ["price", "delivery", "quality"]},
            priority=8,
            deadline=datetime.now() + timedelta(days=3)
        ),
        Task(
            task_id="analysis_001",
            name="data_analysis",
            description="Analyze customer behavior patterns",
            required_capabilities=["data_analysis", "machine_learning"],
            input_data={"dataset_size": 1000000, "features": 50},
            priority=6,
            deadline=datetime.now() + timedelta(days=2)
        )
    ]
    
    print("\nExecuting tasks with hybrid architecture...")
    results = []
    
    for task in tasks:
        print(f"\nExecuting task: {task.name}")
        result = await hybrid_system.execute_task(task)
        results.append(result)
        
        print(f"Result: {result.get('execution_mode', 'unknown')} mode")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Success: {result.get('result', 'Task completed')}")
    
    # Display system status
    print("\n" + "="*50)
    print("SYSTEM STATUS")
    print("="*50)
    
    status = hybrid_system.get_system_status()
    print(f"System: {status['name']}")
    print(f"Total Agents: {status['total_agents']}")
    print(f"Overall Success Rate: {status['overall_success_rate']:.2%}")
    print(f"Adaptation Rules: {status['adaptation_rules']}")
    
    print("\nComponent Status:")
    for comp_id, comp_status in status['components'].items():
        print(f"  {comp_id}: {comp_status['agent_count']} agents, "
              f"Type: {comp_status['type']}")
        if comp_status['performance_metrics']:
            metrics = comp_status['performance_metrics']
            print(f"    Success Rate: {metrics.get('average_success_rate', 0):.2%}")
            print(f"    Avg Time: {metrics.get('average_execution_time', 0):.2f}s")
    
    # Demonstrate adaptive agent learning
    print("\n" + "="*50)
    print("ADAPTIVE AGENT LEARNING")
    print("="*50)
    
    adaptive_agent = next(agent for agent in agents if isinstance(agent, AdaptiveAgent))
    adaptation_summary = adaptive_agent.get_adaptation_summary()
    
    print(f"Agent: {adaptation_summary['agent_id']}")
    print(f"Tasks Executed: {adaptation_summary['total_tasks_executed']}")
    print(f"Success Rate: {adaptation_summary['overall_success_rate']:.2%}")
    print(f"Learned Patterns: {adaptation_summary['learned_patterns']}")
    
    return hybrid_system, results

# Usage example
if __name__ == "__main__":
    import asyncio
    
    async def main():
        hybrid_system, results = await demonstrate_advanced_architectures()
        
        print("\nAdvanced architecture demonstration completed!")
        print(f"Executed {len(results)} tasks with hybrid coordination")
    
    asyncio.run(main())
```

## 6. Hands-on Exercises

### Exercise 1: Custom Hierarchical Structure

```python
class CustomHierarchicalSystem:
    """Custom hierarchical system with domain-specific roles"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.hierarchy = HierarchicalAgentSystem(f"{domain}_hierarchy")
        self.domain_roles = self.define_domain_roles()
    
    def define_domain_roles(self) -> Dict[str, Dict[str, Any]]:
        """Define domain-specific roles and their relationships"""
        if self.domain == "healthcare":
            return {
                "chief_medical_officer": {"level": 0, "span": 3, "specialization": "medical_strategy"},
                "department_head": {"level": 1, "span": 5, "specialization": "department_management"},
                "senior_physician": {"level": 2, "span": 3, "specialization": "clinical_expertise"},
                "physician": {"level": 3, "span": 0, "specialization": "patient_care"},
                "nurse": {"level": 3, "span": 0, "specialization": "patient_support"}
            }
        elif self.domain == "manufacturing":
            return {
                "plant_manager": {"level": 0, "span": 4, "specialization": "operations_strategy"},
                "production_manager": {"level": 1, "span": 6, "specialization": "production_planning"},
                "line_supervisor": {"level": 2, "span": 8, "specialization": "line_management"},
                "operator": {"level": 3, "span": 0, "specialization": "equipment_operation"},
                "quality_inspector": {"level": 3, "span": 0, "specialization": "quality_control"}
            }
        else:
            return {"manager": {"level": 0, "span": 5, "specialization": "general_management"}}
    
    async def create_domain_hierarchy(self) -> Dict[str, BaseAgent]:
        """Create agents based on domain-specific hierarchy"""
        agents = {}
        
        for role_name, role_config in self.domain_roles.items():
            # Create multiple agents for each role if needed
            role_count = role_config.get("count", 1)
            
            for i in range(role_count):
                agent_id = f"{role_name}_{i:03d}"
                
                if role_config["level"] == 0:
                    agent = ExecutiveAgent(agent_id, f"{role_name.title()} {i}", AgentRole.COORDINATOR)
                elif role_config["span"] > 0:
                    agent = ManagerAgent(agent_id, f"{role_name.title()} {i}", AgentRole.COORDINATOR)
                else:
                    agent = SpecialistAgent(agent_id, f"{role_name.title()} {i}", AgentRole.SPECIALIST)
                
                # Add domain-specific capabilities
                specialization = role_config["specialization"]
                agent.add_capability(AgentCapability(
                    name=specialization,
                    description=f"Specialized in {specialization}",
                    input_schema={"domain_data": "object"},
                    output_schema={"domain_result": "object"}
                ))
                
                agents[agent_id] = agent
        
        return agents
    
    # TODO: Implement hierarchy building logic
    # TODO: Add domain-specific task routing
    # TODO: Implement performance metrics for domain
```

### Exercise 2: Advanced Swarm Optimization

```python
class AdvancedSwarmOptimizer:
    """Advanced swarm optimization with multiple algorithms"""
    
    def __init__(self, optimization_type: str):
        self.optimization_type = optimization_type
        self.algorithms = {
            "pso": self.particle_swarm_optimization,
            "aco": self.ant_colony_optimization,
            "abc": self.artificial_bee_colony,
            "hybrid": self.hybrid_optimization
        }
    
    async def particle_swarm_optimization(self, objective_function: str, 
                                        search_space: List[float], 
                                        swarm_size: int = 30) -> Dict[str, Any]:
        """Implement advanced PSO with adaptive parameters"""
        # TODO: Implement adaptive PSO
        # TODO: Add velocity clamping
        # TODO: Implement topology variations
        pass
    
    async def ant_colony_optimization(self, objective_function: str,
                                    search_space: List[float],
                                    colony_size: int = 50) -> Dict[str, Any]:
        """Implement ACO for continuous optimization"""
        # TODO: Implement continuous ACO
        # TODO: Add pheromone evaporation strategies
        # TODO: Implement elite ant strategies
        pass
    
    async def artificial_bee_colony(self, objective_function: str,
                                  search_space: List[float],
                                  colony_size: int = 40) -> Dict[str, Any]:
        """Implement ABC algorithm"""
        # TODO: Implement employed bee phase
        # TODO: Implement onlooker bee phase
        # TODO: Implement scout bee phase
        pass
    
    async def hybrid_optimization(self, objective_function: str,
                                search_space: List[float],
                                swarm_size: int = 50) -> Dict[str, Any]:
        """Implement hybrid swarm optimization"""
        # TODO: Combine multiple algorithms
        # TODO: Implement algorithm switching strategies
        # TODO: Add performance-based selection
        pass
```

### Exercise 3: Intelligent Architecture Selector

```python
class IntelligentArchitectureSelector:
    """AI-powered architecture selection system"""
    
    def __init__(self):
        self.task_history: List[Dict[str, Any]] = []
        self.architecture_performance: Dict[str, Dict[str, float]] = {}
        self.ml_model = None  # Would be a trained ML model
        self.feature_extractors = {
            "task_complexity": self.extract_task_complexity,
            "resource_requirements": self.extract_resource_requirements,
            "time_constraints": self.extract_time_constraints,
            "collaboration_needs": self.extract_collaboration_needs
        }
    
    async def select_optimal_architecture(self, task: Task, 
                                        available_architectures: List[str]) -> str:
        """Select the optimal architecture for a given task"""
        # Extract task features
        features = await self.extract_task_features(task)
        
        # Predict performance for each architecture
        architecture_scores = {}
        for arch in available_architectures:
            score = await self.predict_architecture_performance(features, arch)
            architecture_scores[arch] = score
        
        # Select best architecture
        best_architecture = max(architecture_scores.items(), key=lambda x: x[1])
        
        return best_architecture[0]
    
    async def extract_task_features(self, task: Task) -> Dict[str, float]:
        """Extract numerical features from a task"""
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            features[feature_name] = await extractor(task)
        
        return features
    
    async def extract_task_complexity(self, task: Task) -> float:
        """Extract task complexity score"""
        # TODO: Implement complexity calculation
        # Consider: number of capabilities, data size, dependencies
        return len(task.required_capabilities) / 10.0
    
    async def extract_resource_requirements(self, task: Task) -> float:
        """Extract resource requirement score"""
        # TODO: Implement resource requirement calculation
        return task.priority / 10.0
    
    async def extract_time_constraints(self, task: Task) -> float:
        """Extract time constraint score"""
        # TODO: Implement time constraint calculation
        time_to_deadline = (task.deadline - datetime.now()).total_seconds()
        return min(time_to_deadline / 86400, 1.0)  # Normalize to days
    
    async def extract_collaboration_needs(self, task: Task) -> float:
        """Extract collaboration requirement score"""
        # TODO: Implement collaboration needs calculation
        return 1.0 if len(task.required_capabilities) > 3 else 0.0
    
    async def predict_architecture_performance(self, features: Dict[str, float], 
                                             architecture: str) -> float:
        """Predict performance of an architecture for given features"""
        # TODO: Implement ML-based prediction
        # For now, use simple heuristics
        
        if architecture == "hierarchical":
            return features["task_complexity"] * 0.8 + features["resource_requirements"] * 0.2
        elif architecture == "swarm":
            return features["collaboration_needs"] * 0.6 + (1 - features["time_constraints"]) * 0.4
        elif architecture == "market":
            return features["resource_requirements"] * 0.5 + features["time_constraints"] * 0.5
        else:
            return 0.5  # Default score
    
    async def update_performance_history(self, task: Task, architecture: str, 
                                       result: Dict[str, Any]):
        """Update performance history for learning"""
        # TODO: Implement performance tracking
        # TODO: Retrain ML model periodically
        pass
```

## Module Summary

In this module, we explored advanced agent architectures that enable sophisticated multi-agent coordination and adaptation. Here's what we covered:

### Key Concepts Learned

1. **Multi-Agent System Foundations**
   - Coordination patterns and agent roles
   - Base agent classes and capabilities
   - Task management and execution

2. **Hierarchical Agent Structures**
   - Executive, manager, and worker agent roles
   - Delegation and escalation mechanisms
   - Span of control and organizational depth

3. **Swarm Intelligence Systems**
   - Particle Swarm Optimization (PSO)
   - Ant Colony Optimization (ACO)
   - Collective decision-making and consensus

4. **Hybrid and Adaptive Architectures**
   - Combining multiple coordination patterns
   - Dynamic architecture selection
   - Performance-based adaptation

5. **Market-Based Coordination**
   - Agent marketplaces and auctions
   - Bidding strategies and evaluation
   - Reputation and rating systems

### Practical Skills Developed

1. **Architecture Design**
   - Designing hierarchical agent structures
   - Implementing swarm coordination algorithms
   - Creating hybrid systems that combine multiple patterns

2. **Adaptive Systems**
   - Building self-adapting agents
   - Implementing learning mechanisms
   - Creating performance-based optimization

3. **Market Mechanisms**
   - Implementing auction systems
   - Creating bidding and evaluation algorithms
   - Building reputation systems

4. **System Integration**
   - Combining different architectural patterns
   - Managing complex multi-agent interactions
   - Implementing comprehensive monitoring and adaptation

### Real-World Applications

1. **Enterprise Systems**
   - Organizational workflow automation
   - Resource allocation and management
   - Strategic planning and execution

2. **Optimization Problems**
   - Supply chain optimization
   - Resource scheduling
   - Multi-objective optimization

3. **Distributed Computing**
   - Cloud resource management
   - Edge computing coordination
   - Load balancing and scaling

4. **Smart Systems**
   - Smart city management
   - IoT device coordination
   - Autonomous vehicle fleets

### Next Steps

This module provides the foundation for building sophisticated multi-agent systems. In the next modules, we'll explore:

- **Module IX: Production Deployment and Scaling** - Learn how to deploy and scale agent systems in production environments
- **Module X: Performance Optimization and Monitoring** - Advanced techniques for optimizing agent performance and system monitoring
- **Module XI: Future Trends and Research Directions** - Explore cutting-edge research and emerging trends in agent systems

The advanced architectures covered in this module enable you to build robust, scalable, and adaptive multi-agent systems that can handle complex real-world challenges.
```