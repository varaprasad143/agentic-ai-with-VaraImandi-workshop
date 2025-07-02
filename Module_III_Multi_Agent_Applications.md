# Module III: Multi-Agent Applications

## Learning Objectives

By the end of this module, you will be able to:
- Design and implement multi-agent systems with clear roles and responsibilities
- Coordinate communication and collaboration between multiple AI agents
- Build agent orchestration systems for complex workflows
- Implement agent discovery and registration mechanisms
- Handle conflict resolution and consensus building in multi-agent environments
- Create scalable agent architectures for production systems
- Monitor and debug multi-agent interactions

---

## 1. Introduction to Multi-Agent Systems

### 1.1 What are Multi-Agent Systems?

Multi-Agent Systems (MAS) are computational systems where multiple autonomous agents interact to solve problems that are beyond the capabilities of individual agents. Each agent has its own goals, knowledge, and capabilities, but they work together to achieve common objectives.

### 1.2 Agent Architecture Overview

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Multi-Agent System Architecture</text>
  
  <!-- Agent 1 -->
  <g transform="translate(50, 80)">
    <rect width="150" height="120" rx="10" fill="#3498db" stroke="#2980b9" stroke-width="2"/>
    <text x="75" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Agent 1</text>
    <text x="75" y="45" text-anchor="middle" font-size="12" fill="white">Specialist</text>
    
    <!-- Components -->
    <rect x="10" y="55" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="66" text-anchor="middle" font-size="10" fill="#2c3e50">Knowledge Base</text>
    
    <rect x="10" y="75" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="86" text-anchor="middle" font-size="10" fill="#2c3e50">Decision Engine</text>
    
    <rect x="10" y="95" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="106" text-anchor="middle" font-size="10" fill="#2c3e50">Communication</text>
  </g>
  
  <!-- Agent 2 -->
  <g transform="translate(325, 80)">
    <rect width="150" height="120" rx="10" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
    <text x="75" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Agent 2</text>
    <text x="75" y="45" text-anchor="middle" font-size="12" fill="white">Coordinator</text>
    
    <!-- Components -->
    <rect x="10" y="55" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="66" text-anchor="middle" font-size="10" fill="#2c3e50">Task Manager</text>
    
    <rect x="10" y="75" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="86" text-anchor="middle" font-size="10" fill="#2c3e50">Resource Allocator</text>
    
    <rect x="10" y="95" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="106" text-anchor="middle" font-size="10" fill="#2c3e50">Communication</text>
  </g>
  
  <!-- Agent 3 -->
  <g transform="translate(600, 80)">
    <rect width="150" height="120" rx="10" fill="#27ae60" stroke="#229954" stroke-width="2"/>
    <text x="75" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Agent 3</text>
    <text x="75" y="45" text-anchor="middle" font-size="12" fill="white">Executor</text>
    
    <!-- Components -->
    <rect x="10" y="55" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="66" text-anchor="middle" font-size="10" fill="#2c3e50">Action Engine</text>
    
    <rect x="10" y="75" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="86" text-anchor="middle" font-size="10" fill="#2c3e50">Tool Interface</text>
    
    <rect x="10" y="95" width="130" height="15" rx="3" fill="#ecf0f1"/>
    <text x="75" y="106" text-anchor="middle" font-size="10" fill="#2c3e50">Communication</text>
  </g>
  
  <!-- Communication Layer -->
  <rect x="50" y="250" width="700" height="80" rx="10" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="400" y="275" text-anchor="middle" font-size="16" font-weight="bold" fill="white">Message Bus / Communication Layer</text>
  
  <!-- Communication features -->
  <g transform="translate(70, 290)">
    <circle cx="0" cy="0" r="3" fill="white"/>
    <text x="10" y="5" font-size="12" fill="white">Message Routing</text>
  </g>
  
  <g transform="translate(220, 290)">
    <circle cx="0" cy="0" r="3" fill="white"/>
    <text x="10" y="5" font-size="12" fill="white">Event Broadcasting</text>
  </g>
  
  <g transform="translate(400, 290)">
    <circle cx="0" cy="0" r="3" fill="white"/>
    <text x="10" y="5" font-size="12" fill="white">Protocol Management</text>
  </g>
  
  <g transform="translate(580, 290)">
    <circle cx="0" cy="0" r="3" fill="white"/>
    <text x="10" y="5" font-size="12" fill="white">Security & Auth</text>
  </g>
  
  <!-- Arrows from agents to communication layer -->
  <path d="M 125 200 L 125 250" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M 400 200 L 400 250" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M 675 200 L 675 250" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Bidirectional arrows -->
  <path d="M 125 250 L 125 200" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M 400 250 L 400 200" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M 675 250 L 675 200" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Shared Resources -->
  <rect x="50" y="380" width="200" height="100" rx="10" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
  <text x="150" y="405" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Shared Resources</text>
  <text x="150" y="425" text-anchor="middle" font-size="12" fill="white">• Knowledge Base</text>
  <text x="150" y="445" text-anchor="middle" font-size="12" fill="white">• Vector Database</text>
  <text x="150" y="465" text-anchor="middle" font-size="12" fill="white">• External APIs</text>
  
  <!-- Orchestrator -->
  <rect x="300" y="380" width="200" height="100" rx="10" fill="#e67e22" stroke="#d35400" stroke-width="2"/>
  <text x="400" y="405" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Orchestrator</text>
  <text x="400" y="425" text-anchor="middle" font-size="12" fill="white">• Workflow Management</text>
  <text x="400" y="445" text-anchor="middle" font-size="12" fill="white">• Task Distribution</text>
  <text x="400" y="465" text-anchor="middle" font-size="12" fill="white">• Conflict Resolution</text>
  
  <!-- Monitoring -->
  <rect x="550" y="380" width="200" height="100" rx="10" fill="#34495e" stroke="#2c3e50" stroke-width="2"/>
  <text x="650" y="405" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Monitoring</text>
  <text x="650" y="425" text-anchor="middle" font-size="12" fill="white">• Performance Metrics</text>
  <text x="650" y="445" text-anchor="middle" font-size="12" fill="white">• Health Checks</text>
  <text x="650" y="465" text-anchor="middle" font-size="12" fill="white">• Logging & Alerts</text>
  
  <!-- Arrows from communication layer to components -->
  <path d="M 200 330 L 150 380" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M 400 330 L 400 380" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M 600 330 L 650 380" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Agent interaction arrows -->
  <path d="M 200 140 L 325 140" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead)"/>
  <path d="M 475 140 L 600 140" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead)"/>
  <path d="M 600 160 L 475 160" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead)"/>
  
  <!-- Legend -->
  <g transform="translate(50, 520)">
    <text x="0" y="0" font-size="14" font-weight="bold" fill="#2c3e50">Legend:</text>
    <line x1="0" y1="15" x2="20" y2="15" stroke="#34495e" stroke-width="2"/>
    <text x="25" y="20" font-size="12" fill="#2c3e50">Direct Communication</text>
    
    <line x1="150" y1="15" x2="170" y2="15" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="5,5"/>
    <text x="175" y="20" font-size="12" fill="#2c3e50">Agent Interaction</text>
    
    <circle cx="320" cy="15" r="5" fill="#f39c12"/>
    <text x="330" y="20" font-size="12" fill="#2c3e50">Message Bus</text>
  </g>
  
  <!-- Arrow marker definition -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
</svg>
```

### 1.3 Core Agent Implementation

```python
# core_agent.py
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime

class AgentState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.REQUEST
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: int = 0  # Higher number = higher priority

@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    cost: float = 1.0  # Relative cost of using this capability

class BaseAgent:
    def __init__(self, agent_id: str, name: str, description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.capabilities: Dict[str, AgentCapability] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger(f"Agent.{self.name}")
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'tasks_completed': 0,
            'errors': 0,
            'uptime_start': datetime.now()
        }
        
        # Communication interface (to be set by the system)
        self.communication_interface = None
        
        # Register default message handlers
        self.register_handler("ping", self._handle_ping)
        self.register_handler("get_capabilities", self._handle_get_capabilities)
        self.register_handler("get_status", self._handle_get_status)
    
    def register_capability(self, capability: AgentCapability):
        """Register a new capability for this agent"""
        self.capabilities[capability.name] = capability
        self.logger.info(f"Registered capability: {capability.name}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for: {message_type}")
    
    async def start(self):
        """Start the agent's main processing loop"""
        self.state = AgentState.IDLE
        self.logger.info(f"Agent {self.name} starting...")
        
        # Start the message processing loop
        asyncio.create_task(self._process_messages())
        
        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())
        
        self.logger.info(f"Agent {self.name} started successfully")
    
    async def stop(self):
        """Stop the agent gracefully"""
        self.state = AgentState.OFFLINE
        self.logger.info(f"Agent {self.name} stopping...")
    
    async def send_message(self, message: Message):
        """Send a message through the communication interface"""
        if self.communication_interface:
            message.sender_id = self.agent_id
            await self.communication_interface.send_message(message)
            self.metrics['messages_sent'] += 1
            self.logger.debug(f"Sent message {message.id} to {message.receiver_id}")
        else:
            self.logger.error("No communication interface available")
    
    async def receive_message(self, message: Message):
        """Receive a message and add it to the processing queue"""
        await self.message_queue.put(message)
        self.metrics['messages_received'] += 1
        self.logger.debug(f"Received message {message.id} from {message.sender_id}")
    
    async def _process_messages(self):
        """Main message processing loop"""
        while self.state != AgentState.OFFLINE:
            try:
                # Wait for a message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._handle_message(message)
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self.metrics['errors'] += 1
    
    async def _handle_message(self, message: Message):
        """Handle an incoming message"""
        try:
            self.state = AgentState.BUSY
            
            # Extract message type from content
            msg_type = message.content.get('type', 'unknown')
            
            if msg_type in self.message_handlers:
                response = await self.message_handlers[msg_type](message)
                
                # Send response if one was generated
                if response and message.message_type == MessageType.REQUEST:
                    response_msg = Message(
                        receiver_id=message.sender_id,
                        message_type=MessageType.RESPONSE,
                        content=response,
                        correlation_id=message.id
                    )
                    await self.send_message(response_msg)
            else:
                self.logger.warning(f"No handler for message type: {msg_type}")
                
                # Send error response
                if message.message_type == MessageType.REQUEST:
                    error_response = Message(
                        receiver_id=message.sender_id,
                        message_type=MessageType.ERROR,
                        content={
                            'error': f"Unknown message type: {msg_type}",
                            'original_message_id': message.id
                        }
```

---

## 5. Hands-on Exercises

### Exercise 1: Building a Multi-Agent Task Processing System

```python
# multi_agent_task_system.py
import asyncio
import json
from typing import Dict, List, Any
from datetime import datetime

from core_agent import BaseAgent, AgentCapability, AgentState
from message_bus import MessageBus
from workflow_orchestrator import WorkflowOrchestrator, TaskDefinition, WorkflowDefinition
from agent_registry import AgentRegistry
from conflict_resolution import ConflictResolver

class TaskProcessingAgent(BaseAgent):
    def __init__(self, agent_id: str, specialization: str, processing_time: float = 1.0):
        super().__init__(agent_id, f"{specialization}Agent", f"Processes {specialization} tasks")
        self.specialization = specialization
        self.processing_time = processing_time
        self.current_load = 0.0
        
        # Define capabilities
        self.capabilities = {
            f"process_{specialization}": AgentCapability(
                name=f"process_{specialization}",
                description=f"Process {specialization} related tasks",
                input_schema={"task_data": "object", "priority": "integer"},
                output_schema={"result": "object", "processing_time": "number"},
                cost=processing_time
            )
        }
        
        # Register message handlers
        self.register_handler("process_task", self._handle_process_task)
        self.register_handler("get_load_info", self._handle_get_load_info)
        self.register_handler("get_capabilities", self._handle_get_capabilities)
    
    async def _handle_process_task(self, message) -> Dict[str, Any]:
        """Handle task processing requests"""
        task_data = message.content.get('task_data', {})
        priority = message.content.get('priority', 1)
        
        self.logger.info(f"Processing {self.specialization} task: {task_data.get('id', 'unknown')}")
        
        # Simulate processing time
        self.current_load += 0.1
        start_time = datetime.now()
        
        await asyncio.sleep(self.processing_time)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.current_load = max(0, self.current_load - 0.1)
        
        # Generate result based on specialization
        if self.specialization == "data":
            result = {
                'processed_records': task_data.get('record_count', 100),
                'data_quality_score': 0.95,
                'anomalies_detected': 2
            }
        elif self.specialization == "image":
            result = {
                'images_processed': task_data.get('image_count', 10),
                'objects_detected': 25,
                'confidence_score': 0.89
            }
        elif self.specialization == "text":
            result = {
                'documents_analyzed': task_data.get('document_count', 50),
                'sentiment_score': 0.7,
                'key_topics': ['AI', 'automation', 'efficiency']
            }
        else:
            result = {'status': 'completed', 'data': task_data}
        
        return {
            'type': 'task_completed',
            'task_id': task_data.get('id'),
            'result': result,
            'processing_time': processing_time,
            'agent_id': self.agent_id
        }
    
    async def _handle_get_load_info(self, message) -> Dict[str, Any]:
        """Handle load information requests"""
        return {
            'type': 'load_info_response',
            'load': self.current_load,
            'specialization': self.specialization,
            'processing_time': self.processing_time
        }
    
    async def _handle_get_capabilities(self, message) -> Dict[str, Any]:
        """Handle capability requests"""
        return {
            'type': 'capabilities_response',
            'capabilities': {
                name: {
                    'description': cap.description,
                    'input_schema': cap.input_schema,
                    'output_schema': cap.output_schema,
                    'cost': cap.cost
                }
                for name, cap in self.capabilities.items()
            }
        }

class MultiAgentTaskSystem:
    def __init__(self):
        self.message_bus = MessageBus()
        self.registry = AgentRegistry("registry", self.message_bus)
        self.orchestrator = WorkflowOrchestrator("orchestrator", self.message_bus)
        self.conflict_resolver = ConflictResolver("conflict_resolver", self.message_bus)
        self.agents: Dict[str, TaskProcessingAgent] = {}
        
    async def initialize(self):
        """Initialize the multi-agent system"""
        # Start core services
        await self.registry.start()
        await self.orchestrator.start()
        await self.conflict_resolver.start()
        
        # Create specialized agents
        specializations = [
            ("data", 2.0),
            ("image", 3.0),
            ("text", 1.5),
            ("data", 2.5),  # Second data agent
            ("image", 2.8)  # Second image agent
        ]
        
        for i, (spec, time) in enumerate(specializations):
            agent_id = f"{spec}_agent_{i+1}"
            agent = TaskProcessingAgent(agent_id, spec, time)
            self.agents[agent_id] = agent
            
            await agent.start()
            
            # Register agent with registry
            agent_info = {
                'agent_id': agent_id,
                'name': agent.name,
                'description': agent.description,
                'capabilities': {
                    name: {
                        'description': cap.description,
                        'input_schema': cap.input_schema,
                        'output_schema': cap.output_schema,
                        'cost': cap.cost
                    }
                    for name, cap in agent.capabilities.items()
                },
                'endpoint': f"local://{agent_id}",
                'tags': [spec, 'processor'],
                'metadata': {
                    'specialization': spec,
                    'processing_time': time,
                    'capabilities': list(agent.capabilities.keys())
                }
            }
            
            await self.registry.register_agent(agent_info)
        
        print(f"Initialized {len(self.agents)} agents")
        print(f"Registry stats: {self.registry.get_registry_stats()}")
    
    async def create_complex_workflow(self) -> str:
        """Create a complex multi-step workflow"""
        # Define tasks
        tasks = [
            TaskDefinition(
                task_id="data_ingestion",
                name="Data Ingestion",
                description="Ingest and validate raw data",
                required_capabilities=["process_data"],
                input_data={
                    'task_data': {'id': 'data_001', 'record_count': 1000},
                    'priority': 1
                },
                dependencies=[],
                timeout=30.0
            ),
            TaskDefinition(
                task_id="image_processing",
                name="Image Processing",
                description="Process and analyze images",
                required_capabilities=["process_image"],
                input_data={
                    'task_data': {'id': 'img_001', 'image_count': 50},
                    'priority': 2
                },
                dependencies=[],
                timeout=45.0
            ),
            TaskDefinition(
                task_id="text_analysis",
                name="Text Analysis",
                description="Analyze text documents",
                required_capabilities=["process_text"],
                input_data={
                    'task_data': {'id': 'text_001', 'document_count': 200},
                    'priority': 1
                },
                dependencies=["data_ingestion"],
                timeout=25.0
            ),
            TaskDefinition(
                task_id="final_aggregation",
                name="Final Aggregation",
                description="Aggregate all processing results",
                required_capabilities=["process_data"],
                input_data={
                    'task_data': {'id': 'agg_001', 'sources': ['data', 'image', 'text']},
                    'priority': 3
                },
                dependencies=["data_ingestion", "image_processing", "text_analysis"],
                timeout=20.0
            )
        ]
        
        # Create workflow
        workflow = WorkflowDefinition(
            workflow_id="complex_processing_workflow",
            name="Complex Data Processing Workflow",
            description="Multi-step data processing with different specializations",
            tasks=tasks,
            metadata={'created_by': 'system', 'priority': 'high'}
        )
        
        # Register and execute workflow
        workflow_id = await self.orchestrator.register_workflow(workflow)
        execution_id = await self.orchestrator.execute_workflow(workflow_id)
        
        return execution_id
    
    async def simulate_conflict_scenario(self):
        """Simulate a resource contention conflict"""
        # Create a scenario where multiple agents want the same resource
        conflict_data = {
            'type': 'resource_contention',
            'description': 'Multiple agents competing for high-priority GPU resource',
            'participants': [
                {
                    'agent_id': 'image_agent_1',
                    'position': {'resource': 'gpu_1', 'duration': 300, 'priority': 2},
                    'priority': 2,
                    'weight': 1.0,
                    'metadata': {'capabilities': ['process_image']}
                },
                {
                    'agent_id': 'image_agent_2',
                    'position': {'resource': 'gpu_1', 'duration': 180, 'priority': 3},
                    'priority': 3,
                    'weight': 1.0,
                    'metadata': {'capabilities': ['process_image']}
                },
                {
                    'agent_id': 'data_agent_1',
                    'position': {'resource': 'gpu_1', 'duration': 240, 'priority': 1},
                    'priority': 1,
                    'weight': 0.8,
                    'metadata': {'capabilities': ['process_data']}
                }
            ],
            'context': {
                'resource_type': 'gpu',
                'required_capabilities': ['process_image'],
                'availability_window': 600
            }
        }
        
        conflict_id = await self.conflict_resolver.detect_conflict(conflict_data)
        print(f"Created conflict scenario: {conflict_id}")
        
        # Wait for resolution
        await asyncio.sleep(5)
        
        # Check conflict status
        status_response = await self.conflict_resolver._handle_get_conflict_status(
            type('Message', (), {'content': {'conflict_id': conflict_id}})()
        )
        
        print(f"Conflict resolution: {status_response}")
        return conflict_id
    
    async def monitor_system(self, duration: int = 60):
        """Monitor system performance"""
        print(f"Monitoring system for {duration} seconds...")
        
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < duration:
            # Get registry stats
            registry_stats = self.registry.get_registry_stats()
            
            # Get orchestrator stats
            orchestrator_stats = self.orchestrator.get_orchestrator_stats()
            
            # Get conflict stats
            conflict_stats = self.conflict_resolver.get_conflict_statistics()
            
            print(f"\n--- System Status at {datetime.now().strftime('%H:%M:%S')} ---")
            print(f"Registry: {registry_stats['total_agents']} agents, {registry_stats['healthy_agents']} healthy")
            print(f"Orchestrator: {orchestrator_stats['total_workflows']} workflows, {orchestrator_stats['active_executions']} active")
            print(f"Conflicts: {conflict_stats['total_conflicts']} total, {conflict_stats['resolved_conflicts']} resolved")
            
            await asyncio.sleep(10)
    
    async def shutdown(self):
        """Shutdown the system"""
        print("Shutting down multi-agent system...")
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        # Stop core services
        await self.conflict_resolver.stop()
        await self.orchestrator.stop()
        await self.registry.stop()
        
        print("System shutdown complete")

# Usage example
async def main():
    system = MultiAgentTaskSystem()
    
    try:
        # Initialize system
        await system.initialize()
        
        # Create and execute workflow
        execution_id = await system.create_complex_workflow()
        print(f"Started workflow execution: {execution_id}")
        
        # Simulate conflict
        conflict_id = await system.simulate_conflict_scenario()
        
        # Monitor system
        await system.monitor_system(30)
        
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Exercise 2: Agent Communication Patterns

```python
# communication_patterns.py
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from core_agent import BaseAgent, Message, MessageType
from message_bus import MessageBus

class PublisherAgent(BaseAgent):
    def __init__(self, agent_id: str, topic: str):
        super().__init__(agent_id, "Publisher", f"Publishes data to {topic}")
        self.topic = topic
        self.message_count = 0
        
    async def publish_data(self, data: Dict[str, Any]):
        """Publish data to subscribers"""
        self.message_count += 1
        
        message = {
            'type': 'data_update',
            'topic': self.topic,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'sequence': self.message_count
        }
        
        await self.message_bus.broadcast_message(self.agent_id, message)
        self.logger.info(f"Published message {self.message_count} to {self.topic}")

class SubscriberAgent(BaseAgent):
    def __init__(self, agent_id: str, topics: List[str]):
        super().__init__(agent_id, "Subscriber", f"Subscribes to {topics}")
        self.topics = topics
        self.received_messages: List[Dict[str, Any]] = []
        
        # Register handlers
        self.register_handler("data_update", self._handle_data_update)
        
    async def _handle_data_update(self, message) -> Optional[Dict[str, Any]]:
        """Handle incoming data updates"""
        topic = message.content.get('topic')
        
        if topic in self.topics:
            self.received_messages.append(message.content)
            self.logger.info(f"Received message from {topic}: {message.content.get('sequence')}")
            
            # Process the data
            await self._process_data(message.content)
        
        return None
    
    async def _process_data(self, data: Dict[str, Any]):
        """Process received data"""
        # Simulate processing
        await asyncio.sleep(0.1)
        
        processed_data = {
            'original_sequence': data.get('sequence'),
            'processed_at': datetime.now().isoformat(),
            'processor': self.agent_id
        }
        
        self.logger.info(f"Processed data: {processed_data}")

class RequestResponseAgent(BaseAgent):
    def __init__(self, agent_id: str, service_name: str):
        super().__init__(agent_id, "Service", f"Provides {service_name} service")
        self.service_name = service_name
        self.request_count = 0
        
        # Register handlers
        self.register_handler("service_request", self._handle_service_request)
    
    async def _handle_service_request(self, message) -> Dict[str, Any]:
        """Handle service requests"""
        self.request_count += 1
        request_data = message.content.get('request_data', {})
        
        self.logger.info(f"Processing service request {self.request_count}")
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Generate response based on service type
        if self.service_name == "calculation":
            result = {
                'sum': request_data.get('a', 0) + request_data.get('b', 0),
                'product': request_data.get('a', 0) * request_data.get('b', 0)
            }
        elif self.service_name == "validation":
            result = {
                'valid': len(request_data.get('data', '')) > 0,
                'length': len(request_data.get('data', ''))
            }
        else:
            result = {'status': 'processed', 'data': request_data}
        
        return {
            'type': 'service_response',
            'service': self.service_name,
            'result': result,
            'request_id': message.content.get('request_id'),
            'processed_at': datetime.now().isoformat()
        }

class ClientAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "Client", "Makes service requests")
        self.responses: List[Dict[str, Any]] = []
    
    async def make_request(self, service_agent_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to a service agent"""
        request_id = f"req_{datetime.now().timestamp()}"
        
        request_message = {
            'type': 'service_request',
            'request_id': request_id,
            'request_data': request_data,
            'client_id': self.agent_id
        }
        
        self.logger.info(f"Making request {request_id} to {service_agent_id}")
        
        # Send request and wait for response
        response = await self.message_bus.request_response(
            self.agent_id, service_agent_id, request_message, timeout=10.0
        )
        
        if response:
            self.responses.append(response.content)
            self.logger.info(f"Received response for {request_id}")
            return response.content
        else:
            self.logger.error(f"No response received for {request_id}")
            return {'error': 'timeout'}

class CommunicationDemo:
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
    
    async def setup_publish_subscribe_demo(self):
        """Setup publish-subscribe communication pattern"""
        # Create publishers
        weather_publisher = PublisherAgent("weather_pub", "weather")
        stock_publisher = PublisherAgent("stock_pub", "stocks")
        
        # Create subscribers
        weather_subscriber = SubscriberAgent("weather_sub", ["weather"])
        stock_subscriber = SubscriberAgent("stock_sub", ["stocks"])
        multi_subscriber = SubscriberAgent("multi_sub", ["weather", "stocks"])
        
        # Store agents
        self.agents.update({
            "weather_pub": weather_publisher,
            "stock_pub": stock_publisher,
            "weather_sub": weather_subscriber,
            "stock_sub": stock_subscriber,
            "multi_sub": multi_subscriber
        })
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
        
        print("Publish-Subscribe demo setup complete")
    
    async def run_publish_subscribe_demo(self):
        """Run publish-subscribe demonstration"""
        weather_pub = self.agents["weather_pub"]
        stock_pub = self.agents["stock_pub"]
        
        # Publish weather data
        for i in range(5):
            weather_data = {
                'temperature': 20 + i,
                'humidity': 60 + i * 2,
                'location': 'New York'
            }
            await weather_pub.publish_data(weather_data)
            await asyncio.sleep(1)
        
        # Publish stock data
        for i in range(3):
            stock_data = {
                'symbol': 'AAPL',
                'price': 150.0 + i,
                'volume': 1000000 + i * 100000
            }
            await stock_pub.publish_data(stock_data)
            await asyncio.sleep(1.5)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Print results
        for agent_id, agent in self.agents.items():
            if isinstance(agent, SubscriberAgent):
                print(f"{agent_id} received {len(agent.received_messages)} messages")
    
    async def setup_request_response_demo(self):
        """Setup request-response communication pattern"""
        # Create service agents
        calc_service = RequestResponseAgent("calc_service", "calculation")
        validation_service = RequestResponseAgent("validation_service", "validation")
        
        # Create client agents
        client1 = ClientAgent("client1")
        client2 = ClientAgent("client2")
        
        # Store agents
        self.agents.update({
            "calc_service": calc_service,
            "validation_service": validation_service,
            "client1": client1,
            "client2": client2
        })
        
        # Start new agents
        for agent_id in ["calc_service", "validation_service", "client1", "client2"]:
            await self.agents[agent_id].start()
        
        print("Request-Response demo setup complete")
    
    async def run_request_response_demo(self):
        """Run request-response demonstration"""
        client1 = self.agents["client1"]
        client2 = self.agents["client2"]
        
        # Client 1 makes calculation requests
        calc_requests = [
            {'a': 10, 'b': 5},
            {'a': 20, 'b': 3},
            {'a': 15, 'b': 7}
        ]
        
        for request_data in calc_requests:
            response = await client1.make_request("calc_service", request_data)
            print(f"Calculation result: {response}")
            await asyncio.sleep(0.5)
        
        # Client 2 makes validation requests
        validation_requests = [
            {'data': 'hello world'},
            {'data': ''},
            {'data': 'test123'}
        ]
        
        for request_data in validation_requests:
            response = await client2.make_request("validation_service", request_data)
            print(f"Validation result: {response}")
            await asyncio.sleep(0.5)
        
        # Print summary
        print(f"Client1 received {len(client1.responses)} responses")
        print(f"Client2 received {len(client2.responses)} responses")
    
    async def shutdown(self):
        """Shutdown all agents"""
        for agent in self.agents.values():
            await agent.stop()
        print("Communication demo shutdown complete")

# Usage example
async def main():
    demo = CommunicationDemo()
    
    try:
        # Run publish-subscribe demo
        print("=== Publish-Subscribe Communication Demo ===")
        await demo.setup_publish_subscribe_demo()
        await demo.run_publish_subscribe_demo()
        
        await asyncio.sleep(2)
        
        # Run request-response demo
        print("\n=== Request-Response Communication Demo ===")
        await demo.setup_request_response_demo()
        await demo.run_request_response_demo()
        
    finally:
        await demo.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Exercise 3: Distributed Agent Coordination

```python
# distributed_coordination.py
import asyncio
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from core_agent import BaseAgent, Message, MessageType
from message_bus import MessageBus
from conflict_resolution import ConflictResolver, ConflictType

@dataclass
class DistributedTask:
    task_id: str
    data: Dict[str, Any]
    required_agents: int
    timeout: float
    created_at: datetime
    assigned_agents: List[str] = None
    results: Dict[str, Any] = None
    status: str = "pending"  # pending, assigned, processing, completed, failed

class CoordinatorAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "Coordinator", "Coordinates distributed tasks")
        self.pending_tasks: Dict[str, DistributedTask] = {}
        self.available_workers: List[str] = []
        self.task_assignments: Dict[str, List[str]] = {}  # task_id -> [worker_ids]
        
        # Register handlers
        self.register_handler("worker_available", self._handle_worker_available)
        self.register_handler("worker_unavailable", self._handle_worker_unavailable)
        self.register_handler("task_result", self._handle_task_result)
        self.register_handler("submit_task", self._handle_submit_task)
        self.register_handler("get_task_status", self._handle_get_task_status)
    
    async def submit_distributed_task(self, task_data: Dict[str, Any], required_agents: int = 3) -> str:
        """Submit a task for distributed processing"""
        task_id = f"task_{datetime.now().timestamp()}"
        
        task = DistributedTask(
            task_id=task_id,
            data=task_data,
            required_agents=required_agents,
            timeout=30.0,
            created_at=datetime.now(),
            assigned_agents=[],
            results={}
        )
        
        self.pending_tasks[task_id] = task
        
        # Try to assign workers immediately
        await self._assign_workers_to_task(task_id)
        
        return task_id
    
    async def _assign_workers_to_task(self, task_id: str):
        """Assign available workers to a task"""
        task = self.pending_tasks.get(task_id)
        if not task or task.status != "pending":
            return
        
        # Check if we have enough available workers
        if len(self.available_workers) >= task.required_agents:
            # Select workers (could use more sophisticated selection logic)
            selected_workers = random.sample(self.available_workers, task.required_agents)
            
            # Assign task to workers
            task.assigned_agents = selected_workers
            task.status = "assigned"
            self.task_assignments[task_id] = selected_workers
            
            # Remove assigned workers from available list
            for worker_id in selected_workers:
                self.available_workers.remove(worker_id)
            
            # Send task to workers
            for worker_id in selected_workers:
                task_message = {
                    'type': 'execute_task',
                    'task_id': task_id,
                    'task_data': task.data,
                    'coordinator_id': self.agent_id
                }
                
                await self.message_bus.send_message(
                    self.agent_id, worker_id, task_message
                )
            
            task.status = "processing"
            self.logger.info(f"Assigned task {task_id} to {len(selected_workers)} workers")
            
            # Set timeout for task completion
            asyncio.create_task(self._monitor_task_timeout(task_id))
    
    async def _monitor_task_timeout(self, task_id: str):
        """Monitor task for timeout"""
        task = self.pending_tasks.get(task_id)
        if not task:
            return
        
        await asyncio.sleep(task.timeout)
        
        # Check if task is still processing
        if task.status == "processing":
            self.logger.warning(f"Task {task_id} timed out")
            task.status = "failed"
            
            # Release assigned workers
            if task.assigned_agents:
                self.available_workers.extend(task.assigned_agents)
                del self.task_assignments[task_id]
    
    async def _handle_worker_available(self, message) -> Optional[Dict[str, Any]]:
        """Handle worker availability notifications"""
        worker_id = message.sender_id
        
        if worker_id not in self.available_workers:
            self.available_workers.append(worker_id)
            self.logger.info(f"Worker {worker_id} is now available")
            
            # Try to assign pending tasks
            for task_id, task in self.pending_tasks.items():
                if task.status == "pending":
                    await self._assign_workers_to_task(task_id)
                    break
        
        return None
    
    async def _handle_worker_unavailable(self, message) -> Optional[Dict[str, Any]]:
        """Handle worker unavailability notifications"""
        worker_id = message.sender_id
        
        if worker_id in self.available_workers:
            self.available_workers.remove(worker_id)
            self.logger.info(f"Worker {worker_id} is now unavailable")
        
        return None
    
    async def _handle_task_result(self, message) -> Optional[Dict[str, Any]]:
        """Handle task completion results"""
        task_id = message.content.get('task_id')
        worker_id = message.sender_id
        result = message.content.get('result')
        
        task = self.pending_tasks.get(task_id)
        if not task:
            return None
        
        # Store result
        task.results[worker_id] = result
        
        # Check if all workers have completed
        if len(task.results) == len(task.assigned_agents):
            task.status = "completed"
            
            # Aggregate results
            aggregated_result = self._aggregate_results(task.results)
            task.results['aggregated'] = aggregated_result
            
            # Release workers
            self.available_workers.extend(task.assigned_agents)
            del self.task_assignments[task_id]
            
            self.logger.info(f"Task {task_id} completed with {len(task.results)-1} results")
        
        return None
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple workers"""
        # Simple aggregation - in practice, this would be more sophisticated
        aggregated = {
            'worker_count': len(results),
            'completion_time': datetime.now().isoformat(),
            'results_summary': {}
        }
        
        # Collect numeric values for averaging
        numeric_values = {}
        for worker_id, result in results.items():
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_values:
                        numeric_values[key] = []
                    numeric_values[key].append(value)
        
        # Calculate averages
        for key, values in numeric_values.items():
            aggregated['results_summary'][f'avg_{key}'] = sum(values) / len(values)
            aggregated['results_summary'][f'min_{key}'] = min(values)
            aggregated['results_summary'][f'max_{key}'] = max(values)
        
        return aggregated
    
    async def _handle_submit_task(self, message) -> Dict[str, Any]:
        """Handle task submission requests"""
        task_data = message.content.get('task_data', {})
        required_agents = message.content.get('required_agents', 3)
        
        task_id = await self.submit_distributed_task(task_data, required_agents)
        
        return {
            'type': 'task_submitted',
            'task_id': task_id,
            'status': 'pending'
        }
    
    async def _handle_get_task_status(self, message) -> Dict[str, Any]:
        """Handle task status requests"""
        task_id = message.content.get('task_id')
        
        task = self.pending_tasks.get(task_id)
        if not task:
            return {
                'type': 'task_status_response',
                'error': 'Task not found'
            }
        
        return {
            'type': 'task_status_response',
            'task_id': task_id,
            'status': task.status,
            'assigned_agents': task.assigned_agents,
            'results_count': len(task.results),
            'created_at': task.created_at.isoformat()
        }
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        status_counts = {}
        for task in self.pending_tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        
        return {
            'total_tasks': len(self.pending_tasks),
            'available_workers': len(self.available_workers),
            'active_assignments': len(self.task_assignments),
            'task_status_distribution': status_counts
        }

class WorkerAgent(BaseAgent):
    def __init__(self, agent_id: str, processing_capability: str):
        super().__init__(agent_id, "Worker", f"Processes {processing_capability} tasks")
        self.processing_capability = processing_capability
        self.is_busy = False
        self.completed_tasks = 0
        
        # Register handlers
        self.register_handler("execute_task", self._handle_execute_task)
    
    async def announce_availability(self, coordinator_id: str):
        """Announce availability to coordinator"""
        if not self.is_busy:
            await self.message_bus.send_message(
                self.agent_id, coordinator_id, 
                {'type': 'worker_available', 'capability': self.processing_capability}
            )
    
    async def _handle_execute_task(self, message) -> Optional[Dict[str, Any]]:
        """Handle task execution requests"""
        task_id = message.content.get('task_id')
        task_data = message.content.get('task_data', {})
        coordinator_id = message.content.get('coordinator_id')
        
        if self.is_busy:
            return {
                'type': 'task_rejected',
                'task_id': task_id,
                'reason': 'worker_busy'
            }
        
        self.is_busy = True
        self.logger.info(f"Executing task {task_id}")
        
        try:
            # Simulate task processing
            processing_time = random.uniform(1.0, 5.0)
            await asyncio.sleep(processing_time)
            
            # Generate result based on capability
            result = await self._process_task(task_data, processing_time)
            
            # Send result back to coordinator
            await self.message_bus.send_message(
                self.agent_id, coordinator_id,
                {
                    'type': 'task_result',
                    'task_id': task_id,
                    'result': result,
                    'worker_id': self.agent_id
                }
            )
            
            self.completed_tasks += 1
            self.logger.info(f"Completed task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {e}")
            
            # Send error result
            await self.message_bus.send_message(
                self.agent_id, coordinator_id,
                {
                    'type': 'task_result',
                    'task_id': task_id,
                    'result': {'error': str(e)},
                    'worker_id': self.agent_id
                }
            )
        
        finally:
            self.is_busy = False
            # Announce availability again
            await self.announce_availability(coordinator_id)
        
        return None
    
    async def _process_task(self, task_data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """Process task based on capability"""
        base_result = {
            'worker_id': self.agent_id,
            'capability': self.processing_capability,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.processing_capability == "computation":
            # Simulate computational work
            data_size = task_data.get('data_size', 1000)
            result = {
                **base_result,
                'computed_value': sum(range(data_size)),
                'operations_performed': data_size,
                'efficiency_score': random.uniform(0.8, 1.0)
            }
        
        elif self.processing_capability == "analysis":
            # Simulate data analysis
            records = task_data.get('records', 500)
            result = {
                **base_result,
                'records_analyzed': records,
                'patterns_found': random.randint(5, 15),
                'confidence_score': random.uniform(0.7, 0.95)
            }
        
        elif self.processing_capability == "validation":
            # Simulate data validation
            items = task_data.get('items', 200)
            valid_items = int(items * random.uniform(0.85, 0.98))
            result = {
                **base_result,
                'items_validated': items,
                'valid_items': valid_items,
                'validation_rate': valid_items / items,
                'errors_found': items - valid_items
            }
        
        else:
            result = {
                **base_result,
                'status': 'completed',
                'data': task_data
            }
        
        return result

class DistributedCoordinationDemo:
    def __init__(self):
        self.message_bus = MessageBus()
        self.coordinator = CoordinatorAgent("coordinator")
        self.workers: List[WorkerAgent] = []
        self.conflict_resolver = ConflictResolver("conflict_resolver", self.message_bus)
    
    async def setup(self, num_workers: int = 6):
        """Setup the distributed coordination system"""
        # Start coordinator and conflict resolver
        await self.coordinator.start()
        await self.conflict_resolver.start()
        
        # Create workers with different capabilities
        capabilities = ["computation", "analysis", "validation"]
        
        for i in range(num_workers):
            capability = capabilities[i % len(capabilities)]
            worker = WorkerAgent(f"worker_{i+1}", capability)
            self.workers.append(worker)
            await worker.start()
            
            # Announce availability
            await worker.announce_availability("coordinator")
        
        print(f"Setup complete: 1 coordinator, {len(self.workers)} workers")
    
    async def run_distributed_tasks(self, num_tasks: int = 5):
        """Run multiple distributed tasks"""
        task_ids = []
        
        # Submit tasks
        for i in range(num_tasks):
            task_data = {
                'task_name': f"distributed_task_{i+1}",
                'data_size': random.randint(500, 2000),
                'records': random.randint(100, 1000),
                'items': random.randint(50, 500),
                'priority': random.randint(1, 5)
            }
            
            task_id = await self.coordinator.submit_distributed_task(
                task_data, required_agents=3
            )
            task_ids.append(task_id)
            
            print(f"Submitted task {task_id}")
            await asyncio.sleep(1)
        
        # Monitor task completion
        completed_tasks = 0
        while completed_tasks < num_tasks:
            await asyncio.sleep(2)
            
            stats = self.coordinator.get_coordination_stats()
            completed_tasks = stats['task_status_distribution'].get('completed', 0)
            
            print(f"Progress: {completed_tasks}/{num_tasks} tasks completed")
            print(f"Stats: {stats}")
        
        print("All tasks completed!")
        
        # Print final results
        for task_id in task_ids:
            task = self.coordinator.pending_tasks[task_id]
            if 'aggregated' in task.results:
                print(f"Task {task_id} aggregated result: {task.results['aggregated']}")
    
    async def simulate_worker_failures(self):
        """Simulate worker failures and recovery"""
        print("\nSimulating worker failures...")
        
        # Randomly stop some workers
        failed_workers = random.sample(self.workers, 2)
        
        for worker in failed_workers:
            await worker.stop()
            print(f"Worker {worker.agent_id} failed")
        
        # Submit a task that requires more workers than available
        task_data = {
            'task_name': 'failure_recovery_task',
            'data_size': 1000,
            'priority': 5
        }
        
        task_id = await self.coordinator.submit_distributed_task(
            task_data, required_agents=len(self.workers)
        )
        
        print(f"Submitted high-requirement task {task_id}")
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Restart failed workers
        for worker in failed_workers:
            await worker.start()
            await worker.announce_availability("coordinator")
            print(f"Worker {worker.agent_id} recovered")
        
        # Monitor task completion
        while True:
            task = self.coordinator.pending_tasks[task_id]
            if task.status in ['completed', 'failed']:
                print(f"Recovery task status: {task.status}")
                break
            await asyncio.sleep(2)
    
    async def shutdown(self):
        """Shutdown the system"""
        for worker in self.workers:
            await worker.stop()
        
        await self.conflict_resolver.stop()
        await self.coordinator.stop()
        
        print("Distributed coordination demo shutdown complete")

# Usage example
async def main():
    demo = DistributedCoordinationDemo()
    
    try:
        await demo.setup(num_workers=8)
        await demo.run_distributed_tasks(num_tasks=6)
        await demo.simulate_worker_failures()
        
    finally:
        await demo.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Module Summary

In this module, we've explored the comprehensive world of multi-agent applications, covering:

### Key Concepts Learned

1. **Multi-Agent System Architecture**
   - Agent communication patterns and protocols
   - Message bus implementation for scalable communication
   - Agent lifecycle management and state handling

2. **Workflow Orchestration**
   - Task definition and dependency management
   - Workflow execution with error handling and retries
   - Dynamic agent assignment and load balancing

3. **Agent Discovery and Registration**
   - Service registry for agent capabilities
   - Health monitoring and heartbeat mechanisms
   - Dynamic agent discovery based on criteria

4. **Conflict Resolution and Consensus**
   - Multiple resolution strategies (voting, priority, consensus)
   - Automated conflict detection and escalation
   - Distributed decision-making mechanisms

### Practical Skills Developed

- **System Design**: Architected scalable multi-agent systems
- **Communication Patterns**: Implemented pub-sub and request-response patterns
- **Coordination**: Built distributed task coordination systems
- **Fault Tolerance**: Designed resilient systems with failure recovery
- **Monitoring**: Created comprehensive system monitoring and statistics

### Real-World Applications

- **Distributed Computing**: Task distribution across multiple processing nodes
- **Microservices Orchestration**: Service coordination in cloud environments
- **IoT Systems**: Coordinating multiple sensors and actuators
- **Trading Systems**: Multi-agent financial trading platforms
- **Smart Cities**: Coordinating traffic, utilities, and emergency services

### Next Steps

In the next module, we'll explore **Agentic Workflow Fundamentals**, where we'll dive deeper into:
- Advanced workflow patterns and design principles
- Integration with external systems and APIs
- Performance optimization and scaling strategies
- Security considerations in multi-agent systems
- Real-time monitoring and observability

The foundation you've built in multi-agent applications will be essential for understanding how to create sophisticated, production-ready agentic workflows that can handle complex business processes and integrate seamlessly with existing infrastructure.
```

### 3.2 Agent Discovery and Registration

```python
# agent_registry.py
import asyncio
import json
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from core_agent import BaseAgent, AgentCapability, AgentState
from message_bus import MessageBus

@dataclass
class AgentRegistration:
    agent_id: str
    name: str
    description: str
    capabilities: Dict[str, AgentCapability]
    endpoint: str  # Network endpoint or identifier
    last_heartbeat: datetime
    registration_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    health_status: str = "healthy"  # healthy, degraded, unhealthy
    load_metrics: Dict[str, float] = field(default_factory=dict)

class AgentRegistry(BaseAgent):
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "AgentRegistry", "Manages agent discovery and registration")
        self.message_bus = message_bus
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.capability_index: Dict[str, Set[str]] = {}  # capability -> agent_ids
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> agent_ids
        self.heartbeat_timeout = 60.0  # seconds
        
        # Register message handlers
        self.register_handler("register_agent", self._handle_register_agent)
        self.register_handler("unregister_agent", self._handle_unregister_agent)
        self.register_handler("discover_agents", self._handle_discover_agents)
        self.register_handler("get_agent_info", self._handle_get_agent_info)
        self.register_handler("update_agent_status", self._handle_update_agent_status)
        self.register_handler("heartbeat", self._handle_heartbeat)
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_stale_agents())
    
    async def register_agent(self, agent_info: Dict[str, Any]) -> bool:
        """Register a new agent"""
        try:
            agent_id = agent_info['agent_id']
            
            # Parse capabilities
            capabilities = {}
            for cap_name, cap_data in agent_info.get('capabilities', {}).items():
                capabilities[cap_name] = AgentCapability(
                    name=cap_name,
                    description=cap_data.get('description', ''),
                    input_schema=cap_data.get('input_schema', {}),
                    output_schema=cap_data.get('output_schema', {}),
                    cost=cap_data.get('cost', 1.0)
                )
            
            # Create registration
            registration = AgentRegistration(
                agent_id=agent_id,
                name=agent_info.get('name', agent_id),
                description=agent_info.get('description', ''),
                capabilities=capabilities,
                endpoint=agent_info.get('endpoint', ''),
                last_heartbeat=datetime.now(),
                registration_time=datetime.now(),
                metadata=agent_info.get('metadata', {}),
                tags=set(agent_info.get('tags', []))
            )
            
            # Store registration
            self.registered_agents[agent_id] = registration
            
            # Update indexes
            self._update_capability_index(agent_id, capabilities.keys())
            self._update_tag_index(agent_id, registration.tags)
            
            self.logger.info(f"Registered agent: {registration.name} ({agent_id})")
            
            # Notify other agents about new registration
            await self.message_bus.broadcast_message(self.agent_id, {
                'type': 'agent_registered',
                'agent_id': agent_id,
                'name': registration.name,
                'capabilities': list(capabilities.keys()),
                'tags': list(registration.tags)
            })
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        try:
            if agent_id not in self.registered_agents:
                return False
            
            registration = self.registered_agents[agent_id]
            
            # Remove from indexes
            self._remove_from_capability_index(agent_id)
            self._remove_from_tag_index(agent_id)
            
            # Remove registration
            del self.registered_agents[agent_id]
            
            self.logger.info(f"Unregistered agent: {registration.name} ({agent_id})")
            
            # Notify other agents
            await self.message_bus.broadcast_message(self.agent_id, {
                'type': 'agent_unregistered',
                'agent_id': agent_id,
                'name': registration.name
            })
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def discover_agents(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover agents based on criteria"""
        results = []
        
        for agent_id, registration in self.registered_agents.items():
            if self._matches_criteria(registration, criteria):
                results.append({
                    'agent_id': agent_id,
                    'name': registration.name,
                    'description': registration.description,
                    'capabilities': list(registration.capabilities.keys()),
                    'tags': list(registration.tags),
                    'health_status': registration.health_status,
                    'load_metrics': registration.load_metrics,
                    'endpoint': registration.endpoint,
                    'last_seen': registration.last_heartbeat.isoformat()
                })
        
        # Sort by relevance/health
        results.sort(key=lambda x: (
            x['health_status'] == 'healthy',
            -x['load_metrics'].get('cpu_usage', 0)
        ), reverse=True)
        
        return results
    
    def _matches_criteria(self, registration: AgentRegistration, criteria: Dict[str, Any]) -> bool:
        """Check if an agent matches discovery criteria"""
        # Check capabilities
        required_capabilities = criteria.get('capabilities', [])
        if required_capabilities:
            agent_capabilities = set(registration.capabilities.keys())
            if not set(required_capabilities).issubset(agent_capabilities):
                return False
        
        # Check tags
        required_tags = criteria.get('tags', [])
        if required_tags:
            if not set(required_tags).issubset(registration.tags):
                return False
        
        # Check health status
        min_health = criteria.get('min_health', 'unhealthy')
        health_levels = {'healthy': 3, 'degraded': 2, 'unhealthy': 1}
        if health_levels.get(registration.health_status, 0) < health_levels.get(min_health, 0):
            return False
        
        # Check load constraints
        max_load = criteria.get('max_cpu_load', 1.0)
        current_load = registration.load_metrics.get('cpu_usage', 0)
        if current_load > max_load:
            return False
        
        return True
    
    def _update_capability_index(self, agent_id: str, capabilities: List[str]):
        """Update the capability index"""
        for capability in capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(agent_id)
    
    def _update_tag_index(self, agent_id: str, tags: Set[str]):
        """Update the tag index"""
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(agent_id)
    
    def _remove_from_capability_index(self, agent_id: str):
        """Remove agent from capability index"""
        for capability, agent_set in self.capability_index.items():
            agent_set.discard(agent_id)
    
    def _remove_from_tag_index(self, agent_id: str):
        """Remove agent from tag index"""
        for tag, agent_set in self.tag_index.items():
            agent_set.discard(agent_id)
    
    async def _handle_register_agent(self, message) -> Dict[str, Any]:
        """Handle agent registration requests"""
        agent_info = message.content.get('agent_info', {})
        success = await self.register_agent(agent_info)
        
        return {
            'type': 'registration_response',
            'success': success,
            'agent_id': agent_info.get('agent_id')
        }
    
    async def _handle_unregister_agent(self, message) -> Dict[str, Any]:
        """Handle agent unregistration requests"""
        agent_id = message.content.get('agent_id')
        success = await self.unregister_agent(agent_id)
        
        return {
            'type': 'unregistration_response',
            'success': success,
            'agent_id': agent_id
        }
    
    async def _handle_discover_agents(self, message) -> Dict[str, Any]:
        """Handle agent discovery requests"""
        criteria = message.content.get('criteria', {})
        agents = self.discover_agents(criteria)
        
        return {
            'type': 'discovery_response',
            'agents': agents,
            'count': len(agents)
        }
    
    async def _handle_get_agent_info(self, message) -> Dict[str, Any]:
        """Handle agent info requests"""
        agent_id = message.content.get('agent_id')
        
        if agent_id not in self.registered_agents:
            return {
                'type': 'agent_info_response',
                'error': 'Agent not found'
            }
        
        registration = self.registered_agents[agent_id]
        
        return {
            'type': 'agent_info_response',
            'agent_info': {
                'agent_id': agent_id,
                'name': registration.name,
                'description': registration.description,
                'capabilities': {
                    name: {
                        'description': cap.description,
                        'input_schema': cap.input_schema,
                        'output_schema': cap.output_schema,
                        'cost': cap.cost
                    }
                    for name, cap in registration.capabilities.items()
                },
                'tags': list(registration.tags),
                'health_status': registration.health_status,
                'load_metrics': registration.load_metrics,
                'endpoint': registration.endpoint,
                'registration_time': registration.registration_time.isoformat(),
                'last_heartbeat': registration.last_heartbeat.isoformat(),
                'metadata': registration.metadata
            }
        }
    
    async def _handle_update_agent_status(self, message) -> Dict[str, Any]:
        """Handle agent status updates"""
        agent_id = message.content.get('agent_id')
        
        if agent_id not in self.registered_agents:
            return {
                'type': 'status_update_response',
                'error': 'Agent not found'
            }
        
        registration = self.registered_agents[agent_id]
        
        # Update health status
        if 'health_status' in message.content:
            registration.health_status = message.content['health_status']
        
        # Update load metrics
        if 'load_metrics' in message.content:
            registration.load_metrics.update(message.content['load_metrics'])
        
        # Update metadata
        if 'metadata' in message.content:
            registration.metadata.update(message.content['metadata'])
        
        return {
            'type': 'status_update_response',
            'success': True,
            'agent_id': agent_id
        }
    
    async def _handle_heartbeat(self, message) -> Dict[str, Any]:
        """Handle heartbeat messages"""
        agent_id = message.sender_id
        
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id].last_heartbeat = datetime.now()
            
            # Update load metrics if provided
            load_metrics = message.content.get('load_metrics', {})
            if load_metrics:
                self.registered_agents[agent_id].load_metrics.update(load_metrics)
        
        return {
            'type': 'heartbeat_ack',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _cleanup_stale_agents(self):
        """Remove agents that haven't sent heartbeats"""
        while self.state != AgentState.OFFLINE:
            try:
                current_time = datetime.now()
                stale_agents = []
                
                for agent_id, registration in self.registered_agents.items():
                    time_since_heartbeat = (current_time - registration.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_timeout:
                        stale_agents.append(agent_id)
                
                # Remove stale agents
                for agent_id in stale_agents:
                    await self.unregister_agent(agent_id)
                    self.logger.warning(f"Removed stale agent: {agent_id}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(30)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_agents = len(self.registered_agents)
        healthy_agents = sum(1 for r in self.registered_agents.values() 
                           if r.health_status == 'healthy')
        
        capability_stats = {}
        for capability, agent_set in self.capability_index.items():
            capability_stats[capability] = len(agent_set)
        
        tag_stats = {}
        for tag, agent_set in self.tag_index.items():
            tag_stats[tag] = len(agent_set)
        
        return {
            'total_agents': total_agents,
            'healthy_agents': healthy_agents,
            'degraded_agents': sum(1 for r in self.registered_agents.values() 
                                 if r.health_status == 'degraded'),
            'unhealthy_agents': sum(1 for r in self.registered_agents.values() 
                                  if r.health_status == 'unhealthy'),
            'capabilities': capability_stats,
            'tags': tag_stats,
            'average_load': sum(r.load_metrics.get('cpu_usage', 0) 
                              for r in self.registered_agents.values()) / max(total_agents, 1)
        }
```

---

## 4. Conflict Resolution and Consensus

### 4.1 Conflict Resolution System

```python
# conflict_resolution.py
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime

from core_agent import BaseAgent, Message, MessageType
from message_bus import MessageBus

class ConflictType(Enum):
    RESOURCE_CONTENTION = "resource_contention"
    TASK_ASSIGNMENT = "task_assignment"
    DATA_INCONSISTENCY = "data_inconsistency"
    PRIORITY_CONFLICT = "priority_conflict"
    CAPABILITY_OVERLAP = "capability_overlap"

class ResolutionStrategy(Enum):
    VOTING = "voting"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCING = "load_balancing"
    CONSENSUS = "consensus"
    ARBITRATION = "arbitration"

class ConflictStatus(Enum):
    DETECTED = "detected"
    ANALYZING = "analyzing"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ESCALATED = "escalated"

@dataclass
class ConflictParticipant:
    agent_id: str
    position: Dict[str, Any]  # Agent's position/preference
    priority: int = 0
    weight: float = 1.0  # Voting weight
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conflict:
    conflict_id: str
    conflict_type: ConflictType
    description: str
    participants: List[ConflictParticipant]
    context: Dict[str, Any] = field(default_factory=dict)
    status: ConflictStatus = ConflictStatus.DETECTED
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolution: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    escalation_count: int = 0

class ConflictResolver(BaseAgent):
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "ConflictResolver", "Manages conflict resolution between agents")
        self.message_bus = message_bus
        self.active_conflicts: Dict[str, Conflict] = {}
        self.resolution_strategies: Dict[ConflictType, ResolutionStrategy] = {
            ConflictType.RESOURCE_CONTENTION: ResolutionStrategy.PRIORITY_BASED,
            ConflictType.TASK_ASSIGNMENT: ResolutionStrategy.LOAD_BALANCING,
            ConflictType.DATA_INCONSISTENCY: ResolutionStrategy.CONSENSUS,
            ConflictType.PRIORITY_CONFLICT: ResolutionStrategy.VOTING,
            ConflictType.CAPABILITY_OVERLAP: ResolutionStrategy.ARBITRATION
        }
        
        # Register message handlers
        self.register_handler("report_conflict", self._handle_report_conflict)
        self.register_handler("vote_on_conflict", self._handle_vote_on_conflict)
        self.register_handler("accept_resolution", self._handle_accept_resolution)
        self.register_handler("escalate_conflict", self._handle_escalate_conflict)
        self.register_handler("get_conflict_status", self._handle_get_conflict_status)
    
    async def detect_conflict(self, conflict_data: Dict[str, Any]) -> str:
        """Detect and register a new conflict"""
        conflict_id = str(uuid.uuid4())
        
        # Parse conflict data
        conflict_type = ConflictType(conflict_data['type'])
        participants = []
        
        for p_data in conflict_data['participants']:
            participant = ConflictParticipant(
                agent_id=p_data['agent_id'],
                position=p_data['position'],
                priority=p_data.get('priority', 0),
                weight=p_data.get('weight', 1.0),
                metadata=p_data.get('metadata', {})
            )
            participants.append(participant)
        
        # Create conflict
        conflict = Conflict(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            description=conflict_data['description'],
            participants=participants,
            context=conflict_data.get('context', {}),
            resolution_strategy=self.resolution_strategies.get(conflict_type)
        )
        
        self.active_conflicts[conflict_id] = conflict
        
        self.logger.info(f"Detected conflict: {conflict.description} ({conflict_id})")
        
        # Start resolution process
        asyncio.create_task(self._resolve_conflict(conflict_id))
        
        return conflict_id
    
    async def _resolve_conflict(self, conflict_id: str):
        """Resolve a conflict using the appropriate strategy"""
        try:
            conflict = self.active_conflicts[conflict_id]
            conflict.status = ConflictStatus.ANALYZING
            
            self.logger.info(f"Resolving conflict {conflict_id} using {conflict.resolution_strategy.value}")
            
            # Apply resolution strategy
            if conflict.resolution_strategy == ResolutionStrategy.VOTING:
                resolution = await self._resolve_by_voting(conflict)
            elif conflict.resolution_strategy == ResolutionStrategy.PRIORITY_BASED:
                resolution = await self._resolve_by_priority(conflict)
            elif conflict.resolution_strategy == ResolutionStrategy.LOAD_BALANCING:
                resolution = await self._resolve_by_load_balancing(conflict)
            elif conflict.resolution_strategy == ResolutionStrategy.CONSENSUS:
                resolution = await self._resolve_by_consensus(conflict)
            elif conflict.resolution_strategy == ResolutionStrategy.ARBITRATION:
                resolution = await self._resolve_by_arbitration(conflict)
            else:
                raise ValueError(f"Unknown resolution strategy: {conflict.resolution_strategy}")
            
            # Apply resolution
            if resolution:
                conflict.resolution = resolution
                conflict.status = ConflictStatus.RESOLVED
                conflict.resolved_at = datetime.now()
                
                # Notify participants
                await self._notify_resolution(conflict)
                
                self.logger.info(f"Resolved conflict {conflict_id}: {resolution}")
            else:
                # Escalate if resolution failed
                await self._escalate_conflict(conflict_id)
        
        except Exception as e:
            self.logger.error(f"Error resolving conflict {conflict_id}: {e}")
            await self._escalate_conflict(conflict_id)
    
    async def _resolve_by_voting(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve conflict through weighted voting"""
        conflict.status = ConflictStatus.RESOLVING
        
        # Request votes from participants
        vote_requests = []
        for participant in conflict.participants:
            vote_request = Message(
                receiver_id=participant.agent_id,
                message_type=MessageType.REQUEST,
                content={
                    'type': 'vote_request',
                    'conflict_id': conflict.conflict_id,
                    'conflict_description': conflict.description,
                    'options': [p.position for p in conflict.participants],
                    'context': conflict.context
                }
            )
            vote_requests.append(self.send_message(vote_request))
        
        # Send vote requests
        await asyncio.gather(*vote_requests)
        
        # Wait for votes (with timeout)
        votes = {}
        timeout = 30.0  # 30 seconds
        start_time = datetime.now()
        
        while len(votes) < len(conflict.participants):
            if (datetime.now() - start_time).total_seconds() > timeout:
                break
            await asyncio.sleep(1)
        
        # Count votes
        vote_counts = {}
        for participant in conflict.participants:
            if participant.agent_id in votes:
                vote = votes[participant.agent_id]
                vote_key = json.dumps(vote, sort_keys=True)
                if vote_key not in vote_counts:
                    vote_counts[vote_key] = 0
                vote_counts[vote_key] += participant.weight
        
        # Determine winner
        if vote_counts:
            winning_vote = max(vote_counts.items(), key=lambda x: x[1])
            return {
                'strategy': 'voting',
                'decision': json.loads(winning_vote[0]),
                'vote_count': winning_vote[1],
                'total_votes': len(votes),
                'vote_distribution': vote_counts
            }
        
        return None
    
    async def _resolve_by_priority(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve conflict based on participant priorities"""
        # Find highest priority participant
        highest_priority = max(conflict.participants, key=lambda p: p.priority)
        
        return {
            'strategy': 'priority_based',
            'decision': highest_priority.position,
            'winner': highest_priority.agent_id,
            'priority': highest_priority.priority
        }
    
    async def _resolve_by_load_balancing(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve conflict by distributing load"""
        # Get current load information for participants
        load_info = {}
        for participant in conflict.participants:
            # Request load information
            load_request = Message(
                receiver_id=participant.agent_id,
                message_type=MessageType.REQUEST,
                content={'type': 'get_load_info'}
            )
            
            response = await self.message_bus.request_response(
                self.agent_id, participant.agent_id, 
                {'type': 'get_load_info'}, timeout=5.0
            )
            
            if response:
                load_info[participant.agent_id] = response.content.get('load', 0.0)
            else:
                load_info[participant.agent_id] = 0.5  # Default moderate load
        
        # Assign to least loaded agent
        least_loaded = min(load_info.items(), key=lambda x: x[1])
        winner = next(p for p in conflict.participants if p.agent_id == least_loaded[0])
        
        return {
            'strategy': 'load_balancing',
            'decision': winner.position,
            'winner': winner.agent_id,
            'load_distribution': load_info
        }
    
    async def _resolve_by_consensus(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve conflict through consensus building"""
        # Attempt to find common ground
        positions = [p.position for p in conflict.participants]
        
        # Simple consensus: find intersection of all positions
        if positions:
            consensus = positions[0].copy()
            for position in positions[1:]:
                # Find common keys with same values
                consensus = {
                    k: v for k, v in consensus.items()
                    if k in position and position[k] == v
                }
            
            if consensus:
                return {
                    'strategy': 'consensus',
                    'decision': consensus,
                    'agreement_level': len(consensus) / max(len(p) for p in positions)
                }
        
        return None
    
    async def _resolve_by_arbitration(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve conflict through arbitration (system decision)"""
        # System makes decision based on context and rules
        context = conflict.context
        
        # Example arbitration logic
        if 'resource_type' in context:
            # Allocate resource based on capability match
            best_match = None
            best_score = 0
            
            for participant in conflict.participants:
                score = self._calculate_capability_score(participant, context)
                if score > best_score:
                    best_score = score
                    best_match = participant
            
            if best_match:
                return {
                    'strategy': 'arbitration',
                    'decision': best_match.position,
                    'winner': best_match.agent_id,
                    'arbitration_score': best_score
                }
        
        # Default: random selection
        import random
        winner = random.choice(conflict.participants)
        return {
            'strategy': 'arbitration',
            'decision': winner.position,
            'winner': winner.agent_id,
            'method': 'random_selection'
        }
    
    def _calculate_capability_score(self, participant: ConflictParticipant, context: Dict[str, Any]) -> float:
        """Calculate capability match score for arbitration"""
        # Simple scoring based on metadata
        score = 0.0
        
        required_capabilities = context.get('required_capabilities', [])
        agent_capabilities = participant.metadata.get('capabilities', [])
        
        if required_capabilities and agent_capabilities:
            matches = set(required_capabilities) & set(agent_capabilities)
            score = len(matches) / len(required_capabilities)
        
        return score
    
    async def _notify_resolution(self, conflict: Conflict):
        """Notify all participants about conflict resolution"""
        notification_tasks = []
        
        for participant in conflict.participants:
            notification = Message(
                receiver_id=participant.agent_id,
                message_type=MessageType.BROADCAST,
                content={
                    'type': 'conflict_resolved',
                    'conflict_id': conflict.conflict_id,
                    'resolution': conflict.resolution,
                    'your_position': participant.position,
                    'accepted': participant.position == conflict.resolution.get('decision')
                }
            )
            notification_tasks.append(self.send_message(notification))
        
        await asyncio.gather(*notification_tasks)
    
    async def _escalate_conflict(self, conflict_id: str):
        """Escalate unresolved conflict"""
        conflict = self.active_conflicts[conflict_id]
        conflict.escalation_count += 1
        conflict.status = ConflictStatus.ESCALATED
        
        self.logger.warning(f"Escalating conflict {conflict_id} (escalation #{conflict.escalation_count})")
        
        # Try alternative resolution strategy
        if conflict.escalation_count <= 2:
            strategies = list(ResolutionStrategy)
            current_index = strategies.index(conflict.resolution_strategy)
            next_strategy = strategies[(current_index + 1) % len(strategies)]
            
            conflict.resolution_strategy = next_strategy
            conflict.status = ConflictStatus.DETECTED
            
            # Retry resolution
            asyncio.create_task(self._resolve_conflict(conflict_id))
        else:
            # Final escalation to human intervention
            self.logger.error(f"Conflict {conflict_id} requires human intervention")
    
    async def _handle_report_conflict(self, message) -> Dict[str, Any]:
        """Handle conflict reports"""
        try:
            conflict_data = message.content.get('conflict_data', {})
            conflict_id = await self.detect_conflict(conflict_data)
            
            return {
                'type': 'conflict_reported',
                'conflict_id': conflict_id,
                'status': 'processing'
            }
        
        except Exception as e:
            return {
                'type': 'conflict_report_failed',
                'error': str(e)
            }
    
    async def _handle_vote_on_conflict(self, message) -> Dict[str, Any]:
        """Handle voting responses"""
        conflict_id = message.content.get('conflict_id')
        vote = message.content.get('vote')
        
        # Store vote (simplified - in practice, you'd want proper vote storage)
        if not hasattr(self, '_votes'):
            self._votes = {}
        if conflict_id not in self._votes:
            self._votes[conflict_id] = {}
        
        self._votes[conflict_id][message.sender_id] = vote
        
        return {
            'type': 'vote_recorded',
            'conflict_id': conflict_id
        }
    
    async def _handle_accept_resolution(self, message) -> Dict[str, Any]:
        """Handle resolution acceptance"""
        conflict_id = message.content.get('conflict_id')
        accepted = message.content.get('accepted', True)
        
        return {
            'type': 'acceptance_recorded',
            'conflict_id': conflict_id,
            'accepted': accepted
        }
    
    async def _handle_escalate_conflict(self, message) -> Dict[str, Any]:
        """Handle escalation requests"""
        conflict_id = message.content.get('conflict_id')
        
        if conflict_id in self.active_conflicts:
            await self._escalate_conflict(conflict_id)
            return {
                'type': 'conflict_escalated',
                'conflict_id': conflict_id
            }
        else:
            return {
                'type': 'escalation_failed',
                'error': 'Conflict not found'
            }
    
    async def _handle_get_conflict_status(self, message) -> Dict[str, Any]:
        """Handle conflict status requests"""
        conflict_id = message.content.get('conflict_id')
        
        if conflict_id not in self.active_conflicts:
            return {
                'type': 'conflict_status_response',
                'error': 'Conflict not found'
            }
        
        conflict = self.active_conflicts[conflict_id]
        
        return {
            'type': 'conflict_status_response',
            'conflict_id': conflict_id,
            'status': conflict.status.value,
            'type': conflict.conflict_type.value,
            'description': conflict.description,
            'participants': [p.agent_id for p in conflict.participants],
            'resolution_strategy': conflict.resolution_strategy.value if conflict.resolution_strategy else None,
            'resolution': conflict.resolution,
            'escalation_count': conflict.escalation_count,
            'created_at': conflict.created_at.isoformat(),
            'resolved_at': conflict.resolved_at.isoformat() if conflict.resolved_at else None
        }
    
    def get_conflict_statistics(self) -> Dict[str, Any]:
        """Get conflict resolution statistics"""
        total_conflicts = len(self.active_conflicts)
        resolved_conflicts = sum(1 for c in self.active_conflicts.values() 
                               if c.status == ConflictStatus.RESOLVED)
        
        conflict_types = {}
        resolution_strategies = {}
        
        for conflict in self.active_conflicts.values():
            # Count by type
            type_name = conflict.conflict_type.value
            conflict_types[type_name] = conflict_types.get(type_name, 0) + 1
            
            # Count by resolution strategy
            if conflict.resolution_strategy:
                strategy_name = conflict.resolution_strategy.value
                resolution_strategies[strategy_name] = resolution_strategies.get(strategy_name, 0) + 1
        
        return {
            'total_conflicts': total_conflicts,
            'resolved_conflicts': resolved_conflicts,
            'pending_conflicts': total_conflicts - resolved_conflicts,
            'resolution_rate': resolved_conflicts / max(total_conflicts, 1),
            'conflict_types': conflict_types,
            'resolution_strategies': resolution_strategies
        }
```,
                        correlation_id=message.id
                    )
                    await self.send_message(error_response)
        
        except Exception as e:
            self.logger.error(f"Error handling message {message.id}: {e}")
            self.metrics['errors'] += 1
            
            # Send error response
            if message.message_type == MessageType.REQUEST:
                error_response = Message(
                    receiver_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    content={
                        'error': str(e),
                        'original_message_id': message.id
                    },
                    correlation_id=message.id
                )
                await self.send_message(error_response)
        
        finally:
            self.state = AgentState.IDLE
    
    async def _handle_ping(self, message: Message) -> Dict[str, Any]:
        """Handle ping messages"""
        return {
            'type': 'pong',
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_get_capabilities(self, message: Message) -> Dict[str, Any]:
        """Handle capability inquiry messages"""
        capabilities_info = {}
        for name, capability in self.capabilities.items():
            capabilities_info[name] = {
                'description': capability.description,
                'input_schema': capability.input_schema,
                'output_schema': capability.output_schema,
                'cost': capability.cost
            }
        
        return {
            'type': 'capabilities_response',
            'agent_id': self.agent_id,
            'capabilities': capabilities_info
        }
    
    async def _handle_get_status(self, message: Message) -> Dict[str, Any]:
        """Handle status inquiry messages"""
        uptime = datetime.now() - self.metrics['uptime_start']
        
        return {
            'type': 'status_response',
            'agent_id': self.agent_id,
            'name': self.name,
            'state': self.state.value,
            'capabilities': list(self.capabilities.keys()),
            'metrics': {
                **self.metrics,
                'uptime_seconds': uptime.total_seconds()
            }
        }
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.state != AgentState.OFFLINE:
            try:
                if self.communication_interface:
                    heartbeat = Message(
                        receiver_id="system",
                        message_type=MessageType.HEARTBEAT,
                        content={
                            'type': 'heartbeat',
                            'agent_id': self.agent_id,
                            'state': self.state.value,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    await self.send_message(heartbeat)
                
                # Wait 30 seconds before next heartbeat
                await asyncio.sleep(30)
            
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)
    
    def get_info(self) -> Dict[str, Any]:
        """Get basic information about this agent"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'description': self.description,
            'state': self.state.value,
            'capabilities': list(self.capabilities.keys()),
            'metrics': self.metrics
        }

# Example specialized agent
class DataProcessingAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "DataProcessor", "Specialized agent for data processing tasks")
        
        # Register data processing capabilities
        self.register_capability(AgentCapability(
            name="process_csv",
            description="Process CSV files and extract insights",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "operations": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["file_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "object"},
                    "insights": {"type": "array"}
                }
            },
            cost=2.0
        ))
        
        # Register message handlers
        self.register_handler("process_data", self._handle_process_data)
    
    async def _handle_process_data(self, message: Message) -> Dict[str, Any]:
        """Handle data processing requests"""
        try:
            data_info = message.content.get('data', {})
            operation = message.content.get('operation', 'analyze')
            
            # Simulate data processing
            await asyncio.sleep(1)  # Simulate processing time
            
            result = {
                'type': 'data_processing_result',
                'operation': operation,
                'result': {
                    'status': 'completed',
                    'summary': f"Processed data with operation: {operation}",
                    'records_processed': 1000,
                    'insights': [
                        "Data quality is good",
                        "No missing values detected",
                        "Distribution appears normal"
                    ]
                },
                'processing_time': 1.0
            }
            
            self.metrics['tasks_completed'] += 1
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise

# Usage example
if __name__ == "__main__":
    async def main():
        # Create a data processing agent
        agent = DataProcessingAgent("data_processor_001")
        
        # Start the agent
        await agent.start()
        
        # Simulate receiving a message
        test_message = Message(
            sender_id="test_client",
            receiver_id=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                'type': 'process_data',
                'data': {'file_path': '/data/sample.csv'},
                'operation': 'analyze'
            }
        )
        
        await agent.receive_message(test_message)
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Print agent info
        print("Agent Info:")
        print(json.dumps(agent.get_info(), indent=2, default=str))
        
        # Stop the agent
        await agent.stop()
    
    asyncio.run(main())
```

---

## 3. Agent Orchestration and Workflow Management

### 3.1 Workflow Orchestrator

```python
# workflow_orchestrator.py
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta

from core_agent import Message, MessageType, BaseAgent
from message_bus import MessageBus

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"

class WorkflowStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class TaskDefinition:
    task_id: str
    name: str
    agent_capability: str  # Required capability
    input_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 3
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)  # Conditional execution

@dataclass
class TaskExecution:
    task_id: str
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_attempts: int = 0
    execution_log: List[str] = field(default_factory=list)

@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    description: str
    tasks: List[TaskDefinition]
    global_timeout: float = 3600.0  # 1 hour default
    max_parallel_tasks: int = 10
    failure_strategy: str = "stop"  # "stop", "continue", "retry"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.CREATED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    task_executions: Dict[str, TaskExecution] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # Shared data between tasks
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

class WorkflowOrchestrator(BaseAgent):
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "WorkflowOrchestrator", "Manages and executes multi-agent workflows")
        self.message_bus = message_bus
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}  # agent_id -> capabilities
        self.agent_workload: Dict[str, int] = {}  # agent_id -> current task count
        
        # Register message handlers
        self.register_handler("execute_workflow", self._handle_execute_workflow)
        self.register_handler("task_completed", self._handle_task_completed)
        self.register_handler("task_failed", self._handle_task_failed)
        self.register_handler("get_workflow_status", self._handle_get_workflow_status)
        self.register_handler("cancel_workflow", self._handle_cancel_workflow)
        self.register_handler("agent_capabilities_updated", self._handle_agent_capabilities_updated)
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition"""
        self.workflows[workflow.workflow_id] = workflow
        self.logger.info(f"Registered workflow: {workflow.name} ({workflow.workflow_id})")
    
    async def execute_workflow(self, workflow_id: str, input_context: Dict[str, Any] = None) -> str:
        """Start executing a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        workflow_def = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        
        # Create workflow execution
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now(),
            context=input_context or {},
            task_executions={
                task.task_id: TaskExecution(task_id=task.task_id)
                for task in workflow_def.tasks
            }
        )
        
        self.executions[execution_id] = execution
        self.logger.info(f"Started workflow execution: {execution_id}")
        
        # Start execution
        asyncio.create_task(self._execute_workflow_async(execution_id))
        
        return execution_id
    
    async def _execute_workflow_async(self, execution_id: str):
        """Execute a workflow asynchronously"""
        try:
            execution = self.executions[execution_id]
            workflow_def = self.workflows[execution.workflow_id]
            
            self.logger.info(f"Executing workflow {workflow_def.name} ({execution_id})")
            
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(workflow_def.tasks)
            
            # Execute tasks in dependency order
            completed_tasks = set()
            running_tasks = set()
            
            while len(completed_tasks) < len(workflow_def.tasks):
                # Check for timeout
                if execution.start_time:
                    elapsed = (datetime.now() - execution.start_time).total_seconds()
                    if elapsed > workflow_def.global_timeout:
                        execution.status = WorkflowStatus.FAILED
                        execution.error = "Workflow timeout exceeded"
                        execution.end_time = datetime.now()
                        self.logger.error(f"Workflow {execution_id} timed out")
                        return
                
                # Find tasks ready to execute
                ready_tasks = self._get_ready_tasks(
                    workflow_def.tasks, completed_tasks, running_tasks, dependency_graph
                )
                
                # Limit parallel execution
                available_slots = workflow_def.max_parallel_tasks - len(running_tasks)
                tasks_to_start = ready_tasks[:available_slots]
                
                # Start ready tasks
                for task_def in tasks_to_start:
                    if await self._start_task(execution_id, task_def):
                        running_tasks.add(task_def.task_id)
                
                # Check for completed/failed tasks
                completed_in_iteration = set()
                for task_id in list(running_tasks):
                    task_exec = execution.task_executions[task_id]
                    if task_exec.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        running_tasks.remove(task_id)
                        if task_exec.status == TaskStatus.COMPLETED:
                            completed_tasks.add(task_id)
                            completed_in_iteration.add(task_id)
                        elif task_exec.status == TaskStatus.FAILED:
                            if workflow_def.failure_strategy == "stop":
                                execution.status = WorkflowStatus.FAILED
                                execution.error = f"Task {task_id} failed"
                                execution.end_time = datetime.now()
                                self.logger.error(f"Workflow {execution_id} failed due to task {task_id}")
                                return
                
                # If no progress, wait a bit
                if not completed_in_iteration and not tasks_to_start:
                    await asyncio.sleep(1)
            
            # All tasks completed successfully
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            
            # Calculate metrics
            total_time = (execution.end_time - execution.start_time).total_seconds()
            execution.metrics = {
                'total_execution_time': total_time,
                'task_count': len(workflow_def.tasks),
                'successful_tasks': len(completed_tasks),
                'failed_tasks': sum(1 for t in execution.task_executions.values() if t.status == TaskStatus.FAILED)
            }
            
            self.logger.info(f"Workflow {execution_id} completed successfully in {total_time:.2f}s")
        
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            self.logger.error(f"Workflow {execution_id} failed with error: {e}")
    
    def _build_dependency_graph(self, tasks: List[TaskDefinition]) -> Dict[str, Set[str]]:
        """Build a dependency graph from task definitions"""
        graph = {}
        for task in tasks:
            graph[task.task_id] = set(task.dependencies)
        return graph
    
    def _get_ready_tasks(self, tasks: List[TaskDefinition], completed: Set[str], 
                        running: Set[str], dependency_graph: Dict[str, Set[str]]) -> List[TaskDefinition]:
        """Get tasks that are ready to execute"""
        ready = []
        for task in tasks:
            if (task.task_id not in completed and 
                task.task_id not in running and 
                dependency_graph[task.task_id].issubset(completed)):
                ready.append(task)
        
        # Sort by priority (higher priority first)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready
    
    async def _start_task(self, execution_id: str, task_def: TaskDefinition) -> bool:
        """Start executing a task"""
        try:
            execution = self.executions[execution_id]
            task_exec = execution.task_executions[task_def.task_id]
            
            # Find suitable agent
            suitable_agent = await self._find_suitable_agent(task_def.agent_capability)
            if not suitable_agent:
                self.logger.warning(f"No suitable agent found for task {task_def.task_id}")
                return False
            
            # Assign task to agent
            task_exec.assigned_agent = suitable_agent
            task_exec.status = TaskStatus.RUNNING
            task_exec.start_time = datetime.now()
            task_exec.execution_log.append(f"Assigned to agent {suitable_agent}")
            
            # Update agent workload
            self.agent_workload[suitable_agent] = self.agent_workload.get(suitable_agent, 0) + 1
            
            # Prepare task input
            task_input = {
                **task_def.input_data,
                'execution_id': execution_id,
                'task_id': task_def.task_id,
                'workflow_context': execution.context
            }
            
            # Send task to agent
            task_message = Message(
                receiver_id=suitable_agent,
                message_type=MessageType.REQUEST,
                content={
                    'type': 'execute_task',
                    'capability': task_def.agent_capability,
                    'input': task_input,
                    'timeout': task_def.timeout,
                    'execution_id': execution_id,
                    'task_id': task_def.task_id
                }
            )
            
            await self.send_message(task_message)
            
            self.logger.info(f"Started task {task_def.task_id} on agent {suitable_agent}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start task {task_def.task_id}: {e}")
            task_exec.status = TaskStatus.FAILED
            task_exec.error = str(e)
            return False
    
    async def _find_suitable_agent(self, required_capability: str) -> Optional[str]:
        """Find an agent with the required capability and lowest workload"""
        suitable_agents = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            if required_capability in capabilities:
                workload = self.agent_workload.get(agent_id, 0)
                suitable_agents.append((agent_id, workload))
        
        if not suitable_agents:
            return None
        
        # Sort by workload (ascending) and return the least loaded agent
        suitable_agents.sort(key=lambda x: x[1])
        return suitable_agents[0][0]
    
    async def _handle_execute_workflow(self, message: Message) -> Dict[str, Any]:
        """Handle workflow execution requests"""
        try:
            workflow_id = message.content.get('workflow_id')
            input_context = message.content.get('context', {})
            
            execution_id = await self.execute_workflow(workflow_id, input_context)
            
            return {
                'type': 'workflow_started',
                'execution_id': execution_id,
                'workflow_id': workflow_id
            }
        
        except Exception as e:
            return {
                'type': 'workflow_start_failed',
                'error': str(e)
            }
    
    async def _handle_task_completed(self, message: Message) -> Dict[str, Any]:
        """Handle task completion notifications"""
        try:
            execution_id = message.content.get('execution_id')
            task_id = message.content.get('task_id')
            result = message.content.get('result', {})
            
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                if task_id in execution.task_executions:
                    task_exec = execution.task_executions[task_id]
                    task_exec.status = TaskStatus.COMPLETED
                    task_exec.end_time = datetime.now()
                    task_exec.result = result
                    task_exec.execution_log.append("Task completed successfully")
                    
                    # Update agent workload
                    if task_exec.assigned_agent:
                        self.agent_workload[task_exec.assigned_agent] -= 1
                    
                    # Update workflow context with task result
                    if 'context_updates' in result:
                        execution.context.update(result['context_updates'])
                    
                    self.logger.info(f"Task {task_id} completed in execution {execution_id}")
            
            return {'type': 'task_completion_acknowledged'}
        
        except Exception as e:
            self.logger.error(f"Error handling task completion: {e}")
            return {'type': 'error', 'message': str(e)}
    
    async def _handle_task_failed(self, message: Message) -> Dict[str, Any]:
        """Handle task failure notifications"""
        try:
            execution_id = message.content.get('execution_id')
            task_id = message.content.get('task_id')
            error = message.content.get('error', 'Unknown error')
            
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                if task_id in execution.task_executions:
                    task_exec = execution.task_executions[task_id]
                    
                    # Update agent workload
                    if task_exec.assigned_agent:
                        self.agent_workload[task_exec.assigned_agent] -= 1
                    
                    # Check if we should retry
                    workflow_def = self.workflows[execution.workflow_id]
                    task_def = next(t for t in workflow_def.tasks if t.task_id == task_id)
                    
                    if task_exec.retry_attempts < task_def.retry_count:
                        task_exec.retry_attempts += 1
                        task_exec.status = TaskStatus.PENDING
                        task_exec.assigned_agent = None
                        task_exec.execution_log.append(f"Retrying task (attempt {task_exec.retry_attempts})")
                        self.logger.info(f"Retrying task {task_id} (attempt {task_exec.retry_attempts})")
                    else:
                        task_exec.status = TaskStatus.FAILED
                        task_exec.end_time = datetime.now()
                        task_exec.error = error
                        task_exec.execution_log.append(f"Task failed: {error}")
                        self.logger.error(f"Task {task_id} failed permanently: {error}")
            
            return {'type': 'task_failure_acknowledged'}
        
        except Exception as e:
            self.logger.error(f"Error handling task failure: {e}")
            return {'type': 'error', 'message': str(e)}
    
    async def _handle_get_workflow_status(self, message: Message) -> Dict[str, Any]:
        """Handle workflow status requests"""
        execution_id = message.content.get('execution_id')
        
        if execution_id not in self.executions:
            return {
                'type': 'workflow_status_response',
                'error': 'Execution not found'
            }
        
        execution = self.executions[execution_id]
        
        # Calculate progress
        total_tasks = len(execution.task_executions)
        completed_tasks = sum(1 for t in execution.task_executions.values() 
                            if t.status == TaskStatus.COMPLETED)
        
        return {
            'type': 'workflow_status_response',
            'execution_id': execution_id,
            'workflow_id': execution.workflow_id,
            'status': execution.status.value,
            'progress': {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'progress_percentage': (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
            },
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'error': execution.error,
            'metrics': execution.metrics
        }
    
    async def _handle_cancel_workflow(self, message: Message) -> Dict[str, Any]:
        """Handle workflow cancellation requests"""
        execution_id = message.content.get('execution_id')
        
        if execution_id not in self.executions:
            return {
                'type': 'workflow_cancel_response',
                'error': 'Execution not found'
            }
        
        execution = self.executions[execution_id]
        execution.status = WorkflowStatus.CANCELLED
        execution.end_time = datetime.now()
        
        # Cancel running tasks
        for task_exec in execution.task_executions.values():
            if task_exec.status == TaskStatus.RUNNING:
                task_exec.status = TaskStatus.CANCELLED
                task_exec.end_time = datetime.now()
                
                # Update agent workload
                if task_exec.assigned_agent:
                    self.agent_workload[task_exec.assigned_agent] -= 1
        
        self.logger.info(f"Cancelled workflow execution {execution_id}")
        
        return {
            'type': 'workflow_cancel_response',
            'execution_id': execution_id,
            'status': 'cancelled'
        }
    
    async def _handle_agent_capabilities_updated(self, message: Message) -> Dict[str, Any]:
        """Handle agent capability updates"""
        agent_id = message.content.get('agent_id')
        capabilities = message.content.get('capabilities', [])
        
        self.agent_capabilities[agent_id] = set(capabilities)
        self.logger.info(f"Updated capabilities for agent {agent_id}: {capabilities}")
        
        return {'type': 'capabilities_update_acknowledged'}
    
    async def _monitoring_loop(self):
        """Monitor workflow executions and handle timeouts"""
        while self.state != AgentState.OFFLINE:
            try:
                current_time = datetime.now()
                
                for execution_id, execution in list(self.executions.items()):
                    if execution.status == WorkflowStatus.RUNNING:
                        # Check for workflow timeout
                        if execution.start_time:
                            elapsed = (current_time - execution.start_time).total_seconds()
                            workflow_def = self.workflows[execution.workflow_id]
                            
                            if elapsed > workflow_def.global_timeout:
                                execution.status = WorkflowStatus.FAILED
                                execution.error = "Workflow timeout exceeded"
                                execution.end_time = current_time
                                self.logger.error(f"Workflow {execution_id} timed out")
                        
                        # Check for task timeouts
                        for task_exec in execution.task_executions.values():
                            if (task_exec.status == TaskStatus.RUNNING and 
                                task_exec.start_time):
                                
                                task_elapsed = (current_time - task_exec.start_time).total_seconds()
                                workflow_def = self.workflows[execution.workflow_id]
                                task_def = next(t for t in workflow_def.tasks 
                                              if t.task_id == task_exec.task_id)
                                
                                if task_elapsed > task_def.timeout:
                                    task_exec.status = TaskStatus.FAILED
                                    task_exec.error = "Task timeout exceeded"
                                    task_exec.end_time = current_time
                                    
                                    # Update agent workload
                                    if task_exec.assigned_agent:
                                        self.agent_workload[task_exec.assigned_agent] -= 1
                                    
                                    self.logger.error(f"Task {task_exec.task_id} timed out")
                
                await asyncio.sleep(10)  # Check every 10 seconds
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def get_execution_summary(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of workflow execution"""
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        workflow_def = self.workflows[execution.workflow_id]
        
        task_summaries = []
        for task_def in workflow_def.tasks:
            task_exec = execution.task_executions[task_def.task_id]
            duration = None
            if task_exec.start_time and task_exec.end_time:
                duration = (task_exec.end_time - task_exec.start_time).total_seconds()
            
            task_summaries.append({
                'task_id': task_def.task_id,
                'name': task_def.name,
                'status': task_exec.status.value,
                'assigned_agent': task_exec.assigned_agent,
                'duration': duration,
                'retry_attempts': task_exec.retry_attempts,
                'error': task_exec.error
            })
        
        total_duration = None
        if execution.start_time and execution.end_time:
            total_duration = (execution.end_time - execution.start_time).total_seconds()
        
        return {
            'execution_id': execution_id,
            'workflow_id': execution.workflow_id,
            'workflow_name': workflow_def.name,
            'status': execution.status.value,
            'total_duration': total_duration,
            'task_count': len(workflow_def.tasks),
            'completed_tasks': sum(1 for t in execution.task_executions.values() 
                                 if t.status == TaskStatus.COMPLETED),
            'failed_tasks': sum(1 for t in execution.task_executions.values() 
                              if t.status == TaskStatus.FAILED),
            'tasks': task_summaries,
            'context': execution.context,
            'metrics': execution.metrics,
            'error': execution.error
        }
```

---

## 2. Agent Communication and Coordination

### 2.1 Message Bus Implementation

```python
# message_bus.py
import asyncio
import json
from typing import Dict, List, Set, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import logging
from datetime import datetime, timedelta

from core_agent import Message, MessageType, BaseAgent

@dataclass
class SubscriptionFilter:
    message_types: Set[MessageType] = None
    sender_patterns: Set[str] = None
    content_filters: Dict[str, any] = None

class MessageBus:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.subscriptions: Dict[str, List[SubscriptionFilter]] = defaultdict(list)
        self.message_history: List[Message] = []
        self.max_history_size = 10000
        self.logger = logging.getLogger("MessageBus")
        
        # Performance metrics
        self.metrics = {
            'total_messages': 0,
            'messages_per_type': defaultdict(int),
            'active_agents': 0,
            'failed_deliveries': 0
        }
        
        # Message routing rules
        self.routing_rules: List[Callable[[Message], Optional[str]]] = []
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the message bus"""
        self.agents[agent.agent_id] = agent
        agent.communication_interface = self
        self.metrics['active_agents'] = len(self.agents)
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the message bus"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.subscriptions:
                del self.subscriptions[agent_id]
            self.metrics['active_agents'] = len(self.agents)
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    def subscribe(self, agent_id: str, filter_criteria: SubscriptionFilter):
        """Subscribe an agent to specific message types or patterns"""
        self.subscriptions[agent_id].append(filter_criteria)
        self.logger.debug(f"Agent {agent_id} subscribed with filter: {filter_criteria}")
    
    def add_routing_rule(self, rule: Callable[[Message], Optional[str]]):
        """Add a custom routing rule"""
        self.routing_rules.append(rule)
        self.logger.debug("Added custom routing rule")
    
    async def send_message(self, message: Message):
        """Send a message through the bus"""
        try:
            self.metrics['total_messages'] += 1
            self.metrics['messages_per_type'][message.message_type.value] += 1
            
            # Add to history
            self.message_history.append(message)
            if len(self.message_history) > self.max_history_size:
                self.message_history.pop(0)
            
            self.logger.debug(f"Processing message {message.id} from {message.sender_id}")
            
            # Determine recipients
            recipients = await self._determine_recipients(message)
            
            # Deliver to recipients
            delivery_tasks = []
            for recipient_id in recipients:
                if recipient_id in self.agents:
                    task = asyncio.create_task(
                        self._deliver_message(message, recipient_id)
                    )
                    delivery_tasks.append(task)
            
            # Wait for all deliveries to complete
            if delivery_tasks:
                await asyncio.gather(*delivery_tasks, return_exceptions=True)
            
            self.logger.debug(f"Message {message.id} delivered to {len(recipients)} recipients")
        
        except Exception as e:
            self.logger.error(f"Error sending message {message.id}: {e}")
            self.metrics['failed_deliveries'] += 1
    
    async def _determine_recipients(self, message: Message) -> Set[str]:
        """Determine which agents should receive the message"""
        recipients = set()
        
        # Direct addressing
        if message.receiver_id and message.receiver_id != "*":
            if message.receiver_id == "system":
                # System messages go to all agents
                recipients.update(self.agents.keys())
            else:
                recipients.add(message.receiver_id)
        
        # Broadcast messages
        elif message.receiver_id == "*" or message.message_type == MessageType.BROADCAST:
            recipients.update(self.agents.keys())
        
        # Apply custom routing rules
        for rule in self.routing_rules:
            try:
                rule_result = rule(message)
                if rule_result:
                    recipients.add(rule_result)
            except Exception as e:
                self.logger.error(f"Error in routing rule: {e}")
        
        # Apply subscription filters
        for agent_id, filters in self.subscriptions.items():
            for filter_criteria in filters:
                if self._message_matches_filter(message, filter_criteria):
                    recipients.add(agent_id)
        
        # Remove sender from recipients (avoid self-delivery)
        recipients.discard(message.sender_id)
        
        return recipients
    
    def _message_matches_filter(self, message: Message, filter_criteria: SubscriptionFilter) -> bool:
        """Check if a message matches subscription filter criteria"""
        # Check message type filter
        if filter_criteria.message_types:
            if message.message_type not in filter_criteria.message_types:
                return False
        
        # Check sender pattern filter
        if filter_criteria.sender_patterns:
            sender_match = any(
                pattern in message.sender_id 
                for pattern in filter_criteria.sender_patterns
            )
            if not sender_match:
                return False
        
        # Check content filters
        if filter_criteria.content_filters:
            for key, expected_value in filter_criteria.content_filters.items():
                if key not in message.content or message.content[key] != expected_value:
                    return False
        
        return True
    
    async def _deliver_message(self, message: Message, recipient_id: str):
        """Deliver a message to a specific agent"""
        try:
            if recipient_id in self.agents:
                agent = self.agents[recipient_id]
                await agent.receive_message(message)
            else:
                self.logger.warning(f"Attempted to deliver message to unknown agent: {recipient_id}")
                self.metrics['failed_deliveries'] += 1
        
        except Exception as e:
            self.logger.error(f"Failed to deliver message {message.id} to {recipient_id}: {e}")
            self.metrics['failed_deliveries'] += 1
    
    async def broadcast_message(self, sender_id: str, content: Dict[str, any]):
        """Broadcast a message to all agents"""
        broadcast_msg = Message(
            sender_id=sender_id,
            receiver_id="*",
            message_type=MessageType.BROADCAST,
            content=content
        )
        await self.send_message(broadcast_msg)
    
    def get_agent_list(self) -> List[Dict[str, any]]:
        """Get list of all registered agents"""
        return [
            {
                'agent_id': agent_id,
                'name': agent.name,
                'state': agent.state.value,
                'capabilities': list(agent.capabilities.keys())
            }
            for agent_id, agent in self.agents.items()
        ]
    
    def get_metrics(self) -> Dict[str, any]:
        """Get message bus performance metrics"""
        return {
            **self.metrics,
            'message_history_size': len(self.message_history),
            'subscription_count': sum(len(filters) for filters in self.subscriptions.values())
        }
    
    async def request_response(self, sender_id: str, receiver_id: str, 
                              content: Dict[str, any], timeout: float = 30.0) -> Optional[Message]:
        """Send a request and wait for a response"""
        request_msg = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.REQUEST,
            content=content
        )
        
        # Create a future to wait for the response
        response_future = asyncio.Future()
        correlation_id = request_msg.id
        
        # Store the future for response matching
        if not hasattr(self, '_pending_requests'):
            self._pending_requests = {}
        self._pending_requests[correlation_id] = response_future
        
        # Send the request
        await self.send_message(request_msg)
        
        try:
            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        
        except asyncio.TimeoutError:
            self.logger.warning(f"Request {correlation_id} timed out")
            return None
        
        finally:
            # Clean up
            if correlation_id in self._pending_requests:
                del self._pending_requests[correlation_id]
    
    async def handle_response(self, response_message: Message):
        """Handle incoming response messages"""
        if hasattr(self, '_pending_requests') and response_message.correlation_id:
            correlation_id = response_message.correlation_id
            if correlation_id in self._pending_requests:
                future = self._pending_requests[correlation_id]
                if not future.done():
                    future.set_result(response_message)

# Usage example
if __name__ == "__main__":
    from core_agent import DataProcessingAgent
    
    async def main():
        # Create message bus
        bus = MessageBus()
        
        # Create agents
        agent1 = DataProcessingAgent("processor_1")
        agent2 = DataProcessingAgent("processor_2")
        
        # Register agents with bus
        bus.register_agent(agent1)
        bus.register_agent(agent2)
        
        # Start agents
        await agent1.start()
        await agent2.start()
        
        # Set up subscriptions
        bus.subscribe("processor_2", SubscriptionFilter(
            message_types={MessageType.BROADCAST},
            content_filters={'type': 'data_available'}
        ))
        
        # Send a broadcast message
        await bus.broadcast_message("system", {
            'type': 'data_available',
            'dataset': 'customer_data.csv',
            'priority': 'high'
        })
        
        # Send a direct request
        response = await bus.request_response(
            sender_id="system",
            receiver_id="processor_1",
            content={
                'type': 'process_data',
                'data': {'file_path': '/data/test.csv'},
                'operation': 'summarize'
            },
            timeout=10.0
        )
        
        if response:
            print("Received response:")
            print(json.dumps(response.content, indent=2))
        
        # Wait a bit for message processing
        await asyncio.sleep(3)
        
        # Print metrics
        print("\nMessage Bus Metrics:")
        print(json.dumps(bus.get_metrics(), indent=2))
        
        print("\nRegistered Agents:")
        print(json.dumps(bus.get_agent_list(), indent=2))
        
        # Stop agents
        await agent1.stop()
        await agent2.stop()
    
    asyncio.run(main())
```