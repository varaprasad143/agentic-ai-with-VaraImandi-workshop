# Module IV: Agentic Workflow Fundamentals

## Learning Objectives

By the end of this module, you will be able to:

1. **Design Advanced Workflow Patterns**: Create sophisticated workflow architectures for complex business processes
2. **Implement External System Integration**: Connect agentic workflows with APIs, databases, and third-party services
3. **Optimize Performance and Scalability**: Build high-performance workflows that scale efficiently
4. **Ensure Security and Compliance**: Implement robust security measures and compliance frameworks
5. **Monitor and Observe Workflows**: Create comprehensive monitoring and observability systems
6. **Handle Error Recovery**: Design resilient workflows with advanced error handling and recovery mechanisms

---

## 1. Advanced Workflow Patterns

### 1.1 Workflow Architecture Overview

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Advanced Agentic Workflow Architecture</text>
  
  <!-- Workflow Engine -->
  <rect x="300" y="60" width="200" height="80" rx="10" fill="#3498db" stroke="#2980b9" stroke-width="2"/>
  <text x="400" y="90" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Workflow Engine</text>
  <text x="400" y="110" text-anchor="middle" font-size="12" fill="white">Orchestration Core</text>
  <text x="400" y="125" text-anchor="middle" font-size="12" fill="white">Pattern Management</text>
  
  <!-- Pattern Types -->
  <g id="patterns">
    <!-- Sequential Pattern -->
    <rect x="50" y="180" width="120" height="60" rx="8" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
    <text x="110" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Sequential</text>
    <text x="110" y="215" text-anchor="middle" font-size="10" fill="white">Linear Execution</text>
    <text x="110" y="230" text-anchor="middle" font-size="10" fill="white">Step-by-Step</text>
    
    <!-- Parallel Pattern -->
    <rect x="190" y="180" width="120" height="60" rx="8" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
    <text x="250" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Parallel</text>
    <text x="250" y="215" text-anchor="middle" font-size="10" fill="white">Concurrent Tasks</text>
    <text x="250" y="230" text-anchor="middle" font-size="10" fill="white">Fork & Join</text>
    
    <!-- Conditional Pattern -->
    <rect x="330" y="180" width="120" height="60" rx="8" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
    <text x="390" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Conditional</text>
    <text x="390" y="215" text-anchor="middle" font-size="10" fill="white">Decision Trees</text>
    <text x="390" y="230" text-anchor="middle" font-size="10" fill="white">Branch Logic</text>
    
    <!-- Event-Driven Pattern -->
    <rect x="470" y="180" width="120" height="60" rx="8" fill="#1abc9c" stroke="#16a085" stroke-width="2"/>
    <text x="530" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Event-Driven</text>
    <text x="530" y="215" text-anchor="middle" font-size="10" fill="white">Reactive Flow</text>
    <text x="530" y="230" text-anchor="middle" font-size="10" fill="white">Trigger-Based</text>
    
    <!-- Loop Pattern -->
    <rect x="610" y="180" width="120" height="60" rx="8" fill="#34495e" stroke="#2c3e50" stroke-width="2"/>
    <text x="670" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Loop</text>
    <text x="670" y="215" text-anchor="middle" font-size="10" fill="white">Iterative Tasks</text>
    <text x="670" y="230" text-anchor="middle" font-size="10" fill="white">Repeat Logic</text>
  </g>
  
  <!-- Integration Layer -->
  <rect x="100" y="280" width="600" height="80" rx="10" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2"/>
  <text x="400" y="305" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Integration & Adaptation Layer</text>
  <text x="400" y="325" text-anchor="middle" font-size="12" fill="white">External APIs • Database Connectors • Message Queues • File Systems</text>
  <text x="400" y="345" text-anchor="middle" font-size="12" fill="white">Authentication • Rate Limiting • Circuit Breakers • Retry Logic</text>
  
  <!-- Monitoring & Observability -->
  <rect x="50" y="390" width="200" height="80" rx="10" fill="#e67e22" stroke="#d35400" stroke-width="2"/>
  <text x="150" y="415" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Monitoring</text>
  <text x="150" y="435" text-anchor="middle" font-size="10" fill="white">Metrics Collection</text>
  <text x="150" y="450" text-anchor="middle" font-size="10" fill="white">Performance Tracking</text>
  <text x="150" y="465" text-anchor="middle" font-size="10" fill="white">Health Checks</text>
  
  <!-- Security & Compliance -->
  <rect x="300" y="390" width="200" height="80" rx="10" fill="#c0392b" stroke="#a93226" stroke-width="2"/>
  <text x="400" y="415" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Security</text>
  <text x="400" y="435" text-anchor="middle" font-size="10" fill="white">Access Control</text>
  <text x="400" y="450" text-anchor="middle" font-size="10" fill="white">Data Encryption</text>
  <text x="400" y="465" text-anchor="middle" font-size="10" fill="white">Audit Logging</text>
  
  <!-- Scaling & Performance -->
  <rect x="550" y="390" width="200" height="80" rx="10" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="650" y="415" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Scaling</text>
  <text x="650" y="435" text-anchor="middle" font-size="10" fill="white">Auto-scaling</text>
  <text x="650" y="450" text-anchor="middle" font-size="10" fill="white">Load Balancing</text>
  <text x="650" y="465" text-anchor="middle" font-size="10" fill="white">Resource Optimization</text>
  
  <!-- Storage Layer -->
  <rect x="200" y="500" width="400" height="60" rx="10" fill="#8e44ad" stroke="#7d3c98" stroke-width="2"/>
  <text x="400" y="525" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Persistent Storage Layer</text>
  <text x="400" y="545" text-anchor="middle" font-size="12" fill="white">Workflow State • Execution History • Configuration • Metadata</text>
  
  <!-- Connections -->
  <g stroke="#34495e" stroke-width="2" fill="none" marker-end="url(#arrowhead)">
    <!-- Engine to Patterns -->
    <line x1="350" y1="140" x2="110" y2="180"/>
    <line x1="380" y1="140" x2="250" y2="180"/>
    <line x1="400" y1="140" x2="390" y2="180"/>
    <line x1="420" y1="140" x2="530" y2="180"/>
    <line x1="450" y1="140" x2="670" y2="180"/>
    
    <!-- Patterns to Integration -->
    <line x1="110" y1="240" x2="200" y2="280"/>
    <line x1="250" y1="240" x2="300" y2="280"/>
    <line x1="390" y1="240" x2="400" y2="280"/>
    <line x1="530" y1="240" x2="500" y2="280"/>
    <line x1="670" y1="240" x2="600" y2="280"/>
    
    <!-- Integration to Services -->
    <line x1="250" y1="360" x2="150" y2="390"/>
    <line x1="400" y1="360" x2="400" y2="390"/>
    <line x1="550" y1="360" x2="650" y2="390"/>
    
    <!-- Services to Storage -->
    <line x1="150" y1="470" x2="300" y2="500"/>
    <line x1="400" y1="470" x2="400" y2="500"/>
    <line x1="650" y1="470" x2="500" y2="500"/>
  </g>
  
  <!-- Arrow marker -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
</svg>
```

### 1.2 Core Workflow Engine Implementation

```python
# workflow_engine.py
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
from abc import ABC, abstractmethod

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class PatternType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    EVENT_DRIVEN = "event_driven"
    LOOP = "loop"
    CUSTOM = "custom"

@dataclass
class TaskDefinition:
    task_id: str
    name: str
    description: str
    pattern_type: PatternType
    handler: str  # Function or class to execute
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskExecution:
    execution_id: str
    task_id: str
    status: TaskStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    execution_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    description: str
    version: str
    tasks: List[TaskDefinition]
    global_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    task_executions: Dict[str, TaskExecution] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_context: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None

class TaskHandler(ABC):
    """Abstract base class for task handlers"""
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task with given input data and context"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data against task requirements"""
        pass
    
    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration for this task"""
        return {
            'max_retries': 3,
            'retry_delay': 1.0,
            'backoff_factor': 2.0,
            'max_delay': 60.0
        }

class WorkflowEngine:
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.task_handlers: Dict[str, TaskHandler] = {}
        self.event_listeners: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._execution_queue = asyncio.Queue()
        
    async def start(self):
        """Start the workflow engine"""
        self._running = True
        # Start background task for processing executions
        asyncio.create_task(self._process_executions())
        self.logger.info("Workflow engine started")
    
    async def stop(self):
        """Stop the workflow engine"""
        self._running = False
        self.logger.info("Workflow engine stopped")
    
    def register_workflow(self, workflow: WorkflowDefinition) -> str:
        """Register a workflow definition"""
        self.workflows[workflow.workflow_id] = workflow
        self.logger.info(f"Registered workflow: {workflow.workflow_id}")
        return workflow.workflow_id
    
    def register_task_handler(self, handler_name: str, handler: TaskHandler):
        """Register a task handler"""
        self.task_handlers[handler_name] = handler
        self.logger.info(f"Registered task handler: {handler_name}")
    
    def add_event_listener(self, event_type: str, listener: Callable):
        """Add an event listener"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(listener)
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event to all listeners"""
        if event_type in self.event_listeners:
            for listener in self.event_listeners[event_type]:
                try:
                    await listener(event_data)
                except Exception as e:
                    self.logger.error(f"Error in event listener: {e}")
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> str:
        """Start workflow execution"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        execution_id = str(uuid.uuid4())
        workflow = self.workflows[workflow_id]
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            input_data=input_data,
            start_time=datetime.now()
        )
        
        self.executions[execution_id] = execution
        
        # Queue for execution
        await self._execution_queue.put(execution_id)
        
        await self.emit_event("workflow_started", {
            'execution_id': execution_id,
            'workflow_id': workflow_id,
            'input_data': input_data
        })
        
        return execution_id
    
    async def _process_executions(self):
        """Background task to process workflow executions"""
        while self._running:
            try:
                execution_id = await asyncio.wait_for(
                    self._execution_queue.get(), timeout=1.0
                )
                await self._execute_workflow_internal(execution_id)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing execution: {e}")
    
    async def _execute_workflow_internal(self, execution_id: str):
        """Internal workflow execution logic"""
        execution = self.executions[execution_id]
        workflow = self.workflows[execution.workflow_id]
        
        try:
            execution.status = WorkflowStatus.RUNNING
            
            # Build task dependency graph
            task_graph = self._build_task_graph(workflow.tasks)
            
            # Execute tasks based on pattern and dependencies
            await self._execute_task_graph(execution, workflow, task_graph)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            
            await self.emit_event("workflow_completed", {
                'execution_id': execution_id,
                'workflow_id': execution.workflow_id,
                'output_data': execution.output_data
            })
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.error_info = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            await self.emit_event("workflow_failed", {
                'execution_id': execution_id,
                'workflow_id': execution.workflow_id,
                'error_info': execution.error_info
            })
            
            self.logger.error(f"Workflow execution failed: {e}")
    
    def _build_task_graph(self, tasks: List[TaskDefinition]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        for task in tasks:
            graph[task.task_id] = task.dependencies.copy()
        return graph
    
    async def _execute_task_graph(self, execution: WorkflowExecution, 
                                 workflow: WorkflowDefinition, 
                                 task_graph: Dict[str, List[str]]):
        """Execute tasks based on dependency graph"""
        completed_tasks = set()
        running_tasks = set()
        task_map = {task.task_id: task for task in workflow.tasks}
        
        while len(completed_tasks) < len(workflow.tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task_id, dependencies in task_graph.items():
                if (task_id not in completed_tasks and 
                    task_id not in running_tasks and
                    all(dep in completed_tasks for dep in dependencies)):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Check if we have running tasks
                if running_tasks:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # Deadlock or circular dependency
                    raise RuntimeError("Workflow deadlock detected")
            
            # Execute ready tasks based on their pattern type
            for task_id in ready_tasks:
                task_def = task_map[task_id]
                
                if task_def.pattern_type == PatternType.PARALLEL:
                    # Start parallel execution
                    running_tasks.add(task_id)
                    asyncio.create_task(
                        self._execute_task_async(execution, task_def, completed_tasks, running_tasks)
                    )
                else:
                    # Sequential execution
                    running_tasks.add(task_id)
                    await self._execute_task(execution, task_def)
                    running_tasks.remove(task_id)
                    completed_tasks.add(task_id)
            
            await asyncio.sleep(0.1)
    
    async def _execute_task_async(self, execution: WorkflowExecution, 
                                 task_def: TaskDefinition,
                                 completed_tasks: set, running_tasks: set):
        """Execute task asynchronously for parallel patterns"""
        try:
            await self._execute_task(execution, task_def)
            completed_tasks.add(task_def.task_id)
        finally:
            running_tasks.discard(task_def.task_id)
    
    async def _execute_task(self, execution: WorkflowExecution, task_def: TaskDefinition):
        """Execute a single task"""
        task_execution = TaskExecution(
            execution_id=str(uuid.uuid4()),
            task_id=task_def.task_id,
            status=TaskStatus.PENDING,
            input_data=self._prepare_task_input(execution, task_def),
            start_time=datetime.now()
        )
        
        execution.task_executions[task_def.task_id] = task_execution
        
        try:
            # Check conditions
            if not self._evaluate_conditions(task_def.conditions, execution):
                task_execution.status = TaskStatus.SKIPPED
                task_execution.end_time = datetime.now()
                return
            
            task_execution.status = TaskStatus.RUNNING
            
            # Get task handler
            if task_def.handler not in self.task_handlers:
                raise ValueError(f"Task handler {task_def.handler} not found")
            
            handler = self.task_handlers[task_def.handler]
            
            # Execute with retry logic
            result = await self._execute_with_retry(handler, task_execution, task_def)
            
            task_execution.output_data = result
            task_execution.status = TaskStatus.COMPLETED
            task_execution.end_time = datetime.now()
            
            await self.emit_event("task_completed", {
                'execution_id': execution.execution_id,
                'task_id': task_def.task_id,
                'output_data': result
            })
            
        except Exception as e:
            task_execution.status = TaskStatus.FAILED
            task_execution.end_time = datetime.now()
            task_execution.error_info = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            await self.emit_event("task_failed", {
                'execution_id': execution.execution_id,
                'task_id': task_def.task_id,
                'error_info': task_execution.error_info
            })
            
            raise
    
    async def _execute_with_retry(self, handler: TaskHandler, 
                                 task_execution: TaskExecution,
                                 task_def: TaskDefinition) -> Dict[str, Any]:
        """Execute task with retry logic"""
        retry_config = {**handler.get_retry_config(), **task_def.retry_config}
        max_retries = retry_config.get('max_retries', 3)
        retry_delay = retry_config.get('retry_delay', 1.0)
        backoff_factor = retry_config.get('backoff_factor', 2.0)
        max_delay = retry_config.get('max_delay', 60.0)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    task_execution.status = TaskStatus.RETRYING
                    task_execution.retry_count = attempt
                    
                    # Calculate delay with exponential backoff
                    delay = min(retry_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    await asyncio.sleep(delay)
                
                # Validate input
                if not handler.validate_input(task_execution.input_data):
                    raise ValueError("Invalid input data for task")
                
                # Execute with timeout
                if task_def.timeout:
                    result = await asyncio.wait_for(
                        handler.execute(task_execution.input_data, task_execution.execution_context),
                        timeout=task_def.timeout
                    )
                else:
                    result = await handler.execute(
                        task_execution.input_data, task_execution.execution_context
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Task {task_def.task_id} attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries:
                    break
        
        raise last_exception

class RestApiAdapter(BaseIntegrationAdapter):
    """Adapter for REST API integrations"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.session = None
        self.auth_headers = {}
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Setup authentication
            auth_config = self.config.auth_config
            if auth_config.get('type') == 'bearer':
                self.auth_headers['Authorization'] = f"Bearer {auth_config['token']}"
            elif auth_config.get('type') == 'api_key':
                self.auth_headers[auth_config['header']] = auth_config['key']
            elif auth_config.get('type') == 'basic':
                import base64
                credentials = f"{auth_config['username']}:{auth_config['password']}"
                encoded = base64.b64encode(credentials.encode()).decode()
                self.auth_headers['Authorization'] = f"Basic {encoded}"
            
            # Test connection
            health_result = await self.health_check()
            return health_result.get('status') == 'healthy'
            
        except Exception as e:
            self.logger.error(f"Failed to connect to REST API: {e}")
            return False
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            health_endpoint = self.config.metadata.get('health_endpoint', '/health')
            url = f"{self.config.endpoint.rstrip('/')}{health_endpoint}"
            
            async with self.session.get(url, headers=self.auth_headers) as response:
                if response.status == 200:
                    return {'status': 'healthy', 'response_time': response.headers.get('X-Response-Time')}
                else:
                    return {'status': 'unhealthy', 'status_code': response.status}
                    
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def execute_operation(self, operation: str, **kwargs) -> Any:
        """Execute REST API operation"""
        method = kwargs.get('method', 'GET').upper()
        path = kwargs.get('path', '')
        data = kwargs.get('data')
        params = kwargs.get('params')
        headers = {**self.auth_headers, **kwargs.get('headers', {})}
        
        url = f"{self.config.endpoint.rstrip('/')}/{path.lstrip('/')}"
        
        async with self.session.request(
            method=method,
            url=url,
            json=data if method in ['POST', 'PUT', 'PATCH'] else None,
            params=params,
            headers=headers
        ) as response:
            response.raise_for_status()
            
            if response.content_type == 'application/json':
                return await response.json()
            else:
                return await response.text()

class DatabaseAdapter(BaseIntegrationAdapter):
    """Adapter for database integrations"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.connection_pool = None
        self.db_type = config.metadata.get('db_type', 'postgresql')
    
    async def connect(self) -> bool:
        """Initialize database connection pool"""
        try:
            if self.db_type == 'postgresql':
                import asyncpg
                self.connection_pool = await asyncpg.create_pool(
                    self.config.endpoint,
                    min_size=1,
                    max_size=10,
                    command_timeout=self.config.timeout
                )
            elif self.db_type == 'mongodb':
                import motor.motor_asyncio
                self.connection_pool = motor.motor_asyncio.AsyncIOMotorClient(
                    self.config.endpoint,
                    serverSelectionTimeoutMS=int(self.config.timeout * 1000)
                )
            
            # Test connection
            health_result = await self.health_check()
            return health_result.get('status') == 'healthy'
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    async def disconnect(self):
        """Close database connections"""
        if self.connection_pool:
            if self.db_type == 'postgresql':
                await self.connection_pool.close()
            elif self.db_type == 'mongodb':
                self.connection_pool.close()
            self.connection_pool = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            if self.db_type == 'postgresql':
                async with self.connection_pool.acquire() as conn:
                    result = await conn.fetchval('SELECT 1')
                    return {'status': 'healthy', 'result': result}
            elif self.db_type == 'mongodb':
                await self.connection_pool.admin.command('ping')
                return {'status': 'healthy'}
                
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def execute_operation(self, operation: str, **kwargs) -> Any:
        """Execute database operation"""
        if self.db_type == 'postgresql':
            return await self._execute_postgresql_operation(operation, **kwargs)
        elif self.db_type == 'mongodb':
            return await self._execute_mongodb_operation(operation, **kwargs)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    async def _execute_postgresql_operation(self, operation: str, **kwargs) -> Any:
        """Execute PostgreSQL operation"""
        query = kwargs.get('query')
        params = kwargs.get('params', [])
        
        async with self.connection_pool.acquire() as conn:
            if operation == 'fetch':
                return await conn.fetch(query, *params)
            elif operation == 'fetchrow':
                return await conn.fetchrow(query, *params)
            elif operation == 'fetchval':
                return await conn.fetchval(query, *params)
            elif operation == 'execute':
                return await conn.execute(query, *params)
            elif operation == 'transaction':
                async with conn.transaction():
                    results = []
                    for q, p in zip(kwargs.get('queries', []), kwargs.get('params_list', [])):
                        result = await conn.execute(q, *p)
                        results.append(result)
                    return results
            else:
                raise ValueError(f"Unknown PostgreSQL operation: {operation}")
    
    async def _execute_mongodb_operation(self, operation: str, **kwargs) -> Any:
        """Execute MongoDB operation"""
        database = kwargs.get('database')
        collection = kwargs.get('collection')
        
        db = self.connection_pool[database]
        coll = db[collection]
        
        if operation == 'find':
            filter_doc = kwargs.get('filter', {})
            cursor = coll.find(filter_doc)
            return await cursor.to_list(length=kwargs.get('limit', 100))
        elif operation == 'find_one':
            filter_doc = kwargs.get('filter', {})
            return await coll.find_one(filter_doc)
        elif operation == 'insert_one':
            document = kwargs.get('document')
            result = await coll.insert_one(document)
            return str(result.inserted_id)
        elif operation == 'insert_many':
            documents = kwargs.get('documents')
            result = await coll.insert_many(documents)
            return [str(id) for id in result.inserted_ids]
        elif operation == 'update_one':
            filter_doc = kwargs.get('filter')
            update_doc = kwargs.get('update')
            result = await coll.update_one(filter_doc, update_doc)
            return result.modified_count
        elif operation == 'delete_one':
            filter_doc = kwargs.get('filter')
            result = await coll.delete_one(filter_doc)
            return result.deleted_count
        else:
            raise ValueError(f"Unknown MongoDB operation: {operation}")

class MessageQueueAdapter(BaseIntegrationAdapter):
    """Adapter for message queue integrations"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.connection = None
        self.channel = None
        self.queue_type = config.metadata.get('queue_type', 'rabbitmq')
    
    async def connect(self) -> bool:
        """Initialize message queue connection"""
        try:
            if self.queue_type == 'rabbitmq':
                import aio_pika
                self.connection = await aio_pika.connect_robust(
                    self.config.endpoint,
                    timeout=self.config.timeout
                )
                self.channel = await self.connection.channel()
            
            # Test connection
            health_result = await self.health_check()
            return health_result.get('status') == 'healthy'
            
        except Exception as e:
            self.logger.error(f"Failed to connect to message queue: {e}")
            return False
    
    async def disconnect(self):
        """Close message queue connection"""
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()
        self.connection = None
        self.channel = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check message queue health"""
        try:
            if self.queue_type == 'rabbitmq':
                # Try to declare a temporary queue
                temp_queue = await self.channel.declare_queue('', exclusive=True)
                await temp_queue.delete()
                return {'status': 'healthy'}
                
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def execute_operation(self, operation: str, **kwargs) -> Any:
        """Execute message queue operation"""
        if operation == 'publish':
            return await self._publish_message(**kwargs)
        elif operation == 'consume':
            return await self._consume_messages(**kwargs)
        elif operation == 'declare_queue':
            return await self._declare_queue(**kwargs)
        elif operation == 'declare_exchange':
            return await self._declare_exchange(**kwargs)
        else:
            raise ValueError(f"Unknown message queue operation: {operation}")
    
    async def _publish_message(self, **kwargs) -> Any:
        """Publish message to queue/exchange"""
        import aio_pika
        
        message_body = kwargs.get('message')
        queue_name = kwargs.get('queue_name')
        exchange_name = kwargs.get('exchange_name', '')
        routing_key = kwargs.get('routing_key', queue_name)
        
        message = aio_pika.Message(
            json.dumps(message_body).encode(),
            content_type='application/json'
        )
        
        if exchange_name:
            exchange = await self.channel.get_exchange(exchange_name)
            await exchange.publish(message, routing_key=routing_key)
        else:
            await self.channel.default_exchange.publish(
                message, routing_key=routing_key
            )
        
        return {'status': 'published', 'routing_key': routing_key}
    
    async def _consume_messages(self, **kwargs) -> Any:
        """Consume messages from queue"""
        queue_name = kwargs.get('queue_name')
        max_messages = kwargs.get('max_messages', 10)
        timeout = kwargs.get('timeout', 5.0)
        
        queue = await self.channel.declare_queue(queue_name)
        messages = []
        
        async def process_message(message):
            async with message.process():
                body = json.loads(message.body.decode())
                messages.append({
                    'body': body,
                    'routing_key': message.routing_key,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Set up consumer
        consumer_tag = await queue.consume(process_message)
        
        # Wait for messages or timeout
        start_time = time.time()
        while len(messages) < max_messages and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        # Cancel consumer
        await queue.cancel(consumer_tag)
        
        return {'messages': messages, 'count': len(messages)}
    
    async def _declare_queue(self, **kwargs) -> Any:
        """Declare a queue"""
        queue_name = kwargs.get('queue_name')
        durable = kwargs.get('durable', True)
        
        queue = await self.channel.declare_queue(
            queue_name, durable=durable
        )
        
        return {'queue_name': queue.name, 'message_count': queue.declaration_result.message_count}
    
    async def _declare_exchange(self, **kwargs) -> Any:
        """Declare an exchange"""
        import aio_pika
        
        exchange_name = kwargs.get('exchange_name')
        exchange_type = kwargs.get('exchange_type', 'direct')
        durable = kwargs.get('durable', True)
        
        exchange = await self.channel.declare_exchange(
            exchange_name,
            type=getattr(aio_pika.ExchangeType, exchange_type.upper()),
            durable=durable
        )
        
        return {'exchange_name': exchange.name, 'type': exchange_type}

class IntegrationManager:
    """Manager for all integration adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, BaseIntegrationAdapter] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_adapter(self, name: str, adapter: BaseIntegrationAdapter):
        """Register an integration adapter"""
        self.adapters[name] = adapter
        self.logger.info(f"Registered adapter: {name}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all registered adapters"""
        results = {}
        for name, adapter in self.adapters.items():
            try:
                results[name] = await adapter.connect()
                self.logger.info(f"Adapter {name} connection: {results[name]}")
            except Exception as e:
                results[name] = False
                self.logger.error(f"Failed to connect adapter {name}: {e}")
        return results
    
    async def disconnect_all(self):
        """Disconnect all adapters"""
        for name, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                self.logger.info(f"Disconnected adapter: {name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting adapter {name}: {e}")
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all adapters"""
        results = {}
        for name, adapter in self.adapters.items():
            try:
                results[name] = await adapter.health_check()
            except Exception as e:
                results[name] = {'status': 'unhealthy', 'error': str(e)}
        return results
    
    async def execute_operation(self, adapter_name: str, operation: str, **kwargs) -> Any:
        """Execute operation on specific adapter"""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter {adapter_name} not found")
        
        adapter = self.adapters[adapter_name]
        return await adapter.execute_with_resilience(operation, **kwargs)
    
    def get_adapter(self, name: str) -> Optional[BaseIntegrationAdapter]:
        """Get adapter by name"""
        return self.adapters.get(name)
    
    def list_adapters(self) -> List[str]:
        """List all registered adapter names"""
        return list(self.adapters.keys())
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        health_results = await self.health_check_all()
        
        healthy_count = sum(1 for result in health_results.values() 
                          if result.get('status') == 'healthy')
        total_count = len(health_results)
        
        return {
            'total_adapters': total_count,
            'healthy_adapters': healthy_count,
            'unhealthy_adapters': total_count - healthy_count,
            'overall_health': 'healthy' if healthy_count == total_count else 'degraded',
            'adapter_details': health_results,
            'timestamp': datetime.now().isoformat()
        }

### Usage Example

```python
# Example: Setting up integration adapters
async def setup_integrations():
    # Initialize integration manager
    integration_manager = IntegrationManager()
    
    # REST API adapter
    api_config = IntegrationConfig(
        name="user_service",
        endpoint="https://api.example.com",
        timeout=30.0,
        auth_config={
            'type': 'bearer',
            'token': 'your-api-token'
        },
        metadata={'health_endpoint': '/health'}
    )
    api_adapter = RestApiAdapter(api_config)
    integration_manager.register_adapter("user_service", api_adapter)
    
    # Database adapter
    db_config = IntegrationConfig(
        name="main_db",
        endpoint="postgresql://user:pass@localhost:5432/dbname",
        timeout=10.0,
        metadata={'db_type': 'postgresql'}
    )
    db_adapter = DatabaseAdapter(db_config)
    integration_manager.register_adapter("main_db", db_adapter)
    
    # Message queue adapter
    mq_config = IntegrationConfig(
        name="task_queue",
        endpoint="amqp://guest:guest@localhost:5672/",
        timeout=15.0,
        metadata={'queue_type': 'rabbitmq'}
    )
    mq_adapter = MessageQueueAdapter(mq_config)
    integration_manager.register_adapter("task_queue", mq_adapter)
    
    # Connect all adapters
    connection_results = await integration_manager.connect_all()
    print(f"Connection results: {connection_results}")
    
    # Execute operations
    try:
        # REST API call
        user_data = await integration_manager.execute_operation(
            "user_service", "api_call",
            method="GET",
            path="/users/123"
        )
        
        # Database query
        db_result = await integration_manager.execute_operation(
            "main_db", "fetch",
            query="SELECT * FROM users WHERE id = $1",
            params=[123]
        )
        
        # Message queue publish
        mq_result = await integration_manager.execute_operation(
            "task_queue", "publish",
            queue_name="user_updates",
            message={"user_id": 123, "action": "updated"}
        )
        
        print(f"Operations completed successfully")
        
    except Exception as e:
        print(f"Operation failed: {e}")
    
    finally:
        # Cleanup
        await integration_manager.disconnect_all()

# Run the example
asyncio.run(setup_integrations())
```

## Performance Optimization

### Workflow Performance Monitoring

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time
import statistics
import asyncio
from collections import defaultdict, deque

@dataclass
class PerformanceMetrics:
    """Performance metrics for workflow execution"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    task_count: int
    success_rate: float
    error_count: int
    throughput: float  # tasks per second
    latency_p50: float
    latency_p95: float
    latency_p99: float

@dataclass
class TaskMetrics:
    """Metrics for individual task execution"""
    task_id: str
    task_type: str
    start_time: float
    end_time: float
    execution_time: float
    memory_peak: float
    status: str
    error_message: Optional[str] = None
    retry_count: int = 0

class PerformanceMonitor:
    """Monitor and analyze workflow performance"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.task_metrics: deque = deque(maxlen=window_size)
        self.workflow_metrics: deque = deque(maxlen=window_size)
        self.real_time_metrics: Dict[str, Any] = {}
        self.alert_thresholds = {
            'max_execution_time': 300.0,  # 5 minutes
            'max_memory_usage': 1024 * 1024 * 1024,  # 1GB
            'min_success_rate': 0.95,  # 95%
            'max_error_rate': 0.05  # 5%
        }
        self.logger = logging.getLogger(__name__)
    
    def record_task_start(self, task_id: str, task_type: str) -> float:
        """Record task start time"""
        start_time = time.time()
        self.real_time_metrics[task_id] = {
            'start_time': start_time,
            'task_type': task_type
        }
        return start_time
    
    def record_task_completion(self, task_id: str, status: str, 
                             error_message: Optional[str] = None,
                             retry_count: int = 0):
        """Record task completion"""
        if task_id not in self.real_time_metrics:
            self.logger.warning(f"Task {task_id} not found in real-time metrics")
            return
        
        end_time = time.time()
        task_info = self.real_time_metrics.pop(task_id)
        
        metrics = TaskMetrics(
            task_id=task_id,
            task_type=task_info['task_type'],
            start_time=task_info['start_time'],
            end_time=end_time,
            execution_time=end_time - task_info['start_time'],
            memory_peak=self._get_memory_usage(),
            status=status,
            error_message=error_message,
            retry_count=retry_count
        )
        
        self.task_metrics.append(metrics)
        self._check_alerts(metrics)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    def calculate_performance_metrics(self, time_window: Optional[float] = None) -> PerformanceMetrics:
        """Calculate performance metrics for a time window"""
        current_time = time.time()
        
        # Filter metrics by time window if specified
        if time_window:
            cutoff_time = current_time - time_window
            filtered_metrics = [m for m in self.task_metrics if m.end_time >= cutoff_time]
        else:
            filtered_metrics = list(self.task_metrics)
        
        if not filtered_metrics:
            return PerformanceMetrics(
                execution_time=0, memory_usage=0, cpu_usage=0,
                task_count=0, success_rate=0, error_count=0,
                throughput=0, latency_p50=0, latency_p95=0, latency_p99=0
            )
        
        # Calculate metrics
        execution_times = [m.execution_time for m in filtered_metrics]
        successful_tasks = [m for m in filtered_metrics if m.status == 'completed']
        failed_tasks = [m for m in filtered_metrics if m.status == 'failed']
        
        total_time = max(m.end_time for m in filtered_metrics) - min(m.start_time for m in filtered_metrics)
        throughput = len(filtered_metrics) / total_time if total_time > 0 else 0
        
        return PerformanceMetrics(
            execution_time=statistics.mean(execution_times),
            memory_usage=statistics.mean([m.memory_peak for m in filtered_metrics]),
            cpu_usage=self._get_cpu_usage(),
            task_count=len(filtered_metrics),
            success_rate=len(successful_tasks) / len(filtered_metrics),
            error_count=len(failed_tasks),
            throughput=throughput,
            latency_p50=statistics.median(execution_times),
            latency_p95=self._percentile(execution_times, 0.95),
            latency_p99=self._percentile(execution_times, 0.99)
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _check_alerts(self, metrics: TaskMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.execution_time > self.alert_thresholds['max_execution_time']:
            alerts.append(f"Task {metrics.task_id} exceeded max execution time: {metrics.execution_time:.2f}s")
        
        if metrics.memory_peak > self.alert_thresholds['max_memory_usage']:
            alerts.append(f"Task {metrics.task_id} exceeded max memory usage: {metrics.memory_peak / 1024 / 1024:.2f}MB")
        
        if metrics.status == 'failed':
            alerts.append(f"Task {metrics.task_id} failed: {metrics.error_message}")
        
        for alert in alerts:
            self.logger.warning(f"PERFORMANCE ALERT: {alert}")
    
    def get_task_type_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics grouped by task type"""
        task_groups = defaultdict(list)
        
        for metrics in self.task_metrics:
            task_groups[metrics.task_type].append(metrics)
        
        statistics_by_type = {}
        for task_type, metrics_list in task_groups.items():
            execution_times = [m.execution_time for m in metrics_list]
            success_count = sum(1 for m in metrics_list if m.status == 'completed')
            
            statistics_by_type[task_type] = {
                'count': len(metrics_list),
                'avg_execution_time': statistics.mean(execution_times),
                'success_rate': success_count / len(metrics_list),
                'p95_execution_time': self._percentile(execution_times, 0.95)
            }
        
        return statistics_by_type
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        metrics = self.calculate_performance_metrics()
        task_stats = self.get_task_type_statistics()
        
        export_data = {
            'overall_metrics': {
                'execution_time': metrics.execution_time,
                'memory_usage': metrics.memory_usage,
                'cpu_usage': metrics.cpu_usage,
                'task_count': metrics.task_count,
                'success_rate': metrics.success_rate,
                'error_count': metrics.error_count,
                'throughput': metrics.throughput,
                'latency_p50': metrics.latency_p50,
                'latency_p95': metrics.latency_p95,
                'latency_p99': metrics.latency_p99
            },
            'task_type_statistics': task_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        elif format == 'csv':
            # Convert to CSV format
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(['metric', 'value'])
            
            # Write overall metrics
            for key, value in export_data['overall_metrics'].items():
                writer.writerow([key, value])
            
            return output.getvalue()
         else:
             raise ValueError(f"Unsupported export format: {format}")

class WorkflowOptimizer:
    """Optimize workflow performance based on metrics"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.optimization_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze workflow bottlenecks"""
        task_stats = self.performance_monitor.get_task_type_statistics()
        bottlenecks = []
        
        for task_type, stats in task_stats.items():
            # Identify slow tasks
            if stats['avg_execution_time'] > 30.0:  # 30 seconds threshold
                bottlenecks.append({
                    'type': 'slow_execution',
                    'task_type': task_type,
                    'avg_execution_time': stats['avg_execution_time'],
                    'recommendation': 'Consider optimizing task logic or increasing resources'
                })
            
            # Identify high failure rate tasks
            if stats['success_rate'] < 0.9:  # 90% threshold
                bottlenecks.append({
                    'type': 'high_failure_rate',
                    'task_type': task_type,
                    'success_rate': stats['success_rate'],
                    'recommendation': 'Review error handling and retry logic'
                })
            
            # Identify high variance tasks
            if stats['p95_execution_time'] > stats['avg_execution_time'] * 3:
                bottlenecks.append({
                    'type': 'high_variance',
                    'task_type': task_type,
                    'p95_time': stats['p95_execution_time'],
                    'avg_time': stats['avg_execution_time'],
                    'recommendation': 'Investigate inconsistent performance'
                })
        
        return bottlenecks
    
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """Suggest workflow optimizations"""
        metrics = self.performance_monitor.calculate_performance_metrics()
        suggestions = []
        
        # Resource optimization
        if metrics.memory_usage > 512 * 1024 * 1024:  # 512MB
            suggestions.append({
                'type': 'memory_optimization',
                'current_usage': metrics.memory_usage,
                'suggestion': 'Consider implementing memory pooling or reducing batch sizes'
            })
        
        # Throughput optimization
        if metrics.throughput < 10:  # 10 tasks per second
            suggestions.append({
                'type': 'throughput_optimization',
                'current_throughput': metrics.throughput,
                'suggestion': 'Consider increasing parallelism or optimizing task execution'
            })
        
        # Error rate optimization
        if metrics.success_rate < 0.95:  # 95% success rate
            suggestions.append({
                'type': 'reliability_optimization',
                'current_success_rate': metrics.success_rate,
                'suggestion': 'Implement better error handling and retry mechanisms'
            })
        
        return suggestions
    
    def apply_optimization(self, optimization_type: str, parameters: Dict[str, Any]) -> bool:
        """Apply optimization based on type"""
        try:
            if optimization_type == 'increase_parallelism':
                # Implementation would depend on the specific workflow engine
                self.logger.info(f"Applying parallelism optimization: {parameters}")
                
            elif optimization_type == 'adjust_batch_size':
                self.logger.info(f"Applying batch size optimization: {parameters}")
                
            elif optimization_type == 'enable_caching':
                self.logger.info(f"Applying caching optimization: {parameters}")
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': optimization_type,
                'parameters': parameters,
                'status': 'applied'
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization {optimization_type}: {e}")
            return False
```

## Security and Error Recovery

### Security Framework

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import hashlib
import hmac
import secrets
import jwt
from datetime import datetime, timedelta

@dataclass
class SecurityConfig:
    """Security configuration for workflows"""
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_encryption: bool = True
    token_expiry_minutes: int = 60
    secret_key: str = ""
    allowed_roles: List[str] = None
    rate_limit_per_minute: int = 100

@dataclass
class UserContext:
    """User context for security"""
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    session_id: str
    created_at: datetime
    expires_at: datetime

class SecurityManager:
    """Manage security for agentic workflows"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_sessions: Dict[str, UserContext] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def generate_token(self, user_context: UserContext) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user_context.user_id,
            'username': user_context.username,
            'roles': user_context.roles,
            'session_id': user_context.session_id,
            'exp': user_context.expires_at.timestamp(),
            'iat': datetime.now().timestamp()
        }
        
        return jwt.encode(payload, self.config.secret_key, algorithm='HS256')
    
    def validate_token(self, token: str) -> Optional[UserContext]:
        """Validate JWT token and return user context"""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=['HS256'])
            
            user_context = UserContext(
                user_id=payload['user_id'],
                username=payload['username'],
                roles=payload['roles'],
                permissions=[],  # Would be loaded from database
                session_id=payload['session_id'],
                created_at=datetime.fromtimestamp(payload['iat']),
                expires_at=datetime.fromtimestamp(payload['exp'])
            )
            
            # Check if session is still active
            if user_context.session_id in self.active_sessions:
                return user_context
            
            return None
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserContext]:
        """Authenticate user and create session"""
        # In real implementation, this would check against a user database
        # For demo purposes, we'll use a simple check
        if self._verify_credentials(username, password):
            session_id = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(minutes=self.config.token_expiry_minutes)
            
            user_context = UserContext(
                user_id=f"user_{username}",
                username=username,
                roles=self._get_user_roles(username),
                permissions=self._get_user_permissions(username),
                session_id=session_id,
                created_at=datetime.now(),
                expires_at=expires_at
            )
            
            self.active_sessions[session_id] = user_context
            self._log_security_event('authentication_success', user_context.user_id)
            
            return user_context
        
        self._log_security_event('authentication_failure', username)
        return None
    
    def authorize_action(self, user_context: UserContext, action: str, resource: str) -> bool:
        """Check if user is authorized to perform action on resource"""
        if not self.config.enable_authorization:
            return True
        
        # Check role-based permissions
        required_permission = f"{action}:{resource}"
        
        if required_permission in user_context.permissions:
            self._log_security_event('authorization_success', user_context.user_id, {
                'action': action,
                'resource': resource
            })
            return True
        
        # Check role-based access
        if self.config.allowed_roles:
            for role in user_context.roles:
                if role in self.config.allowed_roles:
                    return True
        
        self._log_security_event('authorization_failure', user_context.user_id, {
            'action': action,
            'resource': resource
        })
        return False
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=1)
        
        # Clean old entries
        if user_id in self.rate_limits:
            self.rate_limits[user_id] = [
                timestamp for timestamp in self.rate_limits[user_id]
                if timestamp > cutoff_time
            ]
        else:
            self.rate_limits[user_id] = []
        
        # Check current rate
        if len(self.rate_limits[user_id]) >= self.config.rate_limit_per_minute:
            self._log_security_event('rate_limit_exceeded', user_id)
            return False
        
        # Add current request
        self.rate_limits[user_id].append(current_time)
        return True
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.config.enable_encryption:
            return data
        
        # Simple encryption using Fernet (in production, use proper key management)
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        f = Fernet(key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data.decode()
    
    def decrypt_data(self, encrypted_data: str, key: str) -> str:
        """Decrypt sensitive data"""
        if not self.config.enable_encryption:
            return encrypted_data
        
        from cryptography.fernet import Fernet
        f = Fernet(key.encode())
        decrypted_data = f.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials (mock implementation)"""
        # In real implementation, this would hash the password and check against database
        mock_users = {
            'admin': 'admin_password',
            'user1': 'user1_password',
            'user2': 'user2_password'
        }
        return mock_users.get(username) == password
    
    def _get_user_roles(self, username: str) -> List[str]:
        """Get user roles (mock implementation)"""
        role_mapping = {
            'admin': ['admin', 'user'],
            'user1': ['user'],
            'user2': ['user']
        }
        return role_mapping.get(username, [])
    
    def _get_user_permissions(self, username: str) -> List[str]:
        """Get user permissions (mock implementation)"""
        permission_mapping = {
            'admin': ['read:*', 'write:*', 'execute:*', 'admin:*'],
            'user1': ['read:workflows', 'execute:workflows'],
            'user2': ['read:workflows']
        }
        return permission_mapping.get(username, [])
    
    def _log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any] = None):
        """Log security events for auditing"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'details': details or {}
        }
        self.audit_log.append(event)
        self.logger.info(f"Security event: {event_type} for user {user_id}")
    
    def get_audit_log(self, start_time: Optional[datetime] = None, 
                     end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        filtered_log = self.audit_log
        
        if start_time:
            filtered_log = [
                event for event in filtered_log
                if datetime.fromisoformat(event['timestamp']) >= start_time
            ]
        
        if end_time:
            filtered_log = [
                event for event in filtered_log
                if datetime.fromisoformat(event['timestamp']) <= end_time
            ]
        
        return filtered_log

class ErrorRecoveryManager:
    """Manage error recovery for workflows"""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_patterns: Dict[str, str] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for specific error type"""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type}")
    
    def register_error_pattern(self, pattern: str, error_type: str):
        """Register error pattern for classification"""
        self.error_patterns[pattern] = error_type
    
    def classify_error(self, error: Exception) -> str:
        """Classify error based on registered patterns"""
        error_message = str(error)
        error_class = type(error).__name__
        
        # Check registered patterns
        for pattern, error_type in self.error_patterns.items():
            if pattern in error_message or pattern in error_class:
                return error_type
        
        # Default classification based on exception type
        if isinstance(error, ConnectionError):
            return 'connection_error'
        elif isinstance(error, TimeoutError):
            return 'timeout_error'
        elif isinstance(error, ValueError):
            return 'validation_error'
        elif isinstance(error, PermissionError):
            return 'permission_error'
        else:
            return 'unknown_error'
    
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from error"""
        error_type = self.classify_error(error)
        
        if error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error_type]
                recovery_result = await strategy(error, context)
                
                self.recovery_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'error_type': error_type,
                    'error_message': str(error),
                    'context': context,
                    'recovery_successful': recovery_result,
                    'strategy_used': strategy.__name__
                })
                
                if recovery_result:
                    self.logger.info(f"Successfully recovered from {error_type}")
                else:
                    self.logger.warning(f"Recovery failed for {error_type}")
                
                return recovery_result
                
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
                return False
        
        self.logger.warning(f"No recovery strategy found for {error_type}")
        return False
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        if not self.recovery_history:
            return {'total_attempts': 0, 'success_rate': 0, 'error_types': {}}
        
        total_attempts = len(self.recovery_history)
        successful_recoveries = sum(1 for entry in self.recovery_history 
                                  if entry['recovery_successful'])
        
        error_type_stats = {}
        for entry in self.recovery_history:
            error_type = entry['error_type']
            if error_type not in error_type_stats:
                error_type_stats[error_type] = {'attempts': 0, 'successes': 0}
            
            error_type_stats[error_type]['attempts'] += 1
            if entry['recovery_successful']:
                error_type_stats[error_type]['successes'] += 1
        
        return {
            'total_attempts': total_attempts,
            'successful_recoveries': successful_recoveries,
            'success_rate': successful_recoveries / total_attempts,
            'error_types': error_type_stats
        }
```

## Hands-on Exercises

### Exercise 1: Building a Complete Agentic Workflow System

```python
# Exercise: Build a comprehensive workflow system with all components

class ComprehensiveWorkflowSystem:
    """Complete workflow system integrating all components"""
    
    def __init__(self):
        # Initialize all components
        self.workflow_engine = WorkflowEngine()
        self.integration_manager = IntegrationManager()
        self.performance_monitor = PerformanceMonitor()
        self.optimizer = WorkflowOptimizer(self.performance_monitor)
        self.security_manager = SecurityManager(SecurityConfig(
            secret_key="your-secret-key-here",
            allowed_roles=["admin", "user"]
        ))
        self.error_recovery = ErrorRecoveryManager()
        
        self._setup_recovery_strategies()
        self._setup_integrations()
    
    def _setup_recovery_strategies(self):
        """Setup error recovery strategies"""
        async def connection_recovery(error, context):
            # Attempt to reconnect
            await asyncio.sleep(5)
            return True
        
        async def timeout_recovery(error, context):
            # Increase timeout and retry
            context['timeout'] = context.get('timeout', 30) * 2
            return True
        
        self.error_recovery.register_recovery_strategy('connection_error', connection_recovery)
        self.error_recovery.register_recovery_strategy('timeout_error', timeout_recovery)
    
    async def _setup_integrations(self):
        """Setup integration adapters"""
        # Database integration
        db_config = IntegrationConfig(
            name="main_db",
            endpoint="postgresql://localhost:5432/workflow_db",
            timeout=30.0,
            metadata={'db_type': 'postgresql'}
        )
        db_adapter = DatabaseAdapter(db_config)
        self.integration_manager.register_adapter("database", db_adapter)
        
        # API integration
        api_config = IntegrationConfig(
            name="external_api",
            endpoint="https://api.example.com",
            timeout=30.0,
            auth_config={'type': 'bearer', 'token': 'api-token'}
        )
        api_adapter = RestApiAdapter(api_config)
        self.integration_manager.register_adapter("api", api_adapter)
    
    async def execute_secure_workflow(self, workflow_id: str, input_data: Dict[str, Any], 
                                    user_token: str) -> Dict[str, Any]:
        """Execute workflow with security and monitoring"""
        # Authenticate user
        user_context = self.security_manager.validate_token(user_token)
        if not user_context:
            raise PermissionError("Invalid or expired token")
        
        # Check authorization
        if not self.security_manager.authorize_action(user_context, "execute", "workflow"):
            raise PermissionError("Insufficient permissions")
        
        # Check rate limit
        if not self.security_manager.check_rate_limit(user_context.user_id):
            raise Exception("Rate limit exceeded")
        
        # Start performance monitoring
        execution_id = f"exec_{secrets.token_urlsafe(8)}"
        start_time = self.performance_monitor.record_task_start(execution_id, "workflow")
        
        try:
            # Execute workflow with error recovery
            result = await self._execute_with_recovery(workflow_id, input_data)
            
            # Record successful completion
            self.performance_monitor.record_task_completion(execution_id, "completed")
            
            return {
                'status': 'success',
                'result': result,
                'execution_id': execution_id,
                'user_id': user_context.user_id
            }
            
        except Exception as e:
            # Record failure
            self.performance_monitor.record_task_completion(
                execution_id, "failed", str(e)
            )
            
            # Attempt recovery
            recovery_successful = await self.error_recovery.attempt_recovery(
                e, {'workflow_id': workflow_id, 'user_id': user_context.user_id}
            )
            
            if recovery_successful:
                # Retry execution
                result = await self.workflow_engine.execute_workflow(workflow_id, input_data)
                self.performance_monitor.record_task_completion(execution_id, "completed")
                return {
                    'status': 'success_after_recovery',
                    'result': result,
                    'execution_id': execution_id
                }
            else:
                raise e
    
    async def _execute_with_recovery(self, workflow_id: str, input_data: Dict[str, Any]):
        """Execute workflow with built-in recovery"""
        try:
            return await self.workflow_engine.execute_workflow(workflow_id, input_data)
        except Exception as e:
            # Log error and attempt recovery
            self.error_recovery.logger.error(f"Workflow execution failed: {e}")
            
            # Check if recovery is possible
            recovery_successful = await self.error_recovery.attempt_recovery(
                e, {'workflow_id': workflow_id}
            )
            
            if recovery_successful:
                # Retry once
                return await self.workflow_engine.execute_workflow(workflow_id, input_data)
            else:
                raise e
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        # Integration health
        integration_health = await self.integration_manager.get_system_status()
        
        # Performance metrics
        performance_metrics = self.performance_monitor.calculate_performance_metrics()
        
        # Recovery statistics
        recovery_stats = self.error_recovery.get_recovery_statistics()
        
        # Optimization suggestions
        bottlenecks = self.optimizer.analyze_bottlenecks()
        suggestions = self.optimizer.suggest_optimizations()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'integration_health': integration_health,
            'performance_metrics': {
                'execution_time': performance_metrics.execution_time,
                'throughput': performance_metrics.throughput,
                'success_rate': performance_metrics.success_rate,
                'memory_usage': performance_metrics.memory_usage
            },
            'recovery_statistics': recovery_stats,
            'bottlenecks': bottlenecks,
            'optimization_suggestions': suggestions
        }

# Usage example
async def demo_comprehensive_system():
    system = ComprehensiveWorkflowSystem()
    
    # Create a sample workflow
    workflow_def = WorkflowDefinition(
        workflow_id="data_processing",
        name="Data Processing Pipeline",
        description="Process and analyze data",
        tasks=[
            TaskDefinition(
                task_id="fetch_data",
                name="Fetch Data",
                task_type="api_call",
                handler_name="api",
                config={"endpoint": "/data", "method": "GET"}
            ),
            TaskDefinition(
                task_id="process_data",
                name="Process Data",
                task_type="data_processing",
                handler_name="processor",
                dependencies=["fetch_data"]
            ),
            TaskDefinition(
                task_id="store_results",
                name="Store Results",
                task_type="database_insert",
                handler_name="database",
                dependencies=["process_data"]
            )
        ]
    )
    
    # Register workflow
    system.workflow_engine.register_workflow(workflow_def)
    
    # Authenticate user
    user_context = system.security_manager.authenticate_user("admin", "admin_password")
    if user_context:
        token = system.security_manager.generate_token(user_context)
        
        # Execute workflow
        try:
            result = await system.execute_secure_workflow(
                "data_processing",
                {"source": "api", "format": "json"},
                token
            )
            print(f"Workflow executed successfully: {result}")
            
            # Get system health
            health = await system.get_system_health()
            print(f"System health: {health}")
            
        except Exception as e:
            print(f"Workflow execution failed: {e}")

# Run the demo
# asyncio.run(demo_comprehensive_system())
```

### Exercise 2: Performance Optimization Challenge

```python
# Exercise: Optimize a slow workflow using performance monitoring

class PerformanceOptimizationChallenge:
    """Challenge to optimize workflow performance"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.optimizer = WorkflowOptimizer(self.monitor)
        self.workflow_engine = WorkflowEngine()
    
    async def simulate_slow_workflow(self):
        """Simulate a workflow with performance issues"""
        # Simulate various task types with different performance characteristics
        task_types = [
            ('slow_task', 45.0),      # Very slow task
            ('memory_heavy', 15.0),   # Memory intensive
            ('unreliable', 10.0),     # High failure rate
            ('variable', 5.0),        # High variance
            ('normal', 2.0)           # Normal task
        ]
        
        for i in range(100):  # Simulate 100 task executions
            for task_type, base_time in task_types:
                task_id = f"{task_type}_{i}"
                
                # Start monitoring
                start_time = self.monitor.record_task_start(task_id, task_type)
                
                # Simulate task execution
                await self._simulate_task_execution(task_type, base_time)
                
                # Determine success/failure
                success = self._determine_task_success(task_type)
                status = 'completed' if success else 'failed'
                error_msg = None if success else f"{task_type} simulation error"
                
                # Record completion
                self.monitor.record_task_completion(task_id, status, error_msg)
    
    async def _simulate_task_execution(self, task_type: str, base_time: float):
        """Simulate task execution with different characteristics"""
        if task_type == 'slow_task':
            # Consistently slow
            await asyncio.sleep(base_time + random.uniform(-5, 5))
        elif task_type == 'memory_heavy':
            # Simulate memory usage
            await asyncio.sleep(base_time + random.uniform(-2, 2))
        elif task_type == 'unreliable':
            # Sometimes takes much longer
            if random.random() < 0.3:  # 30% chance of being very slow
                await asyncio.sleep(base_time * 3)
            else:
                await asyncio.sleep(base_time)
        elif task_type == 'variable':
            # High variance in execution time
            variance = random.uniform(0.1, 10.0)
            await asyncio.sleep(base_time * variance)
        else:
            # Normal execution
            await asyncio.sleep(base_time + random.uniform(-0.5, 0.5))
    
    def _determine_task_success(self, task_type: str) -> bool:
        """Determine if task succeeds based on type"""
        failure_rates = {
            'slow_task': 0.05,      # 5% failure rate
            'memory_heavy': 0.08,   # 8% failure rate
            'unreliable': 0.25,     # 25% failure rate
            'variable': 0.10,       # 10% failure rate
            'normal': 0.02          # 2% failure rate
        }
        
        return random.random() > failure_rates.get(task_type, 0.05)
    
    def analyze_and_optimize(self) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations"""
        # Get overall metrics
        metrics = self.monitor.calculate_performance_metrics()
        
        # Analyze bottlenecks
        bottlenecks = self.optimizer.analyze_bottlenecks()
        
        # Get optimization suggestions
        suggestions = self.optimizer.suggest_optimizations()
        
        # Get task type statistics
        task_stats = self.monitor.get_task_type_statistics()
        
        return {
            'overall_metrics': {
                'avg_execution_time': metrics.execution_time,
                'throughput': metrics.throughput,
                'success_rate': metrics.success_rate,
                'p95_latency': metrics.latency_p95
            },
            'bottlenecks': bottlenecks,
            'optimization_suggestions': suggestions,
            'task_statistics': task_stats
        }
    
    def apply_optimizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Apply optimizations based on analysis"""
        applied_optimizations = []
        
        # Apply optimizations based on bottlenecks
        for bottleneck in analysis['bottlenecks']:
            if bottleneck['type'] == 'slow_execution':
                # Increase parallelism for slow tasks
                self.optimizer.apply_optimization('increase_parallelism', {
                    'task_type': bottleneck['task_type'],
                    'parallel_workers': 4
                })
                applied_optimizations.append(f"Increased parallelism for {bottleneck['task_type']}")
            
            elif bottleneck['type'] == 'high_failure_rate':
                # Implement better retry logic
                self.optimizer.apply_optimization('improve_retry_logic', {
                    'task_type': bottleneck['task_type'],
                    'max_retries': 5,
                    'backoff_factor': 2.0
                })
                applied_optimizations.append(f"Improved retry logic for {bottleneck['task_type']}")
            
            elif bottleneck['type'] == 'high_variance':
                # Implement caching
                self.optimizer.apply_optimization('enable_caching', {
                    'task_type': bottleneck['task_type'],
                    'cache_ttl': 300
                })
                applied_optimizations.append(f"Enabled caching for {bottleneck['task_type']}")
        
        return applied_optimizations

# Run the optimization challenge
async def run_optimization_challenge():
    challenge = PerformanceOptimizationChallenge()
    
    print("Simulating slow workflow...")
    await challenge.simulate_slow_workflow()
    
    print("Analyzing performance...")
    analysis = challenge.analyze_and_optimize()
    
    print("\nPerformance Analysis:")
    print(f"Average execution time: {analysis['overall_metrics']['avg_execution_time']:.2f}s")
    print(f"Throughput: {analysis['overall_metrics']['throughput']:.2f} tasks/sec")
    print(f"Success rate: {analysis['overall_metrics']['success_rate']:.2%}")
    print(f"P95 latency: {analysis['overall_metrics']['p95_latency']:.2f}s")
    
    print("\nBottlenecks found:")
    for bottleneck in analysis['bottlenecks']:
        print(f"- {bottleneck['type']}: {bottleneck['task_type']} - {bottleneck['recommendation']}")
    
    print("\nApplying optimizations...")
    optimizations = challenge.apply_optimizations(analysis)
    for opt in optimizations:
        print(f"- {opt}")

# Run the challenge
# asyncio.run(run_optimization_challenge())
```

### Exercise 3: Security and Error Recovery Implementation

```python
# Exercise: Implement comprehensive security and error recovery

class SecurityErrorRecoveryExercise:
    """Exercise for implementing security and error recovery"""
    
    def __init__(self):
        self.security_manager = SecurityManager(SecurityConfig(
            secret_key="exercise-secret-key",
            allowed_roles=["admin", "operator", "viewer"],
            rate_limit_per_minute=50
        ))
        self.error_recovery = ErrorRecoveryManager()
        self.workflow_engine = WorkflowEngine()
        
        self._setup_error_recovery()
    
    def _setup_error_recovery(self):
        """Setup comprehensive error recovery strategies"""
        
        async def network_error_recovery(error, context):
            """Recover from network errors"""
            print(f"Attempting network error recovery: {error}")
            # Simulate network recovery
            await asyncio.sleep(2)
            return random.random() > 0.3  # 70% success rate
        
        async def database_error_recovery(error, context):
            """Recover from database errors"""
            print(f"Attempting database error recovery: {error}")
            # Simulate database reconnection
            await asyncio.sleep(3)
            return random.random() > 0.2  # 80% success rate
        
        async def validation_error_recovery(error, context):
            """Recover from validation errors"""
            print(f"Attempting validation error recovery: {error}")
            # Simulate data correction
            if 'data' in context:
                context['data'] = self._sanitize_data(context['data'])
                return True
            return False
        
        async def timeout_error_recovery(error, context):
            """Recover from timeout errors"""
            print(f"Attempting timeout error recovery: {error}")
            # Increase timeout and retry
            context['timeout'] = context.get('timeout', 30) * 1.5
            return True
        
        # Register recovery strategies
        self.error_recovery.register_recovery_strategy('connection_error', network_error_recovery)
        self.error_recovery.register_recovery_strategy('database_error', database_error_recovery)
        self.error_recovery.register_recovery_strategy('validation_error', validation_error_recovery)
        self.error_recovery.register_recovery_strategy('timeout_error', timeout_error_recovery)
        
        # Register error patterns
        self.error_recovery.register_error_pattern('connection refused', 'connection_error')
        self.error_recovery.register_error_pattern('database', 'database_error')
        self.error_recovery.register_error_pattern('invalid', 'validation_error')
        self.error_recovery.register_error_pattern('timeout', 'timeout_error')
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data to fix validation errors"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Remove special characters
                sanitized[key] = ''.join(c for c in value if c.isalnum() or c.isspace())
            elif isinstance(value, (int, float)):
                # Ensure positive values
                sanitized[key] = abs(value)
            else:
                sanitized[key] = value
        return sanitized
    
    async def simulate_secure_workflow_execution(self, username: str, password: str, 
                                               workflow_data: Dict[str, Any]):
        """Simulate secure workflow execution with error recovery"""
        
        # Step 1: Authentication
        print(f"Authenticating user: {username}")
        user_context = self.security_manager.authenticate_user(username, password)
        
        if not user_context:
            print("Authentication failed!")
            return {'status': 'authentication_failed'}
        
        print(f"User authenticated: {user_context.username} with roles {user_context.roles}")
        
        # Step 2: Authorization
        if not self.security_manager.authorize_action(user_context, "execute", "workflow"):
            print("Authorization failed!")
            return {'status': 'authorization_failed'}
        
        # Step 3: Rate limiting
        if not self.security_manager.check_rate_limit(user_context.user_id):
            print("Rate limit exceeded!")
            return {'status': 'rate_limit_exceeded'}
        
        # Step 4: Execute workflow with error recovery
        execution_results = []
        
        for i in range(5):  # Simulate 5 workflow steps
            step_name = f"step_{i+1}"
            print(f"\nExecuting {step_name}...")
            
            try:
                # Simulate step execution with potential errors
                result = await self._simulate_workflow_step(step_name, workflow_data)
                execution_results.append({
                    'step': step_name,
                    'status': 'success',
                    'result': result
                })
                print(f"{step_name} completed successfully")
                
            except Exception as e:
                print(f"{step_name} failed with error: {e}")
                
                # Attempt error recovery
                recovery_context = {
                    'step': step_name,
                    'data': workflow_data,
                    'user_id': user_context.user_id
                }
                
                recovery_successful = await self.error_recovery.attempt_recovery(e, recovery_context)
                
                if recovery_successful:
                    print(f"Recovery successful for {step_name}, retrying...")
                    try:
                        # Retry the step
                        result = await self._simulate_workflow_step(step_name, recovery_context.get('data', workflow_data))
                        execution_results.append({
                            'step': step_name,
                            'status': 'success_after_recovery',
                            'result': result
                        })
                        print(f"{step_name} completed after recovery")
                    except Exception as retry_error:
                        execution_results.append({
                            'step': step_name,
                            'status': 'failed_after_recovery',
                            'error': str(retry_error)
                        })
                        print(f"{step_name} failed even after recovery")
                else:
                    execution_results.append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
                    print(f"Recovery failed for {step_name}")
        
        # Step 5: Generate execution report
        successful_steps = sum(1 for result in execution_results 
                             if result['status'] in ['success', 'success_after_recovery'])
        
        return {
            'status': 'completed',
            'user': user_context.username,
            'total_steps': len(execution_results),
            'successful_steps': successful_steps,
            'success_rate': successful_steps / len(execution_results),
            'execution_results': execution_results,
            'recovery_stats': self.error_recovery.get_recovery_statistics()
        }
    
    async def _simulate_workflow_step(self, step_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a workflow step that might fail"""
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Simulate different types of errors
        error_chance = random.random()
        
        if error_chance < 0.1:  # 10% chance of connection error
            raise ConnectionError("Connection refused by remote server")
        elif error_chance < 0.15:  # 5% chance of database error
            raise Exception("Database connection lost")
        elif error_chance < 0.20:  # 5% chance of validation error
            raise ValueError("Invalid data format detected")
        elif error_chance < 0.25:  # 5% chance of timeout error
            raise TimeoutError("Operation timeout exceeded")
        
        # Successful execution
        return {
            'step': step_name,
            'processed_data': len(str(data)),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        audit_log = self.security_manager.get_audit_log()
        recovery_stats = self.error_recovery.get_recovery_statistics()
        
        # Analyze security events
        event_types = {}
        for event in audit_log:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            'security_events': {
                'total_events': len(audit_log),
                'event_breakdown': event_types,
                'recent_events': audit_log[-10:]  # Last 10 events
            },
            'error_recovery': recovery_stats,
            'active_sessions': len(self.security_manager.active_sessions),
            'timestamp': datetime.now().isoformat()
        }

# Run the security and error recovery exercise
async def run_security_exercise():
    exercise = SecurityErrorRecoveryExercise()
    
    # Test different user scenarios
    test_scenarios = [
        ('admin', 'admin_password', {'data_type': 'sensitive', 'volume': 1000}),
        ('user1', 'user1_password', {'data_type': 'normal', 'volume': 500}),
        ('invalid_user', 'wrong_password', {'data_type': 'test', 'volume': 100})
    ]
    
    for username, password, workflow_data in test_scenarios:
        print(f"\n{'='*50}")
        print(f"Testing scenario: {username}")
        print(f"{'='*50}")
        
        result = await exercise.simulate_secure_workflow_execution(
            username, password, workflow_data
        )
        
        print(f"\nExecution result: {result['status']}")
        if 'success_rate' in result:
            print(f"Success rate: {result['success_rate']:.2%}")
    
    # Generate security report
    print(f"\n{'='*50}")
    print("Security Report")
    print(f"{'='*50}")
    
    report = exercise.generate_security_report()
    print(f"Total security events: {report['security_events']['total_events']}")
    print(f"Event breakdown: {report['security_events']['event_breakdown']}")
    print(f"Error recovery stats: {report['error_recovery']}")
    print(f"Active sessions: {report['active_sessions']}")

# Run the exercise
# asyncio.run(run_security_exercise())
```

## Module Summary

In this module, we've covered the fundamental concepts and practical implementation of agentic workflow systems:

### Key Concepts Learned

1. **Advanced Workflow Patterns**
   - Sequential, parallel, conditional, event-driven, and loop patterns
   - Pattern-specific task handlers and execution strategies
   - Workflow orchestration and dependency management

2. **External System Integration**
   - Circuit breaker pattern for resilience
   - Rate limiting and caching mechanisms
   - Adapter pattern for different system types (REST APIs, databases, message queues)
   - Integration management and health monitoring

3. **Performance Optimization**
   - Real-time performance monitoring and metrics collection
   - Bottleneck analysis and optimization suggestions
   - Resource usage tracking and alerting
   - Performance-driven workflow optimization

4. **Security Framework**
   - Authentication and authorization mechanisms
   - JWT token management and session handling
   - Rate limiting and audit logging
   - Data encryption and security best practices

5. **Error Recovery**
   - Error classification and pattern matching
   - Recovery strategy registration and execution
   - Automatic retry mechanisms with backoff
   - Recovery statistics and success tracking

### Practical Skills Developed

- Building resilient workflow engines with comprehensive error handling
- Implementing secure authentication and authorization systems
- Creating performance monitoring and optimization frameworks
- Designing adapter patterns for external system integration
- Developing automated error recovery mechanisms

### Real-world Applications

- **Enterprise Automation**: Building robust automation pipelines for business processes
- **Data Processing**: Creating scalable data processing workflows with monitoring
- **API Orchestration**: Coordinating multiple API calls with error recovery
- **DevOps Pipelines**: Implementing CI/CD workflows with security and monitoring
- **Microservices Coordination**: Managing complex microservice interactions

### Next Steps

With the foundation from all four modules, you're now equipped to:

1. **Build Production Systems**: Create enterprise-grade agentic applications
2. **Scale Applications**: Implement distributed and high-performance systems
3. **Ensure Security**: Apply comprehensive security frameworks
4. **Monitor and Optimize**: Implement observability and optimization strategies
5. **Handle Complexity**: Manage complex multi-agent and workflow systems

Continue practicing with real-world projects and explore advanced topics like distributed computing, machine learning integration, and cloud-native deployments to further enhance your agentic AI application development skills.
    
    def _prepare_task_input(self, execution: WorkflowExecution, task_def: TaskDefinition) -> Dict[str, Any]:
        """Prepare input data for task execution"""
        input_data = execution.input_data.copy()
        
        # Add outputs from dependent tasks
        for dep_task_id in task_def.dependencies:
            if dep_task_id in execution.task_executions:
                dep_execution = execution.task_executions[dep_task_id]
                if dep_execution.output_data:
                    input_data[f"{dep_task_id}_output"] = dep_execution.output_data
        
        return input_data
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], execution: WorkflowExecution) -> bool:
        """Evaluate task execution conditions"""
        if not conditions:
            return True
        
        # Simple condition evaluation - can be extended
        for condition_type, condition_value in conditions.items():
            if condition_type == "always":
                return condition_value
            elif condition_type == "on_success":
                # Check if specified tasks completed successfully
                for task_id in condition_value:
                    if (task_id not in execution.task_executions or 
                        execution.task_executions[task_id].status != TaskStatus.COMPLETED):
                        return False
            elif condition_type == "on_failure":
                # Check if specified tasks failed
                for task_id in condition_value:
                    if (task_id not in execution.task_executions or 
                        execution.task_executions[task_id].status != TaskStatus.FAILED):
                        return False
        
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status"""
        return self.executions.get(execution_id)
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow engine statistics"""
        status_counts = {}
        for execution in self.executions.values():
            status = execution.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_workflows': len(self.workflows),
            'total_executions': len(self.executions),
            'execution_status_distribution': status_counts,
            'registered_handlers': len(self.task_handlers),
            'event_listeners': {event: len(listeners) for event, listeners in self.event_listeners.items()}
        }
```

### 1.3 Pattern-Specific Implementations

```python
# workflow_patterns.py
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from workflow_engine import TaskHandler, WorkflowEngine, TaskDefinition, PatternType

class SequentialTaskHandler(TaskHandler):
    """Handler for sequential task execution"""
    
    def __init__(self, steps: List[Callable]):
        self.steps = steps
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute steps sequentially"""
        result = input_data.copy()
        
        for i, step in enumerate(self.steps):
            step_result = await step(result, context)
            result[f'step_{i}_output'] = step_result
            result.update(step_result if isinstance(step_result, dict) else {'result': step_result})
        
        return result
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict)

class ParallelTaskHandler(TaskHandler):
    """Handler for parallel task execution"""
    
    def __init__(self, parallel_tasks: List[Callable]):
        self.parallel_tasks = parallel_tasks
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks in parallel"""
        # Create tasks for parallel execution
        tasks = []
        for i, task_func in enumerate(self.parallel_tasks):
            task = asyncio.create_task(task_func(input_data.copy(), context))
            tasks.append((i, task))
        
        # Wait for all tasks to complete
        results = {}
        for i, task in tasks:
            try:
                result = await task
                results[f'parallel_task_{i}'] = result
            except Exception as e:
                results[f'parallel_task_{i}'] = {'error': str(e)}
        
        return {
            'parallel_results': results,
            'completed_tasks': len([r for r in results.values() if 'error' not in r]),
            'failed_tasks': len([r for r in results.values() if 'error' in r])
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict)

class ConditionalTaskHandler(TaskHandler):
    """Handler for conditional task execution"""
    
    def __init__(self, condition_func: Callable, true_handler: Callable, false_handler: Optional[Callable] = None):
        self.condition_func = condition_func
        self.true_handler = true_handler
        self.false_handler = false_handler
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute based on condition evaluation"""
        condition_result = await self.condition_func(input_data, context)
        
        if condition_result:
            result = await self.true_handler(input_data, context)
            return {
                'condition_result': True,
                'executed_branch': 'true',
                'result': result
            }
        elif self.false_handler:
            result = await self.false_handler(input_data, context)
            return {
                'condition_result': False,
                'executed_branch': 'false',
                'result': result
            }
        else:
            return {
                'condition_result': False,
                'executed_branch': 'none',
                'result': None
            }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict)

class EventDrivenTaskHandler(TaskHandler):
    """Handler for event-driven task execution"""
    
    def __init__(self, event_processor: Callable, timeout: float = 30.0):
        self.event_processor = event_processor
        self.timeout = timeout
        self.event_queue = asyncio.Queue()
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute based on incoming events"""
        start_time = datetime.now()
        processed_events = []
        
        # Start event processing
        processing_task = asyncio.create_task(
            self._process_events(input_data, context, processed_events)
        )
        
        try:
            # Wait for completion or timeout
            await asyncio.wait_for(processing_task, timeout=self.timeout)
        except asyncio.TimeoutError:
            processing_task.cancel()
        
        end_time = datetime.now()
        
        return {
            'processed_events': processed_events,
            'processing_time': (end_time - start_time).total_seconds(),
            'timeout_reached': not processing_task.done() or processing_task.cancelled()
        }
    
    async def _process_events(self, input_data: Dict[str, Any], context: Dict[str, Any], processed_events: List):
        """Process events from the queue"""
        while True:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                result = await self.event_processor(event, input_data, context)
                processed_events.append({
                    'event': event,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Check if processing should stop
                if result and result.get('stop_processing'):
                    break
                    
            except asyncio.TimeoutError:
                continue
    
    async def send_event(self, event: Dict[str, Any]):
        """Send an event to the handler"""
        await self.event_queue.put(event)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict)

class LoopTaskHandler(TaskHandler):
    """Handler for loop-based task execution"""
    
    def __init__(self, loop_body: Callable, condition_func: Callable, max_iterations: int = 100):
        self.loop_body = loop_body
        self.condition_func = condition_func
        self.max_iterations = max_iterations
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute loop until condition is met"""
        iteration_results = []
        current_data = input_data.copy()
        iteration = 0
        
        while iteration < self.max_iterations:
            # Check loop condition
            should_continue = await self.condition_func(current_data, context, iteration)
            if not should_continue:
                break
            
            # Execute loop body
            iteration_result = await self.loop_body(current_data, context, iteration)
            iteration_results.append({
                'iteration': iteration,
                'result': iteration_result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update data for next iteration
            if isinstance(iteration_result, dict):
                current_data.update(iteration_result)
            
            iteration += 1
        
        return {
            'total_iterations': iteration,
            'max_iterations_reached': iteration >= self.max_iterations,
            'iteration_results': iteration_results,
            'final_data': current_data
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict)

# Example usage and pattern implementations
class WorkflowPatternExamples:
    def __init__(self, engine: WorkflowEngine):
        self.engine = engine
        self._register_example_handlers()
    
    def _register_example_handlers(self):
        """Register example task handlers"""
        
        # Sequential pattern example
        async def data_validation(data, context):
            await asyncio.sleep(0.5)  # Simulate processing
            return {'validation_status': 'passed', 'validated_records': len(data.get('records', []))}
        
        async def data_transformation(data, context):
            await asyncio.sleep(1.0)
            return {'transformed_records': data.get('validated_records', 0) * 2}
        
        async def data_storage(data, context):
            await asyncio.sleep(0.3)
            return {'stored_records': data.get('transformed_records', 0), 'storage_location': 'database'}
        
        sequential_handler = SequentialTaskHandler([data_validation, data_transformation, data_storage])
        self.engine.register_task_handler('data_pipeline', sequential_handler)
        
        # Parallel pattern example
        async def image_resize(data, context):
            await asyncio.sleep(2.0)
            return {'resized_images': data.get('image_count', 0), 'format': 'thumbnail'}
        
        async def image_compress(data, context):
            await asyncio.sleep(1.5)
            return {'compressed_images': data.get('image_count', 0), 'compression_ratio': 0.7}
        
        async def image_watermark(data, context):
            await asyncio.sleep(1.0)
            return {'watermarked_images': data.get('image_count', 0), 'watermark': 'company_logo'}
        
        parallel_handler = ParallelTaskHandler([image_resize, image_compress, image_watermark])
        self.engine.register_task_handler('image_processing', parallel_handler)
        
        # Conditional pattern example
        async def check_data_quality(data, context):
            quality_score = data.get('quality_score', 0.5)
            return quality_score > 0.8
        
        async def high_quality_processing(data, context):
            await asyncio.sleep(1.0)
            return {'processing_type': 'premium', 'result': 'high_quality_output'}
        
        async def standard_processing(data, context):
            await asyncio.sleep(0.5)
            return {'processing_type': 'standard', 'result': 'standard_output'}
        
        conditional_handler = ConditionalTaskHandler(
            check_data_quality, high_quality_processing, standard_processing
        )
        self.engine.register_task_handler('quality_based_processing', conditional_handler)
        
        # Event-driven pattern example
        async def process_user_action(event, data, context):
            action_type = event.get('action_type')
            if action_type == 'click':
                return {'processed': True, 'action': 'click_processed'}
            elif action_type == 'purchase':
                return {'processed': True, 'action': 'purchase_processed', 'stop_processing': True}
            else:
                return {'processed': False, 'action': 'unknown_action'}
        
        event_handler = EventDrivenTaskHandler(process_user_action, timeout=10.0)
        self.engine.register_task_handler('user_interaction_processor', event_handler)
        
        # Loop pattern example
        async def batch_process_data(data, context, iteration):
            batch_size = data.get('batch_size', 100)
            await asyncio.sleep(0.2)  # Simulate processing
            return {
                'processed_batch': iteration,
                'batch_size': batch_size,
                'remaining_items': max(0, data.get('total_items', 1000) - (iteration + 1) * batch_size)
            }
        
        async def check_remaining_items(data, context, iteration):
            remaining = data.get('remaining_items', 0)
            return remaining > 0 and iteration < 50  # Max 50 iterations
        
        loop_handler = LoopTaskHandler(batch_process_data, check_remaining_items, max_iterations=50)
        self.engine.register_task_handler('batch_processor', loop_handler)
    
    def create_data_processing_workflow(self) -> str:
        """Create a comprehensive data processing workflow"""
        tasks = [
            TaskDefinition(
                task_id="data_pipeline",
                name="Data Pipeline",
                description="Sequential data validation, transformation, and storage",
                pattern_type=PatternType.SEQUENTIAL,
                handler="data_pipeline",
                input_schema={"records": "array"},
                output_schema={"stored_records": "integer", "storage_location": "string"},
                dependencies=[],
                timeout=10.0
            ),
            TaskDefinition(
                task_id="quality_check",
                name="Quality-based Processing",
                description="Process data based on quality score",
                pattern_type=PatternType.CONDITIONAL,
                handler="quality_based_processing",
                input_schema={"quality_score": "number"},
                output_schema={"processing_type": "string", "result": "string"},
                dependencies=["data_pipeline"],
                timeout=5.0
            ),
            TaskDefinition(
                task_id="parallel_processing",
                name="Parallel Image Processing",
                description="Process images in parallel",
                pattern_type=PatternType.PARALLEL,
                handler="image_processing",
                input_schema={"image_count": "integer"},
                output_schema={"parallel_results": "object"},
                dependencies=["quality_check"],
                timeout=15.0
            ),
            TaskDefinition(
                task_id="batch_processing",
                name="Batch Data Processing",
                description="Process data in batches using loop",
                pattern_type=PatternType.LOOP,
                handler="batch_processor",
                input_schema={"total_items": "integer", "batch_size": "integer"},
                output_schema={"total_iterations": "integer", "final_data": "object"},
                dependencies=["parallel_processing"],
                timeout=30.0
            )
        ]
        
        from workflow_engine import WorkflowDefinition
        workflow = WorkflowDefinition(
            workflow_id="comprehensive_data_workflow",
            name="Comprehensive Data Processing Workflow",
            description="A workflow demonstrating all pattern types",
            version="1.0",
            tasks=tasks,
            metadata={'created_by': 'pattern_examples', 'complexity': 'high'}
        )
        
        return self.engine.register_workflow(workflow)
```

---

## 2. External System Integration

### 2.1 Integration Architecture

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">External System Integration Architecture</text>
  
  <!-- Workflow Engine Core -->
  <rect x="300" y="60" width="200" height="80" rx="10" fill="#3498db" stroke="#2980b9" stroke-width="2"/>
  <text x="400" y="85" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Workflow Engine</text>
  <text x="400" y="105" text-anchor="middle" font-size="12" fill="white">Integration Hub</text>
  <text x="400" y="125" text-anchor="middle" font-size="12" fill="white">Adapter Management</text>
  
  <!-- Integration Adapters -->
  <g id="adapters">
    <!-- API Adapter -->
    <rect x="50" y="180" width="120" height="80" rx="8" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
    <text x="110" y="205" text-anchor="middle" font-size="12" font-weight="bold" fill="white">API Adapter</text>
    <text x="110" y="220" text-anchor="middle" font-size="10" fill="white">REST/GraphQL</text>
    <text x="110" y="235" text-anchor="middle" font-size="10" fill="white">Rate Limiting</text>
    <text x="110" y="250" text-anchor="middle" font-size="10" fill="white">Auth Handling</text>
    
    <!-- Database Adapter -->
    <rect x="190" y="180" width="120" height="80" rx="8" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
    <text x="250" y="205" text-anchor="middle" font-size="12" font-weight="bold" fill="white">DB Adapter</text>
    <text x="250" y="220" text-anchor="middle" font-size="10" fill="white">SQL/NoSQL</text>
    <text x="250" y="235" text-anchor="middle" font-size="10" fill="white">Connection Pool</text>
    <text x="250" y="250" text-anchor="middle" font-size="10" fill="white">Transactions</text>
    
    <!-- Message Queue Adapter -->
    <rect x="330" y="180" width="120" height="80" rx="8" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
    <text x="390" y="205" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Queue Adapter</text>
    <text x="390" y="220" text-anchor="middle" font-size="10" fill="white">RabbitMQ/Kafka</text>
    <text x="390" y="235" text-anchor="middle" font-size="10" fill="white">Pub/Sub</text>
    <text x="390" y="250" text-anchor="middle" font-size="10" fill="white">Dead Letter</text>
    
    <!-- File System Adapter -->
    <rect x="470" y="180" width="120" height="80" rx="8" fill="#1abc9c" stroke="#16a085" stroke-width="2"/>
    <text x="530" y="205" text-anchor="middle" font-size="12" font-weight="bold" fill="white">File Adapter</text>
    <text x="530" y="220" text-anchor="middle" font-size="10" fill="white">Local/Cloud</text>
    <text x="530" y="235" text-anchor="middle" font-size="10" fill="white">S3/Azure/GCS</text>
    <text x="530" y="250" text-anchor="middle" font-size="10" fill="white">Streaming</text>
    
    <!-- Cloud Services Adapter -->
    <rect x="610" y="180" width="120" height="80" rx="8" fill="#34495e" stroke="#2c3e50" stroke-width="2"/>
    <text x="670" y="205" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Cloud Adapter</text>
    <text x="670" y="220" text-anchor="middle" font-size="10" fill="white">AWS/Azure/GCP</text>
    <text x="670" y="235" text-anchor="middle" font-size="10" fill="white">Serverless</text>
    <text x="670" y="250" text-anchor="middle" font-size="10" fill="white">AI Services</text>
  </g>
  
  <!-- Resilience Layer -->
  <rect x="100" y="300" width="600" height="60" rx="10" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2"/>
  <text x="400" y="320" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Resilience & Reliability Layer</text>
  <text x="400" y="340" text-anchor="middle" font-size="12" fill="white">Circuit Breakers • Retry Logic • Bulkhead Pattern • Timeout Management</text>
  <text x="400" y="355" text-anchor="middle" font-size="12" fill="white">Health Checks • Fallback Strategies • Load Balancing • Caching</text>
  
  <!-- External Systems -->
  <g id="external_systems">
    <!-- REST APIs -->
    <rect x="50" y="400" width="100" height="60" rx="8" fill="#e67e22" stroke="#d35400" stroke-width="2"/>
    <text x="100" y="420" text-anchor="middle" font-size="11" font-weight="bold" fill="white">REST APIs</text>
    <text x="100" y="435" text-anchor="middle" font-size="9" fill="white">Third-party</text>
    <text x="100" y="450" text-anchor="middle" font-size="9" fill="white">Services</text>
    
    <!-- Databases -->
    <rect x="170" y="400" width="100" height="60" rx="8" fill="#c0392b" stroke="#a93226" stroke-width="2"/>
    <text x="220" y="420" text-anchor="middle" font-size="11" font-weight="bold" fill="white">Databases</text>
    <text x="220" y="435" text-anchor="middle" font-size="9" fill="white">PostgreSQL</text>
    <text x="220" y="450" text-anchor="middle" font-size="9" fill="white">MongoDB</text>
    
    <!-- Message Brokers -->
    <rect x="290" y="400" width="100" height="60" rx="8" fill="#8e44ad" stroke="#7d3c98" stroke-width="2"/>
    <text x="340" y="420" text-anchor="middle" font-size="11" font-weight="bold" fill="white">Brokers</text>
    <text x="340" y="435" text-anchor="middle" font-size="9" fill="white">RabbitMQ</text>
    <text x="340" y="450" text-anchor="middle" font-size="9" fill="white">Apache Kafka</text>
    
    <!-- Cloud Storage -->
    <rect x="410" y="400" width="100" height="60" rx="8" fill="#16a085" stroke="#138d75" stroke-width="2"/>
    <text x="460" y="420" text-anchor="middle" font-size="11" font-weight="bold" fill="white">Storage</text>
    <text x="460" y="435" text-anchor="middle" font-size="9" fill="white">Amazon S3</text>
    <text x="460" y="450" text-anchor="middle" font-size="9" fill="white">Azure Blob</text>
    
    <!-- AI Services -->
    <rect x="530" y="400" width="100" height="60" rx="8" fill="#2c3e50" stroke="#1b2631" stroke-width="2"/>
    <text x="580" y="420" text-anchor="middle" font-size="11" font-weight="bold" fill="white">AI Services</text>
    <text x="580" y="435" text-anchor="middle" font-size="9" fill="white">OpenAI</text>
    <text x="580" y="450" text-anchor="middle" font-size="9" fill="white">Azure AI</text>
    
    <!-- Legacy Systems -->
    <rect x="650" y="400" width="100" height="60" rx="8" fill="#7f8c8d" stroke="#566573" stroke-width="2"/>
    <text x="700" y="420" text-anchor="middle" font-size="11" font-weight="bold" fill="white">Legacy</text>
    <text x="700" y="435" text-anchor="middle" font-size="9" fill="white">SOAP/XML</text>
    <text x="700" y="450" text-anchor="middle" font-size="9" fill="white">Mainframe</text>
  </g>
  
  <!-- Monitoring & Observability -->
  <rect x="250" y="500" width="300" height="60" rx="10" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="400" y="520" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Monitoring & Observability</text>
  <text x="400" y="540" text-anchor="middle" font-size="12" fill="white">Metrics • Tracing • Logging • Alerting</text>
  <text x="400" y="555" text-anchor="middle" font-size="12" fill="white">Performance Analytics • Error Tracking</text>
  
  <!-- Connections -->
  <g stroke="#34495e" stroke-width="2" fill="none" marker-end="url(#arrowhead)">
    <!-- Engine to Adapters -->
    <line x1="350" y1="140" x2="110" y2="180"/>
    <line x1="380" y1="140" x2="250" y2="180"/>
    <line x1="400" y1="140" x2="390" y2="180"/>
    <line x1="420" y1="140" x2="530" y2="180"/>
    <line x1="450" y1="140" x2="670" y2="180"/>
    
    <!-- Adapters to Resilience -->
    <line x1="110" y1="260" x2="200" y2="300"/>
    <line x1="250" y1="260" x2="300" y2="300"/>
    <line x1="390" y1="260" x2="400" y2="300"/>
    <line x1="530" y1="260" x2="500" y2="300"/>
    <line x1="670" y1="260" x2="600" y2="300"/>
    
    <!-- Resilience to External Systems -->
    <line x1="200" y1="360" x2="100" y2="400"/>
    <line x1="300" y1="360" x2="220" y2="400"/>
    <line x1="400" y1="360" x2="340" y2="400"/>
    <line x1="500" y1="360" x2="460" y2="400"/>
    <line x1="600" y1="360" x2="580" y2="400"/>
    <line x1="650" y1="360" x2="700" y2="400"/>
    
    <!-- Monitoring connections -->
    <line x1="400" y1="360" x2="400" y2="500"/>
  </g>
  
  <!-- Arrow marker -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
</svg>
```

### 2.2 Integration Framework Implementation

```python
# integration_framework.py
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from contextlib import asynccontextmanager
import time
import hashlib

@dataclass
class IntegrationConfig:
    name: str
    endpoint: str
    auth_config: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    retry_config: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[Dict[str, Any]] = None
    circuit_breaker: Optional[Dict[str, Any]] = None
    cache_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, 
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitBreakerState.CLOSED
        self.last_failure_time = None
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info("Circuit breaker reset to CLOSED")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class RateLimiter:
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire rate limit permission"""
        async with self._lock:
            now = time.time()
            # Remove old requests outside time window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    async def wait_for_slot(self):
        """Wait until a slot becomes available"""
        while not await self.acquire():
            await asyncio.sleep(0.1)

class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = json.dumps([args, sorted(kwargs.items())], sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] < self.ttl:
                    self.access_times[key] = time.time()  # Update access time
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
            return None
    
    async def set(self, key: str, value: Any):
        """Set value in cache"""
        async with self._lock:
            # Evict if at max size
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest entry
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    async def cached_call(self, func: Callable, *args, **kwargs):
        """Execute function with caching"""
        cache_key = self._generate_key(func.__name__, *args, **kwargs)
        
        # Try to get from cache
        cached_result = await self.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute function and cache result
        result = await func(*args, **kwargs)
        await self.set(cache_key, result)
        return result

class BaseIntegrationAdapter(ABC):
    """Base class for all integration adapters"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        
        # Initialize resilience components
        if config.circuit_breaker:
            self.circuit_breaker = CircuitBreaker(**config.circuit_breaker)
        else:
            self.circuit_breaker = None
        
        if config.rate_limit:
            self.rate_limiter = RateLimiter(**config.rate_limit)
        else:
            self.rate_limiter = None
        
        if config.cache_config:
            self.cache_manager = CacheManager(**config.cache_config)
        else:
            self.cache_manager = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to external system"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to external system"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check health of external system"""
        pass
    
    @abstractmethod
    async def execute_operation(self, operation: str, **kwargs) -> Any:
        """Execute operation on external system"""
        pass
    
    async def execute_with_resilience(self, operation: str, **kwargs) -> Any:
        """Execute operation with resilience patterns"""
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.wait_for_slot()
        
        # Try cache first
        if self.cache_manager and kwargs.get('use_cache', True):
            cached_result = await self.cache_manager.cached_call(
                self._execute_operation_internal, operation, **kwargs
            )
            return cached_result
        
        # Execute with circuit breaker
        if self.circuit_breaker:
            return await self.circuit_breaker.call(
                self._execute_operation_internal, operation, **kwargs
            )
        else:
            return await self._execute_operation_internal(operation, **kwargs)
    
    async def _execute_operation_internal(self, operation: str, **kwargs) -> Any:
        """Internal operation execution with retry logic"""
        retry_config = self.config.retry_config
        max_retries = retry_config.get('max_retries', 3)
        retry_delay = retry_config.get('retry_delay', 1.0)
        backoff_factor = retry_config.get('backoff_factor', 2.0)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = retry_delay * (backoff_factor ** (attempt - 1))
                    await asyncio.sleep(delay)
                
                return await asyncio.wait_for(
                    self.execute_operation(operation, **kwargs),
                    timeout=self.config.timeout
                )
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Operation {operation} attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries:
                    break
        
        raise last_exception