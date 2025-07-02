# Module VII: Interoperability of Agents

## Learning Objectives

By the end of this module, you will be able to:

1. **Design standardized agent communication protocols**
2. **Implement cross-platform agent interfaces**
3. **Build agent discovery and registry systems**
4. **Create protocol adapters for different agent frameworks**
5. **Develop secure inter-agent communication channels**
6. **Design federation patterns for multi-agent ecosystems**
7. **Implement agent capability negotiation mechanisms**
8. **Build monitoring systems for interoperable agent networks**

## Introduction to Agent Interoperability

Agent interoperability is the ability of different agent systems to communicate, collaborate, and work together effectively, regardless of their underlying implementation, framework, or platform. In modern AI ecosystems, agents often need to:

- **Communicate across different frameworks** (LangChain, AutoGen, CrewAI)
- **Integrate with external systems** (APIs, databases, services)
- **Collaborate in heterogeneous environments**
- **Share capabilities and resources**
- **Maintain security and trust boundaries**

### Interoperability Architecture Overview

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Agent Interoperability Architecture</text>
  
  <!-- Agent Ecosystems -->
  <g id="ecosystem1">
    <rect x="50" y="80" width="150" height="120" fill="#e3f2fd" stroke="#1976d2" stroke-width="2" rx="10"/>
    <text x="125" y="100" text-anchor="middle" font-weight="bold" fill="#1976d2">LangChain</text>
    <text x="125" y="115" text-anchor="middle" font-weight="bold" fill="#1976d2">Ecosystem</text>
    <circle cx="80" cy="140" r="15" fill="#42a5f5"/>
    <text x="80" y="145" text-anchor="middle" font-size="10" fill="white">A1</text>
    <circle cx="125" cy="140" r="15" fill="#42a5f5"/>
    <text x="125" y="145" text-anchor="middle" font-size="10" fill="white">A2</text>
    <circle cx="170" cy="140" r="15" fill="#42a5f5"/>
    <text x="170" y="145" text-anchor="middle" font-size="10" fill="white">A3</text>
  </g>
  
  <g id="ecosystem2">
    <rect x="250" y="80" width="150" height="120" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2" rx="10"/>
    <text x="325" y="100" text-anchor="middle" font-weight="bold" fill="#7b1fa2">AutoGen</text>
    <text x="325" y="115" text-anchor="middle" font-weight="bold" fill="#7b1fa2">Ecosystem</text>
    <circle cx="280" cy="140" r="15" fill="#ab47bc"/>
    <text x="280" y="145" text-anchor="middle" font-size="10" fill="white">B1</text>
    <circle cx="325" cy="140" r="15" fill="#ab47bc"/>
    <text x="325" y="145" text-anchor="middle" font-size="10" fill="white">B2</text>
    <circle cx="370" cy="140" r="15" fill="#ab47bc"/>
    <text x="370" y="145" text-anchor="middle" font-size="10" fill="white">B3</text>
  </g>
  
  <g id="ecosystem3">
    <rect x="450" y="80" width="150" height="120" fill="#e8f5e8" stroke="#388e3c" stroke-width="2" rx="10"/>
    <text x="525" y="100" text-anchor="middle" font-weight="bold" fill="#388e3c">CrewAI</text>
    <text x="525" y="115" text-anchor="middle" font-weight="bold" fill="#388e3c">Ecosystem</text>
    <circle cx="480" cy="140" r="15" fill="#66bb6a"/>
    <text x="480" y="145" text-anchor="middle" font-size="10" fill="white">C1</text>
    <circle cx="525" cy="140" r="15" fill="#66bb6a"/>
    <text x="525" y="145" text-anchor="middle" font-size="10" fill="white">C2</text>
    <circle cx="570" cy="140" r="15" fill="#66bb6a"/>
    <text x="570" y="145" text-anchor="middle" font-size="10" fill="white">C3</text>
  </g>
  
  <!-- Interoperability Layer -->
  <rect x="100" y="250" width="500" height="100" fill="#fff3e0" stroke="#f57c00" stroke-width="3" rx="15"/>
  <text x="350" y="275" text-anchor="middle" font-size="16" font-weight="bold" fill="#f57c00">Interoperability Layer</text>
  
  <!-- Components -->
  <rect x="120" y="290" width="80" height="40" fill="#ffcc02" stroke="#f57c00" rx="5"/>
  <text x="160" y="305" text-anchor="middle" font-size="10" fill="#333">Protocol</text>
  <text x="160" y="318" text-anchor="middle" font-size="10" fill="#333">Adapters</text>
  
  <rect x="220" y="290" width="80" height="40" fill="#ffcc02" stroke="#f57c00" rx="5"/>
  <text x="260" y="305" text-anchor="middle" font-size="10" fill="#333">Message</text>
  <text x="260" y="318" text-anchor="middle" font-size="10" fill="#333">Router</text>
  
  <rect x="320" y="290" width="80" height="40" fill="#ffcc02" stroke="#f57c00" rx="5"/>
  <text x="360" y="305" text-anchor="middle" font-size="10" fill="#333">Discovery</text>
  <text x="360" y="318" text-anchor="middle" font-size="10" fill="#333">Service</text>
  
  <rect x="420" y="290" width="80" height="40" fill="#ffcc02" stroke="#f57c00" rx="5"/>
  <text x="460" y="305" text-anchor="middle" font-size="10" fill="#333">Security</text>
  <text x="460" y="318" text-anchor="middle" font-size="10" fill="#333">Gateway</text>
  
  <rect x="520" y="290" width="60" height="40" fill="#ffcc02" stroke="#f57c00" rx="5"/>
  <text x="550" y="305" text-anchor="middle" font-size="10" fill="#333">Registry</text>
  <text x="550" y="318" text-anchor="middle" font-size="10" fill="#333">Store</text>
  
  <!-- External Systems -->
  <rect x="50" y="400" width="120" height="80" fill="#fce4ec" stroke="#c2185b" stroke-width="2" rx="10"/>
  <text x="110" y="425" text-anchor="middle" font-weight="bold" fill="#c2185b">External APIs</text>
  <rect x="60" y="440" width="30" height="20" fill="#e91e63" rx="3"/>
  <text x="75" y="453" text-anchor="middle" font-size="8" fill="white">REST</text>
  <rect x="100" y="440" width="30" height="20" fill="#e91e63" rx="3"/>
  <text x="115" y="453" text-anchor="middle" font-size="8" fill="white">GraphQL</text>
  <rect x="140" y="440" width="20" height="20" fill="#e91e63" rx="3"/>
  <text x="150" y="453" text-anchor="middle" font-size="8" fill="white">gRPC</text>
  
  <rect x="200" y="400" width="120" height="80" fill="#e1f5fe" stroke="#0277bd" stroke-width="2" rx="10"/>
  <text x="260" y="425" text-anchor="middle" font-weight="bold" fill="#0277bd">Databases</text>
  <rect x="210" y="440" width="30" height="20" fill="#0288d1" rx="3"/>
  <text x="225" y="453" text-anchor="middle" font-size="8" fill="white">SQL</text>
  <rect x="250" y="440" width="30" height="20" fill="#0288d1" rx="3"/>
  <text x="265" y="453" text-anchor="middle" font-size="8" fill="white">NoSQL</text>
  <rect x="290" y="440" width="20" height="20" fill="#0288d1" rx="3"/>
  <text x="300" y="453" text-anchor="middle" font-size="8" fill="white">Vector</text>
  
  <rect x="350" y="400" width="120" height="80" fill="#f1f8e9" stroke="#558b2f" stroke-width="2" rx="10"/>
  <text x="410" y="425" text-anchor="middle" font-weight="bold" fill="#558b2f">Message Queues</text>
  <rect x="360" y="440" width="30" height="20" fill="#689f38" rx="3"/>
  <text x="375" y="453" text-anchor="middle" font-size="8" fill="white">Kafka</text>
  <rect x="400" y="440" width="30" height="20" fill="#689f38" rx="3"/>
  <text x="415" y="453" text-anchor="middle" font-size="8" fill="white">Redis</text>
  <rect x="440" y="440" width="20" height="20" fill="#689f38" rx="3"/>
  <text x="450" y="453" text-anchor="middle" font-size="8" fill="white">MQTT</text>
  
  <rect x="500" y="400" width="120" height="80" fill="#fff8e1" stroke="#f9a825" stroke-width="2" rx="10"/>
  <text x="560" y="425" text-anchor="middle" font-weight="bold" fill="#f9a825">Cloud Services</text>
  <rect x="510" y="440" width="30" height="20" fill="#fbc02d" rx="3"/>
  <text x="525" y="453" text-anchor="middle" font-size="8" fill="white">AWS</text>
  <rect x="550" y="440" width="30" height="20" fill="#fbc02d" rx="3"/>
  <text x="565" y="453" text-anchor="middle" font-size="8" fill="white">Azure</text>
  <rect x="590" y="440" width="20" height="20" fill="#fbc02d" rx="3"/>
  <text x="600" y="453" text-anchor="middle" font-size="8" fill="white">GCP</text>
  
  <!-- Connection Lines -->
  <!-- From ecosystems to interop layer -->
  <line x1="125" y1="200" x2="200" y2="250" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="325" y1="200" x2="350" y2="250" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="525" y1="200" x2="500" y2="250" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- From interop layer to external systems -->
  <line x1="200" y1="350" x2="110" y2="400" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="300" y1="350" x2="260" y2="400" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="350" x2="410" y2="400" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="500" y1="350" x2="560" y2="400" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Cross-ecosystem communication -->
  <path d="M 200 140 Q 275 120 350 140" stroke="#ff6b6b" stroke-width="2" fill="none" stroke-dasharray="5,5" marker-end="url(#arrowhead-red)"/>
  <path d="M 400 140 Q 475 120 550 140" stroke="#ff6b6b" stroke-width="2" fill="none" stroke-dasharray="5,5" marker-end="url(#arrowhead-red)"/>
  <path d="M 200 160 Q 350 180 500 160" stroke="#ff6b6b" stroke-width="2" fill="none" stroke-dasharray="5,5" marker-end="url(#arrowhead-red)"/>
  
  <!-- Arrow markers -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
    <marker id="arrowhead-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#ff6b6b"/>
    </marker>
  </defs>
  
  <!-- Legend -->
  <rect x="650" y="80" width="130" height="120" fill="white" stroke="#ccc" rx="5"/>
  <text x="715" y="100" text-anchor="middle" font-weight="bold" fill="#333">Legend</text>
  <line x1="660" y1="115" x2="680" y2="115" stroke="#666" stroke-width="2"/>
  <text x="685" y="119" font-size="12" fill="#333">Standard Comm.</text>
  <line x1="660" y1="135" x2="680" y2="135" stroke="#ff6b6b" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="685" y="139" font-size="12" fill="#333">Cross-Platform</text>
  <circle cx="670" cy="155" r="8" fill="#42a5f5"/>
  <text x="685" y="159" font-size="12" fill="#333">Agent Instance</text>
  <rect x="660" y="170" width="20" height="15" fill="#ffcc02" stroke="#f57c00" rx="2"/>
  <text x="685" y="181" font-size="12" fill="#333">Interop Component</text>
  
  <!-- Status indicators -->
  <text x="400" y="580" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Enabling seamless collaboration across diverse agent ecosystems</text>
</svg>
```

## 1. Agent Communication Protocols

### 1.1 Standardized Message Format

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime
import asyncio

class MessageType(Enum):
    """Standard message types for agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentCapability:
    """Represents an agent's capability"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "version": self.version,
            "tags": self.tags
        }

@dataclass
class AgentMessage:
    """Standardized agent message format"""
    id: str
    type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast messages
    payload: Dict[str, Any]
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None  # For request-response correlation
    reply_to: Optional[str] = None  # For response routing
    ttl: Optional[int] = None  # Time to live in seconds
    headers: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def create_request(cls, sender_id: str, recipient_id: str, 
                      action: str, params: Dict[str, Any],
                      correlation_id: str = None) -> 'AgentMessage':
        """Create a request message"""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload={"action": action, "params": params},
            timestamp=datetime.now(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )
    
    @classmethod
    def create_response(cls, sender_id: str, recipient_id: str,
                       result: Any, correlation_id: str,
                       success: bool = True) -> 'AgentMessage':
        """Create a response message"""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.RESPONSE,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload={"success": success, "result": result},
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
    
    @classmethod
    def create_notification(cls, sender_id: str, event: str,
                           data: Dict[str, Any],
                           recipient_id: str = None) -> 'AgentMessage':
        """Create a notification message"""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.NOTIFICATION,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload={"event": event, "data": data},
            timestamp=datetime.now()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl": self.ttl,
            "headers": self.headers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=MessagePriority(data.get("priority", MessagePriority.NORMAL.value)),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl=data.get("ttl"),
            headers=data.get("headers", {})
        )
    
    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Deserialize message from JSON"""
        return cls.from_dict(json.loads(json_str))
```

### 1.2 Protocol Adapters

```python
class ProtocolAdapter(ABC):
    """Abstract base class for protocol adapters"""
    
    @abstractmethod
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through this protocol"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive a message from this protocol"""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection"""
        pass

class HTTPProtocolAdapter(ProtocolAdapter):
    """HTTP-based protocol adapter"""
    
    def __init__(self, base_url: str, port: int = 8080):
        self.base_url = base_url
        self.port = port
        self.session = None
        self.message_queue = asyncio.Queue()
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        import aiohttp
        self.session = aiohttp.ClientSession()
        return True
    
    async def disconnect(self) -> bool:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
        return True
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message via HTTP POST"""
        if not self.session:
            await self.connect()
        
        try:
            url = f"{self.base_url}:{self.port}/messages"
            async with self.session.post(url, json=message.to_dict()) as response:
                return response.status == 200
        except Exception as e:
            print(f"Failed to send HTTP message: {e}")
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive message from queue (populated by HTTP server)"""
        try:
            message_dict = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return AgentMessage.from_dict(message_dict)
        except asyncio.TimeoutError:
            return None
    
    def enqueue_received_message(self, message_dict: Dict[str, Any]):
        """Called by HTTP server to enqueue received messages"""
        self.message_queue.put_nowait(message_dict)

class WebSocketProtocolAdapter(ProtocolAdapter):
    """WebSocket-based protocol adapter"""
    
    def __init__(self, url: str):
        self.url = url
        self.websocket = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to WebSocket"""
        try:
            import websockets
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
        return True
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message via WebSocket"""
        if not self.connected:
            await self.connect()
        
        try:
            await self.websocket.send(message.to_json())
            return True
        except Exception as e:
            print(f"Failed to send WebSocket message: {e}")
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive message from WebSocket"""
        if not self.connected:
            return None
        
        try:
            message_json = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
            return AgentMessage.from_json(message_json)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"Failed to receive WebSocket message: {e}")
            return None

class MessageQueueProtocolAdapter(ProtocolAdapter):
    """Message queue-based protocol adapter (Redis/RabbitMQ)"""
    
    def __init__(self, queue_url: str, queue_name: str):
        self.queue_url = queue_url
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
    
    async def connect(self) -> bool:
        """Connect to message queue"""
        try:
            # Example with Redis
            import aioredis
            self.connection = await aioredis.from_url(self.queue_url)
            return True
        except Exception as e:
            print(f"Failed to connect to message queue: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from message queue"""
        if self.connection:
            await self.connection.close()
        return True
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to queue"""
        if not self.connection:
            await self.connect()
        
        try:
            await self.connection.lpush(self.queue_name, message.to_json())
            return True
        except Exception as e:
            print(f"Failed to send queue message: {e}")
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive message from queue"""
        if not self.connection:
            return None
        
        try:
            message_json = await self.connection.brpop(self.queue_name, timeout=1)
            if message_json:
                return AgentMessage.from_json(message_json[1])
            return None
        except Exception as e:
            print(f"Failed to receive queue message: {e}")
            return None

## 2. Agent Discovery and Registry

### 2.1 Agent Registry System

```python
from typing import Set, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json

@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    endpoint: str
    protocol: str
    version: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, inactive, maintenance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "version": self.version,
            "tags": self.tags,
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentRegistration':
        capabilities = [AgentCapability(**cap) for cap in data["capabilities"]]
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            description=data["description"],
            capabilities=capabilities,
            endpoint=data["endpoint"],
            protocol=data["protocol"],
            version=data["version"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            registered_at=datetime.fromisoformat(data["registered_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            status=data.get("status", "active")
        )

class AgentRegistry:
    """Central registry for agent discovery and management"""
    
    def __init__(self, heartbeat_timeout: int = 300):
        self.agents: Dict[str, AgentRegistration] = {}
        self.capability_index: Dict[str, Set[str]] = {}  # capability_name -> agent_ids
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> agent_ids
        self.heartbeat_timeout = heartbeat_timeout
        self._cleanup_task = None
    
    async def start(self):
        """Start the registry with background cleanup"""
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_agents())
    
    async def stop(self):
        """Stop the registry"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def register_agent(self, registration: AgentRegistration) -> bool:
        """Register a new agent"""
        try:
            # Store agent registration
            self.agents[registration.agent_id] = registration
            
            # Update capability index
            for capability in registration.capabilities:
                if capability.name not in self.capability_index:
                    self.capability_index[capability.name] = set()
                self.capability_index[capability.name].add(registration.agent_id)
            
            # Update tag index
            for tag in registration.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(registration.agent_id)
            
            print(f"Agent {registration.agent_id} registered successfully")
            return True
        except Exception as e:
            print(f"Failed to register agent {registration.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id not in self.agents:
            return False
        
        registration = self.agents[agent_id]
        
        # Remove from capability index
        for capability in registration.capabilities:
            if capability.name in self.capability_index:
                self.capability_index[capability.name].discard(agent_id)
                if not self.capability_index[capability.name]:
                    del self.capability_index[capability.name]
        
        # Remove from tag index
        for tag in registration.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(agent_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
        
        # Remove agent
        del self.agents[agent_id]
        print(f"Agent {agent_id} unregistered successfully")
        return True
    
    async def update_heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat"""
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.now()
            self.agents[agent_id].status = "active"
            return True
        return False
    
    async def find_agents_by_capability(self, capability_name: str) -> List[AgentRegistration]:
        """Find agents that have a specific capability"""
        agent_ids = self.capability_index.get(capability_name, set())
        return [self.agents[agent_id] for agent_id in agent_ids 
                if agent_id in self.agents and self.agents[agent_id].status == "active"]
    
    async def find_agents_by_tag(self, tag: str) -> List[AgentRegistration]:
        """Find agents with a specific tag"""
        agent_ids = self.tag_index.get(tag, set())
        return [self.agents[agent_id] for agent_id in agent_ids 
                if agent_id in self.agents and self.agents[agent_id].status == "active"]
    
    async def find_agents_by_criteria(self, 
                                    capabilities: List[str] = None,
                                    tags: List[str] = None,
                                    protocol: str = None) -> List[AgentRegistration]:
        """Find agents matching multiple criteria"""
        candidate_agents = set(self.agents.keys())
        
        # Filter by capabilities
        if capabilities:
            for capability in capabilities:
                if capability in self.capability_index:
                    candidate_agents &= self.capability_index[capability]
                else:
                    return []  # No agents have this capability
        
        # Filter by tags
        if tags:
            for tag in tags:
                if tag in self.tag_index:
                    candidate_agents &= self.tag_index[tag]
                else:
                    return []  # No agents have this tag
        
        # Filter by protocol and status
        result = []
        for agent_id in candidate_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if (agent.status == "active" and 
                    (protocol is None or agent.protocol == protocol)):
                    result.append(agent)
        
        return result
    
    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration by ID"""
        return self.agents.get(agent_id)
    
    async def list_all_agents(self, include_inactive: bool = False) -> List[AgentRegistration]:
        """List all registered agents"""
        if include_inactive:
            return list(self.agents.values())
        return [agent for agent in self.agents.values() if agent.status == "active"]
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        active_agents = [a for a in self.agents.values() if a.status == "active"]
        return {
            "total_agents": len(self.agents),
            "active_agents": len(active_agents),
            "inactive_agents": len(self.agents) - len(active_agents),
            "total_capabilities": len(self.capability_index),
            "total_tags": len(self.tag_index),
            "protocols": list(set(a.protocol for a in active_agents))
        }
    
    async def _cleanup_inactive_agents(self):
        """Background task to cleanup inactive agents"""
        while True:
            try:
                current_time = datetime.now()
                inactive_agents = []
                
                for agent_id, registration in self.agents.items():
                    time_since_heartbeat = current_time - registration.last_heartbeat
                    if time_since_heartbeat.total_seconds() > self.heartbeat_timeout:
                        inactive_agents.append(agent_id)
                
                for agent_id in inactive_agents:
                    self.agents[agent_id].status = "inactive"
                    print(f"Agent {agent_id} marked as inactive due to missed heartbeat")
                
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

### 2.2 Discovery Service

class AgentDiscoveryService:
    """Service for agent discovery and capability matching"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.discovery_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def discover_capability_providers(self, capability: str, 
                                          max_results: int = 10) -> List[AgentRegistration]:
        """Discover agents that can provide a specific capability"""
        cache_key = f"capability:{capability}"
        
        # Check cache first
        if cache_key in self.discovery_cache:
            cached_result, timestamp = self.discovery_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return cached_result[:max_results]
        
        # Query registry
        agents = await self.registry.find_agents_by_capability(capability)
        
        # Sort by some criteria (e.g., last heartbeat, load)
        agents.sort(key=lambda a: a.last_heartbeat, reverse=True)
        
        # Cache result
        self.discovery_cache[cache_key] = (agents, datetime.now())
        
        return agents[:max_results]
    
    async def discover_best_agent_for_task(self, 
                                         required_capabilities: List[str],
                                         preferred_tags: List[str] = None,
                                         load_balancing: str = "round_robin") -> Optional[AgentRegistration]:
        """Find the best agent for a specific task"""
        # Find agents with all required capabilities
        candidates = await self.registry.find_agents_by_criteria(
            capabilities=required_capabilities,
            tags=preferred_tags
        )
        
        if not candidates:
            return None
        
        # Apply load balancing strategy
        if load_balancing == "round_robin":
            # Simple round-robin (in practice, you'd track state)
            return candidates[0]
        elif load_balancing == "least_recent":
            # Choose agent with oldest last heartbeat (assuming it's less busy)
            return min(candidates, key=lambda a: a.last_heartbeat)
        elif load_balancing == "random":
            import random
            return random.choice(candidates)
        else:
            return candidates[0]
    
    async def get_capability_graph(self) -> Dict[str, List[str]]:
        """Get a graph of capabilities and their providers"""
        graph = {}
        for capability, agent_ids in self.registry.capability_index.items():
            active_agents = []
            for agent_id in agent_ids:
                if (agent_id in self.registry.agents and 
                    self.registry.agents[agent_id].status == "active"):
                    active_agents.append(agent_id)
            if active_agents:
                graph[capability] = active_agents
        return graph
    
    async def suggest_agent_composition(self, 
                                      required_capabilities: List[str]) -> Dict[str, List[str]]:
        """Suggest how to compose agents to fulfill all required capabilities"""
        composition = {}
        missing_capabilities = []
        
        for capability in required_capabilities:
            providers = await self.discover_capability_providers(capability)
            if providers:
                composition[capability] = [p.agent_id for p in providers]
            else:
                missing_capabilities.append(capability)
        
        result = {"composition": composition}
        if missing_capabilities:
            result["missing_capabilities"] = missing_capabilities
        
        return result
```

## 3. Security and Trust Management

### 3.1 Agent Authentication and Authorization

```python
import hashlib
import hmac
import jwt
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import secrets

@dataclass
class AgentCredentials:
    """Agent authentication credentials"""
    agent_id: str
    api_key: str
    secret_key: str
    permissions: Set[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if credentials are expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

class SecurityManager:
    """Manages agent security, authentication, and authorization"""
    
    def __init__(self, jwt_secret: str):
        self.jwt_secret = jwt_secret
        self.agent_credentials: Dict[str, AgentCredentials] = {}
        self.revoked_tokens: Set[str] = set()
        self.permission_hierarchy = {
            "admin": {"read", "write", "execute", "manage", "discover"},
            "operator": {"read", "write", "execute", "discover"},
            "worker": {"read", "execute"},
            "observer": {"read", "discover"}
        }
    
    def generate_credentials(self, agent_id: str, 
                           permissions: Set[str],
                           expires_in_days: int = 365) -> AgentCredentials:
        """Generate new credentials for an agent"""
        api_key = secrets.token_urlsafe(32)
        secret_key = secrets.token_urlsafe(64)
        expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        credentials = AgentCredentials(
            agent_id=agent_id,
            api_key=api_key,
            secret_key=secret_key,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        self.agent_credentials[agent_id] = credentials
        return credentials
    
    def create_jwt_token(self, agent_id: str, 
                        additional_claims: Dict[str, Any] = None) -> str:
        """Create JWT token for agent"""
        if agent_id not in self.agent_credentials:
            raise ValueError(f"No credentials found for agent {agent_id}")
        
        credentials = self.agent_credentials[agent_id]
        if credentials.is_expired():
            raise ValueError(f"Credentials expired for agent {agent_id}")
        
        payload = {
            "agent_id": agent_id,
            "permissions": list(credentials.permissions),
            "iat": datetime.now(),
            "exp": datetime.now() + timedelta(hours=24),
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token is revoked
            if payload.get("jti") in self.revoked_tokens:
                raise ValueError("Token has been revoked")
            
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"], 
                               options={"verify_exp": False})
            jti = payload.get("jti")
            if jti:
                self.revoked_tokens.add(jti)
                return True
        except jwt.InvalidTokenError:
            pass
        return False
    
    def check_permission(self, agent_id: str, required_permission: str) -> bool:
        """Check if agent has required permission"""
        if agent_id not in self.agent_credentials:
            return False
        
        credentials = self.agent_credentials[agent_id]
        if credentials.is_expired():
            return False
        
        return required_permission in credentials.permissions
    
    def sign_message(self, message: str, secret_key: str) -> str:
        """Sign a message using HMAC"""
        return hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_message_signature(self, message: str, signature: str, 
                                secret_key: str) -> bool:
        """Verify message signature"""
        expected_signature = self.sign_message(message, secret_key)
        return hmac.compare_digest(signature, expected_signature)

### 3.2 Secure Communication Channel

class SecureMessageChannel:
    """Secure communication channel for agent messages"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.encryption_key = secrets.token_bytes(32)  # AES-256 key
    
    def encrypt_message(self, message: AgentMessage, 
                       sender_secret: str) -> Dict[str, Any]:
        """Encrypt and sign a message"""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Create Fernet cipher
            key = base64.urlsafe_b64encode(self.encryption_key)
            cipher = Fernet(key)
            
            # Serialize and encrypt message
            message_json = message.to_json()
            encrypted_data = cipher.encrypt(message_json.encode())
            
            # Sign the encrypted data
            signature = self.security_manager.sign_message(
                encrypted_data.decode(), sender_secret
            )
            
            return {
                "encrypted_data": encrypted_data.decode(),
                "signature": signature,
                "sender_id": message.sender_id,
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            # Fallback without encryption if cryptography not available
            message_json = message.to_json()
            signature = self.security_manager.sign_message(message_json, sender_secret)
            return {
                "data": message_json,
                "signature": signature,
                "sender_id": message.sender_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def decrypt_message(self, encrypted_envelope: Dict[str, Any],
                       sender_secret: str) -> AgentMessage:
        """Decrypt and verify a message"""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            if "encrypted_data" in encrypted_envelope:
                # Verify signature
                if not self.security_manager.verify_message_signature(
                    encrypted_envelope["encrypted_data"],
                    encrypted_envelope["signature"],
                    sender_secret
                ):
                    raise ValueError("Message signature verification failed")
                
                # Decrypt message
                key = base64.urlsafe_b64encode(self.encryption_key)
                cipher = Fernet(key)
                decrypted_data = cipher.decrypt(encrypted_envelope["encrypted_data"].encode())
                message_json = decrypted_data.decode()
            else:
                # Fallback for unencrypted messages
                if not self.security_manager.verify_message_signature(
                    encrypted_envelope["data"],
                    encrypted_envelope["signature"],
                    sender_secret
                ):
                    raise ValueError("Message signature verification failed")
                message_json = encrypted_envelope["data"]
            
            return AgentMessage.from_json(message_json)
        except ImportError:
            # Fallback without encryption
            if not self.security_manager.verify_message_signature(
                encrypted_envelope["data"],
                encrypted_envelope["signature"],
                sender_secret
            ):
                raise ValueError("Message signature verification failed")
            return AgentMessage.from_json(encrypted_envelope["data"])

## 4. Federation and Cross-Platform Integration

### 4.1 Agent Federation Manager

class AgentFederation:
    """Manages federation of multiple agent ecosystems"""
    
    def __init__(self, federation_id: str, security_manager: SecurityManager):
        self.federation_id = federation_id
        self.security_manager = security_manager
        self.member_registries: Dict[str, AgentRegistry] = {}
        self.federation_policies: Dict[str, Any] = {}
        self.cross_registry_cache = {}
    
    async def add_member_registry(self, registry_id: str, 
                                registry: AgentRegistry,
                                trust_level: str = "trusted") -> bool:
        """Add a member registry to the federation"""
        self.member_registries[registry_id] = registry
        self.federation_policies[registry_id] = {
            "trust_level": trust_level,
            "allowed_capabilities": "*" if trust_level == "trusted" else [],
            "rate_limits": {"requests_per_minute": 1000 if trust_level == "trusted" else 100}
        }
        return True
    
    async def federated_discovery(self, capability: str,
                                max_results: int = 10) -> List[AgentRegistration]:
        """Discover agents across all federated registries"""
        all_agents = []
        
        for registry_id, registry in self.member_registries.items():
            policy = self.federation_policies[registry_id]
            
            # Check if capability is allowed for this registry
            allowed_caps = policy["allowed_capabilities"]
            if allowed_caps != "*" and capability not in allowed_caps:
                continue
            
            try:
                agents = await registry.find_agents_by_capability(capability)
                # Add federation metadata
                for agent in agents:
                    agent.metadata["federation_id"] = self.federation_id
                    agent.metadata["source_registry"] = registry_id
                    agent.metadata["trust_level"] = policy["trust_level"]
                all_agents.extend(agents)
            except Exception as e:
                print(f"Error querying registry {registry_id}: {e}")
        
        # Sort by trust level and other criteria
        all_agents.sort(key=lambda a: (
            a.metadata.get("trust_level") == "trusted",
            a.last_heartbeat
        ), reverse=True)
        
        return all_agents[:max_results]
    
    async def federated_agent_lookup(self, agent_id: str) -> Optional[AgentRegistration]:
        """Look up an agent across all federated registries"""
        for registry_id, registry in self.member_registries.items():
            try:
                agent = await registry.get_agent(agent_id)
                if agent:
                    agent.metadata["source_registry"] = registry_id
                    return agent
            except Exception as e:
                print(f"Error looking up agent in registry {registry_id}: {e}")
        return None
    
    async def get_federation_stats(self) -> Dict[str, Any]:
        """Get statistics across the entire federation"""
        total_stats = {
            "federation_id": self.federation_id,
            "member_registries": len(self.member_registries),
            "total_agents": 0,
            "total_capabilities": set(),
            "registry_stats": {}
        }
        
        for registry_id, registry in self.member_registries.items():
            try:
                stats = await registry.get_registry_stats()
                total_stats["total_agents"] += stats["active_agents"]
                total_stats["registry_stats"][registry_id] = stats
                
                # Collect unique capabilities
                agents = await registry.list_all_agents()
                for agent in agents:
                    for cap in agent.capabilities:
                        total_stats["total_capabilities"].add(cap.name)
            except Exception as e:
                print(f"Error getting stats from registry {registry_id}: {e}")
        
        total_stats["total_capabilities"] = len(total_stats["total_capabilities"])
        return total_stats

### 4.2 Cross-Platform Message Router

class CrossPlatformMessageRouter:
    """Routes messages between different agent platforms and protocols"""
    
    def __init__(self, federation: AgentFederation):
        self.federation = federation
        self.protocol_adapters: Dict[str, ProtocolAdapter] = {}
        self.routing_table: Dict[str, str] = {}  # agent_id -> protocol
        self.message_transformers: Dict[str, callable] = {}
    
    def register_protocol_adapter(self, protocol_name: str, 
                                adapter: ProtocolAdapter):
        """Register a protocol adapter"""
        self.protocol_adapters[protocol_name] = adapter
    
    def register_message_transformer(self, from_protocol: str, 
                                   to_protocol: str, 
                                   transformer: callable):
        """Register a message transformer between protocols"""
        key = f"{from_protocol}->{to_protocol}"
        self.message_transformers[key] = transformer
    
    async def route_message(self, message: AgentMessage) -> bool:
        """Route a message to the appropriate agent"""
        # Look up recipient agent
        recipient = await self.federation.federated_agent_lookup(message.recipient_id)
        if not recipient:
            print(f"Recipient agent {message.recipient_id} not found")
            return False
        
        # Determine target protocol
        target_protocol = recipient.protocol
        if target_protocol not in self.protocol_adapters:
            print(f"No adapter available for protocol {target_protocol}")
            return False
        
        # Transform message if needed
        source_protocol = message.headers.get("source_protocol", "standard")
        if source_protocol != target_protocol:
            transformer_key = f"{source_protocol}->{target_protocol}"
            if transformer_key in self.message_transformers:
                message = self.message_transformers[transformer_key](message)
        
        # Send message via appropriate adapter
        adapter = self.protocol_adapters[target_protocol]
        return await adapter.send_message(message)
    
    async def broadcast_message(self, message: AgentMessage, 
                              target_capabilities: List[str] = None) -> Dict[str, bool]:
        """Broadcast a message to multiple agents"""
        results = {}
        
        if target_capabilities:
            # Broadcast to agents with specific capabilities
            for capability in target_capabilities:
                agents = await self.federation.federated_discovery(capability)
                for agent in agents:
                    targeted_message = AgentMessage(
                        id=str(uuid.uuid4()),
                        type=message.type,
                        sender_id=message.sender_id,
                        recipient_id=agent.agent_id,
                        payload=message.payload,
                        timestamp=datetime.now(),
                        correlation_id=message.correlation_id
                    )
                    results[agent.agent_id] = await self.route_message(targeted_message)
        else:
            # Broadcast to all agents
            for registry_id, registry in self.federation.member_registries.items():
                agents = await registry.list_all_agents()
                for agent in agents:
                    targeted_message = AgentMessage(
                        id=str(uuid.uuid4()),
                        type=message.type,
                        sender_id=message.sender_id,
                        recipient_id=agent.agent_id,
                        payload=message.payload,
                        timestamp=datetime.now(),
                        correlation_id=message.correlation_id
                    )
                    results[agent.agent_id] = await self.route_message(targeted_message)
        
        return results
```

## 5. Practical Implementation Examples

### 5.1 Complete Interoperability Setup

```python
async def setup_interoperability_infrastructure():
    """Set up a complete interoperability infrastructure"""
    
    # Initialize security manager
    security_manager = SecurityManager(jwt_secret="your-secret-key")
    
    # Create agent registries for different ecosystems
    langchain_registry = AgentRegistry(heartbeat_timeout=300)
    autogen_registry = AgentRegistry(heartbeat_timeout=300)
    crewai_registry = AgentRegistry(heartbeat_timeout=300)
    
    # Start registries
    await langchain_registry.start()
    await autogen_registry.start()
    await crewai_registry.start()
    
    # Create federation
    federation = AgentFederation("multi-ecosystem-federation", security_manager)
    await federation.add_member_registry("langchain", langchain_registry, "trusted")
    await federation.add_member_registry("autogen", autogen_registry, "trusted")
    await federation.add_member_registry("crewai", crewai_registry, "limited")
    
    # Set up protocol adapters
    http_adapter = HTTPProtocolAdapter("http://localhost", 8080)
    ws_adapter = WebSocketProtocolAdapter("ws://localhost:8081")
    mq_adapter = MessageQueueProtocolAdapter("redis://localhost:6379", "agent_messages")
    
    # Create message router
    router = CrossPlatformMessageRouter(federation)
    router.register_protocol_adapter("http", http_adapter)
    router.register_protocol_adapter("websocket", ws_adapter)
    router.register_protocol_adapter("redis", mq_adapter)
    
    # Set up discovery service
    discovery_service = AgentDiscoveryService(langchain_registry)  # Primary registry
    
    # Create secure communication channel
    secure_channel = SecureMessageChannel(security_manager)
    
    return {
        "security_manager": security_manager,
        "federation": federation,
        "router": router,
        "discovery_service": discovery_service,
        "secure_channel": secure_channel,
        "registries": {
            "langchain": langchain_registry,
            "autogen": autogen_registry,
            "crewai": crewai_registry
        }
    }

async def register_sample_agents(infrastructure):
    """Register sample agents in different ecosystems"""
    security_manager = infrastructure["security_manager"]
    registries = infrastructure["registries"]
    
    # LangChain agents
    langchain_agents = [
        AgentRegistration(
            agent_id="langchain-research-agent",
            name="Research Assistant",
            description="Specialized in web research and data analysis",
            capabilities=[
                AgentCapability(
                    name="web_search",
                    description="Search the web for information",
                    input_schema={"query": "string", "max_results": "integer"},
                    output_schema={"results": "array", "summary": "string"}
                ),
                AgentCapability(
                    name="data_analysis",
                    description="Analyze structured data",
                    input_schema={"data": "object", "analysis_type": "string"},
                    output_schema={"insights": "array", "visualizations": "array"}
                )
            ],
            endpoint="http://localhost:8080/langchain/research",
            protocol="http",
            version="1.0.0",
            tags=["research", "analysis", "langchain"]
        ),
        AgentRegistration(
            agent_id="langchain-code-agent",
            name="Code Generator",
            description="Generates and reviews code",
            capabilities=[
                AgentCapability(
                    name="code_generation",
                    description="Generate code in various languages",
                    input_schema={"requirements": "string", "language": "string"},
                    output_schema={"code": "string", "explanation": "string"}
                )
            ],
            endpoint="ws://localhost:8081/langchain/code",
            protocol="websocket",
            version="1.0.0",
            tags=["coding", "generation", "langchain"]
        )
    ]
    
    # AutoGen agents
    autogen_agents = [
        AgentRegistration(
            agent_id="autogen-conversation-agent",
            name="Conversation Manager",
            description="Manages multi-agent conversations",
            capabilities=[
                AgentCapability(
                    name="conversation_management",
                    description="Orchestrate multi-agent conversations",
                    input_schema={"participants": "array", "topic": "string"},
                    output_schema={"conversation_log": "array", "summary": "string"}
                )
            ],
            endpoint="redis://localhost:6379/autogen/conversation",
            protocol="redis",
            version="1.0.0",
            tags=["conversation", "orchestration", "autogen"]
        )
    ]
    
    # CrewAI agents
    crewai_agents = [
        AgentRegistration(
            agent_id="crewai-task-agent",
            name="Task Executor",
            description="Executes specific tasks in a crew",
            capabilities=[
                AgentCapability(
                    name="task_execution",
                    description="Execute assigned tasks",
                    input_schema={"task": "object", "context": "object"},
                    output_schema={"result": "object", "status": "string"}
                )
            ],
            endpoint="http://localhost:8082/crewai/task",
            protocol="http",
            version="1.0.0",
            tags=["task", "execution", "crewai"]
        )
    ]
    
    # Register agents
    for agent in langchain_agents:
        await registries["langchain"].register_agent(agent)
        # Generate credentials
        security_manager.generate_credentials(
            agent.agent_id, 
            {"read", "write", "execute"}
        )
    
    for agent in autogen_agents:
        await registries["autogen"].register_agent(agent)
        security_manager.generate_credentials(
            agent.agent_id, 
            {"read", "write", "execute", "manage"}
        )
    
    for agent in crewai_agents:
        await registries["crewai"].register_agent(agent)
        security_manager.generate_credentials(
            agent.agent_id, 
            {"read", "execute"}
        )

async def demonstrate_cross_platform_communication(infrastructure):
    """Demonstrate communication between agents from different platforms"""
    federation = infrastructure["federation"]
    router = infrastructure["router"]
    security_manager = infrastructure["security_manager"]
    
    print("\n=== Cross-Platform Agent Communication Demo ===")
    
    # 1. Discover agents with specific capabilities
    print("\n1. Discovering agents with 'web_search' capability:")
    search_agents = await federation.federated_discovery("web_search")
    for agent in search_agents:
        print(f"  - {agent.name} ({agent.agent_id}) from {agent.metadata.get('source_registry')}")
    
    # 2. Create a cross-platform message
    message = AgentMessage.create_request(
        sender_id="system-orchestrator",
        recipient_id="langchain-research-agent",
        action="web_search",
        params={"query": "latest AI developments", "max_results": 5}
    )
    
    print(f"\n2. Sending message from system to LangChain agent:")
    print(f"  Message ID: {message.id}")
    print(f"  Action: {message.payload['action']}")
    
    # 3. Route the message
    success = await router.route_message(message)
    print(f"  Routing success: {success}")
    
    # 4. Demonstrate secure communication
    print(f"\n3. Demonstrating secure communication:")
    sender_credentials = security_manager.agent_credentials.get("langchain-research-agent")
    if sender_credentials:
        secure_channel = infrastructure["secure_channel"]
        encrypted_envelope = secure_channel.encrypt_message(message, sender_credentials.secret_key)
        print(f"  Message encrypted: {bool(encrypted_envelope.get('encrypted_data'))}")
        print(f"  Signature present: {bool(encrypted_envelope.get('signature'))}")
    
    # 5. Get federation statistics
    print(f"\n4. Federation Statistics:")
    stats = await federation.get_federation_stats()
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Member registries: {stats['member_registries']}")
    print(f"  Total capabilities: {stats['total_capabilities']}")
    
    return stats

# Usage example
async def main():
    """Main demonstration function"""
    print("Setting up interoperability infrastructure...")
    infrastructure = await setup_interoperability_infrastructure()
    
    print("Registering sample agents...")
    await register_sample_agents(infrastructure)
    
    print("Demonstrating cross-platform communication...")
    stats = await demonstrate_cross_platform_communication(infrastructure)
    
    # Cleanup
    print("\nCleaning up...")
    for registry in infrastructure["registries"].values():
        await registry.stop()
    
    print("Demo completed successfully!")

# Run the demo
# asyncio.run(main())
```

## 6. Hands-on Exercises

### Exercise 1: Custom Protocol Adapter

```python
class CustomProtocolAdapter(ProtocolAdapter):
    """Implement a custom protocol adapter for your specific needs"""
    
    def __init__(self, custom_config: Dict[str, Any]):
        self.config = custom_config
        self.connected = False
        # TODO: Initialize your custom protocol connection
    
    async def connect(self) -> bool:
        """TODO: Implement connection logic for your protocol"""
        # Example: Connect to your custom message broker, API, etc.
        self.connected = True
        return True
    
    async def disconnect(self) -> bool:
        """TODO: Implement disconnection logic"""
        self.connected = False
        return True
    
    async def send_message(self, message: AgentMessage) -> bool:
        """TODO: Implement message sending for your protocol"""
        if not self.connected:
            await self.connect()
        
        # Example implementation:
        # 1. Serialize the message
        # 2. Send via your protocol
        # 3. Handle errors
        return True
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """TODO: Implement message receiving for your protocol"""
        if not self.connected:
            return None
        
        # Example implementation:
        # 1. Check for incoming messages
        # 2. Deserialize to AgentMessage
        # 3. Return the message
        return None
```

### Exercise 2: Advanced Security Policy

```python
class AdvancedSecurityPolicy:
    """Implement advanced security policies for agent interactions"""
    
    def __init__(self):
        self.rate_limits = {}  # agent_id -> rate limit info
        self.capability_restrictions = {}  # agent_id -> allowed capabilities
        self.trust_scores = {}  # agent_id -> trust score (0-100)
    
    def set_rate_limit(self, agent_id: str, requests_per_minute: int):
        """Set rate limit for an agent"""
        self.rate_limits[agent_id] = {
            "limit": requests_per_minute,
            "requests": [],
            "last_reset": datetime.now()
        }
    
    def check_rate_limit(self, agent_id: str) -> bool:
        """TODO: Implement rate limiting logic"""
        # Check if agent has exceeded rate limit
        # Return True if within limit, False otherwise
        return True
    
    def calculate_trust_score(self, agent_id: str, 
                            interaction_history: List[Dict]) -> float:
        """TODO: Implement trust score calculation"""
        # Analyze interaction history
        # Calculate trust score based on:
        # - Success rate of interactions
        # - Response times
        # - Error rates
        # - Feedback from other agents
        return 85.0  # Example score
    
    def authorize_capability_access(self, agent_id: str, 
                                  capability: str) -> bool:
        """TODO: Implement capability-based authorization"""
        # Check if agent is authorized to use specific capability
        # Consider trust score, permissions, and restrictions
        return True
```

### Exercise 3: Intelligent Agent Orchestrator

```python
class IntelligentAgentOrchestrator:
    """Orchestrate complex multi-agent workflows across platforms"""
    
    def __init__(self, federation: AgentFederation, 
                 router: CrossPlatformMessageRouter):
        self.federation = federation
        self.router = router
        self.workflow_templates = {}
        self.active_workflows = {}
    
    def define_workflow_template(self, template_id: str, 
                               workflow_definition: Dict[str, Any]):
        """Define a reusable workflow template"""
        self.workflow_templates[template_id] = workflow_definition
    
    async def execute_workflow(self, template_id: str, 
                             input_data: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: Implement intelligent workflow execution"""
        # 1. Load workflow template
        # 2. Discover and select best agents for each step
        # 3. Execute workflow with error handling and retries
        # 4. Aggregate and return results
        
        workflow_id = str(uuid.uuid4())
        self.active_workflows[workflow_id] = {
            "template_id": template_id,
            "status": "running",
            "steps_completed": 0,
            "results": {}
        }
        
        # Example workflow execution logic
        template = self.workflow_templates.get(template_id)
        if not template:
            raise ValueError(f"Workflow template {template_id} not found")
        
        results = {"workflow_id": workflow_id, "status": "completed"}
        return results
    
    async def optimize_agent_selection(self, required_capabilities: List[str],
                                     performance_history: Dict[str, Any]) -> List[str]:
        """TODO: Implement intelligent agent selection"""
        # Use ML/heuristics to select best agents based on:
        # - Historical performance
        # - Current load
        # - Capability match quality
        # - Trust scores
        return []
    
    async def handle_workflow_failures(self, workflow_id: str, 
                                     failed_step: str, 
                                     error: Exception):
        """TODO: Implement intelligent failure handling"""
        # 1. Analyze failure type
        # 2. Attempt recovery strategies
        # 3. Select alternative agents
        # 4. Resume or restart workflow
        pass
```

## Module Summary

In this module, you have learned how to build comprehensive interoperability solutions for multi-agent systems:

### Key Concepts Learned

1. **Standardized Communication Protocols**
   - Universal message formats for agent communication
   - Protocol adapters for different transport mechanisms
   - Message routing and transformation capabilities

2. **Agent Discovery and Registry Systems**
   - Centralized agent registration and discovery
   - Capability-based agent matching
   - Distributed registry federation

3. **Security and Trust Management**
   - Authentication and authorization mechanisms
   - Secure communication channels with encryption
   - Trust-based access control policies

4. **Federation and Cross-Platform Integration**
   - Multi-ecosystem agent federation
   - Cross-platform message routing
   - Protocol translation and adaptation

### Practical Skills Developed

1. **Building Interoperability Infrastructure**
   - Implementing protocol adapters
   - Creating agent registries
   - Setting up secure communication channels

2. **Security Implementation**
   - JWT-based authentication
   - Message signing and encryption
   - Permission-based access control

3. **Federation Management**
   - Multi-registry coordination
   - Cross-platform agent discovery
   - Intelligent message routing

4. **Advanced Orchestration**
   - Workflow-based agent coordination
   - Intelligent agent selection
   - Failure handling and recovery

### Real-World Applications

1. **Enterprise Integration**
   - Connecting agents across different departments
   - Legacy system integration
   - Multi-vendor agent ecosystems

2. **Cloud and Edge Computing**
   - Distributed agent deployments
   - Edge-to-cloud agent communication
   - Hybrid cloud architectures

3. **Industry Collaboration**
   - Cross-organization agent sharing
   - Industry-standard protocols
   - Regulatory compliance frameworks

4. **Research and Development**
   - Academic collaboration platforms
   - Open-source agent ecosystems
   - Experimental protocol development

### Next Steps

With a solid foundation in agent interoperability, you're ready to explore:

- **Module VIII: Advanced Agent Architectures** - Complex agent designs and patterns
- **Module IX: Production Deployment** - Scaling and deploying agent systems
- **Module X: Future Trends and Research** - Emerging technologies and research directions

The interoperability patterns and techniques you've learned will be essential for building robust, scalable, and collaborative agent ecosystems that can adapt to changing requirements and integrate with diverse technological landscapes.