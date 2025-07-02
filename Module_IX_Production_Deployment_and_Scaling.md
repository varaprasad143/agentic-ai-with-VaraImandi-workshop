# Module IX: Production Deployment and Scaling

## Learning Objectives

By the end of this module, you will be able to:

1. **Deploy Agent Systems to Production**
   - Design production-ready agent architectures
   - Implement containerization and orchestration
   - Set up CI/CD pipelines for agent systems

2. **Scale Multi-Agent Systems**
   - Implement horizontal and vertical scaling strategies
   - Design load balancing for agent workloads
   - Optimize resource allocation and utilization

3. **Ensure System Reliability**
   - Implement fault tolerance and recovery mechanisms
   - Design backup and disaster recovery strategies
   - Monitor system health and performance

4. **Manage Production Operations**
   - Implement configuration management
   - Set up logging and monitoring infrastructure
   - Handle security and compliance requirements

## Introduction to Production Deployment

Deploying multi-agent systems in production environments requires careful consideration of scalability, reliability, security, and maintainability. This module covers the essential practices and technologies needed to successfully deploy and operate agent systems at scale.

### Production Architecture Overview

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Production Agent System Architecture</text>
  
  <!-- Load Balancer -->
  <rect x="350" y="60" width="100" height="40" fill="#3498db" stroke="#2980b9" stroke-width="2" rx="5"/>
  <text x="400" y="85" text-anchor="middle" font-size="12" fill="white" font-weight="bold">Load Balancer</text>
  
  <!-- API Gateway -->
  <rect x="320" y="130" width="160" height="40" fill="#e74c3c" stroke="#c0392b" stroke-width="2" rx="5"/>
  <text x="400" y="155" text-anchor="middle" font-size="12" fill="white" font-weight="bold">API Gateway</text>
  
  <!-- Agent Clusters -->
  <g id="cluster1">
    <rect x="50" y="200" width="200" height="120" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2" rx="10"/>
    <text x="150" y="220" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Agent Cluster 1</text>
    <rect x="70" y="240" width="60" height="30" fill="#27ae60" stroke="#229954" rx="3"/>
    <text x="100" y="258" text-anchor="middle" font-size="10" fill="white">Agent A</text>
    <rect x="140" y="240" width="60" height="30" fill="#27ae60" stroke="#229954" rx="3"/>
    <text x="170" y="258" text-anchor="middle" font-size="10" fill="white">Agent B</text>
    <rect x="70" y="280" width="60" height="30" fill="#27ae60" stroke="#229954" rx="3"/>
    <text x="100" y="298" text-anchor="middle" font-size="10" fill="white">Agent C</text>
    <rect x="140" y="280" width="60" height="30" fill="#27ae60" stroke="#229954" rx="3"/>
    <text x="170" y="298" text-anchor="middle" font-size="10" fill="white">Agent D</text>
  </g>
  
  <g id="cluster2">
    <rect x="300" y="200" width="200" height="120" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2" rx="10"/>
    <text x="400" y="220" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Agent Cluster 2</text>
    <rect x="320" y="240" width="60" height="30" fill="#f39c12" stroke="#e67e22" rx="3"/>
    <text x="350" y="258" text-anchor="middle" font-size="10" fill="white">Agent E</text>
    <rect x="390" y="240" width="60" height="30" fill="#f39c12" stroke="#e67e22" rx="3"/>
    <text x="420" y="258" text-anchor="middle" font-size="10" fill="white">Agent F</text>
    <rect x="320" y="280" width="60" height="30" fill="#f39c12" stroke="#e67e22" rx="3"/>
    <text x="350" y="298" text-anchor="middle" font-size="10" fill="white">Agent G</text>
    <rect x="390" y="280" width="60" height="30" fill="#f39c12" stroke="#e67e22" rx="3"/>
    <text x="420" y="298" text-anchor="middle" font-size="10" fill="white">Agent H</text>
  </g>
  
  <g id="cluster3">
    <rect x="550" y="200" width="200" height="120" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2" rx="10"/>
    <text x="650" y="220" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Agent Cluster 3</text>
    <rect x="570" y="240" width="60" height="30" fill="#9b59b6" stroke="#8e44ad" rx="3"/>
    <text x="600" y="258" text-anchor="middle" font-size="10" fill="white">Agent I</text>
    <rect x="640" y="240" width="60" height="30" fill="#9b59b6" stroke="#8e44ad" rx="3"/>
    <text x="670" y="258" text-anchor="middle" font-size="10" fill="white">Agent J</text>
    <rect x="570" y="280" width="60" height="30" fill="#9b59b6" stroke="#8e44ad" rx="3"/>
    <text x="600" y="298" text-anchor="middle" font-size="10" fill="white">Agent K</text>
    <rect x="640" y="280" width="60" height="30" fill="#9b59b6" stroke="#8e44ad" rx="3"/>
    <text x="670" y="298" text-anchor="middle" font-size="10" fill="white">Agent L</text>
  </g>
  
  <!-- Message Queue -->
  <rect x="320" y="350" width="160" height="40" fill="#34495e" stroke="#2c3e50" stroke-width="2" rx="5"/>
  <text x="400" y="375" text-anchor="middle" font-size="12" fill="white" font-weight="bold">Message Queue</text>
  
  <!-- Databases -->
  <rect x="100" y="420" width="120" height="40" fill="#16a085" stroke="#138d75" stroke-width="2" rx="5"/>
  <text x="160" y="445" text-anchor="middle" font-size="12" fill="white" font-weight="bold">Primary DB</text>
  
  <rect x="250" y="420" width="120" height="40" fill="#16a085" stroke="#138d75" stroke-width="2" rx="5"/>
  <text x="310" y="445" text-anchor="middle" font-size="12" fill="white" font-weight="bold">Cache</text>
  
  <rect x="400" y="420" width="120" height="40" fill="#16a085" stroke="#138d75" stroke-width="2" rx="5"/>
  <text x="460" y="445" text-anchor="middle" font-size="12" fill="white" font-weight="bold">Analytics DB</text>
  
  <!-- Monitoring -->
  <rect x="580" y="420" width="120" height="40" fill="#e67e22" stroke="#d35400" stroke-width="2" rx="5"/>
  <text x="640" y="445" text-anchor="middle" font-size="12" fill="white" font-weight="bold">Monitoring</text>
  
  <!-- External Services -->
  <rect x="50" y="500" width="100" height="30" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" rx="5"/>
  <text x="100" y="520" text-anchor="middle" font-size="10" fill="white">External API</text>
  
  <rect x="200" y="500" width="100" height="30" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" rx="5"/>
  <text x="250" y="520" text-anchor="middle" font-size="10" fill="white">ML Services</text>
  
  <rect x="350" y="500" width="100" height="30" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" rx="5"/>
  <text x="400" y="520" text-anchor="middle" font-size="10" fill="white">Cloud Storage</text>
  
  <rect x="500" y="500" width="100" height="30" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" rx="5"/>
  <text x="550" y="520" text-anchor="middle" font-size="10" fill="white">Notification</text>
  
  <rect x="650" y="500" width="100" height="30" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2" rx="5"/>
  <text x="700" y="520" text-anchor="middle" font-size="10" fill="white">Security</text>
  
  <!-- Connections -->
  <line x1="400" y1="100" x2="400" y2="130" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="170" x2="150" y2="200" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="170" x2="400" y2="200" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="170" x2="650" y2="200" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <line x1="150" y1="320" x2="370" y2="350" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="320" x2="400" y2="350" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="650" y1="320" x2="430" y2="350" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <line x1="370" y1="390" x2="160" y2="420" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="390" x2="310" y2="420" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="430" y1="390" x2="460" y2="420" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="450" y1="390" x2="640" y2="420" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Arrow marker -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Legend -->
  <text x="50" y="580" font-size="12" font-weight="bold" fill="#2c3e50">Legend:</text>
  <rect x="120" y="570" width="15" height="15" fill="#3498db"/>
  <text x="140" y="582" font-size="10" fill="#2c3e50">Load Balancer</text>
  <rect x="220" y="570" width="15" height="15" fill="#27ae60"/>
  <text x="240" y="582" font-size="10" fill="#2c3e50">Agents</text>
  <rect x="300" y="570" width="15" height="15" fill="#16a085"/>
  <text x="320" y="582" font-size="10" fill="#2c3e50">Storage</text>
  <rect x="380" y="570" width="15" height="15" fill="#34495e"/>
  <text x="400" y="582" font-size="10" fill="#2c3e50">Messaging</text>
  <rect x="470" y="570" width="15" height="15" fill="#95a5a6"/>
  <text x="490" y="582" font-size="10" fill="#2c3e50">External</text>
</svg>
```

## 1. Containerization and Orchestration

### 1.1 Docker Containerization

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import docker
import yaml
import json
import os
from datetime import datetime

class ContainerStatus(Enum):
    """Container status enumeration"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"

@dataclass
class ContainerConfig:
    """Configuration for agent containers"""
    image_name: str
    tag: str = "latest"
    environment_vars: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, str] = field(default_factory=dict)  # container_port: host_port
    volumes: Dict[str, str] = field(default_factory=dict)  # host_path: container_path
    memory_limit: str = "512m"
    cpu_limit: str = "0.5"
    restart_policy: str = "unless-stopped"
    network: str = "agent-network"
    labels: Dict[str, str] = field(default_factory=dict)
    health_check: Optional[Dict[str, Any]] = None

@dataclass
class AgentContainer:
    """Represents a containerized agent"""
    container_id: str
    agent_id: str
    config: ContainerConfig
    status: ContainerStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

class AgentContainerManager:
    """Manages containerized agents using Docker"""
    
    def __init__(self, docker_client: Optional[docker.DockerClient] = None):
        self.docker_client = docker_client or docker.from_env()
        self.containers: Dict[str, AgentContainer] = {}
        self.network_name = "agent-network"
        self.registry_url = "localhost:5000"  # Local registry
        
        # Ensure network exists
        self.ensure_network()
    
    def ensure_network(self):
        """Ensure the agent network exists"""
        try:
            self.docker_client.networks.get(self.network_name)
        except docker.errors.NotFound:
            self.docker_client.networks.create(
                self.network_name,
                driver="bridge",
                labels={"purpose": "agent-communication"}
            )
    
    def build_agent_image(self, agent_type: str, dockerfile_path: str, 
                         build_context: str = ".") -> str:
        """Build Docker image for an agent type"""
        image_tag = f"{self.registry_url}/agent-{agent_type}:latest"
        
        try:
            # Build the image
            image, build_logs = self.docker_client.images.build(
                path=build_context,
                dockerfile=dockerfile_path,
                tag=image_tag,
                rm=True,
                forcerm=True
            )
            
            # Push to registry
            self.docker_client.images.push(image_tag)
            
            return image_tag
        except Exception as e:
            raise RuntimeError(f"Failed to build image for {agent_type}: {str(e)}")
    
    async def deploy_agent(self, agent_id: str, config: ContainerConfig) -> AgentContainer:
        """Deploy an agent in a container"""
        try:
            # Prepare container configuration
            container_config = {
                "image": f"{config.image_name}:{config.tag}",
                "name": f"agent-{agent_id}",
                "environment": config.environment_vars,
                "ports": config.ports,
                "volumes": config.volumes,
                "mem_limit": config.memory_limit,
                "cpu_period": 100000,
                "cpu_quota": int(float(config.cpu_limit) * 100000),
                "restart_policy": {"Name": config.restart_policy},
                "network": config.network,
                "labels": {**config.labels, "agent_id": agent_id},
                "detach": True
            }
            
            # Add health check if specified
            if config.health_check:
                container_config["healthcheck"] = config.health_check
            
            # Create and start container
            container = self.docker_client.containers.run(**container_config)
            
            # Create agent container record
            agent_container = AgentContainer(
                container_id=container.id,
                agent_id=agent_id,
                config=config,
                status=ContainerStatus.RUNNING,
                created_at=datetime.now(),
                started_at=datetime.now()
            )
            
            self.containers[agent_id] = agent_container
            
            return agent_container
            
        except Exception as e:
            raise RuntimeError(f"Failed to deploy agent {agent_id}: {str(e)}")
    
    async def scale_agent_type(self, agent_type: str, target_count: int, 
                              base_config: ContainerConfig) -> List[AgentContainer]:
        """Scale a specific agent type to target count"""
        current_agents = [ac for ac in self.containers.values() 
                         if ac.config.labels.get("agent_type") == agent_type]
        current_count = len(current_agents)
        
        deployed_containers = []
        
        if target_count > current_count:
            # Scale up
            for i in range(current_count, target_count):
                agent_id = f"{agent_type}_{i:03d}"
                
                # Customize config for this instance
                instance_config = ContainerConfig(
                    image_name=base_config.image_name,
                    tag=base_config.tag,
                    environment_vars={**base_config.environment_vars, "AGENT_ID": agent_id},
                    ports={port: str(int(host_port) + i) for port, host_port in base_config.ports.items()},
                    volumes=base_config.volumes,
                    memory_limit=base_config.memory_limit,
                    cpu_limit=base_config.cpu_limit,
                    restart_policy=base_config.restart_policy,
                    network=base_config.network,
                    labels={**base_config.labels, "agent_type": agent_type, "instance": str(i)},
                    health_check=base_config.health_check
                )
                
                container = await self.deploy_agent(agent_id, instance_config)
                deployed_containers.append(container)
        
        elif target_count < current_count:
            # Scale down
            agents_to_remove = current_agents[target_count:]
            for agent_container in agents_to_remove:
                await self.stop_agent(agent_container.agent_id)
        
        return deployed_containers
    
    async def stop_agent(self, agent_id: str, timeout: int = 30) -> bool:
        """Stop and remove an agent container"""
        if agent_id not in self.containers:
            return False
        
        try:
            container = self.docker_client.containers.get(
                self.containers[agent_id].container_id
            )
            
            # Stop container gracefully
            container.stop(timeout=timeout)
            
            # Remove container
            container.remove()
            
            # Update status
            self.containers[agent_id].status = ContainerStatus.EXITED
            self.containers[agent_id].finished_at = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"Error stopping agent {agent_id}: {str(e)}")
            return False
    
    def get_agent_logs(self, agent_id: str, tail: int = 100) -> List[str]:
        """Get logs from an agent container"""
        if agent_id not in self.containers:
            return []
        
        try:
            container = self.docker_client.containers.get(
                self.containers[agent_id].container_id
            )
            
            logs = container.logs(tail=tail, timestamps=True).decode('utf-8')
            return logs.split('\n')
            
        except Exception as e:
            return [f"Error retrieving logs: {str(e)}"]
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get resource metrics for an agent container"""
        if agent_id not in self.containers:
            return {}
        
        try:
            container = self.docker_client.containers.get(
                self.containers[agent_id].container_id
            )
            
            stats = container.stats(stream=False)
            
            # Calculate CPU usage percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * \
                             len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100
            
            return {
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_mb": round(memory_usage / 1024 / 1024, 2),
                "memory_limit_mb": round(memory_limit / 1024 / 1024, 2),
                "memory_percent": round(memory_percent, 2),
                "network_rx_bytes": stats['networks']['eth0']['rx_bytes'],
                "network_tx_bytes": stats['networks']['eth0']['tx_bytes'],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        total_containers = len(self.containers)
        running_containers = sum(1 for c in self.containers.values() 
                               if c.status == ContainerStatus.RUNNING)
        
        # Aggregate metrics
        total_cpu = 0.0
        total_memory_mb = 0.0
        
        for agent_id in self.containers:
            metrics = self.get_agent_metrics(agent_id)
            if "cpu_percent" in metrics:
                total_cpu += metrics["cpu_percent"]
                total_memory_mb += metrics["memory_usage_mb"]
        
        return {
            "total_containers": total_containers,
            "running_containers": running_containers,
            "stopped_containers": total_containers - running_containers,
            "cluster_cpu_percent": round(total_cpu, 2),
            "cluster_memory_mb": round(total_memory_mb, 2),
            "network_name": self.network_name,
            "registry_url": self.registry_url
        }
```

### 1.2 Kubernetes Orchestration

```python
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import base64

class KubernetesAgentOrchestrator:
    """Orchestrates agents using Kubernetes"""
    
    def __init__(self, namespace: str = "agent-system"):
        # Load Kubernetes configuration
        try:
            config.load_incluster_config()  # For in-cluster execution
        except:
            config.load_kube_config()  # For local development
        
        self.namespace = namespace
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        
        # Ensure namespace exists
        self.ensure_namespace()
    
    def ensure_namespace(self):
        """Ensure the agent namespace exists"""
        try:
            self.core_v1.read_namespace(name=self.namespace)
        except ApiException as e:
            if e.status == 404:
                namespace_body = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=self.namespace)
                )
                self.core_v1.create_namespace(body=namespace_body)
    
    def create_agent_deployment(self, agent_type: str, replicas: int, 
                               image: str, config: Dict[str, Any]) -> str:
        """Create a Kubernetes deployment for an agent type"""
        
        # Define container
        container = client.V1Container(
            name=f"agent-{agent_type}",
            image=image,
            ports=[client.V1ContainerPort(container_port=8080)],
            env=[
                client.V1EnvVar(name=k, value=v) 
                for k, v in config.get("environment", {}).items()
            ],
            resources=client.V1ResourceRequirements(
                requests={
                    "memory": config.get("memory_request", "256Mi"),
                    "cpu": config.get("cpu_request", "100m")
                },
                limits={
                    "memory": config.get("memory_limit", "512Mi"),
                    "cpu": config.get("cpu_limit", "500m")
                }
            ),
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path="/health",
                    port=8080
                ),
                initial_delay_seconds=30,
                period_seconds=10
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path="/ready",
                    port=8080
                ),
                initial_delay_seconds=5,
                period_seconds=5
            )
        )
        
        # Define pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": f"agent-{agent_type}",
                    "version": "v1",
                    "component": "agent"
                }
            ),
            spec=client.V1PodSpec(
                containers=[container],
                restart_policy="Always"
            )
        )
        
        # Define deployment
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=f"agent-{agent_type}-deployment",
                namespace=self.namespace
            ),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"agent-{agent_type}"}
                ),
                template=pod_template,
                strategy=client.V1DeploymentStrategy(
                    type="RollingUpdate",
                    rolling_update=client.V1RollingUpdateDeployment(
                        max_surge="25%",
                        max_unavailable="25%"
                    )
                )
            )
        )
        
        # Create deployment
        try:
            response = self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            return response.metadata.name
        except ApiException as e:
            raise RuntimeError(f"Failed to create deployment: {e}")
    
    def create_agent_service(self, agent_type: str, port: int = 8080) -> str:
        """Create a Kubernetes service for an agent type"""
        
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=f"agent-{agent_type}-service",
                namespace=self.namespace
            ),
            spec=client.V1ServiceSpec(
                selector={"app": f"agent-{agent_type}"},
                ports=[
                    client.V1ServicePort(
                        port=port,
                        target_port=8080,
                        protocol="TCP"
                    )
                ],
                type="ClusterIP"
            )
        )
        
        try:
            response = self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            return response.metadata.name
        except ApiException as e:
            raise RuntimeError(f"Failed to create service: {e}")
    
    def create_horizontal_pod_autoscaler(self, agent_type: str, 
                                       min_replicas: int = 1, 
                                       max_replicas: int = 10, 
                                       target_cpu_percent: int = 70) -> str:
        """Create HPA for automatic scaling"""
        
        hpa = client.V1HorizontalPodAutoscaler(
            api_version="autoscaling/v1",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(
                name=f"agent-{agent_type}-hpa",
                namespace=self.namespace
            ),
            spec=client.V1HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V1CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=f"agent-{agent_type}-deployment"
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                target_cpu_utilization_percentage=target_cpu_percent
            )
        )
        
        try:
            response = self.autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
            return response.metadata.name
        except ApiException as e:
            raise RuntimeError(f"Failed to create HPA: {e}")
    
    def scale_deployment(self, agent_type: str, replicas: int) -> bool:
        """Manually scale a deployment"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=f"agent-{agent_type}-deployment",
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=f"agent-{agent_type}-deployment",
                namespace=self.namespace,
                body=deployment
            )
            
            return True
        except ApiException as e:
            print(f"Failed to scale deployment: {e}")
            return False
    
    def get_deployment_status(self, agent_type: str) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=f"agent-{agent_type}-deployment",
                namespace=self.namespace
            )
            
            return {
                "name": deployment.metadata.name,
                "namespace": deployment.metadata.namespace,
                "replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "updated_replicas": deployment.status.updated_replicas or 0,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message
                    }
                    for condition in (deployment.status.conditions or [])
                ]
            }
        except ApiException as e:
            return {"error": str(e)}
    
    def get_pod_logs(self, agent_type: str, lines: int = 100) -> Dict[str, List[str]]:
        """Get logs from all pods of an agent type"""
        try:
            # Get pods for the deployment
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app=agent-{agent_type}"
            )
            
            pod_logs = {}
            for pod in pods.items:
                try:
                    logs = self.core_v1.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=self.namespace,
                        tail_lines=lines
                    )
                    pod_logs[pod.metadata.name] = logs.split('\n')
                except ApiException as e:
                    pod_logs[pod.metadata.name] = [f"Error retrieving logs: {e}"]
            
            return pod_logs
        except ApiException as e:
            return {"error": [str(e)]}
    
    def delete_agent_deployment(self, agent_type: str) -> bool:
        """Delete an agent deployment and associated resources"""
        try:
            # Delete HPA
            try:
                self.autoscaling_v1.delete_namespaced_horizontal_pod_autoscaler(
                    name=f"agent-{agent_type}-hpa",
                    namespace=self.namespace
                )
            except ApiException:
                pass  # HPA might not exist
            
            # Delete service
            try:
                self.core_v1.delete_namespaced_service(
                    name=f"agent-{agent_type}-service",
                    namespace=self.namespace
                )
            except ApiException:
                pass  # Service might not exist
            
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=f"agent-{agent_type}-deployment",
                namespace=self.namespace
            )
            
            return True
        except ApiException as e:
            print(f"Failed to delete deployment: {e}")
            return False
```

## 2. Load Balancing and Traffic Management

### 2.1 Agent Load Balancer

```python
import asyncio
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import statistics

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    CONSISTENT_HASHING = "consistent_hashing"

@dataclass
class AgentEndpoint:
    """Represents an agent endpoint for load balancing"""
    agent_id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    last_health_check: Optional[float] = None
    is_healthy: bool = True
    capabilities: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times[-100:])  # Last 100 requests
    
    @property
    def load_score(self) -> float:
        """Calculate load score (lower is better)"""
        connection_load = self.current_connections / self.max_connections
        response_time_factor = min(self.average_response_time / 1000, 1.0)  # Normalize to seconds
        failure_rate = 1 - self.success_rate
        
        return (connection_load * 0.4 + response_time_factor * 0.4 + failure_rate * 0.2)

class AgentLoadBalancer:
    """Load balancer for distributing requests across agent instances"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.endpoints: Dict[str, AgentEndpoint] = {}
        self.round_robin_index = 0
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5  # seconds
        self.max_response_time_history = 100
        
        # Start health checking
        asyncio.create_task(self.health_check_loop())
    
    def register_endpoint(self, endpoint: AgentEndpoint):
        """Register an agent endpoint"""
        self.endpoints[endpoint.agent_id] = endpoint
        print(f"Registered endpoint: {endpoint.agent_id} at {endpoint.host}:{endpoint.port}")
    
    def unregister_endpoint(self, agent_id: str):
        """Unregister an agent endpoint"""
        if agent_id in self.endpoints:
            del self.endpoints[agent_id]
            print(f"Unregistered endpoint: {agent_id}")
    
    def get_healthy_endpoints(self, required_capabilities: Optional[List[str]] = None) -> List[AgentEndpoint]:
        """Get list of healthy endpoints that match required capabilities"""
        healthy_endpoints = [ep for ep in self.endpoints.values() if ep.is_healthy]
        
        if required_capabilities:
            # Filter by capabilities
            capable_endpoints = []
            for endpoint in healthy_endpoints:
                if all(cap in endpoint.capabilities for cap in required_capabilities):
                    capable_endpoints.append(endpoint)
            return capable_endpoints
        
        return healthy_endpoints
    
    async def select_endpoint(self, required_capabilities: Optional[List[str]] = None) -> Optional[AgentEndpoint]:
        """Select an endpoint based on the load balancing strategy"""
        healthy_endpoints = self.get_healthy_endpoints(required_capabilities)
        
        if not healthy_endpoints:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASHING:
            return self._consistent_hashing_selection(healthy_endpoints)
        else:
            return random.choice(healthy_endpoints)
    
    def _round_robin_selection(self, endpoints: List[AgentEndpoint]) -> AgentEndpoint:
        """Round-robin selection"""
        endpoint = endpoints[self.round_robin_index % len(endpoints)]
        self.round_robin_index += 1
        return endpoint
    
    def _least_connections_selection(self, endpoints: List[AgentEndpoint]) -> AgentEndpoint:
        """Select endpoint with least connections"""
        return min(endpoints, key=lambda ep: ep.current_connections)
    
    def _weighted_round_robin_selection(self, endpoints: List[AgentEndpoint]) -> AgentEndpoint:
        """Weighted round-robin selection"""
        total_weight = sum(ep.weight for ep in endpoints)
        random_weight = random.uniform(0, total_weight)
        
        current_weight = 0
        for endpoint in endpoints:
            current_weight += endpoint.weight
            if random_weight <= current_weight:
                return endpoint
        
        return endpoints[-1]  # Fallback
    
    def _least_response_time_selection(self, endpoints: List[AgentEndpoint]) -> AgentEndpoint:
        """Select endpoint with least average response time"""
        return min(endpoints, key=lambda ep: ep.average_response_time)
    
    def _resource_based_selection(self, endpoints: List[AgentEndpoint]) -> AgentEndpoint:
        """Select endpoint based on overall load score"""
        return min(endpoints, key=lambda ep: ep.load_score)
    
    def _consistent_hashing_selection(self, endpoints: List[AgentEndpoint], 
                                    key: str = None) -> AgentEndpoint:
        """Consistent hashing selection (simplified)"""
        if not key:
            key = str(time.time())
        
        # Simple hash-based selection
        hash_value = hash(key) % len(endpoints)
        return endpoints[hash_value]
    
    async def execute_request(self, request_data: Dict[str, Any], 
                            required_capabilities: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a request through the load balancer"""
        endpoint = await self.select_endpoint(required_capabilities)
        
        if not endpoint:
            return {
                "error": "No healthy endpoints available",
                "required_capabilities": required_capabilities
            }
        
        # Track connection
        endpoint.current_connections += 1
        endpoint.total_requests += 1
        
        start_time = time.time()
        
        try:
            # Simulate request execution (replace with actual HTTP/gRPC call)
            result = await self._execute_agent_request(endpoint, request_data)
            
            # Record successful response time
            response_time = (time.time() - start_time) * 1000  # milliseconds
            endpoint.response_times.append(response_time)
            
            # Limit response time history
            if len(endpoint.response_times) > self.max_response_time_history:
                endpoint.response_times = endpoint.response_times[-self.max_response_time_history:]
            
            return {
                "result": result,
                "endpoint": endpoint.agent_id,
                "response_time_ms": response_time
            }
            
        except Exception as e:
            endpoint.failed_requests += 1
            return {
                "error": str(e),
                "endpoint": endpoint.agent_id,
                "response_time_ms": (time.time() - start_time) * 1000
            }
        
        finally:
            endpoint.current_connections -= 1
    
    async def _execute_agent_request(self, endpoint: AgentEndpoint, 
                                   request_data: Dict[str, Any]) -> Any:
        """Execute request to specific agent endpoint"""
        # This would be replaced with actual HTTP/gRPC client call
        # For demonstration, simulate processing time
        processing_time = random.uniform(0.1, 2.0)  # 100ms to 2s
        await asyncio.sleep(processing_time)
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated agent failure")
        
        return {
            "processed_by": endpoint.agent_id,
            "request_id": request_data.get("request_id", "unknown"),
            "result": "Request processed successfully",
            "processing_time": processing_time
        }
    
    async def health_check_loop(self):
        """Continuous health checking of endpoints"""
        while True:
            await asyncio.sleep(self.health_check_interval)
            await self.perform_health_checks()
    
    async def perform_health_checks(self):
        """Perform health checks on all endpoints"""
        for endpoint in self.endpoints.values():
            try:
                # Simulate health check (replace with actual health check)
                health_check_start = time.time()
                is_healthy = await self._check_endpoint_health(endpoint)
                health_check_time = time.time() - health_check_start
                
                endpoint.is_healthy = is_healthy and health_check_time < self.health_check_timeout
                endpoint.last_health_check = time.time()
                
                if not endpoint.is_healthy:
                    print(f"Endpoint {endpoint.agent_id} failed health check")
                
            except Exception as e:
                endpoint.is_healthy = False
                print(f"Health check error for {endpoint.agent_id}: {e}")
    
    async def _check_endpoint_health(self, endpoint: AgentEndpoint) -> bool:
        """Check health of a specific endpoint"""
        # Simulate health check with random success/failure
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return random.random() > 0.1  # 90% success rate
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        total_requests = sum(ep.total_requests for ep in self.endpoints.values())
        total_failures = sum(ep.failed_requests for ep in self.endpoints.values())
        healthy_endpoints = len([ep for ep in self.endpoints.values() if ep.is_healthy])
        
        endpoint_stats = []
        for endpoint in self.endpoints.values():
            endpoint_stats.append({
                "agent_id": endpoint.agent_id,
                "host": endpoint.host,
                "port": endpoint.port,
                "is_healthy": endpoint.is_healthy,
                "current_connections": endpoint.current_connections,
                "total_requests": endpoint.total_requests,
                "failed_requests": endpoint.failed_requests,
                "success_rate": endpoint.success_rate,
                "average_response_time_ms": endpoint.average_response_time,
                "load_score": endpoint.load_score
            })
        
        return {
            "strategy": self.strategy.value,
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": healthy_endpoints,
            "total_requests": total_requests,
            "total_failures": total_failures,
            "overall_success_rate": (total_requests - total_failures) / total_requests if total_requests > 0 else 1.0,
            "endpoints": endpoint_stats
        }
```

## 3. Auto-Scaling and Resource Management

### 3.1 Dynamic Auto-Scaling System

```python
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

class ScalingDirection(Enum):
    """Scaling direction enumeration"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class MetricType(Enum):
    """Types of metrics for scaling decisions"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"

@dataclass
class ScalingMetric:
    """Represents a metric used for scaling decisions"""
    metric_type: MetricType
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    window_size: int = 5  # Number of data points to consider
    history: List[float] = field(default_factory=list)
    
    def add_value(self, value: float):
        """Add a new metric value"""
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        self.current_value = value
    
    @property
    def average_value(self) -> float:
        """Get average value over the window"""
        return statistics.mean(self.history) if self.history else 0.0
    
    @property
    def trend(self) -> float:
        """Calculate trend (positive = increasing, negative = decreasing)"""
        if len(self.history) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(self.history)
        x_sum = sum(range(n))
        y_sum = sum(self.history)
        xy_sum = sum(i * self.history[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope

@dataclass
class ScalingRule:
    """Defines a scaling rule"""
    name: str
    metrics: List[ScalingMetric]
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    scale_up_step: int = 1
    scale_down_step: int = 1
    evaluation_period: int = 60  # seconds
    
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed"""
        for metric in self.metrics:
            if metric.average_value > metric.threshold_up:
                return True
        return False
    
    def should_scale_down(self) -> bool:
        """Determine if scaling down is needed"""
        for metric in self.metrics:
            if metric.average_value < metric.threshold_down:
                return True
        return False
    
    def calculate_scaling_score(self) -> float:
        """Calculate overall scaling score"""
        total_score = 0.0
        total_weight = 0.0
        
        for metric in self.metrics:
            # Normalize metric value relative to thresholds
            if metric.average_value > metric.threshold_up:
                score = (metric.average_value - metric.threshold_up) / metric.threshold_up
            elif metric.average_value < metric.threshold_down:
                score = (metric.threshold_down - metric.average_value) / metric.threshold_down * -1
            else:
                score = 0.0
            
            total_score += score * metric.weight
            total_weight += metric.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

class AutoScaler:
    """Automatic scaling system for agent clusters"""
    
    def __init__(self, container_manager, orchestrator=None):
        self.container_manager = container_manager
        self.orchestrator = orchestrator
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.last_scaling_action: Dict[str, float] = {}
        self.current_instances: Dict[str, int] = {}
        self.scaling_history: List[Dict[str, Any]] = []
        self.is_running = False
        
        # Metrics collection
        self.metrics_collectors: Dict[str, Callable] = {}
        
    def add_scaling_rule(self, agent_type: str, rule: ScalingRule):
        """Add a scaling rule for an agent type"""
        self.scaling_rules[agent_type] = rule
        self.current_instances[agent_type] = rule.min_instances
        print(f"Added scaling rule for {agent_type}: {rule.name}")
    
    def register_metrics_collector(self, agent_type: str, collector: Callable):
        """Register a metrics collector function"""
        self.metrics_collectors[agent_type] = collector
    
    async def start_auto_scaling(self):
        """Start the auto-scaling loop"""
        self.is_running = True
        print("Auto-scaler started")
        
        while self.is_running:
            try:
                await self.evaluate_scaling_decisions()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
            except Exception as e:
                print(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def stop_auto_scaling(self):
        """Stop the auto-scaling loop"""
        self.is_running = False
        print("Auto-scaler stopped")
    
    async def evaluate_scaling_decisions(self):
        """Evaluate scaling decisions for all agent types"""
        for agent_type, rule in self.scaling_rules.items():
            try:
                # Collect current metrics
                await self.collect_metrics(agent_type, rule)
                
                # Make scaling decision
                scaling_decision = await self.make_scaling_decision(agent_type, rule)
                
                if scaling_decision != ScalingDirection.STABLE:
                    await self.execute_scaling_action(agent_type, rule, scaling_decision)
                    
            except Exception as e:
                print(f"Error evaluating scaling for {agent_type}: {e}")
    
    async def collect_metrics(self, agent_type: str, rule: ScalingRule):
        """Collect metrics for an agent type"""
        if agent_type in self.metrics_collectors:
            try:
                metrics_data = await self.metrics_collectors[agent_type]()
                
                # Update rule metrics with collected data
                for metric in rule.metrics:
                    if metric.metric_type.value in metrics_data:
                        metric.add_value(metrics_data[metric.metric_type.value])
                        
            except Exception as e:
                print(f"Error collecting metrics for {agent_type}: {e}")
    
    async def make_scaling_decision(self, agent_type: str, rule: ScalingRule) -> ScalingDirection:
        """Make scaling decision based on metrics"""
        current_time = time.time()
        last_action_time = self.last_scaling_action.get(agent_type, 0)
        current_count = self.current_instances.get(agent_type, rule.min_instances)
        
        # Check cooldown periods
        if rule.should_scale_up():
            if (current_time - last_action_time) >= rule.scale_up_cooldown:
                if current_count < rule.max_instances:
                    return ScalingDirection.UP
        
        elif rule.should_scale_down():
            if (current_time - last_action_time) >= rule.scale_down_cooldown:
                if current_count > rule.min_instances:
                    return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    async def execute_scaling_action(self, agent_type: str, rule: ScalingRule, 
                                   direction: ScalingDirection):
        """Execute the scaling action"""
        current_count = self.current_instances.get(agent_type, rule.min_instances)
        
        if direction == ScalingDirection.UP:
            new_count = min(current_count + rule.scale_up_step, rule.max_instances)
        else:  # ScalingDirection.DOWN
            new_count = max(current_count - rule.scale_down_step, rule.min_instances)
        
        if new_count != current_count:
            try:
                # Execute scaling through container manager or orchestrator
                if self.orchestrator:
                    success = self.orchestrator.scale_deployment(agent_type, new_count)
                else:
                    # Use container manager for scaling
                    success = await self._scale_containers(agent_type, new_count)
                
                if success:
                    self.current_instances[agent_type] = new_count
                    self.last_scaling_action[agent_type] = time.time()
                    
                    # Record scaling action
                    scaling_record = {
                        "timestamp": time.time(),
                        "agent_type": agent_type,
                        "direction": direction.value,
                        "from_count": current_count,
                        "to_count": new_count,
                        "metrics": {
                            metric.metric_type.value: metric.average_value 
                            for metric in rule.metrics
                        },
                        "scaling_score": rule.calculate_scaling_score()
                    }
                    
                    self.scaling_history.append(scaling_record)
                    
                    print(f"Scaled {agent_type} {direction.value}: {current_count} -> {new_count}")
                else:
                    print(f"Failed to scale {agent_type} {direction.value}")
                    
            except Exception as e:
                print(f"Error executing scaling action for {agent_type}: {e}")
    
    async def _scale_containers(self, agent_type: str, target_count: int) -> bool:
        """Scale containers using container manager"""
        try:
            # This would integrate with the container manager
            # For demonstration, we'll simulate the scaling
            await asyncio.sleep(1)  # Simulate scaling time
            return True
        except Exception as e:
            print(f"Container scaling error: {e}")
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        status = {
            "is_running": self.is_running,
            "agent_types": {},
            "recent_actions": self.scaling_history[-10:] if self.scaling_history else []
        }
        
        for agent_type, rule in self.scaling_rules.items():
            current_count = self.current_instances.get(agent_type, rule.min_instances)
            last_action = self.last_scaling_action.get(agent_type, 0)
            
            status["agent_types"][agent_type] = {
                "current_instances": current_count,
                "min_instances": rule.min_instances,
                "max_instances": rule.max_instances,
                "last_scaling_action": last_action,
                "scaling_score": rule.calculate_scaling_score(),
                "metrics": {
                    metric.metric_type.value: {
                        "current": metric.current_value,
                        "average": metric.average_value,
                        "threshold_up": metric.threshold_up,
                        "threshold_down": metric.threshold_down,
                        "trend": metric.trend
                    }
                    for metric in rule.metrics
                }
            }
        
        return status

### 3.2 Resource Optimization

class ResourceOptimizer:
    """Optimizes resource allocation across agent clusters"""
    
    def __init__(self, auto_scaler: AutoScaler):
        self.auto_scaler = auto_scaler
        self.resource_pools: Dict[str, Dict[str, float]] = {}
        self.allocation_history: List[Dict[str, Any]] = []
        
    def register_resource_pool(self, pool_name: str, total_cpu: float, 
                              total_memory: float, total_storage: float):
        """Register a resource pool"""
        self.resource_pools[pool_name] = {
            "total_cpu": total_cpu,
            "total_memory": total_memory,
            "total_storage": total_storage,
            "allocated_cpu": 0.0,
            "allocated_memory": 0.0,
            "allocated_storage": 0.0,
            "agents": {}
        }
    
    async def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation across all pools"""
        optimization_results = {}
        
        for pool_name, pool in self.resource_pools.items():
            # Calculate current utilization
            cpu_utilization = pool["allocated_cpu"] / pool["total_cpu"]
            memory_utilization = pool["allocated_memory"] / pool["total_memory"]
            storage_utilization = pool["allocated_storage"] / pool["total_storage"]
            
            # Identify optimization opportunities
            recommendations = []
            
            if cpu_utilization > 0.8:
                recommendations.append({
                    "type": "scale_up",
                    "resource": "cpu",
                    "reason": "High CPU utilization",
                    "current_utilization": cpu_utilization
                })
            
            if memory_utilization > 0.8:
                recommendations.append({
                    "type": "scale_up",
                    "resource": "memory",
                    "reason": "High memory utilization",
                    "current_utilization": memory_utilization
                })
            
            if cpu_utilization < 0.3 and memory_utilization < 0.3:
                recommendations.append({
                    "type": "scale_down",
                    "resource": "general",
                    "reason": "Low resource utilization",
                    "cpu_utilization": cpu_utilization,
                    "memory_utilization": memory_utilization
                })
            
            optimization_results[pool_name] = {
                "utilization": {
                    "cpu": cpu_utilization,
                    "memory": memory_utilization,
                    "storage": storage_utilization
                },
                "recommendations": recommendations,
                "efficiency_score": self._calculate_efficiency_score(pool)
            }
        
        return optimization_results
    
    def _calculate_efficiency_score(self, pool: Dict[str, Any]) -> float:
        """Calculate efficiency score for a resource pool"""
        cpu_util = pool["allocated_cpu"] / pool["total_cpu"]
        memory_util = pool["allocated_memory"] / pool["total_memory"]
        storage_util = pool["allocated_storage"] / pool["total_storage"]
        
        # Optimal utilization is around 70%
        optimal_util = 0.7
        
        cpu_efficiency = 1 - abs(cpu_util - optimal_util) / optimal_util
        memory_efficiency = 1 - abs(memory_util - optimal_util) / optimal_util
        storage_efficiency = 1 - abs(storage_util - optimal_util) / optimal_util
        
        return (cpu_efficiency + memory_efficiency + storage_efficiency) / 3
    
    async def allocate_resources(self, agent_type: str, pool_name: str, 
                               cpu_request: float, memory_request: float, 
                               storage_request: float) -> bool:
        """Allocate resources for an agent"""
        if pool_name not in self.resource_pools:
            return False
        
        pool = self.resource_pools[pool_name]
        
        # Check if resources are available
        if (pool["allocated_cpu"] + cpu_request > pool["total_cpu"] or
            pool["allocated_memory"] + memory_request > pool["total_memory"] or
            pool["allocated_storage"] + storage_request > pool["total_storage"]):
            return False
        
        # Allocate resources
        pool["allocated_cpu"] += cpu_request
        pool["allocated_memory"] += memory_request
        pool["allocated_storage"] += storage_request
        
        # Track allocation
        pool["agents"][agent_type] = {
            "cpu": cpu_request,
            "memory": memory_request,
            "storage": storage_request,
            "allocated_at": time.time()
        }
        
        return True
    
    def deallocate_resources(self, agent_type: str, pool_name: str) -> bool:
        """Deallocate resources for an agent"""
        if pool_name not in self.resource_pools:
            return False
        
        pool = self.resource_pools[pool_name]
        
        if agent_type not in pool["agents"]:
            return False
        
        allocation = pool["agents"][agent_type]
        
        # Deallocate resources
        pool["allocated_cpu"] -= allocation["cpu"]
        pool["allocated_memory"] -= allocation["memory"]
        pool["allocated_storage"] -= allocation["storage"]
        
        # Remove tracking
        del pool["agents"][agent_type]
        
        return True
```

## 4. Monitoring and Observability

### 4.1 Comprehensive Monitoring System

```python
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics
import logging

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricAggregation(Enum):
    """Metric aggregation methods"""
    AVERAGE = "average"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"

@dataclass
class MetricPoint:
    """Represents a single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Represents an alert"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    threshold: float
    current_value: float
    agent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    is_resolved: bool = False
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Defines an alerting rule"""
    name: str
    metric_name: str
    condition: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: float
    severity: AlertSeverity
    duration: int = 300  # seconds - how long condition must be true
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    
    def evaluate(self, metric_value: float, duration_met: bool) -> bool:
        """Evaluate if alert should be triggered"""
        condition_met = False
        
        if self.condition == ">":
            condition_met = metric_value > self.threshold
        elif self.condition == "<":
            condition_met = metric_value < self.threshold
        elif self.condition == ">=":
            condition_met = metric_value >= self.threshold
        elif self.condition == "<=":
            condition_met = metric_value <= self.threshold
        elif self.condition == "==":
            condition_met = abs(metric_value - self.threshold) < 0.001
        elif self.condition == "!=":
            condition_met = abs(metric_value - self.threshold) >= 0.001
        
        return condition_met and duration_met

class MetricsCollector:
    """Collects and stores metrics from agents"""
    
    def __init__(self, retention_period: int = 86400):  # 24 hours
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.retention_period = retention_period
        self.collectors: Dict[str, Callable] = {}
        
    def register_collector(self, metric_name: str, collector_func: Callable):
        """Register a metric collector function"""
        self.collectors[metric_name] = collector_func
    
    async def collect_metric(self, metric_name: str, value: float, 
                           labels: Optional[Dict[str, str]] = None):
        """Collect a single metric point"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        metric_point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        
        self.metrics[metric_name].append(metric_point)
        
        # Clean old metrics
        await self._cleanup_old_metrics(metric_name)
    
    async def collect_all_metrics(self):
        """Collect all registered metrics"""
        for metric_name, collector_func in self.collectors.items():
            try:
                result = await collector_func()
                if isinstance(result, dict):
                    for key, value in result.items():
                        await self.collect_metric(f"{metric_name}.{key}", value)
                else:
                    await self.collect_metric(metric_name, result)
            except Exception as e:
                logging.error(f"Error collecting metric {metric_name}: {e}")
    
    async def _cleanup_old_metrics(self, metric_name: str):
        """Remove old metric points beyond retention period"""
        cutoff_time = time.time() - self.retention_period
        self.metrics[metric_name] = [
            point for point in self.metrics[metric_name]
            if point.timestamp > cutoff_time
        ]
    
    def get_metric_values(self, metric_name: str, 
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> List[MetricPoint]:
        """Get metric values within time range"""
        if metric_name not in self.metrics:
            return []
        
        points = self.metrics[metric_name]
        
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        return points
    
    def aggregate_metric(self, metric_name: str, aggregation: MetricAggregation,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None) -> Optional[float]:
        """Aggregate metric values"""
        points = self.get_metric_values(metric_name, start_time, end_time)
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        if aggregation == MetricAggregation.AVERAGE:
            return statistics.mean(values)
        elif aggregation == MetricAggregation.SUM:
            return sum(values)
        elif aggregation == MetricAggregation.MIN:
            return min(values)
        elif aggregation == MetricAggregation.MAX:
            return max(values)
        elif aggregation == MetricAggregation.COUNT:
            return len(values)
        elif aggregation == MetricAggregation.PERCENTILE_95:
            return statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0]
        elif aggregation == MetricAggregation.PERCENTILE_99:
            return statistics.quantiles(values, n=100)[98] if len(values) > 1 else values[0]
        
        return None

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        self.rule_states: Dict[str, Dict[str, Any]] = {}  # Track rule evaluation state
        
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        self.rule_states[rule.name] = {
            "condition_start_time": None,
            "last_evaluation": None
        }
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    async def evaluate_alerts(self):
        """Evaluate all alert rules"""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Get current metric value
                metric_value = self.metrics_collector.aggregate_metric(
                    rule.metric_name,
                    MetricAggregation.AVERAGE,
                    start_time=current_time - 60  # Last minute
                )
                
                if metric_value is None:
                    continue
                
                # Check if condition is met
                condition_met = self._evaluate_condition(rule, metric_value)
                
                # Track condition duration
                rule_state = self.rule_states[rule_name]
                
                if condition_met:
                    if rule_state["condition_start_time"] is None:
                        rule_state["condition_start_time"] = current_time
                    
                    duration_met = (current_time - rule_state["condition_start_time"]) >= rule.duration
                else:
                    rule_state["condition_start_time"] = None
                    duration_met = False
                
                # Evaluate alert
                should_alert = rule.evaluate(metric_value, duration_met)
                
                if should_alert and rule_name not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        id=f"{rule_name}_{int(current_time)}",
                        severity=rule.severity,
                        title=f"Alert: {rule.name}",
                        description=rule.description or f"{rule.metric_name} {rule.condition} {rule.threshold}",
                        metric_name=rule.metric_name,
                        threshold=rule.threshold,
                        current_value=metric_value,
                        labels=rule.labels
                    )
                    
                    self.active_alerts[rule_name] = alert
                    self.alert_history.append(alert)
                    
                    # Send notifications
                    await self._send_notifications(alert)
                    
                elif not should_alert and rule_name in self.active_alerts:
                    # Resolve alert
                    alert = self.active_alerts[rule_name]
                    alert.is_resolved = True
                    alert.resolved_at = current_time
                    
                    del self.active_alerts[rule_name]
                    
                    # Send resolution notification
                    await self._send_resolution_notification(alert)
                
                rule_state["last_evaluation"] = current_time
                
            except Exception as e:
                logging.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, rule: AlertRule, value: float) -> bool:
        """Evaluate if condition is met"""
        if rule.condition == ">":
            return value > rule.threshold
        elif rule.condition == "<":
            return value < rule.threshold
        elif rule.condition == ">=":
            return value >= rule.threshold
        elif rule.condition == "<=":
            return value <= rule.threshold
        elif rule.condition == "==":
            return abs(value - rule.threshold) < 0.001
        elif rule.condition == "!=":
            return abs(value - rule.threshold) >= 0.001
        return False
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for handler in self.notification_handlers:
            try:
                await handler(alert, "triggered")
            except Exception as e:
                logging.error(f"Error sending notification: {e}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notifications"""
        for handler in self.notification_handlers:
            try:
                await handler(alert, "resolved")
            except Exception as e:
                logging.error(f"Error sending resolution notification: {e}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts"""
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_active_alerts": len(self.active_alerts),
            "severity_breakdown": severity_counts,
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "created_at": alert.created_at
                }
                for alert in self.active_alerts.values()
            ],
            "recent_resolved": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "resolved_at": alert.resolved_at
                }
                for alert in self.alert_history[-10:]
                if alert.is_resolved
            ]
        }

class MonitoringDashboard:
    """Provides monitoring dashboard functionality"""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview for dashboard"""
        current_time = time.time()
        last_hour = current_time - 3600
        
        # Get key metrics
        cpu_usage = self.metrics_collector.aggregate_metric(
            "system.cpu_percent", MetricAggregation.AVERAGE, last_hour
        )
        memory_usage = self.metrics_collector.aggregate_metric(
            "system.memory_percent", MetricAggregation.AVERAGE, last_hour
        )
        request_rate = self.metrics_collector.aggregate_metric(
            "agents.request_rate", MetricAggregation.AVERAGE, last_hour
        )
        error_rate = self.metrics_collector.aggregate_metric(
            "agents.error_rate", MetricAggregation.AVERAGE, last_hour
        )
        
        return {
            "timestamp": current_time,
            "system_health": {
                "cpu_usage_percent": cpu_usage or 0,
                "memory_usage_percent": memory_usage or 0,
                "request_rate_per_second": request_rate or 0,
                "error_rate_percent": error_rate or 0
            },
            "alerts": self.alert_manager.get_alert_summary(),
            "agent_status": await self._get_agent_status()
        }
    
    async def _get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        # This would integrate with the container manager or orchestrator
        # For demonstration, return mock data
        return {
            "total_agents": 15,
            "healthy_agents": 14,
            "unhealthy_agents": 1,
            "agent_types": {
                "worker": {"count": 8, "healthy": 8},
                "coordinator": {"count": 3, "healthy": 3},
                "specialist": {"count": 4, "healthy": 3}
            }
        }
    
    async def get_metric_chart_data(self, metric_name: str, 
                                  duration_hours: int = 1) -> Dict[str, Any]:
        """Get metric data for charting"""
        end_time = time.time()
        start_time = end_time - (duration_hours * 3600)
        
        points = self.metrics_collector.get_metric_values(
            metric_name, start_time, end_time
        )
        
        # Group by time intervals for charting
        interval_seconds = max(60, (duration_hours * 3600) // 100)  # Max 100 data points
        
        chart_data = []
        current_interval = start_time
        
        while current_interval < end_time:
            interval_end = current_interval + interval_seconds
            interval_points = [
                p for p in points 
                if current_interval <= p.timestamp < interval_end
            ]
            
            if interval_points:
                avg_value = statistics.mean([p.value for p in interval_points])
                chart_data.append({
                    "timestamp": current_interval,
                    "value": avg_value
                })
            
            current_interval = interval_end
        
        return {
            "metric_name": metric_name,
            "start_time": start_time,
            "end_time": end_time,
            "interval_seconds": interval_seconds,
            "data_points": chart_data
        }
```

## 5. Fault Tolerance and Disaster Recovery

### 5.1 Circuit Breaker Pattern

```python
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import logging
from functools import wraps

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Request timeout
    expected_exception: type = Exception # Exception type that counts as failure

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: List[Dict[str, Any]] = field(default_factory=list)

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.stats.total_requests += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.stats.rejected_requests += 1
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs) if asyncio.iscoroutinefunction(func) 
                else asyncio.create_task(asyncio.coroutine(lambda: func(*args, **kwargs))()),
                timeout=self.config.timeout
            )
            
            # Record success
            await self._on_success()
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            await self._on_failure()
            raise e
        except asyncio.TimeoutError:
            # Timeout counts as failure
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    async def _on_success(self):
        """Handle successful request"""
        self.stats.successful_requests += 1
        self.stats.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        
        # Reset failure count on success
        self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed request"""
        self.stats.failed_requests += 1
        self.stats.last_failure_time = time.time()
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self._record_state_change("OPEN", "Failure threshold exceeded")
        logging.warning(f"Circuit breaker '{self.name}' opened")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        self._record_state_change("HALF_OPEN", "Attempting recovery")
        logging.info(f"Circuit breaker '{self.name}' half-opened")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._record_state_change("CLOSED", "Recovery successful")
        logging.info(f"Circuit breaker '{self.name}' closed")
    
    def _record_state_change(self, new_state: str, reason: str):
        """Record state change for monitoring"""
        self.stats.state_changes.append({
            "timestamp": time.time(),
            "from_state": self.state.value if hasattr(self.state, 'value') else str(self.state),
            "to_state": new_state,
            "reason": reason
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = (
            self.stats.successful_requests / self.stats.total_requests 
            if self.stats.total_requests > 0 else 0
        )
        
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": {
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "rejected_requests": self.stats.rejected_requests,
                "success_rate": success_rate,
                "failure_count": self.failure_count,
                "last_failure_time": self.stats.last_failure_time,
                "last_success_time": self.stats.last_success_time
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            },
            "recent_state_changes": self.stats.state_changes[-5:]
        }

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for applying circuit breaker pattern"""
    if config is None:
        config = CircuitBreakerConfig()
    
    breaker = CircuitBreaker(name, config)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        # Attach breaker to function for monitoring
        wrapper._circuit_breaker = breaker
        return wrapper
    
    return decorator

### 5.2 Retry and Backoff Strategies

class BackoffStrategy(Enum):
    """Backoff strategies for retries"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    JITTERED = "jittered"

@dataclass
class RetryConfig:
    """Configuration for retry mechanism"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    
class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted"""
    pass

class RetryManager:
    """Manages retry logic with various backoff strategies"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt failed
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                
                logging.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise RetryExhaustedException(
            f"All {self.config.max_attempts} retry attempts failed. "
            f"Last error: {last_exception}"
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on backoff strategy"""
        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.config.base_delay
            
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
            
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (2 ** attempt)
            
        elif self.config.backoff_strategy == BackoffStrategy.JITTERED:
            base_delay = self.config.base_delay * (2 ** attempt)
            # Add jitter (±25% of base delay)
            import random
            jitter = random.uniform(-0.25, 0.25) * base_delay
            delay = base_delay + jitter
        
        # Cap at max delay
        return min(delay, self.config.max_delay)

def retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to functions"""
    if config is None:
        config = RetryConfig()
    
    retry_manager = RetryManager(config)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_manager.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    return decorator

### 5.3 Health Checks and Self-Healing

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    is_healthy: bool
    message: str
    timestamp: float
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)

class HealthChecker:
    """Manages health checks for agents and services"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.healing_actions: Dict[str, Callable] = {}
        
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
        self.health_history[name] = []
        
    def register_healing_action(self, name: str, healing_func: Callable):
        """Register a self-healing action"""
        self.healing_actions[name] = healing_func
        
    async def run_health_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                is_healthy=False,
                message="Health check not found",
                timestamp=time.time(),
                response_time=0.0
            )
        
        start_time = time.time()
        
        try:
            check_func = self.health_checks[name]
            
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            response_time = time.time() - start_time
            
            if isinstance(result, bool):
                health_result = HealthCheckResult(
                    name=name,
                    is_healthy=result,
                    message="OK" if result else "Health check failed",
                    timestamp=time.time(),
                    response_time=response_time
                )
            elif isinstance(result, dict):
                health_result = HealthCheckResult(
                    name=name,
                    is_healthy=result.get("healthy", False),
                    message=result.get("message", "No message"),
                    timestamp=time.time(),
                    response_time=response_time,
                    details=result.get("details", {})
                )
            else:
                health_result = HealthCheckResult(
                    name=name,
                    is_healthy=False,
                    message=f"Invalid health check result: {result}",
                    timestamp=time.time(),
                    response_time=response_time
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            health_result = HealthCheckResult(
                name=name,
                is_healthy=False,
                message=f"Health check error: {e}",
                timestamp=time.time(),
                response_time=response_time
            )
        
        # Store result in history
        self.health_history[name].append(health_result)
        
        # Keep only last 100 results
        if len(self.health_history[name]) > 100:
            self.health_history[name] = self.health_history[name][-100:]
        
        # Trigger self-healing if needed
        if not health_result.is_healthy and name in self.healing_actions:
            await self._trigger_healing(name, health_result)
        
        return health_result
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name in self.health_checks:
            results[name] = await self.run_health_check(name)
        
        return results
    
    async def _trigger_healing(self, name: str, health_result: HealthCheckResult):
        """Trigger self-healing action"""
        try:
            healing_func = self.healing_actions[name]
            
            logging.warning(f"Triggering healing action for {name}: {health_result.message}")
            
            if asyncio.iscoroutinefunction(healing_func):
                await healing_func(health_result)
            else:
                healing_func(health_result)
                
            logging.info(f"Healing action completed for {name}")
            
        except Exception as e:
            logging.error(f"Healing action failed for {name}: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        total_checks = len(self.health_checks)
        healthy_checks = 0
        
        latest_results = {}
        
        for name, history in self.health_history.items():
            if history:
                latest_result = history[-1]
                latest_results[name] = {
                    "is_healthy": latest_result.is_healthy,
                    "message": latest_result.message,
                    "timestamp": latest_result.timestamp,
                    "response_time": latest_result.response_time
                }
                
                if latest_result.is_healthy:
                    healthy_checks += 1
        
        overall_health = healthy_checks == total_checks if total_checks > 0 else True
        
        return {
            "overall_healthy": overall_health,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "unhealthy_checks": total_checks - healthy_checks,
            "health_percentage": (healthy_checks / total_checks * 100) if total_checks > 0 else 100,
            "checks": latest_results
        }

### 5.4 Disaster Recovery System

class DisasterRecoveryManager:
    """Manages disaster recovery procedures"""
    
    def __init__(self, container_manager, orchestrator=None):
        self.container_manager = container_manager
        self.orchestrator = orchestrator
        self.backup_strategies: Dict[str, Callable] = {}
        self.recovery_procedures: Dict[str, Callable] = {}
        self.disaster_scenarios: Dict[str, Dict[str, Any]] = {}
        
    def register_backup_strategy(self, name: str, backup_func: Callable):
        """Register a backup strategy"""
        self.backup_strategies[name] = backup_func
        
    def register_recovery_procedure(self, scenario: str, recovery_func: Callable):
        """Register a recovery procedure for a disaster scenario"""
        self.recovery_procedures[scenario] = recovery_func
        
    def define_disaster_scenario(self, name: str, triggers: List[str], 
                               severity: str, recovery_time_objective: int):
        """Define a disaster scenario"""
        self.disaster_scenarios[name] = {
            "triggers": triggers,
            "severity": severity,
            "rto": recovery_time_objective,  # Recovery Time Objective in seconds
            "last_occurrence": None,
            "recovery_count": 0
        }
    
    async def create_backup(self, strategy_name: str) -> Dict[str, Any]:
        """Create a backup using specified strategy"""
        if strategy_name not in self.backup_strategies:
            raise ValueError(f"Backup strategy '{strategy_name}' not found")
        
        backup_func = self.backup_strategies[strategy_name]
        
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(backup_func):
                result = await backup_func()
            else:
                result = backup_func()
            
            backup_time = time.time() - start_time
            
            backup_info = {
                "strategy": strategy_name,
                "timestamp": time.time(),
                "duration": backup_time,
                "success": True,
                "details": result if isinstance(result, dict) else {"result": result}
            }
            
            logging.info(f"Backup '{strategy_name}' completed in {backup_time:.2f}s")
            return backup_info
            
        except Exception as e:
            logging.error(f"Backup '{strategy_name}' failed: {e}")
            return {
                "strategy": strategy_name,
                "timestamp": time.time(),
                "duration": 0,
                "success": False,
                "error": str(e)
            }
    
    async def detect_disaster(self, alerts: List[Dict[str, Any]]) -> Optional[str]:
        """Detect if current alerts indicate a disaster scenario"""
        for scenario_name, scenario in self.disaster_scenarios.items():
            triggered_count = 0
            
            for alert in alerts:
                alert_type = alert.get("type", "")
                if alert_type in scenario["triggers"]:
                    triggered_count += 1
            
            # If majority of triggers are active, consider it a disaster
            trigger_threshold = len(scenario["triggers"]) * 0.6  # 60% of triggers
            
            if triggered_count >= trigger_threshold:
                logging.critical(f"Disaster scenario detected: {scenario_name}")
                return scenario_name
        
        return None
    
    async def execute_recovery(self, scenario_name: str) -> Dict[str, Any]:
        """Execute recovery procedure for a disaster scenario"""
        if scenario_name not in self.recovery_procedures:
            return {
                "success": False,
                "error": f"No recovery procedure defined for scenario '{scenario_name}'"
            }
        
        scenario = self.disaster_scenarios.get(scenario_name, {})
        recovery_func = self.recovery_procedures[scenario_name]
        
        try:
            start_time = time.time()
            
            logging.critical(f"Starting disaster recovery for scenario: {scenario_name}")
            
            if asyncio.iscoroutinefunction(recovery_func):
                result = await recovery_func(scenario)
            else:
                result = recovery_func(scenario)
            
            recovery_time = time.time() - start_time
            
            # Update scenario statistics
            if scenario_name in self.disaster_scenarios:
                self.disaster_scenarios[scenario_name]["last_occurrence"] = time.time()
                self.disaster_scenarios[scenario_name]["recovery_count"] += 1
            
            recovery_info = {
                "scenario": scenario_name,
                "success": True,
                "recovery_time": recovery_time,
                "rto_met": recovery_time <= scenario.get("rto", float('inf')),
                "details": result if isinstance(result, dict) else {"result": result}
            }
            
            logging.info(
                f"Disaster recovery completed for '{scenario_name}' in {recovery_time:.2f}s"
            )
            
            return recovery_info
            
        except Exception as e:
            logging.error(f"Disaster recovery failed for '{scenario_name}': {e}")
            return {
                "scenario": scenario_name,
                "success": False,
                "error": str(e),
                "recovery_time": time.time() - start_time
            }
    
    def get_disaster_status(self) -> Dict[str, Any]:
        """Get current disaster recovery status"""
        return {
            "scenarios": self.disaster_scenarios,
            "backup_strategies": list(self.backup_strategies.keys()),
            "recovery_procedures": list(self.recovery_procedures.keys()),
            "last_backup_times": {},  # Would track actual backup times
            "system_resilience_score": self._calculate_resilience_score()
        }
    
    def _calculate_resilience_score(self) -> float:
        """Calculate system resilience score (0-100)"""
        # Simple scoring based on coverage
        scenario_coverage = len(self.recovery_procedures) / max(len(self.disaster_scenarios), 1)
        backup_coverage = min(len(self.backup_strategies) / 3, 1)  # Assume 3 is ideal
        
        return (scenario_coverage + backup_coverage) / 2 * 100
```

## 6. Practical Implementation Examples

### 6.1 Complete Production Setup

```python
import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def setup_production_infrastructure():
    """Set up complete production infrastructure for agent systems"""
    
    # 1. Initialize container management
    container_manager = AgentContainerManager()
    
    # 2. Set up Kubernetes orchestration
    k8s_orchestrator = KubernetesAgentOrchestrator(
        namespace="agent-system",
        cluster_name="production-cluster"
    )
    
    # 3. Configure load balancing
    load_balancer = AgentLoadBalancer(
        strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
    )
    
    # Register agent endpoints
    load_balancer.register_endpoint(AgentEndpoint(
        id="worker-pool-1",
        host="worker-pool-1.agent-system.svc.cluster.local",
        port=8080,
        weight=100,
        agent_type="worker"
    ))
    
    load_balancer.register_endpoint(AgentEndpoint(
        id="coordinator-pool-1",
        host="coordinator-pool-1.agent-system.svc.cluster.local",
        port=8080,
        weight=150,
        agent_type="coordinator"
    ))
    
    # 4. Set up auto-scaling
    auto_scaler = AutoScaler(container_manager, k8s_orchestrator)
    
    # Configure scaling rules for worker agents
    worker_scaling_rule = ScalingRule(
        name="worker-auto-scale",
        metrics=[
            ScalingMetric(
                metric_type=MetricType.CPU_UTILIZATION,
                current_value=0.0,
                threshold_up=70.0,
                threshold_down=30.0,
                weight=1.0
            ),
            ScalingMetric(
                metric_type=MetricType.QUEUE_LENGTH,
                current_value=0.0,
                threshold_up=50.0,
                threshold_down=10.0,
                weight=1.5
            )
        ],
        min_instances=2,
        max_instances=20,
        scale_up_cooldown=300,
        scale_down_cooldown=600
    )
    
    auto_scaler.add_scaling_rule("worker", worker_scaling_rule)
    
    # 5. Set up monitoring and alerting
    metrics_collector = MetricsCollector(retention_period=86400)  # 24 hours
    alert_manager = AlertManager(metrics_collector)
    monitoring_dashboard = MonitoringDashboard(metrics_collector, alert_manager)
    
    # Register metrics collectors
    async def collect_system_metrics():
        # Simulate system metrics collection
        import random
        return {
            "cpu_utilization": random.uniform(20, 80),
            "memory_utilization": random.uniform(30, 70),
            "request_rate": random.uniform(100, 1000),
            "error_rate": random.uniform(0, 5)
        }
    
    metrics_collector.register_collector("system", collect_system_metrics)
    
    # Configure alert rules
    high_cpu_alert = AlertRule(
        name="high-cpu-usage",
        metric_name="system.cpu_utilization",
        condition=">",
        threshold=80.0,
        severity=AlertSeverity.WARNING,
        duration=300,
        description="CPU usage is above 80% for 5 minutes"
    )
    
    high_error_rate_alert = AlertRule(
        name="high-error-rate",
        metric_name="system.error_rate",
        condition=">",
        threshold=10.0,
        severity=AlertSeverity.CRITICAL,
        duration=60,
        description="Error rate is above 10% for 1 minute"
    )
    
    alert_manager.add_alert_rule(high_cpu_alert)
    alert_manager.add_alert_rule(high_error_rate_alert)
    
    # 6. Set up fault tolerance
    health_checker = HealthChecker()
    
    # Register health checks
    async def check_database_health():
        # Simulate database health check
        import random
        return {
            "healthy": random.choice([True, True, True, False]),  # 75% healthy
            "message": "Database connection OK" if random.random() > 0.25 else "Database timeout",
            "details": {"connection_pool_size": 10, "active_connections": 7}
        }
    
    async def check_message_queue_health():
        # Simulate message queue health check
        import random
        return random.choice([True, True, False])  # 66% healthy
    
    health_checker.register_health_check("database", check_database_health)
    health_checker.register_health_check("message_queue", check_message_queue_health)
    
    # Register healing actions
    async def heal_database(health_result: HealthCheckResult):
        logging.info("Attempting to heal database connection...")
        # Simulate database healing (restart connection pool, etc.)
        await asyncio.sleep(2)
        logging.info("Database healing completed")
    
    async def heal_message_queue(health_result: HealthCheckResult):
        logging.info("Attempting to heal message queue...")
        # Simulate message queue healing
        await asyncio.sleep(1)
        logging.info("Message queue healing completed")
    
    health_checker.register_healing_action("database", heal_database)
    health_checker.register_healing_action("message_queue", heal_message_queue)
    
    # 7. Set up disaster recovery
    disaster_recovery = DisasterRecoveryManager(container_manager, k8s_orchestrator)
    
    # Define disaster scenarios
    disaster_recovery.define_disaster_scenario(
        name="data_center_failure",
        triggers=["high-cpu-usage", "high-error-rate", "database-down"],
        severity="critical",
        recovery_time_objective=1800  # 30 minutes
    )
    
    # Register backup strategies
    async def backup_agent_state():
        logging.info("Backing up agent state...")
        # Simulate state backup
        await asyncio.sleep(5)
        return {"backup_size": "2.5GB", "agent_count": 15}
    
    async def backup_configuration():
        logging.info("Backing up configuration...")
        # Simulate configuration backup
        await asyncio.sleep(2)
        return {"config_files": 25, "size": "50MB"}
    
    disaster_recovery.register_backup_strategy("agent_state", backup_agent_state)
    disaster_recovery.register_backup_strategy("configuration", backup_configuration)
    
    # Register recovery procedures
    async def recover_from_data_center_failure(scenario):
        logging.critical("Executing data center failure recovery...")
        
        # 1. Failover to backup data center
        logging.info("Failing over to backup data center...")
        await asyncio.sleep(10)
        
        # 2. Restore agent state
        logging.info("Restoring agent state from backup...")
        await asyncio.sleep(15)
        
        # 3. Restart critical services
        logging.info("Restarting critical services...")
        await asyncio.sleep(5)
        
        return {"failover_time": 30, "agents_restored": 15, "services_restarted": 8}
    
    disaster_recovery.register_recovery_procedure(
        "data_center_failure", 
        recover_from_data_center_failure
    )
    
    return {
        "container_manager": container_manager,
        "orchestrator": k8s_orchestrator,
        "load_balancer": load_balancer,
        "auto_scaler": auto_scaler,
        "metrics_collector": metrics_collector,
        "alert_manager": alert_manager,
        "monitoring_dashboard": monitoring_dashboard,
        "health_checker": health_checker,
        "disaster_recovery": disaster_recovery
    }

async def demonstrate_production_operations():
    """Demonstrate production operations and monitoring"""
    
    print("Setting up production infrastructure...")
    infrastructure = await setup_production_infrastructure()
    
    # Extract components
    auto_scaler = infrastructure["auto_scaler"]
    metrics_collector = infrastructure["metrics_collector"]
    alert_manager = infrastructure["alert_manager"]
    monitoring_dashboard = infrastructure["monitoring_dashboard"]
    health_checker = infrastructure["health_checker"]
    disaster_recovery = infrastructure["disaster_recovery"]
    
    print("\n=== Starting Production Operations ===")
    
    # Start auto-scaling
    auto_scaling_task = asyncio.create_task(auto_scaler.start_auto_scaling())
    
    # Simulate production operations
    for cycle in range(5):
        print(f"\n--- Operation Cycle {cycle + 1} ---")
        
        # Collect metrics
        await metrics_collector.collect_all_metrics()
        
        # Evaluate alerts
        await alert_manager.evaluate_alerts()
        
        # Run health checks
        health_results = await health_checker.run_all_health_checks()
        
        # Get system overview
        system_overview = await monitoring_dashboard.get_system_overview()
        
        # Display status
        print(f"System Health: {system_overview['system_health']}")
        print(f"Active Alerts: {system_overview['alerts']['total_active_alerts']}")
        print(f"Agent Status: {system_overview['agent_status']}")
        
        # Check for disasters
        active_alerts = system_overview['alerts']['active_alerts']
        disaster_scenario = await disaster_recovery.detect_disaster(active_alerts)
        
        if disaster_scenario:
            print(f"\n🚨 DISASTER DETECTED: {disaster_scenario}")
            recovery_result = await disaster_recovery.execute_recovery(disaster_scenario)
            print(f"Recovery Result: {recovery_result}")
        
        # Create periodic backups
        if cycle % 2 == 0:  # Every other cycle
            backup_result = await disaster_recovery.create_backup("agent_state")
            print(f"Backup Status: {backup_result['success']}")
        
        # Wait before next cycle
        await asyncio.sleep(10)
    
    # Stop auto-scaling
    auto_scaler.stop_auto_scaling()
    await auto_scaling_task
    
    print("\n=== Production Operations Complete ===")
    
    # Final status report
    scaling_status = auto_scaler.get_scaling_status()
    health_summary = health_checker.get_health_summary()
    disaster_status = disaster_recovery.get_disaster_status()
    
    print(f"\nFinal Status:")
    print(f"- Auto-scaling: {len(scaling_status['agent_types'])} agent types managed")
    print(f"- Health: {health_summary['health_percentage']:.1f}% healthy")
    print(f"- Resilience Score: {disaster_status['system_resilience_score']:.1f}%")

# Example usage
if __name__ == "__main__":
    asyncio.run(demonstrate_production_operations())
```

## 7. Hands-on Exercises

### Exercise 1: Custom Monitoring Dashboard

```python
class CustomMonitoringDashboard:
    """Custom monitoring dashboard with advanced features"""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.custom_widgets: Dict[str, Callable] = {}
        self.dashboard_config: Dict[str, Any] = {}
        
    def add_custom_widget(self, name: str, widget_func: Callable):
        """Add a custom dashboard widget"""
        self.custom_widgets[name] = widget_func
        
    async def generate_custom_report(self, report_type: str) -> Dict[str, Any]:
        """Generate custom reports based on type"""
        if report_type == "performance":
            return await self._generate_performance_report()
        elif report_type == "capacity":
            return await self._generate_capacity_report()
        elif report_type == "reliability":
            return await self._generate_reliability_report()
        else:
            return {"error": f"Unknown report type: {report_type}"}
    
    async def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance analysis report"""
        # TODO: Implement performance metrics analysis
        # - Response time trends
        # - Throughput analysis
        # - Resource utilization patterns
        # - Performance bottleneck identification
        pass
    
    async def _generate_capacity_report(self) -> Dict[str, Any]:
        """Generate capacity planning report"""
        # TODO: Implement capacity analysis
        # - Current resource usage trends
        # - Projected capacity needs
        # - Scaling recommendations
        # - Cost optimization suggestions
        pass
    
    async def _generate_reliability_report(self) -> Dict[str, Any]:
        """Generate system reliability report"""
        # TODO: Implement reliability analysis
        # - Uptime statistics
        # - Error rate trends
        # - MTTR (Mean Time To Recovery) analysis
        # - SLA compliance metrics
        pass
```

### Exercise 2: Advanced Auto-Scaling Strategy

```python
class PredictiveAutoScaler(AutoScaler):
    """Auto-scaler with predictive capabilities"""
    
    def __init__(self, container_manager, orchestrator=None):
        super().__init__(container_manager, orchestrator)
        self.prediction_models: Dict[str, Any] = {}
        self.historical_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
    def train_prediction_model(self, agent_type: str, historical_data: List[Dict[str, Any]]):
        """Train a prediction model for an agent type"""
        # TODO: Implement machine learning model training
        # - Time series forecasting for load prediction
        # - Seasonal pattern recognition
        # - Anomaly detection for unusual load spikes
        # - Multi-variate analysis of scaling factors
        pass
    
    async def predict_scaling_needs(self, agent_type: str, 
                                  prediction_horizon: int = 3600) -> Dict[str, Any]:
        """Predict scaling needs for the next time period"""
        # TODO: Implement predictive scaling
        # - Use trained models to forecast load
        # - Consider historical patterns and trends
        # - Account for known events (deployments, maintenance)
        # - Provide confidence intervals for predictions
        pass
    
    async def proactive_scaling(self, agent_type: str):
        """Perform proactive scaling based on predictions"""
        # TODO: Implement proactive scaling logic
        # - Scale before demand increases
        # - Gradual scaling to avoid resource waste
        # - Integration with cost optimization
        # - Rollback mechanisms for incorrect predictions
        pass
```

### Exercise 3: Comprehensive Disaster Recovery Testing

```python
class DisasterRecoveryTester:
    """Automated disaster recovery testing system"""
    
    def __init__(self, disaster_recovery: DisasterRecoveryManager):
        self.disaster_recovery = disaster_recovery
        self.test_scenarios: Dict[str, Dict[str, Any]] = {}
        self.test_results: List[Dict[str, Any]] = []
        
    def define_test_scenario(self, name: str, scenario_config: Dict[str, Any]):
        """Define a disaster recovery test scenario"""
        self.test_scenarios[name] = scenario_config
        
    async def run_chaos_engineering_test(self, test_name: str) -> Dict[str, Any]:
        """Run chaos engineering tests to validate disaster recovery"""
        # TODO: Implement chaos engineering tests
        # - Simulate various failure modes
        # - Test recovery procedures under load
        # - Validate RTO/RPO compliance
        # - Measure system resilience
        pass
    
    async def validate_backup_integrity(self, backup_name: str) -> Dict[str, Any]:
        """Validate backup integrity and recoverability"""
        # TODO: Implement backup validation
        # - Test backup restoration procedures
        # - Verify data consistency
        # - Check backup completeness
        # - Measure recovery time
        pass
    
    async def generate_dr_compliance_report(self) -> Dict[str, Any]:
        """Generate disaster recovery compliance report"""
        # TODO: Implement compliance reporting
        # - RTO/RPO compliance analysis
        # - Test coverage assessment
        # - Gap analysis and recommendations
        # - Regulatory compliance status
        pass
```

## Module Summary

This module covered the essential aspects of production deployment and scaling for multi-agent systems:

### Key Concepts Learned:
1. **Containerization and Orchestration**: Docker containers, Kubernetes deployment, service management
2. **Load Balancing**: Traffic distribution strategies, health checks, endpoint management
3. **Auto-Scaling**: Dynamic scaling based on metrics, resource optimization, predictive scaling
4. **Monitoring and Observability**: Metrics collection, alerting, dashboard creation, system health tracking
5. **Fault Tolerance**: Circuit breaker patterns, retry mechanisms, self-healing systems
6. **Disaster Recovery**: Backup strategies, recovery procedures, chaos engineering

### Practical Skills Developed:
1. **Infrastructure Setup**: Complete production environment configuration
2. **Monitoring Implementation**: Real-time metrics and alerting systems
3. **Scaling Strategies**: Automated and predictive scaling mechanisms
4. **Fault Tolerance Design**: Resilient system architecture patterns
5. **Disaster Recovery Planning**: Comprehensive backup and recovery procedures

### Real-world Applications:
1. **Enterprise Systems**: Large-scale agent deployments in corporate environments
2. **Cloud-Native Applications**: Microservices-based agent architectures
3. **High-Availability Services**: Mission-critical systems requiring 99.9%+ uptime
4. **Global Distributed Systems**: Multi-region agent deployments with disaster recovery

### Next Steps:
The next module will focus on **Performance Optimization and Advanced Monitoring**, covering:
- Performance profiling and optimization techniques
- Advanced monitoring and observability patterns
- Cost optimization strategies
- Security hardening for production systems
- Compliance and governance frameworks