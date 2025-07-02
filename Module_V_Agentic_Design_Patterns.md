# Module V: Agentic Design Patterns

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement ReAct Framework**: Build reasoning and acting agents that can think through problems step-by-step
2. **Apply Reflection Patterns**: Create self-improving agents that learn from their mistakes
3. **Develop CodeAct Systems**: Build agents capable of dynamic code execution and debugging
4. **Combine Design Patterns**: Integrate multiple patterns for sophisticated agent behaviors
5. **Optimize Pattern Performance**: Fine-tune patterns for specific use cases and requirements
6. **Handle Pattern Failures**: Implement robust error handling and fallback mechanisms

---

## 1. ReAct Framework Implementation

### 1.1 Understanding ReAct (Reasoning + Acting)

The ReAct framework combines reasoning and acting in a synergistic loop, allowing agents to think through problems while taking actions to gather information and make progress toward their goals.

#### ReAct Architecture

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">ReAct Framework Architecture</text>
  
  <!-- Central Loop -->
  <circle cx="400" cy="300" r="150" fill="none" stroke="#3498db" stroke-width="3" stroke-dasharray="10,5"/>
  
  <!-- Thought Phase -->
  <rect x="320" y="120" width="160" height="80" rx="10" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
  <text x="400" y="150" text-anchor="middle" font-size="14" font-weight="bold" fill="white">THOUGHT</text>
  <text x="400" y="170" text-anchor="middle" font-size="12" fill="white">Reasoning Phase</text>
  <text x="400" y="185" text-anchor="middle" font-size="10" fill="white">Analyze • Plan • Decide</text>
  
  <!-- Action Phase -->
  <rect x="520" y="260" width="160" height="80" rx="10" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="600" y="290" text-anchor="middle" font-size="14" font-weight="bold" fill="white">ACTION</text>
  <text x="600" y="310" text-anchor="middle" font-size="12" fill="white">Execution Phase</text>
  <text x="600" y="325" text-anchor="middle" font-size="10" fill="white">Execute • Observe • Collect</text>
  
  <!-- Observation Phase -->
  <rect x="320" y="400" width="160" height="80" rx="10" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="400" y="430" text-anchor="middle" font-size="14" font-weight="bold" fill="white">OBSERVATION</text>
  <text x="400" y="450" text-anchor="middle" font-size="12" fill="white">Feedback Phase</text>
  <text x="400" y="465" text-anchor="middle" font-size="10" fill="white">Evaluate • Learn • Update</text>
  
  <!-- Reflection Phase -->
  <rect x="120" y="260" width="160" height="80" rx="10" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
  <text x="200" y="290" text-anchor="middle" font-size="14" font-weight="bold" fill="white">REFLECTION</text>
  <text x="200" y="310" text-anchor="middle" font-size="12" fill="white">Meta-Cognitive Phase</text>
  <text x="200" y="325" text-anchor="middle" font-size="10" fill="white">Review • Improve • Adapt</text>
  
  <!-- Arrows -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Thought to Action -->
  <path d="M 480 160 Q 550 200 560 260" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Action to Observation -->
  <path d="M 560 340 Q 550 380 480 440" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Observation to Reflection -->
  <path d="M 320 440 Q 250 380 240 340" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Reflection to Thought -->
  <path d="M 240 260 Q 250 200 320 160" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
</svg>
```

### 1.2 Core ReAct Components

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
import logging
from datetime import datetime

class ActionType(Enum):
    """Types of actions an agent can take"""
    SEARCH = "search"
    CALCULATE = "calculate"
    ANALYZE = "analyze"
    GENERATE = "generate"
    VALIDATE = "validate"
    COMMUNICATE = "communicate"
    TOOL_USE = "tool_use"

class ThoughtType(Enum):
    """Types of reasoning thoughts"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    HYPOTHESIS = "hypothesis"
    EVALUATION = "evaluation"
    DECISION = "decision"
    REFLECTION = "reflection"

@dataclass
class Thought:
    """Represents a reasoning step in the ReAct framework"""
    content: str
    thought_type: ThoughtType
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "type": self.thought_type.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }

@dataclass
class Action:
    """Represents an action taken by the agent"""
    action_type: ActionType
    parameters: Dict[str, Any]
    expected_outcome: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.action_type.value,
            "parameters": self.parameters,
            "expected_outcome": self.expected_outcome,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class Observation:
    """Represents the result of an action"""
    content: str
    success: bool
    metadata: Dict[str, Any]
    timestamp: datetime
    action_reference: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "success": self.success,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "action_reference": self.action_reference
        }

class ReActStep:
    """Represents a complete ReAct cycle step"""
    
    def __init__(self, step_id: str):
        self.step_id = step_id
        self.thought: Optional[Thought] = None
        self.action: Optional[Action] = None
        self.observation: Optional[Observation] = None
        self.reflection: Optional[str] = None
        self.created_at = datetime.now()
    
    def add_thought(self, content: str, thought_type: ThoughtType, 
                   confidence: float = 0.8, context: Dict[str, Any] = None):
        """Add a thought to this step"""
        self.thought = Thought(
            content=content,
            thought_type=thought_type,
            confidence=confidence,
            timestamp=datetime.now(),
            context=context or {}
        )
    
    def add_action(self, action_type: ActionType, parameters: Dict[str, Any], 
                   expected_outcome: str):
        """Add an action to this step"""
        self.action = Action(
            action_type=action_type,
            parameters=parameters,
            expected_outcome=expected_outcome,
            timestamp=datetime.now()
        )
    
    def add_observation(self, content: str, success: bool, 
                       metadata: Dict[str, Any] = None):
        """Add an observation to this step"""
        self.observation = Observation(
            content=content,
            success=success,
            metadata=metadata or {},
            timestamp=datetime.now(),
            action_reference=self.step_id
        )
    
    def add_reflection(self, reflection: str):
        """Add a reflection to this step"""
        self.reflection = reflection
    
    def is_complete(self) -> bool:
        """Check if this step has all required components"""
        return all([self.thought, self.action, self.observation])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation"""
        return {
            "step_id": self.step_id,
            "thought": self.thought.to_dict() if self.thought else None,
            "action": self.action.to_dict() if self.action else None,
            "observation": self.observation.to_dict() if self.observation else None,
            "reflection": self.reflection,
            "created_at": self.created_at.isoformat()
        }

class ToolInterface(ABC):
    """Abstract interface for tools that can be used by ReAct agents"""
    
    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of what this tool does"""
        pass
    
    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the schema for parameters this tool accepts"""
        pass

class SearchTool(ToolInterface):
    """Example search tool implementation"""
    
    def __init__(self, search_engine):
        self.search_engine = search_engine
    
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search with given query"""
        query = parameters.get("query", "")
        max_results = parameters.get("max_results", 5)
        
        try:
            # Simulate search execution
            results = self.search_engine.search(query, max_results)
            return {
                "success": True,
                "results": results,
                "query": query,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def get_description(self) -> str:
        return "Search for information using a query string"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "required": True, "description": "Search query"},
            "max_results": {"type": "integer", "default": 5, "description": "Maximum number of results"}
        }

class CalculatorTool(ToolInterface):
    """Example calculator tool implementation"""
    
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mathematical calculation"""
        expression = parameters.get("expression", "")
        
        try:
            # Safe evaluation of mathematical expressions
            result = eval(expression, {"__builtins__": {}}, {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "sqrt": lambda x: x**0.5
            })
            return {
                "success": True,
                "result": result,
                "expression": expression
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }
    
    def get_description(self) -> str:
        return "Perform mathematical calculations"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "expression": {"type": "string", "required": True, "description": "Mathematical expression to evaluate"}
        }

class ReActAgent:
    """Main ReAct agent implementation"""
    
    def __init__(self, name: str, llm_client, max_steps: int = 10):
        self.name = name
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.tools: Dict[str, ToolInterface] = {}
        self.steps: List[ReActStep] = []
        self.current_goal: Optional[str] = None
        self.logger = logging.getLogger(f"ReActAgent.{name}")
    
    def register_tool(self, name: str, tool: ToolInterface):
        """Register a tool for the agent to use"""
        self.tools[name] = tool
        self.logger.info(f"Registered tool: {name}")
    
    def set_goal(self, goal: str):
        """Set the current goal for the agent"""
        self.current_goal = goal
        self.steps = []  # Reset steps for new goal
        self.logger.info(f"New goal set: {goal}")
    
    def _generate_thought(self, context: str) -> Thought:
        """Generate a reasoning thought based on current context"""
        prompt = f"""
        Goal: {self.current_goal}
        Current Context: {context}
        
        Available Tools: {list(self.tools.keys())}
        
        Think step by step about what you should do next to achieve the goal.
        Consider:
        1. What information do you have?
        2. What information do you need?
        3. What action would be most helpful?
        4. What are the potential outcomes?
        
        Provide your reasoning in a clear, structured way.
        """
        
        response = self.llm_client.generate(prompt)
        
        return Thought(
            content=response,
            thought_type=ThoughtType.PLANNING,
            confidence=0.8,
            timestamp=datetime.now(),
            context={"goal": self.current_goal, "step_count": len(self.steps)}
        )
    
    def _decide_action(self, thought: Thought) -> Action:
        """Decide on an action based on the current thought"""
        tools_info = "\n".join([
            f"- {name}: {tool.get_description()}"
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""
        Goal: {self.current_goal}
        Current Thought: {thought.content}
        
        Available Tools:
        {tools_info}
        
        Based on your reasoning, decide on the next action to take.
        Respond with a JSON object containing:
        {{
            "tool_name": "name_of_tool_to_use",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "expected_outcome": "what_you_expect_to_achieve"
        }}
        """
        
        response = self.llm_client.generate(prompt)
        
        try:
            action_data = json.loads(response)
            tool_name = action_data["tool_name"]
            
            if tool_name not in self.tools:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            return Action(
                action_type=ActionType.TOOL_USE,
                parameters={
                    "tool_name": tool_name,
                    "tool_parameters": action_data["parameters"]
                },
                expected_outcome=action_data["expected_outcome"],
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Failed to parse action: {e}")
            # Fallback action
            return Action(
                action_type=ActionType.ANALYZE,
                parameters={"analysis_target": "current_situation"},
                expected_outcome="Better understanding of the situation",
                timestamp=datetime.now()
            )
    
    def _execute_action(self, action: Action) -> Observation:
        """Execute the decided action and return observation"""
        try:
            if action.action_type == ActionType.TOOL_USE:
                tool_name = action.parameters["tool_name"]
                tool_parameters = action.parameters["tool_parameters"]
                
                if tool_name in self.tools:
                    result = self.tools[tool_name].execute(tool_parameters)
                    
                    return Observation(
                        content=json.dumps(result, indent=2),
                        success=result.get("success", True),
                        metadata={"tool_used": tool_name, "parameters": tool_parameters},
                        timestamp=datetime.now(),
                        action_reference=f"step_{len(self.steps)}"
                    )
                else:
                    return Observation(
                        content=f"Tool '{tool_name}' not found",
                        success=False,
                        metadata={"error": "tool_not_found"},
                        timestamp=datetime.now(),
                        action_reference=f"step_{len(self.steps)}"
                    )
            else:
                # Handle other action types
                return Observation(
                    content=f"Executed {action.action_type.value} action",
                    success=True,
                    metadata={"action_type": action.action_type.value},
                    timestamp=datetime.now(),
                    action_reference=f"step_{len(self.steps)}"
                )
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return Observation(
                content=f"Action failed: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                action_reference=f"step_{len(self.steps)}"
            )
    
    def _generate_reflection(self, step: ReActStep) -> str:
        """Generate a reflection on the completed step"""
        prompt = f"""
        Goal: {self.current_goal}
        
        Step Summary:
        - Thought: {step.thought.content if step.thought else 'None'}
        - Action: {step.action.action_type.value if step.action else 'None'}
        - Observation: {step.observation.content if step.observation else 'None'}
        - Success: {step.observation.success if step.observation else False}
        
        Reflect on this step:
        1. Was the action appropriate for the thought?
        2. Did the observation match the expected outcome?
        3. What can be learned from this step?
        4. How should this inform the next step?
        
        Provide a brief reflection (2-3 sentences).
        """
        
        return self.llm_client.generate(prompt)
    
    def _check_goal_completion(self) -> bool:
        """Check if the goal has been achieved"""
        if not self.steps:
            return False
        
        recent_observations = [step.observation.content for step in self.steps[-3:] 
                             if step.observation]
        
        prompt = f"""
        Goal: {self.current_goal}
        
        Recent Observations:
        {chr(10).join(recent_observations)}
        
        Has the goal been achieved based on these observations?
        Respond with 'YES' if the goal is complete, 'NO' if more work is needed.
        """
        
        response = self.llm_client.generate(prompt).strip().upper()
        return response.startswith('YES')
    
    def solve(self, goal: str) -> Dict[str, Any]:
        """Main method to solve a goal using ReAct framework"""
        self.set_goal(goal)
        
        for step_num in range(self.max_steps):
            step = ReActStep(f"step_{step_num}")
            
            # Generate context from previous steps
            context = self._build_context()
            
            # THOUGHT phase
            thought = self._generate_thought(context)
            step.add_thought(
                thought.content, 
                thought.thought_type, 
                thought.confidence, 
                thought.context
            )
            
            # ACTION phase
            action = self._decide_action(thought)
            step.add_action(
                action.action_type,
                action.parameters,
                action.expected_outcome
            )
            
            # OBSERVATION phase
            observation = self._execute_action(action)
            step.add_observation(
                observation.content,
                observation.success,
                observation.metadata
            )
            
            # REFLECTION phase
            reflection = self._generate_reflection(step)
            step.add_reflection(reflection)
            
            self.steps.append(step)
            
            self.logger.info(f"Completed step {step_num}: {step.thought.content[:100]}...")
            
            # Check if goal is achieved
            if self._check_goal_completion():
                self.logger.info(f"Goal achieved in {step_num + 1} steps")
                break
        
        return self._generate_solution_summary()
    
    def _build_context(self) -> str:
        """Build context string from previous steps"""
        if not self.steps:
            return "This is the first step."
        
        context_parts = []
        for i, step in enumerate(self.steps[-3:]):  # Last 3 steps
            context_parts.append(f"Step {i}: {step.thought.content if step.thought else 'No thought'}")
            if step.observation:
                context_parts.append(f"Result: {step.observation.content[:200]}...")
        
        return "\n".join(context_parts)
    
    def _generate_solution_summary(self) -> Dict[str, Any]:
        """Generate a summary of the solution process"""
        return {
            "goal": self.current_goal,
            "steps_taken": len(self.steps),
            "success": self._check_goal_completion(),
            "steps": [step.to_dict() for step in self.steps],
            "final_reflection": self._generate_final_reflection()
        }
    
    def _generate_final_reflection(self) -> str:
        """Generate a final reflection on the entire process"""
        prompt = f"""
        Goal: {self.current_goal}
        Steps Taken: {len(self.steps)}
        
        Process Summary:
        {chr(10).join([f"Step {i}: {step.thought.content[:100]}..." for i, step in enumerate(self.steps)])}
        
        Provide a final reflection on the problem-solving process:
        1. Was the goal achieved?
        2. What worked well?
        3. What could be improved?
        4. Key insights gained?
        """
        
        return self.llm_client.generate(prompt)
```

### 1.3 ReAct Usage Example

```python
class MockLLMClient:
    """Mock LLM client for demonstration"""
    
    def generate(self, prompt: str) -> str:
        # This would normally call an actual LLM
        # For demo purposes, return structured responses
        if "Think step by step" in prompt:
            return "I need to search for information about the topic to provide an accurate answer."
        elif "decide on the next action" in prompt:
            return '{"tool_name": "search", "parameters": {"query": "example query"}, "expected_outcome": "Find relevant information"}'
        elif "Has the goal been achieved" in prompt:
            return "NO"
        else:
            return "This is a mock response for demonstration purposes."

class MockSearchEngine:
    """Mock search engine for demonstration"""
    
    def search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        return [
            {"title": f"Result 1 for {query}", "content": "Mock content 1"},
            {"title": f"Result 2 for {query}", "content": "Mock content 2"}
        ]

# Example usage
def demonstrate_react_agent():
    """Demonstrate ReAct agent functionality"""
    
    # Initialize components
    llm_client = MockLLMClient()
    search_engine = MockSearchEngine()
    
    # Create agent
    agent = ReActAgent("research_agent", llm_client, max_steps=5)
    
    # Register tools
    agent.register_tool("search", SearchTool(search_engine))
    agent.register_tool("calculator", CalculatorTool())
    
    # Solve a problem
    goal = "Find information about machine learning algorithms and calculate the accuracy of a model with 85 correct predictions out of 100 total predictions"
    
    result = agent.solve(goal)
    
    print("=== ReAct Agent Solution ===")
    print(f"Goal: {result['goal']}")
    print(f"Steps taken: {result['steps_taken']}")
    print(f"Success: {result['success']}")
    print(f"Final reflection: {result['final_reflection']}")
    
    return result

if __name__ == "__main__":
    demonstrate_react_agent()
```

---

## 2. Reflection Patterns

### 2.1 Understanding Self-Reflection in AI Agents

Reflection patterns enable agents to examine their own reasoning processes, learn from mistakes, and improve their performance over time. This meta-cognitive capability is crucial for building adaptive and self-improving systems.

#### Reflection Architecture

```svg
<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="500" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Reflection Pattern Architecture</text>
  
  <!-- Main Process Flow -->
  <rect x="50" y="80" width="120" height="60" rx="8" fill="#3498db" stroke="#2980b9" stroke-width="2"/>
  <text x="110" y="105" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Initial</text>
  <text x="110" y="120" text-anchor="middle" font-size="12" fill="white">Response</text>
  
  <rect x="220" y="80" width="120" height="60" rx="8" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
  <text x="280" y="105" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Self</text>
  <text x="280" y="120" text-anchor="middle" font-size="12" fill="white">Evaluation</text>
  
  <rect x="390" y="80" width="120" height="60" rx="8" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="450" y="105" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Identify</text>
  <text x="450" y="120" text-anchor="middle" font-size="12" fill="white">Issues</text>
  
  <rect x="560" y="80" width="120" height="60" rx="8" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="620" y="105" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Refined</text>
  <text x="620" y="120" text-anchor="middle" font-size="12" fill="white">Response</text>
  
  <!-- Reflection Loop -->
  <rect x="200" y="200" width="400" height="120" rx="10" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2"/>
  <text x="400" y="225" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Reflection Engine</text>
  
  <!-- Reflection Components -->
  <rect x="220" y="240" width="100" height="30" rx="5" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="270" y="258" text-anchor="middle" font-size="10" fill="white">Error Analysis</text>
  
  <rect x="340" y="240" width="100" height="30" rx="5" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="390" y="258" text-anchor="middle" font-size="10" fill="white">Quality Check</text>
  
  <rect x="460" y="240" width="100" height="30" rx="5" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="510" y="258" text-anchor="middle" font-size="10" fill="white">Improvement</text>
  
  <rect x="220" y="280" width="100" height="30" rx="5" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="270" y="298" text-anchor="middle" font-size="10" fill="white">Learning</text>
  
  <rect x="340" y="280" width="100" height="30" rx="5" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="390" y="298" text-anchor="middle" font-size="10" fill="white">Memory Update</text>
  
  <rect x="460" y="280" width="100" height="30" rx="5" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="510" y="298" text-anchor="middle" font-size="10" fill="white">Strategy Adapt</text>
  
  <!-- Arrows -->
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Forward flow -->
  <path d="M 170 110 L 210 110" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M 340 110 L 380 110" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M 510 110 L 550 110" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#arrow)"/>
  
  <!-- Reflection connections -->
  <path d="M 280 140 L 280 190 L 400 190 L 400 200" fill="none" stroke="#9b59b6" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M 450 140 L 450 190 L 400 190 L 400 200" fill="none" stroke="#9b59b6" stroke-width="2" marker-end="url(#arrow)"/>
  
  <!-- Feedback loop -->
  <path d="M 400 320 Q 300 380 110 380 Q 50 380 50 110 L 50 110" fill="none" stroke="#e67e22" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrow)"/>
  
  <text x="150" y="400" font-size="12" fill="#e67e22">Feedback Loop</text>
</svg>
```

### 2.2 Reflection Pattern Implementation

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

class ReflectionType(Enum):
    """Types of reflection analysis"""
    ERROR_ANALYSIS = "error_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"
    COMPLETENESS_CHECK = "completeness_check"
    ACCURACY_VERIFICATION = "accuracy_verification"
    IMPROVEMENT_IDENTIFICATION = "improvement_identification"
    LEARNING_EXTRACTION = "learning_extraction"

class IssueType(Enum):
    """Types of issues that can be identified"""
    FACTUAL_ERROR = "factual_error"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    INCOMPLETE_INFORMATION = "incomplete_information"
    UNCLEAR_REASONING = "unclear_reasoning"
    INEFFICIENT_APPROACH = "inefficient_approach"
    MISSING_CONTEXT = "missing_context"

@dataclass
class ReflectionCriteria:
    """Criteria for evaluating responses"""
    accuracy_weight: float = 0.3
    completeness_weight: float = 0.25
    clarity_weight: float = 0.2
    efficiency_weight: float = 0.15
    relevance_weight: float = 0.1
    
    def validate_weights(self) -> bool:
        """Ensure weights sum to 1.0"""
        total = (self.accuracy_weight + self.completeness_weight + 
                self.clarity_weight + self.efficiency_weight + self.relevance_weight)
        return abs(total - 1.0) < 0.001

@dataclass
class Issue:
    """Represents an identified issue in a response"""
    issue_type: IssueType
    description: str
    severity: float  # 0.0 to 1.0
    location: str  # Where in the response
    suggested_fix: str
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type.value,
            "description": self.description,
            "severity": self.severity,
            "location": self.location,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence
        }

@dataclass
class QualityScores:
    """Quality assessment scores"""
    accuracy: float
    completeness: float
    clarity: float
    efficiency: float
    relevance: float
    overall: float = field(init=False)
    
    def __post_init__(self):
        criteria = ReflectionCriteria()
        self.overall = (
            self.accuracy * criteria.accuracy_weight +
            self.completeness * criteria.completeness_weight +
            self.clarity * criteria.clarity_weight +
            self.efficiency * criteria.efficiency_weight +
            self.relevance * criteria.relevance_weight
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "efficiency": self.efficiency,
            "relevance": self.relevance,
            "overall": self.overall
        }

@dataclass
class ReflectionResult:
    """Result of reflection analysis"""
    original_response: str
    quality_scores: QualityScores
    issues_identified: List[Issue]
    improvement_suggestions: List[str]
    learning_points: List[str]
    needs_revision: bool
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_response": self.original_response,
            "quality_scores": self.quality_scores.to_dict(),
            "issues_identified": [issue.to_dict() for issue in self.issues_identified],
            "improvement_suggestions": self.improvement_suggestions,
            "learning_points": self.learning_points,
            "needs_revision": self.needs_revision,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }

class ReflectionEngine:
    """Engine for performing self-reflection on agent responses"""
    
    def __init__(self, llm_client, criteria: ReflectionCriteria = None):
        self.llm_client = llm_client
        self.criteria = criteria or ReflectionCriteria()
        self.reflection_history: List[ReflectionResult] = []
        self.learned_patterns: Dict[str, List[str]] = {
            "common_errors": [],
            "effective_strategies": [],
            "improvement_patterns": []
        }
    
    def reflect_on_response(self, original_response: str, 
                          context: str = "", 
                          expected_outcome: str = "") -> ReflectionResult:
        """Perform comprehensive reflection on a response"""
        
        # Analyze quality scores
        quality_scores = self._assess_quality(original_response, context, expected_outcome)
        
        # Identify specific issues
        issues = self._identify_issues(original_response, context, expected_outcome)
        
        # Generate improvement suggestions
        improvements = self._generate_improvements(original_response, issues, quality_scores)
        
        # Extract learning points
        learning_points = self._extract_learning_points(original_response, issues, improvements)
        
        # Determine if revision is needed
        needs_revision = self._needs_revision(quality_scores, issues)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(quality_scores, issues)
        
        result = ReflectionResult(
            original_response=original_response,
            quality_scores=quality_scores,
            issues_identified=issues,
            improvement_suggestions=improvements,
            learning_points=learning_points,
            needs_revision=needs_revision,
            confidence=confidence
        )
        
        self.reflection_history.append(result)
        self._update_learned_patterns(result)
        
        return result
    
    def _assess_quality(self, response: str, context: str, expected_outcome: str) -> QualityScores:
        """Assess the quality of a response across multiple dimensions"""
        
        prompt = f"""
        Evaluate the following response across these dimensions (score 0.0 to 1.0):
        
        Context: {context}
        Expected Outcome: {expected_outcome}
        Response: {response}
        
        Rate each dimension:
        1. Accuracy: How factually correct is the response?
        2. Completeness: Does it fully address the question/task?
        3. Clarity: How clear and understandable is it?
        4. Efficiency: How concise yet comprehensive is it?
        5. Relevance: How relevant is it to the context?
        
        Respond with JSON:
        {{
            "accuracy": 0.0-1.0,
            "completeness": 0.0-1.0,
            "clarity": 0.0-1.0,
            "efficiency": 0.0-1.0,
            "relevance": 0.0-1.0
        }}
        """
        
        response_text = self.llm_client.generate(prompt)
        
        try:
            scores_data = json.loads(response_text)
            return QualityScores(
                accuracy=scores_data["accuracy"],
                completeness=scores_data["completeness"],
                clarity=scores_data["clarity"],
                efficiency=scores_data["efficiency"],
                relevance=scores_data["relevance"]
            )
        except Exception:
            # Fallback to default scores
            return QualityScores(
                accuracy=0.7, completeness=0.7, clarity=0.7, 
                efficiency=0.7, relevance=0.7
            )
    
    def _identify_issues(self, response: str, context: str, expected_outcome: str) -> List[Issue]:
        """Identify specific issues in the response"""
        
        prompt = f"""
        Analyze the following response for potential issues:
        
        Context: {context}
        Expected Outcome: {expected_outcome}
        Response: {response}
        
        Look for these types of issues:
        - Factual errors
        - Logical inconsistencies
        - Incomplete information
        - Unclear reasoning
        - Inefficient approaches
        - Missing context
        
        For each issue found, provide:
        {{
            "type": "issue_type",
            "description": "what's wrong",
            "severity": 0.0-1.0,
            "location": "where in response",
            "suggested_fix": "how to fix it"
        }}
        
        Respond with JSON array of issues (empty array if no issues):
        [{{...}}, {{...}}]
        """
        
        response_text = self.llm_client.generate(prompt)
        
        try:
            issues_data = json.loads(response_text)
            issues = []
            
            for issue_data in issues_data:
                issue_type = IssueType(issue_data["type"]) if issue_data["type"] in [e.value for e in IssueType] else IssueType.UNCLEAR_REASONING
                
                issues.append(Issue(
                    issue_type=issue_type,
                    description=issue_data["description"],
                    severity=issue_data["severity"],
                    location=issue_data["location"],
                    suggested_fix=issue_data["suggested_fix"]
                ))
            
            return issues
        except Exception:
            return []  # Return empty list if parsing fails
    
    def _generate_improvements(self, response: str, issues: List[Issue], 
                             quality_scores: QualityScores) -> List[str]:
        """Generate specific improvement suggestions"""
        
        issues_summary = "\n".join([f"- {issue.description}" for issue in issues])
        scores_summary = f"Accuracy: {quality_scores.accuracy:.2f}, Completeness: {quality_scores.completeness:.2f}, Clarity: {quality_scores.clarity:.2f}"
        
        prompt = f"""
        Based on the analysis below, suggest specific improvements:
        
        Original Response: {response}
        
        Quality Scores: {scores_summary}
        
        Issues Identified:
        {issues_summary}
        
        Provide 3-5 specific, actionable improvement suggestions.
        Focus on the most impactful changes.
        """
        
        response_text = self.llm_client.generate(prompt)
        
        # Parse suggestions (assume they're in a list format)
        suggestions = [line.strip() for line in response_text.split('\n') 
                      if line.strip() and (line.strip().startswith('-') or line.strip().startswith('*'))]
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _extract_learning_points(self, response: str, issues: List[Issue], 
                               improvements: List[str]) -> List[str]:
        """Extract learning points from the reflection process"""
        
        prompt = f"""
        Based on this reflection analysis, what are the key learning points?
        
        Original Response: {response}
        Issues Found: {len(issues)}
        Improvements Suggested: {len(improvements)}
        
        Extract 2-3 key learning points that can inform future responses.
        Focus on patterns, strategies, or principles.
        """
        
        response_text = self.llm_client.generate(prompt)
        
        learning_points = [line.strip() for line in response_text.split('\n') 
                          if line.strip() and (line.strip().startswith('-') or line.strip().startswith('*'))]
        
        return learning_points[:3]  # Limit to 3 learning points
    
    def _needs_revision(self, quality_scores: QualityScores, issues: List[Issue]) -> bool:
        """Determine if the response needs revision"""
        # Revision needed if overall quality is low or high-severity issues exist
        if quality_scores.overall < 0.6:
            return True
        
        high_severity_issues = [issue for issue in issues if issue.severity > 0.7]
        return len(high_severity_issues) > 0
    
    def _calculate_confidence(self, quality_scores: QualityScores, issues: List[Issue]) -> float:
        """Calculate confidence in the reflection analysis"""
        base_confidence = quality_scores.overall
        
        # Reduce confidence based on number and severity of issues
        issue_penalty = sum(issue.severity for issue in issues) * 0.1
        
        return max(0.0, min(1.0, base_confidence - issue_penalty))
    
    def _update_learned_patterns(self, result: ReflectionResult):
        """Update learned patterns based on reflection results"""
        # Track common error patterns
        for issue in result.issues_identified:
            if issue.severity > 0.5:
                self.learned_patterns["common_errors"].append(issue.description)
        
        # Track effective improvements
        if result.quality_scores.overall > 0.8:
            self.learned_patterns["effective_strategies"].extend(result.improvement_suggestions)
        
        # Track improvement patterns
        self.learned_patterns["improvement_patterns"].extend(result.learning_points)
        
        # Keep only recent patterns (last 100 entries)
        for key in self.learned_patterns:
            self.learned_patterns[key] = self.learned_patterns[key][-100:]
    
    def get_learned_insights(self) -> Dict[str, Any]:
        """Get insights from accumulated learning"""
        return {
            "total_reflections": len(self.reflection_history),
            "average_quality": sum(r.quality_scores.overall for r in self.reflection_history) / len(self.reflection_history) if self.reflection_history else 0,
            "common_error_patterns": list(set(self.learned_patterns["common_errors"])),
            "effective_strategies": list(set(self.learned_patterns["effective_strategies"])),
            "improvement_patterns": list(set(self.learned_patterns["improvement_patterns"]))
        }

class ReflectiveAgent:
    """Agent that uses reflection to improve its responses"""
    
    def __init__(self, name: str, llm_client, max_iterations: int = 3):
        self.name = name
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.reflection_engine = ReflectionEngine(llm_client)
        self.response_history: List[Dict[str, Any]] = []
    
    def generate_response(self, prompt: str, context: str = "") -> Dict[str, Any]:
        """Generate a response with iterative reflection and improvement"""
        
        current_response = self.llm_client.generate(prompt)
        iteration = 0
        
        while iteration < self.max_iterations:
            # Reflect on current response
            reflection = self.reflection_engine.reflect_on_response(
                current_response, context, prompt
            )
            
            # If no revision needed, we're done
            if not reflection.needs_revision:
                break
            
            # Generate improved response
            improvement_prompt = self._create_improvement_prompt(
                prompt, current_response, reflection
            )
            
            improved_response = self.llm_client.generate(improvement_prompt)
            current_response = improved_response
            iteration += 1
        
        # Store the final result
        result = {
            "final_response": current_response,
            "iterations": iteration + 1,
            "final_reflection": reflection.to_dict(),
            "improvement_history": [r.to_dict() for r in self.reflection_engine.reflection_history[-iteration-1:]]
        }
        
        self.response_history.append(result)
        return result
    
    def _create_improvement_prompt(self, original_prompt: str, 
                                 current_response: str, 
                                 reflection: ReflectionResult) -> str:
        """Create a prompt for improving the response based on reflection"""
        
        issues_text = "\n".join([f"- {issue.description}: {issue.suggested_fix}" 
                                for issue in reflection.issues_identified])
        
        improvements_text = "\n".join([f"- {suggestion}" 
                                     for suggestion in reflection.improvement_suggestions])
        
        return f"""
        Original Question: {original_prompt}
        
        Previous Response: {current_response}
        
        Issues Identified:
        {issues_text}
        
        Improvement Suggestions:
        {improvements_text}
        
        Please provide an improved response that addresses these issues and incorporates the suggestions.
        Focus on accuracy, completeness, clarity, and relevance.
        """
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's performance"""
        if not self.response_history:
            return {"message": "No responses generated yet"}
        
        avg_iterations = sum(r["iterations"] for r in self.response_history) / len(self.response_history)
        
        final_qualities = [r["final_reflection"]["quality_scores"]["overall"] 
                          for r in self.response_history]
        avg_quality = sum(final_qualities) / len(final_qualities)
        
        return {
            "total_responses": len(self.response_history),
            "average_iterations": avg_iterations,
            "average_final_quality": avg_quality,
            "learned_insights": self.reflection_engine.get_learned_insights()
        }
```

### 2.3 Reflection Pattern Usage Example

```python
class MockLLMClient:
    """Enhanced mock LLM client for reflection demonstration"""
    
    def __init__(self):
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        self.call_count += 1
        
        if "Evaluate the following response" in prompt:
            return '''
            {
                "accuracy": 0.7,
                "completeness": 0.6,
                "clarity": 0.8,
                "efficiency": 0.7,
                "relevance": 0.9
            }
            '''
        elif "Analyze the following response for potential issues" in prompt:
            return '''
            [
                {
                    "type": "incomplete_information",
                    "description": "Missing specific examples",
                    "severity": 0.6,
                    "location": "Throughout the response",
                    "suggested_fix": "Add concrete examples to illustrate points"
                }
            ]
            '''
        elif "suggest specific improvements" in prompt:
            return "- Add more specific examples\n- Improve structure and organization\n- Include relevant statistics"
        elif "key learning points" in prompt:
            return "- Always include concrete examples\n- Structure responses clearly\n- Verify factual accuracy"
        elif "improved response" in prompt:
            return "This is an improved response with better structure, specific examples, and clearer explanations."
        else:
            return "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."

def demonstrate_reflection_pattern():
    """Demonstrate the reflection pattern"""
    
    llm_client = MockLLMClient()
    agent = ReflectiveAgent("reflective_assistant", llm_client, max_iterations=2)
    
    # Test the reflective agent
    prompt = "Explain machine learning and provide examples of its applications"
    context = "Educational context for beginners"
    
    result = agent.generate_response(prompt, context)
    
    print("=== Reflective Agent Demonstration ===")
    print(f"Original Prompt: {prompt}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final Response: {result['final_response']}")
    print(f"Final Quality Score: {result['final_reflection']['quality_scores']['overall']:.2f}")
    
    # Show performance summary
    performance = agent.get_performance_summary()
    print("\n=== Performance Summary ===")
    print(f"Total Responses: {performance['total_responses']}")
    print(f"Average Iterations: {performance['average_iterations']:.1f}")
    print(f"Average Quality: {performance['average_final_quality']:.2f}")
    
    return result

if __name__ == "__main__":
    demonstrate_reflection_pattern()
```

---

## 3. CodeAct Dynamic Execution

### 3.1 Understanding CodeAct Pattern

CodeAct (Code Acting) is a design pattern where agents can dynamically generate, execute, and debug code to solve problems. This pattern is particularly powerful for tasks requiring computation, data analysis, or complex logic.

#### CodeAct Architecture

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">CodeAct Dynamic Execution Architecture</text>
  
  <!-- Problem Analysis -->
  <rect x="50" y="80" width="150" height="80" rx="10" fill="#3498db" stroke="#2980b9" stroke-width="2"/>
  <text x="125" y="110" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Problem</text>
  <text x="125" y="125" text-anchor="middle" font-size="12" fill="white">Analysis</text>
  <text x="125" y="140" text-anchor="middle" font-size="10" fill="white">Understand Requirements</text>
  
  <!-- Code Generation -->
  <rect x="250" y="80" width="150" height="80" rx="10" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
  <text x="325" y="110" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Code</text>
  <text x="325" y="125" text-anchor="middle" font-size="12" fill="white">Generation</text>
  <text x="325" y="140" text-anchor="middle" font-size="10" fill="white">Create Solution</text>
  
  <!-- Execution Engine -->
  <rect x="450" y="80" width="150" height="80" rx="10" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="525" y="110" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Execution</text>
  <text x="525" y="125" text-anchor="middle" font-size="12" fill="white">Engine</text>
  <text x="525" y="140" text-anchor="middle" font-size="10" fill="white">Run & Monitor</text>
  
  <!-- Result Validation -->
  <rect x="650" y="80" width="120" height="80" rx="10" fill="#27ae60" stroke="229954" stroke-width="2"/>
  <text x="710" y="110" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Result</text>
  <text x="710" y="125" text-anchor="middle" font-size="12" fill="white">Validation</text>
  <text x="710" y="140" text-anchor="middle" font-size="10" fill="white">Verify Output</text>
  
  <!-- Security Sandbox -->
  <rect x="300" y="200" width="200" height="100" rx="10" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2"/>
  <text x="400" y="225" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Security Sandbox</text>
  
  <!-- Sandbox Components -->
  <rect x="320" y="240" width="80" height="25" rx="3" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="360" y="255" text-anchor="middle" font-size="9" fill="white">Isolation</text>
  
  <rect x="420" y="240" width="80" height="25" rx="3" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="460" y="255" text-anchor="middle" font-size="9" fill="white">Monitoring</text>
  
  <rect x="320" y="270" width="80" height="25" rx="3" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="360" y="285" text-anchor="middle" font-size="9" fill="white">Limits</text>
  
  <rect x="420" y="270" width="80" height="25" rx="3" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="460" y="285" text-anchor="middle" font-size="9" fill="white">Cleanup</text>
  
  <!-- Debug Loop -->
  <rect x="250" y="350" width="300" height="80" rx="10" fill="#fff3cd" stroke="#ffc107" stroke-width="2"/>
  <text x="400" y="375" text-anchor="middle" font-size="14" font-weight="bold" fill="#856404">Debug & Iterate Loop</text>
  <text x="400" y="395" text-anchor="middle" font-size="12" fill="#856404">Error Detection → Analysis → Fix → Retry</text>
  <text x="400" y="410" text-anchor="middle" font-size="10" fill="#856404">Automatic debugging and code improvement</text>
  
  <!-- Arrows -->
  <defs>
    <marker id="codeact-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Forward flow -->
  <path d="M 200 120 L 240 120" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#codeact-arrow)"/>
  <path d="M 400 120 L 440 120" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#codeact-arrow)"/>
  <path d="M 600 120 L 640 120" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#codeact-arrow)"/>
  
  <!-- Sandbox connections -->
  <path d="M 525 160 L 525 180 L 400 180 L 400 200" fill="none" stroke="#9b59b6" stroke-width="2" marker-end="url(#codeact-arrow)"/>
  
  <!-- Debug loop -->
   <path d="M 400 300 L 400 340" fill="none" stroke="#ffc107" stroke-width="2" marker-end="url(#codeact-arrow)"/>
   <path d="M 250 390 Q 150 450 150 120 L 200 120" fill="none" stroke="#ffc107" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#codeact-arrow)"/>
 </svg>
 ```

### 3.2 Core CodeAct Components

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
import subprocess
import tempfile
import os
import sys
import ast
import traceback
import time
import threading
from datetime import datetime, timedelta
import json
import logging

class ExecutionStatus(Enum):
    """Status of code execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class CodeType(Enum):
    """Types of code that can be executed"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    SQL = "sql"
    R = "r"

@dataclass
class ExecutionLimits:
    """Limits for code execution"""
    max_execution_time: float = 30.0  # seconds
    max_memory_mb: int = 512
    max_output_size: int = 10000  # characters
    allowed_imports: List[str] = field(default_factory=lambda: [
        'math', 'random', 'datetime', 'json', 'csv', 'os', 'sys',
        'numpy', 'pandas', 'matplotlib', 'requests', 'sqlite3'
    ])
    forbidden_functions: List[str] = field(default_factory=lambda: [
        'exec', 'eval', 'compile', 'open', '__import__', 'input',
        'raw_input', 'file', 'execfile', 'reload'
    ])

@dataclass
class ExecutionResult:
    """Result of code execution"""
    status: ExecutionStatus
    output: str
    error: Optional[str]
    execution_time: float
    memory_used: Optional[int]
    return_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "memory_used": self.memory_used,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

class SecurityValidator:
    """Validates code for security issues before execution"""
    
    def __init__(self, limits: ExecutionLimits):
        self.limits = limits
        self.logger = logging.getLogger("SecurityValidator")
    
    def validate_python_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python code for security issues"""
        issues = []
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Check for forbidden functions and imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.limits.allowed_imports:
                            issues.append(f"Forbidden import: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.limits.allowed_imports:
                        issues.append(f"Forbidden import from: {node.module}")
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.limits.forbidden_functions:
                            issues.append(f"Forbidden function: {node.func.id}")
                
                elif isinstance(node, ast.Attribute):
                    # Check for dangerous attribute access
                    if node.attr in ['__globals__', '__locals__', '__dict__']:
                        issues.append(f"Dangerous attribute access: {node.attr}")
            
            return len(issues) == 0, issues
            
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
            return False, issues
    
    def validate_code(self, code: str, code_type: CodeType) -> Tuple[bool, List[str]]:
        """Validate code based on its type"""
        if code_type == CodeType.PYTHON:
            return self.validate_python_code(code)
        else:
            # Basic validation for other languages
            dangerous_patterns = ['rm -rf', 'del /f', 'format c:', 'sudo', 'chmod 777']
            issues = []
            
            for pattern in dangerous_patterns:
                if pattern in code.lower():
                    issues.append(f"Dangerous pattern detected: {pattern}")
            
            return len(issues) == 0, issues

class ExecutionSandbox:
    """Secure sandbox for code execution"""
    
    def __init__(self, limits: ExecutionLimits):
        self.limits = limits
        self.validator = SecurityValidator(limits)
        self.logger = logging.getLogger("ExecutionSandbox")
        self._active_processes: Dict[str, subprocess.Popen] = {}
    
    def execute_python(self, code: str, execution_id: str) -> ExecutionResult:
        """Execute Python code in a controlled environment"""
        start_time = time.time()
        
        # Validate code first
        is_valid, issues = self.validator.validate_python_code(code)
        if not is_valid:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Security validation failed: {'; '.join(issues)}",
                execution_time=time.time() - start_time
            )
        
        # Create a restricted execution environment
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed
            }
        }
        
        # Add allowed imports
        for module_name in self.limits.allowed_imports:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
        
        try:
            # Capture output
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute with timeout
            result = None
            error = None
            
            def target():
                nonlocal result, error
                try:
                    result = exec(code, restricted_globals)
                except Exception as e:
                    error = str(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.limits.max_execution_time)
            
            if thread.is_alive():
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    output=stdout_capture.getvalue(),
                    error="Execution timed out",
                    execution_time=self.limits.max_execution_time
                )
            
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            output = stdout_capture.getvalue()
            error_output = stderr_capture.getvalue()
            
            if error:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    output=output,
                    error=error + "\n" + error_output if error_output else error,
                    execution_time=time.time() - start_time
                )
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output,
                error=error_output if error_output else None,
                execution_time=time.time() - start_time,
                return_value=result
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def execute_code(self, code: str, code_type: CodeType, execution_id: str) -> ExecutionResult:
        """Execute code based on its type"""
        if code_type == CodeType.PYTHON:
            return self.execute_python(code, execution_id)
        else:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Code type {code_type.value} not supported yet",
                execution_time=0.0
            )
    
    def cleanup(self, execution_id: str):
        """Clean up resources for a specific execution"""
        if execution_id in self._active_processes:
            process = self._active_processes[execution_id]
            if process.poll() is None:  # Still running
                process.terminate()
                process.wait(timeout=5)
            del self._active_processes[execution_id]

class CodeActAgent:
    """Main CodeAct agent that can reason and act through code"""
    
    def __init__(self, llm_client, execution_limits: Optional[ExecutionLimits] = None):
        self.llm_client = llm_client
        self.execution_limits = execution_limits or ExecutionLimits()
        self.sandbox = ExecutionSandbox(self.execution_limits)
        self.execution_history: List[ExecutionResult] = []
        self.logger = logging.getLogger("CodeActAgent")
        self._execution_counter = 0
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        self._execution_counter += 1
        return f"exec_{int(time.time())}_{self._execution_counter}"
    
    def _create_code_prompt(self, task: str, context: str = "") -> str:
        """Create prompt for code generation"""
        prompt = f"""
You are a CodeAct agent that solves problems by writing and executing code.

Task: {task}

{f"Context: {context}" if context else ""}

Please provide Python code to solve this task. Follow these guidelines:
1. Write clean, well-commented code
2. Handle errors gracefully
3. Print intermediate results for debugging
4. Use only allowed imports: {', '.join(self.execution_limits.allowed_imports)}
5. Avoid forbidden functions: {', '.join(self.execution_limits.forbidden_functions)}

Provide your code in the following format:
```python
# Your code here
```

Explain your approach before the code.
"""
        return prompt
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response"""
        import re
        
        # Look for code blocks
        code_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: look for any code-like content
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code or line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return None
    
    def solve_task(self, task: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Solve a task using the CodeAct approach"""
        self.logger.info(f"Starting CodeAct task: {task}")
        
        results = {
            "task": task,
            "iterations": [],
            "final_status": "pending",
            "solution": None,
            "total_execution_time": 0.0
        }
        
        context = ""
        
        for iteration in range(max_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Generate code
            prompt = self._create_code_prompt(task, context)
            
            try:
                response = self.llm_client.generate(prompt)
                code = self._extract_code_from_response(response)
                
                if not code:
                    results["iterations"].append({
                        "iteration": iteration + 1,
                        "status": "error",
                        "error": "No code found in LLM response",
                        "response": response
                    })
                    continue
                
                # Execute code
                execution_id = self._generate_execution_id()
                execution_result = self.sandbox.execute_code(
                    code, CodeType.PYTHON, execution_id
                )
                
                self.execution_history.append(execution_result)
                results["total_execution_time"] += execution_result.execution_time
                
                iteration_result = {
                    "iteration": iteration + 1,
                    "code": code,
                    "execution_result": execution_result.to_dict(),
                    "llm_response": response
                }
                
                results["iterations"].append(iteration_result)
                
                # Check if task is solved
                if execution_result.status == ExecutionStatus.SUCCESS:
                    if self._is_task_solved(task, execution_result.output):
                        results["final_status"] = "success"
                        results["solution"] = {
                            "code": code,
                            "output": execution_result.output,
                            "execution_time": execution_result.execution_time
                        }
                        break
                
                # Update context for next iteration
                context = self._build_context(results["iterations"])
                
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "status": "error",
                    "error": str(e)
                })
        
        if results["final_status"] == "pending":
            results["final_status"] = "failed"
        
        return results
    
    def _is_task_solved(self, task: str, output: str) -> bool:
        """Determine if the task has been solved based on output"""
        # Simple heuristic - can be enhanced with more sophisticated checking
        if "error" in output.lower() or "exception" in output.lower():
            return False
        
        # Check if output contains meaningful results
        if len(output.strip()) > 0:
            return True
        
        return False
    
    def _build_context(self, iterations: List[Dict]) -> str:
        """Build context from previous iterations"""
        if not iterations:
            return ""
        
        context_parts = ["Previous attempts:"]
        
        for i, iteration in enumerate(iterations[-2:], 1):  # Last 2 iterations
            exec_result = iteration.get("execution_result", {})
            status = exec_result.get("status", "unknown")
            
            if status == "error":
                error = exec_result.get("error", "Unknown error")
                context_parts.append(f"Attempt {i}: Failed with error: {error}")
            elif status == "success":
                output = exec_result.get("output", "")
                context_parts.append(f"Attempt {i}: Succeeded with output: {output[:200]}...")
        
        return "\n".join(context_parts)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about code executions"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for r in self.execution_history if r.status == ExecutionStatus.SUCCESS)
        total_time = sum(r.execution_time for r in self.execution_history)
        avg_time = total_time / total_executions if total_executions > 0 else 0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "error_types": self._get_error_types()
        }
    
    def _get_error_types(self) -> Dict[str, int]:
        """Get frequency of different error types"""
        error_types = {}
        
        for result in self.execution_history:
            if result.status == ExecutionStatus.ERROR and result.error:
                # Simple error categorization
                if "syntax" in result.error.lower():
                    error_types["syntax_error"] = error_types.get("syntax_error", 0) + 1
                elif "name" in result.error.lower() and "not defined" in result.error.lower():
                    error_types["name_error"] = error_types.get("name_error", 0) + 1
                elif "type" in result.error.lower():
                    error_types["type_error"] = error_types.get("type_error", 0) + 1
                else:
                    error_types["other_error"] = error_types.get("other_error", 0) + 1
        
        return error_types
```

### 3.3 Usage Example

```python
# Mock LLM client for demonstration
class MockLLMClient:
    def generate(self, prompt: str) -> str:
        # Simulate different responses based on the task
        if "calculate" in prompt.lower() and "fibonacci" in prompt.lower():
            return """
I'll calculate the Fibonacci sequence using an iterative approach for efficiency.

```python
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    
    print(f"Fibonacci sequence of length {n}: {fib_sequence}")
    return fib_sequence

# Calculate first 10 Fibonacci numbers
result = fibonacci(10)
print(f"Result: {result}")
```
"""
        else:
            return """
I'll solve this step by step.

```python
# Simple calculation
result = 2 + 2
print(f"The result is: {result}")
```
"""

# Example usage
def demonstrate_codeact():
    # Initialize the agent
    llm_client = MockLLMClient()
    limits = ExecutionLimits(
        max_execution_time=10.0,
        max_memory_mb=256
    )
    
    agent = CodeActAgent(llm_client, limits)
    
    # Solve a task
    task = "Calculate the first 10 numbers in the Fibonacci sequence"
    result = agent.solve_task(task, max_iterations=2)
    
    print("=== CodeAct Task Results ===")
    print(f"Task: {result['task']}")
    print(f"Final Status: {result['final_status']}")
    print(f"Total Execution Time: {result['total_execution_time']:.3f}s")
    
    if result['solution']:
        print("\n=== Solution ===")
        print("Code:")
        print(result['solution']['code'])
        print("\nOutput:")
        print(result['solution']['output'])
    
    print("\n=== Execution Statistics ===")
    stats = agent.get_execution_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    demonstrate_codeact()
```

## 4. Planning Patterns

### 4.1 Hierarchical Task Planning

Hierarchical planning breaks down complex tasks into manageable subtasks, creating a tree-like structure of goals and actions.

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#2c3e50">Hierarchical Task Planning Architecture</text>
  
  <!-- Define arrow marker -->
  <defs>
    <marker id="planning-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Main Goal -->
  <rect x="300" y="60" width="200" height="60" rx="10" fill="#3498db" stroke="#2980b9" stroke-width="2"/>
  <text x="400" y="85" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Main Goal</text>
  <text x="400" y="105" text-anchor="middle" font-size="12" fill="white">High-level Objective</text>
  
  <!-- Level 1 Subgoals -->
  <rect x="100" y="180" width="150" height="50" rx="8" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
  <text x="175" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Subgoal 1</text>
  <text x="175" y="215" text-anchor="middle" font-size="10" fill="white">Planning</text>
  
  <rect x="325" y="180" width="150" height="50" rx="8" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
  <text x="400" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Subgoal 2</text>
  <text x="400" y="215" text-anchor="middle" font-size="10" fill="white">Execution</text>
  
  <rect x="550" y="180" width="150" height="50" rx="8" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
  <text x="625" y="200" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Subgoal 3</text>
  <text x="625" y="215" text-anchor="middle" font-size="10" fill="white">Monitoring</text>
  
  <!-- Level 2 Tasks -->
  <rect x="50" y="300" width="100" height="40" rx="6" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="100" y="315" text-anchor="middle" font-size="10" font-weight="bold" fill="white">Task 1.1</text>
  <text x="100" y="330" text-anchor="middle" font-size="9" fill="white">Analyze</text>
  
  <rect x="200" y="300" width="100" height="40" rx="6" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="250" y="315" text-anchor="middle" font-size="10" font-weight="bold" fill="white">Task 1.2</text>
  <text x="250" y="330" text-anchor="middle" font-size="9" fill="white">Design</text>
  
  <rect x="350" y="300" width="100" height="40" rx="6" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="400" y="315" text-anchor="middle" font-size="10" font-weight="bold" fill="white">Task 2.1</text>
  <text x="400" y="330" text-anchor="middle" font-size="9" fill="white">Implement</text>
  
  <rect x="500" y="300" width="100" height="40" rx="6" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="550" y="315" text-anchor="middle" font-size="10" font-weight="bold" fill="white">Task 2.2</text>
  <text x="550" y="330" text-anchor="middle" font-size="9" fill="white">Test</text>
  
  <rect x="650" y="300" width="100" height="40" rx="6" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="700" y="315" text-anchor="middle" font-size="10" font-weight="bold" fill="white">Task 3.1</text>
  <text x="700" y="330" text-anchor="middle" font-size="9" fill="white">Monitor</text>
  
  <!-- Level 3 Actions -->
  <circle cx="75" cy="420" r="20" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="75" y="425" text-anchor="middle" font-size="9" fill="white">A1</text>
  
  <circle cx="125" cy="420" r="20" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="125" y="425" text-anchor="middle" font-size="9" fill="white">A2</text>
  
  <circle cx="225" cy="420" r="20" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="225" y="425" text-anchor="middle" font-size="9" fill="white">A3</text>
  
  <circle cx="275" cy="420" r="20" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="275" y="425" text-anchor="middle" font-size="9" fill="white">A4</text>
  
  <circle cx="375" cy="420" r="20" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="375" y="425" text-anchor="middle" font-size="9" fill="white">A5</text>
  
  <circle cx="425" cy="420" r="20" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="425" y="425" text-anchor="middle" font-size="9" fill="white">A6</text>
  
  <!-- Arrows from Main Goal to Subgoals -->
  <path d="M 350 120 L 200 180" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 400 120 L 400 180" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 450 120 L 600 180" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  
  <!-- Arrows from Subgoals to Tasks -->
  <path d="M 150 230 L 100 300" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 200 230 L 250 300" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 375 230 L 400 300" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 425 230 L 550 300" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 625 230 L 700 300" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  
  <!-- Arrows from Tasks to Actions -->
  <path d="M 80 340 L 75 400" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 120 340 L 125 400" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 230 340 L 225 400" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 270 340 L 275 400" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 380 340 L 375 400" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  <path d="M 420 340 L 425 400" fill="none" stroke="#34495e" stroke-width="2" marker-end="url(#planning-arrow)"/>
  
  <!-- Legend -->
  <rect x="50" y="500" width="700" height="80" rx="5" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="1"/>
  <text x="60" y="520" font-size="12" font-weight="bold" fill="#2c3e50">Planning Hierarchy:</text>
  
  <rect x="70" y="530" width="15" height="15" fill="#3498db"/>
  <text x="95" y="542" font-size="10" fill="#2c3e50">Main Goal (Strategic Level)</text>
  
  <rect x="250" y="530" width="15" height="15" fill="#e74c3c"/>
  <text x="275" y="542" font-size="10" fill="#2c3e50">Subgoals (Tactical Level)</text>
  
  <rect x="430" y="530" width="15" height="15" fill="#f39c12"/>
  <text x="455" y="542" font-size="10" fill="#2c3e50">Tasks (Operational Level)</text>
  
  <circle cx="625" cy="537" r="7" fill="#27ae60"/>
  <text x="645" y="542" font-size="10" fill="#2c3e50">Actions (Execution Level)</text>
  
  <text x="70" y="565" font-size="10" fill="#2c3e50">• Each level provides increasing detail and specificity</text>
  <text x="70" y="575" font-size="10" fill="#2c3e50">• Higher levels set context and constraints for lower levels</text>
</svg>
```

### 4.2 Hierarchical Planning Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Callable
from enum import Enum
import uuid
import time
from datetime import datetime
import logging

class TaskStatus(Enum):
    """Status of a task in the hierarchy"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TaskConstraint:
    """Represents a constraint on task execution"""
    constraint_type: str  # "time", "resource", "dependency", "condition"
    value: Any
    description: str

@dataclass
class TaskResult:
    """Result of task execution"""
    success: bool
    output: Any
    error: Optional[str]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class Task:
    """Represents a task in the hierarchical planning system"""
    
    def __init__(self, 
                 task_id: str,
                 name: str,
                 description: str,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 estimated_duration: float = 0.0,
                 constraints: List[TaskConstraint] = None):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.priority = priority
        self.estimated_duration = estimated_duration
        self.constraints = constraints or []
        
        # Hierarchy relationships
        self.parent: Optional['Task'] = None
        self.children: List['Task'] = []
        self.dependencies: Set[str] = set()  # Task IDs this task depends on
        
        # Execution state
        self.status = TaskStatus.PENDING
        self.result: Optional[TaskResult] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Execution function
        self.executor: Optional[Callable] = None
        
        self.logger = logging.getLogger(f"Task-{task_id}")
    
    def add_child(self, child_task: 'Task'):
        """Add a child task"""
        child_task.parent = self
        self.children.append(child_task)
    
    def add_dependency(self, task_id: str):
        """Add a dependency on another task"""
        self.dependencies.add(task_id)
    
    def is_ready_to_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute based on dependencies"""
        if self.status != TaskStatus.PENDING:
            return False
        
        # Check if all dependencies are completed
        return self.dependencies.issubset(completed_tasks)
    
    def can_execute_constraints(self) -> Tuple[bool, List[str]]:
        """Check if task can execute based on constraints"""
        issues = []
        
        for constraint in self.constraints:
            if constraint.constraint_type == "time":
                current_time = datetime.now()
                if current_time < constraint.value:
                    issues.append(f"Time constraint not met: {constraint.description}")
            elif constraint.constraint_type == "condition":
                if not constraint.value():  # Assume it's a callable
                    issues.append(f"Condition not met: {constraint.description}")
        
        return len(issues) == 0, issues
    
    def execute(self) -> TaskResult:
        """Execute the task"""
        if not self.executor:
            return TaskResult(
                success=False,
                output=None,
                error="No executor defined for task",
                execution_time=0.0
            )
        
        self.status = TaskStatus.IN_PROGRESS
        self.start_time = datetime.now()
        
        try:
            start_time = time.time()
            output = self.executor(self)
            execution_time = time.time() - start_time
            
            result = TaskResult(
                success=True,
                output=output,
                error=None,
                execution_time=execution_time
            )
            
            self.status = TaskStatus.COMPLETED
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TaskResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time
            )
            
            self.status = TaskStatus.FAILED
            self.logger.error(f"Task execution failed: {str(e)}")
        
        finally:
            self.end_time = datetime.now()
            self.result = result
        
        return result
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress information for this task and its children"""
        if not self.children:
            # Leaf task
            return {
                "task_id": self.task_id,
                "name": self.name,
                "status": self.status.value,
                "progress": 1.0 if self.status == TaskStatus.COMPLETED else 0.0,
                "is_leaf": True
            }
        
        # Parent task - calculate progress based on children
        total_children = len(self.children)
        completed_children = sum(1 for child in self.children if child.status == TaskStatus.COMPLETED)
        
        child_progress = [child.get_progress() for child in self.children]
        avg_child_progress = sum(cp["progress"] for cp in child_progress) / total_children if total_children > 0 else 0.0
        
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "progress": avg_child_progress,
            "is_leaf": False,
            "children": child_progress,
            "completed_children": completed_children,
            "total_children": total_children
        }

class HierarchicalPlanner:
    """Manages hierarchical task planning and execution"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.root_tasks: List[Task] = []
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.execution_queue: List[Task] = []
        self.logger = logging.getLogger("HierarchicalPlanner")
    
    def create_task(self, 
                   name: str, 
                   description: str,
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   estimated_duration: float = 0.0,
                   constraints: List[TaskConstraint] = None,
                   executor: Callable = None) -> Task:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        task = Task(task_id, name, description, priority, estimated_duration, constraints)
        task.executor = executor
        
        self.tasks[task_id] = task
        return task
    
    def add_root_task(self, task: Task):
        """Add a root-level task"""
        if task.task_id not in self.tasks:
            self.tasks[task.task_id] = task
        self.root_tasks.append(task)
    
    def build_execution_plan(self) -> List[Task]:
        """Build an execution plan based on dependencies and priorities"""
        execution_plan = []
        visited = set()
        
        def add_task_and_dependencies(task: Task):
            if task.task_id in visited:
                return
            
            visited.add(task.task_id)
            
            # Add dependencies first
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    add_task_and_dependencies(self.tasks[dep_id])
            
            # Add children in order
            for child in task.children:
                add_task_and_dependencies(child)
            
            # Add the task itself if it's a leaf task (has executor)
            if task.executor and not task.children:
                execution_plan.append(task)
        
        # Process all root tasks
        for root_task in self.root_tasks:
            add_task_and_dependencies(root_task)
        
        # Sort by priority and estimated duration
        execution_plan.sort(key=lambda t: (t.priority.value, t.estimated_duration), reverse=True)
        
        return execution_plan
    
    def execute_plan(self, max_parallel: int = 1) -> Dict[str, Any]:
        """Execute the planned tasks"""
        execution_plan = self.build_execution_plan()
        
        results = {
            "total_tasks": len(execution_plan),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "execution_results": [],
            "start_time": datetime.now(),
            "end_time": None,
            "total_execution_time": 0.0
        }
        
        self.logger.info(f"Starting execution of {len(execution_plan)} tasks")
        
        for task in execution_plan:
            # Check if task is ready to execute
            if not task.is_ready_to_execute(self.completed_tasks):
                self.logger.warning(f"Task {task.name} not ready - dependencies not met")
                continue
            
            # Check constraints
            can_execute, constraint_issues = task.can_execute_constraints()
            if not can_execute:
                self.logger.warning(f"Task {task.name} blocked by constraints: {constraint_issues}")
                task.status = TaskStatus.BLOCKED
                continue
            
            # Execute task
            self.logger.info(f"Executing task: {task.name}")
            result = task.execute()
            
            results["execution_results"].append({
                "task_id": task.task_id,
                "task_name": task.name,
                "result": result.__dict__
            })
            
            results["total_execution_time"] += result.execution_time
            
            if result.success:
                self.completed_tasks.add(task.task_id)
                results["completed_tasks"] += 1
                self.logger.info(f"Task {task.name} completed successfully")
            else:
                self.failed_tasks.add(task.task_id)
                results["failed_tasks"] += 1
                self.logger.error(f"Task {task.name} failed: {result.error}")
        
        results["end_time"] = datetime.now()
        results["success_rate"] = results["completed_tasks"] / results["total_tasks"] if results["total_tasks"] > 0 else 0
        
        return results
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress of all tasks"""
        root_progress = [task.get_progress() for task in self.root_tasks]
        
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_tasks)
        failed_tasks = len(self.failed_tasks)
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "in_progress_tasks": total_tasks - completed_tasks - failed_tasks,
            "overall_progress": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "root_tasks_progress": root_progress
        }
    
    def visualize_hierarchy(self) -> str:
        """Generate a text visualization of the task hierarchy"""
        def format_task(task: Task, indent: int = 0) -> str:
            prefix = "  " * indent
            status_symbol = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.CANCELLED: "🚫",
                TaskStatus.BLOCKED: "🔒"
            }.get(task.status, "❓")
            
            lines = [f"{prefix}{status_symbol} {task.name} ({task.status.value})"]
            
            for child in task.children:
                lines.extend(format_task(child, indent + 1))
            
            return lines
        
        all_lines = []
        for root_task in self.root_tasks:
            all_lines.extend(format_task(root_task))
        
        return "\n".join(all_lines)
```

### 4.3 Usage Example

```python
# Example task executors
def analyze_requirements(task: Task) -> str:
    """Simulate requirements analysis"""
    time.sleep(0.1)  # Simulate work
    return "Requirements analyzed: User needs data processing pipeline"

def design_architecture(task: Task) -> str:
    """Simulate architecture design"""
    time.sleep(0.2)  # Simulate work
    return "Architecture designed: Microservices with event-driven communication"

def implement_service(task: Task) -> str:
    """Simulate service implementation"""
    time.sleep(0.3)  # Simulate work
    return f"Service {task.name} implemented successfully"

def run_tests(task: Task) -> str:
    """Simulate testing"""
    time.sleep(0.1)  # Simulate work
    return f"Tests for {task.name} passed"

def deploy_service(task: Task) -> str:
    """Simulate deployment"""
    time.sleep(0.2)  # Simulate work
    return f"Service {task.name} deployed to production"

# Example usage
def demonstrate_hierarchical_planning():
    planner = HierarchicalPlanner()
    
    # Create main goal
    main_goal = planner.create_task(
        "Build Data Processing System",
        "Complete data processing system with microservices",
        TaskPriority.HIGH
    )
    
    # Create planning phase
    planning_phase = planner.create_task(
        "Planning Phase",
        "Analyze requirements and design architecture",
        TaskPriority.HIGH
    )
    
    requirements_task = planner.create_task(
        "Analyze Requirements",
        "Gather and analyze system requirements",
        TaskPriority.HIGH,
        estimated_duration=1.0,
        executor=analyze_requirements
    )
    
    design_task = planner.create_task(
        "Design Architecture",
        "Design system architecture",
        TaskPriority.HIGH,
        estimated_duration=2.0,
        executor=design_architecture
    )
    
    # Create implementation phase
    implementation_phase = planner.create_task(
        "Implementation Phase",
        "Implement all services",
        TaskPriority.MEDIUM
    )
    
    api_service = planner.create_task(
        "API Service",
        "Implement REST API service",
        TaskPriority.HIGH,
        estimated_duration=3.0,
        executor=implement_service
    )
    
    data_service = planner.create_task(
        "Data Service",
        "Implement data processing service",
        TaskPriority.HIGH,
        estimated_duration=4.0,
        executor=implement_service
    )
    
    # Create testing phase
    testing_phase = planner.create_task(
        "Testing Phase",
        "Test all services",
        TaskPriority.MEDIUM
    )
    
    api_tests = planner.create_task(
        "API Tests",
        "Test API service",
        TaskPriority.MEDIUM,
        estimated_duration=1.0,
        executor=run_tests
    )
    
    data_tests = planner.create_task(
        "Data Tests",
        "Test data service",
        TaskPriority.MEDIUM,
        estimated_duration=1.0,
        executor=run_tests
    )
    
    # Create deployment phase
    deployment_phase = planner.create_task(
        "Deployment Phase",
        "Deploy all services",
        TaskPriority.LOW
    )
    
    deploy_api = planner.create_task(
        "Deploy API",
        "Deploy API service",
        TaskPriority.MEDIUM,
        estimated_duration=1.0,
        executor=deploy_service
    )
    
    deploy_data = planner.create_task(
        "Deploy Data Service",
        "Deploy data service",
        TaskPriority.MEDIUM,
        estimated_duration=1.0,
        executor=deploy_service
    )
    
    # Build hierarchy
    main_goal.add_child(planning_phase)
    main_goal.add_child(implementation_phase)
    main_goal.add_child(testing_phase)
    main_goal.add_child(deployment_phase)
    
    planning_phase.add_child(requirements_task)
    planning_phase.add_child(design_task)
    
    implementation_phase.add_child(api_service)
    implementation_phase.add_child(data_service)
    
    testing_phase.add_child(api_tests)
    testing_phase.add_child(data_tests)
    
    deployment_phase.add_child(deploy_api)
    deployment_phase.add_child(deploy_data)
    
    # Set dependencies
    design_task.add_dependency(requirements_task.task_id)
    api_service.add_dependency(design_task.task_id)
    data_service.add_dependency(design_task.task_id)
    api_tests.add_dependency(api_service.task_id)
    data_tests.add_dependency(data_service.task_id)
    deploy_api.add_dependency(api_tests.task_id)
    deploy_data.add_dependency(data_tests.task_id)
    
    # Add to planner
    planner.add_root_task(main_goal)
    
    print("=== Task Hierarchy ===")
    print(planner.visualize_hierarchy())
    
    print("\n=== Executing Plan ===")
    results = planner.execute_plan()
    
    print(f"\nExecution completed:")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Completed: {results['completed_tasks']}")
    print(f"Failed: {results['failed_tasks']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Total execution time: {results['total_execution_time']:.2f}s")
    
    print("\n=== Final Progress ===")
    progress = planner.get_overall_progress()
    print(f"Overall progress: {progress['overall_progress']:.2%}")
    
    print("\n=== Updated Hierarchy ===")
    print(planner.visualize_hierarchy())

if __name__ == "__main__":
    demonstrate_hierarchical_planning()
```

## 5. Hands-on Exercises

### Exercise 1: Building a Complete ReAct Agent

Create a ReAct agent that can solve multi-step reasoning problems.

```python
class AdvancedReActAgent:
    """Advanced ReAct agent with enhanced reasoning capabilities"""
    
    def __init__(self, llm_client, tools: List[ToolInterface]):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.reasoning_history: List[ReActStep] = []
        self.max_steps = 10
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Solve a complex problem using ReAct methodology"""
        # Implementation exercise for students
        pass
    
    def _generate_thought(self, problem: str, context: str) -> Thought:
        """Generate reasoning thought"""
        # Implementation exercise for students
        pass
    
    def _select_action(self, thought: Thought) -> Action:
        """Select appropriate action based on thought"""
        # Implementation exercise for students
        pass

# Exercise: Implement the missing methods
```

### Exercise 2: Implementing Self-Reflection

Build a reflection system that can improve agent performance over time.

```python
class SelfImprovingAgent:
    """Agent that learns from its mistakes through reflection"""
    
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.performance_history = []
        self.reflection_insights = []
    
    def reflect_on_performance(self, task_result: Dict[str, Any]):
        """Analyze performance and generate insights"""
        # Implementation exercise for students
        pass
    
    def apply_insights(self, new_task: str) -> str:
        """Apply learned insights to new tasks"""
        # Implementation exercise for students
        pass

# Exercise: Implement reflection mechanisms
```

### Exercise 3: Advanced Planning System

Create a planning system that can handle dynamic environments and re-planning.

```python
class AdaptivePlanner(HierarchicalPlanner):
    """Planner that can adapt to changing conditions"""
    
    def __init__(self):
        super().__init__()
        self.environment_state = {}
        self.contingency_plans = {}
    
    def monitor_environment(self) -> Dict[str, Any]:
        """Monitor environment for changes"""
        # Implementation exercise for students
        pass
    
    def replan_if_needed(self) -> bool:
        """Check if replanning is needed and execute if so"""
        # Implementation exercise for students
        pass
    
    def create_contingency_plan(self, scenario: str, alternative_tasks: List[Task]):
        """Create contingency plans for different scenarios"""
        # Implementation exercise for students
        pass

# Exercise: Implement adaptive planning capabilities
```

## Module Summary

In this module, we explored advanced agentic design patterns that form the foundation of sophisticated AI systems:

### Key Concepts Learned

1. **ReAct Framework**: The Reasoning and Acting paradigm that enables agents to think through problems step-by-step while taking actions based on their reasoning.

2. **Reflection Patterns**: Self-improvement mechanisms that allow agents to analyze their performance, learn from mistakes, and adapt their behavior over time.

3. **CodeAct Pattern**: A powerful approach where agents solve problems by writing and executing code, providing precise control and verifiable results.

4. **Hierarchical Planning**: Breaking down complex goals into manageable subtasks with proper dependency management and execution coordination.

### Practical Skills Developed

- Implementing reasoning loops with thought-action-observation cycles
- Building secure code execution environments with proper sandboxing
- Creating self-reflective systems that improve over time
- Designing hierarchical task decomposition and execution systems
- Managing dependencies and constraints in complex planning scenarios

### Real-world Applications

- **Research Assistants**: Agents that can reason through complex research questions and gather information systematically
- **Code Generation Systems**: AI that can write, test, and debug code to solve programming challenges
- **Project Management**: Automated systems that can break down projects into tasks and manage execution
- **Problem-Solving Agents**: AI systems that can tackle multi-step problems in domains like mathematics, science, and engineering

### Next Steps

These design patterns provide the building blocks for creating sophisticated agentic AI applications. In the next modules, we'll explore how to:

- Combine multiple agents in collaborative systems
- Implement observability and monitoring for production deployments
- Handle interoperability between different agent frameworks
- Scale agentic systems for enterprise applications

The patterns learned here will serve as the foundation for building production-grade agentic AI systems that can handle real-world complexity and requirements.