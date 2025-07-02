# Module I: Building Context-Aware LLM Applications and MCP

## 📚 Learning Objectives

By the end of this module, you will:
- ✅ Master the Model I/O pipeline and its components
- ✅ Implement sophisticated memory systems for context retention
- ✅ Build intelligent retrieval chains using advanced loaders and retrievers
- ✅ Integrate Model Context Protocol (MCP) into production applications
- ✅ Create hands-on projects demonstrating real-world context-aware capabilities
- ✅ Deploy scalable, production-ready context-aware applications

---

## 🛠️ Prerequisites and Complete Setup Guide

### System Requirements
- **Python**: 3.9 or higher (3.11 recommended)
- **RAM**: 8GB minimum (16GB recommended for large models)
- **Storage**: 15GB free disk space
- **Internet**: Stable connection for API access and model downloads
- **OS**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+

### 🚀 Step-by-Step Environment Setup

#### Step 1: Create and Activate Virtual Environment
```bash
# Create a dedicated virtual environment
python -m venv context_aware_llm_env

# Activate the environment
# On macOS/Linux:
source context_aware_llm_env/bin/activate
# On Windows:
context_aware_llm_env\Scripts\activate

# Verify activation (should show your env path)
which python
```

#### Step 2: Install Core Dependencies
```bash
# Update pip first
pip install --upgrade pip

# Core LangChain ecosystem
pip install langchain==0.1.0
pip install langchain-openai==0.0.5
pip install langchain-community==0.0.10
pip install langchain-experimental==0.0.50

# Vector databases and embeddings
pip install chromadb==0.4.22
pip install sentence-transformers==2.2.2
pip install faiss-cpu==1.7.4

# Document processing
pip install pypdf==3.17.4
pip install python-docx==0.8.11
pip install beautifulsoup4==4.12.2
pip install unstructured==0.11.8

# Utilities and APIs
pip install tiktoken==0.5.2
pip install requests==2.31.0
pip install python-dotenv==1.0.0
pip install pydantic==2.5.2

# Visualization and UI
pip install streamlit==1.29.0
pip install plotly==5.17.0
pip install matplotlib==3.8.2

# Data processing
pip install pandas==2.1.4
pip install numpy==1.24.3

# Optional: Advanced features
pip install redis==5.0.1
pip install openai==1.6.1
pip install anthropic==0.8.1
```

#### Step 3: Environment Configuration
Create a `.env` file in your project root:
```bash
# API Keys (get these from respective providers)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Optional: Vector Database Services
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env_here

# Optional: Other Services
REDIS_URL=redis://localhost:6379
MONGODB_URI=mongodb://localhost:27017

# Application Settings
LOG_LEVEL=INFO
MAX_TOKENS=4000
TEMPERATURE=0.1
```

#### Step 4: Verify Installation
```python
# test_setup.py - Run this to verify your setup
import sys
import os
from datetime import datetime

def test_installation():
    """Comprehensive installation test"""
    print("🔍 Testing Installation...\n")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Test time: {datetime.now()}\n")
    
    # Test core imports
    tests = [
        ("langchain", "LangChain"),
        ("langchain_openai", "LangChain OpenAI"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("tiktoken", "TikToken"),
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
        ("dotenv", "Python Dotenv")
    ]
    
    results = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"✅ {name}: Installed")
            results.append(True)
        except ImportError as e:
            print(f"❌ {name}: Not installed - {e}")
            results.append(False)
    
    # Test environment variables
    print("\n🔑 Environment Variables:")
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_API_TOKEN"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set (length: {len(value)})")
        else:
            print(f"⚠️ {var}: Not set")
    
    # Summary
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Installation Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Setup verification complete! You're ready to proceed.")
        return True
    else:
        print("⚠️ Some components failed. Please check the installation.")
        return False

if __name__ == "__main__":
    test_installation()
```

#### Step 5: Create Project Structure
```bash
# Create organized project structure
mkdir -p context_aware_llm/{
data/{raw,processed,embeddings},
src/{models,utils,agents},
notebooks,
tests,
configs,
logs,
outputs
}

# Create essential files
touch context_aware_llm/{__init__.py,main.py,requirements.txt}
touch context_aware_llm/src/{__init__.py,models/__init__.py,utils/__init__.py,agents/__init__.py}
```

---

## 1. Understanding Model I/O: The Foundation of LLM Applications

### The Model I/O Pipeline: Your AI Application's Nervous System

Imagine the Model I/O pipeline as the nervous system of your AI application. Just as your brain processes sensory input, reasons about it, and produces actions, the Model I/O pipeline transforms raw user input into intelligent, structured responses.

```svg
<svg width="1000" height="400" xmlns="http://www.w3.org/2000/svg">
  <!-- Background with gradient -->
  <defs>
    <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f8f9fa;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#e9ecef;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="3" dy="3" stdDeviation="3" flood-color="#00000020"/>
    </filter>
  </defs>
  
  <rect width="1000" height="400" fill="url(#bgGradient)" stroke="#dee2e6" stroke-width="2" rx="15"/>
  
  <!-- Title -->
  <text x="500" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#212529">🧠 Model I/O Pipeline: From Raw Input to Intelligent Output</text>
  
  <!-- Input Stage -->
  <g filter="url(#shadow)">
    <rect x="60" y="80" width="180" height="120" fill="#e3f2fd" stroke="#1976d2" stroke-width="3" rx="15"/>
    <text x="150" y="105" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#1976d2">📥 INPUT LAYER</text>
    <text x="150" y="125" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#1976d2">• Raw user queries</text>
    <text x="150" y="142" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#1976d2">• Context data</text>
    <text x="150" y="159" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#1976d2">• System state</text>
    <text x="150" y="176" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#1976d2">• User preferences</text>
    <text x="150" y="193" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#1976d2">• Historical context</text>
  </g>
  
  <!-- Prompt Engineering -->
  <g filter="url(#shadow)">
    <rect x="280" y="80" width="180" height="120" fill="#fff3e0" stroke="#f57c00" stroke-width="3" rx="15"/>
    <text x="370" y="105" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#f57c00">🔧 PROMPT ENGINE</text>
    <text x="370" y="125" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#f57c00">• Template design</text>
    <text x="370" y="142" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#f57c00">• Context injection</text>
    <text x="370" y="159" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#f57c00">• Format specification</text>
    <text x="370" y="176" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#f57c00">• Few-shot examples</text>
    <text x="370" y="193" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#f57c00">• Chain-of-thought</text>
  </g>
  
  <!-- LLM Processing -->
  <g filter="url(#shadow)">
    <rect x="500" y="80" width="180" height="120" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="3" rx="15"/>
    <text x="590" y="105" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#7b1fa2">🧠 LLM CORE</text>
    <text x="590" y="125" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#7b1fa2">• Token generation</text>
    <text x="590" y="142" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#7b1fa2">• Context reasoning</text>
    <text x="590" y="159" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#7b1fa2">• Pattern matching</text>
    <text x="590" y="176" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#7b1fa2">• Knowledge retrieval</text>
    <text x="590" y="193" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#7b1fa2">• Inference</text>
  </g>
  
  <!-- Output Parsing -->
  <g filter="url(#shadow)">
    <rect x="720" y="80" width="180" height="120" fill="#e8f5e8" stroke="#388e3c" stroke-width="3" rx="15"/>
    <text x="810" y="105" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#388e3c">📤 OUTPUT PARSER</text>
    <text x="810" y="125" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#388e3c">• Structure extraction</text>
    <text x="810" y="142" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#388e3c">• Validation</text>
    <text x="810" y="159" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#388e3c">• Format conversion</text>
    <text x="810" y="176" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#388e3c">• Error handling</text>
    <text x="810" y="193" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#388e3c">• Type safety</text>
  </g>
  
  <!-- Flow Arrows -->
  <defs>
    <marker id="arrowhead" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto">
      <polygon points="0 0, 12 4, 0 8" fill="#495057"/>
    </marker>
    <marker id="arrowhead-red" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto">
      <polygon points="0 0, 12 4, 0 8" fill="#dc3545"/>
    </marker>
  </defs>
  
  <path d="M 240 140 L 270 140" stroke="#495057" stroke-width="4" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 460 140 L 490 140" stroke="#495057" stroke-width="4" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 680 140 L 710 140" stroke="#495057" stroke-width="4" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Process Labels -->
  <text x="255" y="235" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#6c757d">Structure &amp; Format</text>
  <text x="475" y="235" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#6c757d">Generate &amp; Reason</text>
  <text x="695" y="235" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#6c757d">Extract &amp; Validate</text>
  
  <!-- Feedback Loop -->
  <path d="M 810 220 Q 500 320 150 220" stroke="#dc3545" stroke-width="4" fill="none" stroke-dasharray="10,5" marker-end="url(#arrowhead-red)"/>
  <text x="500" y="350" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#dc3545">🔄 Continuous Context Feedback Loop</text>
  
  <!-- Performance Indicators -->
  <circle cx="150" cy="260" r="8" fill="#28a745"/>
  <text x="150" y="280" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#28a745">Fast</text>
  
  <circle cx="370" cy="260" r="8" fill="#ffc107"/>
  <text x="370" y="280" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#ffc107">Critical</text>
  
  <circle cx="590" cy="260" r="8" fill="#dc3545"/>
  <text x="590" y="280" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#dc3545">Intensive</text>
  
  <circle cx="810" cy="260" r="8" fill="#17a2b8"/>
  <text x="810" y="280" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#17a2b8">Reliable</text>
</svg>
```

### 🎯 Hands-On Exercise 1: Building Your First Production-Ready Model I/O Pipeline

Let's build a sophisticated, real-world Model I/O pipeline that you can actually use in production:

```python
# model_io_pipeline.py
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

# Core imports
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator
import tiktoken

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_io_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class TaskType(Enum):
    """Types of tasks the system can handle"""
    ANALYSIS = "analysis"
    CREATION = "creation"
    PROBLEM_SOLVING = "problem_solving"
    EXPLANATION = "explanation"
    PLANNING = "planning"
    CODING = "coding"
    RESEARCH = "research"

class TaskAnalysis(BaseModel):
    """Structured output for comprehensive task analysis"""
    task_type: TaskType = Field(description="Primary type of the task")
    complexity: TaskComplexity = Field(description="Complexity level assessment")
    estimated_time: str = Field(description="Realistic completion time estimate")
    key_steps: List[str] = Field(description="Detailed step-by-step breakdown")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)
    recommendations: List[str] = Field(description="Actionable recommendations")
    prerequisites: List[str] = Field(description="Required knowledge or tools")
    potential_challenges: List[str] = Field(description="Anticipated difficulties")
    success_metrics: List[str] = Field(description="How to measure success")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 2)

@dataclass
class UserContext:
    """Comprehensive user context information"""
    user_id: str
    session_id: str
    preferences: Dict[str, Any]
    history: List[str]
    timestamp: datetime
    expertise_level: str = "intermediate"
    communication_style: str = "detailed"
    goals: List[str] = None
    constraints: List[str] = None
    
    def __post_init__(self):
        if self.goals is None:
            self.goals = []
        if self.constraints is None:
            self.constraints = []

class PerformanceMetrics:
    """Track pipeline performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.token_count = 0
        self.processing_stages = {}
        self.errors = []
    
    def start_timing(self, stage: str):
        if self.start_time is None:
            self.start_time = time.time()
        self.processing_stages[stage] = {'start': time.time()}
    
    def end_timing(self, stage: str):
        if stage in self.processing_stages:
            self.processing_stages[stage]['end'] = time.time()
            self.processing_stages[stage]['duration'] = (
                self.processing_stages[stage]['end'] - 
                self.processing_stages[stage]['start']
            )
    
    def add_error(self, error: str):
        self.errors.append({
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_total_duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_duration': self.get_total_duration(),
            'token_count': self.token_count,
            'stages': self.processing_stages,
            'error_count': len(self.errors),
            'errors': self.errors
        }

class AdvancedModelIOPipeline:
    """Production-ready Model I/O pipeline with comprehensive features"""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 4000,
                 enable_caching: bool = True):
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize components
        self.parser = PydanticOutputParser(pydantic_object=TaskAnalysis)
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.metrics = PerformanceMetrics()
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        
        # Setup prompt template
        self.setup_advanced_prompt_template()
        
        logger.info(f"Initialized AdvancedModelIOPipeline with model: {model_name}")
    
    def setup_advanced_prompt_template(self):
        """Create a sophisticated, context-aware prompt template"""
        template = """
You are an expert AI assistant specializing in comprehensive task analysis and strategic planning.

USER PROFILE:
- User ID: {user_id}
- Expertise Level: {expertise_level}
- Communication Style: {communication_style}
- Session: {session_id}

CONTEXT INFORMATION:
- User Preferences: {preferences}
- Goals: {goals}
- Constraints: {constraints}
- Recent History: {history}
- Current Timestamp: {timestamp}

TASK REQUEST:
{user_request}

ANALYSIS FRAMEWORK:
1. **Task Classification**: Identify the primary task type and complexity level
2. **Decomposition**: Break down into manageable, sequential steps
3. **Resource Assessment**: Identify required knowledge, tools, and prerequisites
4. **Risk Analysis**: Anticipate potential challenges and mitigation strategies
5. **Success Planning**: Define clear metrics and milestones
6. **Personalization**: Tailor recommendations to user's expertise and preferences

IMPORTANT GUIDELINES:
- Consider the user's expertise level when providing recommendations
- Align suggestions with stated goals and respect constraints
- Provide realistic time estimates based on complexity
- Include specific, actionable steps rather than vague advice
- Anticipate common pitfalls and provide preventive guidance

{format_instructions}

Provide a thorough, personalized analysis that empowers the user to succeed:
"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=[
                "user_id", "expertise_level", "communication_style", "session_id",
                "preferences", "goals", "constraints", "history", "timestamp", "user_request"
            ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def process_request(self, user_request: str, context: UserContext) -> TaskAnalysis:
        """Process user request through the complete I/O pipeline"""
        
        self.metrics.reset()
        
        try:
            # Step 1: Input Processing
            self.metrics.start_timing("input_processing")
            processed_input = self._process_input(user_request, context)
            self.metrics.end_timing("input_processing")
            
            # Step 2: Check cache
            cache_key = self._generate_cache_key(user_request, context)
            if self.enable_caching and cache_key in self.cache:
                logger.info("Cache hit - returning cached result")
                return self.cache[cache_key]
            
            # Step 3: Prompt Engineering
            self.metrics.start_timing("prompt_engineering")
            formatted_prompt = self._engineer_prompt(processed_input, context)
            self.metrics.end_timing("prompt_engineering")
            
            # Step 4: Token counting and validation
            token_count = len(self.tokenizer.encode(formatted_prompt))
            self.metrics.token_count = token_count
            
            if token_count > 3500:  # Leave room for response
                logger.warning(f"High token count: {token_count}")
            
            # Step 5: LLM Processing
            self.metrics.start_timing("llm_processing")
            raw_response = self._llm_process(formatted_prompt)
            self.metrics.end_timing("llm_processing")
            
            # Step 6: Output Parsing
            self.metrics.start_timing("output_parsing")
            structured_output = self._parse_output(raw_response)
            self.metrics.end_timing("output_parsing")
            
            # Step 7: Post-processing and validation
            self.metrics.start_timing("post_processing")
            validated_output = self._post_process(structured_output, context)
            self.metrics.end_timing("post_processing")
            
            # Cache the result
            if self.enable_caching:
                self.cache[cache_key] = validated_output
            
            self.metrics.end_time = time.time()
            logger.info(f"Pipeline completed successfully in {self.metrics.get_total_duration():.2f}s")
            
            return validated_output
            
        except Exception as e:
            self.metrics.add_error(str(e))
            logger.error(f"Pipeline error: {e}")
            return self._create_fallback_response(user_request, str(e))
    
    def _process_input(self, request: str, context: UserContext) -> Dict[str, Any]:
        """Advanced input processing with context analysis"""
        
        # Clean and normalize input
        cleaned_request = request.strip()
        
        # Extract metadata
        word_count = len(cleaned_request.split())
        char_count = len(cleaned_request)
        
        # Analyze complexity indicators
        complexity_indicators = {
            'question_marks': cleaned_request.count('?'),
            'technical_terms': self._count_technical_terms(cleaned_request),
            'length_complexity': 'high' if word_count > 100 else 'medium' if word_count > 30 else 'low'
        }
        
        # Calculate context relevance
        context_relevance = self._calculate_context_relevance(cleaned_request, context)
        
        return {
            "cleaned_request": cleaned_request,
            "metadata": {
                "word_count": word_count,
                "char_count": char_count,
                "complexity_indicators": complexity_indicators,
                "context_relevance": context_relevance,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _engineer_prompt(self, processed_input: Dict[str, Any], context: UserContext) -> str:
        """Create the final prompt with comprehensive context"""
        
        return self.prompt.format(
            user_id=context.user_id,
            expertise_level=context.expertise_level,
            communication_style=context.communication_style,
            session_id=context.session_id,
            preferences=json.dumps(context.preferences, indent=2),
            goals="\n".join(f"- {goal}" for goal in context.goals),
            constraints="\n".join(f"- {constraint}" for constraint in context.constraints),
            history="\n".join(f"- {item}" for item in context.history[-5:]),  # Last 5 interactions
            timestamp=context.timestamp.isoformat(),
            user_request=processed_input["cleaned_request"]
        )
    
    def _llm_process(self, prompt: str) -> str:
        """Send prompt to LLM with error handling and retries"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                logger.warning(f"LLM processing attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _parse_output(self, raw_response: str) -> TaskAnalysis:
        """Parse LLM response with robust error handling"""
        
        try:
            return self.parser.parse(raw_response)
        except Exception as e:
            logger.warning(f"Primary parsing failed: {e}")
            
            # Attempt fallback parsing
            try:
                return self._fallback_parse(raw_response)
            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed: {fallback_error}")
                raise
    
    def _post_process(self, output: TaskAnalysis, context: UserContext) -> TaskAnalysis:
        """Post-process and validate the output"""
        
        # Adjust recommendations based on user expertise
        if context.expertise_level == "beginner":
            output.recommendations = [f"[Beginner-friendly] {rec}" for rec in output.recommendations]
        elif context.expertise_level == "expert":
            output.recommendations = [f"[Advanced] {rec}" for rec in output.recommendations]
        
        # Ensure minimum quality standards
        if len(output.key_steps) < 3:
            output.key_steps.append("Review and refine your approach based on initial results")
        
        if len(output.recommendations) < 2:
            output.recommendations.append("Consider seeking additional resources or expert guidance")
        
        return output
    
    def _count_technical_terms(self, text: str) -> int:
        """Count technical terms in the text"""
        technical_terms = [
            'api', 'database', 'algorithm', 'framework', 'library', 'function',
            'class', 'method', 'variable', 'parameter', 'authentication', 'authorization',
            'deployment', 'scaling', 'optimization', 'performance', 'security'
        ]
        
        text_lower = text.lower()
        return sum(1 for term in technical_terms if term in text_lower)
    
    def _calculate_context_relevance(self, request: str, context: UserContext) -> float:
        """Calculate how relevant the request is to user context"""
        
        relevance_score = 0.5  # Base score
        request_lower = request.lower()
        
        # Check against user preferences
        for pref_key, pref_value in context.preferences.items():
            if str(pref_value).lower() in request_lower:
                relevance_score += 0.1
        
        # Check against goals
        for goal in context.goals:
            common_words = set(request_lower.split()) & set(goal.lower().split())
            if len(common_words) > 1:
                relevance_score += 0.15
        
        # Check against history
        for hist_item in context.history:
            common_words = set(request_lower.split()) & set(hist_item.lower().split())
            if len(common_words) > 2:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _generate_cache_key(self, request: str, context: UserContext) -> str:
        """Generate a cache key for the request"""
        import hashlib
        
        cache_data = {
            'request': request,
            'user_id': context.user_id,
            'expertise_level': context.expertise_level,
            'preferences': context.preferences
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _fallback_parse(self, raw_response: str) -> TaskAnalysis:
        """Fallback parsing when primary parsing fails"""
        
        # Simple extraction logic
        lines = raw_response.split('\n')
        
        return TaskAnalysis(
            task_type=TaskType.ANALYSIS,
            complexity=TaskComplexity.MODERATE,
            estimated_time="Unable to estimate precisely",
            key_steps=["Review the request", "Gather necessary information", "Implement solution"],
            confidence=0.3,
            recommendations=["Please provide more specific details", "Consider breaking down the task"],
            prerequisites=["Basic understanding of the domain"],
            potential_challenges=["Unclear requirements"],
            success_metrics=["Task completion", "User satisfaction"]
        )
    
    def _create_fallback_response(self, request: str, error: str) -> TaskAnalysis:
        """Create a fallback response when pipeline fails"""
        
        return TaskAnalysis(
            task_type=TaskType.ANALYSIS,
            complexity=TaskComplexity.MODERATE,
            estimated_time="Unable to estimate due to processing error",
            key_steps=[
                "Review the original request",
                "Clarify requirements",
                "Retry with more specific information"
            ],
            confidence=0.2,
            recommendations=[
                "Please rephrase your request more clearly",
                "Provide additional context if possible",
                f"Technical note: {error}"
            ],
            prerequisites=["Clear problem statement"],
            potential_challenges=["Processing limitations", "Unclear requirements"],
            success_metrics=["Successful processing", "Clear output"]
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return self.metrics.get_summary()
    
    def clear_cache(self):
        """Clear the response cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")

# 🧪 Comprehensive Test Suite
def test_advanced_pipeline():
    """Comprehensive test of the advanced Model I/O pipeline"""
    
    print("🚀 Testing Advanced Model I/O Pipeline\n")
    print("=" * 70)
    
    # Initialize pipeline
    try:
        pipeline = AdvancedModelIOPipeline(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            enable_caching=True
        )
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Create comprehensive user context
    user_context = UserContext(
        user_id="test_user_123",
        session_id="session_456",
        expertise_level="intermediate",
        communication_style="detailed",
        preferences={
            "programming_languages": ["Python", "JavaScript"],
            "learning_style": "hands-on",
            "time_availability": "2-3 hours daily",
            "project_focus": "web development"
        },
        goals=[
            "Build a full-stack web application",
            "Learn modern development practices",
            "Deploy to production"
        ],
        constraints=[
            "Limited to free tools and services",
            "Must be completed within 2 weeks",
            "No prior experience with deployment"
        ],
        history=[
            "How do I set up a Python development environment?",
            "What's the difference between frontend and backend?",
            "Can you explain REST APIs?",
            "How do I handle user authentication?",
            "What are the best practices for database design?"
        ],
        timestamp=datetime.now()
    )
    
    # Test cases with varying complexity
    test_cases = [
        {
            "name": "Simple Query",
            "request": "How do I create a Python virtual environment?",
            "expected_complexity": TaskComplexity.SIMPLE
        },
        {
            "name": "Moderate Project",
            "request": "I want to build a web application that allows users to upload and share photos with comments. What's the best approach?",
            "expected_complexity": TaskComplexity.MODERATE
        },
        {
            "name": "Complex System",
            "request": "Design a scalable microservices architecture for an e-commerce platform that can handle 100,000 concurrent users, with real-time inventory management, payment processing, and recommendation engine.",
            "expected_complexity": TaskComplexity.COMPLEX
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            # Process the request
            start_time = time.time()
            result = pipeline.process_request(test_case["request"], user_context)
            end_time = time.time()
            
            # Display results
            print(f"✅ Processing completed in {end_time - start_time:.2f}s")
            print(f"📊 Task Type: {result.task_type.value}")
            print(f"📊 Complexity: {result.complexity.value}")
            print(f"📊 Estimated Time: {result.estimated_time}")
            print(f"📊 Confidence: {result.confidence:.2f}")
            
            print(f"\n🔧 Key Steps ({len(result.key_steps)}):")
            for j, step in enumerate(result.key_steps[:3], 1):  # Show first 3
                print(f"  {j}. {step}")
            if len(result.key_steps) > 3:
                print(f"  ... and {len(result.key_steps) - 3} more steps")
            
            print(f"\n💡 Recommendations ({len(result.recommendations)}):")
            for j, rec in enumerate(result.recommendations[:2], 1):  # Show first 2
                print(f"  {j}. {rec}")
            
            print(f"\n⚠️ Potential Challenges ({len(result.potential_challenges)}):")
            for j, challenge in enumerate(result.potential_challenges[:2], 1):
                print(f"  {j}. {challenge}")
            
            # Performance metrics
            metrics = pipeline.get_performance_metrics()
            print(f"\n📈 Performance Metrics:")
            print(f"  • Total Duration: {metrics['total_duration']:.2f}s")
            print(f"  • Token Count: {metrics['token_count']}")
            print(f"  • Errors: {metrics['error_count']}")
            
            results.append({
                'test_case': test_case['name'],
                'success': True,
                'duration': end_time - start_time,
                'complexity': result.complexity,
                'confidence': result.confidence
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                'test_case': test_case['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
    print(f"📈 Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        avg_duration = sum(r['duration'] for r in results if r['success']) / successful_tests
        avg_confidence = sum(r['confidence'] for r in results if r['success']) / successful_tests
        print(f"⏱️ Average Duration: {avg_duration:.2f}s")
        print(f"🎯 Average Confidence: {avg_confidence:.2f}")
    
    # Cache test
    print(f"\n🧪 Testing Cache Performance...")
    cache_start = time.time()
    cached_result = pipeline.process_request(test_cases[0]["request"], user_context)
    cache_end = time.time()
    print(f"✅ Cache retrieval: {cache_end - cache_start:.2f}s")
    
    print(f"\n🎉 Advanced Pipeline testing completed!")
    return successful_tests == total_tests

if __name__ == "__main__":
    # Run the test
    success = test_advanced_pipeline()
    
    if success:
        print("\n🌟 All tests passed! Your Model I/O pipeline is ready for production.")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration and try again.")
```

### 🎨 Advanced Prompt Engineering: The Art of AI Communication

Prompt engineering is like being a translator between human intent and AI understanding. Let's master the most effective techniques:

#### 1. The CLEAR Framework: Your Prompt Engineering Blueprint

```python
# advanced_prompting.py
from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
from enum import Enum
import json

class PromptType(Enum):
    """Different types of prompts for various use cases"""
    ANALYSIS = "analysis"
    CREATION = "creation"
    EXPLANATION = "explanation"
    PROBLEM_SOLVING = "problem_solving"
    CODE_REVIEW = "code_review"
    PLANNING = "planning"

class CLEARPromptFramework:
    """
    CLEAR Framework for Systematic Prompt Engineering:
    C - Context (comprehensive background information)
    L - Length (appropriate response length and detail level)
    E - Examples (relevant few-shot examples)
    A - Audience (target audience and expertise level)
    R - Role (AI's role, persona, and capabilities)
    """
    
    @staticmethod
    def create_clear_prompt(
        context: str,
        length_instruction: str,
        examples: str,
        audience: str,
        role: str,
        task: str,
        additional_constraints: List[str] = None
    ) -> PromptTemplate:
        
        constraints_section = ""
        if additional_constraints:
            constraints_section = f"""
        CONSTRAINTS:
        {chr(10).join(f"- {constraint}" for constraint in additional_constraints)}
        """
        
        template = f"""
        ROLE & EXPERTISE: {role}
        
        CONTEXT & BACKGROUND:
        {context}
        
        TARGET AUDIENCE: {audience}
        
        TASK DESCRIPTION:
        {task}
        
        EXAMPLES FOR REFERENCE:
        {examples}
        
        RESPONSE REQUIREMENTS:
        {length_instruction}
        {constraints_section}
        
        USER INPUT: {{user_input}}
        
        RESPONSE:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["user_input"]
        )
    
    @staticmethod
    def create_specialized_prompts() -> Dict[str, PromptTemplate]:
        """Create a library of specialized prompts for common use cases"""
        
        prompts = {}
        
        # Code Review Prompt
        prompts["code_review"] = CLEARPromptFramework.create_clear_prompt(
            context="""
            You are reviewing code for a production web application. Focus on:
            - Security vulnerabilities and best practices
            - Performance optimization opportunities
            - Code maintainability and readability
            - Error handling and edge cases
            - Adherence to coding standards
            """,
            length_instruction="""
            Provide a structured review with:
            - Overall assessment (1-2 sentences)
            - 3-5 specific issues with severity levels
            - 2-3 improvement suggestions
            - Security considerations (if applicable)
            """,
            examples="""
            Example 1 - SQL Injection Issue:
            Code: `query = f"SELECT * FROM users WHERE id = {user_id}"`
            Issue: 🔴 CRITICAL - SQL injection vulnerability
            Fix: Use parameterized queries: `cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))`
            
            Example 2 - Performance Issue:
            Code: `for item in large_list: if expensive_operation(item): results.append(item)`
            Issue: 🟡 MODERATE - Inefficient loop with expensive operations
            Fix: Consider using list comprehension with filtering or vectorized operations
            """,
            audience="Intermediate to senior developers",
            role="Senior Software Engineer and Security Expert with 10+ years experience",
            task="Review the provided code and provide actionable feedback",
            additional_constraints=[
                "Use emoji indicators for severity: 🔴 Critical, 🟡 Moderate, 🟢 Minor",
                "Provide specific code examples for fixes",
                "Consider both immediate fixes and long-term architectural improvements"
            ]
        )
        
        # Architecture Design Prompt
        prompts["architecture_design"] = CLEARPromptFramework.create_clear_prompt(
            context="""
            You are designing software architecture for scalable, maintainable applications.
            Consider: scalability, security, maintainability, performance, cost-effectiveness,
            team expertise, deployment complexity, and future growth requirements.
            """,
            length_instruction="""
            Provide a comprehensive architecture design including:
            - High-level architecture overview
            - Component breakdown with responsibilities
            - Technology stack recommendations
            - Scalability considerations
            - Security measures
            - Deployment strategy
            - Estimated timeline and resources
            """,
            examples="""
            Example Request: "E-commerce platform for 10,000 daily users"
            Response Structure:
            🏗️ ARCHITECTURE OVERVIEW
            - Microservices architecture with API Gateway
            - Event-driven communication between services
            - Containerized deployment with Kubernetes
            
            📦 CORE COMPONENTS
            - User Service (authentication, profiles)
            - Product Service (catalog, inventory)
            - Order Service (cart, checkout, fulfillment)
            - Payment Service (processing, refunds)
            
            🛠️ TECHNOLOGY STACK
            - Backend: Node.js/Express or Python/FastAPI
            - Database: PostgreSQL + Redis for caching
            - Frontend: React/Next.js
            - Infrastructure: AWS/GCP with Docker + Kubernetes
            """,
            audience="Technical leads, senior developers, and stakeholders",
            role="Principal Software Architect with expertise in distributed systems",
            task="Design a comprehensive software architecture based on requirements"
        )
        
        # Learning Path Prompt
        prompts["learning_path"] = CLEARPromptFramework.create_clear_prompt(
            context="""
            You are creating personalized learning paths for software development.
            Consider: current skill level, learning style, time availability, career goals,
            practical project opportunities, and industry trends.
            """,
            length_instruction="""
            Create a detailed learning path with:
            - Skill assessment and gap analysis
            - Phase-by-phase learning plan (3-6 months)
            - Specific resources and projects for each phase
            - Milestones and success metrics
            - Time estimates and scheduling recommendations
            """,
            examples="""
            Example: "Junior developer wanting to learn full-stack development"
            
            📊 CURRENT SKILLS ASSESSMENT
            - Strengths: Basic programming, problem-solving
            - Gaps: Web frameworks, databases, deployment
            
            🎯 LEARNING PATH (4 months)
            
            Phase 1 (Month 1): Frontend Foundations
            - HTML/CSS mastery (2 weeks)
            - JavaScript fundamentals (2 weeks)
            - Project: Personal portfolio website
            
            Phase 2 (Month 2): Backend Basics
            - Node.js/Express or Python/Flask (3 weeks)
            - Database fundamentals (1 week)
            - Project: REST API for todo app
            """,
            audience="Developers at various skill levels seeking career advancement",
            role="Senior Developer and Technical Mentor with teaching experience",
            task="Create a personalized learning roadmap based on goals and constraints"
        )
        
        return prompts

# Test the CLEAR framework
def test_clear_framework():
    """Test the CLEAR prompt framework with real examples"""
    
    print("🧪 Testing CLEAR Prompt Framework\n")
    print("=" * 60)
    
    # Get specialized prompts
    prompts = CLEARPromptFramework.create_specialized_prompts()
    
    # Test cases
    test_cases = [
        {
            "prompt_type": "code_review",
            "input": """
            def authenticate_user(username, password):
                query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
                result = db.execute(query)
                if result:
                    session['user_id'] = result[0]['id']
                    return True
                return False
            """,
            "description": "Authentication function with potential security issues"
        },
        {
            "prompt_type": "architecture_design",
            "input": """
            I need to build a social media platform that can handle:
            - 50,000 daily active users
            - Real-time messaging
            - Photo/video sharing
            - News feed with personalized content
            - Mobile app support
            Budget: $10,000/month for infrastructure
            Team: 5 developers (2 backend, 2 frontend, 1 mobile)
            Timeline: 6 months to MVP
            """,
            "description": "Social media platform architecture"
        },
        {
            "prompt_type": "learning_path",
            "input": """
            I'm a data analyst with strong SQL and Excel skills, but I want to transition
            to machine learning engineering. I have:
            - 3 years of data analysis experience
            - Basic Python knowledge
            - Statistics background
            - 10-15 hours per week for learning
            - Goal: ML Engineer role within 8 months
            """,
            "description": "Career transition to ML engineering"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['description']}")
        print("-" * 50)
        
        prompt_template = prompts[test_case["prompt_type"]]
        formatted_prompt = prompt_template.format(user_input=test_case["input"])
        
        print(f"📝 Prompt Type: {test_case['prompt_type']}")
        print(f"📏 Prompt Length: {len(formatted_prompt)} characters")
        print(f"🎯 Input Preview: {test_case['input'][:100]}...")
        
        # Show prompt structure (first 500 chars)
        print(f"\n📋 Prompt Structure Preview:")
        print(formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt)
        
        print(f"\n✅ Prompt generated successfully")
    
    print(f"\n🎉 CLEAR Framework testing completed!")
    print(f"📊 Generated {len(test_cases)} specialized prompts")
    
    return True

if __name__ == "__main__":
    test_clear_framework()
```

#### 2. Chain-of-Thought Prompting: Teaching AI to Think Step-by-Step

```python
# chain_of_thought.py
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any
import os
import re

class ChainOfThoughtPrompting:
    """Advanced Chain-of-Thought prompting for complex reasoning tasks"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def create_cot_prompt(self, problem_type: str) -> PromptTemplate:
        """Create Chain-of-Thought prompts for different problem types"""
        
        templates = {
            "math": """
            Solve this mathematical problem step by step, showing your reasoning at each stage.
            
            Problem: {problem}
            
            Let me work through this systematically:
            
            Step 1: Understand what we know
            [Identify given information and constraints]
            
            Step 2: Identify what we need to find
            [Clearly state the goal]
            
            Step 3: Choose the appropriate method
            [Select the mathematical approach or formula]
            
            Step 4: Apply the method step by step
            [Show detailed calculations]
            
            Step 5: Verify the answer
            [Check if the result makes sense]
            
            Final Answer: [Clear, concise answer]
            """,
            
            "programming": """
            Solve this programming problem using systematic thinking.
            
            Problem: {problem}
            
            Let me break this down step by step:
            
            Step 1: Understand the requirements
            [What exactly needs to be accomplished?]
            
            Step 2: Identify inputs and outputs
            [What data do we receive? What should we return?]
            
            Step 3: Consider edge cases
            [What unusual inputs might break our solution?]
            
            Step 4: Design the algorithm
            [High-level approach and data structures]
            
            Step 5: Implement the solution
            [Write clean, readable code]
            
            Step 6: Test and verify
            [Check with examples and edge cases]
            
            Solution:
            ```python
            [Your code here]
            ```
            
            Explanation: [Why this approach works]
            """,
            
            "analysis": """
            Analyze this situation using structured reasoning.
            
            Situation: {problem}
            
            Let me analyze this systematically:
            
            Step 1: Gather and organize information
            [What facts do we have? What's missing?]
            
            Step 2: Identify key factors and relationships
            [What elements influence the situation?]
            
            Step 3: Consider multiple perspectives
            [How might different stakeholders view this?]
            
            Step 4: Evaluate options and trade-offs
            [What are the possible approaches and their pros/cons?]
            
            Step 5: Draw conclusions and recommendations
            [Based on the analysis, what's the best path forward?]
            
            Analysis Summary: [Key insights and recommendations]
            """,
            
            "debugging": """
            Debug this code issue using systematic troubleshooting.
            
            Problem: {problem}
            
            Let me debug this step by step:
            
            Step 1: Reproduce the issue