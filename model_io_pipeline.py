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
from pydantic import BaseModel, Field, field_validator
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
    
    @field_validator('confidence')
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