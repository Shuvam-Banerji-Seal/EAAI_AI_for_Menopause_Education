# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Breaking the Silence: Technical Implementation of an AI-Powered Menopause Education System
# 
# Welcome to the technical implementation phase of the AI-Powered Menopause Education System. This notebook will guide you through building the core components of the system as outlined in the assignment. Our goal is to create a prototype that is not only functional but also embodies the principles of cultural sensitivity and ethical AI.
# 
# ### Project Goal
# 
# To design and develop a culturally-sensitive AI system for menopause education, addressing global health education gaps through personalized Large Language Model applications.
# 
# ### System Architecture Overview
# 
# Below is the high-level architecture we will be implementing. This notebook will focus on creating a simplified, working version of each component.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ```mermaid
# graph TB
#     A[User Input] --> B(Cultural Assessment Module);
#     B --> C{Determine Cultural Pathway};
#     C -- Liberation --> D[Liberation Content Pathway];
#     C -- Loss/Medical --> E[Loss/Medical Content Pathway];
#     C -- Neutral --> F[Neutral Content Pathway];
#     D --> G(Simple LLM Engine);
#     E --> G;
#     F --> G;
#     G --> H(Safety & Bias Validator);
#     H --> I[Final Response];
# ```

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Learning Objectives for this Notebook:
# 
# 1.  **Component Implementation**: Build Python classes for the core modules of our system.
# 2.  **Cultural Logic**: Implement the logic for cultural assessment and content routing.
# 3.  **Simulated AI**: Create a simple, rule-based "LLM" to simulate response generation without needing external APIs.
# 4.  **Safety First**: Integrate a simplified safety and bias checker.
# 5.  **End-to-End Pipeline**: Combine all components into a single, functional pipeline.
# 6.  **Testing with Personas**: Demonstrate the system's adaptability using predefined user personas.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 1: Environment Setup
# 
# First, let's import the necessary libraries. We'll use standard Python libraries to keep this notebook accessible.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:32:42.295133Z","iopub.execute_input":"2025-08-22T18:32:42.295956Z","iopub.status.idle":"2025-08-22T18:32:42.302241Z","shell.execute_reply.started":"2025-08-22T18:32:42.295881Z","shell.execute_reply":"2025-08-22T18:32:42.300975Z"},"jupyter":{"outputs_hidden":false}}
import json
import random
from textwrap import dedent

print("✅ Environment setup complete. Libraries are imported.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 2: Loading Phase 1 Artifacts (Simulated)
# 
# In a real project, the data from Phase 1 (Research and Design) would be loaded from external files (like CSVs or JSONs). For this educational notebook, we will define this data directly in our code. This includes our cultural assessment questions and user personas.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:33:13.325551Z","iopub.execute_input":"2025-08-22T18:33:13.325960Z","iopub.status.idle":"2025-08-22T18:33:13.334923Z","shell.execute_reply.started":"2025-08-22T18:33:13.325936Z","shell.execute_reply":"2025-08-22T18:33:13.333793Z"},"jupyter":{"outputs_hidden":false}}
# Simulated data from cultural_assessment_question_bank.html
cultural_assessment_questions = [
    {
        'id': 'q1',
        'question': "When you think about this next stage of life, what comes to mind first?",
        'options': {
            'a': {"text": "A sense of freedom and new possibilities.", "pathway": "liberation"},
            'b': {"text": "Concerns about health and physical changes.", "pathway": "medical"},
            'c': {"text": "A natural part of life with pros and cons.", "pathway": "neutral"}
        }
    },
    {
        'id': 'q2',
        'question': "How do you prefer to receive health information?",
        'options': {
            'a': {"text": "Empowering stories and traditional wisdom.", "pathway": "liberation"},
            'b': {"text": "Medical facts, data, and treatment options.", "pathway": "medical"},
            'c': {"text": "A balanced mix of personal stories and facts.", "pathway": "neutral"}
        }
    }
]

# Simulated User Personas from user_persona_template.md
user_personas = {
    "persona1": {
        "name": "Aditi",
        "description": "Rural Indian woman, 47, views menopause as a liberation from menstrual taboos.",
        "simulated_answers": {"q1": "a", "q2": "a"} # Answers lean towards 'liberation'
    },
    "persona2": {
        "name": "Priya",
        "description": "Urban professional woman, 50, anxious about symptoms and seeking medical solutions.",
        "simulated_answers": {"q1": "b", "q2": "b"} # Answers lean towards 'medical'
    },
    "persona3": {
        "name": "Susan",
        "description": "Western woman, 52, tech-savvy and looking for balanced, factual information.",
        "simulated_answers": {"q1": "c", "q2": "c"} # Answers lean towards 'neutral'
    }
}

print("✅ Phase 1 artifacts (Questions and Personas) loaded.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 3: Component 1 - Cultural Assessment Module
# 
# This module is responsible for determining the user's perspective on menopause. It asks questions and scores the answers to assign a cultural pathway. For this prototype, we will simulate the Q&A and focus on the scoring logic.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:33:36.381759Z","iopub.execute_input":"2025-08-22T18:33:36.382844Z","iopub.status.idle":"2025-08-22T18:33:36.394064Z","shell.execute_reply.started":"2025-08-22T18:33:36.382809Z","shell.execute_reply":"2025-08-22T18:33:36.392938Z"},"jupyter":{"outputs_hidden":false}}
class CulturalAssessmentModule:
    """Determines the user's cultural pathway based on their answers."""
    
    def __init__(self, questions):
        self.questions = questions
        print("CulturalAssessmentModule initialized.")

    def get_pathway(self, answers):
        """Calculates the dominant pathway from a dictionary of answers."""
        scores = {"liberation": 0, "medical": 0, "neutral": 0}
        
        for question_id, answer_key in answers.items():
            # Find the question corresponding to the ID
            question = next((q for q in self.questions if q['id'] == question_id), None)
            if question:
                pathway = question['options'][answer_key]['pathway']
                if pathway in scores:
                    scores[pathway] += 1
        
        # Determine the dominant pathway
        if not any(scores.values()):
            return "neutral" # Default pathway
            
        dominant_pathway = max(scores, key=scores.get)
        return dominant_pathway

# --- Demo of the Module ---
assessment_module = CulturalAssessmentModule(cultural_assessment_questions)

# Test with Persona 1 (Aditi)
persona1_answers = user_personas['persona1']['simulated_answers']
pathway1 = assessment_module.get_pathway(persona1_answers)
print(f"Persona 1 (Aditi) pathway: {pathway1}")

# Test with Persona 2 (Priya)
persona2_answers = user_personas['persona2']['simulated_answers']
pathway2 = assessment_module.get_pathway(persona2_answers)
print(f"Persona 2 (Priya) pathway: {pathway2}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 4: Component 2 - Content Pathway and Routing
# 
# This component defines the different types of content to be delivered based on the cultural pathway. The `ContentRouter` will select the appropriate content template that will be used to structure the AI's response.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:33:58.641150Z","iopub.execute_input":"2025-08-22T18:33:58.641441Z","iopub.status.idle":"2025-08-22T18:33:58.651167Z","shell.execute_reply.started":"2025-08-22T18:33:58.641421Z","shell.execute_reply":"2025-08-22T18:33:58.650003Z"},"jupyter":{"outputs_hidden":false}}
class ContentRouter:
    """Routes the user to the correct content pathway."""

    def __init__(self):
        self.pathways = {
            "liberation": self.liberation_pathway,
            "medical": self.medical_pathway,
            "neutral": self.neutral_pathway
        }
        print("ContentRouter initialized.")

    def liberation_pathway(self, query):
        return dedent(f"""
        Viewing menopause as a time of empowerment is a wonderful perspective. In response to your query about '{query}', let's explore the positive aspects and opportunities for growth during this natural life stage.
        --- (Simulated AI Content Generation) ---
        Many find new energy and focus when menstrual cycles cease. It's a great time to explore new hobbies or deepen spiritual practices. Your body is changing, and embracing this can lead to greater self-awareness and wisdom.
        """)

    def medical_pathway(self, query):
        return dedent(f"""
        It's completely understandable to focus on the medical aspects and symptoms of menopause. Regarding your question about '{query}', here is some information focused on symptom management and health.
        --- (Simulated AI Content Generation) ---
        Managing symptoms is key to maintaining quality of life. For many symptoms, lifestyle adjustments can be very effective. It is also important to discuss options like Hormone Therapy with a doctor to see if they are right for you.
        """)

    def neutral_pathway(self, query):
        return dedent(f"""
        Menopause is a multifaceted experience with both challenges and benefits. Regarding your question about '{query}', here is a balanced overview.
        --- (Simulated AI Content Generation) ---
        Menopause marks the end of fertility and brings hormonal shifts that can cause symptoms like hot flashes. However, it can also mean freedom from menstruation. Understanding both sides helps you navigate this transition effectively.
        """)

    def get_content(self, pathway, query):
        """Returns the content for the specified pathway."""
        return self.pathways.get(pathway, self.neutral_pathway)(query)

# --- Demo of the Module ---
content_router = ContentRouter()
test_query = "hot flashes"

print("--- Testing Content Router ---")
print("\n--- LIBERATION PATHWAY ---")
print(content_router.get_content("liberation", test_query))

print("\n--- MEDICAL PATHWAY ---")
print(content_router.get_content("medical", test_query))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 5: Component 3 - Simplified LLM Engine
# 
# In a full-scale application, this component would make a call to a Large Language Model (LLM) API like OpenAI or a locally hosted Hugging Face model. To keep this notebook simple and self-contained, we will create a **mock LLM**. This mock engine will take the content template from the `ContentRouter` and simply present it as the "generated" response. This allows us to focus on the system's architecture rather than the complexities of API calls.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:34:20.047416Z","iopub.execute_input":"2025-08-22T18:34:20.047817Z","iopub.status.idle":"2025-08-22T18:34:20.055405Z","shell.execute_reply.started":"2025-08-22T18:34:20.047787Z","shell.execute_reply":"2025-08-22T18:34:20.054104Z"},"jupyter":{"outputs_hidden":false}}
class SimpleLLMEngine:
    """A mock LLM that simulates generating a response."""
    def __init__(self):
        print("SimpleLLMEngine (Mock) initialized.")
        
    def generate_response(self, content_template):
        """'Generates' a response by simply returning the provided template."""
        print("\n[SimpleLLMEngine] 'Generating' response...")
        # In a real system, this is where you would call an LLM API.
        # For example: response = openai.Completion.create(...)
        return content_template

# --- Demo of the Module ---
llm_engine = SimpleLLMEngine()
template = content_router.get_content("neutral", "sleep problems")
generated_text = llm_engine.generate_response(template)

print("--- Testing Simple LLM Engine ---")
print(generated_text)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 6: Component 4 - Safety & Bias Validator
# 
# This is a critical component for any AI system, especially in healthcare. It checks the generated response for harmful content, biases, and medical inaccuracies. Our version will be a simplified, rule-based checker.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:34:46.013130Z","iopub.execute_input":"2025-08-22T18:34:46.013588Z","iopub.status.idle":"2025-08-22T18:34:46.023208Z","shell.execute_reply.started":"2025-08-22T18:34:46.013561Z","shell.execute_reply":"2025-08-22T18:34:46.021939Z"},"jupyter":{"outputs_hidden":false}}
class SafetyValidator:
    """A simplified validator to check for bias and unsafe language."""
    
    def __init__(self):
        # Keywords that might indicate bias or overly prescriptive advice
        self.western_centric_bias_words = ["always", "must", "never", "should"]
        # Words that can be negative in a medical context
        self.medical_bias_words = ["abnormal", "problem", "failure", "disorder"]
        print("SafetyValidator initialized.")

    def validate_response(self, response_text):
        """Validates the response and returns a safety report."""
        issues = []
        response_lower = response_text.lower()
        
        for word in self.western_centric_bias_words:
            if f' {word} ' in response_lower:
                issues.append(f"Potential Western-centric bias detected: '{word}'")

        for word in self.medical_bias_words:
            if f' {word} ' in response_lower:
                issues.append(f"Potential negative medical bias detected: '{word}'")
        
        # Medical disclaimer check (essential)
        if "not medical advice" not in response_lower and "consult a healthcare provider" not in response_lower:
             # In a real system, this would be a hard failure. Here we add it as a major issue.
            pass # For the final integrated system, we add this disclaimer at the end.

        is_safe = len(issues) == 0
        return {"is_safe": is_safe, "issues": issues}

# --- Demo of the Module ---
safety_validator = SafetyValidator()

safe_text = "Menopause is a natural transition. It can be a positive experience."
unsafe_text = "Menopause is a hormonal failure. You should always seek therapy."

print("--- Testing Safety Validator ---")
print(f"Validation for safe text: {safety_validator.validate_response(safe_text)}")
print(f"Validation for unsafe text: {safety_validator.validate_response(unsafe_text)}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 7: Integrating the System - The Full Pipeline
# 
# Now, we'll bring all the components together into a single, cohesive system. This class will orchestrate the entire process from receiving a user query to delivering a final, validated response.
# 
# ### Pipeline Flow
# ```mermaid
# sequenceDiagram
#     participant User
#     participant MenopauseAISystem
#     participant CulturalModule
#     participant ContentRouter
#     participant LLMEngine
#     participant SafetyValidator
# 
#     User->>MenopauseAISystem: handle_query(query, persona)
#     MenopauseAISystem->>CulturalModule: get_pathway(answers)
#     CulturalModule-->>MenopauseAISystem: pathway (e.g., 'liberation')
#     MenopauseAISystem->>ContentRouter: get_content(pathway, query)
#     ContentRouter-->>MenopauseAISystem: content_template
#     MenopauseAISystem->>LLMEngine: generate_response(template)
#     LLMEngine-->>MenopauseAISystem: generated_text
#     MenopauseAISystem->>SafetyValidator: validate_response(text)
#     SafetyValidator-->>MenopauseAISystem: safety_report
#     MenopauseAISystem-->>User: Final, Safe Response
# ```

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:35:12.239588Z","iopub.execute_input":"2025-08-22T18:35:12.239926Z","iopub.status.idle":"2025-08-22T18:35:12.251022Z","shell.execute_reply.started":"2025-08-22T18:35:12.239904Z","shell.execute_reply":"2025-08-22T18:35:12.249791Z"},"jupyter":{"outputs_hidden":false}}
class MenopauseAIEducationSystem:
    """The complete, integrated system for menopause education."""

    def __init__(self, questions, personas, router, llm, validator):
        self.assessment_module = CulturalAssessmentModule(questions)
        self.personas = personas
        self.content_router = router
        self.llm_engine = llm
        self.safety_validator = validator
        self.medical_disclaimer = ("\n\n---\n**Disclaimer**: This is for educational purposes only and is not medical advice. "
                                 "Please consult a healthcare provider for any health concerns.")
        print("\nMenopauseAIEducationSystem is ready.")

    def handle_query(self, persona_id, query):
        """Processes a user query from start to finish."""
        
        print(f"\n{'='*50}")
        print(f"Handling query '{query}' for Persona: {self.personas[persona_id]['name']}")
        print(f"Description: {self.personas[persona_id]['description']}")
        print(f"{'='*50}")

        # 1. Cultural Assessment
        persona_answers = self.personas[persona_id]['simulated_answers']
        pathway = self.assessment_module.get_pathway(persona_answers)
        print(f"[Step 1] Cultural Pathway determined: '{pathway.upper()}'")

        # 2. Get Content Template
        content_template = self.content_router.get_content(pathway, query)
        print("\n[Step 2] Content template selected based on pathway.")

        # 3. Generate Response (Mocked)
        generated_response = self.llm_engine.generate_response(content_template)

        # 4. Validate Response
        safety_report = self.safety_validator.validate_response(generated_response)
        print(f"\n[Step 3] Safety validation complete. Safe: {safety_report['is_safe']}")
        if not safety_report['is_safe']:
            print(f"Issues found: {safety_report['issues']}")
            final_response = "I am unable to provide a response at this time due to safety concerns."
        else:
            final_response = generated_response
        
        # 5. Add disclaimer and deliver
        final_response_with_disclaimer = final_response + self.medical_disclaimer
        
        print("\n--- FINAL RESPONSE ---")
        print(final_response_with_disclaimer)
        print(f"{'='*50}\n")
        
        return final_response_with_disclaimer

# --- Initialize the full system ---
system = MenopauseAIEducationSystem(
    questions=cultural_assessment_questions,
    personas=user_personas,
    router=ContentRouter(),
    llm=SimpleLLMEngine(),
    validator=SafetyValidator()
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 8: Demonstration with User Personas
# 
# Now, let's test our integrated system with the different user personas. This will demonstrate how the AI adapts its responses based on the culturally-informed pathways. We will ask each persona the same question about "weight gain" and observe the different outputs.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:35:30.246173Z","iopub.execute_input":"2025-08-22T18:35:30.246569Z","iopub.status.idle":"2025-08-22T18:35:30.257418Z","shell.execute_reply.started":"2025-08-22T18:35:30.246541Z","shell.execute_reply":"2025-08-22T18:35:30.255812Z"},"jupyter":{"outputs_hidden":false}}
test_query_for_all = "weight gain"

# Persona 1: Aditi (Rural Indian, Liberation perspective)
system.handle_query("persona1", test_query_for_all)

# Persona 2: Priya (Urban Professional, Medical perspective)
system.handle_query("persona2", test_query_for_all)

# Persona 3: Susan (Western, Neutral perspective)
system.handle_query("persona3", test_query_for_all)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Conclusion and Next Steps
# 
# Congratulations! You have successfully built a prototype of the AI-Powered Menopause Education System. 
# 
# In this notebook, we have:
# - Structured the project based on the assignment's design.
# - Implemented core components for cultural assessment, content routing, and safety.
# - Used a mock LLM to focus on the system's logic and architecture.
# - Demonstrated how the system can adapt its responses to different cultural perspectives using user personas.
# 
# ### Extension Opportunities (For Advanced Students)
# 
# 1.  **Integrate a Real LLM**: Replace the `SimpleLLMEngine` with a class that calls the Hugging Face `pipeline` or OpenAI API. This will provide more dynamic and nuanced responses. [2, 6]
# 2.  **Build a Simple UI**: Use a library like `Streamlit` or `Flask` to create a simple web interface for your chatbot, making it interactive.
# 3.  **Expand the Knowledge Base**: Instead of hardcoding content, create a simple knowledge base (e.g., a JSON file or a Python dictionary) that the `ContentRouter` can query. For a more advanced approach, use a vector database like `ChromaDB`. [23, 17]
# 4.  **Enhance the Safety Validator**: Improve the `SafetyValidator` by adding more sophisticated checks, such as sentiment analysis or more comprehensive bias keyword lists.
# 5.  **Multi-Language Support**: Modify the content pathways to include responses in different languages, adding another layer of personalization.

# %% [code] {"jupyter":{"outputs_hidden":false}}
