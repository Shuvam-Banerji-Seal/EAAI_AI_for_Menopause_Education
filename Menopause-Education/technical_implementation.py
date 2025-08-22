# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Cell 1: Environment Setup and Dependencies
# This cell installs and imports all necessary libraries for the agentic AI system. It sets up the environment with `camel-ai` for the multi-agent framework, `chromadb` for the vector knowledge base, and `transformers` for local model inference.

# %% [code]
# ====================================================================
# MENOPAUSE AI EDUCATION SYSTEM - AGENTIC IMPLEMENTATION
# Breaking the Silence: Multi-Agent Culturally-Sensitive AI System
# Using: Camel AI + ChromaDB + Hugging Face Transformers
# ====================================================================

import warnings
warnings.filterwarnings('ignore')

# Install required packages for agentic AI system
import subprocess
import sys

def install_packages():
    packages = [
        'camel-ai[all]',           # Multi-agent framework with all features
        'chromadb',
        'transformers>=4.35.0',
        'torch',
        'sentence-transformers',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'python-dotenv'          
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
    
    print("✅ All packages installed successfully!")

# Uncomment the line below to run the installation
# install_packages()

# Import core libraries
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import asyncio

# Set up OpenAI API Key (replace with your actual key or use environment variables)
# os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'

# Camel AI imports
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.configs import ChatGPTConfig # Corrected config import

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

# Hugging Face imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

print("🚀 Menopause AI Education System - Agentic Architecture Ready!")
print("🔧 Using: Camel AI + ChromaDB + Hugging Face Transformers")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Cell 2: ChromaDB Knowledge Base Setup
# This cell defines the `ChromaDBKnowledgeBase` class, which manages the system's memory. It creates a persistent vector database with collections for medical facts, cultural contexts, and safety guidelines. This allows the agents to retrieve relevant information based on semantic similarity to user queries.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:33:13.325551Z","iopub.execute_input":"2025-08-22T18:33:13.325960Z","iopub.status.idle":"2025-08-22T18:33:13.334923Z","shell.execute_reply.started":"2025-08-22T18:33:13.325936Z","shell.execute_reply":"2025-08-22T18:33:13.333793Z"},"jupyter":{"outputs_hidden":false}}
# ====================================================================
# CHROMADB VECTOR DATABASE SETUP
# ====================================================================

class ChromaDBKnowledgeBase:
    def __init__(self, persist_directory="./menopause_chromadb"):
        """Initialize ChromaDB with persistent storage"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collections = {}
        self._initialize_collections()
        print(f"✅ ChromaDB initialized with persistent storage: {persist_directory}")

    def _initialize_collections(self):
        collection_names = ["medical_knowledge", "cultural_contexts", "symptom_management", "safety_guidelines"]
        for name in collection_names:
            try:
                collection = self.client.get_collection(name=name, embedding_function=self.embedding_function)
                print(f"📂 Loaded existing collection: {name}")
            except ValueError:
                collection = self.client.create_collection(name=name, embedding_function=self.embedding_function, metadata={"description": f"Knowledge base for {name}"})
                print(f"🆕 Created new collection: {name}")
            self.collections[name] = collection

    def add_knowledge(self, collection_name: str, documents: List[str], metadatas: List[Dict], ids: List[str]):
        if collection_name not in self.collections: raise ValueError(f"Collection {collection_name} not found")
        self.collections[collection_name].add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"📚 Added {len(documents)} documents to {collection_name}")

    def query_knowledge(self, collection_name: str, query: str, n_results: int = 3) -> Dict:
        if collection_name not in self.collections: raise ValueError(f"Collection {collection_name} not found")
        return self.collections[collection_name].query(query_texts=[query], n_results=n_results)

    def populate_initial_knowledge(self):
        # Medical knowledge
        medical_docs = [
            "Menopause typically occurs between ages 45-55. The average age in developed countries is 51.",
            "Hot flashes, a common symptom, affect about 75% of women and are caused by hormonal fluctuations.",
            "Perimenopause is the transition phase before menopause and can last from a few years to a decade.",
            "Hormone Replacement Therapy (HRT) is an effective treatment for many symptoms but requires consultation with a doctor due to potential risks.",
            "After menopause, the risk of osteoporosis and cardiovascular disease increases due to lower estrogen levels."
        ]
        self.add_knowledge("medical_knowledge", medical_docs, [{"source": "medical_literature"}]*len(medical_docs), [f"med_{i}" for i in range(len(medical_docs))])

        # Cultural contexts
        cultural_docs = [
            "In some traditional Indian cultures, menopause is seen as a 'liberation' from menstrual restrictions, leading to higher social status.",
            "Western cultures often frame menopause through a 'loss' perspective, focusing on loss of youth, fertility, and attractiveness.",
            "For many rural Indian women, menopause is a positive event, freeing them from the physical and financial burden of menstruation.",
            "In Japan, the term 'konenki' refers to this life stage, and it is often managed with a focus on balance and harmony, using both medical and traditional approaches.",
            "A culture of silence often surrounds menopause globally, preventing women from seeking timely education and support."
        ]
        self.add_knowledge("cultural_contexts", cultural_docs, [{"source": "anthropological_studies"}]*len(cultural_docs), [f"cult_{i}" for i in range(len(cultural_docs))])
        print("🌍 Initial knowledge base populated!")

# Initialize and populate the knowledge base
knowledge_base = ChromaDBKnowledgeBase()
if knowledge_base.collections['medical_knowledge'].count() == 0:
    knowledge_base.populate_initial_knowledge()

# Test query
test_results = knowledge_base.query_knowledge("cultural_contexts", "How is menopause viewed in India?")
print(f"\n🔍 Test Query Results: Found {len(test_results['documents'][0])} relevant documents.")
print(test_results['documents'][0][0])

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Cell 3: Hugging Face Model Engine
# This cell defines the `HuggingFaceModelEngine`. It loads local, open-source language models that can be used for text generation if a cloud-based API (like OpenAI) is unavailable. This provides flexibility and reduces reliance on paid services for core functionality.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:33:36.381759Z","iopub.execute_input":"2025-08-22T18:33:36.382844Z","iopub.status.idle":"2025-08-22T18:33:36.394064Z","shell.execute_reply.started":"2025-08-22T18:33:36.382809Z","shell.execute_reply":"2025-08-22T18:33:36.392938Z"},"jupyter":{"outputs_hidden":false}}
# ====================================================================
# HUGGING FACE MODELS SETUP
# ====================================================================

class HuggingFaceModelEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Hugging Face Engine using device: {self.device}")
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        model_name = "microsoft/DialoGPT-medium"
        try:
            print(f"📥 Loading Hugging Face model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if self.device == "cuda" else -1)
            print(f"✅ Successfully loaded {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to load Hugging Face model: {e}. Text generation will be disabled.")

    def generate_text(self, prompt: str, max_length: int = 150) -> str:
        if self.pipeline is None:
            return f"(Fallback) I understand you are asking about: {prompt}. Please consult our documented resources."
        try:
            # For chat models, we often need to format the input
            result = self.pipeline(prompt, max_length=max_length, num_return_sequences=1, pad_token_id=self.pipeline.tokenizer.eos_token_id)
            return result[0]['generated_text'].replace(prompt, "").strip()
        except Exception as e:
            print(f"⚠️ Error during text generation: {e}")
            return f"(Fallback) There was an issue processing your request about: {prompt}."

# Initialize Hugging Face Model Engine
hf_engine = HuggingFaceModelEngine()

# Test text generation
test_prompt = "Menopause is a natural transition which means"
generated_text = hf_engine.generate_text(test_prompt)
print(f"\n🧪 Test Generation:")
print(f"Prompt: {test_prompt}")
print(f"Generated: {generated_text}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Cell 4: Camel AI Multi-Agent System
# This cell defines the core of the system: the `MenopauseAIAgentSociety`. It creates a team of specialized AI agents using the Camel AI framework. Each agent has a specific role:
# - **Cultural Expert**: Assesses the user's cultural context.
# - **Medical Expert**: Provides accurate medical information.
# - **Safety Validator**: Checks responses for bias and harmful content.
# - **Coordinator**: Integrates the inputs from all other agents into a single, coherent response.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:33:58.641150Z","iopub.execute_input":"2025-08-22T18:33:58.641441Z","iopub.status.idle":"2025-08-22T18:33:58.651167Z","shell.execute_reply.started":"2025-08-22T18:33:58.641421Z","shell.execute_reply":"2025-08-22T18:33:58.650003Z"},"jupyter":{"outputs_hidden":false}}
# ====================================================================
# CAMEL AI MULTI-AGENT SYSTEM
# ====================================================================

class MenopauseAIAgentSociety:
    def __init__(self, knowledge_base: ChromaDBKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.model = None
        if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY']:
            try:
                model_config = ChatGPTConfig(temperature=0.3, max_tokens=400)
                self.model = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=ModelType.GPT_4O_MINI,
                    model_config_dict=model_config.as_dict()
                )
                print("🤖 Multi-Agent Society initialized with OpenAI GPT-4o-mini.")
            except Exception as e:
                print(f"⚠️ OpenAI model initialization failed: {e}. Agents will use rule-based logic.")
        else:
            print("🤖 Multi-Agent Society initialized in rule-based mode (OpenAI API key not found).")
        
        self.agents = self._create_agents()

    def _create_agents(self):
        agents = {}
        # Define system prompts for each agent
        cultural_expert_sys_msg = BaseMessage.make_assistant_message(role_name="Cultural Expert", content="You are a cultural sociologist specializing in global women's health. Your task is to analyze user queries to determine the cultural perspective on menopause (liberation, loss, or neutral) and guide the response framing accordingly.")
        medical_expert_sys_msg = BaseMessage.make_assistant_message(role_name="Medical Expert", content="You are a clinical nurse specialist in gynecology. Your task is to provide medically accurate, evidence-based information about menopause symptoms, risks, and treatments, drawing only from the provided knowledge base.")
        safety_validator_sys_msg = BaseMessage.make_assistant_message(role_name="Safety Validator", content="You are an AI safety and ethics officer. Your task is to review generated responses for medical misinformation, cultural bias, and harmful or definitive language. Approve or reject responses with clear reasoning.")
        coordinator_sys_msg = BaseMessage.make_assistant_message(role_name="Coordinator", content="You are the lead content strategist. Your task is to synthesize inputs from cultural and medical experts into a single, coherent, empathetic, and safe response for the user.")
        
        # Create ChatAgent instances
        agents['cultural_expert'] = ChatAgent(self.model, system_message=cultural_expert_sys_msg) if self.model else None
        agents['medical_expert'] = ChatAgent(self.model, system_message=medical_expert_sys_msg) if self.model else None
        agents['safety_validator'] = ChatAgent(self.model, system_message=safety_validator_sys_msg) if self.model else None
        agents['coordinator'] = ChatAgent(self.model, system_message=coordinator_sys_msg) if self.model else None
        
        return agents

    async def _run_agent_task(self, agent_name: str, prompt: str, fallback_response: str) -> str:
        agent = self.agents.get(agent_name)
        if agent:
            try:
                response = await agent.step(prompt)
                return response.msgs[0].content if response.msgs else fallback_response
            except Exception as e:
                print(f"⚠️ Agent '{agent_name}' failed: {e}")
                return fallback_response
        return fallback_response

    async def process_user_query(self, query: str, context: Dict) -> Dict[str, Any]:
        # 1. Cultural Assessment
        cultural_knowledge = self.knowledge_base.query_knowledge("cultural_contexts", query)
        cultural_prompt = f"User query: '{query}'. User context: {context}. Knowledge: {cultural_knowledge['documents'][0]}. Determine the cultural pathway (Liberation, Loss, Neutral) and reasoning. Respond in JSON."
        cultural_assessment_str = await self._run_agent_task('cultural_expert', cultural_prompt, '{"pathway": "neutral", "reasoning": "Defaulting due to error."}')
        try: cultural_assessment = json.loads(cultural_assessment_str)
        except json.JSONDecodeError: cultural_assessment = {"pathway": "neutral", "reasoning": cultural_assessment_str}

        # 2. Medical Guidance
        medical_knowledge = self.knowledge_base.query_knowledge("medical_knowledge", query)
        medical_prompt = f"Based on this knowledge: {medical_knowledge['documents'][0]}, draft a medically accurate response to the query: '{query}'."
        medical_guidance = await self._run_agent_task('medical_expert', medical_prompt, "Please consult a healthcare provider for medical advice.")

        # 3. Preliminary Response Generation (by Coordinator)
        coord_prompt_1 = f"User query: '{query}'. Cultural pathway: {cultural_assessment.get('pathway')}. Medical guidance: '{medical_guidance}'. Draft an integrated, empathetic response."
        draft_response = await self._run_agent_task('coordinator', coord_prompt_1, medical_guidance)

        # 4. Safety Validation
        safety_prompt = f"Review this draft response for safety, bias, and accuracy: '{draft_response}'. Respond in JSON: {{"approved": boolean, "concerns": []}}."
        safety_validation_str = await self._run_agent_task('safety_validator', safety_prompt, '{"approved": false, "concerns": ["Safety check failed."]}')
        try: safety_validation = json.loads(safety_validation_str)
        except json.JSONDecodeError: safety_validation = {"approved": False, "concerns": [safety_validation_str]}

        # 5. Final Response Coordination
        if safety_validation.get('approved'):
            final_response = draft_response
        else:
            final_response = f"I understand you have a question about menopause. For your safety, I recommend discussing this with a qualified healthcare professional. Concerns noted: {safety_validation.get('concerns')}"
        
        return {
            "final_response": final_response,
            "cultural_pathway": cultural_assessment.get('pathway'),
            "safety_approved": safety_validation.get('approved')
        }

# Initialize the agent society
agent_society = MenopauseAIAgentSociety(knowledge_base)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Cell 5: Integrated System Pipeline & Simulation
# This cell defines the main `MenopauseAISystem` class that orchestrates the entire process. It takes a user query, routes it through the multi-agent society, and produces a final, culturally-sensitive, and safety-validated response. The cell concludes with a simulation of a multi-user session to demonstrate the system's capabilities with diverse inputs.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:34:20.047416Z","iopub.execute_input":"2025-08-22T18:34:20.047817Z","iopub.status.idle":"2025-08-22T18:34:20.055405Z","shell.execute_reply.started":"2025-08-22T18:34:20.047787Z","shell.execute_reply":"2025-08-22T18:34:20.054104Z"},"jupyter":{"outputs_hidden":false}}
# ====================================================================
# INTEGRATED AGENTIC MENOPAUSE AI SYSTEM - PIPELINE
# ====================================================================

class MenopauseAISystem:
    def __init__(self, agent_society: MenopauseAIAgentSociety, hf_engine: HuggingFaceModelEngine):
        self.agent_society = agent_society
        self.hf_engine = hf_engine
        self.stats = {"total_queries": 0, "pathways": {"liberation": 0, "loss": 0, "neutral": 0}, "safety_flags": 0}
        print("✅ Menopause AI System Pipeline is live!")

    async def process_interaction(self, user_query: str, user_context: Dict = None) -> Dict[str, Any]:
        print(f"\n🔄 Processing query: '{user_query}' with context: {user_context}")
        self.stats["total_queries"] += 1
        
        # Use the agent society if an API key is provided, otherwise use local engine
        if self.agent_society.model:
            print("🧠 Using Camel AI multi-agent society...")
            result = await self.agent_society.process_user_query(user_query, user_context)
            final_response = result["final_response"]
            pathway = result["cultural_pathway"]
            if not result["safety_approved"]:
                self.stats["safety_flags"] += 1
        else:
            print("🤖 Using local Hugging Face model...")
            # Simplified local logic
            pathway = "liberation" if "freedom" in user_query.lower() else "neutral"
            prompt = f"From a {pathway} perspective, discuss: {user_query}"
            generated_text = self.hf_engine.generate_text(prompt)
            final_response = f"As a starting point for discussion: {generated_text}\n\n---\nFor comprehensive and personalized advice, please consult a healthcare provider."
        
        self.stats["pathways"][pathway.lower()] = self.stats["pathways"].get(pathway.lower(), 0) + 1
        
        print(f"🎯 Pathway determined: {pathway.upper()}")
        print(f"📝 Final Response: {final_response}")
        return {"query": user_query, "response": final_response, "pathway": pathway}

async def simulate_session():
    # Initialize the full system
    system = MenopauseAISystem(agent_society, hf_engine)

    test_queries = [
        {"query": "I feel so free now that my periods have stopped. What should I focus on for my health?", "context": {"culture": "Indian", "perspective": "liberation"}},
        {"query": "I'm really struggling with hot flashes and mood swings. What are my treatment options?", "context": {"culture": "Western", "perspective": "loss"}},
        {"query": "What is perimenopause?", "context": {"culture": "Unknown", "perspective": "neutral"}}
    ]

    for tq in test_queries:
        await system.process_interaction(tq["query"], tq["context"])

    print("\n📊 Simulation Stats:")
    print(json.dumps(system.stats, indent=2))

# Run the simulation
print("🎬 Starting Multi-User Session Simulation...")
asyncio.run(simulate_session())

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Cell 6: System Analytics and Monitoring
# This final cell provides a conceptual `MenopauseAIAnalytics` class to demonstrate how the system's performance could be monitored. It includes functions to create visualizations for the distribution of cultural pathways, agent performance metrics, and safety validation results. This illustrates the importance of continuous evaluation and improvement in a production environment.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-22T18:34:46.013130Z","iopub.execute_input":"2025-08-22T18:34:46.013588Z","iopub.status.idle":"2025-08-22T18:34:46.023208Z","shell.execute_reply.started":"2025-08-22T18:34:46.013561Z","shell.execute_reply":"2025-08-22T18:34:46.021939Z"},"jupyter":{"outputs_hidden":false}}
# ====================================================================
# SYSTEM ANALYTICS AND MONITORING DASHBOARD (CONCEPTUAL)
# ====================================================================

import matplotlib.pyplot as plt
import seaborn as sns

class MenopauseAIAnalytics:
    def __init__(self, system_stats: Dict):
        self.stats = system_stats
        plt.style.use('seaborn-v0_8-whitegrid')

    def display_dashboard(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Menopause AI System Analytics Dashboard', fontsize=16)

        # 1. Cultural Pathway Distribution
        pathways = list(self.stats['pathways'].keys())
        counts = list(self.stats['pathways'].values())
        colors = sns.color_palette("viridis", len(pathways))
        
        ax1.pie(counts, labels=pathways, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(width=0.4))
        ax1.set_title('Cultural Pathway Distribution')

        # 2. Safety Intervention Rate
        approved_count = self.stats['total_queries'] - self.stats['safety_flags']
        safety_counts = [approved_count, self.stats['safety_flags']]
        safety_labels = ['Approved', 'Flagged for Safety']

        ax2.bar(safety_labels, safety_counts, color=['#4CAF50', '#F44336'])
        ax2.set_title('Safety Validation Results')
        ax2.set_ylabel('Number of Interactions')
        for i, count in enumerate(safety_counts):
            ax2.text(i, count + 0.1, str(count), ha='center', fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# To run this cell, you would first run the simulation in Cell 5
# and pass the resulting stats to the analytics class.
print("📋 Analytics module ready. Run simulation to generate stats.")

# Example usage after running the simulation:
# sim_stats = asyncio.run(simulate_session()) # This would be run in a real script
# analytics = MenopauseAIAnalytics(sim_stats)
# analytics.display_dashboard()

# %% [code] {"jupyter":{"outputs_hidden":false}}
