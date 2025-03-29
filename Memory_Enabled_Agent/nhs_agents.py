"""
NHS Virtual Assistant Voice Agent

This module implements a virtual assistant for NHS doctors and patients using LiveKit with:
- Vector-based memory storage using Qdrant
- User-specific memory collections
- Medical knowledge base integration
- API integration for patient and doctor data
- Consent management for medical records

Author: Avijit Sarkar (Modified version)
"""

import os
import re
import asyncio
import logging
import datetime
import json
import requests
from typing import List, Dict, Any, Optional, Annotated, Union

# Load environment variables
from dotenv import load_dotenv

# LiveKit imports
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, turn_detector
from livekit.plugins.cartesia import tts as cartesia_tts

# Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# OpenAI for embeddings
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nhs_agent")

# Load environment variables
load_dotenv()

# Environment settings
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Room validation settings
DOCTOR_SUFFIX = "-doctor"
PATIENT_SUFFIX = "-patient"
ROOM_VALIDATION_ERROR = "Room name does not match required pattern"

# API endpoints
NHS_API_BASE = "https://nhsapi.kno2gether.com/api"
PATIENT_VERIFY_ENDPOINT = f"{NHS_API_BASE}/patients/verify"
DOCTOR_VERIFY_ENDPOINT = f"{NHS_API_BASE}/doctors/verify"
MEDICAL_RECORDS_ENDPOINT = f"{NHS_API_BASE}/medical-records/patient"

# Qdrant settings
QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_TLS = os.environ.get("QDRANT_TLS", "true").lower() == "true"

# Knowledge base collections
COMMON_KNOWLEDGE_COLLECTION = "patient_assessment_commonknowledgebase"

# Create Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=QDRANT_TLS
)

# Knowledge base mapping
KNOWLEDGE_BASE_MAP = {
    "memory_map": {
        "knowledgebases": [
            {
                "id": "nhs-demo_VasoplegicShockKnowledgeBase",
                "domain": "Critical Care",
                "content": "Management of vasoplegic shock: pathophysiology, diagnosis, vasopressors, adjuvant therapies, hemodynamic monitoring strategies"
            },
            {
                "id": "nhs-demo_CaesareanPainKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Managing intraoperative pain during Caesarean under neuraxial anesthesia: risk assessment, technique selection, block testing, breakthrough pain management, incidence rates"
            },
            {
                "id": "nhs-demo_TraumaInformedCareKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Trauma-informed care in obstetric anesthesia: psychological trauma recognition, communication strategies, consent processes, preventing retraumatization in vulnerable patients"
            },
            {
                "id": "nhs-demo_SpinalPathologyKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Neuraxial anesthesia for patients with spinal pathology: mechanical back pain, disc disease, scoliosis, previous surgery, spinal dysraphism, technique modifications"
            },
            {
                "id": "nhs-demo_IntracranialPathologyKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Neuraxial anesthesia for patients with intracranial pathology: hydrocephalus, brain tumors, Chiari malformations, elevated ICP management during labor/delivery"
            },
            {
                "id": "nhs-demo_MaternalSepsisKnowledgeBase",
                "domain": "Obstetric Critical Care",
                "content": "Maternal sepsis management: risk factors, diagnosis, early warning systems, antibiotics, fluid resuscitation, source control, care location decisions"
            },
            {
                "id": "nhs-demo_CriticalCareEchocardiographyKnowledgeBase",
                "domain": "Critical Care",
                "content": "Critical care echocardiography: transthoracic/transoesophageal techniques, training requirements, cardiac views, interpretation, applications in shock and cardiac arrest"
            },
            {
                "id": "nhs-demo_NeuroanaesthesiaKnowledgeBase",
                "domain": "Neuroanesthesia",
                "content": "Anesthetic management for pituitary surgery: gland anatomy/physiology, hormone hypersecretion, preoperative assessment, airway considerations, complications"
            },
            {
                "id": "nhs-demo_PediatricCardiacERASKnowledgeBase",
                "domain": "Pediatric Cardiac",
                "content": "Enhanced recovery after pediatric cardiac surgery: patient selection, preoperative preparation, perioperative/postoperative management for accelerated recovery"
            },
            {
                "id": "nhs-demo_PediatricCardiacAnaesthesiaKnowledgeBase",
                "domain": "Pediatric Cardiac",
                "content": "Anesthetic management for children with congenital heart disease undergoing non-cardiac procedures: risk stratification, assessment, management for different CHD physiologies"
            },
            {
                "id": "nhs-demo_AirwayUltrasoundKnowledgeBase",
                "domain": "Airway Management",
                "content": "Airway ultrasound techniques: cricothyroid membrane identification, tracheostomy guidance, intubation confirmation, difficult laryngoscopy prediction, protocols"
            }
        ]
    }
}

class UserData:
    """Base class for user data"""
    
    def __init__(self, user_id: str, user_type: str):
        self.user_id = user_id
        self.user_type = user_type
        self.full_name = ""
        self.email = ""
        self.phone = ""
        self.address = ""
        self.date_of_birth = ""
        self.created_at = ""
        self.id = ""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert user data to dictionary"""
        return {
            "user_id": self.user_id,
            "user_type": self.user_type,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "date_of_birth": self.date_of_birth,
            "created_at": self.created_at,
            "id": self.id
        }
        
    def __str__(self) -> str:
        """String representation"""
        name = self.full_name if self.full_name else f"Unknown {self.user_type}"
        return f"{self.user_type.capitalize()}: {name} ({self.user_id})"

class PatientData(UserData):
    """Patient-specific data"""
    
    def __init__(self, nhs_number: str):
        super().__init__(nhs_number, "patient")
        self.nhs_number = nhs_number
        self.medical_records = []
        self.has_consent_for_records = False
        
    @classmethod
    def from_api_response(cls, response_data: Dict[str, Any]) -> 'PatientData':
        """Create PatientData instance from API response"""
        patient = cls(response_data.get("nhs_number", ""))
        patient.full_name = response_data.get("full_name", "")
        patient.email = response_data.get("email", "")
        patient.phone = response_data.get("phone", "")
        patient.address = response_data.get("address", "")
        patient.date_of_birth = response_data.get("date_of_birth", "")
        patient.created_at = response_data.get("created_at", "")
        patient.id = response_data.get("id", "")
        return patient
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert patient data to dictionary"""
        data = super().to_dict()
        data.update({
            "nhs_number": self.nhs_number,
            "has_consent_for_records": self.has_consent_for_records
        })
        if self.medical_records:
            data["medical_records"] = self.medical_records
        return data

class DoctorData(UserData):
    """Doctor-specific data"""
    
    def __init__(self, registration_number: str):
        super().__init__(registration_number, "doctor")
        self.registration_number = registration_number
        self.hospital = ""
        self.specialty = ""
        
    @classmethod
    def from_api_response(cls, response_data: Dict[str, Any]) -> 'DoctorData':
        """Create DoctorData instance from API response"""
        doctor = cls(response_data.get("registration_number", ""))
        doctor.full_name = response_data.get("full_name", "")
        doctor.hospital = response_data.get("hospital", "")
        doctor.specialty = response_data.get("specialty", "")
        doctor.email = response_data.get("email", "")
        doctor.phone = response_data.get("phone", "")
        doctor.address = response_data.get("address", "")
        doctor.date_of_birth = response_data.get("date_of_birth", "")
        doctor.created_at = response_data.get("created_at", "")
        doctor.id = response_data.get("id", "")
        return doctor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert doctor data to dictionary"""
        data = super().to_dict()
        data.update({
            "registration_number": self.registration_number,
            "hospital": self.hospital,
            "specialty": self.specialty
        })
        return data

class VectorMemory:
    """Vector memory system using Qdrant"""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._init_collection()
    
    def _init_collection(self):
        """Initialize the vector collection"""
        try:
            # Check if collection exists
            collections = qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=qmodels.Distance.COSINE
                    )
                )
                logger.info(f"Created new vector collection: {self.collection_name}")
            else:
                logger.info(f"Using existing vector collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing vector collection: {e}")
            raise
    
    def add_to_memory(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add text to vector memory"""
        try:
            # Create default metadata if none provided
            if metadata is None:
                metadata = {}
            
            # Add timestamp to metadata
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
            # Get embedding from OpenAI
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            
            # Generate a UUID for the point ID
            import uuid
            point_id = str(uuid.uuid4())
            
            # Add to collection
            qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            **metadata
                        }
                    )
                ]
            )
            
            logger.info(f"Added to memory collection {self.collection_name}: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            return False
    
    def query_memory(self, query: str, limit: int = 3) -> str:
        """Query memory for relevant information"""
        try:
            # Get embedding from OpenAI
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            
            # Search collection
            search_results = qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            if not search_results:
                return ""
            
            # Format results
            results = []
            for hit in search_results:
                text = hit.payload.get("text", "")
                timestamp = hit.payload.get("timestamp", "")
                results.append(f"({timestamp[:10]}) {text}")
            
            logger.info(f"Retrieved memory for query: {query[:50]}...")
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return ""

class KnowledgeBase:
    """Medical knowledge base integration"""
    
    def __init__(self):
        self.knowledge_map = KNOWLEDGE_BASE_MAP
    
    def get_relevant_knowledge_bases(self, query: str) -> List[str]:
        """Get relevant knowledge base IDs based on query"""
        try:
            # For now, use a simple keyword matching approach
            # In a production system, this would use embeddings and semantic search
            query = query.lower()
            relevant_kbs = []
            
            for kb in self.knowledge_map["memory_map"]["knowledgebases"]:
                # Check content and domain for keyword matches
                content = kb["content"].lower()
                domain = kb["domain"].lower()
                
                # Simple scoring system - count word matches
                score = 0
                for word in query.split():
                    if len(word) > 3:  # Skip short words
                        if word in content:
                            score += 1
                        if word in domain:
                            score += 2  # Weight domain matches higher
                
                if score > 0:
                    relevant_kbs.append({
                        "id": kb["id"],
                        "score": score,
                        "domain": kb["domain"]
                    })
            
            # Sort by relevance score
            relevant_kbs.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top 3 knowledge base IDs
            return [kb["id"] for kb in relevant_kbs[:3]]
            
        except Exception as e:
            logger.error(f"Error finding relevant knowledge bases: {e}")
            return []
    
    def query_knowledge_base(self, collection_id: str, query: str, limit: int = 3) -> str:
        """Query a specific knowledge base collection"""
        try:
            # Get embedding from OpenAI
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            
            # Search collection
            search_results = qdrant_client.search(
                collection_name=collection_id,
                query_vector=query_embedding,
                limit=limit
            )
            
            if not search_results:
                return ""
            
            # Format results
            results = []
            for hit in search_results:
                text = hit.payload.get("text", "")
                source = hit.payload.get("source", "NHS Guidelines")
                results.append(f"Source: {source}\n{text}")
            
            logger.info(f"Retrieved knowledge from {collection_id} for query: {query[:50]}...")
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error querying knowledge base {collection_id}: {e}")
            return ""
    
    def get_comprehensive_knowledge(self, query: str) -> str:
        """Get comprehensive knowledge from relevant sources"""
        try:
            # Get relevant knowledge base IDs
            kb_ids = self.get_relevant_knowledge_bases(query)
            
            # Always include common knowledge base
            if COMMON_KNOWLEDGE_COLLECTION not in kb_ids:
                kb_ids.append(COMMON_KNOWLEDGE_COLLECTION)
            
            # Query each knowledge base
            all_results = []
            for kb_id in kb_ids:
                result = self.query_knowledge_base(kb_id, query)
                if result:
                    all_results.append(result)
            
            if not all_results:
                return "No relevant information found in the knowledge base. I can only provide general NHS guidance on this topic."
            
            return "\n\n".join(all_results)
            
        except Exception as e:
            logger.error(f"Error getting comprehensive knowledge: {e}")
            return "I encountered an issue accessing the medical knowledge base. I can only provide general NHS guidance on this topic."

class NHSFunctions(llm.FunctionContext):
    """Function context for NHS virtual assistant"""
    
    def __init__(self, user_data: UserData, memory: VectorMemory, knowledge_base: KnowledgeBase):
        super().__init__()
        self.user_data = user_data
        self.memory = memory
        self.knowledge_base = knowledge_base
        self.medical_record_consent = False
    
    @llm.ai_callable(
        description="Query medical knowledge for a specific topic"
    )
    async def query_medical_knowledge(
        self,
        query: Annotated[
            str,
            llm.TypeInfo(
                description="Medical topic or question to search for"
            )
        ]
    ) -> str:
        """Query medical knowledge base"""
        try:
            knowledge = self.knowledge_base.get_comprehensive_knowledge(query)
            return f"Medical knowledge on '{query}':\n\n{knowledge}"
        except Exception as e:
            return f"Failed to query medical knowledge: {str(e)}"
    
    @llm.ai_callable(
        description="Record patient consent for accessing medical records"
    )
    async def record_patient_consent(
        self,
        has_consent: Annotated[
            bool,
            llm.TypeInfo(
                description="Whether the patient has given consent"
            )
        ]
    ) -> str:
        """Record patient's consent for accessing medical records"""
        try:
            # Only available for patients
            if self.user_data.user_type != "patient":
                return "This function is only available for patients."
            
            self.medical_record_consent = has_consent
            patient_data = self.user_data
            if isinstance(patient_data, PatientData):
                patient_data.has_consent_for_records = has_consent
            
            if has_consent:
                return "Patient consent for accessing medical records has been recorded. You may now retrieve the patient's medical records."
            else:
                return "Patient has not given consent for accessing medical records."
        except Exception as e:
            return f"Failed to record consent: {str(e)}"
    
    @llm.ai_callable(
        description="Get patient medical records"
    )
    async def get_medical_records(
        self
    ) -> str:
        """Get patient medical records"""
        try:
            # Only available for patients
            if self.user_data.user_type != "patient":
                return "This function is only available for patients."
            
            # Check for consent
            patient_data = self.user_data
            if isinstance(patient_data, PatientData):
                if not patient_data.has_consent_for_records:
                    return "You don't have the patient's consent to access their medical records. Please obtain consent first."
                
                # Get medical records from API
                nhs_number = patient_data.nhs_number
                url = f"{MEDICAL_RECORDS_ENDPOINT}/{nhs_number}"
                response = requests.get(url)
                
                if response.status_code != 200:
                    return f"Failed to retrieve medical records: {response.status_code}"
                
                records = response.json()
                patient_data.medical_records = records
                
                # Format records for display
                if not records:
                    return "No medical records found for this patient."
                
                formatted_records = []
                for record in records:
                    record_date = record.get("record_date", "Unknown date")[:10]
                    med_history = record.get("medical_history", {})
                    
                    allergies = ", ".join(med_history.get("allergies", ["None"]))
                    conditions = ", ".join(med_history.get("chronic_conditions", ["None"]))
                    medications = ", ".join(med_history.get("medications", ["None"]))
                    
                    formatted_record = (
                        f"Record Date: {record_date}\n"
                        f"Allergies: {allergies}\n"
                        f"Chronic Conditions: {conditions}\n"
                        f"Current Medications: {medications}\n"
                        f"Notes: {record.get('notes', 'No notes')}"
                    )
                    formatted_records.append(formatted_record)
                
                return "Medical Records:\n\n" + "\n\n".join(formatted_records)
            else:
                return "Invalid patient data."
        except Exception as e:
            return f"Failed to retrieve medical records: {str(e)}"
    
    @llm.ai_callable(
        description="Save important information to memory for future reference"
    )
    async def save_to_memory(
        self,
        information: Annotated[
            str,
            llm.TypeInfo(
                description="Non-sensitive information to save for future conversations"
            )
        ],
        category: Annotated[
            str,
            llm.TypeInfo(
                description="Category of information such as 'preference', 'general_health', or 'scheduling'"
            )
        ]
    ) -> str:
        """Save important non-sensitive information to memory"""
        try:
            # Add to vector memory
            success = self.memory.add_to_memory(
                information,
                metadata={
                    "category": category,
                    "user_id": self.user_data.user_id,
                    "user_type": self.user_data.user_type
                }
            )
            
            if success:
                return f"Information saved to memory under category '{category}'"
            else:
                return "Failed to save information to memory"
        except Exception as e:
            return f"Error saving to memory: {str(e)}"

class NHSAgent:
    """NHS virtual assistant for doctors and patients"""
    
    def __init__(self, user_data: UserData):
        # Store user data
        self.user_data = user_data
        
        # Initialize vector memory with user-specific collection
        collection_name = f"{user_data.user_type}_{user_data.user_id.replace('-', '_')}"
        self.memory = VectorMemory(collection_name)
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase()
        
        # Initialize function context
        self.function_context = NHSFunctions(user_data, self.memory, self.knowledge_base)
        
        # Store last user message for processing
        self.last_user_message = ""
        
        # Track full conversation history
        self.conversation_history = []
        
        logger.info(f"NHS agent initialized for {user_data}")
    
    async def before_llm_callback(self, assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """Process context before sending to LLM"""
        try:
            # Get the latest user message
            for msg in reversed(chat_ctx.messages):
                if msg.role == "user" and msg.content:
                    user_message = msg.content
                    if isinstance(user_message, list):
                        # Handle potential image content
                        user_message = "\n".join([
                            str(item) for item in user_message 
                            if not isinstance(item, llm.ChatImage)
                        ])
                    
                    # Store for later use
                    self.last_user_message = user_message
                    
                    # Retrieve memory context
                    memory_context = self.memory.query_memory(user_message)
                    
                    # Build context
                    context_parts = []
                    
                    # Add user data context
                    user_context = self._get_user_context()
                    if user_context:
                        context_parts.append(user_context)
                    
                    # Add memory context if available
                    if memory_context:
                        context_parts.append(f"Previous conversation context:\n{memory_context}")
                    
                    # For patient-specific queries, try to add knowledge base context
                    if self.user_data.user_type == "patient" or "patient" in user_message.lower():
                        # Check if this is a medical query
                        medical_terms = self._identify_medical_terms(user_message)
                        if medical_terms:
                            knowledge = self.knowledge_base.get_comprehensive_knowledge(user_message)
                            if knowledge:
                                context_parts.append(f"Relevant medical information:\n{knowledge}")
                    
                    if context_parts:
                        # Create context message
                        context_text = "Here is important context for this conversation:\n\n" + "\n\n".join(context_parts)
                        
                        # Add as system message before the user message
                        for i in range(len(chat_ctx.messages)):
                            if chat_ctx.messages[i].role == "user" and chat_ctx.messages[i].content == user_message:
                                # Insert context message before this user message
                                chat_ctx.messages.insert(i, llm.ChatMessage(
                                    role="system",
                                    content=context_text
                                ))
                                logger.info("Added context to chat")
                                break
                    
                    break
            
            # Limit context length to avoid token issues
            if len(chat_ctx.messages) > 15:
                # Keep system messages and most recent messages
                system_messages = [msg for msg in chat_ctx.messages if msg.role == "system"]
                recent_messages = chat_ctx.messages[-12:]  # Keep last 12 messages
                
                # Combine them, keeping at most one system message at the start
                if system_messages:
                    chat_ctx.messages = [system_messages[0]] + recent_messages
                else:
                    chat_ctx.messages = recent_messages
                
                logger.info("Truncated chat context to reduce token usage")
            
        except Exception as e:
            logger.error(f"Error in before_llm_callback: {e}")
    
    def _get_user_context(self) -> str:
        """Get context based on user data"""
        if self.user_data.user_type == "patient":
            if isinstance(self.user_data, PatientData):
                context = f"Patient Information:\n- Name: {self.user_data.full_name}\n- NHS Number: {self.user_data.nhs_number}\n- Date of Birth: {self.user_data.date_of_birth[:10] if self.user_data.date_of_birth else 'Not provided'}"
                
                if self.user_data.has_consent_for_records and hasattr(self.user_data, 'medical_records') and self.user_data.medical_records:
                    # Add recent medical information
                    latest_record = self.user_data.medical_records[0]
                    med_history = latest_record.get("medical_history", {})
                    
                    allergies = ", ".join(med_history.get("allergies", ["None"]))
                    conditions = ", ".join(med_history.get("chronic_conditions", ["None"]))
                    medications = ", ".join(med_history.get("medications", ["None"]))
                    
                    context += f"\n\nMedical Information:\n- Allergies: {allergies}\n- Chronic Conditions: {conditions}\n- Current Medications: {medications}"
                
                return context
        elif self.user_data.user_type == "doctor":
            if isinstance(self.user_data, DoctorData):
                return f"Doctor Information:\n- Name: {self.user_data.full_name}\n- Registration: {self.user_data.registration_number}\n- Specialty: {self.user_data.specialty}\n- Hospital: {self.user_data.hospital}"
        
        return ""
    
    def _identify_medical_terms(self, text: str) -> List[str]:
        """Identify medical terms in text"""
        # This is a simple implementation - in production, you would use a medical NER model
        medical_keywords = [
            "pain", "symptom", "disease", "infection", "medication", 
            "treatment", "diagnosis", "surgery", "prescription", "allergy",
            "condition", "chronic", "acute", "therapy", "vaccine",
            "heart", "lung", "liver", "kidney", "blood pressure", "diabetes",
            "cancer", "arthritis", "asthma", "fever", "headache"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in medical_keywords:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def add_user_message(self, message: str):
        """Add a user message to the conversation history"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Store for context in the upcoming agent response
            self.last_user_message = message
            
            logger.info(f"Added user message to conversation history: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding user message to history: {e}")
    
    def add_agent_message(self, message: str):
        """Add an agent message to the conversation history"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            logger.info(f"Added agent message to conversation history: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding agent message to history: {e}")
    
    async def process_conversation(self):
        """Process the entire conversation history at the end of the session"""
        try:
            logger.info("Processing complete conversation...")
            
            # Skip if no conversation happened
            if not self.conversation_history:
                logger.info("No conversation to process")
                return
            
            # Extract key information from the conversation
            # For both doctors and patients, we'll save the full conversation 
            # but filter out any sensitive medical information
            
            # Format the conversation for storage
            conversation_text = ""
            for message in self.conversation_history:
                prefix = "User" if message["role"] == "user" else "Assistant"
                conversation_text += f"{prefix}: {message['text']}\n\n"
            
            # Try to store the full conversation in memory for context continuity
            try:
                success = self.memory.add_to_memory(
                    text=conversation_text,
                    metadata={
                        "type": "conversation_history", 
                        "user_id": self.user_data.user_id,
                        "user_type": self.user_data.user_type,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                if success:
                    logger.info("Successfully stored conversation history in memory")
                else:
                    logger.warning("Failed to store conversation history in memory")
            except Exception as e:
                logger.error(f"Error storing conversation: {e}")
            
            logger.info("Conversation processing complete")
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")

def validate_room_name(room_name: str) -> Union[str, None]:
    """Validate that the room name ends with one of the required suffixes"""
    if room_name.endswith(DOCTOR_SUFFIX):
        return "doctor"
    elif room_name.endswith(PATIENT_SUFFIX):
        return "patient"
    else:
        return None

def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    # Load VAD model for voice activity detection
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarmed models loaded")

async def fetch_patient_data(nhs_number: str) -> Optional[PatientData]:
    """Fetch patient data from the API"""
    try:
        url = f"{PATIENT_VERIFY_ENDPOINT}/{nhs_number}"
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch patient data: {response.status_code}")
            return None
        
        patient_data = PatientData.from_api_response(response.json())
        logger.info(f"Successfully fetched patient data for {nhs_number}")
        return patient_data
        
    except Exception as e:
        logger.error(f"Error fetching patient data: {e}")
        return None

async def fetch_doctor_data(registration_number: str) -> Optional[DoctorData]:
    """Fetch doctor data from the API"""
    try:
        url = f"{DOCTOR_VERIFY_ENDPOINT}/{registration_number}"
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch doctor data: {response.status_code}")
            return None
        
        doctor_data = DoctorData.from_api_response(response.json())
        logger.info(f"Successfully fetched doctor data for {registration_number}")
        return doctor_data
        
    except Exception as e:
        logger.error(f"Error fetching doctor data: {e}")
        return None

async def entrypoint(ctx: JobContext):
    """Main entry point for the NHS LiveKit agent"""
    logger.info(f"Connecting to room {ctx.room.name}")
    
    # Validate room name to determine user type
    user_type = validate_room_name(ctx.room.name)
    if not user_type:
        logger.error(f"Room name '{ctx.room.name}' does not have the required suffix")
        ctx.error = ROOM_VALIDATION_ERROR
        return
    
    logger.info(f"Detected user type: {user_type}")
    
    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting NHS virtual assistant for participant {participant.identity}")
    
    # Get participant metadata
    user_id = ""
    user_data = None
    metadata = participant.metadata
    
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
            logger.info(f"Parsed metadata: {parsed_metadata}")
            
            if user_type == "patient":
                nhs_number = parsed_metadata.get('nhs_number', '')
                if not nhs_number:
                    # Try alternate casing
                    nhs_number = parsed_metadata.get('nhsNumber', '')
                if nhs_number:
                    # Try to fetch patient data from API
                    user_id = nhs_number
                    user_data = await fetch_patient_data(nhs_number)
                    logger.info(f"Fetched patient data: {user_data}")
            elif user_type == "doctor":
                registration_number = parsed_metadata.get('registration_number', '')
                if not registration_number:
                    # Try alternate casing
                    registration_number = parsed_metadata.get('registrationNumber', '')
                if registration_number:
                    # Try to fetch doctor data from API
                    user_id = registration_number
                    user_data = await fetch_doctor_data(registration_number)
                    logger.info(f"Fetched doctor data: {user_data}")
        except json.JSONDecodeError:
            logger.error("Failed to parse participant metadata")
    
    # If we couldn't get user data from metadata, create a basic instance
    if not user_data:
        if user_type == "patient":
            # Use a default NHS number if none provided
            user_id = "unknown_patient"
            user_data = PatientData(user_id)
            # Populate from metadata if available
            if metadata and 'parsed_metadata' in locals():
                user_data.full_name = parsed_metadata.get('name', '')
                user_data.date_of_birth = parsed_metadata.get('dateOfBirth', '')
        else:
            # Use a default registration number if none provided
            user_id = "unknown_doctor" if not registration_number else registration_number
            user_data = DoctorData(user_id)
            # Populate from metadata if available
            if metadata and 'parsed_metadata' in locals():
                user_data.full_name = parsed_metadata.get('name', '')
                user_data.hospital = parsed_metadata.get('hospitalName', '')
                user_data.specialty = parsed_metadata.get('speciality', '')
    
    logger.info(f"Initialized user data: {user_data.to_dict()}")
    
    # Initialize NHS agent with user data
    nhs_agent = NHSAgent(user_data)
    
    # Create initial system prompt based on user type
    if user_type == "patient":
        system_prompt = create_patient_system_prompt(user_data)
    else:
        system_prompt = create_doctor_system_prompt(user_data)
    
    # Initialize the chat context with the system prompt
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    
    # Create the voice pipeline agent
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
        stt=deepgram.STT(
            model="nova-2-general",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            language="en-US",
        ),
        llm=openai.LLM(
            model=MODEL_NAME,
        ),
        tts=cartesia_tts.TTS(
            model="sonic-2",
            voice="7e19344f-9f17-47d7-a13a-4366ad06ebf3",
            sample_rate=24000,
            speed="normal",
            emotion=["curiosity", "positivity:high"],
        ),
        chat_ctx=initial_ctx,
        fnc_ctx=nhs_agent.function_context,
        turn_detector=turn_detector.EOUModel(),
        before_llm_cb=nhs_agent.before_llm_callback,
    )
    
    # Set up event handlers for recording messages
    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        if isinstance(msg.content, list):
            content = "\n".join(
                "[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content
            )
        else:
            content = msg.content
            
        logger.info(f"User speech committed: {content[:50]}...")
        nhs_agent.add_user_message(content)
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: llm.ChatMessage):
        content = msg.content
        logger.info(f"Agent speech committed: {content[:50]}...")
        nhs_agent.add_agent_message(content)
    
    # Set up metrics collection
    usage_collector = metrics.UsageCollector()
    @agent.on("metrics_collected")
    def on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)
    
    # Process conversation at end of session
    async def end_of_session():
        # Process conversation for memory storage
        await nhs_agent.process_conversation()
        
        # Log conversation summary
        user_messages = [msg for msg in nhs_agent.conversation_history if msg["role"] == "user"]
        agent_messages = [msg for msg in nhs_agent.conversation_history if msg["role"] == "assistant"]
        
        if user_messages:
            # Calculate conversation statistics
            conversation_duration = None
            if len(nhs_agent.conversation_history) >= 2:
                first_msg_time = datetime.datetime.fromisoformat(nhs_agent.conversation_history[0]["timestamp"])
                last_msg_time = datetime.datetime.fromisoformat(nhs_agent.conversation_history[-1]["timestamp"])
                conversation_duration = (last_msg_time - first_msg_time).total_seconds()
            
            # Log summary
            logger.info("=" * 50)
            logger.info("CONVERSATION SUMMARY")
            logger.info("=" * 50)
            logger.info(f"User type: {user_type}")
            logger.info(f"User ID: {user_id}")
            logger.info(f"Total messages: {len(nhs_agent.conversation_history)}")
            logger.info(f"User messages: {len(user_messages)}")
            logger.info(f"Agent messages: {len(agent_messages)}")
            
            if conversation_duration:
                minutes = int(conversation_duration // 60)
                seconds = int(conversation_duration % 60)
                logger.info(f"Conversation duration: {minutes}m {seconds}s")
            
            logger.info("=" * 50)
        
        # Log usage metrics
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    # Add to shutdown callbacks
    ctx.add_shutdown_callback(end_of_session)
    
    # Start the agent
    agent.start(ctx.room, participant)
    
    # Create welcome message based on user type
    if user_type == "patient":
        welcome_message = create_patient_welcome(user_data)
    else:
        welcome_message = create_doctor_welcome(user_data)
    
    # Send welcome message
    await agent.say(welcome_message, allow_interruptions=True)

def create_patient_system_prompt(patient_data: PatientData) -> str:
    """Create system prompt for patient interactions"""
    name = patient_data.full_name if patient_data.full_name else "there"
    
    system_prompt = (
        f"You are an NHS virtual health assistant providing information and support to patients. Your interface with users will be voice. Pretend we're having a conversation, no special formatting or headings, just natural speech."
        f"\n\nNever use astericks, or other special characters, or emojis, or non-verbal expressions, or special formatting. Just natural speech."
        f"\n\nSometime it may take a while for you to respond, Let the user know you are finding more information and thinking about the best way to answer the question."
        f"You're currently speaking with a patient named {name}."
        
        f"\n\nGuiding Principles:"
        f"\n1. Medical Accuracy - Only provide information that is medically accurate and from reliable sources. "
        f"Cite your sources when giving medical information."
        f"\n2. Empathetic Support - Be warm, understanding and compassionate. Many patients are anxious or concerned."
        f"\n3. Clear Communication - Use simple, clear language avoiding medical jargon where possible."
        f"\n4. Consent First - Always ask for explicit consent before accessing or discussing personal medical records."
        f"\n5. Limitations - Be clear about your limitations. You cannot diagnose, prescribe medication, or schedule appointments."

        f"\n\nImportant Guidelines:"
        f"\n- Don't make diagnoses or suggest treatments that haven't been prescribed by a doctor"
        f"\n- Don't offer services you can't provide such as booking appointments or issuing prescriptions"
        f"\n- Always recommend consulting with a healthcare professional for specific medical issues"
        f"\n- Be respectful of the patient's emotional state and concerns"
        f"\n- For urgent medical issues, advise contacting emergency services (999) or NHS 111"
        f"\n- If you don't know something, be honest rather than speculating"
        f"\n- Save important but non-sensitive information for future conversations using the save_to_memory function"
        
        f"\n\nFunctionality Available:"
        f"\n- You can access medical knowledge bases for general health information"
        f"\n- You can access the patient's medical records ONLY after explicit consent"
        f"\n- You can explain NHS services and standard procedures"
        f"\n- You can provide general health advice backed by NHS guidelines"
        f"\n- You can save important information to memory for future conversations"
        
        f"\n\nMemory Management:"
        f"\n- Use previous conversation context to personalize interactions"
        f"\n- Proactively store useful non-sensitive information about the patient's preferences, general health concerns, and conversation details"
        f"\n- NEVER store sensitive medical information or personally identifiable data in memory"
        f"\n- Appropriate data for memory: communication preferences, general topics discussed, follow-up items"
        f"\n- Inappropriate data for memory: specific test results, detailed medical history, exact medications"
        
        f"\n\nVoice Communication Guidelines:"
        f"\n- Use short, clear sentences with natural pauses"
        f"\n- Speak in a warm, reassuring tone"
        f"\n- Use verbal acknowledgments ('I understand', 'I see', etc.)"
        f"\n- Avoid using technical medical terminology where possible"
        f"\n- Don't use emojis, special characters, or non-verbal expressions"
    )
    
    return system_prompt

def create_doctor_system_prompt(doctor_data: DoctorData) -> str:
    """Create system prompt for doctor interactions"""
    name = doctor_data.full_name if doctor_data.full_name else "Doctor"
    specialty = doctor_data.specialty if doctor_data.specialty else "medicine"
    
    system_prompt = (
        f"You are an NHS virtual clinical assistant supporting healthcare professionals. Your interface with users will be voice. Pretend we're having a conversation, no special formatting or headings, just natural speech."
        f"\n\nNever use astericks(** for Bold text etc or dashes(-) for bullet points), or other special characters, or emojis, or non-verbal expressions, or special formatting. Just natural speech."
        f"\n\nSometime it may take a while for you to respond, Let the user know you are finding more information and thinking about the best way to answer the question."
        f"You're currently speaking with {name}, a healthcare professional specializing in {specialty}."
        
        f"\n\nGuiding Principles:"
        f"\n1. Clinical Accuracy - Provide medically accurate information backed by evidence. Always cite guidelines and sources."
        f"\n2. Professional Support - Maintain a professional and efficient manner befitting clinical discussions."
        f"\n3. Detail-Oriented - Provide comprehensive information when discussing clinical topics, including relevant medical details."
        f"\n4. Evidence-Based - Base all information on current clinical evidence and established medical guidelines."
        f"\n5. Limitations - Be clear about your limitations. You cannot access patient records or make clinical decisions."

        f"\n\nImportant Guidelines:"
        f"\n- Focus on providing clinical information from established medical guidelines"
        f"\n- You cannot access specific patient information or records"
        f"\n- You can discuss general case scenarios but cannot provide specific patient advice"
        f"\n- Suggest appropriate clinical resources and references"
        f"\n- Be precise with medical terminology when appropriate for professional discussion"
        f"\n- Acknowledge when information might be outside your knowledge base"
        f"\n- Save important but non-sensitive information for future conversations using the save_to_memory function"
        
        f"\n\nFunctionality Available:"
        f"\n- You can access medical knowledge bases and NHS guidelines"
        f"\n- You can explain clinical procedures and protocols based on NHS standards"
        f"\n- You can provide information on best practices from medical literature"
        f"\n- You can assist with general clinical questions relevant to the doctor's specialty"
        f"\n- You can save important information to memory for future conversations"
        
        f"\n\nMemory Management:"
        f"\n- Use previous conversation context to maintain continuity"
        f"\n- Proactively store useful information about the doctor's clinical interests, preferred resources, and common queries"
        f"\n- NEVER store patient-specific information or sensitive clinical data in memory"
        f"\n- Appropriate data for memory: clinical areas of interest, frequently referenced guidelines, preferred clinical resources"
        f"\n- Inappropriate data for memory: specific patient cases, hypothetical scenarios with identifiable details"
        
        f"\n\nVoice Communication Guidelines:"
        f"\n- Use professional medical terminology appropriate for healthcare professionals"
        f"\n- Be concise and direct in your responses"
        f"\n- Maintain a professional tone throughout the conversation"
        f"\n- Structure information in a logical clinical format"
        f"\n- Don't use emojis, special characters, or non-verbal expressions"
    )
    
    return system_prompt

def create_patient_welcome(patient_data: PatientData) -> str:
    """Create welcome message for patients"""
    if patient_data.full_name:
        welcome_message = f"Hello {patient_data.full_name}, welcome to the NHS virtual health assistant. I'm here to provide you with health information and support. How can I help you today?"
    else:
        welcome_message = "Hello, welcome to the NHS virtual health assistant. I'm here to provide you with health information and support. May I know your name to better assist you?"
    
    return welcome_message

def create_doctor_welcome(doctor_data: DoctorData) -> str:
    """Create welcome message for doctors"""
    if doctor_data.full_name:
        specialty = f" in {doctor_data.specialty}" if doctor_data.specialty else ""
        welcome_message = f"Hello {doctor_data.full_name}, welcome to the NHS clinical assistant. I'm here to support your work{specialty}. How can I assist you today?"
    else:
        welcome_message = "Hello Doctor, welcome to the NHS clinical assistant. I'm here to support your clinical work. How can I assist you today?"
    
    return welcome_message

if __name__ == "__main__":
    # Run the LiveKit agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            num_idle_processes=2  # Keep 2 processes warm for better response time
        ),
    )
