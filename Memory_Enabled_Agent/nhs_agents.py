"""
NHS Virtual Assistant Voice Agent

This module implements a virtual assistant for NHS doctors and patients using LiveKit with:
- Vector-based memory storage using Qdrant
- User-specific memory collections
- Medical knowledge base integration
- API integration for patient and doctor data
- Consent management for medical records
- Cost-efficient AI for knowledge base selection using Cerebras
- Fallback to OpenAI when needed

Environment variables required:
- OPENAI_API_KEY: For embeddings and fallback LLM
- CEREBRAS_API_KEY: For cost-efficient knowledge base selection (optional, will fall back to OpenAI)
- QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY: For vector storage

Author: Avijit Sarkar (Modified version)
"""

import os
import re
import asyncio
import logging
import datetime
import json
import requests
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Annotated, Union, Set, Tuple

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
import livekit.rtc as rtc

# Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# OpenAI for embeddings
from openai import OpenAI

# Cerebras for knowledge base selection (cost-effective alternative)
from cerebras.cloud.sdk import Cerebras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nhs_agent")

# Create logs directory if it doesn't exist
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

class StructuredFileLogger:
    """Structured file logger for NHS agent conversations"""
    
    def __init__(self, user_id: str, user_type: str):
        """Initialize the structured logger
        
        Args:
            user_id (str): User's ID (NHS number or registration number)
            user_type (str): Type of user ('patient' or 'doctor')
        """
        self.user_id = user_id
        self.user_type = user_type
        
        # Create unique log filename with timestamp and user info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_id = str(uuid.uuid4())[:8]  # Short unique ID
        safe_user_id = "".join([c if c.isalnum() else "_" for c in user_id])
        
        filename = f"{timestamp}_{user_type}_{safe_user_id}_{self.log_id}.log"
        self.log_file = logs_dir / filename
        
        # Create the log file with header
        with open(self.log_file, "w", encoding="utf-8") as f:
            header = (
                f"=== NHS AGENT CONVERSATION LOG ===\n"
                f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"User Type: {user_type}\n"
                f"User ID: {user_id}\n"
                f"Log ID: {self.log_id}\n"
                f"===============================\n\n"
            )
            f.write(header)
        
        logger.info(f"Created structured log file: {self.log_file}")
        
        # Track knowledge bases used
        self.knowledge_bases_used: Set[str] = set()
    
    async def log_event(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log an event to the structured log file asynchronously
        
        Args:
            event_type (str): Type of event (e.g., 'USER_QUERY', 'CONTEXT', 'AGENT_RESPONSE')
            content (str): Main content of the event
            metadata (Dict[str, Any], optional): Additional metadata for the event
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Format the log entry
            log_entry = f"[{timestamp}] {event_type}\n"
            
            if content:
                # Format content with indentation for readability
                formatted_content = "\n".join(f"    {line}" for line in content.split("\n"))
                log_entry += f"{formatted_content}\n"
            
            # Add metadata if provided
            if metadata:
                # Format metadata as indented JSON for readability
                try:
                    metadata_str = json.dumps(metadata, indent=4)
                    formatted_metadata = "\n".join(f"    {line}" for line in metadata_str.split("\n"))
                    log_entry += f"METADATA:\n{formatted_metadata}\n"
                except:
                    # Fallback if JSON conversion fails
                    log_entry += f"METADATA: {str(metadata)}\n"
            
            log_entry += "---\n\n"
            
            # Write to file asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_to_file,
                log_entry
            )
            
        except Exception as e:
            logger.error(f"Error writing to structured log file: {e}")
    
    def _write_to_file(self, content: str):
        """Write content to log file (called by run_in_executor)"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
    
    async def log_user_query(self, query: str):
        """Log a user query"""
        await self.log_event("USER_QUERY", query)
    
    async def log_context(self, context: str, context_type: str = "GENERAL"):
        """Log context added to the agent"""
        await self.log_event("CONTEXT", context, {"type": context_type})
    
    async def log_agent_response(self, response: str):
        """Log agent text response"""
        await self.log_event("AGENT_RESPONSE", response)
    
    async def log_tts_output(self, text: str, duration_ms: int = None):
        """Log Text-to-Speech output"""
        metadata = {"duration_ms": duration_ms} if duration_ms else {}
        await self.log_event("TTS_OUTPUT", text, metadata)
    
    async def log_knowledge_retrieval(self, kb_id: str, query: str, content: str):
        """Log knowledge retrieval from vector database"""
        # Track that this knowledge base was used
        self.knowledge_bases_used.add(kb_id)
        
        await self.log_event(
            "KNOWLEDGE_RETRIEVAL", 
            content,
            {
                "knowledge_base_id": kb_id,
                "query": query
            }
        )
    
    async def log_memory_retrieval(self, query: str, content: str):
        """Log memory retrieval"""
        await self.log_event(
            "MEMORY_RETRIEVAL", 
            content,
            {"query": query}
        )
    
    async def log_conversation_summary(self, num_turns: int, duration_sec: int = None):
        """Log conversation summary at the end"""
        summary = (
            f"Conversation Summary:\n"
            f"- Total turns: {num_turns}\n"
            f"- Knowledge bases used: {', '.join(self.knowledge_bases_used) if self.knowledge_bases_used else 'None'}\n"
        )
        
        if duration_sec is not None:
            minutes = int(duration_sec // 60)
            seconds = int(duration_sec % 60)
            summary += f"- Duration: {minutes}m {seconds}s\n"
        
        await self.log_event("CONVERSATION_SUMMARY", summary)

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

# Knowledge base maps for different user types
DOCTOR_KNOWLEDGE_BASE_MAP = {
    "memory_map": {
        "knowledgebases": [
            {
                "id": "nhs-demo_VasoplegicShockKnowledgeBase",
                "domain": "Critical Care",
                "content": "Management of vasoplegic shock: pathophysiology, diagnosis, vasopressors, adjuvant therapies, hemodynamic monitoring strategies",
                "document_name": "Management-of-vasoplegic-shock_2024_bjae",
                "document_title": "Management of vasoplegic shock",
                "authors": "R.N. Mistry and J.E. Winearls, Gold Coast University Hospital, Australia"
            },
            {
                "id": "nhs-demo_CaesareanPainKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Managing intraoperative pain during Caesarean under neuraxial anesthesia: risk assessment, technique selection, block testing, breakthrough pain management, incidence rates",
                "document_name": "Patient-centred-strategies-in-obstetric-anaesthesi",
                "document_title": "Prevention and management of intraoperative pain during Caesarean section",
                "authors": "S. Orbach-Zinger and Y. Binyamin, Israel"
            },
            {
                "id": "nhs-demo_TraumaInformedCareKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Trauma-informed care in obstetric anesthesia: psychological trauma recognition, communication strategies, consent processes, preventing retraumatization in vulnerable patients",
                "document_name": "Patient-centred-strategies-in-obstetric-anaesthesi",
                "document_title": "Patient-centred strategies in obstetric anaesthesia",
                "authors": "B.D. Mergler, C.C. Duffy and R.J. Mergler, USA"
            },
            {
                "id": "nhs-demo_SpinalPathologyKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Neuraxial anesthesia for patients with spinal pathology: mechanical back pain, disc disease, scoliosis, previous surgery, spinal dysraphism, technique modifications",
                "document_name": "Neuraxial-anaesthesia-in-the-parturient-with-pre-e",
                "document_title": "Neuraxial anaesthesia in the parturient with pre-existing structural spinal pathology",
                "authors": "G. Crowe and T. Drew, Ireland"
            },
            {
                "id": "nhs-demo_IntracranialPathologyKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Neuraxial anesthesia for patients with intracranial pathology: hydrocephalus, brain tumors, Chiari malformations, elevated ICP management during labor/delivery",
                "document_name": "Neuraxial-anaesthesia-for-the-parturient-with-intr",
                "document_title": "Neuraxial anaesthesia for the parturient with intracranial pathology",
                "authors": "C. Warrick, W. Schievink and M. Zakowski, USA"
            },
            {
                "id": "nhs-demo_AirwayUltrasoundKnowledgeBase",
                "domain": "Airway Management",
                "content": "Airway ultrasound techniques for laryngoscopy, larynx, trachea, and tracheostomy procedures",
                "document_name": "Airway ultrasound",
                "document_title": "Airway ultrasound",
                "authors": "R. Lohse, W.H. Teoh and M.S. Kristensen, Copenhagen University Hospital, Denmark"
            },
            {
                "id": "nhs-demo_PediatricCardiacAnaesthesiaKnowledgeBase",
                "domain": "Pediatric Anesthesia",
                "content": "Anesthesia for children with congenital heart disease undergoing non-cardiac surgery",
                "document_name": "Anaesthesia-for-children-with-congenital-heart-dis",
                "document_title": "Anaesthesia for children with congenital heart disease undergoing non-cardiac surgery",
                "authors": "J. Spiro, J. Bauerle and D. Njoku, St. Louis Children's Hospital, USA"
            },
            {
                "id": "nhs-demo_NeuroanaesthesiaKnowledgeBase",
                "domain": "Neuroanesthesia",
                "content": "Anesthesia for pituitary surgery: perioperative considerations and management",
                "document_name": "Anaesthesia-for-pituitary-surgery_2024_bjae",
                "document_title": "Anaesthesia for pituitary surgery",
                "authors": "K. Raveendran, S. Kwok and L. Glancz, UK"
            },
            {
                "id": "nhs-demo_CriticalCareEchocardiographyKnowledgeBase",
                "domain": "Critical Care",
                "content": "Critical care echocardiography: training, imaging techniques, and clinical indications",
                "document_name": "Critical-care-echocardiography--training,-imaging",
                "document_title": "Critical care echocardiography: training, imaging, and indications",
                "authors": "J.K. Cheng and R. Arntfield, New Zealand and Canada"
            },
            {
                "id": "nhs-demo_PediatricCardiacERASKnowledgeBase",
                "domain": "Pediatric Anesthesia",
                "content": "Enhanced recovery protocols after pediatric cardiac surgery",
                "document_name": "Enhanced-recovery-after-paediatric-cardiac-surgery",
                "document_title": "Enhanced recovery after paediatric cardiac surgery",
                "authors": "L. Foote, L. Hepburn and C. Goodison, Great Ormond Street Hospital, UK"
            },
            {
                "id": "nhs-demo_MaternalSepsisKnowledgeBase",
                "domain": "Obstetric Anesthesia",
                "content": "Maternal sepsis: background, diagnosis, and management approaches",
                "document_name": "Maternal-sepsis--background,-diagnosis-and-managemt",
                "document_title": "Maternal sepsis: background, diagnosis and management",
                "authors": "J. Manigrasso, N. Desai and E. Naoum, USA and UK"
            }
        ]
    }
}

# Simplified knowledge base map for patients - using only common knowledge collection
PATIENT_KNOWLEDGE_BASE_MAP = {
    "memory_map": {
        "knowledgebases": [
            {
                "id": "patient_assessment_generalanaestheticsrisksknowledgebase",
                "domain": "Anaesthesia",
                "content": "General anaesthetics risks and side effects: frequency of common side effects (shivering, nausea, sore throat), rare complications (dental damage, nerve injury, allergic reactions), and very rare risks (accidental awareness, visual loss, mortality rates), with statistical incidence data",
                "document_name": "General-anaesthetics-Risks-and-side-effects",
                "document_title": "General anaesthetics: Risks and side effects",
                "authors": "Royal College of Anaesthetists (RCoA)",
                "date": "2024"
            },
            {
                "id": "patient_assessment_pediatricanaestheticsrisksknowledgebase",
                "domain": "Pediatric Anaesthesia",
                "content": "Common events and risks for children and young people having general anaesthesia: categorized by frequency (very common, common, uncommon, rare, very rare), including sore throat, behavioral changes, minor injuries, breathing problems, need for intensive care, anaphylaxis, and long-term risks",
                "document_name": "Common-events-and-risks-for-children-and-young-people-having-a-general-anaesthetic",
                "document_title": "Common events and risks for children and young people having a general anaesthetic",
                "authors": "Royal College of Anaesthetists (RCoA) and Association of Paediatric Anaesthetists of Great Britain and Ireland",
                "date": "2022-03"
            },
            {
                "id": "medical_assessment_epiduralanaesthesiaknowledgebase",
                "domain": "Regional Anaesthesia",
                "content": "Epidural anaesthesia during and after surgery: explanation of procedure, benefits compared to other pain relief methods, contraindications, insertion technique, potential side effects and risks, and shared decision-making process",
                "document_name": "Epidural-anaesthesia-during-and-after-surgery",
                "document_title": "Epidural anaesthesia during and after surgery",
                "authors": "Royal College of Anaesthetists (RCoA) and Association of Anaesthetists",
                "date": "2023-06"
            },
            {
                "id": "medical_assessment_accidentalawarenessknowledgebase",
                "domain": "Anaesthesia",
                "content": "Waking up during a general anaesthetic (accidental awareness): explanation of what it is, how likely it is to happen, what it feels like, causes, risk reduction strategies, and what to do if it happens including where to seek help and support",
                "document_name": "Anaesthetics-risks-and-side-effects-Waking-up-during-a-general-anaesthetic",
                "document_title": "Anaesthetics – risks and side effects: Waking up during a general anaesthetic (accidental awareness)",
                "authors": "Leila Finikarides for the Royal College of Anaesthetists (RCoA)",
                "date": "2024-11"
            },
            {
                "id": "patient_assessment_anaestheticdeathseriousharmknowledgebase",
                "domain": "Anaesthesia",
                "content": "Death and serious harm risks during anaesthesia and surgery: mortality statistics, risk factors, mechanisms of serious harm (allergic reactions, airway problems, reduced blood supply), risk reduction strategies by anaesthetists and patients",
                "document_name": "Anaesthetics-risks-and-side-effects-Death-and-serious-harm",
                "document_title": "Anaesthetics – risks and side effects: Death and serious harm",
                "authors": "Leila Finikarides for the Royal College of Anaesthetists (RCoA)",
                "date": "2024-11"
            },
            {
                "id": "patient_assessment_peripheralnerveblockdamageknowledgebase",
                "domain": "Regional Anaesthesia",
                "content": "Nerve damage after peripheral nerve blocks: symptoms, duration of effects, incidence rates for temporary and permanent damage, causes of nerve damage (needle trauma, vascular damage, medication effects), management, and treatment options",
                "document_name": "Anaesthetics-risks-and-side-effects-Nerve-damage-after-a-peripheral-nerve-block",
                "document_title": "Anaesthetics – risks and side effects: Nerve damage after a peripheral nerve block",
                "authors": "Leila Finikarides for the Royal College of Anaesthetists (RCoA)",
                "date": "2024-11"
            },
            {
                "id": "patient_assessment_spinalanaesthesiaknowledgebase",
                "domain": "Regional Anaesthesia",
                "content": "Spinal anaesthesia: explanation of procedure, suitable operations, benefits compared to general anaesthesia, administration technique, patient experience during and after the procedure, recovery process, and shared decision-making",
                "document_name": "Your-spinal-anaesthetic",
                "document_title": "Your spinal anaesthetic",
                "authors": "Royal College of Anaesthetists (RCoA), Association of Anaesthetists and RA-UK",
                "date": "2023-04"
            }
        ]
    }
}

# Create Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=QDRANT_TLS
)

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
    
    def __init__(self, collection_name: str, file_logger=None):
        self.collection_name = collection_name
        self.file_logger = file_logger  # Will be set by NHSAgent
        self._init_collection()
    
    def set_file_logger(self, file_logger):
        """Set the file logger for this memory system"""
        self.file_logger = file_logger
    
    async def _log_to_file(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log memory operations to file if logger is available"""
        if self.file_logger:
            await self.file_logger.log_event(f"MEMORY_{event_type}", content, metadata)
    
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
            
            # Log embedding generation asynchronously
            if self.file_logger:
                text_preview = text[:200] + "..." if len(text) > 200 else text
                asyncio.create_task(self._log_to_file(
                    "EMBED_GENERATION",
                    f"Generated embedding for memory text: {text_preview}",
                    {
                        "collection": self.collection_name,
                        "text_length": len(text),
                        "embedding_model": "text-embedding-3-small",
                        "metadata_keys": list(metadata.keys())
                    }
                ))
            
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
            
            # Log successful memory addition asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_to_file(
                    "STORAGE_SUCCESS",
                    f"Successfully stored in memory collection: {self.collection_name}",
                    {
                        "point_id": point_id,
                        "collection": self.collection_name
                    }
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            
            # Log error asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_to_file(
                    "STORAGE_ERROR",
                    f"Error adding to memory: {e}",
                    {"collection": self.collection_name}
                ))
            
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
            
            # Log query embedding generation asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_to_file(
                    "QUERY_EMBED",
                    f"Generated embedding for memory query: {query}",
                    {
                        "collection": self.collection_name,
                        "query": query,
                        "embedding_model": "text-embedding-3-small",
                        "limit": limit
                    }
                ))
            
            # Search collection
            search_results = qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            if not search_results:
                # Log no results asynchronously
                if self.file_logger:
                    asyncio.create_task(self._log_to_file(
                        "QUERY_NO_RESULTS",
                        f"No memory found for query: {query}",
                        {"collection": self.collection_name}
                    ))
                return ""
            
            # Format results
            results = []
            point_ids = []
            for hit in search_results:
                text = hit.payload.get("text", "")
                timestamp = hit.payload.get("timestamp", "")
                results.append(f"({timestamp[:10]}) {text}")
                
                # Track point IDs for logging
                if hasattr(hit, "id"):
                    point_ids.append(str(hit.id))
            
            logger.info(f"Retrieved memory for query: {query[:50]}...")
            
            # Log memory retrieval results asynchronously
            if self.file_logger:
                results_preview = "\n".join(results)
                if len(results_preview) > 500:
                    results_preview = results_preview[:500] + "..."
                
                asyncio.create_task(self._log_to_file(
                    "QUERY_RESULTS",
                    f"Retrieved memory for query: {query}\n\nResults:\n{results_preview}",
                    {
                        "collection": self.collection_name,
                        "result_count": len(results),
                        "point_ids": point_ids
                    }
                ))
            
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            
            # Log error asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_to_file(
                    "QUERY_ERROR",
                    f"Error querying memory: {e}",
                    {
                        "collection": self.collection_name,
                        "query": query
                    }
                ))
            
            return ""

class KnowledgeBase:
    """Vector database based medical knowledge retrieval system"""
    
    def __init__(self, knowledge_map=None, file_logger=None):
        """Initialize with the appropriate knowledge base map based on user type"""
        self.knowledge_map = knowledge_map or DOCTOR_KNOWLEDGE_BASE_MAP
        self.file_logger = file_logger  # Optional file logger (will be set by NHSAgent)
        self._last_ai_analysis = {
            "refined_query": "",
            "selected_kb_ids": [],
            "explanation": ""
        }
    
    def set_file_logger(self, file_logger):
        """Set the file logger for this knowledge base"""
        self.file_logger = file_logger
    
    async def _log_knowledge_event(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log knowledge base event if file logger is available"""
        if self.file_logger:
            await self.file_logger.log_event(f"KB_{event_type}", content, metadata)
    
    def ai_analyze_query(self, query: str) -> Dict[str, Any]:
        """Use AI to analyze the query and determine most relevant knowledge bases
        
        Args:
            query: The user's medical query or question
            
        Returns:
            Dictionary containing:
                - refined_query: Query optimized for vector search
                - selected_kb_ids: List of relevant knowledge base IDs
                - explanation: Brief explanation of knowledge base selection
        """
        try:
            logger.info(f"AI analyzing query: {query[:50]}...")
            
            # Get the knowledgebases from the map
            kb_map_key = next(iter(self.knowledge_map.keys()))
            knowledgebases = self.knowledge_map[kb_map_key]["knowledgebases"]
            
            # Create a description of available knowledge bases for the model
            kb_descriptions = []
            for kb in knowledgebases:
                kb_descriptions.append({
                    "id": kb["id"],
                    "domain": kb["domain"],
                    "content": kb["content"],
                    "document_title": kb.get("document_title", "Unknown")
                })
            
            # Create prompt for the AI model
            system_prompt = """You are an expert medical library assistant. Your job is to:
1. Analyze medical queries to understand their core information needs
2. Select the most relevant knowledge bases that would contain the answer
3. Reformulate the query to be optimal for vector search retrieval
4. Provide a brief explanation for your selections

Select between 1-3 knowledge bases that are most likely to contain relevant information.
If no knowledge bases are relevant, return an empty list.
"""
            
            user_prompt = f"""Medical query: {query}

Available knowledge bases:
{json.dumps(kb_descriptions, indent=2)}

Return your response as a JSON object with these fields:
- refined_query: A reformulated version of the query optimized for vector search
- selected_kb_ids: Array of IDs for the most relevant knowledge base(s), up to 3 max
- explanation: Brief explanation of why these knowledge bases were selected

Example:
{{
  "refined_query": "pathophysiology and management of vasoplegic shock in critical care",
  "selected_kb_ids": ["nhs-demo_VasoplegicShockKnowledgeBase"],
  "explanation": "The query is about vasoplegic shock management which directly matches this knowledge base."
}}"""

            # Use Cerebras instead of OpenAI for knowledge base selection (cost-effective)
            try:
                # Initialize Cerebras client (using API key from environment variable)
                cerebras_client = Cerebras(
                    api_key=os.environ.get("CEREBRAS_API_KEY")
                )
                
                # Try up to 3 times with increasing backoff
                max_retries = 3
                retry_count = 0
                response_content = None
                
                while retry_count < max_retries and response_content is None:
                    try:
                        # Make the API request to Cerebras
                        response = cerebras_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            model="llama-3.3-70b",  # Use Llama 3.3 70B model for best results
                            response_format={"type": "json_object"}
                        )
                        
                        # Extract response content
                        response_content = response.choices[0].message.content
                        logger.info(f"Cerebras response received: {len(response_content)} chars")
                        
                    except Exception as retry_error:
                        retry_count += 1
                        if retry_count < max_retries:
                            # Exponential backoff: 1s, 2s, 4s, etc.
                            wait_time = 2 ** (retry_count - 1)
                            logger.warning(f"Cerebras API call failed (attempt {retry_count}/{max_retries}). Retrying in {wait_time}s: {str(retry_error)}")
                            # Use synchronous sleep instead of asyncio.sleep since this is not an async function
                            time.sleep(wait_time)
                        else:
                            # Last attempt failed, re-raise
                            raise retry_error
                
            except Exception as cerebras_error:
                # Fall back to OpenAI if Cerebras fails
                logger.warning(f"Cerebras API failed after retries, falling back to OpenAI: {str(cerebras_error)}")
                
                # Use OpenAI as fallback
                openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Using smaller model for speed and cost efficiency
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                response_content = response.choices[0].message.content
                logger.info("Using OpenAI fallback for KB selection")
            
            # Parse response
            response_json = json.loads(response_content)
            
            # Get required fields with defaults
            refined_query = response_json.get("refined_query", query)
            selected_kb_ids = response_json.get("selected_kb_ids", [])
            explanation = response_json.get("explanation", "")
            
            # Log the result
            logger.info(f"AI query analysis: Selected {len(selected_kb_ids)} knowledge bases for query")
            if selected_kb_ids:
                logger.info(f"Selected KBs: {selected_kb_ids}")
                logger.info(f"Refined query: '{refined_query}'")
                logger.info(f"Reason: {explanation}")
            
            # Store the analysis results as an instance attribute
            result = {
                "refined_query": refined_query,
                "selected_kb_ids": selected_kb_ids,
                "explanation": explanation
            }
            self._last_ai_analysis = result
            
            # Log knowledge base selection asynchronously
            if self.file_logger:
                # This is a sync function, so we need to use create_task for async logging
                asyncio.create_task(self._log_knowledge_event(
                    "SELECTION", 
                    f"Selected {len(selected_kb_ids)} knowledge bases for query: {query[:100]}...",
                    {
                        "query": query,
                        "refined_query": refined_query,
                        "selected_kb_ids": selected_kb_ids,
                        "explanation": explanation
                    }
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI query analysis: {e}")
            # Fall back to the original query and no knowledge bases
            result = {
                "refined_query": query,
                "selected_kb_ids": [],
                "explanation": "Error in AI analysis"
            }
            self._last_ai_analysis = result
            return result
    
    def get_relevant_knowledge_bases(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant knowledge base IDs based on query"""
        try:
            # For now, use a keyword matching approach
            # In a production system, this could use embeddings for better matching
            query = query.lower()
            relevant_kbs = []
            
            # Get the knowledgebases from whichever map is being used
            kb_map_key = next(iter(self.knowledge_map.keys()))
            knowledgebases = self.knowledge_map[kb_map_key]["knowledgebases"]
            
            for kb in knowledgebases:
                # Check content and domain for keyword matches
                content = kb["content"].lower()
                domain = kb["domain"].lower()
                title = kb.get("document_title", "").lower()
                
                # Simple scoring system - count word matches
                score = 0
                for word in query.split():
                    if len(word) > 3:  # Skip short words
                        if word in content:
                            score += 1.5
                        if word in domain:
                            score += 2  # Weight domain matches higher
                        if word in title:
                            score += 2.5  # Weight title matches highest
                
                if score > 0:
                    relevant_kbs.append({
                        "id": kb["id"],
                        "score": score,
                        "domain": kb["domain"],
                        "document_title": kb.get("document_title", ""),
                        "document_name": kb.get("document_name", ""),
                        "authors": kb.get("authors", "")
                    })
            
            # Sort by relevance score
            relevant_kbs.sort(key=lambda x: x["score"], reverse=True)
            
            # Log the selected knowledge bases
            if relevant_kbs:
                logger.info(f"Selected knowledge bases for query '{query[:30]}...': {[kb['id'] for kb in relevant_kbs[:3]]}")
                # Log knowledge base selection asynchronously
                if self.file_logger:
                    asyncio.create_task(self._log_knowledge_event(
                        "KEYWORD_SELECTION", 
                        f"Selected knowledge bases via keyword matching for: {query[:100]}...",
                        {
                            "query": query,
                            "selected_kb_ids": [kb['id'] for kb in relevant_kbs[:3]]
                        }
                    ))
            else:
                logger.warning(f"No relevant knowledge bases found for query: {query[:30]}...")
                if self.file_logger:
                    asyncio.create_task(self._log_knowledge_event(
                        "NO_KB_FOUND", 
                        f"No relevant knowledge bases found for: {query[:100]}..."
                    ))
                
            # Return top 3 knowledge base details
            return relevant_kbs[:3]
        
        except Exception as e:
            logger.error(f"Error finding relevant knowledge bases: {e}")
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "ERROR", 
                    f"Error finding relevant knowledge bases: {e}",
                    {"query": query}
                ))
            return []
    
    def get_kb_info_by_id(self, kb_id: str) -> Dict[str, Any]:
        """Get knowledge base info by ID"""
        kb_map_key = next(iter(self.knowledge_map.keys()))
        knowledgebases = self.knowledge_map[kb_map_key]["knowledgebases"]
        
        for kb in knowledgebases:
            if kb["id"] == kb_id:
                # Create a comprehensive info dictionary with all available fields
                kb_info = {
                    "id": kb["id"],
                    "score": 10,  # Assign high score since it was selected by AI
                    "domain": kb["domain"]
                }
                
                # Add optional fields only if they exist in the knowledge map
                if "document_title" in kb:
                    kb_info["document_title"] = kb["document_title"]
                
                if "document_name" in kb:
                    kb_info["document_name"] = kb["document_name"]
                
                if "authors" in kb and kb["authors"]:
                    kb_info["authors"] = kb["authors"]
                
                if "date" in kb:
                    kb_info["date"] = kb["date"]
                
                return kb_info
        
        return None
    
    def query_knowledge_base(self, kb_info: Dict[str, Any], query: str, limit: int = 5) -> Dict[str, Any]:
        """Query a specific knowledge base collection"""
        collection_id = kb_info["id"]
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
                # Log empty result asynchronously
                if self.file_logger:
                    asyncio.create_task(self._log_knowledge_event(
                        "RETRIEVAL_EMPTY", 
                        f"No results found in {collection_id} for query: {query[:100]}..."
                    ))
                return {"text": "", "source_info": kb_info}
            
            # Format results with page numbers or section information if available
            results = []
            for hit in search_results:
                text = hit.payload.get("text", "")
                page = hit.payload.get("page", "")
                section = hit.payload.get("section", "")
                
                # Add page/section context if available
                context = ""
                if page:
                    context += f"(Page {page})"
                if section:
                    context += f" Section: {section}"
                
                if text:
                    if context:
                        results.append(f"{context}\n{text}")
                    else:
                        results.append(text)
            
            logger.info(f"Retrieved knowledge from {collection_id} for query: {query[:50]}...")
            
            # Log search results asynchronously
            if self.file_logger:
                combined_text = "\n\n".join(results)
                asyncio.create_task(self.file_logger.log_knowledge_retrieval(
                    collection_id, 
                    query, 
                    combined_text
                ))
            
            return {
                "text": "\n\n".join(results),
                "source_info": kb_info,
                "score": kb_info["score"]
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge base {collection_id}: {e}")
            
            # Log error asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "ERROR", 
                    f"Error querying knowledge base {collection_id}: {e}",
                    {"query": query}
                ))
                
            return {"text": "", "source_info": kb_info, "score": 0}
    
    def get_comprehensive_knowledge(self, query: str) -> Dict[str, Any]:
        """Get comprehensive knowledge from relevant sources using AI for knowledge base selection"""
        try:
            # Use AI to analyze the query and select knowledge bases
            ai_analysis = self.ai_analyze_query(query)
            refined_query = ai_analysis["refined_query"]
            selected_kb_ids = ai_analysis["selected_kb_ids"]
            explanation = ai_analysis["explanation"]
            
            all_results = []
            sources_used = []
            
            # Log the AI's decision
            if selected_kb_ids:
                logger.info(f"AI selected {len(selected_kb_ids)} knowledge bases for query: {query}")
                logger.info(f"Refined query: {refined_query}")
                logger.info(f"Explanation: {explanation}")
                
                # Query each selected knowledge base
                for kb_id in selected_kb_ids:
                    kb_info = self.get_kb_info_by_id(kb_id)
                    
                    if kb_info:
                        result = self.query_knowledge_base(kb_info, refined_query)
                        if result["text"]:
                            all_results.append({
                                "text": result["text"],
                                "score": result.get("score", 0),
                                "source_info": result["source_info"]
                            })
                            
                            # Add source information directly from knowledge map
                            source_info = result["source_info"]
                            source_entry = {
                                "id": source_info["id"],
                                "title": source_info.get("document_title", "Unknown"),
                                "domain": source_info.get("domain", "")
                            }
                            
                            # Only add authors if they exist in the knowledge map
                            if "authors" in source_info and source_info["authors"]:
                                source_entry["authors"] = source_info["authors"]
                            
                            sources_used.append(source_entry)
            else:
                # Fallback to traditional method if AI didn't select any knowledge bases
                logger.info(f"AI didn't select any knowledge bases, falling back to keyword matching")
                
                # Log fallback
                if self.file_logger:
                    asyncio.create_task(self._log_knowledge_event(
                        "FALLBACK", 
                        f"AI didn't select any knowledge bases, falling back to keyword matching for: {query[:100]}..."
                    ))
                
                # Get relevant knowledge base details using keyword matching
                kb_details = self.get_relevant_knowledge_bases(query)
                
                # Query each knowledge base
            for kb_info in kb_details:
                result = self.query_knowledge_base(kb_info, query)
                if result["text"]:
                    all_results.append({
                        "text": result["text"],
                        "score": result.get("score", 0),
                        "source_info": result["source_info"]
                    })
                    
                        # Add source information directly from knowledge map
                    source_info = result["source_info"]
                        source_entry = {
                        "id": source_info["id"],
                        "title": source_info.get("document_title", "Unknown"),
                        "domain": source_info.get("domain", "")
                        }
                        
                        # Only add authors if they exist in the knowledge map
                        if "authors" in source_info and source_info["authors"]:
                            source_entry["authors"] = source_info["authors"]
                        
                        sources_used.append(source_entry)
            
            if not all_results:
                # Log no results
                if self.file_logger:
                    asyncio.create_task(self._log_knowledge_event(
                        "NO_RESULTS", 
                        f"No knowledge found for query: {query[:100]}..."
                    ))
                    
                return {
                    "text": "I don't have specific information about that in my knowledge base. I can only provide information based on the medical literature I have access to.",
                    "sources": []
                }
            
            # Sort results by relevance score
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Combine texts from all sources
            combined_text = "\n\n".join([result["text"] for result in all_results])
            
            # Log final results
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "FINAL_RESULT", 
                    f"Found knowledge from {len(sources_used)} sources for: {query[:100]}...",
                    {
                        "query": query,
                        "sources_count": len(sources_used),
                        "source_ids": [source["id"] for source in sources_used]
                    }
                ))
            
            return {
                "text": combined_text,
                "sources": sources_used
            }
            
        except Exception as e:
            error_msg = f"Error getting comprehensive knowledge: {e}"
            logger.error(error_msg)
            
            # Log error
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "ERROR", 
                    error_msg,
                    {"query": query}
                ))
                
            return {
                "text": "I encountered an issue accessing the medical knowledge base. I can only provide information based on the medical literature I have access to.",
                "sources": []
            }

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
            knowledge_result = self.knowledge_base.get_comprehensive_knowledge(query)
            knowledge_text = knowledge_result.get("text", "")
            sources = knowledge_result.get("sources", [])
            
            if not knowledge_text:
                return "I'm sorry, I don't have specific information about that topic in my knowledge base. Please consider consulting official medical resources or speaking with a healthcare provider for more information."
            
            # Format response with source citations
            response = f"Here's information on '{query}':\n\n{knowledge_text}\n\n"
            
            # Add source citations
            if sources:
                response += "This information is sourced from:\n"
                for idx, source in enumerate(sources):
                    title = source.get("title", "Unknown")
                    authors = source.get("authors", "")
                    domain = source.get("domain", "")
                    response += f"{idx+1}. {title} by {authors} ({domain})\n"
                
                response += "\nWhen using this information in clinical practice, always refer to these source documents for complete details and context."
            else:
                response += "Note: This information is provided for educational purposes only and should be verified with primary medical literature sources."
            
            return response
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
        
        # Initialize the structured file logger
        self.file_logger = StructuredFileLogger(user_data.user_id, user_data.user_type)
        
        # Initialize vector memory with user-specific collection
        collection_name = f"{user_data.user_type}_{user_data.user_id.replace('-', '_')}"
        self.memory = VectorMemory(collection_name, self.file_logger)
        
        # Set logger for vector memory
        self.memory.set_file_logger(self.file_logger)
        
        # Load appropriate knowledge base map based on user type
        if user_data.user_type == "patient":
            logger.info("Initializing patient knowledge base")
            self.knowledge_base = KnowledgeBase(PATIENT_KNOWLEDGE_BASE_MAP, self.file_logger)
        else:
            logger.info("Initializing doctor knowledge base")
            self.knowledge_base = KnowledgeBase(DOCTOR_KNOWLEDGE_BASE_MAP, self.file_logger)
        
        # Set logger for knowledge base
        self.knowledge_base.set_file_logger(self.file_logger)
        
        # Initialize function context
        self.function_ctx = NHSFunctions(user_data, self.memory, self.knowledge_base)
        
        # Initialize conversation history
        self.conversation_history = []
        self.last_user_message = ""
        
        # Keep track of sources used in last query for validation
        self.last_query_sources = []
        
        # Flag to track if knowledge was found for the current query
        self.knowledge_found = False
        self.current_query_type = None
        
        # Log initialization
        logger.info(f"NHS agent initialized for {user_data}")
        
        # Log initialization in the structured file
        asyncio.create_task(self.file_logger.log_event(
            "AGENT_INITIALIZED", 
            f"NHS agent initialized for {user_data}",
            {
                "user_type": user_data.user_type,
                "user_id": user_data.user_id,
                "memory_collection": collection_name,
                "knowledge_base_type": "PATIENT" if user_data.user_type == "patient" else "DOCTOR"
            }
        ))
        
    def get_restricted_response_for_no_knowledge(self, query: str) -> str:
        """
        Get a restricted response when no knowledge is found
        
        This ensures the model never makes up information when our knowledge base doesn't have answers
        
        Args:
            query: The user's query
            
        Returns:
            A standard restricted response that doesn't fabricate information
        """
        # Standard response options to ensure no fabrication occurs
        responses = [
            "I don't have specific information about that in my knowledge base. I can only provide information based on the NHS medical literature I have access to. For this specific question, I recommend speaking with a healthcare professional.",
            
            "I don't have information about that in my NHS knowledge sources. To get accurate information on this topic, I'd recommend speaking with a healthcare provider or contacting NHS 111.",
            
            "My NHS knowledge sources don't contain information to answer that specific question. For medical questions that I can't address with my available sources, it's best to consult with a healthcare professional directly.",
            
            "That information isn't available in my NHS knowledge base. Since I can only provide information from verified NHS sources that I have access to, I suggest discussing this with your doctor or healthcare team.",
            
            "I can only provide information from specific NHS medical sources in my knowledge base, and I don't have information about that particular topic. For this question, it would be best to speak with a healthcare professional who can give you personalized advice."
        ]
        
        # Get a consistent response based on the query hash (so repeated queries get the same response)
        query_hash = hash(query) % len(responses)
        response = responses[query_hash]
        
        logger.info(f"Providing restricted response for query with no knowledge: {query[:50]}...")
        return response
    
    def validate_citation(self, response_text: str) -> Tuple[bool, str]:
        """
        Validate if the response correctly cites sources
        
        Args:
            response_text: The agent's response text
            
        Returns:
            Tuple containing:
            - Boolean indicating if the citation is valid
            - Error message or empty string if valid
        """
        # Special handling for medical queries with no knowledge found
        # If this was a medical query and we found no knowledge, the response
        # should ONLY be our restricted response with no fabricated information
        if self.current_query_type == "medical" and not self.knowledge_found:
            # Get the expected restricted response
            expected_response = self.get_restricted_response_for_no_knowledge(self.last_user_message)
            
            # Allow minor variations (whitespace, capitalization)
            normalized_expected = ' '.join(expected_response.lower().split())
            normalized_actual = ' '.join(response_text.lower().split())
            
            # Check if the response is substantially different from our restricted response
            if normalized_expected not in normalized_actual:
                return False, "Response contains fabricated medical information when no knowledge was found"
            
            # Also check for specific indicators of fabrication
            fabrication_indicators = [
                "according to", "study", "research", "published", "journal",
                "article", "publication", "literature", "report", "findings",
                "evidence", "clinical trial", "meta-analysis", "review",
                "guideline", "recommendation", "statistical", "data shows",
                "statistics indicate", "mortality rate", "morbidity rate",
                "by", "author", "paper", "publication"
            ]
            
            response_lower = response_text.lower()
            for indicator in fabrication_indicators:
                if indicator in response_lower:
                    return False, f"Response fabricates medical information by referencing '{indicator}' when no knowledge was found"
            
            return True, ""
        
        # Skip validation if there are no sources to validate against
        if not self.last_query_sources:
            # If this is a medical query but we have no sources, do an extra check for fabrication
            if self.current_query_type == "medical":
                # Check for indicators of fabricated medical sources
                fabrication_indicators = [
                    "according to", "study", "research", "published", "journal",
                    "article", "publication", "literature", "report", "findings",
                    "evidence", "clinical trial", "meta-analysis", "review",
                    "guideline", "recommendation", "statistical", "data shows",
                    "statistics indicate", "mortality rate", "morbidity rate"
                ]
                
                response_lower = response_text.lower()
                for indicator in fabrication_indicators:
                    if indicator in response_lower:
                        return False, f"Response appears to cite non-existent medical sources using '{indicator}'"
                        
            return True, ""
        
        # Extract all valid document titles from the last query
        valid_titles = []
        valid_authors = []
        for source in self.last_query_sources:
            if "title" in source and source["title"]:
                valid_titles.append(source["title"].lower())
            if "authors" in source and source["authors"]:
                valid_authors.append(source["authors"].lower())
        
        # If there are no valid titles to check against, skip validation
        if not valid_titles:
            return True, ""
            
        # Check for known invalid document names that have been fabricated in the past
        known_invalid_names = [
            "anesthesia and children", 
            "anesthesia safety",
            "american society of anesthesiologists",
            "pediatric anesthesia safety",
            "anesthesia risks",
            "anesthesia guidelines",
            "anesthesiology",
            "journal of anesthesiology",
            "anesthesia journal",
            "mortality and morbidity in anaesthesia",
            "lunn and mushin",
            "british journal of anaesthesia",
            "new england journal of medicine",
            "lancet",
            "jama",
            "anesthesiology today",
            "anesthesia statistics",
            "pediatric anesthesia journal",
            "modern anesthesia"
        ]
        
        response_lower = response_text.lower()
        
        # Format valid titles and authors for display
        formatted_titles = ["'" + title + "'" for title in valid_titles]
        valid_titles_str = ", ".join(formatted_titles)
        
        # First, check for known fabricated document names
        for invalid_name in known_invalid_names:
            if invalid_name in response_lower:
                # Look for evidence of citation
                citation_markers = ["according to", "based on", "published by", "from the", "by", "in the", "cited in"]
                for marker in citation_markers:
                    marker_lower = marker.lower()
                    if marker_lower in response_lower:
                        # Check if it's referring to a valid citation
                        is_valid_citation = False
                        for title in valid_titles:
                            # Look for the marker followed by our valid title
                            marker_pos = response_lower.find(marker_lower)
                            if marker_pos >= 0 and title in response_lower[marker_pos:marker_pos + 100]:
                                is_valid_citation = True
                                break
                        
                        # If not a valid citation and the invalid name appears near the marker
                        if not is_valid_citation:
                            marker_pos = response_lower.find(marker_lower)
                            if marker_pos >= 0 and invalid_name in response_lower[marker_pos:marker_pos + 100]:
                                error_msg = f"Response cites a non-existent source: '{invalid_name}'. "
                                error_msg += f"Valid sources are: {valid_titles_str}"
                                return False, error_msg
        
        # Check for any author names that weren't provided
        author_markers = ["by", "authored by", "written by", "published by"]
        for marker in author_markers:
            marker_lower = marker.lower()
            if marker_lower in response_lower:
                # Extract potential author names
                marker_pos = response_lower.find(marker_lower)
                following_text = response_lower[marker_pos + len(marker_lower):marker_pos + 100]
                
                # Check if any of our valid authors are mentioned
                is_valid_author = False
                for author in valid_authors:
                    if author in following_text:
                        is_valid_author = True
                        break
                
                # If it seems to mention authors but none of them are valid
                if not is_valid_author:
                    for invalid_name in ["lunn", "mushin", "jones", "smith", "wilson", "thompson", "brown", "miller"]:
                        if invalid_name in following_text:
                            error_msg = f"Response mentions author '{invalid_name}' who is not in the provided sources."
                            error_msg += f" Valid sources include authors: {', '.join(valid_authors)}"
                            return False, error_msg
        
        # Common citation phrases that might indicate a source
        citation_phrases = [
            "according to", 
            "based on", 
            "as stated in", 
            "as mentioned in", 
            "as described in",
            "from the",
            "the document",
            "published by",
            "report by",
            "study by",
            "research by",
            "article by",
            "data from",
            "findings from"
        ]
        
        # Look for citations in the response
        for phrase in citation_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in response_lower:
                # Check if the citation is to a valid source
                citation_valid = False
                for title in valid_titles:
                    if title in response_lower:
                        citation_valid = True
                        break
                        
                # If we found a citation phrase but no valid title, the citation might be invalid
                if not citation_valid:
                    # Extract what was actually cited (approximate method)
                    cited_source = ""
                    try:
                        phrase_pos = response_lower.find(phrase_lower)
                        if phrase_pos >= 0:
                            # Look for the end of the citation (period, comma, quotation mark)
                            end_markers = [".", ",", "\"", "\n"]
                            start_pos = phrase_pos + len(phrase_lower)
                            
                            # Skip spaces after the phrase
                            while start_pos < len(response_lower) and response_lower[start_pos].isspace():
                                start_pos += 1
                                
                            # Find the end marker
                            end_pos = len(response_lower)
                            for marker in end_markers:
                                marker_pos = response_lower.find(marker, start_pos)
                                if marker_pos >= 0 and marker_pos < end_pos:
                                    end_pos = marker_pos
                            
                            # Extract the cited source
                            if end_pos > start_pos:
                                cited_source = response_text[start_pos:end_pos].strip()
                    except:
                        # If extraction fails, just note there's an invalid citation
                        cited_source = "Unknown"
                    
                    # Check if it's a known medical journal or publication
                    known_journals = [
                        "lancet", "nejm", "new england journal", "jama", "bmj", "british medical journal", 
                        "anesthesiology", "journal of", "american journal", "journal"
                    ]
                    
                    for journal in known_journals:
                        if journal in cited_source.lower():
                            error_msg = f"Response cites a medical journal or publication that was not provided: '{cited_source}'. "
                            error_msg += f"Valid sources are: {valid_titles_str}"
                            return False, error_msg
                    
                    # Prepare error message
                    error_msg = f"Response contains invalid citation: '{cited_source}'. "
                    error_msg += f"Valid sources are: {valid_titles_str}"
                    
                    return False, error_msg
        
        # Check for statistical claims without sources
        stat_patterns = [
            r"\d+ in \d+", r"\d+% of", r"\d+ percent", r"one in \d+", r"1 in \d+", 
            r"mortality rate", r"morbidity rate", r"survival rate", r"death rate", 
            r"\d+,\d+ cases", r"\d+-\d+%"
        ]
        
        for pattern in stat_patterns:
            matches = re.findall(pattern, response_lower)
            if matches and "sources are" not in response_lower and "valid sources" not in response_lower:
                error_msg = f"Response contains statistical claims '{matches[0]}' without proper citation."
                return False, error_msg
        
        return True, ""
    
    async def before_llm_callback(self, assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """Callback executed before sending messages to LLM"""
        try:
            # Skip if no messages
            if not chat_ctx.messages:
                return
            
            # Get the current user message (last one in the context)
            user_message = ""
            for msg in reversed(chat_ctx.messages):
                if msg.role == "user":
                    user_message = msg.content
                    break
            
            if not user_message:
                return
            
            # Add to last user message cache for context
                    self.last_user_message = user_message
                    
            # Reset knowledge tracking for this query
            self.knowledge_found = False
            self.last_query_sources = []
            
            # Determine if this is a medical query
            is_medical_query = self._is_knowledge_query(user_message)
            self.current_query_type = "medical" if is_medical_query else "general"
            
            # Add context based on user message and history
            for i in range(len(chat_ctx.messages)):
                if chat_ctx.messages[i].role == "user" and chat_ctx.messages[i].content == user_message:
                    # Add relevant context before this user message
                    context_parts = []
                    
                    # 1. Add user context (patient/doctor info)
                    user_context = self._get_user_context()
                    if user_context:
                        context_parts.append(user_context)
                        # Log user context
                        await self.file_logger.log_context(user_context, "USER_INFO")
                    
                    # 2. Add memory context only for non-medical queries
                    if not is_medical_query:
                        memory_context = self.memory.query_memory(user_message)
                    if memory_context:
                            context_parts.append(f"Relevant information from previous conversations:\n{memory_context}")
                    
                            # Log memory retrieval
                            await self.file_logger.log_memory_retrieval(user_message, memory_context)
                    
                    # 3. Add knowledge base context for medical queries
                    if is_medical_query:
                        knowledge_result = self.knowledge_base.get_comprehensive_knowledge(user_message)
                        knowledge_text = knowledge_result.get("text", "")
                        sources = knowledge_result.get("sources", [])
                        
                        # Store sources used for validation
                        self.last_query_sources = sources
                        
                        # Mark if knowledge was found
                        self.knowledge_found = bool(knowledge_text and knowledge_text.strip() and 
                                                  not knowledge_text.startswith("I don't have specific information"))
                        
                        # Log if no knowledge was found
                        if not self.knowledge_found:
                            logger.warning(f"No knowledge found for medical query: {user_message[:100]}...")
                            await self.file_logger.log_event(
                                "NO_KNOWLEDGE_FOUND",
                                f"No knowledge found for query: {user_message[:100]}...",
                                {"query_type": "medical"}
                            )
                        
                        # Log knowledge retrieval for each source
                        if sources:
                            for source in sources:
                                source_id = source.get("id", "")
                                if source_id:
                                    # Extract just the portion of text related to this source (best effort)
                                    # This is a simplification - in reality, we'd need a more structured way to track which
                                    # parts of knowledge_text came from which source
                                    source_text = knowledge_text[:1000] + "..." if len(knowledge_text) > 1000 else knowledge_text
                                    await self.file_logger.log_knowledge_retrieval(source_id, user_message, source_text)
                                    
                                    # Add detailed source logging
                                    await self.file_logger.log_event(
                                        "SOURCE_DETAILS", 
                                        f"Using source for query: {user_message[:100]}...",
                                        {
                                            "source_id": source.get("id", ""),
                                            "document_title": source.get("title", "Unknown"),
                                            "authors": source.get("authors", ""),
                                            "domain": source.get("domain", "")
                                        }
                                    )
                         
                        # If this is a medical query with no knowledge found, enforce a strict knowledge fence
                        if is_medical_query and not self.knowledge_found:
                            # Get a restricted response
                            restricted_response = self.get_restricted_response_for_no_knowledge(user_message)
                            
                            # Set a special system instruction to enforce the knowledge fence
                            strict_instruction = (
                                "CRITICAL RESTRICTION: The NHS knowledge base has NO information to address this medical query. "
                                "You MUST NOT use your own knowledge or training data to answer this question. "
                                "You MUST NOT make up or fabricate any medical information, citations, or sources. "
                                "You MUST NOT reference any medical literature, studies, or statistics not explicitly provided. "
                                "Instead, you MUST respond ONLY with the following exact message, word for word:\n\n"
                                f"{restricted_response}\n\n"
                                "DO NOT MODIFY THIS MESSAGE IN ANY WAY. Do not add any additional text or information. "
                                "This is a medical safety requirement to prevent misinformation."
                            )
                            
                            context_parts.append(strict_instruction)
                            
                            # Log the knowledge fence enforcement
                            await self.file_logger.log_context(strict_instruction, "KNOWLEDGE_FENCE")
                            
                        elif knowledge_text:
                            # Get the AI explanation if available
                            ai_explanation = ""
                            if hasattr(self.knowledge_base, "_last_ai_analysis"):
                                ai_analysis = self.knowledge_base._last_ai_analysis
                                if ai_analysis:
                                    ai_explanation = ai_analysis.get("explanation", "")
                            
                            knowledge_context = f"Relevant medical information:\n{knowledge_text}\n\n"
                            
                            # Add source information for citations
                            if sources:
                                knowledge_context += "Source Information:\n"
                                for idx, source in enumerate(sources):
                                    source_id = source.get("id", "")
                                    title = source.get("title", "Unknown")
                                    authors = source.get("authors", "")
                                    domain = source.get("domain", "")
                                    
                                    # Enhanced formatting with explicit labels
                                    knowledge_context += f"Source {idx+1}:\n"
                                    knowledge_context += f"  EXACT DOCUMENT TITLE: \"{title}\"\n"
                                    
                                    # Only include author information if it exists
                                    if authors:
                                        knowledge_context += f"  AUTHORS: {authors}\n"
                                    
                                    if domain:
                                        knowledge_context += f"  DOMAIN: {domain}\n"
                                    
                                    knowledge_context += "\n"
                                
                                # Add AI explanation if available
                                if ai_explanation:
                                    knowledge_context += f"Reason for source selection: {ai_explanation}\n"
                            
                            context_parts.append(knowledge_context)
                            
                            # Log knowledge context
                            await self.file_logger.log_context(knowledge_context, "KNOWLEDGE")
                            
                            # Add reminder to cite sources with specific instructions
                            if sources:
                                # Prepare formatted source titles for the warning
                                formatted_source_titles = []
                                for source in sources:
                                    title = source.get('title', 'Unknown')
                                    formatted_source_titles.append(f"'{title}'")
                                
                                source_titles_str = ", ".join(formatted_source_titles)
                                
                                # Add strict validation warning to citation instructions
                                reminder = (
                                    "CRITICAL: When answering this medical question, you MUST cite each source you use using the EXACT document titles provided.\n"
                                    "- Begin your response with 'According to [EXACT Document Title]' or 'According to [EXACT Document Title] by [Authors]' if author information is available\n"
                                    "- For each separate piece of information from different sources, clearly indicate the source using its EXACT title\n"
                                    "- If synthesizing from multiple sources, list each source by its EXACT title: 'Based on information from [Source 1 EXACT Title], [Source 2 EXACT Title], and [Source 3 EXACT Title]...'\n"
                                    "- ALWAYS refer to the source by the EXACT document title provided above - do not abbreviate, summarize, or rephrase the title\n"
                                    "- NEVER make up or invent document titles or author names that aren't provided\n"
                                    "- NEVER use generic names like 'Anesthesia Safety' when the actual title is different\n"
                                    "- If asked about your sources, ONLY mention the exact titles and authors that were provided to you\n"
                                    "- If you're unsure about the exact title of a source, refer to the specific titles listed in the source information above\n"
                                    f"- WARNING: This is a medical application. Your response will be validated against the following document titles: {source_titles_str}\n"
                                    f"- Responses that cite non-existent documents will be rejected as this is a safety-critical medical application\n"
                                    "This ensures the user knows the information comes from authoritative medical literature and not from your general knowledge."
                                )
                                context_parts.append(reminder)
                                
                                # Log source citation instructions
                                await self.file_logger.log_context(reminder, "CITATION_INSTRUCTIONS")
                    
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
                                
                                # Log the full context added
                                await self.file_logger.log_context(context_text, "FULL_CONTEXT")
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
                await self.file_logger.log_event("CONTEXT_TRUNCATION", "Chat context truncated to reduce token usage")
            
        except Exception as e:
            error_msg = f"Error in before_llm_callback: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
    
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
    
    def _is_knowledge_query(self, text: str) -> bool:
        """Identify if a message is a medical knowledge query"""
        # Keywords that suggest a request for medical information
        knowledge_keywords = [
            "what is", "how does", "explain", "tell me about", "information on", 
            "research on", "studies", "guidelines", "protocol", "procedure",
            "evidence", "treatment", "causes", "symptoms", "diagnosis", "prognosis",
            "what are", "how to", "what should", "best practice", "recommend", "guidance"
        ]
        
        text_lower = text.lower()
        
        # Check for question marks
        has_question = "?" in text
        
        # Check for knowledge query keywords
        for keyword in knowledge_keywords:
            if keyword in text_lower:
                return True
        
        # If it has a question mark and medical terms, it's likely a knowledge query
        medical_terms = self._identify_medical_terms(text)
        if has_question and len(medical_terms) > 0:
            return True
            
        return False
    
    async def add_user_message(self, message: str):
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
            
            # Log user message
            await self.file_logger.log_user_query(message)
            
            logger.info(f"Added user message to conversation history: {message[:50]}...")
            
        except Exception as e:
            error_msg = f"Error adding user message to history: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
    
    async def add_agent_message(self, message: str):
        """Add an agent message to the conversation history"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Log agent response
            await self.file_logger.log_agent_response(message)
            
            logger.info(f"Added agent message to conversation history: {message[:50]}...")
            
        except Exception as e:
            error_msg = f"Error adding agent message to history: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
    
    async def process_conversation(self):
        """Process the entire conversation history at the end of the session"""
        try:
            logger.info("Processing complete conversation...")
            
            # Skip if no conversation happened
            if not self.conversation_history:
                logger.info("No conversation to process")
                await self.file_logger.log_event("INFO", "No conversation to process")
                return
            
            # Extract key information from the conversation
            # For both doctors and patients, we'll save the full conversation 
            # but filter out any sensitive medical information
            
            # Format the conversation for storage
            conversation_text = ""
            for message in self.conversation_history:
                prefix = "User" if message["role"] == "user" else "Assistant"
                conversation_text += f"{prefix}: {message['text']}\n\n"
            
            # Log what's about to be stored in memory
            memory_preview = conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text
            await self.file_logger.log_event(
                "MEMORY_PROCESSING", 
                f"Processing conversation for memory storage:\n{memory_preview}",
                {
                    "message_count": len(self.conversation_history),
                    "first_message_time": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
                    "last_message_time": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
                }
            )
            
            # Try to store the full conversation in memory for context continuity
            try:
                # Extract potential memory topics
                topics = []
                for msg in self.conversation_history:
                    if msg["role"] == "user":
                        # Simple topic extraction - first few words
                        text = msg["text"].strip()
                        topic = " ".join(text.split()[:5]) + "..." if len(text.split()) > 5 else text
                        topics.append(topic)
                
                # Prepare metadata for memory storage
                memory_metadata = {
                        "type": "conversation_history", 
                        "user_id": self.user_data.user_id,
                        "user_type": self.user_data.user_type,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message_count": len(self.conversation_history),
                    "topics": topics[:3]  # Store up to 3 topics
                }
                
                # Log detailed memory metadata
                await self.file_logger.log_event(
                    "MEMORY_METADATA", 
                    f"Memory storage metadata:\n{json.dumps(memory_metadata, indent=2)}",
                    memory_metadata
                )
                
                # Save to memory
                success = self.memory.add_to_memory(
                    text=conversation_text,
                    metadata=memory_metadata
                )
                
                if success:
                    logger.info("Successfully stored conversation history in memory")
                    await self.file_logger.log_event(
                        "MEMORY_SAVE_SUCCESS", 
                        "Successfully stored conversation history in memory",
                        {"collection_name": self.memory.collection_name}
                    )
                else:
                    logger.warning("Failed to store conversation history in memory")
                    await self.file_logger.log_event("WARNING", "Failed to store conversation history in memory")
            except Exception as e:
                error_msg = f"Error storing conversation: {e}"
                logger.error(error_msg)
                await self.file_logger.log_event("ERROR", error_msg)
            
            # Calculate conversation duration
            start_time = None
            end_time = None
            
            if len(self.conversation_history) > 0:
                start_time = datetime.datetime.fromisoformat(self.conversation_history[0]["timestamp"])
                end_time = datetime.datetime.fromisoformat(self.conversation_history[-1]["timestamp"])
            
            # Log conversation summary
            if start_time and end_time:
                duration_sec = (end_time - start_time).total_seconds()
                await self.file_logger.log_conversation_summary(len(self.conversation_history), duration_sec)
            else:
                await self.file_logger.log_conversation_summary(len(self.conversation_history))
            
            logger.info("Conversation processing complete")
            
        except Exception as e:
            error_msg = f"Error processing conversation: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
    
    async def send_chat_message(self, room: rtc.Room, message: str):
        """Send a chat message to all participants in the room"""
        try:
            if not room or not message:
                logger.warning("Cannot send empty message or no room provided")
                await self.file_logger.log_event("WARNING", "Cannot send empty message or no room provided")
                return False
                
            # Send message to all participants in the room
            await room.local_participant.publish_data(message.encode('utf-8'), rtc.DataPacketKind.RELIABLE)
            logger.info(f"Sent chat message: {message[:50]}...")
            
            # Log the message
            await self.file_logger.log_event("CHAT_MESSAGE_SENT", message)
            
            # Add message to conversation history
            await self.add_agent_message(message)
            return True
            
        except Exception as e:
            error_msg = f"Error sending chat message: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
            return False

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
    try:
        logger.info("Loading VAD model...")
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("✅ VAD model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load VAD model: {e}")
        logger.warning("Will attempt to load VAD model at runtime")
        proc.userdata["vad"] = None
    
    # Try to load turn detector model, but make it completely optional
    try:
        logger.info("Loading turn detector model...")
        proc.userdata["turn_detector"] = turn_detector.EOUModel()
        logger.info("✅ Turn detector model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load turn detector model: {e}")
        
        # Try an alternative approach for Docker environments
        try:
            import os
            import sys
            
            # Directory where the model should be
            model_dir = os.path.expanduser("~/.cache/livekit-plugins-turn-detector")
            
            # If the directory doesn't exist, create it
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                logger.info(f"Created model directory: {model_dir}")
            
            # Log available model files
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                logger.info(f"Files in model directory: {files}")
            else:
                logger.warning(f"Model directory does not exist: {model_dir}")
            
            logger.warning("Attempting to load turn detector model again...")
            proc.userdata["turn_detector"] = turn_detector.EOUModel()
            logger.info("✅ Turn detector model loaded on second attempt")
        except Exception as e2:
            logger.warning(f"Second attempt to load turn detector model failed: {e2}")
        logger.warning("Agent will continue using default pause detection for turn detection")
        proc.userdata["turn_detector"] = None

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
    
    # Check required API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    cerebras_api_key = os.environ.get("CEREBRAS_API_KEY")
    
    if not openai_api_key:
        logger.error("OpenAI API key not set in environment variables")
        ctx.error = "OpenAI API key not available. Please provide a valid API key."
        return
    
    if not cerebras_api_key:
        logger.warning("Cerebras API key not set. Will fall back to OpenAI for knowledge base selection.")
    else:
        logger.info("Using Cerebras for knowledge base selection")
    
    # Validate room name to determine user type
    user_type = validate_room_name(ctx.room.name)
    if not user_type:
        logger.error(f"Room name '{ctx.room.name}' does not have the required suffix")
        ctx.error = ROOM_VALIDATION_ERROR
        return
    
    logger.info(f"Detected user type: {user_type}")
    
    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    
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
    try:
        # Use the turn detector from prewarm if available, otherwise None
        turn_detector_instance = ctx.proc.userdata.get("turn_detector")
        if turn_detector_instance:
            logger.info("Using pre-loaded turn detector model")
        else:
            logger.info("Turn detector not available, using default pause detection")
        
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
            llm=openai.LLM.with_cerebras(model="llama-3.3-70b"),
            tts=cartesia_tts.TTS(
                model="sonic-2",
                voice="7e19344f-9f17-47d7-a13a-4366ad06ebf3",
                sample_rate=24000,
                speed="normal",
                emotion=["curiosity", "positivity:high"],
            ),
            chat_ctx=initial_ctx,
            fnc_ctx=nhs_agent.function_ctx,
            turn_detector=turn_detector_instance,  # Can be None if not available
            before_llm_cb=nhs_agent.before_llm_callback,
        )

        chat = rtc.ChatManager(ctx.room)
        
        # Set response for both voice and chat
        last_chat_message_id = None
        
        # Handle voice messages
        @agent.on("user_speech_committed")
        def on_user_speech_committed(msg: llm.ChatMessage):
            if isinstance(msg.content, list):
                content = "\n".join(
                    "[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content
                )
            else:
                content = msg.content
                
            logger.info(f"User speech committed: {content[:50]}...")
            
            # Log user message asynchronously
            asyncio.create_task(nhs_agent.add_user_message(content))
        
        @agent.on("agent_speech_committed")
        def on_agent_speech_committed(msg: llm.ChatMessage):
            """Handler for agent speech commit events"""
            try:
            nonlocal last_chat_message_id
                logger.info(f"Agent response: {msg.content[:100]}...")
                
                # Validate the response for proper citation
                is_valid, error_msg = nhs_agent.validate_citation(msg.content)
                if not is_valid:
                    # Log the validation error
                    logger.error(f"Citation validation failed: {error_msg}")
                    asyncio.create_task(nhs_agent.file_logger.log_event(
                        "CITATION_ERROR",
                        f"Response contained invalid citation and was rejected: {error_msg}",
                        {"original_response": msg.content}
                    ))
                    
                    # Generate a corrected response
                    warning_msg = (
                        "I need to correct my previous statement. "
                        "Let me provide accurate information based only on the medical literature available to me."
                    )
                    
                    # Regenerate the response
                    agent.generate_reply(with_prefix=warning_msg)
                return
            
                # Add to conversation history if the response is valid
                asyncio.create_task(nhs_agent.add_agent_message(msg.content))
                
                # Log the agent's response
                asyncio.create_task(nhs_agent.file_logger.log_agent_response(msg.content))
                
                # If this is a response to a chat message, also send it as a chat message
                if last_chat_message_id is not None:
                    asyncio.create_task(nhs_agent.send_chat_message(ctx.room, msg.content))
                    last_chat_message_id = None
                
            except Exception as e:
                logger.error(f"Error handling agent response: {e}")
                # Try to recover and continue
        
        # Log TTS output
        @agent.on("tts_audio_generated")
        def on_tts_audio_generated(text: str, duration_ms: int):
            logger.info(f"TTS audio generated for: {text[:50]}... ({duration_ms}ms)")
            
            # Log TTS output asynchronously
            asyncio.create_task(nhs_agent.file_logger.log_tts_output(text, duration_ms))
        
        @chat.on("message_received")
        def on_message_received(msg: rtc.ChatMessage):
            logger.info(f"Chat message received: {msg.message}")
            if msg.message:
                # Add message to chat context
                chat_ctx.append(
                    text=msg.message,
                    role="user"
                )
                # Generate reply
                agent.generate_reply()
                logger.info("Response generation created")

        # Handle chat messages (text-based chat)
        @ctx.room.on("message_received")
        def on_message_received(msg: rtc.ChatMessage):
            try:
                nonlocal last_chat_message_id
                sender = msg.sender_sid or "unknown"
                if not msg.message or not msg.message.strip():
                    logger.warning(f"Empty chat message received from {sender}, ignoring")
                    return
                    
                logger.info(f"Chat message received from {sender}: {msg.message}")
                
                # Set the message ID so we know to send the response as a chat message too
                last_chat_message_id = msg.id
                
                # Add message to chat context
                asyncio.create_task(nhs_agent.add_user_message(msg.message))
                logger.info(f"Added chat message to conversation history: {msg.message[:50]}...")
                
                # Generate reply
                logger.info("Generating reply to chat message...")
                agent.generate_reply()
                logger.info("Response generation triggered from chat message")
            except Exception as e:
                logger.error(f"Error handling chat message: {e}")
                # Try to recover and continue
        
        # Set up metrics collection
        usage_collector = metrics.UsageCollector()
        @agent.on("metrics_collected")
        def on_metrics_collected(mtrcs: metrics.AgentMetrics):
            metrics.log_metrics(mtrcs)
            usage_collector.collect(mtrcs)
            
            # Skip logging for metrics that don't have cost/tokens (like PipelineVADMetrics)
            if not hasattr(mtrcs, 'cost') or not hasattr(mtrcs, 'tokens'):
                return
                
            # Log metrics asynchronously
            asyncio.create_task(nhs_agent.file_logger.log_event(
                "METRICS",
                f"Cost: ${mtrcs.cost:.6f}, Tokens: {mtrcs.tokens}",
                {
                    "cost": mtrcs.cost,
                    "tokens": mtrcs.tokens,
                    "latency_ms": mtrcs.latency_ms if hasattr(mtrcs, 'latency_ms') else 0
                }
            ))
        
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
                
                # Log detailed conversation analysis
                summary_text = (
                    f"Conversation Analysis:\n"
                    f"- User type: {user_type}\n"
                    f"- User ID: {user_id}\n"
                    f"- Total messages: {len(nhs_agent.conversation_history)}\n"
                    f"- User messages: {len(user_messages)}\n"
                    f"- Agent messages: {len(agent_messages)}\n"
                )
                
                # Add conversation topics (from first few user messages)
                if len(user_messages) > 0:
                    topics = ", ".join([msg["text"][:30] + "..." for msg in user_messages[:3]])
                    summary_text += f"- Discussion topics: {topics}\n"
                
                # Add duration information
                if conversation_duration:
                    minutes = int(conversation_duration // 60)
                    seconds = int(conversation_duration % 60)
                    logger.info(f"Conversation duration: {minutes}m {seconds}s")
                    summary_text += f"- Duration: {minutes}m {seconds}s\n"
                
                # Log to structured file
                await nhs_agent.file_logger.log_event(
                    "CONVERSATION_ANALYSIS", 
                    summary_text, 
                    {
                        "user_type": user_type,
                        "user_id": user_id,
                        "total_messages": len(nhs_agent.conversation_history),
                        "user_messages": len(user_messages),
                        "agent_messages": len(agent_messages),
                        "duration_sec": conversation_duration if conversation_duration else None
                    }
                )
                
                # Generate conversation memory summary
                memory_text = "Content saved to memory:\n"
                
                # Get the most recent messages (up to 5) for the memory summary
                recent_exchanges = []
                for i in range(min(5, len(nhs_agent.conversation_history))):
                    if i < len(nhs_agent.conversation_history):
                        msg = nhs_agent.conversation_history[-(i+1)]
                        role = "User" if msg["role"] == "user" else "Assistant"
                        text = msg["text"][:100] + "..." if len(msg["text"]) > 100 else msg["text"]
                        recent_exchanges.append(f"{role}: {text}")
                
                # Add recent exchanges to memory text
                if recent_exchanges:
                    memory_text += "- Recent exchanges:\n  " + "\n  ".join(recent_exchanges) + "\n"
                
                # Add KB usage information
                if nhs_agent.file_logger.knowledge_bases_used:
                    memory_text += f"- Knowledge bases used: {', '.join(nhs_agent.file_logger.knowledge_bases_used)}\n"
                
                # Log memory information
                await nhs_agent.file_logger.log_event("MEMORY_STORAGE", memory_text)
                
                logger.info("=" * 50)
            
            # Log usage metrics
            summary = usage_collector.get_summary()
            logger.info(f"Usage: {summary}")
            
            # Log detailed usage statistics
            usage_stats = {}
            if hasattr(usage_collector, "metrics") and usage_collector.metrics:
                total_tokens = sum(m.tokens for m in usage_collector.metrics)
                total_cost = sum(m.cost for m in usage_collector.metrics)
                avg_latency = sum(m.latency_ms for m in usage_collector.metrics) / len(usage_collector.metrics) if usage_collector.metrics else 0
                
                usage_stats = {
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                    "avg_latency_ms": avg_latency,
                    "call_count": len(usage_collector.metrics)
                }
                
                usage_text = (
                    f"Usage Statistics:\n"
                    f"- Total tokens: {total_tokens}\n"
                    f"- Total cost: ${total_cost:.6f}\n"
                    f"- Average latency: {avg_latency:.2f}ms\n"
                    f"- API calls: {len(usage_collector.metrics)}\n"
                )
                
                await nhs_agent.file_logger.log_event("USAGE_STATISTICS", usage_text, usage_stats)
                
            # Log session completion
            await nhs_agent.file_logger.log_event(
                "SESSION_COMPLETED", 
                "Agent session completed successfully",
                {
                    "log_file": str(nhs_agent.file_logger.log_file)
                }
            )
        
        # Add to shutdown callbacks
        ctx.add_shutdown_callback(end_of_session)
        
        # Start the agent
        agent.start(ctx.room, participant)
        
        # Create welcome message based on user type
        if user_type == "patient":
            welcome_message = create_patient_welcome(user_data)
        else:
            welcome_message = create_doctor_welcome(user_data)
        
        # Log the welcome message
        await nhs_agent.file_logger.log_event("WELCOME_MESSAGE", welcome_message)
        
        # Send welcome message
        await agent.say(welcome_message, allow_interruptions=True)
        
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        # Attempt to explain error to the user
        ctx.error = f"Failed to initialize NHS virtual assistant: {str(e)}"

def create_patient_system_prompt(patient_data: PatientData) -> str:
    """Create system prompt for patient interactions"""
    name = patient_data.full_name if patient_data.full_name else "there"
    
    system_prompt = (
        f"You are an NHS virtual health assistant providing information and support to patients. Your interface with users will be voice. Pretend we're having a conversation, no special formatting or headings, just natural speech."
        f"\n\nNever use asterisks, or other special characters, or emojis, or non-verbal expressions, or special formatting. Just natural speech."
        f"\n\nSometimes it may take a while for you to respond, Let the user know you are finding more information and thinking about the best way to answer the question."
        f"You're currently speaking with a patient named {name}."
        
        f"\n\nGuiding Principles:"
        f"\n1. Medical Accuracy - Only provide information that is medically accurate and from reliable sources. "
        f"ALWAYS cite your sources when giving medical information, mentioning the document title and authors (if provided)."
        f"\n2. Empathetic Support - Be warm, understanding and compassionate. Many patients are anxious or concerned."
        f"\n3. Clear Communication - Use simple, clear language avoiding medical jargon where possible."
        f"\n4. Consent First - Always ask for explicit consent before accessing or discussing personal medical records."
        f"\n5. Source Citation - For EVERY medical claim or statement, explicitly cite the source document and authors (if available)."
        f"\n6. Limitations - Be clear about your limitations. You cannot diagnose, prescribe medication, or schedule appointments."
        
        f"\n\nCitation Requirements:"
        f"\n- For ANY medical or clinical information, you MUST cite the specific document source"
        f"\n- Use the format: 'According to [Document Title],...' or 'According to [Document Title] by [Authors],...' if author information is available"
        f"\n- If multiple sources are used, cite each one separately"
        f"\n- If a user asks for more information about a source, provide only the details you were given"
        f"\n- Never invent or fabricate information beyond what is in the knowledge base"
        f"\n- Never invent or make up document titles or author names"
        f"\n- ALWAYS use the EXACT document titles provided to you - do not abbreviate, summarize, or create similar-sounding alternatives"
        f"\n- NEVER refer to sources using generic titles (like 'Anesthesia Safety') when the actual title is different"
        f"\n- If no information is found, say: 'I don't have specific information about that in my knowledge base. I recommend speaking with your healthcare provider for guidance.'"
        f"\n- Be transparent when synthesizing information from multiple sources by listing all sources used"
        f"\n- When explaining complex medical information from sources, maintain accuracy while using patient-friendly language"
        
        f"\n\nKNOWLEDGE RESTRICTION: You are LIMITED to ONLY providing medical information from the specific NHS knowledge base provided to you."
        f"\n- You MUST NOT use any medical information from your general training data."
        f"\n- You MUST NOT fabricate or invent medical information, statistics, or studies."
        f"\n- You MUST NOT cite medical literature, journals, or authors that aren't explicitly provided to you."
        f"\n- If the NHS knowledge base doesn't contain information about a medical topic, you MUST acknowledge this limitation."
        f"\n- You MUST NEVER provide made-up medical information just to be helpful."
        f"\n- You can only discuss statistical data, mortality rates, or success rates IF this information is explicitly provided in the NHS knowledge base."
        f"\n- When you don't have information, clearly say: 'I don't have that specific information in my NHS knowledge base.'"
        
        f"\n\nConversation Flow:"
        f"\n1. For medical questions, first check the NHS knowledge base. If information is available, provide it with clear source citation."
        f"\n2. If asked about a medical topic NOT in your knowledge base, acknowledge the limitation rather than using your general training data."
        f"\n3. For general questions about NHS services or non-medical topics, you can be more flexible in your responses."
        f"\n4. Maintain a friendly, supportive tone while being truthful about the limits of your knowledge."
        
        f"\n\nThis is a SAFETY-CRITICAL system where accuracy is paramount. Providing fabricated medical information could cause real harm to patients."
    )
    
    return system_prompt

def create_doctor_system_prompt(doctor_data: DoctorData) -> str:
    """Create system prompt for doctor interactions"""
    name = doctor_data.full_name if doctor_data.full_name else "Dr."
    
    system_prompt = (
        f"You are an NHS virtual assistant providing information support for healthcare professionals. Your interface with users will be voice. Pretend we're having a conversation, no special formatting or headings, just natural speech."
        f"\n\nNever use asterisks, or other special characters, or emojis, or non-verbal expressions, or special formatting. Just natural speech."
        f"\n\nYou're currently speaking with a healthcare professional named {name}, specializing in {doctor_data.specialty} at {doctor_data.hospital}."
        
        f"\n\nGuiding Principles:"
        f"\n1. Medical Accuracy - Only provide information that is accurate and from reliable sources."
        f"\n2. Professional Support - Provide evidence-based information relevant to the specialist's needs."
        f"\n3. Source Transparency - Always cite your sources when providing clinical information."
        f"\n4. Conciseness - Be clear and direct, while providing comprehensive information."
        f"\n5. Knowledge Limitations - Be transparent about the limitations of your knowledge."
        
        f"\n\nCitation Requirements:"
        f"\n- For ALL clinical information, you MUST cite the specific BJA Education document source"
        f"\n- Use the format: 'According to [Document Title],...' or 'According to [Document Title] by [Authors],...' if author information is available"
        f"\n- If multiple sources are used, cite each one separately"
        f"\n- If asked about sources, provide only the details you were given"
        f"\n- Never invent information beyond what is provided in your knowledge base"
        f"\n- Never make up document titles or author names that weren't provided to you"
        f"\n- ALWAYS use the EXACT document titles provided to you - do not abbreviate, summarize, or create similar-sounding alternatives"
        f"\n- NEVER refer to sources using generic titles (like 'Anesthesia Safety') when the actual title is different" 
        f"\n- If no information is found, say: 'I don't have specific information about that in my knowledge base. You may want to consult BJA Education at bjaed.org for more resources.'"
        f"\n- Be transparent when synthesizing information from multiple sources by listing all sources used"
        
        f"\n\nKNOWLEDGE RESTRICTION: You are LIMITED to ONLY providing medical information from the specific NHS knowledge base provided to you."
        f"\n- You MUST NOT use any medical information from your general training data."
        f"\n- You MUST NOT fabricate or invent medical information, statistics, or studies."
        f"\n- You MUST NOT cite medical literature, journals, or authors that aren't explicitly provided to you."
        f"\n- If the NHS knowledge base doesn't contain information about a medical topic, you MUST acknowledge this limitation."
        f"\n- You MUST NEVER provide made-up medical information just to be helpful."
        f"\n- You can only discuss statistical data, mortality rates, or success rates IF this information is explicitly provided in the NHS knowledge base."
        f"\n- When you don't have information, clearly state: 'I don't have that specific information in my NHS knowledge base.'"
        
        f"\n\nConversation Flow:"
        f"\n1. For medical questions, first check the NHS knowledge base. If information is available, provide it with clear source citation."
        f"\n2. If asked about a medical topic NOT in your knowledge base, acknowledge the limitation rather than using your general training data."
        f"\n3. For general questions about NHS services or non-medical topics, you can be more flexible in your responses."
        f"\n4. Maintain a professional, supportive tone while being truthful about the limits of your knowledge."
        
        f"\n\nThis is a SAFETY-CRITICAL system where accuracy is paramount. Providing fabricated medical information could cause real harm to patients."
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
    # Download models with: python nhs_agents.py download-files
    # Start agent with: python nhs_agents.py start
    logger.info("Starting NHS Virtual Assistant")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            num_idle_processes=2  # Keep 2 processes warm for better response time
        ),
    )
