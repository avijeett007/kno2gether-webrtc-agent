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
    """Vector database based medical knowledge retrieval system"""
    
    def __init__(self, knowledge_map=None):
        """Initialize with the appropriate knowledge base map based on user type"""
        self.knowledge_map = knowledge_map or DOCTOR_KNOWLEDGE_BASE_MAP
        self._last_ai_analysis = {
            "refined_query": "",
            "selected_kb_ids": [],
            "explanation": ""
        }
    
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
            else:
                logger.warning(f"No relevant knowledge bases found for query: {query[:30]}...")
                
            # Return top 3 knowledge base details
            return relevant_kbs[:3]
        
        except Exception as e:
            logger.error(f"Error finding relevant knowledge bases: {e}")
            return []
    
    def get_kb_info_by_id(self, kb_id: str) -> Dict[str, Any]:
        """Get knowledge base info by ID"""
        kb_map_key = next(iter(self.knowledge_map.keys()))
        knowledgebases = self.knowledge_map[kb_map_key]["knowledgebases"]
        
        for kb in knowledgebases:
            if kb["id"] == kb_id:
                return {
                    "id": kb["id"],
                    "score": 10,  # Assign high score since it was selected by AI
                    "domain": kb["domain"],
                    "document_title": kb.get("document_title", ""),
                    "document_name": kb.get("document_name", ""),
                    "authors": kb.get("authors", "")
                }
        
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
            
            return {
                "text": "\n\n".join(results),
                "source_info": kb_info,
                "score": kb_info["score"]
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge base {collection_id}: {e}")
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
                            
                            # Add source information
                            source_info = result["source_info"]
                            sources_used.append({
                                "id": source_info["id"],
                                "title": source_info.get("document_title", "Unknown"),
                                "authors": source_info.get("authors", ""),
                                "domain": source_info.get("domain", "")
                            })
            else:
                # Fallback to traditional method if AI didn't select any knowledge bases
                logger.info(f"AI didn't select any knowledge bases, falling back to keyword matching")
                
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
                        
                        # Add source information
                        source_info = result["source_info"]
                        sources_used.append({
                            "id": source_info["id"],
                            "title": source_info.get("document_title", "Unknown"),
                            "authors": source_info.get("authors", ""),
                            "domain": source_info.get("domain", "")
                        })
            
            if not all_results:
                return {
                    "text": "I don't have specific information about that in my knowledge base. I can only provide information based on the medical literature I have access to.",
                    "sources": []
                }
            
            # Sort results by relevance score
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Combine texts from all sources
            combined_text = "\n\n".join([result["text"] for result in all_results])
            
            return {
                "text": combined_text,
                "sources": sources_used
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive knowledge: {e}")
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
        
        # Initialize vector memory with user-specific collection
        collection_name = f"{user_data.user_type}_{user_data.user_id.replace('-', '_')}"
        self.memory = VectorMemory(collection_name)
        
        # Initialize knowledge base with appropriate map based on user type
        if user_data.user_type == "patient":
            logger.info(f"Using patient knowledge base map for user {user_data.user_id}")
            self.knowledge_base = KnowledgeBase(PATIENT_KNOWLEDGE_BASE_MAP)
        else:
            logger.info(f"Using doctor knowledge base map for user {user_data.user_id}")
            self.knowledge_base = KnowledgeBase(DOCTOR_KNOWLEDGE_BASE_MAP)
        
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
                    
                    # Determine if this is a medical query
                    is_medical_query = False
                    medical_terms = self._identify_medical_terms(user_message)
                    
                    if len(medical_terms) > 0 or self._is_knowledge_query(user_message):
                        is_medical_query = True
                    
                    # For doctor queries or patient medical queries, get relevant information from knowledgebase
                    if is_medical_query:
                        # This is likely a medical query
                        logger.info(f"Processing medical query: {user_message[:50]}...")
                        
                        # Use AI-powered knowledge retrieval
                        knowledge_result = self.knowledge_base.get_comprehensive_knowledge(user_message)
                        knowledge_text = knowledge_result.get("text", "")
                        sources = knowledge_result.get("sources", [])
                        
                        if knowledge_text:
                            # Get the AI explanation if available
                            ai_explanation = ""
                            try:
                                # Try to access the AI analysis result
                                if hasattr(self.knowledge_base, '_last_ai_analysis'):
                                    ai_explanation = self.knowledge_base._last_ai_analysis.get("explanation", "")
                            except:
                                pass
                                
                            # Format the knowledge text with source information
                            knowledge_context = f"Relevant medical information:\n{knowledge_text}\n\n"
                            
                            # Add source information for citation
                            if sources:
                                knowledge_context += "Source Information:\n"
                                for idx, source in enumerate(sources):
                                    title = source.get("title", "Unknown")
                                    authors = source.get("authors", "")
                                    domain = source.get("domain", "")
                                    knowledge_context += f"Source {idx+1}: {title} by {authors} ({domain})\n"
                                
                                # Add AI explanation if available
                                if ai_explanation:
                                    knowledge_context += f"\nReason for source selection: {ai_explanation}\n"
                            
                            context_parts.append(knowledge_context)
                            
                            # Add reminder to cite sources with specific instructions
                            if sources:
                                reminder = (
                                    "CRITICAL: When answering this medical question, you MUST cite each source you use.\n"
                                    "- Begin your response with 'According to [Document Title] by [Authors],'\n"
                                    "- For each separate piece of information from different sources, clearly indicate the source\n"
                                    "- If synthesizing from multiple sources, list each source: 'Based on information from [Source 1], [Source 2], and [Source 3]...'\n"
                                    "- Always refer to the source by the exact document title and authors\n"
                                    "- If the user asks about the source of your information, provide the complete document title and authors\n"
                                    "This ensures the user knows the information comes from authoritative medical literature and not from your general knowledge."
                                )
                                context_parts.append(reminder)
                    
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
    
    async def send_chat_message(self, room: rtc.Room, message: str):
        """Send a chat message to all participants in the room"""
        try:
            if not room or not message:
                logger.warning("Cannot send empty message or no room provided")
                return False
                
            # Send message to all participants in the room
            await room.local_participant.publish_data(message.encode('utf-8'), rtc.DataPacketKind.RELIABLE)
            logger.info(f"Sent chat message: {message[:50]}...")
            
            # Add message to conversation history
            self.add_agent_message(message)
            return True
            
        except Exception as e:
            logger.error(f"Error sending chat message: {e}")
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
            nhs_agent.add_user_message(content)
        
        @agent.on("agent_speech_committed")
        def on_agent_speech_committed(msg: llm.ChatMessage):
            nonlocal last_chat_message_id
            content = msg.content
            logger.info(f"Agent speech committed: {content[:50]}...")
            nhs_agent.add_agent_message(content)
            
            # If this is a response to a chat message, also send it as a chat message
            if last_chat_message_id is not None:
                asyncio.create_task(nhs_agent.send_chat_message(ctx.room, content))
                last_chat_message_id = None

                
        @chat.on("message_received")
        def on_message_received(msg: rtc.ChatMessage):
            if not msg.message or not msg.message.strip():
                logger.warning(f"Empty chat message received, ignoring")
                return
            
            logger.info(f"Chat message received : {msg.message}")
            
            # Set the message ID so we know to send the response as a chat message too
            last_chat_message_id = msg.id
            nhs_agent.add_user_message(msg.message)
            logger.info(f"Added chat message to conversation history: {msg.message[:50]}...")

            # Generate reply
            logger.info("Generating reply to chat message...")
            agent.generate_reply()
            logger.info("Response generation triggered from chat message")
                
       
                   
        

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
                nhs_agent.add_user_message(msg.message)
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
        f"ALWAYS cite your sources when giving medical information, mentioning the document title and authors."
        f"\n2. Empathetic Support - Be warm, understanding and compassionate. Many patients are anxious or concerned."
        f"\n3. Clear Communication - Use simple, clear language avoiding medical jargon where possible."
        f"\n4. Consent First - Always ask for explicit consent before accessing or discussing personal medical records."
        f"\n5. Source Citation - For EVERY medical claim or statement, explicitly cite the source document and authors."
        f"\n6. Limitations - Be clear about your limitations. You cannot diagnose, prescribe medication, or schedule appointments."

        f"\n\nImportant Guidelines:"
        f"\n- Don't make diagnoses or suggest treatments that haven't been prescribed by a doctor"
        f"\n- All medical information MUST be attributed to specific medical literature sources in your knowledge base"
        f"\n- Begin responses to medical questions with 'According to [Document Title] by [Authors],...'"
        f"\n- Don't offer services you can't provide such as booking appointments or issuing prescriptions"
        f"\n- Always recommend consulting with a healthcare professional for specific medical issues"
        f"\n- Be respectful of the patient's emotional state and concerns"
        f"\n- For urgent medical issues, advise contacting emergency services (999) or NHS 111"
        f"\n- If you don't know something, be honest rather than speculating"
        f"\n- Save important but non-sensitive information for future conversations using the save_to_memory function"
        f"\n- ALWAYS cite your information sources using the pattern: 'According to [Document Title] by [Authors]...'"
        f"\n- If the user asks about sources, provide the full document title and authors"
        
        f"\n\nKnowledge Base Usage:"
        f"\n- Your knowledge comes exclusively from vector database collections of medical literature"
        f"\n- You do not have internet access and cannot search online for information"
        f"\n- You can only reference documents that are explicitly provided in your context"
        f"\n- When you don't know something, or cannot find the information from knowledge base, state clearly that you don't have that information in your knowledge base"
        f"\n- Never make up information or cite documents that aren't specified in your context"
        f"\n- Translate complex medical information into patient-friendly language while maintaining accuracy"
        
        f"\n\nFunctionality Available:"
        f"\n- You can access medical knowledge bases for general health information"
        f"\n- You can access the patient's medical records ONLY after explicit consent"
        f"\n- You can explain NHS services and standard procedures"
        f"\n- You can provide general health advice backed by sources in your knowledge base"
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
        
        f"\n\nCitation Requirements:"
        f"\n- For ANY medical or clinical information, you MUST cite the specific document source"
        f"\n- Use the format: 'According to [Document Title] by [Authors], ...'"
        f"\n- If multiple sources are used, cite each one separately"
        f"\n- If a user asks for more information about a source, provide the full document title and authors"
        f"\n- Never invent or fabricate information beyond what is in the knowledge base"
        f"\n- If no information is found, say: 'I don't have specific information about that in my knowledge base. I recommend speaking with your healthcare provider for guidance.'"
        f"\n- Be transparent when synthesizing information from multiple sources by listing all sources used"
        f"\n- When explaining complex medical information from sources, maintain accuracy while using patient-friendly language"
    )
    
    return system_prompt

def create_doctor_system_prompt(doctor_data: DoctorData) -> str:
    """Create system prompt for doctor interactions"""
    name = doctor_data.full_name if doctor_data.full_name else "Doctor"
    specialty = doctor_data.specialty if doctor_data.specialty else "medicine"
    
    system_prompt = (
        f"You are an NHS virtual clinical assistant supporting healthcare professionals. Your interface with users will be voice. Pretend we're having a conversation, no special formatting or headings, just natural speech."
        f"\n\nNever use asterisks(** for Bold text etc or dashes(-) for bullet points), or other special characters, or emojis, or non-verbal expressions, or special formatting. Just natural speech."
        f"\n\nSometimes it may take a while for you to respond, Let the user know you are finding more information and thinking about the best way to answer the question."
        f"You're currently speaking with {name}, a healthcare professional specializing in {specialty}."
        
        f"\n\nGuiding Principles:"
        f"\n1. Clinical Accuracy - Provide medically accurate information backed by evidence. ALWAYS cite guidelines and sources with document title and authors."
        f"\n2. Professional Support - Maintain a professional and efficient manner befitting clinical discussions."
        f"\n3. Detail-Oriented - Provide comprehensive information when discussing clinical topics, including relevant medical details."
        f"\n4. Evidence-Based - Base all information on current clinical evidence and established medical guidelines."
        f"\n5. Source Citation - For EVERY medical claim or statement, explicitly cite the source document and authors."
        f"\n6. Limitations - Be clear about your limitations. You cannot access patient records or make clinical decisions."

        f"\n\nImportant Guidelines:"
        f"\n- Focus on providing clinical information ONLY from the medical literature in your knowledge base"
        f"\n- You cannot access specific patient information or records"
        f"\n- All medical information MUST be attributed to specific BJA Education documents"
        f"\n- Begin responses to medical questions with 'According to [Document Title] by [Authors],...'"
        f"\n- Discuss general case scenarios but cannot provide specific patient advice"
        f"\n- Be precise with medical terminology when appropriate for professional discussion"
        f"\n- Acknowledge when information might be outside your knowledge base"
        f"\n- Save important but non-sensitive information for future conversations using the save_to_memory function"
        f"\n- ALWAYS cite your information sources using the pattern: 'According to [Document Title] by [Authors]...'"
        f"\n- If the doctor asks about sources, provide the full document title and authors"
        
        f"\n\nKnowledge Base Usage:"
        f"\n- Your knowledge comes exclusively from vector database collections of BJA Education documents"
        f"\n- You do not have internet access and cannot search online for information"
        f"\n- You can only reference documents that are explicitly provided in your context"
        f"\n- When you don't know something, state clearly that you don't have that information in your knowledge base"
        f"\n- Never make up information or cite documents that aren't specified in your context"
        
        f"\n\nFunctionality Available:"
        f"\n- You can access medical knowledge bases containing BJA Education articles"
        f"\n- You can explain clinical procedures and protocols based on these sources"
        f"\n- You can provide information on best practices from medical literature in your knowledge base"
        f"\n- You can assist with general clinical questions by citing relevant knowledge base documents"
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
        
        f"\n\nCitation Requirements:"
        f"\n- For ALL clinical information, you MUST cite the specific BJA Education document source"
        f"\n- Use the format: 'According to [Document Title] by [Authors], ...'"
        f"\n- If multiple sources are used, cite each one separately"
        f"\n- If asked about sources, provide the complete document title and authors"
        f"\n- Never invent information beyond what is provided in your knowledge base"
        f"\n- If no information is found, say: 'I don't have specific information about that in my knowledge base. You may want to consult BJA Education at bjaed.org for more resources.'"
        f"\n- Be transparent when synthesizing information from multiple sources by listing all sources used"
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
