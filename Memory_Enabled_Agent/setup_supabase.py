#!/usr/bin/env python
"""
Supabase Setup Script for OYOS Agent

This script sets up the required Supabase tables for the OYOS coach/DISC agent:
- goals: For tracking user goals
- agent_settings: For customizable agent configuration

Usage:
    python setup_supabase.py --url SUPABASE_URL --key SUPABASE_KEY

Or set environment variables:
    SUPABASE_URL and SUPABASE_KEY

Note: This script requires Supabase tables to be created manually or via the Supabase dashboard.
      Instructions are provided in the README.md file.
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("supabase_setup")

try:
    from supabase import create_client, Client
except ImportError:
    logger.error("Supabase client not installed. Please run: pip install supabase>=2.0.0")
    sys.exit(1)

# Default configuration values
DEFAULT_SYSTEM_PROMPTS = {
    "disc": """You are Jenni, a DISC personality assessment expert at OYOS. Conduct an engaging maximum of 3-4 minute conversation to create an effective DISC profile assessment for the user.

Voice Optimized Communication:
- Use short, clear sentences
- Add natural pauses with '...'
- Use verbal backchanneling ('mm-hmm', 'I see', 'right', 'got it')
- Keep responses concise and conversational

Key DISC Profiles to Assess:
D (Dominant):
- Direct, decisive, problem-solver
- Values time and results
- Can be argumentative or overstepping
- Fears being taken advantage of
- Motivated by challenges and authority

I (Influence):
- Enthusiastic, optimistic, persuasive
- Great motivator and team encourager
- May prioritize popularity over results
- Fears rejection
- Motivated by recognition and social interaction

S (Steadiness):
- Good listener, team player, reliable
- Patient and empathetic
- Resists change, may hold grudges
- Fears loss of security
- Motivated by stability and appreciation

C (Compliance):
- Analytical, precise, systematic
- Detail-oriented, high standards
- Can get bogged down in procedures
- Fears criticism
- Motivated by quality and clear expectations

Interview Strategy:
ALWAYS ASK ONE QUESTION AT A TIME. 
1. Start with a warm introduction & first say what you do and then gather basic context
2. Ask focused questions across DISC dimensions
3. Keep each question concise but thought-provoking
4. Listen for behavioral patterns and adapt follow-up questions
5. Maintain engaging conversation flow while gathering key insights
6. Aim for maximum 3-4 minutes of meaningful dialogue

Remember: 
- Keep the conversation naturally flowing
- Use the user's name occasionally
- Never mention the background processing
- Adapt questions based on responses received""",

    "coach": """You are an expert professional coach with deep expertise in multiple coaching domains, including Leadership, Executive, Performance, Career, Team, Business, Agile, Well-being, and DEI Coaching. You guide users using evidence-based coaching models such as T-GROW, SMART goal-setting, OKRs, Agile frameworks, and Cognitive Behavioral Coaching.

Voice Optimized Communication Guidelines:
- Use short, clear sentences with proper punctuation
- Add natural pauses with '...'
- Use verbal backchanneling ('mm-hmm', 'I see', 'right', 'got it')
- Never produce emojis or non-text symbols
- Express dates in MM/DD/YYYY format
- Keep responses concise and conversational

Your coaching approach:
1. Use structured coaching models to break down challenges
2. Ask deep, reflective questions to help users gain clarity
3. Challenge assumptions and biases where needed
4. Provide actionable insights and exercises when applicable
5. Adapt your coaching style based on responses
6. Help set clear, measurable goals
7. Track progress against goals over time
8. Maintain professional boundaries while being supportive

Remember to:
- Use the user's name occasionally
- Refer to previous conversations when relevant (available in context)
- Track goals and commitments
- Ask follow-up questions to deepen understanding
- End sessions with clear takeaways and next steps"""
}

# Default TTS configuration
DEFAULT_TTS_CONFIG = {
    "disc": {
        "provider": "cartesia",
        "model": "sonic",
        "voice_id": "c2ac25f9-ecc4-4f56-9095-651354df60c0",
        "emotion": ["curiosity:highest", "positivity:high"],
        "speed": "normal"
    },
    "coach": {
        "provider": "cartesia",
        "model": "sonic",
        "voice_id": "7e19344f-9f17-47d7-a13a-4366ad06ebf3",
        "emotion": ["curiosity", "positivity:high"],
        "speed": "normal"
    }
}

# Default LLM configuration
DEFAULT_LLM_CONFIG = {
    "disc": {
        "provider": "cerebras",
        "model": "llama-3.3-70b",
        "fallback_provider": "openai",
        "fallback_model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1024
    },
    "coach": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1024
    }
}

def check_tables_exist(supabase: Client) -> bool:
    """
    Check if the required tables exist in Supabase
    
    Args:
        supabase: Supabase client
        
    Returns:
        bool: True if both tables exist
    """
    tables_exist = True
    
    # Check goals table
    try:
        supabase.table("goals").select("*").limit(1).execute()
        logger.info("Goals table exists")
    except Exception as e:
        logger.warning(f"Goals table does not exist: {e}")
        tables_exist = False
    
    # Check agent_settings table
    try:
        supabase.table("agent_settings").select("*").limit(1).execute()
        logger.info("Agent settings table exists")
    except Exception as e:
        logger.warning(f"Agent settings table does not exist: {e}")
        tables_exist = False
    
    return tables_exist

def print_table_creation_instructions():
    """Print instructions for manually creating the required tables"""
    logger.info("The required tables need to be created manually via the Supabase dashboard.")
    logger.info("\n" + "-"*80)
    logger.info("Please create the following tables in your Supabase project:")
    
    # Goals table
    logger.info("\n1. Table name: goals")
    logger.info("   Columns:")
    logger.info("   - id: uuid (Primary Key, Default: uuid_generate_v4())")
    logger.info("   - user_email: text (Not Null)")
    logger.info("   - description: text (Not Null)")
    logger.info("   - goal_type: text (Not Null)")
    logger.info("   - status: text (Not Null)")
    logger.info("   - created_at: timestamptz (Not Null)")
    logger.info("   - last_updated: timestamptz (Not Null)")
    
    # Agent settings table
    logger.info("\n2. Table name: agent_settings")
    logger.info("   Columns:")
    logger.info("   - id: uuid (Primary Key, Default: uuid_generate_v4())")
    logger.info("   - name: text (Not Null)")
    logger.info("   - agent_type: text (Not Null)")
    logger.info("   - setting_type: text (Not Null)")
    logger.info("   - value: jsonb (Not Null)")
    logger.info("   - is_active: boolean (Not Null, Default: true)")
    logger.info("   - created_at: timestamptz (Not Null, Default: now())")
    logger.info("   - updated_at: timestamptz (Not Null, Default: now())")
    
    logger.info("\nCreate indexes (optional but recommended):")
    logger.info("   - goals_user_email_idx ON goals (user_email)")
    logger.info("   - goals_status_idx ON goals (status)")
    logger.info("   - agent_settings_agent_type_idx ON agent_settings (agent_type)")
    logger.info("   - agent_settings_is_active_idx ON agent_settings (is_active)")
    logger.info("   - agent_settings_setting_type_idx ON agent_settings (setting_type)")
    
    logger.info("\nAfter creating the tables, run this script again to insert default settings.")
    logger.info("-"*80 + "\n")

def insert_default_settings(supabase: Client) -> bool:
    """
    Insert default settings into the agent_settings table
    
    Args:
        supabase: Supabase client
        
    Returns:
        bool: Success status
    """
    success = True
    
    # Insert default system prompts
    try:
        logger.info("Adding default system prompts...")
        
        for agent_type, prompt in DEFAULT_SYSTEM_PROMPTS.items():
            # Check if a system prompt already exists for this agent type
            existing = supabase.table("agent_settings").select("*").eq("agent_type", agent_type).eq("setting_type", "system_prompt").eq("is_active", True).execute()
            
            if existing.data:
                logger.info(f"System prompt for {agent_type} agent already exists")
            else:
                # Insert default system prompt
                supabase.table("agent_settings").insert({
                    "name": f"Default {agent_type.upper()} System Prompt",
                    "agent_type": agent_type,
                    "setting_type": "system_prompt",
                    "value": json.dumps({"prompt": prompt}),
                    "is_active": True
                }).execute()
                logger.info(f"Added default system prompt for {agent_type} agent")
    except Exception as e:
        logger.error(f"Failed to add default system prompts: {e}")
        success = False
    
    # Insert default TTS configurations
    try:
        logger.info("Adding default TTS configurations...")
        
        for agent_type, config in DEFAULT_TTS_CONFIG.items():
            # Check if TTS config already exists for this agent type
            existing = supabase.table("agent_settings").select("*").eq("agent_type", agent_type).eq("setting_type", "tts_config").eq("is_active", True).execute()
            
            if existing.data:
                logger.info(f"TTS configuration for {agent_type} agent already exists")
            else:
                # Insert default TTS config
                supabase.table("agent_settings").insert({
                    "name": f"Default {agent_type.upper()} TTS Configuration",
                    "agent_type": agent_type,
                    "setting_type": "tts_config",
                    "value": json.dumps(config),
                    "is_active": True
                }).execute()
                logger.info(f"Added default TTS configuration for {agent_type} agent")
    except Exception as e:
        logger.error(f"Failed to add default TTS configurations: {e}")
        success = False
    
    # Insert default LLM configurations
    try:
        logger.info("Adding default LLM configurations...")
        
        for agent_type, config in DEFAULT_LLM_CONFIG.items():
            # Check if LLM config already exists for this agent type
            existing = supabase.table("agent_settings").select("*").eq("agent_type", agent_type).eq("setting_type", "llm_config").eq("is_active", True).execute()
            
            if existing.data:
                logger.info(f"LLM configuration for {agent_type} agent already exists")
            else:
                # Insert default LLM config
                supabase.table("agent_settings").insert({
                    "name": f"Default {agent_type.upper()} LLM Configuration",
                    "agent_type": agent_type,
                    "setting_type": "llm_config",
                    "value": json.dumps(config),
                    "is_active": True
                }).execute()
                logger.info(f"Added default LLM configuration for {agent_type} agent")
    except Exception as e:
        logger.error(f"Failed to add default LLM configurations: {e}")
        success = False
    
    return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Set up Supabase tables for OYOS agent")
    parser.add_argument("--url", help="Supabase URL")
    parser.add_argument("--key", help="Supabase API key")
    
    args = parser.parse_args()
    
    # Get Supabase credentials
    supabase_url = args.url or os.environ.get("SUPABASE_URL")
    supabase_key = args.key or os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase URL and API key are required")
        logger.error("Please provide them as arguments or set environment variables")
        logger.error("Usage: python setup_supabase.py --url SUPABASE_URL --key SUPABASE_KEY")
        sys.exit(1)
    
    # Connect to Supabase
    try:
        logger.info(f"Connecting to Supabase at {supabase_url}...")
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        sys.exit(1)
    
    # Check if tables exist
    tables_exist = check_tables_exist(supabase)
    if not tables_exist:
        print_table_creation_instructions()
        logger.warning("Tables not found. Please create them manually and run this script again.")
        sys.exit(1)
    
    # Insert default settings
    if not insert_default_settings(supabase):
        logger.warning("Failed to insert all default settings")
        sys.exit(1)
    
    logger.info("Supabase setup completed successfully")

if __name__ == "__main__":
    main() 