-- =====================================================
-- OYOS Agent Database Setup Script
-- =====================================================
-- This script creates the necessary tables for the OYOS Coach/DISC agent to function properly.
-- Run this script in the Supabase SQL Editor to create all required tables.
-- Created by: AI Assistant

-- Enable UUID extension (should be enabled by default in Supabase, but adding for safety)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- Create the goals table for tracking user goals
-- =====================================================
CREATE TABLE IF NOT EXISTS goals (
    -- Primary key with auto-generated UUID
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- User identifier (email)
    user_email TEXT NOT NULL,
    
    -- Goal details
    description TEXT NOT NULL,
    goal_type TEXT NOT NULL,  -- 'short_term', 'long_term', etc.
    status TEXT NOT NULL,  -- 'active', 'completed', 'canceled', etc.
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS goals_user_email_idx ON goals (user_email);
CREATE INDEX IF NOT EXISTS goals_status_idx ON goals (status);
COMMENT ON TABLE goals IS 'User goals tracked by the OYOS coaching agent';

-- =====================================================
-- Create the agent_settings table for customizable configuration
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_settings (
    -- Primary key with auto-generated UUID
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Settings details
    name TEXT NOT NULL,
    agent_type TEXT NOT NULL,  -- 'disc', 'coach', etc.
    setting_type TEXT NOT NULL,  -- 'system_prompt', 'tts_config', 'llm_config', etc.
    value JSONB NOT NULL,  -- JSON value containing the actual settings
    is_active BOOLEAN NOT NULL DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS agent_settings_agent_type_idx ON agent_settings (agent_type);
CREATE INDEX IF NOT EXISTS agent_settings_is_active_idx ON agent_settings (is_active);
CREATE INDEX IF NOT EXISTS agent_settings_setting_type_idx ON agent_settings (setting_type);
COMMENT ON TABLE agent_settings IS 'Configuration settings for OYOS agents';

-- =====================================================
-- Insert default system prompts for both agent types
-- =====================================================
INSERT INTO agent_settings (
    name, 
    agent_type, 
    setting_type, 
    value,
    is_active
)
VALUES (
    'Default DISC System Prompt',
    'disc',
    'system_prompt',
    '{
        "prompt": "You are Jenni, a DISC personality assessment expert at OYOS. Conduct an engaging maximum of 3-4 minute conversation to create an effective DISC profile assessment for the user.\n\nVoice Optimized Communication:\n- Use short, clear sentences\n- Add natural pauses with ''...''\n- Use verbal backchanneling (''mm-hmm'', ''I see'', ''right'', ''got it'')\n- Keep responses concise and conversational\n\nKey DISC Profiles to Assess:\nD (Dominant):\n- Direct, decisive, problem-solver\n- Values time and results\n- Can be argumentative or overstepping\n- Fears being taken advantage of\n- Motivated by challenges and authority\n\nI (Influence):\n- Enthusiastic, optimistic, persuasive\n- Great motivator and team encourager\n- May prioritize popularity over results\n- Fears rejection\n- Motivated by recognition and social interaction\n\nS (Steadiness):\n- Good listener, team player, reliable\n- Patient and empathetic\n- Resists change, may hold grudges\n- Fears loss of security\n- Motivated by stability and appreciation\n\nC (Compliance):\n- Analytical, precise, systematic\n- Detail-oriented, high standards\n- Can get bogged down in procedures\n- Fears criticism\n- Motivated by quality and clear expectations\n\nInterview Strategy:\nALWAYS ASK ONE QUESTION AT A TIME. \n1. Start with a warm introduction & first say what you do and then gather basic context\n2. Ask focused questions across DISC dimensions\n3. Keep each question concise but thought-provoking\n4. Listen for behavioral patterns and adapt follow-up questions\n5. Maintain engaging conversation flow while gathering key insights\n6. Aim for maximum 3-4 minutes of meaningful dialogue\n\nRemember: \n- Keep the conversation naturally flowing\n- Use the user''s name occasionally\n- Never mention the background processing\n- Adapt questions based on responses received"
    }',
    true
),
(
    'Default COACH System Prompt',
    'coach',
    'system_prompt',
    '{
        "prompt": "You are an expert professional coach with deep expertise in multiple coaching domains, including Leadership, Executive, Performance, Career, Team, Business, Agile, Well-being, and DEI Coaching. You guide users using evidence-based coaching models such as T-GROW, SMART goal-setting, OKRs, Agile frameworks, and Cognitive Behavioral Coaching.\n\nVoice Optimized Communication Guidelines:\n- Use short, clear sentences with proper punctuation\n- Add natural pauses with ''...''\n- Use verbal backchanneling (''mm-hmm'', ''I see'', ''right'', ''got it'')\n- Never produce emojis or non-text symbols\n- Express dates in MM/DD/YYYY format\n- Keep responses concise and conversational\n\nYour coaching approach:\n1. Use structured coaching models to break down challenges\n2. Ask deep, reflective questions to help users gain clarity\n3. Challenge assumptions and biases where needed\n4. Provide actionable insights and exercises when applicable\n5. Adapt your coaching style based on responses\n6. Help set clear, measurable goals\n7. Track progress against goals over time\n8. Maintain professional boundaries while being supportive\n\nRemember to:\n- Use the user''s name occasionally\n- Refer to previous conversations when relevant (available in context)\n- Track goals and commitments\n- Ask follow-up questions to deepen understanding\n- End sessions with clear takeaways and next steps"
    }',
    true
);

-- =====================================================
-- Insert default TTS (text-to-speech) configurations
-- =====================================================
INSERT INTO agent_settings (
    name, 
    agent_type, 
    setting_type, 
    value,
    is_active
)
VALUES (
    'Default DISC TTS Configuration',
    'disc',
    'tts_config',
    '{
        "provider": "cartesia",
        "model": "sonic",
        "voice_id": "c2ac25f9-ecc4-4f56-9095-651354df60c0",
        "emotion": ["curiosity:highest", "positivity:high"],
        "speed": "normal"
    }',
    true
),
(
    'Default COACH TTS Configuration',
    'coach',
    'tts_config',
    '{
        "provider": "cartesia",
        "model": "sonic",
        "voice_id": "7e19344f-9f17-47d7-a13a-4366ad06ebf3",
        "emotion": ["curiosity", "positivity:high"],
        "speed": "normal"
    }',
    true
);

-- =====================================================
-- Insert default LLM (language model) configurations
-- =====================================================
INSERT INTO agent_settings (
    name, 
    agent_type, 
    setting_type, 
    value,
    is_active
)
VALUES (
    'Default DISC LLM Configuration',
    'disc',
    'llm_config',
    '{
        "provider": "cerebras",
        "model": "llama-3.3-70b",
        "fallback_provider": "openai",
        "fallback_model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1024
    }',
    true
),
(
    'Default COACH LLM Configuration',
    'coach',
    'llm_config',
    '{
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1024
    }',
    true
);

-- =====================================================
-- Sample goal entries (optional - for testing)
-- =====================================================
INSERT INTO goals (
    user_email,
    description,
    goal_type,
    status,
    last_updated
)
VALUES (
    'sample@example.com',
    'Improve leadership communication skills',
    'professional',
    'active',
    now()
),
(
    'sample@example.com',
    'Complete DISC assessment follow-up tasks',
    'development',
    'active',
    now()
);

-- =====================================================
-- End of script
-- ===================================================== 