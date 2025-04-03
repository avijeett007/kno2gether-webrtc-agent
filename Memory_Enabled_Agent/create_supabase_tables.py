#!/usr/bin/env python
"""
Supabase Tables Creation Script

This script creates the required tables in Supabase using REST API calls
instead of SQL commands. Use this as an alternative to manual table creation.

Usage:
    python create_supabase_tables.py --url SUPABASE_URL --key SUPABASE_KEY

Or set environment variables:
    SUPABASE_URL and SUPABASE_KEY
"""

import os
import sys
import json
import argparse
import logging
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("supabase_tables_creator")

try:
    from supabase import create_client, Client
except ImportError:
    logger.error("Supabase client not installed. Please run: pip install supabase>=2.0.0")
    sys.exit(1)

def create_tables_via_rest(base_url: str, api_key: str) -> bool:
    """
    Create the required tables using REST API calls
    
    Args:
        base_url: Supabase URL
        api_key: Supabase API key
        
    Returns:
        bool: Success status
    """
    success = True
    
    # Set up headers for REST API calls
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    # Define tables to create
    tables = [
        {
            "name": "goals",
            "rpc_endpoint": f"{base_url}/rest/v1/rpc/create_goals_table",
            "body": {
                "table_name": "goals",
                "columns": [
                    {"name": "id", "type": "uuid", "primary": True, "default_value": "uuid_generate_v4()"},
                    {"name": "user_email", "type": "text", "nullable": False},
                    {"name": "description", "type": "text", "nullable": False},
                    {"name": "goal_type", "type": "text", "nullable": False},
                    {"name": "status", "type": "text", "nullable": False},
                    {"name": "created_at", "type": "timestamptz", "nullable": False, "default_value": "now()"},
                    {"name": "last_updated", "type": "timestamptz", "nullable": False, "default_value": "now()"}
                ]
            }
        },
        {
            "name": "agent_settings",
            "rpc_endpoint": f"{base_url}/rest/v1/rpc/create_agent_settings_table",
            "body": {
                "table_name": "agent_settings",
                "columns": [
                    {"name": "id", "type": "uuid", "primary": True, "default_value": "uuid_generate_v4()"},
                    {"name": "name", "type": "text", "nullable": False},
                    {"name": "agent_type", "type": "text", "nullable": False},
                    {"name": "setting_type", "type": "text", "nullable": False},
                    {"name": "value", "type": "jsonb", "nullable": False},
                    {"name": "is_active", "type": "boolean", "nullable": False, "default_value": "true"},
                    {"name": "created_at", "type": "timestamptz", "nullable": False, "default_value": "now()"},
                    {"name": "updated_at", "type": "timestamptz", "nullable": False, "default_value": "now()"}
                ]
            }
        }
    ]
    
    # First, check if the create_table function exists in the database
    logger.info("Setting up table creation functions in PostgreSQL...")
    
    # PostgreSQL function creation query (need to execute as SQL)
    function_query = """
    CREATE OR REPLACE FUNCTION create_goals_table()
    RETURNS VOID AS $$
    BEGIN
        CREATE TABLE IF NOT EXISTS goals (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_email TEXT NOT NULL,
            description TEXT NOT NULL,
            goal_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_updated TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        
        CREATE INDEX IF NOT EXISTS goals_user_email_idx ON goals (user_email);
        CREATE INDEX IF NOT EXISTS goals_status_idx ON goals (status);
    END;
    $$ LANGUAGE plpgsql;
    
    CREATE OR REPLACE FUNCTION create_agent_settings_table()
    RETURNS VOID AS $$
    BEGIN
        CREATE TABLE IF NOT EXISTS agent_settings (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            name TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            setting_type TEXT NOT NULL,
            value JSONB NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT true,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        
        CREATE INDEX IF NOT EXISTS agent_settings_agent_type_idx ON agent_settings (agent_type);
        CREATE INDEX IF NOT EXISTS agent_settings_is_active_idx ON agent_settings (is_active);
        CREATE INDEX IF NOT EXISTS agent_settings_setting_type_idx ON agent_settings (setting_type);
    END;
    $$ LANGUAGE plpgsql;
    """
    
    # Try to install the SQL functions using direct REST API
    try:
        # This is a direct SQL endpoint for Supabase
        sql_url = f"{base_url}/rest/v1/sql"
        response = requests.post(
            sql_url,
            headers=headers,
            json={"query": function_query}
        )
        
        if response.status_code in (200, 201):
            logger.info("✅ SQL functions created successfully")
        else:
            logger.error(f"❌ Failed to create SQL functions: {response.text}")
            logger.warning("Table creation via REST API might not work, but we'll try anyway")
    except Exception as e:
        logger.error(f"❌ Failed to create SQL functions: {e}")
        logger.warning("Table creation via REST API might not work, but we'll try anyway")
    
    # Now try to call each function to create the tables
    for table in tables:
        logger.info(f"Creating table: {table['name']}...")
        
        try:
            # First check if table exists using the Supabase client
            supabase = create_client(base_url, api_key)
            try:
                response = supabase.table(table['name']).select("*").limit(1).execute()
                logger.info(f"✅ Table {table['name']} already exists, skipping creation")
                continue
            except Exception:
                logger.info(f"Table {table['name']} does not exist, creating...")
            
            # Call the RPC function to create the table
            response = requests.post(
                table['rpc_endpoint'],
                headers=headers,
                json={}
            )
            
            if response.status_code in (200, 201, 204):
                logger.info(f"✅ Table {table['name']} created successfully")
            else:
                logger.error(f"❌ Failed to create table {table['name']}: {response.text}")
                logger.info(f"Status code: {response.status_code}")
                success = False
        except Exception as e:
            logger.error(f"❌ Failed to create table {table['name']}: {e}")
            success = False
    
    if not success:
        logger.warning("""
        REST API table creation might not work in all Supabase instances.
        Please try creating the tables manually through the Supabase dashboard.
        See the README for detailed schema information.
        """)
    
    return success

def check_tables_exist(supabase: Client) -> Dict[str, bool]:
    """
    Check which tables exist in Supabase
    
    Args:
        supabase: Supabase client
        
    Returns:
        Dict[str, bool]: Dictionary of table names and existence status
    """
    tables_status = {
        "goals": False,
        "agent_settings": False
    }
    
    for table_name in tables_status.keys():
        try:
            supabase.table(table_name).select("*").limit(1).execute()
            tables_status[table_name] = True
            logger.info(f"✅ Table {table_name} exists")
        except Exception as e:
            logger.warning(f"❓ Table {table_name} does not exist: {e}")
    
    return tables_status

def main():
    """Main function"""
    # Load environment variables from .env file if present
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Create required Supabase tables")
    parser.add_argument("--url", help="Supabase URL")
    parser.add_argument("--key", help="Supabase API key")
    parser.add_argument("--force", action="store_true", help="Force table creation even if they already exist")
    
    args = parser.parse_args()
    
    # Get Supabase credentials
    supabase_url = args.url or os.environ.get("SUPABASE_URL")
    supabase_key = args.key or os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase URL and API key are required")
        logger.error("Please provide them as arguments or set environment variables")
        logger.error("Usage: python create_supabase_tables.py --url SUPABASE_URL --key SUPABASE_KEY")
        sys.exit(1)
    
    # Connect to Supabase to check existing tables
    try:
        logger.info(f"Connecting to Supabase at {supabase_url}...")
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        sys.exit(1)
    
    # Check which tables already exist
    if not args.force:
        tables_status = check_tables_exist(supabase)
        
        if all(tables_status.values()):
            logger.info("All required tables already exist!")
            should_continue = input("Do you want to continue anyway and try recreating tables? (y/n): ")
            if should_continue.lower() != 'y':
                logger.info("Exiting without creating tables.")
                sys.exit(0)
    
    # Create tables using REST API
    logger.info("Creating tables using REST API...")
    success = create_tables_via_rest(supabase_url, supabase_key)
    
    if success:
        logger.info("✅ All tables created successfully!")
    else:
        logger.warning("""
        Some tables could not be created using REST API.
        Please try creating them manually via the Supabase dashboard.
        See the README.md for table schema details.
        
        Alternative command:
        - Go to the SQL Editor in Supabase dashboard
        - Run the following SQL commands:
        
        CREATE TABLE IF NOT EXISTS goals (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_email TEXT NOT NULL,
            description TEXT NOT NULL,
            goal_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_updated TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        
        CREATE INDEX IF NOT EXISTS goals_user_email_idx ON goals (user_email);
        CREATE INDEX IF NOT EXISTS goals_status_idx ON goals (status);
        
        CREATE TABLE IF NOT EXISTS agent_settings (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            name TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            setting_type TEXT NOT NULL,
            value JSONB NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT true,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        
        CREATE INDEX IF NOT EXISTS agent_settings_agent_type_idx ON agent_settings (agent_type);
        CREATE INDEX IF NOT EXISTS agent_settings_is_active_idx ON agent_settings (is_active);
        CREATE INDEX IF NOT EXISTS agent_settings_setting_type_idx ON agent_settings (setting_type);
        """)
    
    # Check if tables exist after creation
    tables_status = check_tables_exist(supabase)
    if all(tables_status.values()):
        logger.info("✅ All required tables are now available!")
        logger.info("You can now run setup_supabase.py to populate default settings.")
    else:
        missing_tables = [table for table, exists in tables_status.items() if not exists]
        logger.error(f"❌ Tables still missing: {', '.join(missing_tables)}")
        logger.error("Please create them manually via the Supabase dashboard.")

if __name__ == "__main__":
    main() 