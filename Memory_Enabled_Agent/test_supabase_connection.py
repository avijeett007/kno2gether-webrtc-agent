#!/usr/bin/env python
"""
Supabase Connection Test Script

This script tests the connection to Supabase and verifies if the required tables
exist and can be accessed using the same methods that the agent uses.

Usage:
    python test_supabase_connection.py --url SUPABASE_URL --key SUPABASE_KEY

Or set environment variables:
    SUPABASE_URL and SUPABASE_KEY
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("supabase_test")

try:
    from supabase import create_client, Client
except ImportError:
    logger.error("Supabase client not installed. Please run: pip install supabase>=2.0.0")
    sys.exit(1)

def test_connection(supabase: Client) -> bool:
    """Test basic connection to Supabase"""
    try:
        # A simple test to check if the connection works
        logger.info("Testing basic connection...")
        
        # Try to get a list of tables (this doesn't actually list tables but tests auth)
        # We'll use a select on a non-existent table with limit 0 just to test auth
        try:
            supabase.table("_test_connection").select("*").limit(0).execute()
        except Exception as e:
            # This will likely fail with a 404, which is fine
            if "404" not in str(e):
                logger.error(f"Unexpected error during connection test: {e}")
                return False
        
        logger.info("✅ Connection successful!")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return False

def test_goals_table(supabase: Client) -> bool:
    """Test access to goals table"""
    success = True
    
    logger.info("\n----- Testing 'goals' table access -----")
    
    # Test 1: Check if goals table exists
    try:
        logger.info("Checking if goals table exists...")
        response = supabase.table("goals").select("*").limit(5).execute()
        logger.info(f"✅ Goals table exists! Found {len(response.data)} records")
        
        # Display sample data if available
        if response.data:
            logger.info(f"Sample record: {json.dumps(response.data[0], indent=2)}")
        else:
            logger.info("No records found in goals table (empty table)")
    except Exception as e:
        logger.error(f"❌ Failed to access goals table: {e}")
        success = False
    
    # Test 2: Test filtering (same as agent uses)
    if success:
        try:
            logger.info("\nTesting filtering on goals table...")
            # Use a generic filter that should work even if no records match
            test_email = "test@example.com"  # Use a test email that likely won't exist
            response = supabase.table("goals").select("*").eq("user_email", test_email).eq("status", "active").execute()
            logger.info(f"✅ Filtering works! Found {len(response.data)} active goals for test user")
        except Exception as e:
            logger.error(f"❌ Failed to filter goals table: {e}")
            success = False
    
    return success

def test_agent_settings_table(supabase: Client) -> bool:
    """Test access to agent_settings table"""
    success = True
    
    logger.info("\n----- Testing 'agent_settings' table access -----")
    
    # Test 1: Check if agent_settings table exists
    try:
        logger.info("Checking if agent_settings table exists...")
        response = supabase.table("agent_settings").select("*").limit(5).execute()
        logger.info(f"✅ Agent settings table exists! Found {len(response.data)} records")
        
        # Display sample data if available
        if response.data:
            logger.info(f"Sample record: {json.dumps(response.data[0], indent=2)}")
        else:
            logger.info("No records found in agent_settings table (empty table)")
    except Exception as e:
        logger.error(f"❌ Failed to access agent_settings table: {e}")
        success = False
    
    # Test 2: Test the exact query that the agent uses to get system prompts
    if success:
        try:
            logger.info("\nTesting system prompt retrieval (same as agent uses)...")
            response = supabase.table("agent_settings").select("*").eq("agent_type", "coach").eq("setting_type", "system_prompt").eq("is_active", True).execute()
            if response.data:
                logger.info(f"✅ System prompt query works! Found {len(response.data)} active system prompts for coach agent")
            else:
                logger.info("✅ Query works but no system prompts found for coach agent")
            
            # Try the same for DISC agent
            response = supabase.table("agent_settings").select("*").eq("agent_type", "disc").eq("setting_type", "system_prompt").eq("is_active", True).execute()
            if response.data:
                logger.info(f"✅ System prompt query works! Found {len(response.data)} active system prompts for DISC agent")
            else:
                logger.info("✅ Query works but no system prompts found for DISC agent")
        except Exception as e:
            logger.error(f"❌ Failed to query system prompts: {e}")
            success = False
    
    return success

def test_table_schema(supabase: Client, table_name: str) -> bool:
    """Test if table schema is compatible with agent expectations"""
    logger.info(f"\n----- Testing '{table_name}' table schema -----")
    
    expected_columns = {
        "goals": [
            "id", "user_email", "description", "goal_type", 
            "status", "created_at", "last_updated"
        ],
        "agent_settings": [
            "id", "name", "agent_type", "setting_type", 
            "value", "is_active", "created_at", "updated_at"
        ]
    }
    
    if table_name not in expected_columns:
        logger.error(f"Unknown table: {table_name}")
        return False
    
    try:
        # Get a sample record to check columns
        response = supabase.table(table_name).select("*").limit(1).execute()
        
        if not response.data:
            logger.warning(f"❓ Cannot verify schema - no records in {table_name} table")
            return True  # Assume it's okay if we can query the table but it's empty
        
        record = response.data[0]
        
        # Check for expected columns
        missing_columns = []
        for col in expected_columns[table_name]:
            if col not in record:
                missing_columns.append(col)
        
        if missing_columns:
            logger.error(f"❌ Schema validation failed! Missing columns: {', '.join(missing_columns)}")
            logger.error(f"Found columns: {', '.join(record.keys())}")
            return False
        else:
            logger.info(f"✅ Schema validation passed! All required columns present")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to validate schema: {e}")
        return False

def main():
    """Main function"""
    # Load environment variables from .env file if present
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test Supabase connection and table access")
    parser.add_argument("--url", help="Supabase URL")
    parser.add_argument("--key", help="Supabase API key")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Get Supabase credentials
    supabase_url = args.url or os.environ.get("SUPABASE_URL")
    supabase_key = args.key or os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase URL and API key are required")
        logger.error("Please provide them as arguments or set environment variables")
        logger.error("Usage: python test_supabase_connection.py --url SUPABASE_URL --key SUPABASE_KEY")
        sys.exit(1)
    
    # Connect to Supabase
    try:
        logger.info(f"Connecting to Supabase at {supabase_url}...")
        supabase = create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        sys.exit(1)
    
    # Run tests
    connection_ok = test_connection(supabase)
    if not connection_ok:
        logger.error("Basic connection test failed. Cannot proceed with table tests.")
        sys.exit(1)
    
    # Test goals table
    goals_ok = test_goals_table(supabase)
    
    # Test agent_settings table
    settings_ok = test_agent_settings_table(supabase)
    
    # If tables exist, test their schema
    if goals_ok:
        schema_goals_ok = test_table_schema(supabase, "goals")
    else:
        schema_goals_ok = False
        
    if settings_ok:
        schema_settings_ok = test_table_schema(supabase, "agent_settings")
    else:
        schema_settings_ok = False
    
    # Print summary
    logger.info("\n----- Test Summary -----")
    logger.info(f"Basic connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    logger.info(f"Goals table access: {'✅ PASS' if goals_ok else '❌ FAIL'}")
    logger.info(f"Agent settings table access: {'✅ PASS' if settings_ok else '❌ FAIL'}")
    logger.info(f"Goals table schema: {'✅ PASS' if schema_goals_ok else '❌ FAIL' if goals_ok else '⚠️ NOT TESTED'}")
    logger.info(f"Agent settings schema: {'✅ PASS' if schema_settings_ok else '❌ FAIL' if settings_ok else '⚠️ NOT TESTED'}")
    
    # Overall assessment
    logger.info("\n----- Conclusion -----")
    if all([connection_ok, goals_ok, settings_ok, schema_goals_ok, schema_settings_ok]):
        logger.info("✅ All tests passed! The agent should be able to access and use the tables correctly.")
    elif all([connection_ok, goals_ok, settings_ok]):
        logger.info("⚠️ Tables exist but schema validation had issues. The agent might encounter errors.")
    elif connection_ok:
        logger.info("❌ Connection works but required tables are missing or inaccessible.")
        logger.info("Please create the missing tables as described in the README.")
    else:
        logger.info("❌ Basic connection failed. Please check your Supabase credentials and network connection.")

if __name__ == "__main__":
    main() 