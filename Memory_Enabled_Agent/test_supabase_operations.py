#!/usr/bin/env python
"""
Supabase Operations Test Script

This script tests real database operations on existing Supabase tables using the 
exact same methods that the OYOS agent uses. It will attempt to:
1. List all tables in your Supabase database
2. Perform SELECT operations on existing tables
3. Perform INSERT operations (and rollback to avoid leaving test data)
4. Perform UPDATE operations (on test data only)
5. Perform DELETE operations (on test data only)

Usage:
    python test_supabase_operations.py --url SUPABASE_URL --key SUPABASE_KEY

Or it will read from .env file or environment variables.
"""

import os
import sys
import json
import uuid
import argparse
import logging
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("supabase_operations_test")

try:
    from supabase import create_client, Client
except ImportError:
    logger.error("Supabase client not installed. Please run: pip install supabase>=2.0.0")
    sys.exit(1)

def list_all_tables(supabase: Client) -> List[str]:
    """
    List all tables in the Supabase database using PostgreSQL information schema
    
    Args:
        supabase: Supabase client
        
    Returns:
        List[str]: List of table names
    """
    tables = []
    
    logger.info("\n----- Listing All Tables in Database -----")
    
    try:
        # First, specifically check for the "profiles" table that we know exists
        try:
            # Just test if we can access the table
            result = supabase.table("profiles").select("count").limit(1).execute()
            tables.append("profiles")
            logger.info(f"✅ Found table: profiles")
        except Exception as e:
            logger.warning(f"❌ Expected 'profiles' table not found - {str(e)}")
        
        # This is a direct SQL endpoint for Supabase that uses REST not the 'sql' method
        # We're making a direct REST call using the Supabase client's internal HTTP client
        # This mimics what would happen in the Supabase dashboard when you click "Tables"
        try:
            response = supabase.table("_metadata").select("*").execute()
            logger.info("Metadata table access successful")
        except Exception:
            logger.info("No metadata table available")
        
        # Try accessing some common tables
        test_tables = ["goals", "agent_settings", "users", "auth", "storage", "buckets", "objects"]
        for table in test_tables:
            if table in tables:
                continue  # Skip if already found
                
            try:
                # Just test if we can access the table
                result = supabase.table(table).select("count").limit(1).execute()
                tables.append(table)
                logger.info(f"✅ Found table: {table}")
            except Exception:
                # Don't log these failures to keep output clean
                pass
        
        if not tables:
            logger.warning("Could not find any tables with standard methods.")
            logger.info("Checking information schema as fallback (requires elevated permissions)...")
            # Try to query information_schema as a last resort
            # This will likely fail in most cases due to permissions
            # But it's worth trying
            try:
                url = supabase.rest_url + "/schemas"
                response = supabase.postgrest.request("GET", url)
                if response and hasattr(response, "json") and callable(response.json):
                    schemas = response.json()
                    logger.info(f"Found schemas: {schemas}")
            except Exception as e:
                logger.warning(f"Could not query information schema: {e}")
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
    
    return tables

def test_select_operations(supabase: Client, table_name: str) -> bool:
    """
    Test SELECT operations on a table
    
    Args:
        supabase: Supabase client
        table_name: Table name to test
        
    Returns:
        bool: Success status
    """
    logger.info(f"\n----- Testing SELECT Operations on {table_name} -----")
    
    success = True
    
    try:
        # Test 1: Simple SELECT with limit
        logger.info(f"Testing simple SELECT with limit on {table_name}...")
        response = supabase.table(table_name).select("*").limit(5).execute()
        count = len(response.data)
        logger.info(f"✅ SELECT with limit successful. Returned {count} rows.")
        
        # Display sample data if available
        if response.data:
            # Redact potentially sensitive data for profiles table
            if table_name == "profiles":
                sample_record = response.data[0].copy()
                # Redact potentially sensitive fields
                for field in ["email", "password", "phone", "address"]:
                    if field in sample_record:
                        sample_record[field] = "REDACTED"
                logger.info(f"Sample record (redacted): {json.dumps(sample_record, indent=2)}")
            else:
                logger.info(f"Sample record: {json.dumps(response.data[0], indent=2)}")
            
            # Get field names from the first record
            if count > 0:
                fields = list(response.data[0].keys())
                logger.info(f"Table fields: {fields}")
                
                # Test 2: SELECT with specific columns
                if len(fields) >= 2:
                    logger.info(f"\nTesting SELECT with specific columns: {fields[:2]}...")
                    field_list = ",".join(fields[:2])
                    response = supabase.table(table_name).select(field_list).limit(2).execute()
                    logger.info(f"✅ SELECT with specific columns successful.")
                    sample_record = response.data[0] if response.data else {}
                    logger.info(f"Sample record: {json.dumps(sample_record, indent=2)}")
                
                # Test 3: SELECT with filtering
                if "id" in fields:
                    logger.info("\nTesting SELECT with filtering on id field...")
                    # Use a random UUID that likely won't exist
                    test_id = str(uuid.uuid4())
                    response = supabase.table(table_name).select("*").eq("id", test_id).execute()
                    logger.info(f"✅ SELECT with filtering successful. Returned {len(response.data)} rows.")
                elif fields:
                    # Use the first field for filtering
                    field = fields[0]
                    logger.info(f"\nTesting SELECT with filtering on {field}...")
                    # Use a value that likely won't exist
                    test_value = "test_value_that_should_not_exist_12345"
                    response = supabase.table(table_name).select("*").eq(field, test_value).execute()
                    logger.info(f"✅ SELECT with filtering successful. Returned {len(response.data)} rows.")
        else:
            logger.info(f"Table {table_name} is empty.")
            
    except Exception as e:
        logger.error(f"❌ SELECT operations failed: {e}")
        success = False
    
    return success

def test_insert_operations(supabase: Client, table_name: str) -> (bool, Optional[str]):
    """
    Test INSERT operations on a table
    
    Args:
        supabase: Supabase client
        table_name: Table name to test
        
    Returns:
        tuple: (Success status, Inserted record ID or None)
    """
    logger.info(f"\n----- Testing INSERT Operations on {table_name} -----")
    
    success = True
    inserted_id = None
    
    try:
        # First get the table structure
        logger.info(f"Getting table structure for {table_name}...")
        response = supabase.table(table_name).select("*").limit(1).execute()
        
        if response.data:
            fields = list(response.data[0].keys())
            logger.info(f"Table fields: {fields}")
            
            # Create test data based on table structure
            # We need to be careful to create valid data for each field
            test_data = {}
            
            # Special handling for common fields
            skip_fields = []
            if "id" in fields:
                # Skip ID field, it's usually auto-generated
                skip_fields.append("id")
            
            if "created_at" in fields:
                # Skip created_at field, it's usually auto-generated
                skip_fields.append("created_at")
            
            if "updated_at" in fields:
                # Skip updated_at field, it's usually auto-generated
                skip_fields.append("updated_at")
            
            # Handle specific tables we know about
            if table_name == "profiles":
                # Special handling for profiles table
                test_data = {
                    "username": f"test_user_{uuid.uuid4()}",
                    "full_name": "Test User (Safe to Delete)",
                    "avatar_url": "https://example.com/test-avatar.jpg",
                    "website": "https://example.com",
                    "email": f"test_{uuid.uuid4()}@example.com",
                    "bio": "This is a test user created by the operations test script. Safe to delete.",
                    "is_test_account": True
                }
                
                # Remove any fields not in the actual table
                test_data = {k: v for k, v in test_data.items() if k in fields and k not in skip_fields}
                
                # Add any required fields that we didn't cover
                for field in fields:
                    if field not in test_data and field not in skip_fields:
                        if field.endswith("_id") or field == "id":
                            test_data[field] = str(uuid.uuid4())
                        elif field.endswith("_email") or field == "email":
                            test_data[field] = f"test_{uuid.uuid4()}@example.com"
                        elif field.endswith("_name") or field == "name":
                            test_data[field] = f"Test Name {uuid.uuid4()}"
                        elif field.endswith("_type") or field == "type":
                            test_data[field] = "test_type"
                        elif field.endswith("_status") or field == "status":
                            test_data[field] = "test_status"
                        elif field.endswith("_active") or field == "is_active":
                            test_data[field] = False
                        elif field.endswith("_value") or field == "value":
                            test_data[field] = json.dumps({"test": "value"})
                        elif field.endswith("_date") or field.endswith("_at"):
                            test_data[field] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                        else:
                            test_data[field] = f"test_value_{uuid.uuid4()}"
                
            elif table_name == "goals":
                test_data = {
                    "user_email": f"test_{uuid.uuid4()}@example.com",
                    "description": "Test goal from operation test script",
                    "goal_type": "test",
                    "status": "active",
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                }
            elif table_name == "agent_settings":
                test_data = {
                    "name": f"Test Setting {uuid.uuid4()}",
                    "agent_type": "test",
                    "setting_type": "test_setting",
                    "value": json.dumps({"test": "value"}),
                    "is_active": False
                }
            else:
                # Generic handling for unknown tables
                logger.warning(f"Unknown table: {table_name}. Creating generic test data.")
                for field in fields:
                    if field in skip_fields:
                        continue
                        
                    if field.endswith("_id") or field == "id":
                        test_data[field] = str(uuid.uuid4())
                    elif field.endswith("_email") or field == "email":
                        test_data[field] = f"test_{uuid.uuid4()}@example.com"
                    elif field.endswith("_name") or field == "name":
                        test_data[field] = f"Test Name {uuid.uuid4()}"
                    elif field.endswith("_type") or field == "type":
                        test_data[field] = "test_type"
                    elif field.endswith("_status") or field == "status":
                        test_data[field] = "test_status"
                    elif field.endswith("_active") or field == "is_active":
                        test_data[field] = False
                    elif field.endswith("_value") or field == "value":
                        test_data[field] = json.dumps({"test": "value"})
                    elif field.endswith("_date") or field.endswith("_at"):
                        test_data[field] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    else:
                        test_data[field] = f"test_value_{uuid.uuid4()}"
            
            logger.info(f"Inserting test data: {json.dumps(test_data, indent=2)}")
            
            # Insert the test data
            response = supabase.table(table_name).insert(test_data).execute()
            
            if response.data:
                logger.info("✅ INSERT operation successful!")
                logger.info(f"Inserted record: {json.dumps(response.data[0], indent=2)}")
                
                # Get the ID of the inserted record for later cleanup
                if "id" in response.data[0]:
                    inserted_id = response.data[0]["id"]
                    logger.info(f"Inserted record ID: {inserted_id}")
            else:
                logger.warning("INSERT operation returned no data, but appeared to succeed")
        else:
            logger.warning(f"Table {table_name} structure could not be determined (empty table). Skipping INSERT test.")
    except Exception as e:
        logger.error(f"❌ INSERT operation failed: {e}")
        success = False
    
    return success, inserted_id

def test_update_operations(supabase: Client, table_name: str, record_id: str) -> bool:
    """
    Test UPDATE operations on a table
    
    Args:
        supabase: Supabase client
        table_name: Table name to test
        record_id: ID of the record to update
        
    Returns:
        bool: Success status
    """
    logger.info(f"\n----- Testing UPDATE Operations on {table_name} -----")
    
    success = True
    
    try:
        # First get the current record
        logger.info(f"Getting current record for {table_name}...")
        response = supabase.table(table_name).select("*").eq("id", record_id).execute()
        
        if response.data:
            current_record = response.data[0]
            logger.info(f"Current record: {json.dumps(current_record, indent=2)}")
            
            # Create update data
            update_data = {}
            
            # Handle specific tables we know about
            if table_name == "profiles":
                # Only update fields that are safe to modify
                update_data = {
                    "bio": f"Updated test bio {uuid.uuid4()} - This account was created by the test script",
                    "website": f"https://example.com/updated-{uuid.uuid4()}"
                }
                
                # Only include fields that actually exist in the table
                update_data = {k: v for k, v in update_data.items() if k in current_record}
                
            elif table_name == "goals":
                update_data = {
                    "description": f"Updated goal from operation test script {uuid.uuid4()}",
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                }
            elif table_name == "agent_settings":
                update_data = {
                    "name": f"Updated Setting {uuid.uuid4()}",
                    "value": json.dumps({"test": "updated_value"})
                }
            else:
                # Generic handling for unknown tables
                for key, value in current_record.items():
                    if key == "id" or key.endswith("_id"):
                        # Skip ID fields
                        continue
                    elif key.endswith("_at") or key.endswith("_date"):
                        # Update date fields
                        update_data[key] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    elif isinstance(value, str) and not key.endswith("_email"):
                        # Update string fields
                        update_data[key] = f"Updated {value} {uuid.uuid4()}"
                    elif key.endswith("_email") or key == "email":
                        # Update email fields
                        update_data[key] = f"updated_{uuid.uuid4()}@example.com"
            
            logger.info(f"Updating with data: {json.dumps(update_data, indent=2)}")
            
            # Update the record
            response = supabase.table(table_name).update(update_data).eq("id", record_id).execute()
            
            if response.data:
                logger.info("✅ UPDATE operation successful!")
                logger.info(f"Updated record: {json.dumps(response.data[0], indent=2)}")
            else:
                logger.warning("UPDATE operation returned no data, but appeared to succeed")
        else:
            logger.warning(f"Record with ID {record_id} not found. Skipping UPDATE test.")
            success = False
    except Exception as e:
        logger.error(f"❌ UPDATE operation failed: {e}")
        success = False
    
    return success

def test_delete_operations(supabase: Client, table_name: str, record_id: str) -> bool:
    """
    Test DELETE operations on a table
    
    Args:
        supabase: Supabase client
        table_name: Table name to test
        record_id: ID of the record to delete
        
    Returns:
        bool: Success status
    """
    logger.info(f"\n----- Testing DELETE Operations on {table_name} -----")
    
    success = True
    
    try:
        # Delete the record
        logger.info(f"Deleting record with ID {record_id}...")
        response = supabase.table(table_name).delete().eq("id", record_id).execute()
        
        if response.data:
            logger.info("✅ DELETE operation successful!")
            logger.info(f"Deleted record: {json.dumps(response.data[0], indent=2)}")
        else:
            logger.warning("DELETE operation returned no data, but appeared to succeed")
        
        # Verify the record was deleted
        logger.info("Verifying record was deleted...")
        response = supabase.table(table_name).select("*").eq("id", record_id).execute()
        
        if not response.data:
            logger.info("✅ Verification successful! Record no longer exists.")
        else:
            logger.warning("❌ Verification failed! Record still exists after deletion.")
            success = False
    except Exception as e:
        logger.error(f"❌ DELETE operation failed: {e}")
        success = False
    
    return success

def main():
    """Main function"""
    # Load environment variables from .env file if present
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test Supabase database operations")
    parser.add_argument("--url", help="Supabase URL")
    parser.add_argument("--key", help="Supabase API key")
    parser.add_argument("--table", help="Specific table to test (default: all)")
    parser.add_argument("--skip-insert", action="store_true", help="Skip INSERT operations")
    parser.add_argument("--skip-update", action="store_true", help="Skip UPDATE operations")
    parser.add_argument("--skip-delete", action="store_true", help="Skip DELETE operations")
    parser.add_argument("--read-only", action="store_true", help="Skip all write operations (equivalent to --skip-insert --skip-update --skip-delete)")
    
    args = parser.parse_args()
    
    # If read-only flag is set, enable all skip flags
    if args.read_only:
        args.skip_insert = True
        args.skip_update = True
        args.skip_delete = True
    
    # Get Supabase credentials
    supabase_url = args.url or os.environ.get("SUPABASE_URL")
    supabase_key = args.key or os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase URL and API key are required")
        logger.error("Please provide them as arguments or set environment variables")
        logger.error("Usage: python test_supabase_operations.py --url SUPABASE_URL --key SUPABASE_KEY")
        sys.exit(1)
    
    # Connect to Supabase
    try:
        logger.info(f"Connecting to Supabase at {supabase_url}...")
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase!")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        sys.exit(1)
    
    # List all tables
    tables = list_all_tables(supabase)
    
    if not tables:
        logger.error("No tables found or accessible. Make sure your Supabase instance has tables and your API key has access to them.")
        sys.exit(1)
    
    # Filter to specific table if provided
    if args.table:
        if args.table in tables:
            tables = [args.table]
        else:
            logger.error(f"Table {args.table} not found in the list of accessible tables.")
            logger.info(f"Available tables: {', '.join(tables)}")
            sys.exit(1)
    
    # Test operations on each table
    results = {}
    
    for table in tables:
        logger.info(f"\n===== Testing Operations on Table: {table} =====")
        
        # Initialize results for this table
        results[table] = {
            "select": False,
            "insert": False,
            "update": False,
            "delete": False
        }
        
        # Test SELECT operations
        results[table]["select"] = test_select_operations(supabase, table)
        
        # Test INSERT, UPDATE, DELETE operations
        if not args.skip_insert:
            insert_success, record_id = test_insert_operations(supabase, table)
            results[table]["insert"] = insert_success
            
            if insert_success and record_id:
                # Only test UPDATE if INSERT was successful and returned an ID
                if not args.skip_update:
                    results[table]["update"] = test_update_operations(supabase, table, record_id)
                
                # Always try to DELETE the test record to clean up
                if not args.skip_delete:
                    results[table]["delete"] = test_delete_operations(supabase, table, record_id)
    
    # Print summary
    logger.info("\n===== Operation Test Summary =====")
    
    for table, ops in results.items():
        logger.info(f"\nTable: {table}")
        logger.info(f"SELECT: {'✅ PASS' if ops['select'] else '❌ FAIL'}")
        logger.info(f"INSERT: {'✅ PASS' if ops['insert'] else '❌ FAIL' if not args.skip_insert else 'SKIPPED'}")
        logger.info(f"UPDATE: {'✅ PASS' if ops['update'] else '❌ FAIL' if not args.skip_update and ops['insert'] else 'SKIPPED'}")
        logger.info(f"DELETE: {'✅ PASS' if ops['delete'] else '❌ FAIL' if not args.skip_delete and ops['insert'] else 'SKIPPED'}")
    
    # Overall assessment
    all_selects = all(ops["select"] for ops in results.values())
    all_inserts = all(ops["insert"] for ops in results.values()) if not args.skip_insert else True
    all_updates = all(ops["update"] for ops in results.values()) if not args.skip_update else True
    all_deletes = all(ops["delete"] for ops in results.values()) if not args.skip_delete else True
    
    logger.info("\n===== Overall Assessment =====")
    if all([all_selects, all_inserts, all_updates, all_deletes]):
        logger.info("✅ ALL OPERATIONS SUCCESSFUL!")
        logger.info("Your Supabase setup is correctly configured and the agent should be able to access and modify the data.")
    elif all_selects:
        logger.info("⚠️ SELECT operations successful, but some write operations failed.")
        logger.info("The agent may be able to read data but might have issues writing to the database.")
    else:
        logger.info("❌ SELECT operations failed on some tables.")
        logger.info("The agent may have issues accessing data from the database.")
    
    # Check agent-specific tables
    agent_tables = ["goals", "agent_settings"]
    missing_agent_tables = [table for table in agent_tables if table not in tables]
    
    if missing_agent_tables:
        logger.warning("\n⚠️ The following tables required by the OYOS agent are missing:")
        for table in missing_agent_tables:
            logger.warning(f"  - {table}")
        logger.warning("You may need to create these tables manually in the Supabase dashboard.")
        logger.warning("See the README.md for table structure details.")

if __name__ == "__main__":
    main() 