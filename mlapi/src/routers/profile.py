import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, List

import aiohttp
import aioredis
import asyncpg
# Set up PostgreSQL connection
import psycopg2
import pypeln
import requests
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware  # middleware helper
from fastapi_sqlalchemy import \
    db  # an object to provide global access to a database session
from pydantic import BaseModel
# from aioredis import create_pool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mlapi.src.models.vc_models import CompanyInput

router = APIRouter()
### -------------------------------------------------------
### STORE COMPANY DATA ABOUT A COMPANY - useful for venture capital firms and startups that want to collect and store data about firms they are watching
# Define the function to store company data in the database
@router.post("/store-company-data")
def store_company_data(company_input: CompanyInput):
    # Check if the table exists in the database
    cur.execute("SELECT to_regclass('public.companies')")
    table_exists = cur.fetchone()[0]

    # If the table does not exist, create it
    if not table_exists:
        cur.execute(
            "CREATE TABLE companies (company_name TEXT, industry TEXT, team_members TEXT, project TEXT, funding_round TEXT, valuation TEXT)"
        )

    # Insert the company data into the database
    query = "INSERT INTO companies (company_name, industry, team_members, project, funding_round, valuation) VALUES (%s, %s, %s, %s, %s, %s)"
    cur.execute(
        query,
        (
            company_input.company_name,
            company_input.industry,
            company_input.team_members,
            company_input.project,
            company_input.funding_round,
            company_input.valuation,
        ),
    )
    conn.commit()


# NOTE: STill requires user to input both fields
# Define the function to query the database
@router.get("/query-database/{company_name}/{industry}")
def query_database(company_name: str | None = None, industry: str | None = None):
    # Check if the table exists in the database
    cur.execute("SELECT to_regclass('public.companies')")
    table_exists = cur.fetchone()[0]

    # If the table does not exist, return an error message
    if not table_exists:
        return {"error": "Table does not exist"}

    # Build the query based on the user's input
    query = "SELECT * FROM companies WHERE"
    if company_name:
        query += f" company_name='{company_name}'"
    if industry:
        if company_name:
            query += " AND"
        query += f" industry='{industry}'"
    if not company_name and not industry:
        return {"error": "Company name or industry is required"}

    # Execute the query and fetch the results
    cur.execute(query)
    results = cur.fetchall()

    # Return the results
    return results
