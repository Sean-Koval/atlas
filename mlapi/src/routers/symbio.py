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

from mlapi.src.models.vc_models import Startup, VentureCapitalFirm

router = APIRouter()

#### ----------------------- API BEGINS
### VC/STARTUP USER INPUT AND DATA COLLECTION
@router.post("/venture_capital_firms")
def create_venture_capital_firm(venture_capital_firm: VentureCapitalFirm):
    # Insert the venture capital firm into the database
    with conn, conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO venture_capital_firms (name, location, website, investment_focus, aum) VALUES (%s, %s, %s, %s, %s)",
            (
                venture_capital_firm.name,
                venture_capital_firm.location,
                venture_capital_firm.website,
                venture_capital_firm.investment_focus,
                venture_capital_firm.aum,
            ),
        )
        # Return the ID of the inserted row
        return {"id": cursor.lastrowid, "venture_capital_firm": venture_capital_firm}


@router.get("/venture_capital_firms/{id}")
def read_venture_capital_firm(id: int):
    # Retrieve the venture capital firm from the database
    with conn, conn.cursor() as cursor:
        cursor.execute("SELECT * FROM venture_capital_firms WHERE id = %s", (id,))
        venture_capital_firm = cursor.fetchone()
        # print(f"venture_capital_firm: {venture_capital_firm}")
        # print(f"type(venture_capital_firm): {type(venture_capital_firm)}")
        if venture_capital_firm is not None:
            # venture_capital_firm_dict = dict(venture_capital_firm)
            id, name, location, website, investment_focus, aum = venture_capital_firm
            # Convert the row to a VentureCapitalFirm object
            # return "Could not find"
            # return VentureCapitalFirm(**venture_capital_firm_dict)
            return VentureCapitalFirm(
                id=id,
                name=name,
                location=location,
                website=website,
                investment_focus=investment_focus,
                aum=aum,
            )
        else:
            return {"error": "Venture capital firm not found"}


@router.post("/startups")
def create_startup(startup: Startup):
    # Insert the startup into the database
    with conn, conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO startups (name, industry, location, website, funding_stage, funding_amount, pitch) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                startup.name,
                startup.industry,
                startup.location,
                startup.website,
                startup.funding_stage,
                startup.funding_amount,
                startup.pitch,
            ),
        )
        # Return the ID of the inserted row
        return {"id": cursor.lastrowid, "startup": startup}


@router.get("/startups/{id}")
def read_startup(id: int):
    # Retrieve the startup from the database
    with conn, conn.cursor() as cursor:
        cursor.execute("SELECT * FROM startups WHERE id = %s", (id,))
        startup = cursor.fetchone()
        # print(f"Startup: {startup}")
        # print(f"type(startup): {type(startup)}")
        if startup is not None:
            # Convert the row to a Startup object
            (
                id,
                name,
                industry,
                location,
                website,
                funding_stage,
                funding_amount,
                pitch,
            ) = startup
            return Startup(
                id=id,
                name=name,
                industry=industry,
                location=location,
                website=website,
                funding_stage=funding_stage,
                funding_amount=funding_amount,
                pitch=pitch,
            )
        else:
            return {"error": "Startup not found"}


@router.get("/startups")
def read_all_startups():
    # Retrieve all startups from the database
    with conn, conn.cursor() as cursor:
        cursor.execute("SELECT * FROM startups")
        # return all startups from db
        startups = cursor.fetchall()
        return startups
