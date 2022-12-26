### -----------------------
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
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware  # middleware helper
from fastapi_sqlalchemy import \
    db  # an object to provide global access to a database session
from pydantic import BaseModel
# from aioredis import create_pool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import Request, Response, APIRouter, HTTPException

from config import settings
from models.vc_models import *
from routers import profile, radar, symbio


SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

# SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

# create fast-api
app = FastAPI(title="atlas", description="Venture Capital + Startup: Research Tool")

# postgres connection
conn = psycopg2.connect(SQLALCHEMY_DATABASE_URL)
cur = conn.cursor()

cur.execute("ROLLBACK")


# app.add_middleware(DBSessionMiddleware, db_url=os.getenv["DATABASE_URL"])


#### ----- THIS WILL BE ADDED
# create the connection pool for redis
# pool = aioredis.ConnectionPool.from_url("redis://localhost", max_connections=10)
# redis = aioredis.Redis(connection_pool=pool)


### ---- STORE COMPANY DATA IN POSTGRES ----- (FOR A USER)

# Create the startups table if it does not exist
with conn, conn.cursor() as cursor:
    try:
        cursor.execute(
            """
            CREATE TABLE startups (
                id serial PRIMARY KEY,
                name varchar(255) NOT NULL,
                industry varchar(255) NOT NULL,
                location varchar(255) NOT NULL,
                website varchar(255) NOT NULL,
                funding_stage varchar(255) NOT NULL,
                funding_amount integer NOT NULL,
                pitch text NOT NULL
            )
            """
        )
    except psycopg2.errors.DuplicateTable:
        pass

    try:
        cursor.execute(
            """
            CREATE TABLE venture_capital_firms (
                id serial PRIMARY KEY,
                name varchar(255) NOT NULL,
                location varchar(255) NOT NULL,
                website varchar(255) NOT NULL,
                investment_focus varchar(255) NOT NULL,
                aum varchar(255) NOT NULL
            )
            """
        )

    except psycopg2.errors.DuplicateTable:
        # Table already exists, do nothing
        pass


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f"{process_time:0.4f} sec")
    return response


# route endpoints
app.include_router(profile.router, prefix="/users", tags=["users"])
app.include_router(radar.router, prefix="/radar", tags=["radar"])
app.include_router(symbio.router, prefix="/symbio", tags=["symbio"])


# main server fn
async def main_pool():
    """Main loop that spins up services related to the application"""
    ### ------------------------------
    # configure and run uvicorn server
    config = uvicorn.Config("main:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main_pool())
