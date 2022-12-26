### -----------------------
### Main Application:
###     1\ promps user to create account and sign in
###     2\ user can see what the application scrapes from github
###     3\ user can select specific projects and save to a folder
###     4\ user can retreive portfolio of saved projects
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
from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware  # middleware helper
from fastapi_sqlalchemy import \
    db  # an object to provide global access to a database session
from pydantic import BaseModel
# from aioredis import create_pool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings
from models.vc_models import *

# start db
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

# SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

# create fast-api
app = FastAPI()
# postgres connection
conn = psycopg2.connect(SQLALCHEMY_DATABASE_URL)
cur = conn.cursor()

cur.execute("ROLLBACK")

# adds process time to response header (for measuring speed of requests)
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f"{process_time:0.4f} sec")
    return response


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


#### ----------------------- API BEGINS
### VC/STARTUP USER INPUT AND DATA COLLECTION
@app.post("/venture_capital_firms")
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


@app.get("/venture_capital_firms/{id}")
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


@app.post("/startups")
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


@app.get("/startups/{id}")
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


@app.get("/startups")
def read_all_startups():
    # Retrieve all startups from the database
    with conn, conn.cursor() as cursor:
        cursor.execute("SELECT * FROM startups")
        # return all startups from db
        startups = cursor.fetchall()
        return startups


### -------------------------------------------------------
### STORE COMPANY DATA ABOUT A COMPANY - useful for venture capital firms and startups that want to collect and store data about firms they are watching
# Define the function to store company data in the database
@app.post("/store-company-data")
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
@app.get("/query-database/{company_name}/{industry}")
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


### --------------------------------------------------------------
### --- DATA SCRAPING: Collect information about github projects
# async function for scraping github (should be expanded to append the url to any endpoint that needs to be scraped)
async def search_github(keywords):
    """
    Searches GitHub using the provided keywords.

    Args:
        keywords (str): The keywords to search for.

    Returns:
        dict: The search results, including the total number of results and a list of items matching the search criteria.
    """
    async with aiohttp.ClientSession() as session:
        # encode the keywords as a query string
        params = {
            "q": keywords,
            # NOTE: fields doesnt work properly yet
            "fields": "id,name,description,tags,stargazers_count,forks_count,watchers_count,language,downloads_url,archived,open_issues_count",
        }
        async with session.get(
            "https://api.github.com/search/repositories", params=params
        ) as response:
            return await response.json()


async def insert_results(results):
    """
    Inserts the provided search results into a PostgreSQL database.

    Parameters:
        results (list of dict): A list of search results, including the keyword, total count, and items.

    Returns:
        None
    """
    # connect to the database
    conn = await asyncpg.connect(database="atlas_db", user="postgres", password="")

    # create the table if it doesn't exist
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_results (
        keyword text PRIMARY KEY,
        total_count integer NOT NULL,
        items jsonb NOT NULL
        )
        """
    )

    # insert the results into the table
    for result in results:
        # serialize to json
        items_json = json.dumps(result["items"])

        await conn.execute(
            """
        INSERT INTO search_results (keyword, total_count, items)
        VALUES ($1, $2, $3)
        ON CONFLICT (keyword) DO UPDATE
        SET total_count = $2, items = $3
        """,
            result["keyword"],
            result["total_count"],
            items_json,  # result["items"]
        )

    # close the connection
    await conn.close()


@app.post("/search")
async def search(keywords: KeywordInput) -> List[SearchResult]:
    """
    Handles a search request made to the API.

    Parameters:
        keywords (KeywordInput): An object containing a list of keywords to search for.

    Returns:
        list of SearchResult: A list of search results, including the total number of results and a list of items matching the search criteria.
    """
    # async scrapping tasks
    tasks = [
        asyncio.create_task(search_github(keyword)) for keyword in keywords.keywords
    ]

    # use asyncio.as_completed to iterate through the tasks as they are completed
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)

    # create a dict for each result with the keyword and the search results
    results_with_keywords = [
        {
            "keyword": keyword,
            "total_count": result["total_count"],
            "items": result["items"],
        }
        for keyword, result in zip(keywords.keywords, results)
    ]

    # insert the results into the database
    await insert_results(results_with_keywords)

    # create a SearchResult object for each result
    search_results = [
        SearchResult(total_count=result["total_count"], items=result["items"])
        for result in results
    ]

    # return the search results
    return search_results


### --- SCRAPE GH PROJECTS USING KEYWORDS ---
# Scrape GitHub for projects containing specific keywords
async def scrape_github(keywords: List[str]) -> List[Project]:
    """Scrape GitHub for projects containing specific keywords.

    Args:
        keywords: List of keywords to search for.

    Returns:
        List of Project objects containing the scraped projects.
    """
    projects = []
    for keyword in keywords:
        url = f"https://api.github.com/search/repositories?q={keyword}"
        r = requests.get(url)
        data = r.json()
        # collect project info
        for item in data["items"]:
            name = item["name"]
            description = item["description"]
            stars = item["stargazers_count"]
            project = Project(name=name, description=description, stars=stars)
            projects.append(project)

    return projects


# NOTE: might not need this function
# Store scraped projects in Redis cache
async def cache_projects(projects: List[Project]) -> None:
    """Store scraped projects in Redis cache.

    Args:
        projects: List of Project objects to store in cache.
    """
    for project in projects:
        await redis.set(project.name, project.description)


# Store scraped projects in PostgreSQL database
def store_projects(projects: List[Project]) -> None:
    """Store scraped projects in PostgreSQL database.

    Args:
        projects: List of Project objects to store in database.
    """
    try:
        # store github project info postgres
        for project in projects:
            cur.execute(
                "INSERT INTO github_projects VALUES (%s, %s, %s)",
                (project.name, project.description, project.stars),
            )
            conn.commit()
    except:
        print("FAILED TO STORE PROJECTS IN ATLAS_DB")
        pass


# Endpoint for scraping and storing projects
@app.post("/scrape")
async def scrape_and_store(keywords: List[str]):  # was -> None
    """Scrape GitHub for projects containing specific keywords and store them in Redis cache and PostgreSQL database.

    Args:
        keywords: List of keywords to search for.
    """
    # scrape github projects w/keyword
    projects = await scrape_github(keywords)
    # store projects: name, description, stars in postgres
    store_projects(projects)
    return projects


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
