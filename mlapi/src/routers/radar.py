import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, List

import aiohttp
import aioredis
import asyncpg

# Set up PostgreSQL connectio
# import psycopg2
import pypeln
import requests
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware  # middleware helper
from fastapi_sqlalchemy import (
    db,
)  # an object to provide global access to a database session
from pydantic import BaseModel

# from aioredis import create_pool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.vc_models import KeywordInput, Project, SearchResult

router = APIRouter()

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


@router.post("/search")
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
@router.post("/scrape")
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
