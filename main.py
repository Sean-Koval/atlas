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

# start db
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

# SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

# create app
app = FastAPI()

# adds process time to response header (for measuring speed of requests)
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f"{process_time:0.4f} sec")
    return response


conn = psycopg2.connect(SQLALCHEMY_DATABASE_URL)
cur = conn.cursor()

cur.execute("ROLLBACK")
# app.add_middleware(DBSessionMiddleware, db_url=os.getenv["DATABASE_URL"])


#### ----- THIS WILL BE ADDED
# create the connection pool for redis
# pool = aioredis.ConnectionPool.from_url("redis://localhost", max_connections=10)
# redis = aioredis.Redis(connection_pool=pool)


class KeywordInput(BaseModel):
    keywords: list[str]


class SearchResult(BaseModel):
    total_count: int
    items: list[dict[str, Any]] = None


# Define model for project information
class Project(BaseModel):
    name: str | None
    description: str | None
    stars: int | None


class PartnerInput(BaseModel):
    company_name: str
    industry: str | None = None


class Partner(BaseModel):
    name: str
    industry: str
    location: str


# Define the input model for the function
class IndustryInput(BaseModel):
    industry_name: str
    timeframe: str


# Define the input model for the function
class IndustryInputCrunchbase(BaseModel):
    industry_name: str


# Define the input model for the function
class CompanyInput(BaseModel):
    company_name: str
    industry: str
    team_members: str
    project: str
    funding_round: str
    valuation: str


# for use by startup/vc finder
class Startup(BaseModel):
    name: str
    industry: str
    location: str
    website: str
    funding_stage: str
    funding_amount: int
    pitch: str


class VentureCapitalFirm(BaseModel):
    name: str
    location: str
    website: str
    investment_focus: str
    aum: int


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


#   BERT short text clustering
#   @app.get("/group-projects-by-topic")
#   def analyze_text():
#
#   """Returns projects from the db by grouped topic
#       - encode/tokenize/label/cluster
#       - BERT for short text analysis
#       - this means run something like BERT for topic modeling
#       - returm the desired groupings/topics
#       - : maybe we have to return a list of of all topics and # of projects for each topic (then have a separate function that can query that topic's projects and their similarities to each other - should we turn this into a graph data structure)
#   """
#
#
#
#

### ---------------------------------
### DATA SCRAPPING: COLLECT INFORMATION ABOUT COMPANIES FROM TECHCRUNCH
### this information can be used to analyze the competitive market and get company info
'''
async def search_for_partners(company_name: str, industry: Optional[str] = None) -> List[Partner]:
    """
    Searches for potential partners or customers for the specified company using the Crunchbase API.

    Parameters:
    - company_name (str): The name of the company to search for partners for.
    - industry (str): An optional industry to filter the search by.

    Returns:
    - list of Partner: A list of potential partners or customers that match the search criteria.
    """
    pass
    # check if the search results are cached in Redis
    async with aioredis.create_redis_pool(f"redis://{REDIS_HOST}:{REDIS_PORT}") as redis:
        cache_key = f"partners:{company_name}:{industry}"
        cached_results = await redis.get(cache_key)
        if cached_results:
            # return the cached results
            return json.loads(cached_results)

    # build the API endpoint URL
    endpoint = "https://api.crunchbase.com/v3.1/organizations"
    params = {"query": company_name}
    if industry:
        params["category_uuids"] = industry

    async with aiohttp.ClientSession() as session:
        # make the API request asynchronously
        async with session.get(endpoint, params=params, headers={"Authorization": f"Bearer {API_KEY}"}) as response:
            # parse the response and extract the relevant data
            results = parse_response(response)
    # NOTE: will want to add this functionality to the search_for_partners() function below
    # store the search results in PostgreSQL
    async with asyncpg.create_pool(f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}") as pool:
        async with pool.acquire() as conn:
            await conn.execute("INSERT INTO search_results (company_name, industry, results) VALUES ($1, $2, $3)", company_name, industry, results)

    # cache the search results in Redis
    async with aioredis.create_redis_pool(f"redis://{REDIS_HOST}:{REDIS_PORT}") as redis:
        await redis.set(cache_key, json.dumps(results))

    return results
'''
# implement this to collect from the asyncio function in a more concise way
#     for result in search_results["data"]["items"]:
#         # retrieve the company's funding history, market and industry data, and product and service offerings
#         funding_history, categories, products, leadership = await asyncio.gather(
#             session.get(result["relationships"]["funding_rounds"]["paging"]["first_page_url"], headers={"Authorization": f"Bearer {API_KEY}"}),
#             session.get(result["relationships"]["categories"]["paging"]["first_page_url"], headers={"Authorization": f"Bearer {API_KEY}"}


# async def search_for_partners(inputs: PartnerInput) -> List[Partner]:
#    """
#    Searches for potential partners or customers for the specified company using the Crunchbase API and returns information about the company's funding history, market and industry data, product and service offerings, and team and leadership. The retrieved data is also stored in Redis and PostgreSQL for faster access in the future.
#
#    Parameters:
#    - inputs (PartnerInput): An object containing the company name and optional industry to search for partners in.
#
#    Returns:
#    - list of Partner: A list of potential partners or customers that match the search criteria, with each object containing information about the company.
#    """
#    # check Redis for cached data
#    redis_key = f"partners:{inputs.company_name}:{inputs.industry}"
#    redis = await aioredis.create_redis_pool("redis://localhost")
#    cached_data = await redis.get(redis_key)
#    if cached_data:
# parse the cached data and return it
#        partners = [Partner.parse_raw(data) for data in json.loads(cached_data)]
#        return partners

#    partners = []

#    async with aiohttp.ClientSession() as session:
# search for potential partners or customers using the Crunchbase API
#        async with session.get("https://api.crunchbase.com/v3.1/organizations", params={"query": inputs.company_name, "category_uuids": inputs.industry}, headers={"Authorization": f"Bearer {API_KEY}"}) as response:
#            search_results = await response.json()

# retrieve information about each potential partner or customer
#        for result in search_results["data"]["items"]:
# retrieve the company's funding history
#            async with session.get(result["relationships"]["funding_rounds"]["paging"]["first_page_url"], headers={"Authorization": f"Bearer {API_KEY}"}) as response:
#                funding_rounds_data = await response.json()
#            funding_history = funding_rounds_data["data"]["items"]

#            # retrieve the company's market and industry data
#            async with session.get(result["relationships"]["categories"]["paging"]["first_page_url"], headers={"Authorization": f"Bearer {API_KEY}"}) as response:
#                categories_data = await response.json()
#            market_and_industry = categories_data["data"]["items"]

# retrieve the company's product and service offerings
#            async with session.get(result["relationships"]["products"]["paging"]["first_page_url"], headers={"Authorization": f"Bearer {API_KEY}"}) as response:
#                products_data = await response.json()
#            products_and_services = products_data["data"]["items"]

# retrieve the company's team and leadership
#            async with session.get(result["relationships"]["people"]["paging"]["first_page_url"], headers={"Authorization": f"Bearer {API_KEY}"}) as response:
#                people_data = await response.json()
#            team_and_leadership = people_data["data"]["items"]

'''
@app.get("/partners")
async def find_partners(input: PartnerInput) -> List[Partner]:
    """
    Finds potential partners or customers for a specific company.

    Parameters:
        company_name (str): The name of the company to search for partners for.
        industry (str): An optional industry to filter the search by.

    Returns:
        list of Partner: A list of Partner objects, with each object representing a specific partner or customer.
    """

    # check Redis for cached data
    redis_key = f"partners:{inputs.company_name}:{inputs.industry}"
    redis = await aioredis.create_redis_pool("redis://localhost")
    cached_data = await redis.get(redis_key)
    if cached_data:
        # parse the cached data and return it
        partners = [Partner.parse_raw(data) for data in json.loads(cached_data)]
        return partners

    partners = []

    # connect to the PostgreSQL database
    conn = await asyncpg.connect(database="atlas_db", user="postgres", password="")

    # retrieve potential partners or customers from the database based on a similarity score of their projects and industry
    query = """
    SELECT * FROM partners
    WHERE similarity(projects, $1) > 0.5 AND industry = $2
    """
    records = await conn.fetch(query, inputs.projects, inputs.industry)
    for record in records:
        # create a Partner object for each record
        partner = Partner(name=record["name"], funding_history=record["funding_history"], categories=record["categories"], products=record["products"], leadership=record["leadership"])
        partners.append(partner)

    # close the database connection
    await conn.close()

    return partners

    # search for potential partners or customers
    partners = await search_for_partners(input.company_name, input.industry)

    # create a Partner object for each result
    formatted_results = [Partner(name=partner.name, industry=partner.industry, location=partner.location) for partner in partners]
    # NOTE: Do we want to do storage of the result in postresql here?
    # NOTE: do we want caching here or in the search_partners furnction

    return formatted_results

'''

'''
import nltk

def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """
    Extracts the most relevant keywords from a short text using the Short Text Keyword Extraction (STKE) algorithm.

    Parameters:
    - text (str): The short text to extract keywords from.
    - num_keywords (int, optional): The number of keywords to extract. Defaults to 10.

    Returns:
    - list of str: A list of the most relevant keywords.
    """
    # tokenize the text using the Punkt tokenizer
    tokens = nltk.tokenize.word_tokenize(text)

    # remove stop words and lemmatize the tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in nltk.corpus.stopwords.words("english")]

    # compute the term frequency of each token
    term_frequency = {}
    for token in filtered_tokens:
        if token in term_frequency:
            term_frequency[token] += 1
        else:
            term_frequency[token] = 1

    # sort the tokens by term frequency in descending order
    sorted_tokens = sorted(term_frequency.items(), key=lambda x: x[1], reverse=True)

    # return the most relevant keywords
    return [token for token, frequency in sorted_tokens[:num_keywords]]


    This function first tokenizes the text using the Punkt tokenizer from the Natural Language Toolkit (nltk) library. It then removes stop words and lemmatizes the tokens using the WordNet lemmatizer from nltk. Next, it computes the term frequency of each token and sorts the tokens by term frequency in descending order. Finally, it returns the most relevant keywords by selecting the top num_keywords tokens from the sorted list.

    You can then use this function in the search_for_partners() function to extract keywords from the company's project descriptions and industry information before searching for potential partners:

'''

'''
import transformers

def analyze_sentiment(text: str) -> float:
    """
    Analyzes the sentiment of a text using the BERT model from the transformers library.

    Parameters:
    - text (str): The text to analyze.

    Returns:
    - float: A value between 0 (negative sentiment) and 1 (positive sentiment) indicating the overall sentiment of the text.
    """
    # load the BERT model
    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-cased")

    # encode the text as input for the model
    input_ids = transformers.BertTokenizer.from_pretrained("bert-base-cased").encode(text, return_tensors="pt")

    # run the model to classify the text as positive or negative sentiment
    outputs = model(input_ids)[0]
    sentiment = outputs[0][0]

    # return the sentiment score
    return sentiment.item()

This function uses the BERT model from the transformers library to analyze the sentiment of a text. It returns a value between 0 (negative sentiment) and 1 (positive sentiment) indicating the overall sentiment of the text.

'''
'''
import fasttext

def identify_topics(text: str, num_topics: int = 5, num_words: int = 5) -> List[Tuple[str, float]]:
    """
    Identifies the main topics in a text using the FastText model from the fasttext library.

    Parameters:
    - text (str): The text to analyze.
    - num_topics (int, optional): The number of topics to identify. Defaults to 5.
    - num_words (int, optional): The number of words to include in each topic. Defaults to 5.

    Returns:
    - list of tuple: A list of tuples containing the most important words for each topic and their weights.
    """
    # load the FastText model
    model = fasttext.load_model("cc.en.300.bin")

    # get the most important words for each topic
    topics = model.get_nearest_neighbors(text, k=num_topics * num_words)

    # group the words by topic and return the most important words for each topic
    return [(topic, weight) for topic, weight in zip(topics[0], topics[1])]

'''

'''
import transformers

def identify_topics(text: str, num_topics: int = 5, num_words: int = 5) -> List[Tuple[str, float]]:
    """
    Identifies the main topics in a text using the BERT model from the transformers library.

    Parameters:
    - text (str): The text to analyze.
    - num_topics (int, optional): The number of topics to identify. Defaults to 5.
    - num_words (int, optional): The number of words to include in each topic. Defaults to 5.

    Returns:
    - list of tuple: A list of tuples containing the most important words for each topic and their weights.
    """
    # load the BERT model
    model = transformers.BertModel.from_pretrained("bert-base-cased")

    # encode the text as input for the model
    input_ids = transformers.BertTokenizer.from_pretrained("bert-base-cased").encode(text, return_tensors="pt")

    # run the model to get the hidden states for each token
    hidden_states = model(input_ids)[2]

    # get the average hidden state for each topic
    topic_vectors = torch.mean(hidden_states, dim=1)

    # compute the cosine similarity between each topic vector and the input text
    similarities = torch.nn.functional.cosine_similarity(topic_vectors, input_ids, dim=1)

    # sort the topics by similarity
    sorted_topics = torch.argsort(similarities, descending=True)

    # get the most important words for each topic
    topics = []
    for topic in sorted_topics[:num_topics]:
        # get the hidden state for the topic
        topic_vector = hidden_states[topic]

        # compute the dot product between the topic vector and each token vector
        token_scores = torch.sum(topic_vector * input_ids, dim=1)

        # get the tokens with the highest scores
        sorted_tokens = torch.argsort(token_scores, descending=True)
        important_tokens = sorted_tokens[:num_words]

        # get the token strings and their scores
        tokens = [transformers.BertTokenizer.from_pretrained("bert-base-cased").decode(t, skip_special_tokens=True) for t in important_tokens]
        scores = token_scores[important_tokens].tolist()


        # add the topic to the list
        topics.append(tuple(zip(tokens, scores)))

    return topics


This function now returns a list of tuples containing the most important words for each topic and their weights. The topics are sorted by their similarity to the input text, and the important words for each topic are sorted by their relevance to the topic.

'''

"""
# Define the function to collect data from Google Trends
@app.get("/market-growth/{industry_name}")
def get_market_growth(industry_input: IndustryInput):
    # Check if the data is already stored in Redis
    data = r.get(industry_input.industry_name)
    if data:
        return data
    else:
        # Set up the pytrends client
        pytrends = TrendReq()

        # Define the keywords for the industry
        keywords = [industry_input.industry_name]

        # Request data from Google Trends
        trend_data = pytrends.trend(keywords, timeframe=industry_input.timeframe)

        # Extract the data for the specific industry from the response
        industry_data = trend_data[industry_input.industry_name]

        # Store the data in Redis
        r.set(industry_input.industry_name, industry_data)

        # Store the data in PostgreSQL
        query = "INSERT INTO market_growth (industry_name, data) VALUES (%s, %s)"
        cur.execute(query, (industry_input.industry_name, industry_data))
        conn.commit()

        return industry_data

# Define an error handler for invalid industry names
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": "Invalid industry name"}


    This code creates a FastAPI app and defines a function that uses the pytrends library to collect data on market growth for a specific industry. The function first checks if the data is already stored in Redis, and if it is, it returns the stored data. If the data is not stored in Redis, the function requests the data from Google Trends and stores it in both Redis and PostgreSQL. The function also includes an error handler that returns an error message if an invalid industry name is passed as an input.

To use this function, you can make a GET request to the endpoint /market-growth/{industry_name} and pass the industry name and timeframe as parameters in the query string. For example:
"""


"""

# Define the function to collect data from Crunchbase
@app.get("/market-growth/{industry_name}")
def get_market_growth(industry_input: IndustryInput):
    # Check if the data is already stored in Redis
    data = r.get(industry_input.industry_name)
    if data:
        return data
    else:
        # Set up the Crunchbase API client
        api_key = "your_api_key"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Define the URL for the Crunchbase API endpoint
        endpoint = "https://api.crunchbase.com/v3.1/odm-organizations"

        # Set the parameters for the API request
        params = {
            "organization_types": "company",
            "query": industry_input.industry_name
        }

        # Send the request to the Crunchbase API
        response = requests.get(endpoint, headers=headers, params=params)

        # Extract the data from the response
        data = response.json()["data"]["items"]

        # Store the data in Redis
        r.set(industry_input.industry_name, data)

        # Store the data in PostgreSQL
        query = "INSERT INTO market_growth (industry_name, data) VALUES (%s, %s)"
        cur.execute(query, (industry_input.industry_name, data))
        conn.commit()

        return data

his code creates a FastAPI app and defines a function that uses the Crunchbase API to collect data on companies in a specific industry. The function first checks if the data is already stored in Redis, and if it is, it returns the stored data. If the data is not stored in Redis, the function sends a request to the Crunchbase API and stores the resulting data in both Redis and PostgreSQL. The function also includes an error handler that returns an error message if an invalid industry name is passed as an input.

To use this function, you can make a GET request to the endpoint /market-growth/{industry_name} and pass the industry name

"""

"""
import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_market_growth(data):
    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Split the data into a training set and a test set
    train_set = df[:int(len(df) * 0.8)]
    test_set = df[int(len(df) * 0.8):]

    # Train a linear regression model on the training set
    model = LinearRegression()
    model.fit(train_set[["Year"]], train_set[["Market Growth"]])

    # Make predictions on the test set
    predictions = model.predict(test_set[["Year"]])

    # Calculate the mean squared error of the predictions
    mse = ((predictions - test_set[["Market Growth"]]) ** 2).mean()

    # Return the mean squared error
    return mse

This code defines a function that takes a data set as input and trains a linear regression model to predict market growth based on the year. It then makes predictions on a test set and calculates the mean squared error of the predictions.

To use this function with the data collected from Google Trends, you would need to pass the data to the function as a parameter. You can then use the returned mean squared error to evaluate the accuracy of the predictions.

"""

"""
import pandas as pd

def analyze_data(data):
    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Calculate statistical metrics
    mean = df["Market Growth"].mean()
    median = df["Market Growth"].median()
    std_dev = df["Market Growth"].std()
    min_val = df["Market Growth"].min()
    max_val = df["Market Growth"].max()
    quartiles = df["Market Growth"].quantile([0.25, 0.5, 0.75])

    # Return the calculated metrics
    return {
        "mean": mean,
        "median": median,
        "standard deviation": std_dev,
        "minimum value": min_val,
        "maximum value": max_val,
        "quartiles": quartiles
    }

This code defines a function that takes a data set as input and calculates a series of statistical metrics, including the mean, median, standard deviation, minimum value, maximum value, and quartiles of the "Market Growth" column. It then returns a dictionary with the calculated metrics.

To use this function with the data collected from Google Trends or the Crunchbase API, you would need to pass the data to the function as a parameter. You can then use the returned metrics to analyze the data and understand market trends.

"""


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
