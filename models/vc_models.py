from pydantic import BaseModel
from typing import Any, List

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
