from typing import List, Dict
from state.research_state import PlanStep, ResearchState
import os
from dotenv import load_dotenv
from tavily.client import TavilyClient

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
client = TavilyClient(TAVILY_API_KEY)


def tavily_search(query: str):
    EXCLUDED_DOMAINS = [
        # Authenticated / Paywalled
        "linkedin.com",
        "facebook.com",
        "instagram.com",
        "x.com",
        "twitter.com",
        "reddit.com",
        "quora.com",
        "medium.com",
        "substack.com",
        "patreon.com",
        "onlyfans.com",
        "bloomberg.com",
        "ft.com",
        "wsj.com",
        "economist.com",
        "jstor.org",
        "ieee.org",
        "sciencedirect.com",
        "springer.com",
        "nature.com",
        "lexisnexis.com",
        "westlaw.com",
        # E-commerce / Marketplaces
        "amazon.com",
        "ebay.com",
        "walmart.com",
        "target.com",
        "bestbuy.com",
        "homedepot.com",
        "lowes.com",
        "aliexpress.com",
        "etsy.com",
        "wayfair.com",
        "costco.com",
        "shopify.com",
        # Ticketing / Travel
        "ticketmaster.com",
        "livenation.com",
        "stubhub.com",
        "seatgeek.com",
        "expedia.com",
        "booking.com",
        "priceline.com",
        "kayak.com",
        "airbnb.com",
        "delta.com",
        "united.com",
        "americanairlines.com",
        # SaaS Dashboards / Cloud Consoles
        "aws.amazon.com",
        "console.aws.amazon.com",
        "azure.microsoft.com",
        "portal.azure.com",
        "cloud.google.com",
        "console.cloud.google.com",
        "stripe.com",
        "dashboard.stripe.com",
        "datadog.com",
        "newrelic.com",
        "grafana.com",
        "notion.so",
        "atlassian.net",
        "jira.com",
        # Social / Multimedia
        "tiktok.com",
        "youtube.com",
        "snapchat.com",
        "pinterest.com",
        "imgur.com",
        "flickr.com",
        "soundcloud.com",
        "spotify.com",
        "twitch.tv",
        # Government / Institutional
        "irs.gov",
        "sec.gov",
        "ssa.gov",
        "cdc.gov",
        "nih.gov",
        "who.int",
        "un.org",
        "loc.gov",
        "europa.eu",
        "gov.uk",
        # Forums / Communities
        "phpbb.com",
        "vbulletin.com",
        "invisioncommunity.com",
        "stackexchange.com",
        "stackoverflow.com",
        "superuser.com",
        "serverfault.com",
        # Media / News
        "nytimes.com",
        "washingtonpost.com",
        "cnn.com",
        "bbc.com",
        "theguardian.com",
        "forbes.com",
        "businessinsider.com",
        "vox.com",
    ]
    response = client.search(query, max_results=7, exclude_domains=EXCLUDED_DOMAINS)
    return response


def tavily_extract(url: str) -> Dict:
    response = client.extract(url)
    return response
