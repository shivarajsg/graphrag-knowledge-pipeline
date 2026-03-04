"""
mock_data.py
------------
Pre-extracted entity / relation / event data for the three Swiggy test
chunks.  Used when --mock flag is set so the pipeline runs end-to-end
without any LLM API calls.

Each entry maps chunk_id → extraction dict that would otherwise be
produced by the LLM extraction step.
"""

MOCK_EXTRACTIONS: dict = {

    # ── Chunk 5 – Platform structure ─────────────────────────────────────────
    "5": {
        "chunk_id": "5",
        "entities": [
            {
                "name": "Swiggy",
                "type": "Company",
                "attributes": {
                    "description": "new-age consumer-first technology company",
                    "platform_type": "unified",
                },
            },
            {
                "name": "Consumer Convenience Platform",
                "type": "Platform",
                "attributes": {
                    "principle": "convenience-first",
                    "service_categories": "4",
                    "partner_categories": "4",
                    "foundational_capabilities": "4",
                },
            },
            {"name": "Food Delivery",                   "type": "Service",     "attributes": {}},
            {"name": "Dining Out and Events",           "type": "Service",     "attributes": {}},
            {"name": "Grocery and Household Items",     "type": "Service",     "attributes": {}},
            {"name": "Other Hyperlocal Services",       "type": "Service",     "attributes": {}},
            {"name": "Restaurant Partners",             "type": "Partner",     "attributes": {}},
            {"name": "Merchant Partners",               "type": "Partner",     "attributes": {}},
            {"name": "Brand Partners",                  "type": "Partner",     "attributes": {}},
            {"name": "Delivery Partners",               "type": "Partner",     "attributes": {}},
            {"name": "Technology",                      "type": "Capability",  "attributes": {}},
            {"name": "Analytics",                       "type": "Capability",  "attributes": {}},
            {"name": "Fulfilment",                      "type": "Capability",  "attributes": {}},
            {
                "name": "Swiggy One",
                "type": "Programme",
                "attributes": {"description": "Membership programme", "brand": "Swiggy One"},
            },
            {"name": "Swiggy Limited IPO Prospectus",  "type": "Source",      "attributes": {}},
        ],
        "relations": [
            {"source": "Swiggy",                          "target": "Consumer Convenience Platform", "relation": "operates"},
            {"source": "Consumer Convenience Platform",   "target": "Food Delivery",                 "relation": "offers"},
            {"source": "Consumer Convenience Platform",   "target": "Dining Out and Events",         "relation": "offers"},
            {"source": "Consumer Convenience Platform",   "target": "Grocery and Household Items",   "relation": "offers"},
            {"source": "Consumer Convenience Platform",   "target": "Other Hyperlocal Services",     "relation": "offers"},
            {"source": "Consumer Convenience Platform",   "target": "Restaurant Partners",           "relation": "supported_by"},
            {"source": "Consumer Convenience Platform",   "target": "Merchant Partners",             "relation": "supported_by"},
            {"source": "Consumer Convenience Platform",   "target": "Brand Partners",                "relation": "supported_by"},
            {"source": "Consumer Convenience Platform",   "target": "Delivery Partners",             "relation": "supported_by"},
            {"source": "Consumer Convenience Platform",   "target": "Technology",                    "relation": "enabled_by"},
            {"source": "Consumer Convenience Platform",   "target": "Analytics",                     "relation": "enabled_by"},
            {"source": "Consumer Convenience Platform",   "target": "Fulfilment",                    "relation": "enabled_by"},
            {"source": "Consumer Convenience Platform",   "target": "Swiggy One",                    "relation": "enabled_by"},
        ],
        "events": [],
        "sources": ["Swiggy Limited IPO Prospectus"],
    },

    # ── Chunk 6 – Strategy, capabilities & products ──────────────────────────
    "6": {
        "chunk_id": "6",
        "entities": [
            {
                "name": "Swiggy",
                "type": "Company",
                "attributes": {
                    "strategy": "High-frequency Hyperlocal Commerce",
                    "culture": "Innovation-led DNA",
                    "approach": "convenience-first",
                },
            },
            {
                "name": "Consumer Convenience Platform",
                "type": "Platform",
                "attributes": {
                    "value_props": "Fast delivery, Quality & assortment",
                    "capability": "Personalised recommendations",
                },
            },
            {
                "name": "Technology",
                "type": "Capability",
                "attributes": {"asset": "Reusable tech stack (quick, low-cost innovation)"},
            },
            {
                "name": "Swiggy Instamart",
                "type": "Product",
                "attributes": {
                    "launch_year": "2020",
                    "scale": "124 cities as of 31-Mar-2025",
                    "initial_cities": "2",
                    "category": "Quick Commerce",
                },
            },
            {
                "name": "Dineout",
                "type": "Acquisition",
                "attributes": {"acquired_year": "2022", "purpose": "expanded platform capabilities"},
            },
            {
                "name": "Swiggy One",
                "type": "Programme",
                "attributes": {"description": "Membership programme"},
            },
            {
                "name": "Bolt",
                "type": "Feature",
                "attributes": {"description": "10-minute food delivery", "launch_year": "2024"},
            },
            {"name": "Quick Bites",                  "type": "Product",  "attributes": {}},
            {
                "name": "Co-branded Credit Card HDFC",
                "type": "Product",
                "attributes": {"partner": "HDFC Bank", "launch_year": "2023"},
            },
            {"name": "Restaurant Reservations",      "type": "Service",  "attributes": {}},
        ],
        "relations": [
            {"source": "Swiggy",                        "target": "Dineout",                       "relation": "acquired"},
            {"source": "Grocery and Household Items",   "target": "Swiggy Instamart",              "relation": "executed_via"},
            {"source": "Food Delivery",                 "target": "Bolt",                          "relation": "feature"},
            {"source": "Swiggy One",                    "target": "Food Delivery",                 "relation": "benefits_across"},
            {"source": "Swiggy One",                    "target": "Grocery and Household Items",   "relation": "benefits_across"},
            {"source": "Swiggy One",                    "target": "Dining Out and Events",         "relation": "benefits_across"},
            {"source": "Swiggy One",                    "target": "Quick Bites",                   "relation": "includes"},
            {"source": "Swiggy One",                    "target": "Co-branded Credit Card HDFC",   "relation": "includes"},
            {"source": "Swiggy One",                    "target": "Bolt",                          "relation": "includes"},
        ],
        "events": [
            {
                "year": "2014",
                "description": "Pioneer in Food Delivery",
                "tags": ["Pioneer", "Launch"],
                "entities_involved": ["Swiggy", "Food Delivery"],
            },
            {
                "year": "2020",
                "description": "Pioneer in Quick Commerce with Instamart",
                "tags": ["Pioneer", "Launch"],
                "entities_involved": ["Swiggy", "Swiggy Instamart"],
            },
        ],
        "sources": [],
    },

    # ── Chunk 7 – Full timeline ───────────────────────────────────────────────
    "7": {
        "chunk_id": "7",
        "entities": [
            {
                "name": "Swiggy",
                "type": "Company",
                "attributes": {
                    "incorporation_city": "Bengaluru",
                    "incorporation_year": "2013",
                    "users_2025": "120 million transacted users",
                },
            },
            {
                "name": "Swiggy Instamart",
                "type": "Product",
                "attributes": {
                    "launch_year": "2020",
                    "scale_2022": "25 cities, 400+ dark stores, 8,400+ SKUs",
                    "scale_2025": "124 cities",
                },
            },
            {"name": "Swiggy Genie",        "type": "Product",     "attributes": {"launch_year": "2020"}},
            {
                "name": "Swiggy One",
                "type": "Programme",
                "attributes": {
                    "launch_year": "2021",
                    "members_2024": "5.7 million",
                },
            },
            {"name": "Dineout",             "type": "Acquisition",  "attributes": {"acquired_year": "2022"}},
            {"name": "Swiggy Minis",        "type": "Product",      "attributes": {"launch_year": "2022"}},
            {
                "name": "Co-branded Credit Card HDFC",
                "type": "Product",
                "attributes": {"launch_year": "2023", "partner": "HDFC Bank"},
            },
            {
                "name": "Lynk",
                "type": "Acquisition",
                "attributes": {"acquired_year": "2023", "stake": "100%"},
            },
            {
                "name": "Swiggy Mall",
                "type": "Product",
                "attributes": {"launch_year": "2023", "integrated_with": "Swiggy Instamart"},
            },
            {
                "name": "Bolt",
                "type": "Feature",
                "attributes": {"launch_year": "2024", "description": "10-minute food delivery"},
            },
            {
                "name": "Swiggy Scenes",
                "type": "Product",
                "attributes": {"launch_year": "2024", "description": "Events and experiences"},
            },
            {"name": "SNACC",               "type": "Product",  "attributes": {"launch_year": "2025"}},
            {"name": "Pyng",                "type": "Product",  "attributes": {"launch_year": "2025"}},
            {"name": "Swiggy Limited IPO Prospectus", "type": "Source", "attributes": {}},
            {"name": "Annual Report FY 2024-25",      "type": "Source", "attributes": {}},
        ],
        "relations": [
            {"source": "Swiggy",                    "target": "Dineout",                        "relation": "acquired"},
            {"source": "Swiggy",                    "target": "Lynk",                           "relation": "acquired"},
            {"source": "Swiggy Mall",               "target": "Swiggy Instamart",               "relation": "integrated_surface"},
            {"source": "Dining Out and Events",     "target": "Swiggy Scenes",                  "relation": "events_experiences_surface"},
            {"source": "Other Hyperlocal Services", "target": "Swiggy Genie",                   "relation": "includes"},
            {"source": "Other Hyperlocal Services", "target": "Swiggy Minis",                   "relation": "includes"},
            {"source": "Grocery and Household Items","target": "Swiggy Mall",                    "relation": "integrated_surface"},
            {"source": "Dineout",                   "target": "Dining Out and Events",          "relation": "expanded_surface"},
        ],
        "events": [
            {
                "year": "2013",
                "description": "Incorporation of the company in Bengaluru",
                "tags": ["Corporate"],
                "entities_involved": ["Swiggy"],
            },
            {
                "year": "2014",
                "description": "Launch of Food Delivery business",
                "tags": ["Launch", "Pioneer"],
                "entities_involved": ["Swiggy", "Food Delivery"],
            },
            {
                "year": "2015",
                "description": "First major fundraise completed",
                "tags": ["Fundraise"],
                "entities_involved": ["Swiggy"],
            },
            {
                "year": "2019",
                "description": "Food delivery business expanded to 500+ cities",
                "tags": ["Milestone", "Scale"],
                "entities_involved": ["Swiggy", "Food Delivery"],
            },
            {
                "year": "2020",
                "description": "Launch of Swiggy Instamart and Swiggy Genie",
                "tags": ["Launch", "Pioneer"],
                "entities_involved": ["Swiggy Instamart", "Swiggy Genie"],
            },
            {
                "year": "2021",
                "description": "Launch of Swiggy One membership programme",
                "tags": ["Launch"],
                "entities_involved": ["Swiggy One"],
            },
            {
                "year": "2022",
                "description": "Acquired and integrated Dineout; Instamart expanded to 25 cities "
                               "with 400+ dark stores and 8,400+ SKUs; launched Swiggy Minis",
                "tags": ["Acquisition", "Launch", "Scale"],
                "entities_involved": ["Dineout", "Swiggy Instamart", "Swiggy Minis"],
            },
            {
                "year": "2023",
                "description": "Launched co-branded HDFC credit card; acquired 100% stake in Lynk; "
                               "launched Swiggy Mall (integrated with Instamart)",
                "tags": ["Launch", "Acquisition"],
                "entities_involved": ["Co-branded Credit Card HDFC", "Lynk", "Swiggy Mall"],
            },
            {
                "year": "2024",
                "description": "Public listing; Swiggy One membership crossed 5.7M members; "
                               "launched Bolt (10-min delivery) and Swiggy Scenes (events)",
                "tags": ["Listing", "Launch", "Milestone"],
                "entities_involved": ["Swiggy One", "Bolt", "Swiggy Scenes"],
            },
            {
                "year": "2025",
                "description": "Surpassed 120M transacted users; launched SNACC and Pyng",
                "tags": ["Milestone", "Launch"],
                "entities_involved": ["SNACC", "Pyng"],
            },
        ],
        "sources": ["Swiggy Limited IPO Prospectus", "Annual Report FY 2024-25"],
    },
}
