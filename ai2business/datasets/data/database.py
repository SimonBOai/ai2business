""" Class type database of importan KPIs.

Due to the problematic of relative and absolute import of paths. Hence, the is no 
collection provided as `json`-file, it is provided as `dict` in a class attribute.
"""


class StockMarket:
    """StockMarket contains companies and their ISIN of the leading stock markets world-
    wide.

    Attributes:
        dowjones (dict): Collection of the 30 leading companies in the Dow Joens.
    """

    dowjones = {
        "American Express Co": "AXP",
        "Amgen Inc": "AMGN",
        "Apple Inc": "AAPL",
        "Boeing Co": "BA",
        "Caterpillar Inc": "CAT",
        "Cisco Systems Inc": "CSCO",
        "Chevron Corp": "CVX",
        "Goldman Sachs Group Inc": "GS",
        "Home Depot Inc": "HD",
        "Honeywell International Inc": "HON",
        "International Business Machines Corp": "IBM",
        "Intel Corp": "INTC",
        "Johnson & Johnson": "JNJ",
        "Coca-Cola Co": "KO",
        "JPMorgan Chase & Co": "JPM",
        "McDonald's Corp": "MCD",
        "3M Co": "MMM",
        "Merck & Co Inc": "MRK",
        "Microsoft Corp": "MSFT",
        "Nike Inc": "NKE",
        "Procter & Gamble Co": "PG",
        "Travelers Companies Inc": "TRV",
        "UnitedHealth Group Inc": "UNH",
        "Salesforce.Com Inc": "CRM",
        "Verizon Communications Inc": "VZ",
        "Visa Inc": "V",
        "Walgreens Boots Alliance Inc": "WBA",
        "Walmart Inc": "WMT",
        "Walt Disney Co": "DIS",
        "Dow Inc": "DOW",
    }
