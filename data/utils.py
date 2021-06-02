"""Utils for data"""

from typing import List


def build_holder_relation(symbols: List[str], api_call, fields, threshold=1):
    """Parses all holder information (institutional, mutual) to construct a relation matrix"""
    companies = list(
        filter(
            None,
            [api_call(symbol) for symbol in fields],
        )
    )

    companies_sorted_and_stripped = [
        sorted(holders, key=lambda holder: int(holder["shares"]), reverse=True)[:10]
        for holders in companies
    ]

    company_holders = {}
    for idx, holders in enumerate(companies_sorted_and_stripped):
        company_holders[symbols[idx]] = [holder["holder"] for holder in holders]

    holder_data = []
    for symbol in symbols:
        holder_dict = {}
        holder_dict["symbol"] = symbol
        for company, holders in company_holders.items():
            # Threshold to make the relation a bit more meaningful
            if len(set(holders) & set(company_holders[symbol])) > threshold:
                holder_dict[company] = 1
            else:
                holder_dict[company] = 0
        holder_data.append(holder_dict)

    return holder_data
