from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Tuple

from pydantic import Field, SecretStr
from pydantic.class_validators import validator

import hummingbot.connector.exchange.kraken_v2.kraken_v2_constants as CONSTANTS
from hummingbot.client.config.config_data_types import BaseConnectorConfigMap, ClientFieldData
from hummingbot.connector.exchange.kraken_v2.kraken_v2_constants import KrakenV2APITier
from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit
from hummingbot.core.data_type.trade_fee import TradeFeeSchema

CENTRALIZED = True

EXAMPLE_PAIR = "ETH-USDC"

DEFAULT_FEES = TradeFeeSchema(
    maker_percent_fee_decimal=Decimal("0.16"),
    taker_percent_fee_decimal=Decimal("0.26"),
)


def convert_from_exchange_symbol(symbol: str) -> str:
    # Assuming if starts with Z or X and has 4 letters then Z/X is removable
    if (symbol[0] == "X" or symbol[0] == "Z") and len(symbol) == 4:
        symbol = symbol[1:]
    return CONSTANTS.KRAKEN_V2_TO_HB_MAP.get(symbol, symbol)

def convert_from_ws_exchange_symbol(symbol: str) -> str:
    # Assuming if starts with Z or X and has 4 letters then Z/X is removable
    if (symbol[0] == "X" or symbol[0] == "Z") and len(symbol) == 4:
        symbol = symbol[1:]
    return CONSTANTS.KRAKEN_V2_TO_HB_MAP.get(symbol, symbol)

def convert_to_exchange_symbol(symbol: str) -> str:
    inverted_kraken_v2_to_hb_map = {v: k for k, v in CONSTANTS.KRAKEN_V2_TO_HB_MAP.items()}
    return inverted_kraken_v2_to_hb_map.get(symbol, symbol)

def split_to_base_quote(exchange_trading_pair: str) -> Tuple[Optional[str], Optional[str]]:
    base, quote = exchange_trading_pair.split("-")
    return base, quote


def convert_from_exchange_trading_pair(exchange_trading_pair: str, available_trading_pairs: Optional[Tuple] = None) -> \
        Optional[str]:
    base, quote = "", ""
    if "-" in exchange_trading_pair:
        base, quote = split_to_base_quote(exchange_trading_pair)
    elif "/" in exchange_trading_pair:
        base, quote = exchange_trading_pair.split("/")
    elif len(available_trading_pairs) > 0:
        # If trading pair has no spaces (i.e. ETHUSDT). Then it will have to match with the existing pairs
        # Option 1: Using traditional naming convention
        connector_trading_pair = {''.join(convert_from_exchange_trading_pair(tp).split('-')): tp for tp in
                                  available_trading_pairs}.get(
            exchange_trading_pair)
        if connector_trading_pair:    
            # print(f"Option 1. There was no splitter in pair {exchange_trading_pair} and now it's {connector_trading_pair}")
            connector_trading_pair = convert_from_exchange_trading_pair(connector_trading_pair)
            # print(f"Option 1 FIX. New pair = {connector_trading_pair}")
        if not connector_trading_pair:
            # Option 2: Using kraken_v2 naming convention ( XXBT for Bitcoin, XXDG for Doge, ZUSD for USD, etc)
            connector_trading_pair = {''.join(tp.split('-')): tp for tp in available_trading_pairs}.get(
                exchange_trading_pair)
            if connector_trading_pair:
                # print(f"Option 2. There was no splitter in pair {exchange_trading_pair} and now it's {connector_trading_pair}")
                connector_trading_pair = convert_from_exchange_trading_pair(connector_trading_pair)
            #     connector_trading_pair = convert_from_exchange_trading_pair(connector_trading_pair)
            if not connector_trading_pair:
                # Option 3: KrakenV2 naming convention but without the initial X and Z
                connector_trading_pair = {''.join(convert_to_exchange_symbol(convert_from_exchange_symbol(s))
                                                  for s in tp.split('-')): tp
                                          for tp in available_trading_pairs}.get(exchange_trading_pair)
                if connector_trading_pair:
                    # print(f"Option 3. There was no splitter in pair {exchange_trading_pair} and now it's {connector_trading_pair}")
                    connector_trading_pair = convert_from_exchange_trading_pair(connector_trading_pair)
        return connector_trading_pair

    if not base or not quote:
        return None
    base = convert_from_exchange_symbol(base)
    quote = convert_from_exchange_symbol(quote)
    # print(f"The trading pair had a splitter initially ({exchange_trading_pair}) and now it's {base}-{quote}")
    return f"{base}-{quote}"


def convert_to_exchange_trading_pair(hb_trading_pair: str, delimiter: str = "") -> str:
    """
    Note: The result of this method can safely be used to submit/make queries.
    Result shouldn't be used to parse responses as KrakenV2 add special formating to most pairs.
    """
    if "-" in hb_trading_pair:
        base, quote = hb_trading_pair.split("-")
    elif "/" in hb_trading_pair:
        base, quote = hb_trading_pair.split("/")
    else:
        return hb_trading_pair
    base = convert_to_exchange_symbol(base)
    quote = convert_to_exchange_symbol(quote)

    exchange_trading_pair = f"{base}{delimiter}{quote}"
    return exchange_trading_pair


def _build_private_rate_limits(tier: KrakenV2APITier = KrakenV2APITier.STARTER) -> List[RateLimit]:
    private_rate_limits = []

    PRIVATE_ENDPOINT_LIMIT, MATCHING_ENGINE_LIMIT = CONSTANTS.KRAKEN_V2_TIER_LIMITS[tier]

    # Private REST endpoints
    private_rate_limits.extend([
        # Private API Pool
        RateLimit(
            limit_id=CONSTANTS.PRIVATE_ENDPOINT_LIMIT_ID,
            limit=PRIVATE_ENDPOINT_LIMIT,
            time_interval=CONSTANTS.PRIVATE_ENDPOINT_LIMIT_INTERVAL,
        ),
        # Private endpoints
        RateLimit(
            limit_id=CONSTANTS.GET_TOKEN_PATH_URL,
            limit=PRIVATE_ENDPOINT_LIMIT,
            time_interval=CONSTANTS.PRIVATE_ENDPOINT_LIMIT_INTERVAL,
            linked_limits=[LinkedLimitWeightPair(CONSTANTS.PRIVATE_ENDPOINT_LIMIT_ID)],
        ),
        RateLimit(
            limit_id=CONSTANTS.BALANCE_PATH_URL,
            limit=PRIVATE_ENDPOINT_LIMIT,
            time_interval=CONSTANTS.PRIVATE_ENDPOINT_LIMIT_INTERVAL,
            weight=2,
            linked_limits=[LinkedLimitWeightPair(CONSTANTS.PRIVATE_ENDPOINT_LIMIT_ID)],
        ),
        RateLimit(
            limit_id=CONSTANTS.OPEN_ORDERS_PATH_URL,
            limit=PRIVATE_ENDPOINT_LIMIT,
            time_interval=CONSTANTS.PRIVATE_ENDPOINT_LIMIT_INTERVAL,
            weight=2,
            linked_limits=[LinkedLimitWeightPair(CONSTANTS.PRIVATE_ENDPOINT_LIMIT_ID)],
        ),
        RateLimit(
            limit_id=CONSTANTS.QUERY_ORDERS_PATH_URL,
            limit=PRIVATE_ENDPOINT_LIMIT,
            time_interval=CONSTANTS.PRIVATE_ENDPOINT_LIMIT_INTERVAL,
            weight=2,
            linked_limits=[LinkedLimitWeightPair(CONSTANTS.PRIVATE_ENDPOINT_LIMIT_ID)],
        ),
        RateLimit(
            limit_id=CONSTANTS.QUERY_TRADES_PATH_URL,
            limit=PRIVATE_ENDPOINT_LIMIT,
            time_interval=CONSTANTS.PRIVATE_ENDPOINT_LIMIT_INTERVAL,
            weight=2,
            linked_limits=[LinkedLimitWeightPair(CONSTANTS.PRIVATE_ENDPOINT_LIMIT_ID)],
        ),
    ])

    # Matching Engine Limits
    private_rate_limits.extend([
        RateLimit(
            limit_id=CONSTANTS.ADD_ORDER_PATH_URL,
            limit=MATCHING_ENGINE_LIMIT,
            time_interval=CONSTANTS.MATCHING_ENGINE_LIMIT_INTERVAL,
            linked_limits=[LinkedLimitWeightPair(CONSTANTS.MATCHING_ENGINE_LIMIT_ID)],
        ),
        RateLimit(
            limit_id=CONSTANTS.CANCEL_ORDER_PATH_URL,
            limit=MATCHING_ENGINE_LIMIT,
            time_interval=CONSTANTS.MATCHING_ENGINE_LIMIT_INTERVAL,
            linked_limits=[LinkedLimitWeightPair(CONSTANTS.MATCHING_ENGINE_LIMIT_ID)],
        ),
        RateLimit(
            limit_id=CONSTANTS.AMEND_ORDER_PATH_URL,
            limit=MATCHING_ENGINE_LIMIT,
            time_interval=CONSTANTS.MATCHING_ENGINE_LIMIT_INTERVAL,
            linked_limits=[LinkedLimitWeightPair(CONSTANTS.MATCHING_ENGINE_LIMIT_ID)],
        ),        
    ])

    return private_rate_limits


def build_rate_limits_by_tier(tier: KrakenV2APITier = KrakenV2APITier.STARTER) -> List[RateLimit]:
    rate_limits = []

    rate_limits.extend(CONSTANTS.PUBLIC_API_LIMITS)
    rate_limits.extend(_build_private_rate_limits(tier=tier))

    return rate_limits


def rfc3339_to_unix(rfc3339_timestamp: str) -> float:
    """
    Convert an RFC3339 timestamp to a Unix timestamp.

    Args:
    rfc3339_timestamp (str): The RFC3339 formatted timestamp string.

    Returns:
    float: The corresponding Unix timestamp.
    """
    # Replace 'Z' with '+00:00' to make the string ISO 8601 compliant
    if rfc3339_timestamp.endswith('Z'):
        rfc3339_timestamp = rfc3339_timestamp.replace('Z', '+00:00')
    
    # Convert the RFC3339 timestamp to a datetime object
    dt = datetime.fromisoformat(rfc3339_timestamp)
    
    # Convert the datetime object to Unix timestamp
    unix_timestamp = dt.timestamp()
    
    return unix_timestamp

class KrakenV2ConfigMap(BaseConnectorConfigMap):
    connector: str = Field(default="kraken_v2", client_data=None)
    kraken_v2_api_key: SecretStr = Field(
        default=...,
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter your KrakenV2 API key",
            is_secure=True,
            is_connect_key=True,
            prompt_on_new=True,
        )
    )
    kraken_v2_secret_key: SecretStr = Field(
        default=...,
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter your KrakenV2 secret key",
            is_secure=True,
            is_connect_key=True,
            prompt_on_new=True,
        )
    )
    kraken_v2_api_tier: str = Field(
        default="Starter",
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter your KrakenV2 API Tier (Starter/Intermediate/Pro)",
            is_connect_key=True,
            prompt_on_new=True,
        )
    )

    class Config:
        title = "kraken_v2"

    @validator("kraken_v2_api_tier", pre=True)
    def _api_tier_validator(cls, value: str) -> Optional[str]:
        """
        Determines if input value is a valid API tier
        """
        try:
            KrakenV2APITier(value.upper())
            return value
        except ValueError:
            raise ValueError("No such KrakenV2 API Tier.")

KEYS = KrakenV2ConfigMap.construct()
