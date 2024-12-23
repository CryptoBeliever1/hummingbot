from enum import Enum
from typing import Dict, Tuple

from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit
from hummingbot.core.data_type.in_flight_order import OrderState

DEFAULT_DOMAIN = "kraken_v2"
MAX_ORDER_ID_LEN = 32
HBOT_ORDER_ID_PREFIX = "HBOT"

MAX_ID_BIT_COUNT = 31


class KrakenV2APITier(Enum):
    """
    kraken_v2's Private Endpoint Rate Limit Tiers, based on the Account Verification level.
    """
    STARTER = "STARTER"
    INTERMEDIATE = "INTERMEDIATE"
    PRO = "PRO"


# Values are calculated by adding the Maxiumum Counter value and the expected count decay(in a minute) of a given tier.
# Reference:
# - API Rate Limits: https://support.kraken.com/hc/en-us/articles/206548367-What-are-the-API-rate-limits
# - Matching Engine Limits: https://support.kraken.com/hc/en-us/articles/360045239571
STARTER_PRIVATE_ENDPOINT_LIMIT = 15 + 20
STARTER_MATCHING_ENGINE_LIMIT = 60 + 60
INTERMEDIATE_PRIVATE_ENDPOINT_LIMIT = 20 + 30
INTERMEDIATE_MATCHING_ENGINE_LIMIT = 125 + 140
PRO_PRIVATE_ENDPOINT_LIMIT = 20 + 60
PRO_MATCHING_ENGINE_LIMIT = 180 + 225

KRAKEN_V2_TIER_LIMITS: Dict[KrakenV2APITier, Tuple[int, int]] = {
    KrakenV2APITier.STARTER: (STARTER_PRIVATE_ENDPOINT_LIMIT, STARTER_MATCHING_ENGINE_LIMIT),
    KrakenV2APITier.INTERMEDIATE: (INTERMEDIATE_PRIVATE_ENDPOINT_LIMIT, INTERMEDIATE_MATCHING_ENGINE_LIMIT),
    KrakenV2APITier.PRO: (PRO_PRIVATE_ENDPOINT_LIMIT, PRO_MATCHING_ENGINE_LIMIT),
}

KRAKEN_V2_TO_HB_MAP = {
    "XBT": "BTC",
    "XDG": "DOGE",
}

BASE_URL = "https://api.kraken.com"
TICKER_PATH_URL = "/0/public/Ticker"
SNAPSHOT_PATH_URL = "/0/public/Depth"
ASSET_PAIRS_PATH_URL = "/0/public/AssetPairs"
TIME_PATH_URL = "/0/public/Time"
GET_TOKEN_PATH_URL = "/0/private/GetWebSocketsToken"
ADD_ORDER_PATH_URL = "/0/private/AddOrder"
AMEND_ORDER_PATH_URL = "/0/private/AmendOrder"
CANCEL_ORDER_PATH_URL = "/0/private/CancelOrder"
BALANCE_PATH_URL = "/0/private/Balance"
OPEN_ORDERS_PATH_URL = "/0/private/OpenOrders"
QUERY_ORDERS_PATH_URL = "/0/private/QueryOrders"
QUERY_TRADES_PATH_URL = "/0/private/QueryTrades"


UNKNOWN_ORDER_MESSAGE = "Unknown order"
# Order States
ORDER_STATE = {
    "pending": OrderState.OPEN,
    "open": OrderState.OPEN,
    "closed": OrderState.FILLED,
    "canceled": OrderState.CANCELED,
    "expired": OrderState.FAILED,
}

WS_ORDER_STATE = {
    "pending_new": OrderState.OPEN,
    "new": OrderState.OPEN,
    "partially_filled": OrderState.PARTIALLY_FILLED,
    "filled": OrderState.FILLED,
    "canceled": OrderState.CANCELED,
    "expired": OrderState.FAILED,
}

ORDER_NOT_EXIST_ERROR_CODE = "Error fetching status update for the order"

WS_URL = "wss://ws.kraken.com/v2"
WS_AUTH_URL = "wss://ws-auth.kraken.com/v2"

OPEN_ORDERS_CHANNEL_NAME = "openOrders"

DIFF_EVENT_TYPE = "book"
TRADE_EVENT_TYPE = "trade"
PING_TIMEOUT = 10
USER_TRADES_ENDPOINT_NAME = "executions"
USER_ORDERS_ENDPOINT_NAME = "executions"
USER_BALANCE_ENDPOINT_NAME = "balances"

PUBLIC_ENDPOINT_LIMIT_ID = "PublicEndpointLimitID"
PUBLIC_ENDPOINT_LIMIT = 2
PUBLIC_ENDPOINT_LIMIT_INTERVAL = 1
PRIVATE_ENDPOINT_LIMIT_ID = "PrivateEndpointLimitID"
PRIVATE_ENDPOINT_LIMIT_INTERVAL = 60
MATCHING_ENGINE_LIMIT_ID = "MatchingEngineLimitID"
MATCHING_ENGINE_LIMIT_INTERVAL = 60
WS_CONNECTION_LIMIT_ID = "WSConnectionLimitID"

PUBLIC_API_LIMITS = [
    # Public API Pool
    RateLimit(
        limit_id=PUBLIC_ENDPOINT_LIMIT_ID,
        limit=PUBLIC_ENDPOINT_LIMIT,
        time_interval=PUBLIC_ENDPOINT_LIMIT_INTERVAL,
    ),
    # Public Endpoints
    RateLimit(
        limit_id=SNAPSHOT_PATH_URL,
        limit=PUBLIC_ENDPOINT_LIMIT,
        time_interval=PUBLIC_ENDPOINT_LIMIT_INTERVAL,
        linked_limits=[LinkedLimitWeightPair(PUBLIC_ENDPOINT_LIMIT_ID)],
    ),
    RateLimit(
        limit_id=ASSET_PAIRS_PATH_URL,
        limit=PUBLIC_ENDPOINT_LIMIT,
        time_interval=PUBLIC_ENDPOINT_LIMIT_INTERVAL,
        linked_limits=[LinkedLimitWeightPair(PUBLIC_ENDPOINT_LIMIT_ID)],
    ),
    RateLimit(
        limit_id=TICKER_PATH_URL,
        limit=PUBLIC_ENDPOINT_LIMIT,
        time_interval=PUBLIC_ENDPOINT_LIMIT_INTERVAL,
        linked_limits=[LinkedLimitWeightPair(PUBLIC_ENDPOINT_LIMIT_ID)],
    ),
    RateLimit(
        limit_id=TIME_PATH_URL,
        limit=PUBLIC_ENDPOINT_LIMIT,
        time_interval=PUBLIC_ENDPOINT_LIMIT_INTERVAL,
        linked_limits=[LinkedLimitWeightPair(PUBLIC_ENDPOINT_LIMIT_ID)],
    ),   
    # WebSocket Connection Limit
    RateLimit(limit_id=WS_CONNECTION_LIMIT_ID,
              limit=150,
              time_interval=60 * 10),
]
