import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from hummingbot.connector.exchange.kraken_v2 import kraken_v2_constants as CONSTANTS, kraken_v2_web_utils as web_utils
from hummingbot.connector.exchange.kraken_v2.kraken_v2_order_book import KrakenV2OrderBook
from hummingbot.connector.exchange.kraken_v2.kraken_v2_utils import (
    convert_from_exchange_trading_pair,
    convert_to_exchange_trading_pair,
    rfc3339_to_unix,
)
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_message import OrderBookMessage
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, WSJSONRequest
from hummingbot.core.web_assistant.rest_assistant import RESTAssistant
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant
from hummingbot.logger import HummingbotLogger

if TYPE_CHECKING:
    from hummingbot.connector.exchange.kraken_v2.kraken_v2_exchange import KrakenV2Exchange


class KrakenV2APIOrderBookDataSource(OrderBookTrackerDataSource):
    MESSAGE_TIMEOUT = 30.0

    # PING_TIMEOUT = 10.0

    def __init__(self,
                 trading_pairs: List[str],
                 connector: 'KrakenV2Exchange',
                 api_factory: WebAssistantsFactory,
                 # throttler: Optional[AsyncThrottler] = None
                 ):
        super().__init__(trading_pairs)
        self._connector = connector
        self._api_factory = api_factory
        self._rest_assistant = None
        self._ws_assistant = None
        self._order_book_create_function = lambda: OrderBook()

    _kraobds_logger: Optional[HummingbotLogger] = None

    async def _get_rest_assistant(self) -> RESTAssistant:
        if self._rest_assistant is None:
            self._rest_assistant = await self._api_factory.get_rest_assistant()
        return self._rest_assistant

    async def get_last_traded_prices(self,
                                     trading_pairs: List[str],
                                     domain: Optional[str] = None) -> Dict[str, float]:
        return await self._connector.get_last_traded_prices(trading_pairs=trading_pairs)

    async def _order_book_snapshot(self, trading_pair: str) -> OrderBook:
        snapshot: Dict[str, Any] = await self._request_order_book_snapshot(trading_pair)
        snapshot_timestamp: float = time.time()
        snapshot_msg: OrderBookMessage = KrakenV2OrderBook.snapshot_message_from_exchange(
            snapshot,
            snapshot_timestamp,
            metadata={"trading_pair": trading_pair}
        )
        return snapshot_msg

    async def _request_order_book_snapshot(self, trading_pair: str, ) -> Dict[str, Any]:
        """
        Retrieves a copy of the full order book from the exchange, for a particular trading pair.

        :param trading_pair: the trading pair for which the order book will be retrieved

        :return: the response from the exchange (JSON dictionary)
        """
        params = {
            "pair": await self._connector.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        }

        rest_assistant = await self._api_factory.get_rest_assistant()
        response_json = await rest_assistant.execute_request(
            url=web_utils.public_rest_url(path_url=CONSTANTS.SNAPSHOT_PATH_URL),
            params=params,
            method=RESTMethod.GET,
            throttler_limit_id=CONSTANTS.SNAPSHOT_PATH_URL,
        )
        if len(response_json["error"]) > 0:
            raise IOError(f"Error fetching KrakenV2 REST market snapshot for {trading_pair}. "
                          f"Error is {response_json['error']}.")
        data: Dict[str, Any] = next(iter(response_json["result"].values()))
        data = {"trading_pair": trading_pair, **data}
        data["latest_update"] = max([*map(lambda x: x[2], data["bids"] + data["asks"])], default=0.)
        return data

    async def _subscribe_channels(self, ws: WSAssistant):
        """
        Subscribes to the trade events and diff orders events through the provided websocket connection.

        :param ws: the websocket assistant used to connect to the exchange
        """
        try:
            trading_pairs: List[str] = []
            for tp in self._trading_pairs:
                # trading_pairs.append(convert_to_exchange_trading_pair(tp, '/'))
                symbol = convert_to_exchange_trading_pair(tp, '/')
                trading_pairs.append(symbol)
            trades_payload = {
                "method": "subscribe",
                # "pair": trading_pairs,
                # "subscription": {"name": 'trade'},
                "params": {
                    "channel": 'trade',
                    "symbol": trading_pairs,
                    "snapshot": True
                }
            }
            subscribe_trade_request: WSJSONRequest = WSJSONRequest(payload=trades_payload)

            # order_book_payload = {
            #     "event": "subscribe",
            #     "pair": trading_pairs,
            #     "subscription": {"name": 'book', "depth": 1000},
            # }

            order_book_payload = {
                "method": "subscribe",
                "params": {
                    "channel": 'book',
                    "symbol": trading_pairs,
                }
            }            

            subscribe_orderbook_request: WSJSONRequest = WSJSONRequest(payload=order_book_payload)

            await ws.send(subscribe_trade_request)
            await ws.send(subscribe_orderbook_request)

            self.logger().info("Subscribing to public order book and trade channels...")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().error("Unexpected error occurred subscribing to order book data streams.")
            raise

    def _channel_originating_message(self, event_message) -> str:
        # self.logger().info(f"event_message: {event_message}")
        channel = ""

        if "data" in event_message and "type" in event_message and "channel" in event_message:
            if event_message["channel"] == CONSTANTS.TRADE_EVENT_TYPE:
                channel = self._trade_messages_queue_key
            elif event_message["channel"] == CONSTANTS.DIFF_EVENT_TYPE: 
                channel = self._diff_messages_queue_key
        else:
            if event_message.get("errorMessage") is not None:
                err_msg = event_message.get("errorMessage")
                raise IOError(f"Error event received from the server ({err_msg})")
            # else:
            #     err_msg = "No data, type or channel fields in the websocket response"
            #     raise IOError(f"Error event received from the server ({err_msg})")                
        return channel

    async def _connected_websocket_assistant(self) -> WSAssistant:
        ws: WSAssistant = await self._api_factory.get_ws_assistant()
        await ws.connect(ws_url=CONSTANTS.WS_URL,
                         ping_timeout=CONSTANTS.PING_TIMEOUT)
        return ws

    async def _parse_trade_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):

        trades = [
            {"pair": convert_from_exchange_trading_pair(trade["symbol"]), "trade": trade}
            for trade in raw_message["data"]
        ]
        for trade in trades:
            trade_msg: OrderBookMessage = KrakenV2OrderBook.trade_message_from_exchange(trade)
            message_queue.put_nowait(trade_msg)

    # async def _parse_order_book_diff_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
    #     msg_dict = {"trading_pair": convert_from_exchange_trading_pair(raw_message[-1]),
    #                 "asks": raw_message[1].get("a", []) or raw_message[1].get("as", []) or [],
    #                 "bids": raw_message[1].get("b", []) or raw_message[1].get("bs", []) or []}
    #     msg_dict["update_id"] = max(
    #         [*map(lambda x: float(x[2]), msg_dict["bids"] + msg_dict["asks"])], default=0.
    #     )
    #     if "as" in raw_message[1] and "bs" in raw_message[1]:
    #         order_book_message: OrderBookMessage = (
    #             KrakenV2OrderBook.snapshot_ws_message_from_exchange(msg_dict, time.time())
    #         )
    #     else:
    #         order_book_message: OrderBookMessage = KrakenV2OrderBook.diff_message_from_exchange(
    #             msg_dict, time.time())
    #     message_queue.put_nowait(order_book_message)

    async def _parse_order_book_diff_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
        data = raw_message["data"][0]
        trading_pair = convert_from_exchange_trading_pair(data["symbol"])

        # Determine update_id based on the presence of a timestamp field
        if "timestamp" in data:
            update_id = int(rfc3339_to_unix(data["timestamp"]))  # Convert to integer Unix timestamp
        else:
            update_id = int(time.time())  # Use current time as integer Unix timestamp if no timestamp field


        bids = [[str(bid["price"]), str(bid["qty"]), update_id] for bid in data["bids"]]
        asks = [[str(ask["price"]), str(ask["qty"]), update_id] for ask in data["asks"]]
        
        msg_dict = {
            "trading_pair": trading_pair,
            "bids": bids,
            "asks": asks,
            "update_id": update_id
        }

        if raw_message["type"] == "snapshot":
            order_book_message: OrderBookMessage = (
                KrakenV2OrderBook.snapshot_ws_message_from_exchange(msg_dict, time.time())
            )
        else:
            order_book_message: OrderBookMessage = KrakenV2OrderBook.diff_message_from_exchange(
                msg_dict, time.time())

        message_queue.put_nowait(order_book_message)