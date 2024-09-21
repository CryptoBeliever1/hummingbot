import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

from hummingbot.connector.exchange.kraken_v2 import kraken_v2_constants as CONSTANTS
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.web_assistant.connections.data_types import WSJSONRequest
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant
from hummingbot.logger import HummingbotLogger

# import json


if TYPE_CHECKING:
    from hummingbot.connector.exchange.kraken_v2.kraken_v2_exchange import KrakenV2Exchange


class KrakenV2APIUserStreamDataSource(UserStreamTrackerDataSource):
    _logger: Optional[HummingbotLogger] = None

    def __init__(self,
                 connector: 'KrakenV2Exchange',
                 api_factory: Optional[WebAssistantsFactory] = None):

        super().__init__()
        self._api_factory = api_factory
        self._connector = connector
        self._current_auth_token: Optional[str] = None
        self._rate_count = 0

    async def _connected_websocket_assistant(self) -> WSAssistant:
        ws: WSAssistant = await self._api_factory.get_ws_assistant()
        await ws.connect(ws_url=CONSTANTS.WS_AUTH_URL, ping_timeout=CONSTANTS.PING_TIMEOUT)
        return ws

    @property
    def last_recv_time(self):
        if self._ws_assistant is None:
            return 0
        else:
            return self._ws_assistant.last_recv_time

    async def get_auth_token(self) -> str:
        try:
            response_json = await self._connector._api_post(path_url=CONSTANTS.GET_TOKEN_PATH_URL, params={},
                                                            is_auth_required=True)
        except Exception:
            raise
        return response_json["token"]

    async def _subscribe_channels(self, websocket_assistant: WSAssistant):
        """
        Subscribes to order events and balance events.

        :param websocket_assistant: the websocket assistant used to connect to the exchange
        """
        try:

            if self._current_auth_token is None:
                self._current_auth_token = await self.get_auth_token()

            # orders_change_payload = {
            #     "event": "subscribe",
            #     "subscription": {
            #         "name": "openOrders",
            #         "token": self._current_auth_token,
            #         "ratecounter": True
            #     }
            # }
            # subscribe_order_change_request: WSJSONRequest = WSJSONRequest(payload=orders_change_payload)

            # trades_payload = {
            #     "event": "subscribe",
            #     "subscription": {
            #         "name": "ownTrades",
            #         "token": self._current_auth_token
            #     }
            # }

            trades_orders_change_payload = {
                "method": "subscribe",
                "params": {
                    "channel": "executions",
                    "token": self._current_auth_token,
                    "snap_orders": True,
                    "snap_trades": False,
                    "order_status": False,
                    "ratecounter": True
                }
            }


            trades_orders_change_request: WSJSONRequest = WSJSONRequest(payload=trades_orders_change_payload)

            balance_payload = {
                "method": "subscribe",
                "params": {
                    "channel": 'balances',
                    "token": self._current_auth_token,
                }
            }            
            subscribe_balance_request: WSJSONRequest = WSJSONRequest(payload=balance_payload)

            await websocket_assistant.send(trades_orders_change_request)
            await websocket_assistant.send(subscribe_balance_request)

            self.logger().info("Subscribed to private order changes trades updates and balance channels...")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().exception("Unexpected error occurred subscribing to user streams...")
            raise

    async def _process_event_message(self, event_message: Dict[str, Any], queue: asyncio.Queue):        
        if "channel" in event_message:            
            if event_message["channel"] in [
                CONSTANTS.USER_TRADES_ENDPOINT_NAME,
                CONSTANTS.USER_ORDERS_ENDPOINT_NAME,
                CONSTANTS.USER_BALANCE_ENDPOINT_NAME,
            ]:
                queue.put_nowait(event_message)
                # self.logger().info("put nowait")
                # self.extract_rate_count(event_message)
                       
        else:
            if event_message.get("errorMessage") is not None:
                err_msg = event_message.get("errorMessage")
                raise IOError({
                    "label": "WSS_ERROR",
                    "message": f"Error received via websocket - {err_msg}."
                })

    # def get_rate_count(self) -> int:
    #     return self._rate_count
    
    # def extract_rate_count(self, event_message):
    #     if event_message[-2] == CONSTANTS.USER_ORDERS_ENDPOINT_NAME:
    #         # self.logger().info(json.dumps(event_message, indent=4))
                
    #         # Check if the event_message contains the necessary nested structures
    #         if len(event_message) > 1:
    #             first_element = event_message[0]
    #             if first_element:
    #                 max_rate_count = -1
    #                 for order in first_element:
    #                     if isinstance(order, dict):
    #                         for order_id, order_details in order.items():
    #                             if "ratecount" in order_details:
    #                                 max_rate_count = max(max_rate_count, order_details["ratecount"])
    #                 if max_rate_count > -1:
    #                     self._rate_count = max_rate_count
    #                     # self.logger().info(f"self._rate_count changed in stream: {self._rate_count}")           