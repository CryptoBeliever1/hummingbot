import asyncio
import re

# import json
from collections import defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from bidict import bidict

from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.connector.constants import s_decimal_0, s_decimal_NaN
from hummingbot.connector.exchange.kraken_v2 import kraken_v2_constants as CONSTANTS, kraken_v2_web_utils as web_utils
from hummingbot.connector.exchange.kraken_v2.kraken_v2_api_order_book_data_source import KrakenV2APIOrderBookDataSource
from hummingbot.connector.exchange.kraken_v2.kraken_v2_api_user_stream_data_source import (
    KrakenV2APIUserStreamDataSource,
)
from hummingbot.connector.exchange.kraken_v2.kraken_v2_auth import KrakenV2Auth
from hummingbot.connector.exchange.kraken_v2.kraken_v2_constants import KrakenV2APITier
from hummingbot.connector.exchange.kraken_v2.kraken_v2_utils import (
    build_rate_limits_by_tier,
    convert_from_exchange_symbol,
    convert_from_exchange_trading_pair,
    convert_from_ws_exchange_symbol,
    rfc3339_to_unix,
)
from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import get_new_numeric_client_order_id
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.data_type.common import OpenOrder, OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.data_type.trade_fee import TokenAmount, TradeFeeBase
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.core.utils.estimate_fee import build_trade_fee
from hummingbot.core.utils.tracking_nonce import NonceCreator
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory

if TYPE_CHECKING:
    from hummingbot.client.config.config_helpers import ClientConfigAdapter


class KrakenV2Exchange(ExchangePyBase):
    UPDATE_ORDER_STATUS_MIN_INTERVAL = 10.0
    SHORT_POLL_INTERVAL = 60.0
    LONG_POLL_INTERVAL = 120.0

    web_utils = web_utils
    REQUEST_ATTEMPTS = 5

    def __init__(self,
                 client_config_map: "ClientConfigAdapter",
                 kraken_v2_api_key: str,
                 kraken_v2_secret_key: str,
                 trading_pairs: Optional[List[str]] = None,
                 trading_required: bool = True,
                 domain: str = CONSTANTS.DEFAULT_DOMAIN,
                 kraken_v2_api_tier: str = "starter"
                 ):
        self._real_time_balance_update = True
        self.api_key = kraken_v2_api_key
        self.secret_key = kraken_v2_secret_key
        self._domain = domain
        self._trading_required = trading_required
        self._trading_pairs = trading_pairs
        self._kraken_v2_api_tier = KrakenV2APITier(kraken_v2_api_tier.upper())
        self._asset_pairs = {}
        self._client_config = client_config_map
        self._client_order_id_nonce_provider = NonceCreator.for_microseconds()
        self._throttler = self._build_async_throttler(api_tier=self._kraken_v2_api_tier)
        self.rate_count = 0
        self.rate_count_update_timestamp = None
        
        super().__init__(client_config_map)

    @staticmethod
    def kraken_v2_order_type(order_type: OrderType) -> str:
        return order_type.name.lower()

    @staticmethod
    def to_hb_order_type(kraken_v2_type: str) -> OrderType:
        return OrderType[kraken_v2_type]

    @property
    def authenticator(self):
        return KrakenV2Auth(
            api_key=self.api_key,
            secret_key=self.secret_key,
            time_provider=self._time_synchronizer)

    @property
    def name(self) -> str:
        return "kraken_v2"

    # not used
    @property
    def rate_limits_rules(self):
        return build_rate_limits_by_tier(self._kraken_v2_api_tier)

    @property
    def domain(self):
        return self._domain

    @property
    def client_order_id_max_length(self):
        return CONSTANTS.MAX_ORDER_ID_LEN

    @property
    def client_order_id_prefix(self):
        return CONSTANTS.HBOT_ORDER_ID_PREFIX

    @property
    def trading_rules_request_path(self):
        return CONSTANTS.ASSET_PAIRS_PATH_URL

    @property
    def trading_pairs_request_path(self):
        return CONSTANTS.ASSET_PAIRS_PATH_URL

    @property
    def check_network_request_path(self):
        return CONSTANTS.TICKER_PATH_URL

    @property
    def trading_pairs(self):
        return self._trading_pairs

    @property
    def is_cancel_request_in_exchange_synchronous(self) -> bool:
        return True

    @property
    def is_trading_required(self) -> bool:
        return self._trading_required

    def supported_order_types(self):
        return [OrderType.LIMIT, OrderType.LIMIT_MAKER, OrderType.MARKET]
        # return [OrderType.LIMIT]

    def _build_async_throttler(self, api_tier: KrakenV2APITier) -> AsyncThrottler:
        limits_pct = self._client_config.rate_limits_share_pct
        if limits_pct < Decimal("100"):
            self.logger().warning(
                f"The KrakenV2 API does not allow enough bandwidth for a reduced rate-limit share percentage."
                f" Current percentage: {limits_pct}."
            )
        throttler = AsyncThrottler(build_rate_limits_by_tier(api_tier))
        return throttler

    def _is_request_exception_related_to_time_synchronizer(self, request_exception: Exception):
        return False

    def _is_order_not_found_during_status_update_error(self, status_update_exception: Exception) -> bool:
        return False

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return CONSTANTS.UNKNOWN_ORDER_MESSAGE in str(cancelation_exception)

    def _create_web_assistants_factory(self) -> WebAssistantsFactory:
        return web_utils.build_api_factory(
            throttler=self._throttler,
            auth=self._auth)

    def _create_order_book_data_source(self) -> OrderBookTrackerDataSource:
        return KrakenV2APIOrderBookDataSource(
            trading_pairs=self._trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory)

    def _create_user_stream_data_source(self) -> UserStreamTrackerDataSource:
        return KrakenV2APIUserStreamDataSource(
            connector=self,
            api_factory=self._web_assistants_factory,
        )

    def _get_fee(self,
                 base_currency: str,
                 quote_currency: str,
                 order_type: OrderType,
                 order_side: TradeType,
                 amount: Decimal,
                 price: Decimal = s_decimal_NaN,
                 is_maker: Optional[bool] = None) -> TradeFeeBase:
        is_maker = order_type is OrderType.LIMIT_MAKER
        trade_base_fee = build_trade_fee(
            exchange=self.name,
            is_maker=is_maker,
            order_side=order_side,
            order_type=order_type,
            amount=amount,
            price=price,
            base_currency=base_currency,
            quote_currency=quote_currency
        )
        return trade_base_fee

    async def _api_get(self, *args, **kwargs):
        kwargs["method"] = RESTMethod.GET
        return await self._api_request_with_retry(*args, **kwargs)

    async def _api_post(self, *args, **kwargs):
        kwargs["method"] = RESTMethod.POST
        return await self._api_request_with_retry(*args, **kwargs)

    async def _api_put(self, *args, **kwargs):
        kwargs["method"] = RESTMethod.PUT
        return await self._api_request_with_retry(*args, **kwargs)

    async def _api_delete(self, *args, **kwargs):
        kwargs["method"] = RESTMethod.DELETE
        return await self._api_request_with_retry(*args, **kwargs)

    @staticmethod
    def is_cloudflare_exception(exception: Exception):
        """
        Error status 5xx or 10xx are related to Cloudflare.
        https://support.kraken.com/hc/en-us/articles/360001491786-API-error-messages#6
        """
        return bool(re.search(r"HTTP status is (5|10)\d\d\.", str(exception)))

    async def get_open_orders_with_userref(self, userref: int):
        data = {'userref': userref}
        return await self._api_request_with_retry(RESTMethod.POST,
                                                  CONSTANTS.OPEN_ORDERS_PATH_URL,
                                                  is_auth_required=True,
                                                  data=data)

    async def get_open_orders(self) -> List[OpenOrder]:
        open_orders = await self._api_request_with_retry(RESTMethod.POST, CONSTANTS.OPEN_ORDERS_PATH_URL,
                                                         is_auth_required=True)

        self.logger().info(f"Open orders message: {open_orders}")
        ret_val = []
        for exchange_order_id in open_orders.get("open"):
            order = open_orders["open"][exchange_order_id]
            if order.get("status") != "open" and order.get("status") != "pending":
                continue
            details = order.get("descr")        
            if details["ordertype"] != "limit":
                self.logger().info(f"Unsupported order type found: {order['type']}")
                continue
            # trading_pair = convert_from_exchange_trading_pair(details["pair"])
            # if trading_pair is None:
            trading_pair = convert_from_exchange_trading_pair(
                    details["pair"], tuple((await self.get_asset_pairs()).keys())
                ) 

            ret_val.append(
                OpenOrder(
                    client_order_id=order["userref"],
                    trading_pair=trading_pair,
                    price=Decimal(str(details["price"])),
                    amount=Decimal(str(order["vol"])),
                    executed_amount=Decimal(str(order["vol_exec"])),
                    status=order["status"],
                    order_type=OrderType.LIMIT,
                    is_buy=True if details["type"].lower() == "buy" else False,
                    time=int(order["opentm"]),
                    exchange_order_id=exchange_order_id
                )
            )

        self.logger().info(f"Parsed open orders: {ret_val}")
        return ret_val


    def open_orders(self):
        safe_ensure_future(self.get_open_orders())

    # custom method for testing
    def say_hello(self):
        return "Custom message 22"

    # === Orders placing ===

    def buy(self,
            trading_pair: str,
            amount: Decimal,
            order_type=OrderType.LIMIT,
            price: Decimal = s_decimal_NaN,
            **kwargs) -> str:
        """
        Creates a promise to create a buy order using the parameters

        :param trading_pair: the token pair to operate with
        :param amount: the order amount
        :param order_type: the type of order to create (MARKET, LIMIT, LIMIT_MAKER)
        :param price: the order price

        :return: the id assigned by the connector to the order (the client id)
        """
        order_id = str(get_new_numeric_client_order_id(
            nonce_creator=self._client_order_id_nonce_provider,
            max_id_bit_count=CONSTANTS.MAX_ID_BIT_COUNT,
        ))
        safe_ensure_future(self._create_order(
            trade_type=TradeType.BUY,
            order_id=order_id,
            trading_pair=trading_pair,
            amount=amount,
            order_type=order_type,
            price=price))
        return order_id

    def sell(self,
             trading_pair: str,
             amount: Decimal,
             order_type: OrderType = OrderType.LIMIT,
             price: Decimal = s_decimal_NaN,
             **kwargs) -> str:
        """
        Creates a promise to create a sell order using the parameters.
        :param trading_pair: the token pair to operate with
        :param amount: the order amount
        :param order_type: the type of order to create (MARKET, LIMIT, LIMIT_MAKER)
        :param price: the order price
        :return: the id assigned by the connector to the order (the client id)
        """
        order_id = str(get_new_numeric_client_order_id(
            nonce_creator=self._client_order_id_nonce_provider,
            max_id_bit_count=CONSTANTS.MAX_ID_BIT_COUNT,
        ))
        safe_ensure_future(self._create_order(
            trade_type=TradeType.SELL,
            order_id=order_id,
            trading_pair=trading_pair,
            amount=amount,
            order_type=order_type,
            price=price))
        return order_id

    async def get_asset_pairs(self) -> Dict[str, Any]:
        if not self._asset_pairs:
            asset_pairs = await self._api_request_with_retry(method=RESTMethod.GET,
                                                             path_url=CONSTANTS.ASSET_PAIRS_PATH_URL)
            self._asset_pairs = {f"{details['base']}-{details['quote']}": details
                                 for _, details in asset_pairs.items() if
                                 web_utils.is_exchange_information_valid(details)}
        return self._asset_pairs

    async def _place_order(self,
                           order_id: str,
                           trading_pair: str,
                           amount: Decimal,
                           trade_type: TradeType,
                           order_type: OrderType,
                           price: Decimal,
                           **kwargs) -> Tuple[str, float]:
        trading_pair = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        data = {
            "pair": trading_pair,
            "type": "buy" if trade_type is TradeType.BUY else "sell",
            "ordertype": "market" if order_type is OrderType.MARKET else "limit",
            "volume": str(amount),
            "userref": order_id,
            "price": str(price)
        }

        if order_type is OrderType.MARKET:
            del data["price"]
        if order_type is OrderType.LIMIT_MAKER:
            data["oflags"] = "post"
        order_result = await self._api_request_with_retry(RESTMethod.POST,
                                                          CONSTANTS.ADD_ORDER_PATH_URL,
                                                          data=data,
                                                          is_auth_required=True)

        o_id = order_result["txid"][0]    
        return (o_id, self.current_timestamp)

    async def _amend_order_old(self,
                           order_id: str,
                           amount: Decimal,
                           price: Decimal,
                           trading_pair: str,
                           **kwargs) -> Tuple[str, float]:
        data = {
            "txid": order_id,
        }        
        if amount is None and price is None:
            return False
        if amount is not None:
#             amount = self.quantize_order_amount(trading_pair=trading_pair, amount=amount)

            data["order_qty"] = str(amount)
        if price is not None:
#             price = self.quantize_order_price(trading_pair, price)
        
            data["limit_price"] = str(price)

        self.logger().info(f"Sending POST request: {data}")
        order_result = await self._api_request_with_retry(RESTMethod.POST,
                                                          CONSTANTS.AMEND_ORDER_PATH_URL,
                                                          data=data,
                                                          is_auth_required=True)
        self.logger().info(f"Received reply: {order_result}")

        a_id = order_result.get("amend_id")
        ret_dic = {
            "amend_id": a_id,
            # "amount": amount,
            # "price": price,
            # "amendment_try_timestamp": self.current_timestamp
        }  
        return ret_dic

    def amend_order(self,
             client_order_id: str,       
             amount: Decimal | None,
             price: Decimal | None,
             **kwargs) -> str:
        """
        Creates a promise to amend a limit order. 

        Args:
            client_order_id (str): The inner identifier of the order to be amended.
            amount (Decimal | None): The new amount for the order. If None, the amount remains unchanged.
            price (Decimal | None): The new price for the order. If None, the price remains unchanged.
            **kwargs: Additional optional parameters for the amendment request.

        Returns:
            str: A promise or confirmation string for the amended order.
        """

        safe_ensure_future(self._amend_order(
            client_order_id=client_order_id,
            amount=amount,
            price=price,
            ))
        return client_order_id

    async def _amend_order(self,
                    client_order_id: str,
                    amount: Decimal | None,
                    price: Decimal | None,
                    **kwargs):
        """
        Amends an order in the exchange
        
        Args:
            client_order_id (str): The inner identifier of the order to be amended.
            amount (Decimal | None): The new amount for the order. If None, the amount remains unchanged.
            price (Decimal | None): The new price for the order. If None, the price remains unchanged.
            **kwargs: Additional optional parameters for the amendment request.

        Returns:
            None
        """
        if amount is None and price is None:
            self.logger().info(f"Invalid or None price ({price}) and amount ({amount}) while trying to amend order {client_order_id}")
            return

        # tracked_order = self._order_tracker.active_orders.get(client_order_id)
        tracked_order = self._order_tracker.fetch_tracked_order(client_order_id)

        if tracked_order is None:
            self.logger().info(f"Amend not successful: order {client_order_id} not found in active orders")
            return
        
        trading_pair = tracked_order.trading_pair
        trading_rule = self._trading_rules[trading_pair]

        quantized_price = None
        quantized_amount = None

        if price is not None:
            quantized_price = self.quantize_order_price(trading_pair, price)
        elif price.is_nan() or price == s_decimal_0:
            quantized_price = None
                      
        if amount is not None:
            quantized_amount = self.quantize_order_amount(trading_pair=trading_pair, amount=amount)

        amount_for_checking = quantized_amount if quantized_amount is not None else tracked_order.amount
        
        price_for_checking = quantized_price if quantized_price is not None else tracked_order.price

        notional_size = price_for_checking * amount_for_checking

        if amount_for_checking < trading_rule.min_order_size:
            self.logger().warning(f"{tracked_order.trade_type.name.title()} order ({client_order_id}) amount {amount_for_checking} is lower than the minimum order "
                                  f"size {trading_rule.min_order_size}. The order will not be amended, increase the "
                                  f"amount to be higher than the minimum order size.")
            await self._execute_order_cancel(tracked_order)
            return

        elif notional_size < trading_rule.min_notional_size:
            self.logger().warning(f"{tracked_order.trade_type.name.title()} order notional {notional_size} is lower than the "
                                  f"minimum notional size {trading_rule.min_notional_size}. The order will not be "
                                  f"created. Increase the amount or the price to be higher than the minimum notional.")
            await self._execute_order_cancel(tracked_order)
            return
        
        elif price_for_checking == tracked_order.price and amount_for_checking == tracked_order.amount:
            self.logger().warning(f"Order amend is not possible: both price ({price_for_checking}) and amount ({amount_for_checking}) are equal to the current order corresponding values. Doing nothing...")
            return 

        # order_price_before_amendment = tracked_order.price
        # order_size_before_amendment = tracked_order.amount        

        try:
            await self._execute_amend_order_and_process_update(order=tracked_order, price=quantized_price, amount=quantized_amount, **kwargs,)

            # self.logger().info(f"Order {client_order_id} amended successfully. ")
            
            # sometimes ws message arrives faster and in_flight_order is already
            # updated by the time the response is received from the above POST request
            # So no message is sent if the order is updated already.

            # amount_message = f"from {order_size_before_amendment} to {quantized_amount}"
            # size_message = f"from {order_price_before_amendment} to {quantized_price}"
            # self.logger().info(
            #     f"Amended order {client_order_id}. "
            #     f"Price: "
            #     f"{'Not changed' if quantized_price is None else size_message}. "
            #     f"Amount: "
            #     f"{'Not changed' if quantized_amount is None else amount_message}"

            # )


        except asyncio.CancelledError:
            raise
        except Exception as ex:
            await self._execute_order_cancel(tracked_order)
            self.logger().error(f"Error while amending order {client_order_id}. Tried to cancel it.", exc_info=True)       

    async def _execute_amend_order_and_process_update(
            self, 
            order: InFlightOrder,
            price: Decimal,
            amount: Decimal, 
            **kwargs) -> str:
        
        amend_id, update_timestamp = await self._execute_amend_order(
            order_id=order.exchange_order_id,
            amount=amount,
            price=price,
            **kwargs,
        )

        if amend_id is not None:
            order_update = self._create_order_update_with_price_and_amount(price, amount, order)
            await self.process_in_flight_order_update(order_update)

        return amend_id

    def _create_order_update_with_price_and_amount(self, 
                                                     price: Decimal | None,
                                                     amount: Decimal | None,
                                                     order: InFlightOrder) -> OrderUpdate:
        order_update: OrderUpdate = OrderUpdate(
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            trading_pair=order.trading_pair,
            update_timestamp=self.current_timestamp,
            new_state=OrderState.OPEN,
            misc_updates={
                "price": price,
                "amount": amount,
            }
        )        
        return order_update

    def sync_mode_process_in_flight_order_update(self, order_update):
        return safe_ensure_future(self.process_in_flight_order_update(order_update))

    async def process_in_flight_order_update(self, order_update: OrderUpdate) -> bool:
        # There was no possibility to modify the in_flight_order.py file,
        # so this method was created

        order = self._order_tracker.fetch_order(order_update.client_order_id, order_update.exchange_order_id)

        if order:
            new_amount = order_update.misc_updates.get("amount")
            new_price = order_update.misc_updates.get("price")

            updated = False

            if new_amount is not None and new_amount != order.amount:
                order.amount = new_amount
                updated = True

            if new_price is not None and new_price != order.price:
                order.price = new_price
                updated = True

            if order.current_state != order_update.new_state:           
                order.current_state = order_update.new_state
                updated = True

        if updated:
            order.last_update_timestamp = order_update.update_timestamp

        else:
            lost_order = self._order_tracker.fetch_lost_order(
                client_order_id=order_update.client_order_id, exchange_order_id=order_update.exchange_order_id
            )
            if lost_order:
                if order_update.new_state in [OrderState.CANCELED, OrderState.FILLED, OrderState.FAILED]:
                    # If the order officially reaches a final state after being lost it should be removed from the lost list
                    del self._lost_orders[lost_order.client_order_id]
            else:
                self.logger().debug(f"Order is not/no longer being tracked ({order_update})")
        return updated

    async def _execute_amend_order(self,
            order_id: str,
            amount: Decimal,
            price: Decimal,
            **kwargs,
        ):
        
        data = {
            "txid": order_id,
        }        

        if amount is not None:
            data["order_qty"] = str(amount)
        if price is not None:
            data["limit_price"] = str(price)

        self.logger().info(f"Sending POST request: {data}")
        order_result = await self._api_request_with_retry(RESTMethod.POST,
                                                          CONSTANTS.AMEND_ORDER_PATH_URL,
                                                          data=data,
                                                          is_auth_required=True)
        self.logger().info(f"Received reply: {order_result}")

        a_id = order_result.get("amend_id")
        return (a_id, self.current_timestamp)        

    async def _api_request_with_retry(self,
                                      method: RESTMethod,
                                      path_url: str,
                                      params: Optional[Dict[str, Any]] = None,
                                      data: Optional[Dict[str, Any]] = None,
                                      is_auth_required: bool = False,
                                      retry_interval=2.0) -> Dict[str, Any]:
        response_json = None
        result = None
        for retry_attempt in range(self.REQUEST_ATTEMPTS):
            try:
                response_json = await self._api_request(path_url=path_url, method=method, params=params, data=data,
                                                        is_auth_required=is_auth_required)

                if response_json.get("error") and "EAPI:Invalid nonce" in response_json.get("error", ""):
                    self.logger().error(f"Invalid nonce error from {path_url}. " +
                                        "Please ensure your KrakenV2 API key nonce window is at least 10, " +
                                        "and if needed reset your API key.")
                result = response_json.get("result")
                if not result or response_json.get("error"):
                    raise IOError({"error": response_json})
                break
            except IOError as e:
                if self.is_cloudflare_exception(e):
                    if path_url == CONSTANTS.ADD_ORDER_PATH_URL:
                        self.logger().info(f"Retrying {path_url}")
                        # Order placement could have been successful despite the IOError, so check for the open order.
                        response = await self.get_open_orders_with_userref(data.get('userref'))
                        if any(response.get("open").values()):
                            return response
                    self.logger().warning(
                        f"Cloudflare error. Attempt {retry_attempt + 1}/{self.REQUEST_ATTEMPTS}"
                        f" API command {method}: {path_url}"
                    )
                    await asyncio.sleep(retry_interval ** retry_attempt)
                    continue
                else:
                    raise e
        if not result:
            raise IOError(f"Error fetching data from {path_url}, msg is {response_json}.")
        return result

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder):
        exchange_order_id = await tracked_order.get_exchange_order_id()
        api_params = {
            "txid": exchange_order_id,
        }
        cancel_result = await self._api_request_with_retry(
            method=RESTMethod.POST,
            path_url=CONSTANTS.CANCEL_ORDER_PATH_URL,
            data=api_params,
            is_auth_required=True)
        if isinstance(cancel_result, dict) and (
                cancel_result.get("count") == 1 or
                cancel_result.get("error") is not None):
            return True
        return False

    async def _format_trading_rules(self, exchange_info_dict: Dict[str, Any]) -> List[TradingRule]:
        """
        Example:
        {
            "XBTUSDT": {
              "altname": "XBTUSDT",
              "wsname": "XBT/USDT",
              "aclass_base": "currency",
              "base": "XXBT",
              "aclass_quote": "currency",
              "quote": "USDT",
              "lot": "unit",
              "pair_decimals": 1,
              "lot_decimals": 8,
              "lot_multiplier": 1,
              "leverage_buy": [2, 3],
              "leverage_sell": [2, 3],
              "fees": [
                [0, 0.26],
                [50000, 0.24],
                [100000, 0.22],
                [250000, 0.2],
                [500000, 0.18],
                [1000000, 0.16],
                [2500000, 0.14],
                [5000000, 0.12],
                [10000000, 0.1]
              ],
              "fees_maker": [
                [0, 0.16],
                [50000, 0.14],
                [100000, 0.12],
                [250000, 0.1],
                [500000, 0.08],
                [1000000, 0.06],
                [2500000, 0.04],
                [5000000, 0.02],
                [10000000, 0]
              ],
              "fee_volume_currency": "ZUSD",
              "margin_call": 80,
              "margin_stop": 40,
              "ordermin": "0.0002"
            }
        }
        """
        retval: list = []
        trading_pair_rules = exchange_info_dict.values()
        for rule in filter(web_utils.is_exchange_information_valid, trading_pair_rules):
            try:
                trading_pair = await self.trading_pair_associated_to_exchange_symbol(symbol=rule.get("altname"))
                min_order_size = Decimal(rule.get('ordermin', 0))
                min_price_increment = Decimal(f"1e-{rule.get('pair_decimals')}")
                min_base_amount_increment = Decimal(f"1e-{rule.get('lot_decimals')}")
                retval.append(
                    TradingRule(
                        trading_pair,
                        min_order_size=min_order_size,
                        min_price_increment=min_price_increment,
                        min_base_amount_increment=min_base_amount_increment,
                    )
                )
            except Exception:
                self.logger().error(f"Error parsing the trading pair rule {rule}. Skipping.", exc_info=True)
        return retval

    async def _update_trading_fees(self):
        """
        Update fees information from the exchange
        """
        pass

    async def _user_stream_event_listener(self):
        """
        Listens to messages from _user_stream_tracker.user_stream queue.
        Traders, Orders, and Balance updates from the WS.
        """
        user_channels = [
            CONSTANTS.USER_TRADES_ENDPOINT_NAME,
            CONSTANTS.USER_ORDERS_ENDPOINT_NAME,
            CONSTANTS.USER_BALANCE_ENDPOINT_NAME,
        ]

        user_orders_trades_channels = [
            CONSTANTS.USER_TRADES_ENDPOINT_NAME,
            CONSTANTS.USER_ORDERS_ENDPOINT_NAME,
        ]

        async for event_message in self._iter_user_event_queue():
            # self.logger().info(f"complex_event_message: {complex_event_message}, type: {type(complex_event_message)}")
            # for balance updates there may be several objects in a single response

            # self.logger().info(f"Got event_message: {event_message}")

            try:
                channel: str = event_message.get("channel", None)          
                
                if "data" in event_message and channel in user_channels:
                    data = event_message['data']

                    # orders and trades updates from 'executions' channel
                    # two of user_orders_trades_channels are equal to 'executions'
                    if channel in user_orders_trades_channels:                    
                        trade_message = []
                        order_message = []                                           

                        for item in data:

                            # if exec_type is equal to "filled", it means this is a message
                            # notifying about order completion. It's created exactly for that purpose.
                            # self.logger().info(f"Received a data Message from channel {channel}: {item}")
                            if "exec_type" in item:
                                if item["exec_type"] in ["trade"]:
                                    trade_message.append(item)
                                    order_message.append(item)
                                elif item["exec_type"] in ["pending_new"]:
                                    continue
                                else:    
                                    order_message.append(item)                                   

                        # trade or order message can have multiple records                
                        if trade_message:
                            self._process_trade_message(trade_message)
                        if order_message:
                            self._process_order_message(order_message)
                    elif channel == CONSTANTS.USER_BALANCE_ENDPOINT_NAME:
                        self._process_balance_message_ws(event_message)    
                
                elif event_message is asyncio.CancelledError:
                    raise asyncio.CancelledError
                else:
                    raise Exception(event_message)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().error(
                    "Unexpected error in user stream listener loop.", exc_info=True)
                await self._sleep(5.0)

    def _process_balance_message_ws(self, event_message):
        
        # self.logger().info(f"Got new Balance message: {event_message}")
        event_message_type = event_message.get("type", None)        
        # for simplicity we process only "main" "spot" balances
        account = event_message.get("data", [])
        try:
            if event_message_type == "snapshot":        
                # for each object in 'data' (for each currency)
                for asset in account:
                    asset_name = asset["asset"]
                    # if it's a snapshot then there's a "wallets" key                
                    if "wallets" in asset:
                        # for each wallet inside one data object (one data object -> one currency)
                        for wallet in asset["wallets"]:
                            wallet_type = wallet.get("type", None)
                            wallet_id = wallet.get("id", None)
                            self._process_balance_single_asset_message(asset_name, wallet_type, wallet_id, wallet)                            
            # it's just an update
            elif event_message_type == "update":
                # for each object in 'data' (for each currency)
                for asset in account:
                    asset_name = asset["asset"]
                    wallet_type = asset.get("wallet_type", None)
                    wallet_id = asset.get("wallet_id", None)
                    
                    self._process_balance_single_asset_message(asset_name, wallet_type, wallet_id, asset)
            else:
                raise ValueError(f"Unknown message type: {event_message_type}")
        except KeyError as e:
            self.logger().error(f"Missing expected key: {e}", exc_info=True)
        except ValueError as e:
            self.logger().error(f"Value error: {e}", exc_info=True)
        except Exception as e:
            self.logger().error("Unexpected error in user stream listener loop.", exc_info=True)

    def _process_balance_single_asset_message(self, asset_name, wallet_type, wallet_id, wallet):
        if wallet_type == "spot" and wallet_id == "main":
            # Check if asset_name already exists in _account_balances
            if asset_name in self._account_balances:
                previous_balance = self._account_balances[asset_name]
            else:
                previous_balance = Decimal('0.0')

            if asset_name in self._account_available_balances:
                previous_available_balance = self._account_available_balances[asset_name]
            else:
                # It's assumed this will never happen
                self.update_available_balance_offline()
                previous_available_balance = self._account_available_balances.get(asset_name, Decimal('0.0'))

            new_balance = Decimal(str(wallet["balance"]))
            self._account_balances[asset_name] = new_balance

            # self.logger().info(f"{asset_name} total balance updated. New: {new_balance}, Old: {previous_balance}")

            # The available balance is approximate and is not always correct.
            # it will work well if no other bots are running on the same assets.
            # For precise and accurate result the information about all open orders
            # is required. Doing POST API requests every time the balance changes 
            # is not an option, it may easily hit the rate limits and the info may
            # be outdated if the balance changes rapidly because of an order fill event. 
            # For example, the ws balance message may be sent and the 
            # POST API request may return the unchanged order status
            self._account_available_balances[asset_name] = (
                previous_available_balance + (new_balance - previous_balance)
            )

            # self.logger().info(f"{asset_name} available balance updated: New: {new_balance}, Old: {previous_available_balance}")

    def _create_trade_update_with_order_fill_data(
            self,
            order_fill: Dict[str, Any],
            order: InFlightOrder):
        fee_asset = order.quote_asset

        fee = TradeFeeBase.new_spot_fee(
            fee_schema=self.trade_fee_schema(),
            trade_type=order.trade_type,
            percent_token=fee_asset,
            flat_fees=[TokenAmount(
                amount=Decimal(order_fill["fee"]),
                token=fee_asset
            )]
        )
        trade_update = TradeUpdate(
            trade_id=str(order_fill["trade_id"]),
            client_order_id=order.client_order_id,
            exchange_order_id=order_fill.get("ordertxid"),
            trading_pair=order.trading_pair,
            fee=fee,
            fill_base_amount=Decimal(order_fill["vol"]),
            fill_quote_amount=Decimal(order_fill["vol"]) * Decimal(order_fill["price"]),
            fill_price=Decimal(order_fill["price"]),
            fill_timestamp=order_fill["time"],
        )
        return trade_update
    
    # modified for ws v2
    def _create_ws_trade_update_with_order_fill_data(
            self,
            order_fill: Dict[str, Any],
            order: InFlightOrder):

        if "exec_id" in order_fill: 

            source_fee = order_fill['fees'][0]
            
            fee_asset = convert_from_ws_exchange_symbol(source_fee["asset"])

            fee = TradeFeeBase.new_spot_fee(
                fee_schema=self.trade_fee_schema(),
                trade_type=order.trade_type,
                percent_token=fee_asset,
                flat_fees=[TokenAmount(
                    amount=Decimal(str(source_fee["qty"])),
                    token=fee_asset
                )]
            )

            trade_update = TradeUpdate(
                trade_id=str(order_fill["exec_id"]),
                client_order_id=order.client_order_id,
                exchange_order_id=str(order_fill.get("order_id", "")),
                trading_pair=order.trading_pair,
                fee=fee,
                fill_base_amount=Decimal(str(order_fill["last_qty"])),
                fill_quote_amount=Decimal(str(order_fill["cost"])),
                fill_price=Decimal(str(order_fill["last_price"])),
                fill_timestamp=rfc3339_to_unix(order_fill["timestamp"]),
            )
        else:

            source_fee = {"asset": "USD", "qty": order_fill["fee_usd_equiv"]}
            
            fee_asset = convert_from_ws_exchange_symbol(source_fee["asset"])

            fee = TradeFeeBase.new_spot_fee(
                fee_schema=self.trade_fee_schema(),
                trade_type=order.trade_type,
                percent_token=fee_asset,
                flat_fees=[TokenAmount(
                    amount=Decimal(source_fee["qty"]),
                    token=fee_asset
                )]
            )

            trade_update = TradeUpdate(
                trade_id=str(order_fill.get("exec_id", "fully_filled")),
                client_order_id=order.client_order_id,
                exchange_order_id=str(order_fill.get("order_id", "")),
                trading_pair=order.trading_pair,
                fee=fee,
                fill_base_amount=Decimal(str(order_fill["cum_qty"])),
                fill_quote_amount=Decimal(str(order_fill["cum_cost"])),
                fill_price=Decimal(str(order_fill["avg_price"])),
                fill_timestamp=rfc3339_to_unix(order_fill["timestamp"]),
            )                

            self.logger().info(f"Created a Trade Update: fill_base_amount = {trade_update.fill_base_amount}, fill_quote_amount = {trade_update.fill_quote_amount}, fill_price = {trade_update.fill_price}")
        rate_counter = order_fill.get("ratecount", None)
        if rate_counter is not None:
            self.rate_count = int(rate_counter)
            self.rate_count_update_timestamp = self.current_timestamp

        return trade_update


    # modified for ws v2
    def _process_trade_message(self, trades: List):
        
        for update in trades:
            # self.logger().info(f"Received a Trade Message. Order or Trade: {update}")
            # trade_id: str = next(iter(update))
            trade: Dict[str, str] = update
            # trade["trade_id"] = update["exec_id"]
            exchange_order_id = trade.get("order_id")
            client_order_id = str(trade.get("order_userref", ""))
            tracked_order = self._order_tracker.all_fillable_orders.get(client_order_id)

            if not tracked_order:
                self.logger().debug(f"Ignoring trade message with id {exchange_order_id}: not in in_flight_orders.")
            else:
                # self.logger().info(f"Received a Trade Message. Order or Trade: {update}")
                trade_update = self._create_ws_trade_update_with_order_fill_data(
                    order_fill=trade,
                    order=tracked_order)
                self._order_tracker.process_trade_update(trade_update)

    def _create_order_update_with_order_status_data(self, order_status: Dict[str, Any], order: InFlightOrder):
        order_update = OrderUpdate(
            trading_pair=order.trading_pair,
            update_timestamp=self.current_timestamp,
            new_state=CONSTANTS.ORDER_STATE[order_status["status"]],
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
        )
        return order_update

    def _create_ws_order_update_with_order_status_data(self, order_status: Dict[str, Any], order: InFlightOrder):
        order_update = OrderUpdate(
            trading_pair=order.trading_pair,
            update_timestamp=self.current_timestamp,
            new_state=CONSTANTS.WS_ORDER_STATE[order_status["order_status"]],
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
        )
        return order_update

    def _process_order_message(self, orders: List):
        # update = orders[0]
        # self.logger().info(f"Received Order Message. Orders: {orders}")
        for order_msg in orders:
            # self.logger().info(f"Received Order Message. Order: {order_msg}")
            
            client_order_id = str(order_msg.get("order_userref", ""))
            exchange_order_id = order_msg.get("order_id")
            tracked_order = self._order_tracker.all_updatable_orders.get(client_order_id)

            if not tracked_order:
                self.logger().debug(
                    f"Ignoring order message with client id {client_order_id}: not in in_flight_orders.")
                continue
            
            self.logger().info(f"Received Order Message. Order: {order_msg}")
            
            if exchange_order_id is not None and tracked_order.exchange_order_id is None:
                tracked_order.exchange_order_id = str(exchange_order_id)            
            
            if order_msg["exec_type"] == "amended":
                if order_msg["amended"]:

                    # tracked_order.amount = Decimal(str(order_msg["order_qty"]))
                    # tracked_order.price = Decimal(str(order_msg["limit_price"]))
                    
                    amend_order_update = self._create_order_update_with_price_and_amount(price=Decimal(str(order_msg["limit_price"])), amount=Decimal(str(order_msg["order_qty"])), order=tracked_order)

                    self.sync_mode_process_in_flight_order_update(amend_order_update)
                    
                    original_datetime_str = str(order_msg['timestamp'])

                    order_side = 'BUY' if tracked_order.trade_type == TradeType.BUY else 'SELL'
                    
                    message = f"Amended {order_side} order {client_order_id}. Price: {str(order_msg['limit_price'])}, Amount: {str(order_msg['order_qty'])}, Amend Timestamp: {original_datetime_str}, Ratecount: {order_msg.get('ratecount')}."

                    self.logger().info(f"{message} ")
                    # self.logger().info(f"Updated tracked order price: {self._order_tracker._in_flight_orders[client_order_id].price}")

            if "order_status" in order_msg and order_msg["exec_type"] != "amended":
                order_update = self._create_ws_order_update_with_order_status_data(order_status=order_msg,
                                                                                order=tracked_order)
                self._order_tracker.process_order_update(order_update=order_update)
            rate_counter = order_msg.get("ratecount", None)
            if rate_counter is not None:
                self.rate_count = int(rate_counter)
                # self.current_timestamp example: 1728233623.0
                # it is in seconds 
                self.rate_count_update_timestamp = self.current_timestamp
                # self.logger().info(f"self.rate_count changed: {self.rate_count}")    

    async def _all_trade_updates_for_order(self, order: InFlightOrder) -> List[TradeUpdate]:
        trade_updates = []

        try:
            exchange_order_id = await order.get_exchange_order_id()
            all_fills_response = await self._api_request_with_retry(
                method=RESTMethod.POST,
                path_url=CONSTANTS.QUERY_TRADES_PATH_URL,
                data={"txid": exchange_order_id},
                is_auth_required=True)

            for trade_id, trade_fill in all_fills_response.items():
                trade: Dict[str, str] = all_fills_response[trade_id]
                trade["trade_id"] = trade_id
                trade_update = self._create_trade_update_with_order_fill_data(
                    order_fill=trade,
                    order=order)
                trade_updates.append(trade_update)

        except asyncio.TimeoutError:
            raise IOError(f"Skipped order update with order fills for {order.client_order_id} "
                          "- waiting for exchange order id.")
        except Exception as e:
            if "EOrder:Unknown order" in str(e) or "EOrder:Invalid order" in str(e):
                return trade_updates
        return trade_updates

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        exchange_order_id = await tracked_order.get_exchange_order_id()
        updated_order_data = await self._api_request_with_retry(
            method=RESTMethod.POST,
            path_url=CONSTANTS.QUERY_ORDERS_PATH_URL,
            data={"txid": exchange_order_id},
            is_auth_required=True)

        update = updated_order_data.get(exchange_order_id)
        new_state = CONSTANTS.ORDER_STATE[update["status"]]

        order_update = OrderUpdate(
            client_order_id=tracked_order.client_order_id,
            exchange_order_id=exchange_order_id,
            trading_pair=tracked_order.trading_pair,
            update_timestamp=self.current_timestamp,
            new_state=new_state,
        )

        return order_update

    async def _update_balances(self):
        local_asset_names = set(self._account_balances.keys())
        remote_asset_names = set()
        balances = await self._api_request_with_retry(RESTMethod.POST, CONSTANTS.BALANCE_PATH_URL,
                                                      is_auth_required=True)
        open_orders = await self._api_request_with_retry(RESTMethod.POST, CONSTANTS.OPEN_ORDERS_PATH_URL,
                                                         is_auth_required=True)

        self.logger().info(f"UPDATED BALANCES with POST API!") # {balances}")

        locked = defaultdict(Decimal)

        for order in open_orders.get("open").values():
            if order.get("status") == "open":
                details = order.get("descr")
                if details.get("ordertype") == "limit":
                    pair = convert_from_exchange_trading_pair(
                        details.get("pair"), tuple((await self.get_asset_pairs()).keys())
                    )
                    (base, quote) = self.split_trading_pair(pair)
                    vol_locked = Decimal(order.get("vol", 0)) - Decimal(order.get("vol_exec", 0))
                    if details.get("type") == "sell":
                        locked[convert_from_exchange_symbol(base)] += vol_locked
                    elif details.get("type") == "buy":
                        locked[convert_from_exchange_symbol(quote)] += vol_locked * Decimal(details.get("price"))

        for asset_name, balance in balances.items():
            cleaned_name = convert_from_exchange_symbol(asset_name).upper()
            total_balance = Decimal(balance)
            free_balance = total_balance - Decimal(locked[cleaned_name])
            self._account_available_balances[cleaned_name] = free_balance
            self._account_balances[cleaned_name] = total_balance
            remote_asset_names.add(cleaned_name)

        asset_names_to_remove = local_asset_names.difference(remote_asset_names)
        for asset_name in asset_names_to_remove:
            del self._account_available_balances[asset_name]
            del self._account_balances[asset_name]
    
    def update_balances_offline(self):
        
        if self._real_time_balance_update:
            return
        balances = self._account_balances
        open_orders = self.in_flight_orders

        self.logger().info(f"UPDATED BALANCES offline! {balances['USD']}, {balances['GNO']}")

        locked = defaultdict(Decimal)
        
        for order in open_orders.values():
            # self.logger().info(f"open_order: {order.current_state}, {order.amount}, {order.executed_amount_base}")
            if order.current_state in [OrderState.OPEN, OrderState.PARTIALLY_FILLED]:
                base_asset = order.base_asset
                quote_asset = order.quote_asset
                remaining_amount = order.amount - order.executed_amount_base

                if order.trade_type == TradeType.SELL:
                    locked[base_asset] += remaining_amount
                    # self.logger().info(f"locked: {locked[base_asset]}")
                elif order.trade_type == TradeType.BUY:
                    locked[quote_asset] += remaining_amount * order.price
                    # self.logger().info(f"locked: {locked[quote_asset]}")
                

        # self.logger().info(f"locked: {locked}")

        for asset_name, balance in balances.items():
            total_balance = Decimal(balance)
            free_balance = total_balance - Decimal(locked[asset_name])
            self._account_available_balances[asset_name] = free_balance
            self._account_balances[asset_name] = total_balance

        self.logger().info(f"UPDATED AVAILABLE BALANCES offline: {self._account_available_balances['USD']}, {self._account_available_balances['GNO']}")

    # this method only uses tracked orders to calculate the available balance
    def update_available_balance_offline(self):
        
        balances = self._account_balances
        open_orders = self.in_flight_orders

        # self.logger().info(f"UPDATED BALANCES offline! {balances['USD']}, {balances['GNO']}")

        locked = defaultdict(Decimal)
        
        for order in open_orders.values():
            # self.logger().info(f"open_order: {order.current_state}, {order.amount}, {order.executed_amount_base}")
            if order.current_state in [OrderState.OPEN, OrderState.PARTIALLY_FILLED]:
                base_asset = order.base_asset
                quote_asset = order.quote_asset
                remaining_amount = order.amount - order.executed_amount_base

                if order.trade_type == TradeType.SELL:
                    locked[base_asset] += remaining_amount
                    # self.logger().info(f"locked: {locked[base_asset]}")
                elif order.trade_type == TradeType.BUY:
                    locked[quote_asset] += remaining_amount * order.price
                    # self.logger().info(f"locked: {locked[quote_asset]}")
                

        # self.logger().info(f"locked: {locked}")

        for asset_name, balance in balances.items():
            total_balance = Decimal(balance)
            free_balance = total_balance - Decimal(locked[asset_name])
            self._account_available_balances[asset_name] = free_balance
            # self._account_balances[asset_name] = total_balance

        # self.logger().info(f"UPDATED AVAILABLE BALANCES offline: {self._account_available_balances['USD']}, {self._account_available_balances['GNO']}")


    def _initialize_trading_pair_symbols_from_exchange_info(self, exchange_info: Dict[str, Any]):
        mapping = bidict()
        for symbol_data in filter(web_utils.is_exchange_information_valid, exchange_info.values()):
            mapping[symbol_data["altname"]] = convert_from_exchange_trading_pair(symbol_data["wsname"])
        self._set_trading_pair_symbol_map(mapping)

    async def _get_last_traded_price(self, trading_pair: str) -> float:
        params = {
            "pair": await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        }
        resp_json = await self._api_request_with_retry(
            method=RESTMethod.GET,
            path_url=CONSTANTS.TICKER_PATH_URL,
            params=params
        )
        record = list(resp_json.values())[0]
        return float(record["c"][0])
