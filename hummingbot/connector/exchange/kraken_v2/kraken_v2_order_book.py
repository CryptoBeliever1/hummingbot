from typing import Dict, Optional

from hummingbot.connector.exchange.kraken_v2.kraken_v2_utils import rfc3339_to_unix
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_message import OrderBookMessage, OrderBookMessageType


class KrakenV2OrderBook(OrderBook):

    @classmethod
    def snapshot_message_from_exchange(cls,
                                       msg: Dict[str, any],
                                       timestamp: float,
                                       metadata: Optional[Dict] = None) -> OrderBookMessage:
        if metadata:
            msg.update(metadata)
        return OrderBookMessage(OrderBookMessageType.SNAPSHOT, {
            "trading_pair": msg["trading_pair"].replace("/", ""),
            "update_id": msg["latest_update"],
            "bids": msg["bids"],
            "asks": msg["asks"]
        }, timestamp=timestamp)

    @classmethod
    def diff_message_from_exchange(cls,
                                   msg: Dict[str, any],
                                   timestamp: Optional[float] = None,
                                   metadata: Optional[Dict] = None) -> OrderBookMessage:
        if metadata:
            msg.update(metadata)
        return OrderBookMessage(OrderBookMessageType.DIFF, {
            "trading_pair": msg["trading_pair"].replace("/", ""),
            "update_id": msg["update_id"],
            "bids": msg["bids"],
            "asks": msg["asks"]
        }, timestamp=timestamp)

    @classmethod
    def snapshot_ws_message_from_exchange(cls,
                                          msg: Dict[str, any],
                                          timestamp: Optional[float] = None,
                                          metadata: Optional[Dict] = None) -> OrderBookMessage:
        if metadata:
            msg.update(metadata)
        return OrderBookMessage(OrderBookMessageType.SNAPSHOT, {
            "trading_pair": msg["trading_pair"].replace("/", ""),
            "update_id": msg["update_id"],
            "bids": msg["bids"],
            "asks": msg["asks"]
        }, timestamp=timestamp)

    @classmethod
    def trade_message_from_exchange(cls, msg: Dict[str, any], metadata: Optional[Dict] = None):
        if metadata:
            msg.update(metadata)
        # ts = int(msg["trade"]["trade_id"])
        ts = rfc3339_to_unix(msg["trade"]["timestamp"])
        return OrderBookMessage(OrderBookMessageType.TRADE, {
            "trading_pair": msg["pair"].replace("/", ""),
            "trade_type": float(TradeType.SELL.value) if msg["trade"]["side"] == "sell" else float(TradeType.BUY.value),
            "trade_id": ts,
            "update_id": ts,
            "price": str(msg["trade"]["price"]),
            "amount": str(msg["trade"]["qty"])
        }, timestamp=ts)

    @classmethod
    def from_snapshot(cls, msg: OrderBookMessage) -> "OrderBook":
        retval = KrakenV2OrderBook()
        retval.apply_snapshot(msg.bids, msg.asks, msg.update_id)
        return retval
