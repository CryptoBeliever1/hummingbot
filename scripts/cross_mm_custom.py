import asyncio
import math
import os
import re
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional, TypedDict, Union

import pandas as pd
from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData
from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.client.ui import version
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderType,
    PositionAction,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from scripts.utility.crossbot import Timer
from scripts.utility.telegram_utils import TelegramUtils

# from hummingbot.core.data_type.order_book_tracker import OrderBookTracker
# from hummingbot.core.data_type.order_book import OrderBook

# from hummingbot.core.utils.trading_pair import TradingPair
# from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
# from hummingbot.connector.connector_base import ConnectorBase

class OrderAmendedPreviouslyDict(TypedDict, total=True):
    order_id: Optional[str]
    amend_id: Optional[str]
    price: Optional[Decimal]
    amount: Optional[Decimal]
    amendment_try_timestamp: Optional[float]

class CrossMMCustomConfig(BaseClientModel):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    # Name of the script file
    # Flow mode options: both, buy, sell
    flow_mode: str = Field("buy", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Flow mode: both, buy, or sell"
    ))
    # Maker exchange
    maker: str = Field("kraken_v2", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maker exchange"
    ))
    # Taker exchange
    taker: str = Field("mexc", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Taker exchange"
    ))
    # Trading pair for maker
    maker_pair: str = Field("GNO-USD", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair for maker"
    ))
    # Trading pair for taker
    taker_pair: str = Field("GNO-USDT", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair for taker"
    ))
    # Round up to this value in best price calculations from taker asks bids
    order_price_precision: int = Field(2, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Order price precision"
    ))
    # Round up to this value in best price calculations from taker asks bids
    order_base_precision: int = Field(2, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Order base precision"
    ))

    # Maker fee
    maker_fee: float = Field(0.0025, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maker fee percentage, (0.001 = 0.1%)"
    ))
    # Taker fee
    taker_fee: float = Field(0.0002, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Taker fee percentage, (0.001 = 0.1%)"
    ))
    # Minimum profit for sell
    min_profit_sell: float = Field(0.01, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum profit for sell in fractions (e.g., 0.001 = 0.1%)"
    ))
    # Minimum profit for buy
    min_profit_buy: float = Field(0.01, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum profit for buy in fractions (e.g., 0.001 = 0.1%)"
    ))
    # Dust volume in base asset units
    dust_vol_sell: float = Field(0.1, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Dust volume for sell in base asset units"
    ))
    # Dust volume in base asset units
    dust_vol_buy: float = Field(0.1, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Dust volume for buy in base asset units"
    ))

    # in quote currency. This setting can be used to put the order with THE SAME
    # PRICE as the current best order. Just increase the accuracy by 10 times
    # compared to the max accuracy of the base asset on maker exchange
    # For example, if min step is 1e-8, put 1e-9
    order_price_step_sell: float = Field(0.2, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order price step for sell"
    ))
    order_price_step_buy: float = Field(0.2, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order price step for buy"
    ))

    # This is the max allowed distance up from the dust price
    # If the order price stays within the distance, it's not edited
    # The purpose is to maximise profit
    # If it's too small the order will be edited frequently
    # and the rate limits can be exceeded
    # If it's too high the order may hang at the top of the order book 
    # for longer periods even if the competitors with high volumes
    # remove their orders from the top    
    order_price_safe_distance_sell: float = Field(0.3, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order price safe distance for sell"
    ))
    order_price_safe_distance_buy: float = Field(0.3, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order price safe distance for buy"
    ))
    
    # if we have to put the order close to the hedge price 
    # this is the distance from the hedge price we put it to
    hedge_price_step_sell: float = Field(0.2, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Hedge price step for sell"
    ))
    hedge_price_step_buy: float = Field(0.1, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Hedge price step for buy"
    ))
        
    # if the hedge price is out of the volume window, the order price will
    # be kept within this distance from the hedge price
    hedge_price_safe_distance_sell: float = Field(1, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Hedge price safe distance for sell"
    ))
    hedge_price_safe_distance_buy: float = Field(0.5, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Hedge price safe distance for buy"
    ))
    maker_order_book_depth: int = Field(30, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Maker order book depth"
    ))
    strategy: str = Field("default", client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Strategy to use: 'default' or 'keep_tick_level'"
    ))

    # at what level to keep the order at maker, starting from 1
    keep_tick_number_buy: int = Field(2, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Tick level to keep for buy orders"
    ))
    keep_tick_number_sell: int = Field(2, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Tick level to keep for sell orders"
    ))

    # in adaptive mode the below values are used as MAXIMAL order size (Base asset)
    amount_sell: float = Field(0.1, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maximal order size for sell in base asset"
    ))
    amount_buy: float = Field(0.1, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maximal order size for buy in base asset"
    ))

    # minimal order amount nominated in quote asset
    min_notional_maker: float = Field(5, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimal order amount for maker in quote asset"
    ))
    min_notional_taker: float = Field(5, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimal order amount for taker in quote asset"
    ))

    # adaptive order amount in fraction of the max available asset volume in the wallet    
    adaptive_amount_sell_fraction: float = Field(1, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Adaptive amount fraction for sell (of max available)"
    ))
    adaptive_amount_buy_fraction: float = Field(1, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Adaptive amount fraction for buy (of max available)"
    ))
    forget_time: int = Field(3600, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Forget time in seconds"
    ))
    base_precision_for_output: int = Field(4, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Base precision for output"
    ))
    quote_precision_for_output: int = Field(2, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Quote precision for output"
    ))
    edit_order_mode: int = Field(1, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Edit order mode: 1 for enabled, 0 for disabled"
    ))
    
    # key is the number of seconds for a measuring timeframe, value is the limit in % per min    
    price_speed_limits: dict = Field({'previous': 6.5, 4: 4.9, 8: 3.1, 16: 2.16, 32: 1.25}, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Price speed limits as a dictionary"
    ))

    # arbitrage is an option to place an order if there's a profitable limit order
    # on the opposite side of the order book
    # if you don't wish to activate it just put some big numbers to profits or min sizes
    arbitrage_profit_buy: float = Field(0.01, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Arbitrage profit for buy"
    ))
    arbitrage_profit_sell: float = Field(0.01, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Arbitrage profit for sell"
    ))
    arbitrage_min_size_buy: float = Field(50, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Minimum size for arbitrage buy"
    ))
    arbitrage_min_size_sell: float = Field(50, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Minimum size for arbitrage sell"
    ))
    taker_trading_mode: str = Field("spot", client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Taker trading mode: 'cross_margin', 'isolated_margin', or 'spot'"
    ))

    # What amount of asset to keep untouched on the exchange, expressed in QUOTE currency
    # It means we keep both base and quote assets not less than the amount set here
    # It's important to keep 1.01*max_order_size on taker if the MARKET BUY order is 
    # not possible on taker with the amount only
    # so taker_min_balance_quote = amount_sell*price*0.01
    taker_min_balance_quote: float = Field(0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum balance for taker in quote currency"
    ))
    maker_min_balance_quote: float = Field(0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum balance for maker in quote currency"
    ))
    total_base_change_notification_limit: float = Field(0.00001, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Total base change detection and notification lower limit, in amount base symbol. Recommended to set to the min base increment (min from maker and taker)"
    ))
    previous_speed_limit: float = Field(0.17, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Previous speed limit in percentage per millisecond"
    ))
    taker_best_ask_price_coef: float = Field(1.01, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Coefficient for taker best ask price"
    ))
    taker_best_bid_price_coef: float = Field(0.99, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Coefficient for taker best bid price"
    ))
    # for some exchanges ther's an API resource use counter
    rate_count_max_limit: int = Field(130, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Max allowed rate limit for the exchange API counter"
    ))
    # milliseconds
    after_order_is_filled_delay: int = Field(2500, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Delay after maker order is filled, in milliseconds"
    ))
    after_failed_order_delay: int = Field(30000, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Delay after maker order submission fail, in milliseconds"
    ))
    after_rate_conter_limit_reached_delay: int = Field(30000, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Delay after rate counter limit reached, in milliseconds"
    ))
    one_order_only: bool = Field(False, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Stop after one trade is executed"
    ))
    debug_output_control: dict = Field(
        {'check_active_orders': False, 
         'calculate_planned_orders_sizes': False, 
         'calculate_hedge_price': False, 
         'adjust_orders_sizes': False, 
         'calc_orders_parameters': False,
         }, 
         client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Debug output from different parts of the code"
    ))
    after_latest_balance_check_delay: int = Field(5, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Delay after the latest balance check (before the next balance check), in seconds"
    ))
    after_latest_order_fill_delay: int = Field(130, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Delay after the latest order fill (before the next balance check), in seconds"
    ))
    notify_about_rate_limit_exceeding: bool = Field(False, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Send telegram notifications if API rate limit custom counter is exceeded for maker"
    ))
    enable_small_base_balance_corrections: bool = Field(False, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Is small base balance (less than min taker increment) changes correction needed"
    ))
    enable_amend_order_for_kraken_v2: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Enable amend order feature instead of cancel and create for kraken_v2 connector"
    ))


s_decimal_nan = Decimal("NaN")
s_decimal_0 = Decimal("0")
# s_decimal_min = 

class MakerTotalBaseBalanceChecker:
    """
    A class to check if the total base balance can be checked based on certain time delays
    after the latest balance check and the latest order fill.

    Attributes:
        latest_balance_check_timestamp (float): The timestamp of the latest balance check.
        latest_order_fill_timestamp (float): The timestamp of the latest order fill.
        after_latest_balance_check_delay (int): The delay in seconds after the latest balance check before it can be checked again.
        after_latest_order_fill_delay (int): The delay in seconds after the latest order fill before it can be checked again.
    """

    def __init__(self, after_latest_balance_check_delay: int = 5, after_latest_order_fill_delay: int = 130):
        """
        Initializes the MakerTotalBaseBalanceChecker instance with the current timestamp for balance check
        and order fill, and the specified or default delay times.

        Args:
            after_latest_balance_check_delay (int): Delay in seconds after the latest balance check before it can be checked again (default 3).
            after_latest_order_fill_delay (int): Delay in seconds after the latest order fill before it can be checked again (default 3).
        """
        self.latest_balance_check_timestamp: float = time.time()
        
        # not to wait for the delay right after the script start if no orders filled
        self.latest_order_fill_timestamp: float = (time.time() - float(after_latest_order_fill_delay))
        self.after_latest_balance_check_delay: int = after_latest_balance_check_delay
        self.after_latest_order_fill_delay: int = after_latest_order_fill_delay
        self.total_balance_during_the_latest_detected_change: float = None
        self.total_balance_before_the_latest_detected_change: float = None
        self.latest_balance_change_detection_timestamp: float = None

    def can_the_total_base_balance_be_checked_now(self) -> bool:
        """
        Determines if the total base balance can be checked based on the elapsed time
        since the latest balance check and order fill, compared to their respective delay times.

        Returns:
            bool: True if both delay conditions are satisfied, otherwise False.
        """
        current_time = time.time()
        time_since_balance_check = current_time - self.latest_balance_check_timestamp
        time_since_order_fill = current_time - self.latest_order_fill_timestamp
        
        return (time_since_balance_check >= self.after_latest_balance_check_delay and
                time_since_order_fill >= self.after_latest_order_fill_delay)

    def check_if_the_balance_changed(self, starting_base_total: float, current_base_total: float, 
                                     meaningful_difference_in_base_symbol: float = 0.1) -> bool:
        """
        Checks if the total base balance has changed from its previous state or from the starting value by more than the specified meaningful percentage.

        Args:
            starting_base_total (float): The starting value of the base total balance.
            current_base_total (float): The current value of the base total balance.
            meaningful_difference_in_base_symbol (float): The meaningful percentage difference to consider (default 0.1%).

        Returns:
            bool: True if a meaningful change in balance is detected, otherwise False.
        """
        # return_value = False

        # We detect the total balance change not depending on how close the final value
        # is to the initial starting balance (when the script started). Any consecutive
        # changes (e.i. from 100 to 120 and then from 120 to 100) are detected unless 
        # they are not meaningful.
        
        if self.can_the_total_base_balance_be_checked_now():
            
            # Update the timestamp of the latest balance check
            self.latest_balance_check_timestamp = time.time()

            if starting_base_total is None:
                return False            
            
            if self.total_balance_during_the_latest_detected_change is None:
                previous_balance = starting_base_total
            else:
                previous_balance = self.total_balance_during_the_latest_detected_change    
            
            # Calculate the percentage difference
            real_value_difference = current_base_total - previous_balance

            # real_value_difference_from_starting_balance = current_base_total - starting_base_total
            # difference_percentage = abs(real_value_difference) / previous_balance * 100
                       
            if abs(real_value_difference)  >= meaningful_difference_in_base_symbol:
                
                self.total_balance_during_the_latest_detected_change = current_base_total               
                self.total_balance_before_the_latest_detected_change = previous_balance
                self.latest_balance_change_detection_timestamp = time.time()

                return True

        return False


class CrossMmCustom(ScriptStrategyBase):

### OPTIONS ###
    # both, buy, sell
    # flow_mode = "sell" #"sell" #"both"

    # maker = 'kraken_v2' # 'kraken_paper_trade'
    # taker = 'mexc'  #'gate_io_paper_trade'

    # maker = 'kraken_v2_paper_trade' # 'kraken_paper_trade'
    # taker = 'mexc_paper_trade'  #'gate_io_paper_trade'

    # maker_pair = "GNO-USD"
    # taker_pair = "GNO-USDT"

    #round up to this value in best price calculations from taker asks bids
    # order_price_precision = 2
    # order_base_precision = 2


    # maker_fee = 0.0025
    # taker_fee = 0.0002 #0.001 #0.00084

    ###### PROFIT in fractions, 0.001 = 0.1% #######
    # maker fee can be added here
    # min_profit_sell = 0.02 #0.0005
    # min_profit_buy = 0.02 #-0.01 #0.001 #0.00085 #was 0.0005 on 06.05.21

    ##### DUST VOLUME ##### in base asset units
    # dust_vol_sell = 0.1 
    # dust_vol_buy = 0.1
    #####             ##### 

    # in quote currency. This setting can be used to put the order with THE SAME
    # PRICE as the current best order. Just increase the accuracy by 10 times
    # compared to the max accuracy of the base asset on maker exchange
    # For example, if min step is 1e-8, put 1e-9
    # order_price_step_sell = 0.2 #9.8e-9 #1e-9
    # order_price_step_buy = 0.2

    # this is the max allowed distance up from the dust price
    # if the order price stays within the distance, it's not edited
    # the purpose is to maximise profit
    # if it's too small the order will be edited frequently
    # and the rate limits can be exceeded
    # if it's too high the order may hang at the top of the order book 
    # for longer periods even if the competitors with high volumes
    # remove their orders from the top
    # order_price_safe_distance_sell = 0.3 #0.8
    # order_price_safe_distance_buy = 0.3 #1

    # if we have to put the order close to the hedge price 
    # this is the distance from the hedge price we put it to
    # hedge_price_step_sell = 0.2
    # hedge_price_step_buy = 0.1

    # if the hedge price is out of the volume window, the order price will
    # be kept within this distance from the hedge price
    # hedge_price_safe_distance_sell = 1
    # hedge_price_safe_distance_buy = 0.5

    # maker_order_book_depth = 30
    # possible options are:
    # - 'default'
    # - 'keep_tick_level'

    # strategy = 'keep_tick_level'
    # strategy = 'default'

    # at what level to keep the order at maker, starting from 1
    # keep_tick_number_buy = 2
    # keep_tick_number_sell = 2


    # in adaptive mode the below values are used as MAXIMAL order size (Base asset)
    # amount_sell = 0.1 #0.00025 #0.0003 # 0.00105 #Base asset
    # amount_buy = 0.1 #0.00026 #0.00028  # 0.00105 #Base asset

    #minimal order amount nominated in quote asset
    # min_notional_maker = 5
    # min_notional_taker = 5

    # adaptive order amount in fraction of the max available asset volume in the wallet
    # adaptive_amount_sell_fraction = 1
    # adaptive_amount_buy_fraction = 1

    # forget_time = 3600

    # base_precision_for_output = 4
    # quote_precision_for_output = 2

    #if exchange supports edit order we can use this feature. Should be 0 or 1
    # edit_order_mode = 1

    # key is the number of seconds for a measuring timeframe, value is the limit in % per min
    # price_speed_limits = {'previous':6.5, 4:4.9, 8:3.1, 16:2.16, 32:1.25}
    #price_speed_limits = {'previous':1.0, 4:1.6, 8:1.8, 16:0.7, 32:0.1}

    #stop_trading_mode_active_timeframe = 30000
    # arbitrage is an option to place an order if there's a profitable limit order
    # on the opposite side of the order book
    # if you don't wish to activate it just put some big numbers to profits or min sizes
    # arbitrage_profit_buy = 0.01#0.002
    # arbitrage_profit_sell = 0.01#0.002
    # arbitrage_min_size_buy = 50
    # arbitrage_min_size_sell = 50

    # "cross_margin", "isolated_margin", or "spot". The default is "spot"
    # taker_trading_mode = "spot"

    # what amount of asset to keep untouched on the exchange, expressed in QUOTE currency
    # it means we keep both base and quote assets not less than the amount set here
    # It's important to keep 1.01*max_order_size on taker if the MARKET BUY order is 
    # not possible on taker with the amount only
    # so taker_min_balance_quote = amount_sell*price*0.01
    # taker_min_balance_quote = 0
    # maker_min_balance_quote = 0

    # total_base_change_notification_limit = 0.01

    # stop trading speed max limit for previous time value, %/millisec
    # previous_speed_limit = 0.17 #0.001717 # = 10.3 %/min

    # markets = {maker: {maker_pair}, taker: {taker_pair}}

    price_source = PriceType.MidPrice

    # sell_profit_coef = (1 + taker_fee)/(1 - maker_fee - min_profit_sell)
    # buy_profit_coef = (1 - taker_fee)/(1 + maker_fee + min_profit_buy)

    # for placing the taker limit order to make sure it will be completed
    # as a market order.
    # taker_best_ask_price_coef = 1.01
    # taker_best_bid_price_coef = 0.99
    
    # buy_order_placed = False
    # sell_order_placed = False
    # one_order_only = False
    exit_bot_flag = False
    start_time = None
    idle_mode = False
    one_time_message_flag = False
    one_time_init_was_launched_before = False

    # for some exchanges ther's an API resource use counter
    # rate_count_max_limit = 130
    previously_processed_rate_count_update_timestamp = None
    # start_time: timestamp with milliseconds
    # how_long: pause in milliseconds
    # idle_timer = Timer()
    idle_timers = []
    
    @classmethod
    def init_markets(cls, config: CrossMMCustomConfig):
        cls.markets = {config.maker: {config.maker_pair}, config.taker: {config.taker_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: CrossMMCustomConfig):
        super().__init__(connectors)
        # Assign all config attributes to self
        # The following line is added due to the history of development
        # At first the config file was not used and class variables 
        # were used instead. So to avoid additional variable names 
        # replacement in the code the config vars are just copied to self. vars
        self.__dict__.update(config.__dict__)
        self.sell_profit_coef = (1 + self.taker_fee)/(1 - self.maker_fee - self.min_profit_sell)
        self.buy_profit_coef = (1 - self.taker_fee)/(1 + self.maker_fee + self.min_profit_buy)
        self._async_executor = ThreadPoolExecutor(max_workers=1)       

    # Helper function to disable execution of any routines inside on_tick() 
    # if certain conditions are met
    def go_passive(self):
        if self.exit_bot_flag:
            return True
        if self.any_of_timers_is_active():
            return True
        return False

    def any_of_timers_is_active(self):
        '''Checks '.timer_is_active' flag for all timers.
        Returns True if any of them is true'''
        for timer in self.idle_timers:
            if timer.timer_is_active:
                # self.logger().info(f"The idle timer {timer.name} is active")
                return True
        return False    


    def disable_outdated_timers(self):
        '''Remove timers from idle_timers if their time has passed.'''
        if self.idle_timers:  # Only proceed if idle_timers is not empty
            self.idle_timers = [timer for timer in self.idle_timers if not timer.has_time_passed()]
            # if not self.idle_timers:
            #     self.logger().info(f"Some timer disabled")  


    def on_tick(self):
        try:
            self.exit_on_exit_bot_flag()
            # self.idle_timer.deactivate_if_time_passed()
            self.disable_outdated_timers()
            self.one_time_init()

            # self.logger().info(f"Current flow mode: {self.flow_mode}")
            # self.logger().info("TICK STARTED!!!!!!!!!!!!!!!!!!!!!!!!!!")             
            # self.logger().info(self.connectors[self.taker].trading_rules)
            self.check_active_orders(debug_output=False) 

            # self.show_tracked_orders()

            self.check_rate_count()
            # self.connectors[self.maker].open_orders()

            # self.exit_after_some_time()
        
            self.custom_init()

            # k = unknown

            self.calculate_planned_orders_sizes(debug_output=False)

            self.calculate_hedge_price(debug_output=False)       

            self.adjust_orders_sizes(debug_output=False) 

            self.calc_orders_parameters(debug_output=False)

            self.buy_order_flow()

            self.sell_order_flow()

            self.check_and_correct_total_base_balance(debug_output=True)
        
        except IOError as e:
            # Static part of the expected message
            expected_static_part = "Skipped order update with order fills for"            
            if expected_static_part in str(e) and "waiting for exchange order id." in str(e):
                self.standard_exception_message_sending(e, exception_source="on tick cycle")
                time.sleep(2)
            else:
                # Re-raise the exception if it's not the specific one you're looking for
                raise

        except Exception as e:
            self.standard_exception_message_sending(e, exception_source="on tick cycle")
            self.soft_exit()                        
        
        return

    def exit_after_some_time(self, time_period=10):
        if self.exit_bot_flag:
            return
        
        if self.start_time is None:
            self.start_time = time.time()
            self.logger().info("Script started. Timer initiated.")
        # Check if time_period seconds have passed
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= time_period:
            self.logger().info(f"{time_period} seconds have passed. Exiting the script.")
            self.soft_exit(cancel_active_orders=True)
            # raise RuntimeError("Stopping the script due to a non-critical error.")
            # sys.exit()

    # Exit routine if exit_bot_flag was set somewhere
    def exit_on_exit_bot_flag(self):
        
        if self.exit_bot_flag:
            self.logger().info(f"Exit bot flag was set. Exiting the script.")
            self.soft_exit(cancel_active_orders=True)

    def soft_exit(self, cancel_active_orders=True):
        # if cancel_active_orders:
            # time.sleep(3)
            # self.custom_cancel_all_orders() 
        
        if not self.exit_bot_flag:
            self.exit_bot_flag = True
        message_text = "Skipping all on_tick routines due to Exit Bot flag. Stopping..."     
        self.logger().info(message_text)
        self.telegram_utils.send_unformatted_message(message_text)   
        HummingbotApplication.main_application().stop()

    def custom_cancel_all_orders(self, debug_output=False):
        if self.active_buy_order is not None:
            self.custom_cancel_order(order=self.active_buy_order, debug_output=debug_output)
        if self.active_sell_order is not None:    
            self.custom_cancel_order(order=self.active_sell_order, debug_output=debug_output)

    def buy_order_flow(self):
        if self.go_passive():
            return        
        if self.flow_mode == "sell":
            return
        
        if self.skip_buy_order_flow_flag:
            self.skip_order_flow(side="buy", activate=False)
            return
        
        if self.active_buy_order is None:
            self.create_new_maker_order(side=TradeType.BUY, debug_output=False)
        elif self.edit_order_condition(side=TradeType.BUY, debug_output=False):
            
            if self.amend_condition(self.in_flight_buy_order):
                buy_order_edit_result = self.amend_order_any_side(
                    order_side="BUY",
                    order_size=self.order_size_buy,
                    planned_order_price=self.planned_order_price_buy,
                    order_exchange_id=self.buy_order_exchange_id,
                    active_order=self.active_buy_order,
                    amended_previously=self.buy_order_amended_previously,
                    debug_output=True
                )
            else:
                buy_order_edit_result = self.edit_order(order=self.active_buy_order, debug_output=False)
            
            if buy_order_edit_result == "cancel":
                self.custom_cancel_order(order=self.active_buy_order, debug_output=False)
            
            # if self.cancel_order_condition(side=TradeType.BUY, debug_output=False):
            #     self.custom_cancel_order(order=self.active_buy_order, debug_output=False) 

    def sell_order_flow(self):
        if self.go_passive():
            return        
        if self.flow_mode == "buy":
            return

        if self.skip_sell_order_flow_flag:
            self.skip_order_flow(side="sell", activate=False)
            return

        if self.active_sell_order is None:
            self.create_new_maker_order(side=TradeType.SELL, debug_output=False)
        elif self.edit_order_condition(side=TradeType.SELL, debug_output=False):
            
            if self.amend_condition(self.in_flight_sell_order):
                sell_order_edit_result = self.amend_order_any_side(
                    order_side="SELL",
                    order_size=self.order_size_sell,
                    planned_order_price=self.planned_order_price_sell,
                    order_exchange_id=self.sell_order_exchange_id,
                    active_order=self.active_sell_order,
                    amended_previously=self.sell_order_amended_previously,
                    debug_output=True
                )
            else:
                sell_order_edit_result = self.edit_order(order=self.active_sell_order, debug_output=False)

            if sell_order_edit_result == "cancel":
                self.custom_cancel_order(order=self.active_sell_order, debug_output=False)
            
            # if self.cancel_order_condition(side=TradeType.SELL, debug_output=False):
            #     self.custom_cancel_order(order=self.active_sell_order, debug_output=False)

    def get_instance_id(self):
        return self.connectors[self.maker]._client_config.hb_config.instance_id

    def one_time_init(self):
        if self.go_passive():
            return

        if self.one_time_init_was_launched_before:
            return
        try: 
            self.telegram_utils = TelegramUtils(self.maker, self.taker, self.maker_pair, self.taker_pair)
            self.hummingbot = HummingbotApplication.main_application()
            # instance_id = hummingbot.instance_id
            instance_id = self.get_instance_id()
            self.logger().info(f"Hummingbot Instance ID: {instance_id}")

            # self.logger().info(f"strategy_file_name: {self.hummingbot.strategy_file_name}")
            # self.logger().info(f"strategy_name: {self.hummingbot.strategy_name}")
            in_flight_orders = self.connectors[self.maker].in_flight_orders
            # self.logger().info(f"######### In Flight Orders: {in_flight_orders} ########")
            
            # If the exchange_id is None and the order is more than 20 seconds old something is wrong with it
            # It was created and failed or cancelled. So it's just marked as Failed
            if in_flight_orders:
                for order in in_flight_orders.values():
                    self.logger().notify(f"Strange In Flight Order Attributes: {order.attributes}")
                    current_timestamp = int(time.time())    
                    if order.exchange_order_id is None and order.current_state in (
                        OrderState.PENDING_CREATE,
                        OrderState.OPEN,
                        OrderState.CREATED    
                    ) and (current_timestamp - order.last_update_timestamp) > 20:
                        order.current_state = OrderState.FAILED
                        self.logger().notify(f"Fix in flight order {order.client_order_id} state to {order.current_state}")        

            self.maker_base_symbol, self.maker_quote_symbol = self.connectors[self.maker].split_trading_pair(self.maker_pair)
            self.taker_base_symbol, self.taker_quote_symbol = self.connectors[self.taker].split_trading_pair(self.taker_pair)

            self.maker_base_free = None
            self.maker_quote_free = None        
            self.taker_base_free = None        
            self.taker_quote_free = None        
            self.maker_base_total = None        
            self.maker_quote_total = None        
            self.taker_base_total = None        
            self.taker_quote_total = None
            self.base_total = None
            self.quote_total = None
            self.starting_base_total = None
            self.starting_quote_total = None
            self.balances_data_dict = None

            self.get_balances()

            self.starting_base_total = self.base_total
            self.starting_quote_total = self.quote_total

            self.base_balance_checker = MakerTotalBaseBalanceChecker(after_latest_balance_check_delay=self.after_latest_balance_check_delay, after_latest_order_fill_delay=self.after_latest_order_fill_delay)


            # loop = asyncio.get_event_loop()
            # asyncio.run_coroutine_threadsafe(
            #     self._run_async_update_balances(7000),
            #     loop
            # )
            # self.logger().info(f"Notifiers: {self.hummingbot.notifiers}")
            self.maker_rules = self.connectors[self.maker].trading_rules.get(self.maker_pair)
            self.taker_rules = self.connectors[self.taker].trading_rules.get(self.taker_pair)
            
            self.trading_rules_min_maker_order_amount = 0
            self.trading_rules_min_taker_order_amount = 0

            if self.maker_rules is not None:
                self.trading_rules_min_maker_order_amount = float(self.maker_rules.min_order_size)
            if self.taker_rules is not None:
                self.trading_rules_min_taker_order_amount = float(self.taker_rules.min_order_size)                


            self.sell_order_exchange_id = None
            self.buy_order_exchange_id = None
            
            telegram_string = self.telegram_utils.bot_started_string(version, self.strategy, self.maker_fee, self.taker_fee, self.flow_mode)
            self.hummingbot.notify(telegram_string)
            self.logger().info(telegram_string)

            # r = 2/0
            # outputting balances
            self.create_balances_data_dict()       

            # self.logger().info(self.balances_data_dict)

            # list for calculating profits for each individual maker fill
            # [{'maker_order': maker_order: LimitOrder, 'taker_order_id': taker_order_inner_id: str}]
            self.order_fills_profits_pairs = []

            # for calculating total sum of all profits from each order
            self.accumulated_profits = {
                'total_order_profit': Decimal("0"),
                'total_order_profit_excluding_maker_fee': Decimal("0"),
                'total_maker_fees': Decimal("0"),
                'total_taker_fees': Decimal("0"),
            }

            self.bot_start_time_timestamp = time.time()

            # if the maker fill order was smaller than the allowed taker min order
            # then this base amount will be added to the next taker order
            self.pending_small_base_amount_to_fix: Decimal = Decimal("0")

            self.notify_about_rate_limit_exceeding = getattr(self, 'notify_about_rate_limit_exceeding', False)
            self.enable_small_base_balance_corrections = getattr(self, 'enable_small_base_balance_corrections', False)
            self.enable_amend_order_for_kraken_v2 = getattr(self, 'enable_amend_order_for_kraken_v2', False)

            if self.maker != "kraken_v2":
                self.enable_amend_order_for_kraken_v2 = False

            self.buy_order_amended_previously: OrderAmendedPreviouslyDict = {
                "order_id": None,
                "amend_id": None,
                "price": None,
                "amount": None,
                "amendment_try_timestamp": None,
            }
            self.sell_order_amended_previously: OrderAmendedPreviouslyDict = {
                "order_id": None,
                "amend_id": None,
                "price": None,
                "amount": None,
                "amendment_try_timestamp": None,
            }
            self.skip_buy_order_flow_flag: bool = False
            self.skip_sell_order_flow_flag: bool = False

            self.in_flight_buy_order = None
            self.in_flight_sell_order = None

            self.in_flight_buy_order_limit = None
            self.in_flight_sell_order_limit = None

            telegram_string = self.telegram_utils.start_balance_data_text(self.balances_data_dict)
            self.hummingbot.notify(telegram_string)
        except Exception as e:
            exc_text = f"There has been an error during initialization: {e}, now stopping..."
            self.soft_exit()
            self.logger().error(exc_text, exc_info=True)            
            self.telegram_utils.send_unformatted_message(exc_text)
            
        
        self.one_time_init_was_launched_before = True

    def create_balances_data_dict(self):
        self.balances_data_dict = {
            'base_total': self.base_total,
            'base_maker_total': self.maker_base_total,
            'base_taker_total': self.taker_base_total,
            'quote_total': self.quote_total,
            'quote_maker_total': self.maker_quote_total,
            'quote_taker_total': self.taker_quote_total,
            'maker_name': self.maker,
            'taker_name': self.taker,
            'maker_base_symbol': self.maker_base_symbol,
            'maker_quote_symbol': self.maker_quote_symbol,
            'taker_base_symbol': self.taker_base_symbol,
            'taker_quote_symbol': self.taker_quote_symbol,
            'base_precision_for_output': self.base_precision_for_output,
            'quote_precision_for_output': self.quote_precision_for_output,
        }        

    def get_balances(self):

        self.maker_base_free = float(self.connectors[self.maker].get_available_balance(self.maker_base_symbol))
        
        self.maker_quote_free = float(self.connectors[self.maker].get_available_balance(self.maker_quote_symbol))

        self.taker_base_free = float(self.connectors[self.taker].get_available_balance(self.taker_base_symbol))
        
        self.taker_quote_free = float(self.connectors[self.taker].get_available_balance(self.taker_quote_symbol))
        # float(self.connectors[self.taker].get_balance(self.taker_quote_symbol)) 

        self.maker_base_total = float(self.connectors[self.maker].get_balance(self.maker_base_symbol))
        
        self.maker_quote_total = float(self.connectors[self.maker].get_balance(self.maker_quote_symbol))

        self.taker_base_total = float(self.connectors[self.taker].get_balance(self.taker_base_symbol))
        
        self.taker_quote_total = float(self.connectors[self.taker].get_balance(self.taker_quote_symbol))

        self.base_total = self.maker_base_total + self.taker_base_total
        self.quote_total = self.maker_quote_total + self.taker_quote_total        

    def custom_init(self):

        if self.go_passive():
            return
        # self.hummingbot.notify("This <b>is</b> a left panel Notification!!!")
        # self.logger().notify("This <b>is</b> a left panel and Telegram Notification!!!")
        # self.maker_base_symbol, self.maker_quote_symbol = self.maker_pair.split("-")
        # self.taker_base_symbol, self.taker_quote_symbol = self.taker_pair.split("-")
        
        # # Log the variables using logger().info
        # self.logger().info(f"Maker base symbol: {self.maker_base_symbol}, Maker quote symbol: {self.maker_quote_symbol}")
        # self.logger().info(f"Taker base symbol: {self.taker_base_symbol}, Taker quote symbol: {self.taker_quote_symbol}")           

        # self.connectors[self.maker].update_balances()
        self.get_balances()

        # We need to adjust the available balance for the amount(size) of an active order
        # Because if there's an active order the available balance should include it
        # Because one active order is allowed only
        if self.maker_quote_total is not None and self.maker_quote_free is not None:
            if self.maker_quote_free < self.maker_quote_total and self.active_buy_order:
                self.maker_quote_free = min((self.maker_quote_free + 
                                         float(self.active_buy_order.price) * float(self.active_buy_order.quantity)),
                                         self.maker_quote_total)
        
        if self.maker_base_total is not None and self.maker_base_free is not None:
            if self.maker_base_free < self.maker_base_total and self.active_sell_order:
                self.maker_base_free = min((self.maker_base_free + 
                                         float(self.active_sell_order.quantity)),
                                         self.maker_base_total)
        # market, trading_pair, base_asset, quote_asset = self.get_market_trading_pair_tuples()[0]
        # self.maker_base_free = float(market.get_available_balance(self.maker_base_symbol))
        # self.maker_quote_free = float(market.get_available_balance(self.maker_quote_symbol))
 
        # self.logger().info(f"self.maker_base_free: {self.maker_base_free}")
        # self.logger().info(f"self.maker_quote_free: {self.maker_quote_free}")

        self.do_not_create_buy_order_because_of_bad_parameters = False
        self.do_not_create_sell_order_because_of_bad_parameters = False
        
        self.create_buy_order_after_cancel_in_current_tick_cycle = False
        self.create_sell_order_after_cancel_in_current_tick_cycle = False

        self.buy_order_client_id_to_edit_in_current_tick_cycle = None
        self.sell_order_client_id_to_edit_in_current_tick_cycle = None
        # self.logger().info(f"{self.maker_base_symbol}: {self.maker_base_free}")
        # self.logger().info(f"{self.maker_quote_symbol}: {self.maker_quote_free}")

    def calculate_planned_orders_sizes(self, debug_output=False):
        if self.go_passive():
            return
        
        debug_output = self.debug_output_value_for_this_function(debug_output, function_name='calculate_planned_orders_sizes')

        koef_for_calculating_amounts = (1 - self.maker_fee - self.taker_fee - self.min_profit_buy)
        # price_for_calc_min_sell_amount = self.connectors[self.maker].get_price_by_type(self.maker_pair, PriceType.BestAsk) * (1 - self.maker_fee)
        price_for_calc_min_buy_amount = float(self.connectors[self.maker].get_price_by_type(self.maker_pair, PriceType.BestBid)) * koef_for_calculating_amounts
        price_for_calc_taker_max_amount = float(self.connectors[self.taker].get_price_by_type(self.taker_pair, PriceType.BestAsk)) * koef_for_calculating_amounts
        
        if debug_output:
            self.logger().info(
                f"""price_for_calc_min_buy_amount: {price_for_calc_min_buy_amount}""")

        # nominated in base currency, this is order amount actually
        # we calculate preliminary approximate volumes that are guaranteed bigger than the final volumes
        # this is done to calculate adequate hedge prices further with slightly bigger volumes    
        self.order_size_sell = min([
                                                (self.maker_base_free * self.adaptive_amount_sell_fraction), 
                                                self.amount_sell,
                                                self.taker_quote_free / price_for_calc_taker_max_amount    
                                                ])
                                            
        if debug_output:
            self.logger().info(f"order_size_sell: {self.order_size_sell}")

        self.order_size_buy = min([
                                                (self.maker_quote_free * self.adaptive_amount_buy_fraction) / price_for_calc_min_buy_amount, 
                                                self.amount_buy,
                                                self.taker_base_free 
                                                ])
                                            
        if debug_output:
            self.logger().info(f"order_size_buy: {self.order_size_buy}")

    def calculate_hedge_price(self, debug_output=False) -> Decimal:
        if self.go_passive():
            return
        
        debug_output = self.debug_output_value_for_this_function(debug_output, function_name='calculate_hedge_price')        
        # ref_price = self.connectors[self.taker].get_price_by_type(self.taker_pair, self.price_source)
        
        # self.taker_ref_order_amount = (self.taker_volume_depth_for_best_ask_bid / ref_price)
        
        taker_buy_result = self.connectors[self.taker].get_price_for_volume(self.taker_pair, True, self.order_size_sell)
        taker_sell_result = self.connectors[self.taker].get_price_for_volume(self.taker_pair, False, self.order_size_buy)

        self.taker_buy_by_volume_price = float(taker_buy_result.result_price)
        self.taker_sell_by_volume_price = float(taker_sell_result.result_price)

        self.float_var_is_valid_and_positive(self.taker_buy_by_volume_price, "self.taker_buy_by_volume_price")
        self.float_var_is_valid_and_positive(self.taker_sell_by_volume_price, "self.taker_sell_by_volume_price")

        self.hedge_price_sell = self.taker_buy_by_volume_price * self.sell_profit_coef
        # if debug_output:
        #     debug_order_book = self.get_order_book_dict(self.taker, self.taker_pair)
        #     self.df_order_book_sell
        #     self.logger().info(debug_order_book)

        if not self.float_var_is_valid_and_positive(self.hedge_price_sell, "self.hedge_price_sell"):
            self.do_not_create_sell_order_because_of_bad_parameters = True
        
        if debug_output:
            self.logger().info(f"self.hedge_price_sell = {self.taker_buy_by_volume_price} * {self.sell_profit_coef} = {self.hedge_price_sell}, taker_buy_result.result_price = {taker_buy_result.result_price}")
                
        self.hedge_price_buy = self.taker_sell_by_volume_price * self.buy_profit_coef

        if not self.float_var_is_valid_and_positive(self.hedge_price_buy, "self.hedge_price_buy"):
            self.do_not_create_sell_order_because_of_bad_parameters = True
            self.do_not_create_buy_order_because_of_bad_parameters = True
        
        if debug_output:
            self.logger().info(f"self.hedge_price_buy = {self.taker_sell_by_volume_price} * {self.buy_profit_coef}, taker_sell_result.result_price = {taker_sell_result.result_price}")
        
        # min order amount in base units
        self.min_notional_maker_amount = self.min_notional_maker / self.hedge_price_buy
        self.min_notional_taker_amount = self.min_notional_taker / self.hedge_price_buy
        self.min_notional_for_maker_order_creation = max(
            self.min_notional_maker_amount, 
            self.min_notional_taker_amount,
            self.trading_rules_min_maker_order_amount,
            self.trading_rules_min_taker_order_amount
            )

        if debug_output:
            output_message = f"""
            min_notional_maker_amount: {self.min_notional_maker_amount}
            min_notional_taker_amount: {self.min_notional_taker_amount}
            trading_rules_min_maker_order_amount: {self.trading_rules_min_maker_order_amount}
            trading_rules_min_taker_order_amount: {self.trading_rules_min_taker_order_amount}
            """
            self.logger().info(output_message)



        if debug_output:
            output_message = f"""
    amount for taker buy calculations: {self.order_size_sell} 
    amount for taker sell calculations: {self.order_size_buy}
    taker_buy_result: {taker_buy_result.result_price}
    taker_sell_result: {taker_sell_result.result_price}
    hedge_price_sell: {self.hedge_price_sell}
    hedge_price_buy: {self.hedge_price_buy}
    """
            self.logger().info(output_message)

    def min_taker_order_amount(self) -> float:
        """
        Calculate the minimum taker order amount based on trading rules fetched from and notional amount from config.

        Returns:
            float: The minimum taker order amount.
        """
        if self.trading_rules_min_taker_order_amount == 0:
            return self.min_notional_taker_amount
        
        return max(
            self.min_notional_taker_amount,
            self.trading_rules_min_taker_order_amount
        )


    # Here we calculate the exact order sizes taking into consideration all maker and taker balances
    # and the required untouched balances on maker and taker        
    def adjust_orders_sizes(self, debug_output=False):
        if self.go_passive():
            return

        debug_output = self.debug_output_value_for_this_function(debug_output, function_name='adjust_orders_sizes')

        maker_untouched_amount = self.maker_min_balance_quote / self.hedge_price_buy
        taker_untouched_amount = self.taker_min_balance_quote / self.hedge_price_buy

        maker_available_minus_untouched = self.maker_base_free * self.adaptive_amount_sell_fraction - maker_untouched_amount
        
        taker_available_minus_untouched = (self.taker_quote_free - self.taker_min_balance_quote) / self.taker_buy_by_volume_price

        taker_available_with_higher_buy_price = self.taker_quote_free / (self.taker_buy_by_volume_price * self.taker_best_ask_price_coef)

        sell_values = [
            maker_available_minus_untouched, 
            self.amount_sell,
            taker_available_minus_untouched,
            taker_available_with_higher_buy_price,  
        ]
        
        self.order_size_sell = min(sell_values)

        if debug_output:
            self.logger().info(f"sell_values: {sell_values}")
            self.logger().info(f"taker_quote_free: {self.taker_quote_free}, taker_min_balance_quote: {self.taker_min_balance_quote}, hedge_price_sell: {self.hedge_price_sell}")
            self.logger().info(f"taker_base_free: {self.taker_base_free}")
            self.logger().info(f"order_size_sell: {self.order_size_sell}")

        buy_values = [
            (self.maker_quote_free * self.adaptive_amount_buy_fraction - self.maker_min_balance_quote) / self.hedge_price_buy, 
            self.amount_buy,
            (self.taker_base_free - taker_untouched_amount)
        ]
        
        self.order_size_buy = min(buy_values)
        
        if debug_output:
            self.logger().info(f"buy_values: {buy_values}")
            self.logger().info(f"order_size_buy: {self.order_size_buy}")


    # Finds if there are any open orders and assigns
    # self.active_buy_order and self.active_sell_order
    def check_active_orders(self, debug_output=False):
        if self.go_passive():
            return
        
        # True or false to output debug messages
        debug_output = self.debug_output_value_for_this_function(debug_output, function_name='check_active_orders')

        self.active_limit_orders = self.get_active_orders(connector_name=self.maker)

        # self.logger().info(f"######### Active Orders: {self.active_limit_orders} ########")
        
        # Cleaning up wrong orders from database

        # trying to get active orders through exchange entity. 
        # It works but the order tracking and balance functionality fails with this approach
        self.active_in_flight_orders = self.connectors[self.maker].in_flight_orders
        # self.logger().info(f"######### In Flight Orders: {self.active_in_flight_orders} ########")
        # self.active_in_flight_orders_converted_to_limit = [order.to_limit_order() for order in self.active_in_flight_orders.values()]        
        # self.active_limit_orders = [order.to_limit_order() for order in self.active_in_flight_orders.values()]

        if debug_output:
            if self.active_limit_orders:
                self.logger().info("There are active orders.")
                self.logger().info(f"Orders: {self.active_limit_orders}")
            else:
                self.logger().info("There are no active orders.")

        # Check for active buy orders
        active_buy_orders = [order for order in self.active_limit_orders if order.is_buy]
        active_sell_orders = [order for order in self.active_limit_orders if not order.is_buy]

        if len(active_buy_orders) > 1 or len(active_sell_orders) > 1:
            # Cancel all orders because one order per side is allowed only
            # Also making a small delay for the orders to cancel
            self.logger().notify("Found more than one order on buy or sell side. Cancelling all orders.")
            self.cancel_all_active_limit_orders(debug_output=True)
            self.active_buy_order = None
            self.active_sell_order = None
            self.idle_timers.append(Timer(name="after_more_than_one_active_order_found_timer", duration=self.after_order_is_filled_delay))
            return           

        if active_buy_orders:
            self.active_buy_order = active_buy_orders[0]
            # self.logger().info(f"######### Active BUY Limit Order: {self.active_buy_order.client_order_id}, order price: {self.active_buy_order.price} ########")
            self.in_flight_buy_order = self.active_in_flight_orders.get(self.active_buy_order.client_order_id)
                
            if self.amend_condition(self.in_flight_buy_order):
                self.buy_order_exchange_id = self.in_flight_buy_order.exchange_order_id

                # self.logger().info(f"######### Active BUY Limit Order: {self.active_buy_order.client_order_id}, order price: {self.active_buy_order.price} ########")

                # We check if the order has been amended successfully by comparing 
                # the price and amount previously sent to the exchange with 
                # the current in_flight_order values
                # If the values are different we just cancel order and skip to the next tick
                # to create a new order
                
                # if self.buy_order_amended_previously["order_id"] == self.active_buy_order.client_order_id and (
                #         (self.buy_order_amended_previously["price"] is not None 
                #             and 
                #         self.buy_order_amended_previously["price"] != self.in_flight_buy_order.price
                #         ) 
                #             or 
                #         (self.buy_order_amended_previously["amount"] is not None 
                #             and 
                #         self.buy_order_amended_previously["amount"] != self.in_flight_buy_order.amount
                #         )
                #     ):
                #     vars_message = (
                #         f'self.buy_order_amended_previously["order_id"] = {self.buy_order_amended_previously["order_id"]}\n'
                #         f'self.active_buy_order.client_order_id = {self.active_buy_order.client_order_id}\n'
                #         f'self.buy_order_amended_previously["price"] = {self.buy_order_amended_previously["price"]}\n'
                #         f'self.in_flight_buy_order.price = {self.in_flight_buy_order.price}\n'
                #         f'self.buy_order_amended_previously["amount"] = {self.buy_order_amended_previously["amount"]}\n'
                #         f'self.in_flight_buy_order.amount = {self.in_flight_buy_order.amount}\n'
                #         f'self.buy_order_amended_previously["amendment_try_timestamp"] = {self.buy_order_amended_previously["amendment_try_timestamp"]}'
                #     )    

                #     self.logger().notify(f"BUY order was not amended successfully, skipping to the next tick and canceling it. Report:\n{vars_message}")
                #     self.custom_cancel_order(self.active_buy_order)
                #     self.skip_order_flow(side="buy", activate=True)

                # self.logger().info(f"######### In Flight BUY Order: {self.buy_order_exchange_id}, order price: {self.in_flight_buy_order.price} ########")

            # a protection from not finding the corresponding in_flight_order.
            # if it's not found everything just goes with create and cancel, without 'amend' feature
            
            if self.in_flight_buy_order is not None:
                self.in_flight_buy_order_limit = self.in_flight_buy_order.to_limit_order()
                # self.logger().info(f"self.active_buy_order: {self.active_buy_order}\nself.in_flight_buy_order_limit: {self.in_flight_buy_order_limit}")

                self.active_buy_order = self.in_flight_buy_order_limit

                # self.logger().info(f"self.active_buy_order after copying from flight order: {self.active_buy_order}")   

            if debug_output:
                self.logger().info("There are active buy orders.")
                self.logger().info(f"{self.active_buy_order}")
        else:
            self.active_buy_order = None
            if debug_output:
                self.logger().info("There are no active buy orders.")

################# START previous no-amend SELL block ###############
        # if active_sell_orders:
        #     self.active_sell_order = active_sell_orders[0]
        #     self.in_flight_sell_order = self.active_in_flight_orders.get(self.active_sell_order.client_order_id)
        #     if self.in_flight_sell_order is not None:
        #         self.sell_order_exchange_id = self.in_flight_sell_order.exchange_order_id            
        #     if debug_output:
        #         self.logger().info("There are active sell orders.")
        # else:
        #     self.active_sell_order = None
        #     if debug_output:
        #         self.logger().info("There are no active sell orders.")
################# END of previous no-amend SELL block ###############

        if active_sell_orders:
            self.active_sell_order = active_sell_orders[0]
            # self.logger().info(f"######### Active SELL Limit Order: {self.active_sell_order.client_order_id}, order price: {self.active_sell_order.price} ########")
            self.in_flight_sell_order = self.active_in_flight_orders.get(self.active_sell_order.client_order_id)
                
            if self.amend_condition(self.in_flight_sell_order):
                self.sell_order_exchange_id = self.in_flight_sell_order.exchange_order_id

                # self.logger().info(f"######### Active SELL Limit Order: {self.active_sell_order.client_order_id}, order price: {self.active_sell_order.price} ########")

                # We check if the order has been amended successfully by comparing 
                # the price and amount previously sent to the exchange with 
                # the current in_flight_order values
                # If the values are different we just cancel order and skip to the next tick
                # to create a new order

                # if self.sell_order_amended_previously["order_id"] == self.active_sell_order.client_order_id and (
                #         (self.sell_order_amended_previously["price"] is not None 
                #             and 
                #         self.sell_order_amended_previously["price"] != self.in_flight_sell_order.price
                #         ) 
                #             or 
                #         (self.sell_order_amended_previously["amount"] is not None 
                #             and 
                #         self.sell_order_amended_previously["amount"] != self.in_flight_sell_order.amount
                #         )
                #     ):
                #     vars_message = (
                #         f'self.sell_order_amended_previously["order_id"] = {self.sell_order_amended_previously["order_id"]}\n'
                #         f'self.active_sell_order.client_order_id = {self.active_sell_order.client_order_id}\n'
                #         f'self.sell_order_amended_previously["price"] = {self.sell_order_amended_previously["price"]}\n'
                #         f'self.in_flight_sell_order.price = {self.in_flight_sell_order.price}\n'
                #         f'self.sell_order_amended_previously["amount"] = {self.sell_order_amended_previously["amount"]}\n'
                #         f'self.in_flight_sell_order.amount = {self.in_flight_sell_order.amount}\n'
                #         f'self.sell_order_amended_previously["amendment_try_timestamp"] = {self.sell_order_amended_previously["amendment_try_timestamp"]}'
                #     )    

                #     self.logger().notify(f"SELL order was not amended successfully, skipping to the next tick and canceling it. Report:\n{vars_message}")
                #     self.custom_cancel_order(self.active_sell_order)
                #     self.skip_order_flow(side="sell", activate=True)

                # self.logger().info(f"######### In Flight SELL Order: {self.sell_order_exchange_id}, order price: {self.in_flight_sell_order.price} ########")

            # a protection from not finding the corresponding in_flight_order.
            # if it's not found everything just goes with create and cancel, without 'amend' feature
            
            if self.in_flight_sell_order is not None:
                self.in_flight_sell_order_limit = self.in_flight_sell_order.to_limit_order()
                # self.logger().info(f"self.active_sell_order: {self.active_sell_order}\nself.in_flight_sell_order_limit: {self.in_flight_sell_order_limit}")

                self.active_sell_order = self.in_flight_sell_order_limit

                # self.logger().info(f"self.active_sell_order after copying from flight order: {self.active_sell_order}")   

            if debug_output:
                self.logger().info("There are active sell orders.")
                self.logger().info(f"{self.active_sell_order}")
        else:
            self.active_sell_order = None
            if debug_output:
                self.logger().info("There are no active sell orders.")
        


    def amend_condition(self, in_flight_order) -> bool:
        result = in_flight_order is not None and self.enable_amend_order_for_kraken_v2
        return result

    def skip_order_flow(self, side="buy", activate=False):
        if activate:
            if side == "buy":
                self.skip_buy_order_flow_flag = True
            else:
                self.skip_sell_order_flow_flag = True
        else:
            if side == "buy":
                self.skip_buy_order_flow_flag = False
            else:
                self.skip_sell_order_flow_flag = False

    def round_to_precision(self, value, precision, rounding_mode='ceil', return_string=True):
        # Calculate the scaling factor based on the precision
        scaling_factor = 10 ** precision
        # Multiply by the scaling factor to shift the decimal places
        scaled_value = value * scaling_factor
        
        if rounding_mode == 'ceil':
            # Apply math.ceil to get the ceiling integer value
            rounded_value = math.ceil(scaled_value)
        elif rounding_mode == 'floor':
            # Apply math.floor to get the floor integer value
            rounded_value = math.floor(scaled_value)
        else:
            raise ValueError("Invalid rounding_mode. Please use 'ceil' or 'floor'.")
        
        # Divide by the scaling factor to shift back to the original number of decimal places
        result = rounded_value / scaling_factor
        if return_string:
            result = f"{result:.{precision}f}"
        return result

    def create_new_maker_order(self, side=TradeType.BUY, debug_output=False):
        if side == TradeType.BUY:
            if self.do_not_create_buy_order_because_of_bad_parameters:
                self.logger().error(f"The {side} order is not created because self.do_not_create_buy_order_because_of_bad_parameters is True")
                return
            if not self.order_size_and_price_are_valid(side):
                return
            buy_price = Decimal(
                self.round_to_precision(self.planned_order_price_buy, self.order_price_precision, rounding_mode='floor'))

            buy_order = OrderCandidate(trading_pair=self.maker_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=Decimal(str(self.order_size_buy)), price=buy_price)
            # buy_order_adjusted = self.adjust_proposal_to_budget(self.maker, [buy_order])
            if self.check_order_min_size_before_placing(self.maker, buy_order, notif_output=False):
                self.place_order(self.maker, buy_order)
        else:
            if self.do_not_create_sell_order_because_of_bad_parameters:
                self.logger().error(f"The {side} order is not created because self.do_not_create_buy_order_because_of_bad_parameters is True")
                return                        
            if not self.order_size_and_price_are_valid(side):
                return
            
            # self.logger().info(f"sell price before placing order: {self.planned_order_price_sell}")
            sell_price = Decimal(
                self.round_to_precision(self.planned_order_price_sell, self.order_price_precision, rounding_mode='ceil'))
            # self.logger().info(f"sell price after converting to Decimal: {sell_price}")
            sell_order = OrderCandidate(trading_pair=self.maker_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=Decimal(str(self.order_size_sell)), price=sell_price)
            # sell_order_adjusted = self.adjust_proposal_to_budget(self.maker, [sell_order])
            if self.check_order_min_size_before_placing(self.maker, sell_order, notif_output=False):
                self.place_order(self.maker, sell_order)

    def float_var_is_valid_and_positive(self, variable, variable_name="Undefined Name"):
            if not isinstance(variable, float):
                self.logger().error(f"The {variable_name} variable value is not float!")
                return False
            if math.isnan(variable):
                self.logger().error(f"The {variable_name} variable value is NaN!")
                return False
            if variable <= 0:
                # self.logger().error(f"The {variable_name} variable value is less or equal to 0!")
                return False
            return True
    
    def order_size_and_price_are_valid(self, side=TradeType.BUY, debug_output=False):
        result = True
        if side == TradeType.BUY:
            result = (self.float_var_is_valid_and_positive(self.order_size_buy, "self.order_size_buy") and 
                    self.float_var_is_valid_and_positive(self.planned_order_price_buy, "self.planned_order_price_buy"))

        elif side == TradeType.SELL:
            result = (self.float_var_is_valid_and_positive(self.order_size_sell, "self.order_size_sell") and 
                    self.float_var_is_valid_and_positive(self.planned_order_price_sell, "self.planned_order_price_sell"))

        else:
            self.logger().error(f"Unknown trade side: {side}")
            return False
        return result



    def place_order(self, connector_name: str, order: OrderCandidate):
        
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

        # self.connectors[self.maker].update_balances_offline()        

    def check_order_min_size_before_placing(self, connector_name: str, order: OrderCandidate, notif_output=False):
        if connector_name == self.maker:
            if self.check_min_order_amount_for_maker_order_creation(order.amount):
                return True
            if notif_output:
                self.logger().info(f"The planned maker {order.order_side} order amount is too low: {order.amount}, can't place a new order")
            return False                   
        else:
            if order.amount >= self.min_notional_taker_amount:
                return True
            if notif_output:
                self.logger().info(f"The planned taker {order.order_side} order amount is too low: {order.amount}, can't place a new order")
            return False 

    def calc_orders_parameters(self, debug_output=False):
        if self.go_passive():
            return

        debug_output = self.debug_output_value_for_this_function(debug_output, function_name='calc_orders_parameters')    

        self.get_order_book_dict(self.maker, self.maker_pair, self.maker_order_book_depth)
        
        # if debug_output:
        #     # self.logger().info(f"df_order_book_sell: {self.df_order_book_sell}")
        #     # self.logger().info(f"df_order_book_buy: {self.df_order_book_buy}")        
        
        # SELL side strategy
        if self.df_order_book_sell.at[self.df_order_book_sell.last_valid_index(), "cumsum"] <= self.dust_vol_sell:
            self.dust_vol_limit_price_sell = self.df_order_book_sell.iat[self.df_order_book_sell.last_valid_index(), 0]
        else:
            self.dust_vol_limit_price_sell = round(self.df_order_book_sell.loc[self.df_order_book_sell['cumsum'] > self.dust_vol_sell].iat[0, 0], self.order_price_precision)
            
        if debug_output:
            self.logger().info(f"dust_vol_limit_price_sell: {self.dust_vol_limit_price_sell}")

        # BUY side strategy
        if self.df_order_book_buy.at[self.df_order_book_buy.last_valid_index(), "cumsum"] <= self.dust_vol_buy:
            self.dust_vol_limit_price_buy = self.df_order_book_buy.iat[self.df_order_book_buy.last_valid_index(), 0]
        else:
            self.dust_vol_limit_price_buy = round(self.df_order_book_buy.loc[self.df_order_book_buy['cumsum'] > self.dust_vol_buy].iat[0, 0], self.order_price_precision)

        if debug_output:
            self.logger().info(f"dust_vol_limit_price_buy: {self.dust_vol_limit_price_buy}")

        if debug_output:
            self.logger().info(f"hedge_price_sell: {self.hedge_price_sell}")
            self.logger().info(f"hedge_price_buy: {self.hedge_price_buy}")   

        # setting the price based on the max allowable volume above it
        self.planned_order_price_sell = self.dust_vol_limit_price_sell - self.order_price_step_sell
        if debug_output:
            self.logger().info(f"Calculated planned_order_price_sell: {self.planned_order_price_sell} = {self.dust_vol_limit_price_sell} - {self.order_price_step_sell}")

        self.planned_order_price_buy = self.dust_vol_limit_price_buy + self.order_price_step_buy
        if debug_output:
            self.logger().info(f"planned_order_price_buy: {self.planned_order_price_buy}")

        # if the planned price (that is within the volume window) is better than the hedge price (if it is profitable, to say differently)
        self.order_cond_sell = self.planned_order_price_sell >= self.hedge_price_sell
        if debug_output:
            self.logger().info(f"order_cond_sell: {self.order_cond_sell}")

        self.order_cond_buy = self.planned_order_price_buy <= self.hedge_price_buy
        if debug_output:
            self.logger().info(f"order_cond_buy: {self.order_cond_buy}")

        # If the planned price is not profitable, set it close to the hedge price within some positive distance
        
        # SELL side strategy. 
        if not self.order_cond_sell:
            self.planned_order_price_sell = float(self.hedge_price_sell) + float(self.hedge_price_step_sell)
            if debug_output:
                self.logger().info(f"Adjusted planned_order_price_sell: {self.planned_order_price_sell} = {self.hedge_price_sell} + {self.hedge_price_step_sell}")
            
        # BUY side strategy
        if not self.order_cond_buy:
            self.planned_order_price_buy = float(self.hedge_price_buy) - float(self.hedge_price_step_buy)
            if self.planned_order_price_buy <= 0:
                self.planned_order_price_buy = float(self.hedge_price_buy) * 0.98    
            if debug_output:
                self.logger().info(f"Adjusted planned_order_price_buy: {self.planned_order_price_buy} = {self.hedge_price_buy} - {self.hedge_price_step_buy}")

    
    def check_min_order_amount_for_maker_order_creation(self, order_amount):
        if order_amount >= self.min_notional_for_maker_order_creation:
            return True
        else:
            return False

    def edit_order_condition(self, side=TradeType.BUY, debug_output=False):
        if side == TradeType.BUY: 
            profit_achieved = float(self.active_buy_order.price) <= self.hedge_price_buy

            if not profit_achieved:
                if debug_output:
                    self.logger().info(f"{side} edit order required: Not Profitable anymore")
                return True 

            # hedge price is above the dust volume - best scenario
            # hedge_above_dust_vol = (self.hedge_price_buy >= self.dust_vol_limit_price_buy)
            hedge_above_dust_vol_buy = (float(self.hedge_price_buy) - float(self.dust_vol_limit_price_buy)) >= float(self.order_price_step_buy)

            # if the existing order price is between the dust_vol_limit_price and the planned newly calc price
            # active_order_price_is_between_the_dust_price_and_one_step_above_price = (self.active_buy_order.price >= self.dust_vol_limit_price_buy and 
            #                                                                          self.active_buy_order.price <= self.planned_order_price_buy)
            
            # The below condition is for attaching the order to the LATEST price
            # active_order_price_is_between_the_dust_price_and_one_step_above_price = (
            #     self.active_buy_order.price >= self.dust_vol_limit_price_buy and 
            #     self.active_buy_order.price <= (self.active_buy_order.price + self.order_price_safe_distance_buy))
            
            # The below condition is for attaching the order to the DUST VOLUME price
            active_order_price_is_between_the_dust_price_and_one_step_above_price = (
                float(self.active_buy_order.price) >= self.dust_vol_limit_price_buy and 
                float(self.active_buy_order.price) <= (self.dust_vol_limit_price_buy + self.order_price_safe_distance_buy))

    #         if debug_output:
    #             self.logger().info(f"active_buy_order.price: {self.active_buy_order.price} ({type(self.active_buy_order.price).__name__}), "
    #   f"dust_vol_limit_price_buy: {self.dust_vol_limit_price_buy} ({type(self.dust_vol_limit_price_buy).__name__}), "
    #   f"order_price_safe_distance_buy: {self.order_price_safe_distance_buy} ({type(self.order_price_safe_distance_buy).__name__}), "
    #   f"active_order_price_is_between_the_dust_price_and_one_step_above_price: {active_order_price_is_between_the_dust_price_and_one_step_above_price} "
    #   f"({type(active_order_price_is_between_the_dust_price_and_one_step_above_price).__name__})")


            # active order price is between the hedge and one step lower (under the hedge condition is met on the first step - must be profitable)
            active_order_price_is_above_one_step_lower_the_hedge = (float(self.active_buy_order.price) >= 
                                                                          (float(self.hedge_price_buy) - float(self.hedge_price_safe_distance_buy)))
            
            if hedge_above_dust_vol_buy:
                if debug_output:
                    self.logger().info(f"""{side} hedge price is above the dust volume - best scenario
                                       hedge: {self.hedge_price_buy} >= dust_vol: {self.dust_vol_limit_price_buy}""")             
                
                if active_order_price_is_between_the_dust_price_and_one_step_above_price:
                    if debug_output:
                        self.logger().info(f"""{side} the existing order price is between the dust_vol_limit_price and the safe distance ({self.order_price_safe_distance_buy}) 
                                       {self.active_buy_order.price} >= {self.dust_vol_limit_price_buy} and                                                           
                                        {self.active_buy_order.price} <= {(self.dust_vol_limit_price_buy + 
                                                                           self.order_price_safe_distance_buy)}""")

                    return False
            else: # hedge price is under the dust volume - edge scenario, not the best
                if debug_output:
                    self.logger().info(f"""{side} hedge price is close to the dust volume - edge scenario, not the best
                                       hedge: {self.hedge_price_buy} < dust_vol: {self.dust_vol_limit_price_buy}""")
                
                if active_order_price_is_above_one_step_lower_the_hedge:
                    if debug_output:
                        self.logger().info(f"""{side} active order price is between the hedge and and the safe distance ({self.hedge_price_safe_distance_buy})
                                       {self.active_buy_order.price} >= 
                                       {self.hedge_price_buy} - {self.hedge_price_safe_distance_buy}""")

                    return False
            # the existing order price is out of the allowable limits
            if debug_output:
                self.logger().info(f"""{side} active order price is out of limits - need editing""") 
            return True


        if side == TradeType.SELL:
            profit_achieved = float(self.active_sell_order.price) >= self.hedge_price_sell
            
            if not profit_achieved:
                if debug_output:
                    self.logger().info(f"{side} edit order required: Not Profitable anymore")
                return True 

            # hedge price is above the dust volume - best scenario
            # hedge_above_dust_vol = (self.hedge_price_sell <= self.dust_vol_limit_price_sell)

            hedge_above_dust_vol_sell = (float(self.dust_vol_limit_price_sell) - float(self.hedge_price_sell)) >= float(self.order_price_step_sell)

            # if the existing order price is between the dust_vol_limit_price and the planned newly calc price
            # active_order_price_is_between_the_dust_price_and_one_step_above_price = (self.active_sell_order.price <= self.dust_vol_limit_price_sell and 
            #                                                                          self.active_sell_order.price >= self.planned_order_price_sell)

            # active_order_price_is_between_the_dust_price_and_one_step_above_price = (
            #     self.active_sell_order.price <= self.dust_vol_limit_price_sell and 
            #     self.active_sell_order.price >= (self.active_sell_order.price - self.order_price_safe_distance_buy))
            active_sell_price_float = float(self.active_sell_order.price)

            active_order_price_is_between_the_dust_price_and_one_step_above_price = (
                active_sell_price_float <= self.dust_vol_limit_price_sell and 
                active_sell_price_float >= (self.dust_vol_limit_price_sell - self.order_price_safe_distance_sell))
            if debug_output:
                self.logger().info(f"active_sell_order.price: {active_sell_price_float} (type: {type(active_sell_price_float)}), "
      f"dust_vol_limit_price_sell: {self.dust_vol_limit_price_sell} (type: {type(self.dust_vol_limit_price_sell)}), "
      f"active_sell_price_float <= self.dust_vol_limit_price_sell is: {active_sell_price_float <= self.dust_vol_limit_price_sell},"
      f"order_price_safe_distance_sell: {self.order_price_safe_distance_sell}, "
      f"(dust_vol_limit_price_sell - order_price_safe_distance_sell): {self.dust_vol_limit_price_sell - self.order_price_safe_distance_sell}, "
      f"active_sell_price_float >= (self.dust_vol_limit_price_sell - self.order_price_safe_distance_sell) is: {active_sell_price_float >= (self.dust_vol_limit_price_sell - self.order_price_safe_distance_sell)}"
      f"Condition result: {active_order_price_is_between_the_dust_price_and_one_step_above_price}")

            # active order price is between the hedge and one step lower (under the hedge condition is met on the first step - must be profitable)
            active_order_price_is_above_one_step_lower_the_hedge = (float(self.active_sell_order.price) <= 
                                                                          (float(self.hedge_price_sell) + float(self.hedge_price_safe_distance_sell)))
            
            if hedge_above_dust_vol_sell:
                if debug_output:
                    self.logger().info(f"""{side} hedge price is above the dust volume - best scenario
                                        hedge: {self.hedge_price_sell} <= dust_vol: {self.dust_vol_limit_price_sell}""")

                if active_order_price_is_between_the_dust_price_and_one_step_above_price:
                    if debug_output:
                        self.logger().info(f"""{side} the existing order price is between the dust_vol_limit_price and the safe distance ({self.order_price_safe_distance_buy}) 
                                    {self.active_sell_order.price} <= {self.dust_vol_limit_price_sell} and                                                           
                                        {self.active_sell_order.price} >= {(self.dust_vol_limit_price_sell - self.order_price_safe_distance_sell)}""")
                    return False
            else: # hedge price is under the dust volume - edge scenario, not the best
                if debug_output:
                    self.logger().info(f""""{side} hedge price is close to the dust volume - edge scenario, not the best
                                    hedge: {self.hedge_price_sell} > dust_vol: {self.dust_vol_limit_price_sell}""")                    
                
                if active_order_price_is_above_one_step_lower_the_hedge:
                    if debug_output:
                        self.logger().info(f"""{side} active order price is between the hedge and safe distance ({self.hedge_price_safe_distance_sell})
                                    {self.active_sell_order.price} <= 
                                    {self.hedge_price_sell} + {self.hedge_price_safe_distance_sell}""")
                                            
                    return False
            # the existing order price is out of the allowable limits
            if debug_output:
                self.logger().info(f"""{side} active order price is out of limits - need editing""")    
            return True
        
        raise TypeError(f"Expected argument of type TradeType, but got {type(side).__name__}")

    def edit_order(self, order, debug_output=False):
        new_order_side = TradeType.BUY if order.is_buy else TradeType.SELL
        
        if new_order_side == TradeType.BUY:
            new_order_size = self.order_size_buy
        else:
            new_order_size = self.order_size_sell
        
        if new_order_side == TradeType.BUY:
            new_order_price = self.planned_order_price_buy
        else:
            new_order_price = self.planned_order_price_sell

        if not self.check_min_order_amount_for_maker_order_creation(new_order_size):
            if debug_output:
                self.logger().info(f"Edit Order: the planned {new_order_side} order amount is too low: {new_order_size}, can't place a new order")
            return "cancel"

        if not self.order_size_and_price_are_valid(new_order_side):
            self.logger().info(f"Edit Order: the planned {new_order_side} order size or price are not valid ({new_order_size} and {new_order_price})")
            return "cancel"

        self.custom_cancel_order(order)


        if debug_output:
            self.logger().info(f"Editing {new_order_side} order.")
        
        if new_order_side == TradeType.BUY: 
            self.create_buy_order_after_cancel_in_current_tick_cycle = True
            self.buy_order_client_id_to_edit_in_current_tick_cycle = order.client_order_id
        if new_order_side == TradeType.SELL:
            self.create_sell_order_after_cancel_in_current_tick_cycle = True
            self.sell_order_client_id_to_edit_in_current_tick_cycle = order.client_order_id
        
        # self.create_new_maker_order(side=new_order_side)
    
    # def amend_buy_order(self, debug_output=False) -> Optional[str]:
    #     # the below two vars are float
    #     new_order_size = self.order_size_buy
    #     new_order_price = self.planned_order_price_buy
   
    #     new_order_side = "BUY"
    #     order_id = self.buy_order_exchange_id
    #     try: 
    #         new_order_size_decimal = Decimal(str(new_order_size))
    #         new_order_size_decimal = self.connectors[self.maker].quantize_order_amount(self.maker_pair, new_order_size_decimal)        
    #     except (ValueError, InvalidOperation) as e:
    #         self.logger().info(f"Amend Order: the planned {new_order_side} order size is not valid ({new_order_size}) - {e}")
    #         return "cancel"
    #     # we only check the order amount if it will be different than before
    #     if self.active_buy_order.quantity != new_order_size_decimal:
        
    #         if not self.check_min_order_amount_for_maker_order_creation(new_order_size):
    #             if debug_output:
    #                 self.logger().info(f"Edit BUY Order: the planned {new_order_side} order amount is too low: {new_order_size}, can't place a new order")
    #             return "cancel"
    #     else:
    #         new_order_size_decimal = None

    #     try: 
    #         new_order_price_decimal = Decimal(
    #             self.round_to_precision(new_order_price, self.order_price_precision, rounding_mode='floor'))
    #         new_order_price_decimal = self.connectors[self.maker].quantize_order_price(self.maker_pair, new_order_price_decimal)        
    #     except Exception as e:
    #         self.logger().info(f"Amend Order: the planned {new_order_price} order price is not valid ({new_order_size}) - {e}")
    #         return "cancel"

    #     # self.logger().info(f"self.active_buy_order.price == new_order_price_decimal -> ({self.active_buy_order.price == new_order_price_decimal}\n{self.active_buy_order.price} == {new_order_price_decimal})")
        
    #     # it means there's nothing to update
    #     if self.active_buy_order.price == new_order_price_decimal and new_order_size_decimal is None:
    #         return

    #     try:
    #         amend_result = self.connectors[self.maker].amend_order(order_id, new_order_size_decimal, new_order_price_decimal, self.maker_pair)
    #     except Exception as e:
    #         error_message = f"Tried to amend order {order_id} buy failed. An error of type {type(e).__name__} occurred while AMENDING order: {e}"
    #         self.logger().error(error_message, exc_info=True)
    #         self.telegram_utils.send_unformatted_message(error_message)
    #         return "cancel"

    #     # ether amount or price can be None, or both
    #     # these values will be used later to check if the order
    #     # has been amended successfully by comparing these values
    #     # to the values of the already presumably amended in_flight_order
    #     # self.logger().info(f"amend_result: {amend_result}")  
    #     self.buy_order_amended_previously = {
    #         "order_id": self.active_buy_order.client_order_id,
    #         "amend_id": None,
    #         "amount": new_order_size_decimal,
    #         "price": new_order_price_decimal,
    #         "amendment_try_timestamp": time.time()
    #     }


    #     self.logger().info(f"Amending {new_order_side} order {self.active_buy_order.client_order_id} ({order_id}). Amount: from {self.active_buy_order.quantity} to {new_order_size_decimal} and price: from {self.active_buy_order.price} to {new_order_price_decimal}")

    # def amend_sell_order(self, debug_output=False) -> Optional[str]:
    #     # the below two vars are float
    #     new_order_size = self.order_size_sell
    #     new_order_price = self.planned_order_price_sell
   
    #     new_order_side = "SELL"
    #     order_id = self.sell_order_exchange_id
    #     try: 
    #         new_order_size_decimal = Decimal(str(new_order_size))
    #         new_order_size_decimal = self.connectors[self.maker].quantize_order_amount(self.maker_pair, new_order_size_decimal)        
    #     except (ValueError, InvalidOperation) as e:
    #         self.logger().info(f"Amend Order: the planned {new_order_side} order size is not valid ({new_order_size}) - {e}")
    #         return "cancel"
    #     # we only check the order amount if it will be different than before
    #     if self.active_sell_order.quantity != new_order_size_decimal:
        
    #         if not self.check_min_order_amount_for_maker_order_creation(new_order_size):
    #             if debug_output:
    #                 self.logger().info(f"Edit SELL Order: the planned {new_order_side} order amount is too low: {new_order_size}, can't place a new order")
    #             return "cancel"
    #     else:
    #         new_order_size_decimal = None

    #     try: 
    #         new_order_price_decimal = Decimal(
    #             self.round_to_precision(new_order_price, self.order_price_precision, rounding_mode='ceil'))
    #         new_order_price_decimal = self.connectors[self.maker].quantize_order_price(self.maker_pair, new_order_price_decimal)        
    #     except Exception as e:
    #         self.logger().info(f"Amend Order: the planned {new_order_price} order price is not valid ({new_order_size}) - {e}")
    #         return "cancel"

    #     # self.logger().info(f"self.active_sell_order.price == new_order_price_decimal -> ({self.active_sell_order.price == new_order_price_decimal}\n{self.active_sell_order.price} == {new_order_price_decimal})")
        
    #     # it means there's nothing to update
    #     if self.active_sell_order.price == new_order_price_decimal and new_order_size_decimal is None:
    #         return

    #     try:
    #         amend_result = self.connectors[self.maker].amend_order(order_id, new_order_size_decimal, new_order_price_decimal, self.maker_pair)
    #     except Exception as e:
    #         error_message = f"Tried to amend order {order_id} sell failed. An error of type {type(e).__name__} occurred while AMENDING order: {e}"
    #         self.logger().error(error_message, exc_info=True)
    #         self.telegram_utils.send_unformatted_message(error_message)
    #         return "cancel"

    #     # ether amount or price can be None, or both
    #     # these values will be used later to check if the order
    #     # has been amended successfully by comparing these values
    #     # to the values of the already presumably amended in_flight_order
    #     # self.logger().info(f"amend_result: {amend_result}")  
    #     self.sell_order_amended_previously = {
    #         "order_id": self.active_sell_order.client_order_id,
    #         "amend_id": None,
    #         "amount": new_order_size_decimal,
    #         "price": new_order_price_decimal,
    #         "amendment_try_timestamp": time.time()
    #     }


    #     self.logger().info(f"Amending {new_order_side} order {self.active_sell_order.client_order_id} ({order_id}). Amount: from {self.active_sell_order.quantity} to {new_order_size_decimal} and price: from {self.active_sell_order.price} to {new_order_price_decimal}")

    def amend_order_any_side(self, 
                            order_side: str, 
                            order_size: float, 
                            planned_order_price: float, 
                            order_exchange_id: str, 
                            active_order: LimitOrder, 
                            amended_previously: dict, 
                            debug_output=False) -> Optional[str]:
        """
        A universal function to amend an order for both BUY and SELL sides.

        Args:
            order_side (str): "BUY" or "SELL" to specify the order side.
            order_size (float): The new order size.
            planned_order_price (float): The new order price.
            order_exchange_id (str): The exchange order ID to amend.
            active_order: The active order object (e.g., self.active_buy_order).
            amended_previously (dict): Dictionary storing amendment history from the previous amendment try.
            debug_output (bool): Whether to print debugging information.

        Returns:
            Optional[str]: Returns "cancel" if the order amendment fails or is invalid. This will later signal to cancel the active order becase there were some problems with it. Sometimes returns None.
        """
        try: 
            # Convert the order size to a Decimal and quantize
            new_order_size_decimal = Decimal(str(order_size))
            new_order_size_decimal = self.connectors[self.maker].quantize_order_amount(self.maker_pair, new_order_size_decimal)        
        except (ValueError, InvalidOperation) as e:
            self.logger().info(f"Amend Order: the planned {order_side} order size is not valid ({order_size}) - {e}")
            return "cancel"

        # Check if the new size is different from the active order
        if active_order.quantity != new_order_size_decimal:
            if not self.check_min_order_amount_for_maker_order_creation(order_size):
                if debug_output:
                    self.logger().info(f"Edit Order: the planned {order_side} order amount is too low: {order_size}, can't place a new order")
                return "cancel"
        else:
            new_order_size_decimal = None

        try: 
            # Adjust rounding mode based on order side
            rounding_mode = 'floor' if order_side == "BUY" else 'ceil'

            # Convert the order price to a Decimal and quantize
            new_order_price_decimal = Decimal(
                self.round_to_precision(planned_order_price, self.order_price_precision, rounding_mode=rounding_mode))
            new_order_price_decimal = self.connectors[self.maker].quantize_order_price(self.maker_pair, new_order_price_decimal)        
        except Exception as e:
            self.logger().info(f"Amend Order: the planned {planned_order_price} order price is not valid ({order_size}) - {e}")
            return "cancel"

        # If both price and size are unchanged, nothing to update
        if active_order.price == new_order_price_decimal and new_order_size_decimal is None:
            return

    # def amend_order(self,
    #          client_order_id: str,       
    #          amount: Decimal,
    #          price: Decimal,
    #          **kwargs) -> str:
        order_price_before_amendment = active_order.price
        order_size_before_amendment = active_order.quantity
        try:
            # Send the amend order request to the exchange
            amend_result = self.connectors[self.maker].amend_order(active_order.client_order_id, new_order_size_decimal, new_order_price_decimal)
        except Exception as e:
            error_message = f"Tried to amend order {order_exchange_id} {order_side.lower()} failed. An error of type {type(e).__name__} occurred while AMENDING order: {e}"
            self.logger().error(error_message, exc_info=True)
            self.telegram_utils.send_unformatted_message(error_message)
            return "cancel"

        # Store amendment details for future comparison
        amended_previously.update({
            "order_id": active_order.client_order_id,
            "amend_id": None,
            "amount": new_order_size_decimal,
            "price": new_order_price_decimal,
            "amendment_try_timestamp": time.time()
        })

        # self.logger().info(
        #     f"Amended {order_side} order {active_order.client_order_id} ({order_exchange_id}). "
        #     f"Amount: from {order_size_before_amendment} to {new_order_size_decimal} and "
        #     f"price: from {order_price_before_amendment} to {new_order_price_decimal}"
        # )
        

    def cancel_all_active_limit_orders(self, debug_output=False):
        for order in self.active_limit_orders:
            self.custom_cancel_order(order=order, debug_output=debug_output)    


    def custom_cancel_order(self, order, debug_output=False):
        if order is not None:
            if debug_output:
                self.logger().info(f"custom_cancel_order: Cancelling order {order.client_order_id}...")
            self.connectors[self.maker].cancel(order.trading_pair, order.client_order_id)
    
    def cancel_order_condition(self, side=TradeType.BUY, debug_output=False):
        return False

    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        # self.logger().info(f"catched buy order!!!!!!!!")
        self.connectors[self.maker].update_balances_offline()
        # pass

    def did_create_sell_order(self, event: SellOrderCreatedEvent):
        self.connectors[self.maker].update_balances_offline()
        # pass

    def did_cancel_order(self, event: OrderCancelledEvent):
        # self.logger().info(f"Catched Order cancel!!!")
        if (self.create_buy_order_after_cancel_in_current_tick_cycle == True 
            and self.buy_order_client_id_to_edit_in_current_tick_cycle == event.order_id):
            if self.amend_condition(self.in_flight_buy_order):
                return
            # self.logger().info(f"Catched Order cancel and creating a new order! Cancelled order id: {self.buy_order_client_id_to_edit_in_current_tick_cycle}")
            self.create_new_maker_order(side=TradeType.BUY)

        if (self.create_sell_order_after_cancel_in_current_tick_cycle == True 
            and self.sell_order_client_id_to_edit_in_current_tick_cycle == event.order_id):
            if self.amend_condition(self.in_flight_sell_order):
                return            
            self.create_new_maker_order(side=TradeType.SELL)

    def did_fail_order(self, event: MarketOrderFailureEvent):
        # self.logger().info(f"Tracked order {event.order_id} FAILED catched event")
        delay = self.after_failed_order_delay
        for order in self.get_active_orders(connector_name=self.maker):        
            if order.client_order_id == event.order_id:
                self.idle_timers.append(Timer(name="failed_order_idle_timer", duration=delay))
                
                notif_text = f"Maker order {order.client_order_id} failed, going idle for {delay/1000} seconds"
                self.telegram_utils.send_unformatted_message(notif_text)
                self.telegram_utils.send_unformatted_message(f"Exception: {self.get_latest_exception_traceback()}")

        
        # orders_from_all_connectors = self.order_tracker.tracked_limit_orders
        # connector = self.connectors[self.maker]
        # limit_orders = [o[1] for o in orders_from_all_connectors if o[0] == connector]

        # for order in limit_orders:
        #     self.logger().info(f"Active orders: {order.client_order_id}")
        #     if order.client_order_id == event.order_id:
        #         self.logger().info(f"Tracked order {event.order_id} FAILED catch event") 

    def get_latest_exception_traceback(self):
        # Get the latest exception information

        config_name = self.hummingbot.strategy_file_name.split(".")[0]
        # config_name = f"conf_{os.path.splitext(self.script_file_name)[0]}"
        # self.logger().info(f"CONFIG NAME: {config_name}")
        log_file = self.find_log_file(config_name)
        latest_error_message = self.extract_latest_full_error_message(log_file)
        return latest_error_message

    def show_tracked_orders(self):
        
        for order in self.get_active_orders(connector_name=self.maker):        
            self.logger().info(f"Active orders: {order.client_order_id}")

        # self.logger().info(f"Showing all tracked orders.......")
        # orders_from_all_connectors = self.order_tracker.tracked_limit_orders
        # connector = self.connectors[self.maker]
        # limit_orders = [o[1] for o in orders_from_all_connectors if o[0] == connector]
        
        # for order in orders_from_all_connectors:
        #     self.logger().info(f"Tracked orders: {order.client_order_id}") 


    def is_active_maker_order(self, event: OrderFilledEvent):
        """
        Helper function that checks if order is an active order on the maker exchange
        """
        for order in self.get_active_orders(connector_name=self.maker):
            if order.client_order_id == event.order_id:
                return True
        return False

    def get_order_by_event(self, event: OrderFilledEvent):
        """
        Returns an order object if it's an active order or None
        """
        for order in self.get_active_orders(connector_name=self.maker):
            if order.client_order_id == event.order_id:
                return order
        return None
    
    def get_taker_order_by_event(self, event: OrderFilledEvent):
        """
        Returns an order object if it's an active taker order or None
        """
        for order in self.get_active_orders(connector_name=self.taker):
            if order.client_order_id == event.order_id:
                return order
        return None

    def get_taker_completed_order_by_event(self, event: Union[SellOrderCompletedEvent, BuyOrderCompletedEvent]):
        """
        Returns an order object if it's an active taker order or None
        """
        for order in self.get_active_orders(connector_name=self.taker):
            if order.client_order_id == event.order_id:
                return order
        return None        
    
    def format_filled_order_message(self, client_order_id, amount, price, quote_currency, base_currency, size=None, is_buy_order=True, is_maker_exchange_order=True, profit_dict={}):
        if is_buy_order:
            order_direction = "BUY"
        else:
            order_direction = "SELL"

        if is_maker_exchange_order:
            maker_or_taker = "MAKER"
        else:
            maker_or_taker = "TAKER"    

        additional_header = ""

        if price is None:
            # size: Decimal
            # amount: Decimal
            price = size / amount
            # price = round(price, self.order_price_precision)
            # additional_header = "Fully Completed! "

            if not is_maker_exchange_order:
                taker_rules = self.connectors[self.taker].trading_rules.get(self.taker_pair)
                if taker_rules:
                    if taker_rules.min_price_increment is not None:
                        significant_digits = -int(f"{taker_rules.min_price_increment:e}".split('e')[-1])
                        price = round(price, significant_digits)


        if size is None:
            size = amount * price    

        size = round(size, self.quote_precision_for_output)

        log_message = f"{additional_header}Filled {maker_or_taker} {order_direction} order for {size} {quote_currency} ({amount} {base_currency} at {price} {quote_currency}). Order ID: {client_order_id}"
        
        telegram_message = f"{additional_header}Filled <b>{maker_or_taker} {order_direction}</b> order for <b>{size} {quote_currency}</b> (<b>{amount} {base_currency}</b> at {price} {quote_currency}). Order ID: {client_order_id}"

        # profits output
        if not is_maker_exchange_order and profit_dict:

                # profit_dict={'order_profit_without_fees': order_profit_without_fees,
                #              'order_maker_fee': order_maker_fee,
                #              'order_taker_fee': order_taker_fee,
                #              'total_order_profit': total_order_profit,
                #              }

                # dictionary structure for accumulated profits:
                # self.accumulated_profits = {
                #     'total_order_profit': Decimal("0"),
                #     'total_order_profit_excluding_maker_fee': Decimal("0"),
                #     'total_maker_fees': Decimal("0"),
                #     'total_taker_fees': Decimal("0"),
                # }


            order_profit_without_fees = profit_dict.get('order_profit_without_fees')
            order_maker_fee = profit_dict.get('order_maker_fee')
            order_taker_fee = profit_dict.get('order_taker_fee')
            total_order_profit = profit_dict.get('total_order_profit')
            maker_order_id = profit_dict.get('maker_order_id')

            if (order_profit_without_fees is not None and 
                order_maker_fee is not None and
                order_taker_fee is not None and
                total_order_profit is not None):

                # profit_in_perc = (total_order_profit / size) * 100
                try:
                    profit_in_perc = (total_order_profit / size) * 100
                except ZeroDivisionError:
                    profit_in_perc = 0

                starting_time = time.strftime('%-d %b, %H:%M', time.gmtime(self.bot_start_time_timestamp))

                elapsed_time = self.get_elapsed_time(self.bot_start_time_timestamp)

                log_message = f"{log_message}\n\nProfit: {total_order_profit} {quote_currency} or {profit_in_perc:.3f}%\n({order_profit_without_fees} - {order_maker_fee} - {order_taker_fee})\nMatched Order ID: {maker_order_id}\n\nSince {starting_time} ({elapsed_time}): {self.accumulated_profits['total_order_profit']} {quote_currency}\nExcl. maker fees: {self.accumulated_profits['total_order_profit_excluding_maker_fee']} {quote_currency}"
                
                telegram_message = f"{telegram_message}\n\nProfit: <b>{total_order_profit} {quote_currency}</b> or {profit_in_perc:.3f}%\n({order_profit_without_fees} - {order_maker_fee} - {order_taker_fee})\nMatched Order ID: {maker_order_id}\n\nSince {starting_time} ({elapsed_time}): <b>{self.accumulated_profits['total_order_profit']} {quote_currency}</b>\nExcl. maker fees: {self.accumulated_profits['total_order_profit_excluding_maker_fee']} {quote_currency}"


        return {'log_message': log_message, 
                'telegram_message': telegram_message}

    def get_elapsed_time(self, start_timestamp):
        elapsed_seconds = time.time() - start_timestamp
        days = int(elapsed_seconds // 86400)  # Seconds in a day
        hours = int((elapsed_seconds % 86400) // 3600)  # Remaining seconds to hours
        minutes = int((elapsed_seconds % 3600) // 60)  # Remaining seconds to minutes
        return f"{days} d. {hours:02}:{minutes:02}"

    def did_fill_order(self, event: OrderFilledEvent):
        '''
        Both maker exchange and taker exchange filled events are processed 
        '''

        # Processing Maker order fill event
        filled_order = self.get_order_by_event(event)

        order_id_for_profits_calculation = None

        ############### BUY ORDER ##################
        if filled_order is not None and event.trade_type == TradeType.BUY:
            order_message = self.format_filled_order_message(
                filled_order.client_order_id, 
                event.amount, 
                event.price, 
                filled_order.quote_currency, 
                filled_order.base_currency, 
                is_buy_order=True, 
                is_maker_exchange_order=True
            )
            
            self.logger().info(order_message['log_message'])
            self.telegram_utils.send_unformatted_message(order_message['telegram_message'])            

            if self.one_order_only:
                self.logger().info("One order only! No more orders should be placed.")
                self.exit_bot_flag = True

            taker_sell_result = str(self.taker_sell_by_volume_price)
            
            sell_price_with_slippage = Decimal(taker_sell_result) * Decimal(str(self.taker_best_bid_price_coef))

            taker_rules = self.connectors[self.taker].trading_rules.get(self.taker_pair)
            if taker_rules:
                taker_price_increment = Decimal(str(taker_rules.min_price_increment))
                if taker_price_increment:
                    sell_price_with_slippage = Decimal(self.connectors[self.taker].quantize_order_price(self.taker_pair, sell_price_with_slippage))    

            taker_sell_order_amount = event.amount

            # correcting base balance for small value of unmatched small maker orders
            if self.enable_small_base_balance_corrections:
                if self.pending_small_base_amount_to_fix > Decimal("0"):
                    taker_sell_order_amount += self.pending_small_base_amount_to_fix
                    self.pending_small_base_amount_to_fix = Decimal("0")
                    
                    message = f"Correcting taker SELL order amount. New value: {taker_sell_order_amount}"
                    # self.logger().notify(message)

                    self.send_beautiful_message_to_log_and_telegram(message, message)

            # check if there's enough base balance on taker
            if event.amount > Decimal(str(self.taker_base_free)):
                taker_sell_order_amount = Decimal(str(self.taker_base_free))
                self.logger().info(f"Correcting SELL LIMIT amount on taker to {taker_sell_order_amount} because the quote balance on taker is not enough")
                
            
            
            sell_order = OrderCandidate(trading_pair=self.taker_pair, is_maker=False, order_type=OrderType.LIMIT, 
                                        order_side=TradeType.SELL, amount=Decimal(taker_sell_order_amount), price=sell_price_with_slippage)
            
            if self.check_order_min_size_before_placing(self.taker, sell_order, notif_output=True):
                try:
                    order_id_for_profits_calculation = self.place_order(self.taker, sell_order)
                    # {'maker_order': maker_order: LimitOrder, 'taker_order_id': taker_order_inner_id: str}
                    self.order_fills_profits_pairs.append(
                            {'maker_order': event, 
                            'taker_order_id': sell_order
                            }
                         )
                except Exception as e:
                    error_message = f"An error of type {type(e).__name__} occurred while placing taker SELL order: {e}"
                    self.logger().error(error_message, exc_info=True)
                    self.telegram_utils.send_unformatted_message(error_message)
                else:
                    self.logger().info(f"Sending TAKER SELL order for {taker_sell_order_amount} {filled_order.base_currency} at price: {sell_price_with_slippage} {filled_order.quote_currency}")
                    # self.logger().info(f"Order fills profits: {self.order_fills_profits_pairs}")
        
        ################# SELL ORDER #################
        elif filled_order is not None and event.trade_type == TradeType.SELL:
            order_message = self.format_filled_order_message(
                filled_order.client_order_id, 
                event.amount, 
                event.price, 
                filled_order.quote_currency, 
                filled_order.base_currency, 
                is_buy_order=False, 
                is_maker_exchange_order=True
            )
            
            self.logger().info(order_message['log_message'])
            self.telegram_utils.send_unformatted_message(order_message['telegram_message'])            

            if self.one_order_only:
                self.logger().info("One order only! No more orders should be placed.")
                self.exit_bot_flag = True

            taker_buy_result = str(self.taker_buy_by_volume_price)
            
            buy_price_with_slippage = Decimal(taker_buy_result) * Decimal(str(self.taker_best_ask_price_coef))

            taker_rules = self.connectors[self.taker].trading_rules.get(self.taker_pair)
            if taker_rules:
                taker_price_increment = Decimal(str(taker_rules.min_price_increment))
                if taker_price_increment:
                    buy_price_with_slippage = Decimal(self.connectors[self.taker].quantize_order_price(self.taker_pair, buy_price_with_slippage) + taker_price_increment)

            taker_buy_order_amount = event.amount

            # correcting base balance for small value of unmatched small maker orders
            if self.enable_small_base_balance_corrections:
                if self.pending_small_base_amount_to_fix < Decimal("0"):
                    taker_buy_order_amount += abs(self.pending_small_base_amount_to_fix)
                    self.pending_small_base_amount_to_fix = Decimal("0")
                    # self.logger().notify(f"correcting taker buy order amount. New value: {taker_buy_order_amount}")

                    message = f"Correcting taker BUY order amount. New value: {taker_buy_order_amount}"
                    # self.logger().notify(message)

                    self.send_beautiful_message_to_log_and_telegram(message, message)                

            # check if there's enough quote balance on taker
            if Decimal(str(self.taker_quote_free)) < Decimal(event.amount) * Decimal(buy_price_with_slippage):
                
                taker_buy_order_amount = (
                    Decimal(str(self.taker_quote_free - self.taker_min_balance_quote)) / Decimal(buy_price_with_slippage)
                )
                self.logger().info(f"Correcting BUY LIMIT amount on taker to {taker_buy_order_amount} because the quote balance on taker is not enough")


            
            buy_order = OrderCandidate(trading_pair=self.taker_pair, is_maker=False, order_type=OrderType.LIMIT, 
                                        order_side=TradeType.BUY, amount=Decimal(taker_buy_order_amount), price=buy_price_with_slippage)
            
            if self.check_order_min_size_before_placing(self.taker, buy_order, notif_output=True):
                try:
                    order_id_for_profits_calculation = self.place_order(self.taker, buy_order)
                    self.order_fills_profits_pairs.append(
                            {'maker_order': event, 
                            'taker_order_id': buy_order
                            }
                         )                    
                except Exception as e:
                    error_message = f"An error of type {type(e).__name__} occurred while placing taker BUY order: {e}"
                    self.logger().error(error_message, exc_info=True)
                    self.telegram_utils.send_unformatted_message(error_message)                    
                else:
                    self.logger().info(f"Sending TAKER BUY order for {taker_buy_order_amount} {filled_order.base_currency} at price: {buy_price_with_slippage} {filled_order.quote_currency}")
                    # self.logger().info(f"Order fills profits: {self.order_fills_profits_pairs}")

        if filled_order is not None:

            # setting the latest order fill timestamp so that the total balance checker
            # don't run immediately after
            self.base_balance_checker.latest_order_fill_timestamp = time.time()    
            # Cancel all orders because balances have changed and 
            # the filled order may not have been filled completely
            # Also making a small delay to wait for the balances update
            # so the new orders could be added correctly
            self.logger().info("Cancelling all orders after maker filled")
            
            if self.amend_condition(self.in_flight_buy_order):
                self.check_in_flight_and_cancel(self.in_flight_buy_order)
            elif self.active_buy_order is not None:
                self.custom_cancel_order(order=self.active_buy_order)
            
            if self.amend_condition(self.in_flight_sell_order):        
                self.check_in_flight_and_cancel(self.in_flight_sell_order)
            elif self.active_sell_order is not None:    
                self.custom_cancel_order(order=self.active_sell_order)

            # self.custom_cancel_all_orders(debug_output=True)

            # If the active order is still not marked as completed but
            # the filled amount is equal to the active order amount we 
            # don't cancel such an order because it was fully completed
            
            # self.check_and_cancel_order(self.active_buy_order, event.trade_type, TradeType.BUY, event.amount)
            # self.check_and_cancel_order(self.active_sell_order, event.trade_type, TradeType.SELL, event.amount)
            
            durtaion = self.after_order_is_filled_delay
            # self.logger().info(f"Starting timer 'after_order_is_filled_timer' for {durtaion} ms")

            # Mexc exchange sends wrong quote balance after the sell order
            # through a websocket. That is why the balance should be updated forcefully
            # through a REST API. The first delay is started at the moment 
            # of the filled maker order. The first delay
            # is needed for the taker order to be fully filled and taker websocket messages to be 
            # sent. The second delay is needed to have enough time to get
            # the taker balance with a rest api.
            # The second delay will be equal to 
            # self.after_order_is_filled_delay - balance_rest_api_request_delay
            #
   
            if self.taker == 'mexc' and event.trade_type == TradeType.BUY:

                duration_ratio = 0.6
                # every calculation is in milliseconds
                balance_rest_api_request_delay = duration_ratio * self.after_order_is_filled_delay

                # The taker order is sent and processed fully,
                # all taker balances updated in less than 700 ms
                # Just to make sure the balances are updated 
                # before they are requested the delay is set
                # to not less than 1000 ms
                min_delay = 1200
                if balance_rest_api_request_delay < min_delay:
                    balance_rest_api_request_delay = min_delay                 
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Handle the case where no event loop is running
                    # For example, you might create a new event loop or log an error
                    self.logger().error("No running event loop found while trying to update MEXC balances")                
                else:
                    asyncio.run_coroutine_threadsafe(
                        self._run_async_update_balances(balance_rest_api_request_delay),
                        loop
                    )
                # durtaion = int(self.after_order_is_filled_delay * (1 - duration_ratio))

            self.idle_timers.append(Timer(name="after_order_is_filled_timer", duration=durtaion))
                  
        
        # Processing Taker order filled event
        # Sending notification if a taker order is filled
        # filled_order = self.get_taker_order_by_event(event)
        
        # if filled_order is not None:
        #     is_buy_order = (event.trade_type == TradeType.BUY)
           
        #     order_message = self.format_filled_order_message(
        #         filled_order.client_order_id, 
        #         event.amount, 
        #         event.price, 
        #         filled_order.quote_currency, 
        #         filled_order.base_currency, 
        #         is_buy_order=is_buy_order, 
        #         is_maker_exchange_order=False
        #     )
            
        #     # self.logger().info(order_message['log_message'])
        #     self.telegram_utils.send_unformatted_message(order_message['telegram_message'])             

#     def check_and_cancel_order(self, active_order, trade_type: TradeType, expected_trade_type: TradeType, amount: Decimal):
#         if active_order is not None and trade_type == expected_trade_type:
#             self.logger().info(f"Order mismatch: {active_order.quantity} != {amount} (condition: {active_order.quantity != amount})"
# )
#             if active_order.quantity != amount:
#                 self.logger().info(f"Canceling order {active_order.client_order_id}")
#                 self.custom_cancel_order(order=active_order, debug_output=False)

    def check_in_flight_and_cancel(self, in_flight_order: InFlightOrder):
        if in_flight_order is not None:
            if not in_flight_order.is_done:
                self.cancel(self.maker, in_flight_order.trading_pair, in_flight_order.client_order_id)

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        self.notify_about_completed_order(event)
        
    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        self.notify_about_completed_order(event)

    def notify_about_completed_order(self, event: Union[SellOrderCompletedEvent, BuyOrderCompletedEvent]):
        filled_order = self.get_taker_completed_order_by_event(event)
        
        if filled_order is not None:

            self.base_balance_checker.latest_order_fill_timestamp = time.time()    
                
                    # self.order_fills_profits_pairs.append(
                    #         {'maker_order': event, (OrderFilledEvent)
                    #         'taker_order_id': buy_order (OrderCandidate)
                    #         }
            is_buy_order = isinstance(event, BuyOrderCompletedEvent)

            expected_trade_side = TradeType.BUY if is_buy_order else TradeType.SELL

            # profit calculation
            order_profit_without_fees = None
            order_maker_fee = None
            order_taker_fee = None
            total_order_profit = None

            # self.logger().info(f"order_fills_profits_pairs: {self.order_fills_profits_pairs}")


            matching_order_pairs = [
            order_pair for order_pair in self.order_fills_profits_pairs

            # event example on taker (event) - BuyOrderCompletedEvent
            # {"timestamp": 1733263659.0, "order_id": "HUMBOTBGAUT62864e4b3a55cc561c30c", "base_asset": "GIGA", "quote_asset": "USDT", "base_asset_amount": "150", "quote_asset_amount": "6.0255", "order_type": "OrderType.LIMIT", "exchange_order_id": "C02__492752804511879170097", "event_name": "BuyOrderCompletedEvent", "event_source": "mexc"}

            # event example on maker: (OrderFilledEvent)
            # {"timestamp": 1733263659.0, "order_id": "666947820", "trading_pair": "GIGA-USD", "trade_type": "TradeType.SELL", "order_type": "OrderType.LIMIT", "price": "0.040327", "amount": "150.0", "trade_fee": {"percent": "0", "percent_token": "USD", "flat_fees": [{"token": "USD", "amount": "0.0121"}]}, "exchange_trade_id": "TDHZZ5-Z56HU-JK6MF2", "exchange_order_id": "OB7BVL-PD3JS-VAVKWS", "leverage": 1, "position": "NIL", "event_name": "OrderFilledEvent", "event_source": "kraken_v2"}

            # order_pair['taker_order_id'] is the OrderCandidate type, it's the object sent to create a taker order, this is not an event
            # order_pair['maker_order'] is OrderFilledEvent object type. This is a partially or fully filled order on maker

            # in the condition below we search for the taker candidate that is similar to the current taker fill by side and amount. Additional condition: the corresponding maker OrderFilledEvent (the exact one the taker candidate was based on) timestamp is close to the current taker OrderCompletedEvent event.
            if (order_pair.get('taker_order_id') and 
                order_pair['taker_order_id'].order_side == expected_trade_side and
                order_pair['taker_order_id'].amount <= event.base_asset_amount * Decimal(str(1.005)) and
                order_pair['taker_order_id'].amount >= event.base_asset_amount * Decimal(str(0.995)) and
                (event.timestamp - order_pair['maker_order'].timestamp) <= 2.0
                )
        ]

            if matching_order_pairs:
                # Take the first matching pair
                order_pair = matching_order_pairs[0]
                corresponding_maker_order = order_pair.get('maker_order')
                
                if corresponding_maker_order:
                    # Consistent calculation for both buy and sell orders
                    corresponding_maker_order_filled_size = corresponding_maker_order.amount * corresponding_maker_order.price
                    
                    if is_buy_order:
                        order_profit_without_fees = corresponding_maker_order_filled_size - event.quote_asset_amount
                    else:
                        order_profit_without_fees = event.quote_asset_amount - corresponding_maker_order_filled_size

                    # calculating maker fee by default using the options fee percent setting self.maker_fee
                    order_maker_fee = corresponding_maker_order_filled_size * Decimal(str(self.maker_fee))

                    # Access the trade_fee from the OrderFilledEvent
                    trade_fee = corresponding_maker_order.trade_fee
                   
                    # First, check flat fees
                    if trade_fee.flat_fees:
                        # Take the first flat fee (most common scenario)
                        flat_fee = trade_fee.flat_fees[0]
                        
                        if flat_fee.token == self.maker_quote_symbol:
                            order_maker_fee = flat_fee.amount
                            # self.logger().notify(f"Taking a flat fee from the maker order event: {order_maker_fee}")                       

                    # Calculate fees consistently for both buy and sell orders

                    order_taker_fee = event.quote_asset_amount * Decimal(str(self.taker_fee))
                    
                    total_order_profit = order_profit_without_fees - order_maker_fee - order_taker_fee
                    total_order_profit_excluding_maker_fee = order_profit_without_fees - order_taker_fee

                    # dictionary structure:
                    # self.accumulated_profits = {
                    #     'total_order_profit': Decimal("0"),
                    #     'total_order_profit_excluding_maker_fee': Decimal("0"),
                    #     'total_maker_fees': Decimal("0"),
                    #     'total_taker_fees': Decimal("0"),
                    # }                    

                    self.accumulated_profits['total_order_profit'] += total_order_profit
                    self.accumulated_profits['total_order_profit_excluding_maker_fee'] += total_order_profit_excluding_maker_fee

                    self.accumulated_profits['total_maker_fees'] += order_maker_fee
                    self.accumulated_profits['total_taker_fees'] += order_taker_fee

                    # Remove the processed order pair from the list
                    self.order_fills_profits_pairs.remove(order_pair)
                    self.logger().info(f"order_fills_profits_pairs after removing the matched one: {self.order_fills_profits_pairs}")

            order_message = self.format_filled_order_message(
                filled_order.client_order_id, 
                event.base_asset_amount, 
                None, 
                event.quote_asset, 
                event.base_asset,
                size = event.quote_asset_amount, 
                is_buy_order=is_buy_order, 
                is_maker_exchange_order=False,
                profit_dict={'order_profit_without_fees': order_profit_without_fees,
                             'order_maker_fee': order_maker_fee,
                             'order_taker_fee': order_taker_fee,
                             'total_order_profit': total_order_profit,
                             'maker_order_id': corresponding_maker_order.order_id,
                             }
            )
            
            # self.logger().info(order_message['log_message'])
            self.telegram_utils.send_unformatted_message(order_message['telegram_message'])        

    def check_rate_count(self):
        """
        Checks the rate count of the maker connector Kraken and determines if it exceeds the allowed limit. 
        If the rate count is too high, the bot will pause for a specified delay. This function is 
        responsible for controlling the rate of order submissions by evaluating the connector's 
        rate count and its timestamp. If the rate count exceeds the 
        maximum limit, it triggers actions such as pausing and canceling orders.
        Since ratecount is ONLY updated asynchronously when the order is created 
        or filled or cancelled (and it's not possible to retreive it with POST API), 
        the ratecount timestamp is used to get out of the idle mode.
        """
        if not hasattr(self.connectors[self.maker], 'rate_count'):
            return
        if self.any_of_timers_is_active():
            return
        
        rate_count = self.connectors[self.maker].rate_count
        # transform in milliseconds
        rate_count_timestamp = self.connectors[self.maker].rate_count_update_timestamp
        
        if rate_count_timestamp is None:
            return        
        rate_count_timestamp = int(rate_count_timestamp) * 1000
                
        # if self.previously_processed_rate_count_update_timestamp is not None:
        #     # if the current 'rate_count' value was processed before
        #     if rate_count_timestamp <= self.previously_processed_rate_count_update_timestamp:
        #         return
        delay = self.after_rate_conter_limit_reached_delay

        # debug_string = (f"Evaluating: ({rate_count_timestamp} + {delay} - 2000) "
        #         f"< ({self.connectors[self.maker].current_timestamp} * 1000 = {self.connectors[self.maker].current_timestamp * 1000})")
    
        # self.logger().info(debug_string)
        if rate_count_timestamp + delay - 2000 < self.connectors[self.maker].current_timestamp * 1000:
            # self.logger().info(f"The latest rate_count_timestamp is too old, skip checking...")
            return
        
        if rate_count > self.rate_count_max_limit:
                        # previous_idle_timer_state = self.idle_timer.timer_is_active
            
            self.idle_timers.append(Timer(name="rate_count_idle_timer", duration=delay))
            
            self.previously_processed_rate_count_update_timestamp = rate_count_timestamp
            self.custom_cancel_all_orders(debug_output=True)
            
            message = f"Rate Count Max limit exceeded ({rate_count} > {self.rate_count_max_limit}). Pausing for {delay / 1000:.3g} seconds."
            # \n Rate_count_timestamp: {self.connectors[self.maker].rate_count_update_timestamp}\n"
            self.logger().info(message)
            if self.notify_about_rate_limit_exceeding:
                self.telegram_utils.send_unformatted_message(message)

    def check_and_correct_total_base_balance(self, notify=True, fix_balance=False, debug_output=False):
        if not self.base_balance_checker.check_if_the_balance_changed(self.starting_base_total, self.base_total, meaningful_difference_in_base_symbol=self.total_base_change_notification_limit):
            return
        
        # the latest change from previous to current values
        base_amount_difference_from_previous_change_till_now = Decimal(str(self.base_total)) - Decimal(str(self.base_balance_checker.total_balance_before_the_latest_detected_change))

        # the difference between the start and current values
        base_amount_difference_from_start_till_now = Decimal(str(self.base_total)) - Decimal(str(self.starting_base_total))

        # notify about the changes bigger than min taker amount increment, do not notify
        # about the 'dust' changes and changes related to float inaccuraces
        if (notify and 
            abs(base_amount_difference_from_previous_change_till_now) >= self.taker_rules.min_base_amount_increment and
            abs(base_amount_difference_from_start_till_now) >= self.taker_rules.min_base_amount_increment
            ):
            message = f"The total base balance changed for more than {self.total_base_change_notification_limit} {self.maker_base_symbol}, from {self.base_balance_checker.total_balance_before_the_latest_detected_change} to {self.base_total} {self.maker_base_symbol}\n(diff: {base_amount_difference_from_previous_change_till_now})\nThe start balance was: {self.starting_base_total} {self.maker_base_symbol}\nThe diff. from the start bal.: {base_amount_difference_from_start_till_now}"
            
            self.send_beautiful_message_to_log_and_telegram(message, message)


        
        
        # self.logger().notify(f"min taker order amount: {Decimal(str(self.min_taker_order_amount()))}, condition result {base_amount_difference_from_start_till_now} < {Decimal(str(self.min_taker_order_amount()))} = {base_amount_difference_from_start_till_now < Decimal(str(self.min_taker_order_amount()))}")
        if not self.enable_small_base_balance_corrections:
            return
        # do not try to fix extra small changes, if they are smaller than the min taker amount increment
        if abs(base_amount_difference_from_start_till_now) < self.taker_rules.min_base_amount_increment:
            return

        if abs(base_amount_difference_from_start_till_now) < Decimal(str(self.min_taker_order_amount())):
            # add verification of unfilled maker small orders
            self.pending_small_base_amount_to_fix = base_amount_difference_from_start_till_now            
            
            message = f"Detected small balance change. New fix amount: {self.pending_small_base_amount_to_fix}"
            # self.logger().notify(message)

            self.send_beautiful_message_to_log_and_telegram(message, message)

    def get_order_book_dict(self, exchange: str, trading_pair: str, depth: int = 50):

        self.order_book_bids = None
        self.order_book_asks = None

        self.df_order_book_sell = None
        self.df_order_book_buy = None
   
        order_book = self.connectors[exchange].get_order_book(trading_pair)
        snapshot = order_book.snapshot

        self.order_book_bids = snapshot[0].loc[:(depth - 1), ["price", "amount"]].values.tolist()
        self.order_book_asks = snapshot[1].loc[:(depth - 1), ["price", "amount"]].values.tolist()

        self.df_order_book_sell = pd.DataFrame(self.order_book_asks)
        self.df_order_book_sell["cumsum"] = self.df_order_book_sell[1].cumsum()

        self.df_order_book_buy = pd.DataFrame(self.order_book_bids)
        self.df_order_book_buy["cumsum"] = self.df_order_book_buy[1].cumsum()       


    def active_orders_df(self) -> pd.DataFrame:
        """
        Returns a custom data frame of all active maker orders for display purposes
        """
        columns = ["Exchange", "Market", "Side", "Price", "Amount", "Spread Mid", "Spread Cancel", "Age"]
        data = []

        active_orders = [order for order in [self.in_flight_sell_order_limit, self.in_flight_buy_order_limit] if order is not None]


        for order in active_orders:
            spread_mid_bps = 0
            spread_cancel_bps = 0
            age_txt = 0
            data.append([
                self.maker,
                order.trading_pair,
                "buy" if order.is_buy else "sell",
                float(order.price),
                float(order.quantity),
                int(spread_mid_bps),
                int(spread_cancel_bps),
                age_txt
            ])
        if not data:
            raise ValueError
        df = pd.DataFrame(data=data, columns=columns)
        df.sort_values(by=["Market", "Side"], inplace=True)
        return df

    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        ord_size = Decimal(1)
        # min_quantum = self.connectors[self.maker].get_order_size_quantum(self.maker_pair, ord_size)
        min_quantum_2 = self.connectors[self.maker].say_hello()
        rate_count = self.connectors[self.maker].rate_count
        min_quantum = self.connectors[self.maker].supported_order_types()
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        base_precision_for_output = self.balances_data_dict.get("base_precision_for_output")
        quote_precision_for_output = self.balances_data_dict.get("quote_precision_for_output")

        if base_precision_for_output is not None and quote_precision_for_output is not None:

            line_total_base = self.telegram_utils.balance_string("", self.maker_base_symbol, self.base_total, base_precision_for_output)

            line_total_quote = self.telegram_utils.balance_string('total Now', self.maker_quote_symbol, self.quote_total, quote_precision_for_output)

            line_total_base_start = self.telegram_utils.balance_string("", self.maker_base_symbol, self.starting_base_total, base_precision_for_output)

            line_total_quote_start = self.telegram_utils.balance_string('at start', self.maker_quote_symbol, self.starting_quote_total, quote_precision_for_output)

            line_total_base_diff = self.telegram_utils.balance_string("", self.maker_base_symbol, (self.base_total - self.starting_base_total), base_precision_for_output)

            line_total_quote_diff = self.telegram_utils.balance_string('Diff', self.maker_quote_symbol, (self.quote_total - self.starting_quote_total), quote_precision_for_output)            


            lines.extend([f"\n{line_total_base.strip()}|    {line_total_quote.strip()}"])
            lines.extend([f"{line_total_base_start.strip()}|    {line_total_quote_start.strip()}"])
            lines.extend([f"{line_total_base_diff.strip()}|    {line_total_quote_diff.strip()}"])


        try:
            orders_df = self.active_orders_df()
            lines.extend(["", "  Active Orders:"] + ["    " + line for line in orders_df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        # lines.extend([f"\n"])
        # lines.extend([f"planned_order_price_buy: {self.planned_order_price_buy}"])
        # lines.extend([f"dust_vol_price_buy: {self.dust_vol_limit_price_buy} - hedge_price_buy: {self.hedge_price_buy}"])
        # lines.extend([f"rate_count: {rate_count}"])
        # lines.extend([f"rate_count_timestamp: {self.connectors[self.maker].rate_count_update_timestamp}"])
        # lines.extend([f"self.maker_base_free: {self.maker_base_free}"])
        # lines.extend([f"self.maker_quote_free: {self.maker_quote_free}"])
        # # lines.extend([f"trading_rules on maker: {self.connectors[self.maker].trading_rules.get(self.maker_pair)}"])
        # # lines.extend([f"say_hello: {min_quantum_2}"])
        # lines.extend([f"self.taker_base_free = {self.taker_base_free}, self.taker_quote_free = {self.taker_quote_free}"])

        # cdef class TradingRule:
        #     def __init__(self,
        #                  trading_pair: str,
        #                  min_order_size: Decimal = s_decimal_0,
        #                  max_order_size: Decimal = s_decimal_max,
        #                  min_price_increment: Decimal = s_decimal_min,
        #                  min_base_amount_increment: Decimal = s_decimal_min,
        #                  min_quote_amount_increment: Decimal = s_decimal_min,
        #                  min_notional_size: Decimal = s_decimal_0,
        #                  min_order_value: Decimal = s_decimal_0,
        #                  max_price_significant_digits: Decimal = s_decimal_max,
        #                  supports_limit_orders: bool = True,
        #                  supports_market_orders: bool = True,
        #                  buy_order_collateral_token: Optional[str] = None,
        #                  sell_order_collateral_token: Optional[str] = None):
        #         self.trading_pair = trading_pair
        #         self.min_order_size = min_order_size
        #         self.max_order_size = max_order_size
        #         self.min_price_increment = min_price_increment
        #         self.min_base_amount_increment = min_base_amount_increment
        #         self.min_quote_amount_increment = min_quote_amount_increment
        #         self.min_notional_size = min_notional_size
        #         self.min_order_value = min_order_value
        #         self.max_price_significant_digits = max_price_significant_digits
        #         self.supports_limit_orders = supports_limit_orders
        #         self.supports_market_orders = supports_market_orders
        #         quote_token = split_hb_trading_pair(self.trading_pair)[1]
        #         self.buy_order_collateral_token = buy_order_collateral_token or quote_token
        #         self.sell_order_collateral_token = sell_order_collateral_token or quote_token




        maker_rules = self.connectors[self.maker].trading_rules.get(self.maker_pair)
        taker_rules = self.connectors[self.taker].trading_rules.get(self.taker_pair)

        maker_rules_lines = self.format_rule(maker_rules)
        taker_rules_lines = self.format_rule(taker_rules)

        lines.extend([
            "",
            f"  Rate Counter: {rate_count}",
            "",
            "  Maker Rules:                Taker Rules:"
        ])

        for maker_rule, taker_rule in zip(maker_rules_lines, taker_rules_lines):
            lines.append(f"    {maker_rule:<30} {taker_rule}")

        return "\n".join(lines)
        # return taker_rules.__repr__()    

    def format_rule(self, rule):
        # significant_digits = -int(f"{rule.min_price_increment:e}".split('e')[-1])
        return [
            f"Trading Pair: {rule.trading_pair}",
            f"Min Order Size: {rule.min_order_size}",
            f"Min Notional Size: {rule.min_notional_size}",
            f"Min Price Increment: {rule.min_price_increment}",
            f"Min Base Increment: {rule.min_base_amount_increment}",
            f"Supports Limit Orders: {rule.supports_limit_orders}",
            f"Supports Market Orders: {rule.supports_market_orders}",
        ]

#################################################
    
    def buy(self,
            connector_name: str,
            trading_pair: str,
            amount: Decimal,
            order_type: OrderType,
            price=s_decimal_nan,
            position_action=PositionAction.OPEN) -> str:
        """
        A wrapper function to buy_with_specific_market.

        :param connector_name: The name of the connector
        :param trading_pair: The market trading pair
        :param amount: An order amount in base token value
        :param order_type: The type of the order
        :param price: An order price
        :param position_action: A position action (for perpetual market only)

        :return: The client assigned id for the new order
        """
        market_pair = self._market_trading_pair_tuple(connector_name, trading_pair)
        if order_type in [OrderType.LIMIT, OrderType.LIMIT_MAKER]:
            price = self.connectors[connector_name].quantize_order_price(trading_pair, price)
        quantized_amount = self.connectors[connector_name].quantize_order_amount(trading_pair=trading_pair, amount=amount)
        self.logger().info(f"{connector_name}: Creating {trading_pair} buy order: price: {price} amount: {quantized_amount}.")
        order_result = self.buy_with_specific_market(market_pair, amount, order_type, price, position_action=position_action)
        
        # if not self.connectors[self.maker].real_time_balance_update:
        #     # This is only required for exchanges that do not provide balance update notifications through websocket
        #     self.connectors[self.maker]._in_flight_orders_snapshot = {k: copy.copy(v) for k, v in self.in_flight_orders.items()}
        #     self.connectors[self.maker]._in_flight_orders_snapshot_timestamp = self.current_timestamp
    
        return order_result 
        # return self.connectors[self.maker].buy(trading_pair, quantized_amount, order_type, price, position_action=position_action)

    def sell(self,
             connector_name: str,
             trading_pair: str,
             amount: Decimal,
             order_type: OrderType,
             price=s_decimal_nan,
             position_action=PositionAction.OPEN) -> str:
        """
        A wrapper function to sell_with_specific_market.

        :param connector_name: The name of the connector
        :param trading_pair: The market trading pair
        :param amount: An order amount in base token value
        :param order_type: The type of the order
        :param price: An order price
        :param position_action: A position action (for perpetual market only)

        :return: The client assigned id for the new order
        """
        market_pair = self._market_trading_pair_tuple(connector_name, trading_pair)
        if order_type in [OrderType.LIMIT, OrderType.LIMIT_MAKER]:
            price = self.connectors[connector_name].quantize_order_price(trading_pair, price)
        quantized_amount = self.connectors[connector_name].quantize_order_amount(trading_pair=trading_pair, amount=amount)
        self.logger().info(f"{connector_name}: Creating {trading_pair} sell order: price: {price} amount: {quantized_amount}.")        
        return self.sell_with_specific_market(market_pair, amount, order_type, price, position_action=position_action)
        # return self.connectors[self.maker].sell(trading_pair, amount, order_type, price, position_action=position_action)
    
    def standard_exception_message_sending(self, exception_object, exception_source="somewhere", send_notification=True):
        exc_info = sys.exc_info()
        exc_text = f"Caught an exception <b>'{exception_object}'</b> in <b>{exception_source}.</b>"
        cleaned_string = exc_text.replace("<b>", "").replace("</b>", "")
        self.logger().error(cleaned_string, exc_info=True)
        if send_notification:
            exc_text += "\n" + "".join(traceback.format_exception(*exc_info))           
            self.telegram_utils.send_unformatted_message(exc_text)

    def find_log_file(self, config_name):
        # Construct the log file name pattern based on the config_name variable
        log_file_pattern = rf'^logs_{re.escape(config_name)}\.log$'
        # Get the directory of the current file
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Go one level up
        parent_directory = os.path.dirname(current_directory)

        # Go to the 'logs' directory in the parent
        log_dir = os.path.join(parent_directory, 'logs')
        
        # Find the latest matching log file in the specified directory
        log_files = [f for f in os.listdir(log_dir) if re.match(log_file_pattern, f)]
        latest_file = max(log_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x))) if log_files else None
        return os.path.join(log_dir, latest_file) if latest_file else None

    def extract_latest_full_error_message(self, log_file):
        if not log_file:
            return "Log file not found."

        # Pattern to match the entire error message block, including the traceback
        error_block_pattern = re.compile(r"(Traceback \(most recent call last\):[\s\S]+?(?=\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}|\Z))")
        latest_error_message = None

        with open(log_file, 'r') as f:
            # Read the file from the end using a deque to handle large files efficiently
            lines = deque(f, maxlen=10000)  # Adjust maxlen based on the expected size of error blocks
            log_content = ''.join(lines)
            
            # Search for the error block pattern in the reversed content
            error_blocks = re.findall(error_block_pattern, log_content)

            # Get the first occurrence from the end (latest error block)
            if error_blocks:
                latest_error_message = error_blocks[-1]

        return f"{latest_error_message}" or "No error message with traceback found in the log."

    async def _run_async_update_balances(self, sleep_duration=0):
        """
        Async method to run the balance update
        sleep_duration is in milliseconds
        """
        connector_name = self.taker
        await asyncio.sleep(sleep_duration/1000)
        try:
            # Run with a specific 10-second timeout
            await asyncio.wait_for(
                self.connectors[connector_name]._update_balances(), 
                timeout=10.0
            )
            self.logger().info(f"Connector balances on {connector_name} updated successfully")
        
        except asyncio.TimeoutError:
            self.logger().error(f"Balance update timed out after 10 seconds for {connector_name}")
        
        except Exception as e:
            self.logger().error(f"Error updating balances on {connector_name}: {e}")


    def debug_output_value_for_this_function(self, debug_output: bool, function_name: str=None):
        output_value = debug_output
        if not hasattr(self, 'debug_output_control'):
            return output_value
        if function_name in self.debug_output_control:
            output_value = self.debug_output_control[function_name]
        return output_value

    def send_beautiful_message_to_log_and_telegram(self, message_to_log: str, message_to_telegram=None):
        self.logger().info(message_to_log)
        if message_to_telegram is not None:    
            self.telegram_utils.send_unformatted_message(message_to_telegram)            
                       