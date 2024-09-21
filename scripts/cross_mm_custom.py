import math
import sys
import time
from decimal import Decimal
from typing import List

import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import (
    BuyOrderCreatedEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderType,
    PositionAction,
    SellOrderCreatedEvent,
)
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

# from hummingbot.core.data_type.order_book_tracker import OrderBookTracker
# from hummingbot.core.data_type.order_book import OrderBook

# from hummingbot.core.utils.trading_pair import TradingPair
# from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
# from hummingbot.connector.connector_base import ConnectorBase

s_decimal_nan = Decimal("NaN")

class CrossMmCustom(ScriptStrategyBase):

### OPTIONS ###
    # both, buy, sell
    flow_mode = "buy" #"sell" #"both"

    maker = 'kraken_v2' # 'kraken_paper_trade'
    taker = 'mexc'  #'gate_io_paper_trade'

    # maker = 'kraken_v2_paper_trade' # 'kraken_paper_trade'
    # taker = 'mexc_paper_trade'  #'gate_io_paper_trade'

    maker_pair = "GNO-USD"
    taker_pair = "GNO-USDT"

    #round up to this value in best price calculations from taker asks bids
    order_price_precision = 2
    order_base_precision = 2

    #in milliseconds, delay after which to check for unprocessed maker orders
    trade_timestamp_delay = 10

    maker_fee = 0.0025
    taker_fee = 0.0002 #0.001 #0.00084

    ###### PROFIT in fractions, 0.001 = 0.1% #######
    # maker fee can be added here
    min_profit_sell = -0.02 #0.0005
    min_profit_buy = -0.02 #-0.01 #0.001 #0.00085 #was 0.0005 on 06.05.21

    ##### DUST VOLUME ##### in base asset units
    dust_vol_sell = 0.1 
    dust_vol_buy = 0.1
    #####             ##### 

    # in quote currency. This setting can be used to put the order with THE SAME
    # PRICE as the current best order. Just increase the accuracy by 10 times
    # compared to the max accuracy of the base asset on maker exchange
    # For example, if min step is 1e-8, put 1e-9
    order_price_step_sell = 0.2 #9.8e-9 #1e-9
    order_price_step_buy = 0.2

    # this is the max allowed distance up from the dust price
    # if the order price stays within the distance, it's not edited
    # the purpose is to maximise profit
    # if it's too small the order will be edited frequently
    # and the rate limits can be exceeded
    # if it's too high the order may hang at the top of the order book 
    # for longer periods even if the competitors with high volumes
    # remove their orders from the top
    order_price_safe_distance_sell = 0.8
    order_price_safe_distance_buy = 1

    # if we have to put the order close to the hedge price 
    # this is the distance from the hedge price we put it to
    hedge_price_step_sell = 0.03
    hedge_price_step_buy = 0.03

    # if the hedge price is out of the volume window, the order price will
    # be kept within this distance from the hedge price
    hedge_price_safe_distance_sell = 1
    hedge_price_safe_distance_buy = 1

    maker_order_book_depth = 30
    # possible options are:
    # - 'default'
    # - 'keep_tick_level'

    # strategy = 'keep_tick_level'
    strategy = 'default'

    # at what level to keep the order at maker, starting from 1
    keep_tick_number_buy = 2
    keep_tick_number_sell = 2


    # in adaptive mode the below values are used as MAXIMAL order size (Base asset)
    amount_sell = 0.1 #0.00025 #0.0003 # 0.00105 #Base asset
    amount_buy = 0.1 #0.00026 #0.00028  # 0.00105 #Base asset

    #minimal order amount nominated in quote asset
    min_notional_maker = 5
    min_notional_taker = 5

    #Adaptive order size mode. If enabled, the order amount depends on the wallet available aseets
    advaptive_order_amount_mode = 1 # 1 - enabled, 0 - disabled
    # adaptive order amount in fraction of the max available asset volume in the wallet
    adaptive_amount_sell_fraction = 1
    adaptive_amount_buy_fraction = 1

    low_balance_duration = 1800

    forget_time = 3600

    base_precision_for_output = 1
    quote_precision_for_output = 2

    #if exchange supports edit order we can use this feature. Should be 0 or 1
    edit_order_mode = 1

    # how many levels to fetch. For default value just comment this line
    taker_order_book_depth = 10 # can be 20 or 100 for kucoin
    # in qoute currency, order size for the correct profits calculation
    taker_volume_depth_for_best_ask_bid = 2800

    # key is the number of seconds for a measuring timeframe, value is the limit in % per min
    price_speed_limits = {'previous':6.5, 4:4.9, 8:3.1, 16:2.16, 32:1.25}
    #price_speed_limits = {'previous':1.0, 4:1.6, 8:1.8, 16:0.7, 32:0.1}

    #stop_trading_mode_active_timeframe = 30000
    # arbitrage is an option to place an order if there's a profitable limit order
    # on the opposite side of the order book
    # if you don't wish to activate it just put some big numbers to profits or min sizes
    arbitrage_profit_buy = 0.01#0.002
    arbitrage_profit_sell = 0.01#0.002
    arbitrage_min_size_buy = 50
    arbitrage_min_size_sell = 50

    # "cross_margin", "isolated_margin", or "spot". The default is "spot"
    taker_trading_mode = "spot"

    # what part of the taker asset to take to limit the order size in adaptive mode
    # not mandatory. The default value is set in the main code
    taker_max_order_size_fraction = 1

    # what amount of asset to keep untouched on the exchange, expressed in QUOTE currency
    # it means we keep both base and quote assets not less than the amount set here
    # It's important to keep 1.01*max_order_size on taker if the MARKET BUY order is 
    # not possible on taker with the amount only
    # so taker_min_balance_quote = amount_sell*price*0.01
    taker_min_balance_quote = 0
    maker_min_balance_quote = 0

    # what part of max borrowable base amount to take for limiting max order amount
    max_borrowable_initial_part = 1
    # how much is allowed to borrow total (during the whole trading session) in base units
    # this is a hard upper borrowing limit. Optional.
    # max_borrowable_absolule_base_limit = 1

    total_base_change_notification_limit = 0.01

    # stop trading speed max limit for previous time value, %/millisec
    previous_speed_limit = 0.17 #0.001717 # = 10.3 %/min

    # min delay before any create or arb order after any create or arb order, ms
    min_delay_before_create_order = 2000

    # how long (milliseconds) to wait for the balances from redis in maker_part
    maker_part_startup_delay = 25*1000

    # some exchanges return trades timestamps with SECONDS accuracy, not ms
    # in order to address the issue please set this variable to 1000 if the accuracy is SECONDS
    # the default value is 1, that is milliseconds accuracy
    # acceptable values are 1 or 1000
    taker_exchange_trades_timestamp_accuracy = 1000

    # If you need to calculate the profits for a given period of time
    # They will be displayed after the bot termination
    # after the session profits
    # if the calculation is not needed just comment the following two parameters
    # the datetime format is "2024-03-01 00:00:00"
    # the word "now" can be used for current time
    # the time is in UTC usually

    # profit_calculation_start_date = "2024-03-24 00:00:00"
    # profit_calculation_end_date = "now"

    # order_amount = 0.1                  # amount for each order
    # spread_bps = 10                     # bot places maker orders at this spread to taker price
    # min_spread_bps = 0                  # bot refreshes order if spread is lower than min-spread
    # slippage_buffer_spread_bps = 100    # buffer applied to limit taker hedging trades on taker exchange
    # max_order_age = 120                 # bot refreshes orders after this age

    markets = {maker: {maker_pair}, taker: {taker_pair}}

    price_source = PriceType.MidPrice

    sell_profit_coef = (1 + taker_fee)/(1 - maker_fee - min_profit_sell)
    buy_profit_coef = (1 - taker_fee)/(1 + maker_fee + min_profit_buy)

    # for placing the taker limit order to make sure it will be completed
    # as a market order.
    taker_best_ask_price_coef = 1.01
    taker_best_bid_price_coef = 0.99
    # buy_order_placed = False
    # sell_order_placed = False
    one_order_only = True
    exit_bot_flag = False
    start_time = None
    idle_mode = False
    one_time_message_flag = False
    
    def on_tick(self):
        
        # self.logger().info("TICK STARTED!!!!!!!!!!!!!!!!!!!!!!!!!!")             
        # self.logger().info(self.connectors[self.taker].trading_rules)
        self.check_active_orders(debug_output=False) 

        # self.exit_after_some_time()

        # if self.exit_bot_flag and self.one_order_only:
        #     self.custom_cancel_all_orders()            
        #     self.logger().info("Stopping the bot due to One Order Only flag.")
        #     raise RuntimeError("Stopping the script due to a non-critical error.")
        
        if self.exit_bot_flag:
            
            self.custom_cancel_all_orders()
            if not self.one_time_message_flag: 
                self.logger().info("Skipping all on_tick routines due to One Order Only flag.")
                self.one_time_message_flag = True
            return         

        self.custom_init()

        self.calculate_planned_orders_sizes(debug_output=False)

        self.calculate_hedge_price(debug_output=False)       

        self.adjust_orders_sizes(debug_output=False) 

        self.calc_orders_parameters(debug_output=False)

        self.buy_order_flow()

        self.sell_order_flow()

        return

    def exit_after_some_time(self, time_period=7):
        if self.start_time is None:
            self.start_time = time.time()
            self.logger().info("Script started. Timer initiated.")
        # Check if 20 seconds have passed
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= time_period:
            self.logger().info(f"{time_period} seconds have passed. Exiting the script.")
            self.custom_cancel_all_orders()
            raise RuntimeError("Stopping the script due to a non-critical error.")
            # sys.exit()

    def custom_cancel_all_orders(self):
        if self.active_buy_order is not None:
            self.custom_cancel_order(order=self.active_buy_order, debug_output=False)
        if self.active_buy_order is not None:    
            self.custom_cancel_order(order=self.active_sell_order, debug_output=False)

    def buy_order_flow(self):
        if self.flow_mode == "sell":
            return
        if self.active_buy_order is None:
            self.create_new_maker_order(side=TradeType.BUY, debug_output=False)
        elif self.edit_order_condition(side=TradeType.BUY, debug_output=False):
            buy_order_edit_result = self.edit_order(order=self.active_buy_order, debug_output=False)
            
            if buy_order_edit_result == "cancel":
                self.custom_cancel_order(order=self.active_buy_order, debug_output=False)
            
            if self.cancel_order_condition(side=TradeType.BUY, debug_output=False):
                self.custom_cancel_order(order=self.active_buy_order, debug_output=False) 

    def sell_order_flow(self):
        if self.flow_mode == "buy":
            return
        if self.active_sell_order is None:
            self.create_new_maker_order(side=TradeType.SELL, debug_output=False)
        elif self.edit_order_condition(side=TradeType.SELL, debug_output=False):
            sell_order_edit_result = self.edit_order(order=self.active_sell_order, debug_output=False)
            
            if sell_order_edit_result == "cancel":
                self.custom_cancel_order(order=self.active_sell_order, debug_output=False)
            
            if self.cancel_order_condition(side=TradeType.SELL, debug_output=False):
                self.custom_cancel_order(order=self.active_sell_order, debug_output=False)

    def custom_init(self):
        # self.maker_base_symbol, self.maker_quote_symbol = self.maker_pair.split("-")
        # self.taker_base_symbol, self.taker_quote_symbol = self.taker_pair.split("-")
        
        self.maker_base_symbol, self.maker_quote_symbol = self.connectors[self.maker].split_trading_pair(self.maker_pair)
        self.taker_base_symbol, self.taker_quote_symbol = self.connectors[self.taker].split_trading_pair(self.taker_pair)

        # # Log the variables using logger().info
        # self.logger().info(f"Maker base symbol: {self.maker_base_symbol}, Maker quote symbol: {self.maker_quote_symbol}")
        # self.logger().info(f"Taker base symbol: {self.taker_base_symbol}, Taker quote symbol: {self.taker_quote_symbol}")           

        # self.connectors[self.maker].update_balances()
        self.maker_base_free = None
        self.maker_base_free = float(self.connectors[self.maker].get_available_balance(self.maker_base_symbol))
        
        self.maker_quote_free = None
        self.maker_quote_free = float(self.connectors[self.maker].get_available_balance(self.maker_quote_symbol))

        self.taker_base_free = None
        self.taker_base_free = float(self.connectors[self.taker].get_available_balance(self.taker_base_symbol))
        
        self.taker_quote_free = None
        self.taker_quote_free = float(self.connectors[self.taker].get_available_balance(self.taker_quote_symbol))
        # float(self.connectors[self.taker].get_balance(self.taker_quote_symbol)) 

        self.maker_base_total = None
        self.maker_base_total = float(self.connectors[self.maker].get_balance(self.maker_base_symbol))
        
        self.maker_quote_total = None
        self.maker_quote_total = float(self.connectors[self.maker].get_balance(self.maker_quote_symbol))

        self.taker_base_total = None
        self.taker_base_total = float(self.connectors[self.taker].get_balance(self.taker_base_symbol))
        
        self.taker_quote_total = None
        self.taker_quote_total = float(self.connectors[self.taker].get_balance(self.taker_quote_symbol))

        if self.maker_quote_total is not None and self.maker_quote_free is not None:
            if self.maker_quote_free < self.maker_quote_total and self.active_buy_order:
                self.maker_quote_free = min((self.maker_quote_free + 
                                         float(self.active_buy_order.price) * float(self.active_buy_order.quantity)),
                                         self.maker_quote_total)

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
            self.logger().info(f"self.hedge_price_sell = {self.taker_buy_by_volume_price} * {self.sell_profit_coef}, taker_buy_result.result_price = {taker_buy_result.result_price}")
                
        self.hedge_price_buy = self.taker_sell_by_volume_price * self.buy_profit_coef

        if not self.float_var_is_valid_and_positive(self.hedge_price_buy, "self.hedge_price_buy"):
            self.do_not_create_sell_order_because_of_bad_parameters = True
            self.do_not_create_buy_order_because_of_bad_parameters = True
        
        if debug_output:
            self.logger().info(f"self.hedge_price_buy = {self.taker_sell_by_volume_price} * {self.buy_profit_coef}, taker_sell_result.result_price = {taker_sell_result.result_price}")
        
        # min order amount in base units
        self.min_notional_maker_amount = self.min_notional_maker / self.hedge_price_buy
        self.min_notional_taker_amount = self.min_notional_taker / self.hedge_price_buy
        self.min_notional_for_maker_order_creation = max(self.min_notional_maker_amount, self.min_notional_taker_amount)

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

    # Here we calculate the exact order sizes taking into consideration all maker and taker balances
    # and the required untouched balances on maker and taker        
    def adjust_orders_sizes(self, debug_output=False):
        maker_untouched_amount = self.maker_min_balance_quote / self.hedge_price_buy
        taker_untouched_amount = self.taker_min_balance_quote / self.hedge_price_buy

        sell_values = [
            (self.maker_base_free * self.adaptive_amount_sell_fraction - maker_untouched_amount), 
            self.amount_sell,
            (self.taker_quote_free - self.taker_min_balance_quote) / self.hedge_price_sell  
        ]
        
        self.order_size_sell = min(sell_values)

        if debug_output:
            self.logger().info(f"sell_values: {sell_values}")
            self.logger().info(f"taker_quote_free: {self.taker_quote_free}, taker_min_balance_quote: {self.taker_min_balance_quote}, hedge_price_sell: {self.hedge_price_sell}")
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

        self.active_limit_orders = self.get_active_orders(connector_name=self.maker)

        # trying to get active orders through exchange entity. 
        # It works but the order tracking and balance functionality fails with this approach
        # self.active_in_flight_orders = self.connectors[self.maker].in_flight_orders
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

        if active_buy_orders:
            self.active_buy_order = active_buy_orders[0]
            if debug_output:
                self.logger().info("There are active buy orders.")
                self.logger().info(f"{self.active_buy_order}")
        else:
            self.active_buy_order = None
            if debug_output:
                self.logger().info("There are no active buy orders.")

        if active_sell_orders:
            self.active_sell_order = active_sell_orders[0]
            if debug_output:
                self.logger().info("There are active sell orders.")
        else:
            self.active_sell_order = None
            if debug_output:
                self.logger().info("There are no active sell orders.")


    def create_new_maker_order(self, side=TradeType.BUY, debug_output=False):
        if side == TradeType.BUY:
            if self.do_not_create_buy_order_because_of_bad_parameters:
                self.logger().error(f"The {side} order is not created because self.do_not_create_buy_order_because_of_bad_parameters is True")
                return
            if not self.order_size_and_price_are_valid(side):
                return
            buy_price = Decimal(self.planned_order_price_buy)

            buy_order = OrderCandidate(trading_pair=self.maker_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=Decimal(self.order_size_buy), price=buy_price)
            # buy_order_adjusted = self.adjust_proposal_to_budget(self.maker, [buy_order])
            if self.check_order_min_size_before_placing(self.maker, buy_order, notif_output=False):
                self.place_order(self.maker, buy_order)
        else:
            if self.do_not_create_sell_order_because_of_bad_parameters:
                self.logger().error(f"The {side} order is not created because self.do_not_create_buy_order_because_of_bad_parameters is True")
                return                        
            if not self.order_size_and_price_are_valid(side):
                return
            sell_price = Decimal(self.planned_order_price_sell)

            sell_order = OrderCandidate(trading_pair=self.maker_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=Decimal(self.order_size_sell), price=sell_price)
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
                self.active_buy_order.price >= self.dust_vol_limit_price_buy and 
                self.active_buy_order.price <= (self.dust_vol_limit_price_buy + self.order_price_safe_distance_buy))

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

            active_order_price_is_between_the_dust_price_and_one_step_above_price = (
                self.active_sell_order.price <= self.dust_vol_limit_price_sell and 
                self.active_sell_order.price >= (self.dust_vol_limit_price_sell - self.order_price_safe_distance_sell))

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

    def custom_cancel_order(self, order, debug_output=False):
        
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
        self.logger().info(f"Catched Order cancel!!!")
        if (self.create_buy_order_after_cancel_in_current_tick_cycle == True 
            and self.buy_order_client_id_to_edit_in_current_tick_cycle == event.order_id):
            self.logger().info(f"Catched Order cancel and creating a new order! Cancelled order id: {self.buy_order_client_id_to_edit_in_current_tick_cycle}")
            self.create_new_maker_order(side=TradeType.BUY)

        if (self.create_sell_order_after_cancel_in_current_tick_cycle == True 
            and self.sell_order_client_id_to_edit_in_current_tick_cycle == event.order_id):
            self.create_new_maker_order(side=TradeType.SELL)



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
        Returns an order object it's an active order or None
        """
        for order in self.get_active_orders(connector_name=self.maker):
            if order.client_order_id == event.order_id:
                return order
        return None
    
    def did_fill_order(self, event: OrderFilledEvent):
        
        filled_order = self.get_order_by_event(event)

        if event.trade_type == TradeType.BUY and (filled_order is not None):
            
            self.logger().info(f"Filled MAKER BUY order {filled_order.client_order_id} for {event.amount * event.price} {filled_order.quote_currency} ({event.amount} {filled_order.base_currency} at {event.price} {filled_order.quote_currency})")            

            if self.one_order_only:
                self.logger().info("One order only! No more orders should be placed.")
                self.exit_bot_flag = True

            taker_sell_result = self.taker_sell_by_volume_price
            
            sell_price_with_slippage = Decimal(taker_sell_result) * Decimal(self.taker_best_bid_price_coef)

            taker_rules = self.connectors[self.taker].trading_rules.get(self.taker_pair)
            if taker_rules:
                taker_price_increment = Decimal(str(taker_rules.min_price_increment))
                if taker_price_increment:
                    sell_price_with_slippage = Decimal(self.connectors[self.taker].quantize_order_price(self.taker_pair, sell_price_with_slippage))    

            taker_sell_order_amount = event.amount

            # check if there's enough base balance on taker
            if event.amount > self.taker_base_free:
                taker_sell_order_amount = self.taker_base_free
                self.logger().info(f"Correcting SELL LIMIT amount on taker to {taker_sell_order_amount} because the quote balance on taker is not enough")


            self.logger().info(f"Sending TAKER SELL order for {taker_sell_order_amount} {filled_order.base_currency} at price: {sell_price_with_slippage} {filled_order.quote_currency}")
            
            sell_order = OrderCandidate(trading_pair=self.taker_pair, is_maker=False, order_type=OrderType.LIMIT, 
                                        order_side=TradeType.SELL, amount=Decimal(taker_sell_order_amount), price=sell_price_with_slippage)
            
            if self.check_order_min_size_before_placing(self.taker, sell_order, notif_output=True):
                try:
                    self.place_order(self.taker, sell_order)
                except Exception as e:
                    self.logger().warning(f"An error of type {type(e).__name__} occurred while placing SELL order: {e}", exc_info=True)
        
        elif event.trade_type == TradeType.SELL and self.is_active_maker_order(event):

            self.logger().info(f"Filled MAKER SELL order {filled_order.client_order_id} for {event.amount * event.price} {filled_order.quote_currency} ({event.amount} {filled_order.base_currency} at {event.price} {filled_order.quote_currency})")

            if self.one_order_only:
                self.logger().info("One order only! No more orders should be placed.")
                self.exit_bot_flag = True

            taker_buy_result = self.taker_buy_by_volume_price
            
            buy_price_with_slippage = Decimal(taker_buy_result) * Decimal(self.taker_best_ask_price_coef)

            taker_rules = self.connectors[self.taker].trading_rules.get(self.taker_pair)
            if taker_rules:
                taker_price_increment = Decimal(str(taker_rules.min_price_increment))
                if taker_price_increment:
                    buy_price_with_slippage = Decimal(self.connectors[self.taker].quantize_order_price(self.taker_pair, buy_price_with_slippage) + taker_price_increment)

            taker_buy_order_amount = event.amount


            # check if there's enough quote balance on taker
            if Decimal(self.taker_quote_free) < Decimal(event.amount) * Decimal(buy_price_with_slippage):
                
                taker_buy_order_amount = (Decimal(self.taker_quote_free - self.min_notional_taker) / Decimal(buy_price_with_slippage)                
                )
                self.logger().info(f"Correcting BUY LIMIT amount on taker to {taker_buy_order_amount} because the quote balance on taker is not enough")


            self.logger().info(f"Sending TAKER BUY order for {taker_buy_order_amount} {filled_order.base_currency} at price: {buy_price_with_slippage} {filled_order.quote_currency}")

            buy_order = OrderCandidate(trading_pair=self.taker_pair, is_maker=False, order_type=OrderType.LIMIT, 
                                        order_side=TradeType.BUY, amount=Decimal(taker_buy_order_amount), price=buy_price_with_slippage)
            
            if self.check_order_min_size_before_placing(self.taker, buy_order, notif_output=True):
                try:
                    self.place_order(self.taker, buy_order)
                except Exception as e:
                    self.logger().warning(f"An error of type {type(e).__name__} occurred while placing BUY order: {e}", exc_info=True)


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


        for order in self.active_limit_orders:
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

        try:
            orders_df = self.active_orders_df()
            lines.extend(["", "  Active Orders:"] + ["    " + line for line in orders_df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        lines.extend([f"\n"])
        lines.extend([f"planned_order_price_buy: {self.planned_order_price_buy}"])
        lines.extend([f"dust_vol_price_buy: {self.dust_vol_limit_price_buy} - hedge_price_buy: {self.hedge_price_buy}"])
        lines.extend([f"rate_count: {rate_count}"])
        lines.extend([f"self.maker_base_free: {self.maker_base_free}"])
        lines.extend([f"self.maker_quote_free: {self.maker_quote_free}"])
        # lines.extend([f"trading_rules on maker: {self.connectors[self.maker].trading_rules.get(self.maker_pair)}"])
        # lines.extend([f"say_hello: {min_quantum_2}"])
        lines.extend([f"self.taker_base_free = {self.taker_base_free}, self.taker_quote_free = {self.taker_quote_free}"])
        maker_rules = self.connectors[self.maker].trading_rules.get(self.maker_pair)
        taker_rules = self.connectors[self.taker].trading_rules.get(self.taker_pair)

        maker_rules_lines = self.format_rule(maker_rules)
        taker_rules_lines = self.format_rule(taker_rules)

        lines.extend([
            "",
            "  Trading Rules:",
            "",
            "  Maker Rules:                Taker Rules:"
        ])

        for maker_rule, taker_rule in zip(maker_rules_lines, taker_rules_lines):
            lines.append(f"    {maker_rule:<30} {taker_rule}")

        return "\n".join(lines)

    def format_rule(self, rule):
        return [
            f"Trading Pair: {rule.trading_pair}",
            f"Min Order Size: {rule.min_order_size}",
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