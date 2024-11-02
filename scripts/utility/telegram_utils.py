from datetime import datetime

from hummingbot.client.hummingbot_application import HummingbotApplication


class TelegramUtils():
    def __init__(self, maker: str, taker: str, maker_pair: str, taker_pair: str):
        self.header_string = (f"<b>{maker_pair}</b> at <b>{maker.capitalize()}</b> -> "
                              f"<b>{taker.capitalize()}</b> ({taker_pair})")
        self.hb = HummingbotApplication.main_application()

    def format_string(self, string: str, include_time: bool = False):
        output = f"{self.header_string}\n\n{string}"
        if include_time:
            current_timestamp = datetime.now()
            # Format time with milliseconds
            formatted_time = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            output += f"\n\n({formatted_time})"  # Use f-string for better readability
        
        return output
    
    def send_unformatted_message(self, message, include_time: bool = True):
        message = self.format_string(message, include_time)
        self.hb.notify(message)


    def start_balance_data_text(self, data_dict: dict): 
        self.base_total = data_dict['base_total']
        self.base_maker_total = data_dict['base_maker_total']
        self.base_taker_total = data_dict['base_taker_total']
        self.quote_total = data_dict['quote_total']
        self.quote_maker_total = data_dict['quote_maker_total']
        self.quote_taker_total = data_dict['quote_taker_total']
        self.maker_name = data_dict['maker_name']
        self.taker_name = data_dict['taker_name']
        self.maker_base_symbol = data_dict['maker_base_symbol']
        self.maker_quote_symbol = data_dict['maker_quote_symbol']
        self.taker_base_symbol = data_dict['taker_base_symbol']
        self.taker_quote_symbol = data_dict['taker_quote_symbol']
        self.base_precision_for_output = data_dict['base_precision_for_output']
        self.quote_precision_for_output = data_dict['quote_precision_for_output']

        # print(f"maker_base_symbol: {self.maker_base_symbol}")

        self.symbol_column_width = max([len(self.maker_base_symbol), len(self.taker_base_symbol), len(self.maker_quote_symbol), len(self.taker_quote_symbol)])

        # print(f"maker_base_symbol: {self.maker_base_symbol}")
        
        self.unified_column_width = max([
                                    len(self.format_base(self.base_total)), 
                                    len(self.format_base(self.base_maker_total)), 
                                    len(self.format_base(self.base_taker_total)),
                                    len(self.format_quote(self.quote_total)), 
                                    len(self.format_quote(self.quote_maker_total)), 
                                    len(self.format_quote(self.quote_taker_total)),
                                    ]) + 1
        self.name_column_width = max([len("Total"), len(self.maker_name), len(self.taker_name)])

        clean_string_0 = "Starting Balances:\n"

        # Base balances
        bold_string = self.balance_string("total", self.maker_base_symbol, self.base_total, self.base_precision_for_output)
        clean_string_0 += f"<b>{bold_string}</b>"

        clean_string_0 += self.balance_string(self.maker_name, self.maker_base_symbol, self.base_maker_total, self.base_precision_for_output)
        
        clean_string_0 += self.balance_string(self.taker_name, self.taker_base_symbol, self.base_taker_total, self.base_precision_for_output)        
        
        # Quote balances
        clean_string_0 += f"\n<b>{self.balance_string('total', self.maker_quote_symbol, self.quote_total, self.quote_precision_for_output)}</b>"

        clean_string_0 += self.balance_string(self.maker_name, self.maker_quote_symbol, self.quote_maker_total, self.quote_precision_for_output)
        
        clean_string_0 += self.balance_string(self.taker_name, self.taker_quote_symbol, self.quote_taker_total, self.quote_precision_for_output) 
        
        output_string = clean_string_0
        return self.format_string(output_string)
    
    def balance_string(self, exchange_name, symbol, amount, precision):
        # print(f"exchange_name: {exchange_name} (type: {type(exchange_name)})")
        # print(f"symbol: {symbol} (type: {type(symbol)})")
        # print(f"amount: {amount} (type: {type(amount)})")
        # print(f"precision: {precision} (type: {type(precision)})")
        # print(f"self.name_column_width: {self.name_column_width} (type: {type(self.name_column_width)})")
        # print(f"self.symbol_column_width: {self.symbol_column_width} (type: {type(self.symbol_column_width)})")
        # output_string = "\n{:<{name_column_width}} | {:>{symbol_column_width}}".format(exchange_name, symbol, 
        #                                                          name_column_width=self.name_column_width,
        #                                                          symbol_column_width=self.symbol_column_width)
        # output_string = f"\n{amount:<{self.unified_column_width},.{precision}f} {symbol:<{self.symbol_column_width}} | {exchange_name.capitalize()}"
        output_string = f"\n{amount:<{self.unified_column_width},.{precision}f} {symbol:<{self.symbol_column_width}} | {exchange_name.capitalize()}"        
        
        return output_string

    def format_base(self, amount):
        formatted_amount = f"{amount:,.{self.base_precision_for_output}f}"
        return formatted_amount

    def format_quote(self, amount):
        formatted_amount = f"{amount:,.{self.quote_precision_for_output}f}"
        return formatted_amount    
    
    def bot_started_string(self, version, strategy, maker_fee, taker_fee, trading_side=""):
        output_string = f"Bot Started. Version: {version}\nStrategy: {strategy}\nTrading Side(s): {trading_side}\nMaker fee: {maker_fee * 100:.{10}g}%, Taker fee: {taker_fee * 100:.{10}g}%"
        return self.format_string(output_string, True)