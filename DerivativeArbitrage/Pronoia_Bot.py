import ccxt
from strategies import *
from ftx_portfolio import *
import matplotlib.pyplot as plt
import dataframe_image as dfi

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Pronoia_Bot to reply to Telegram messages with all ftx basis

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('coucou')

def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('example requests:')
    update.message.reply_text('* risk [echange] [subaccount]-> live risk')
    update.message.reply_text('* plex [echange] [subaccount]-> compute plex')
    update.message.reply_text('* hist BTC [7] [15m] [ftx]-> history of BTC and related futures/borrow every 15m for past 7d')
    update.message.reply_text('* basis [future] [10000] [ftx] -> futures basis on ftx in size 10000')
    update.message.reply_text('* SysPerp [holding period] [signal horizon]: optimal perps')
    update.message.reply_text('* execreport: live portoflio vs target')

def echo(update, context):
    try:
        split_message = update.effective_message.text.lower().split()
        whitelist = ['daviidarr']
        if not update.effective_message.chat['first_name'] in whitelist:
            update.message.reply_text("https://www.voltz.xyz/litepaper")
            update.message.reply_text("Hey " + update.effective_message.chat['first_name'] + ": my code is so slow you have time to read that")
            log=pd.DataFrame({'first_name':[update.effective_message.chat['first_name']],
                              'date':[str(update.effective_message['date'])],
                              'message':[update.effective_message['text']]})
            outputit(log,'','Runtime/chatlog',params={'pickleit':True,'excelit':False})
            excelit("Runtime/chatlog.pickle","Runtime/chathistory.xlsx")

        if split_message[0]=='risk':
            if update.effective_message.chat['first_name'] in whitelist:
                risk = ftx_portoflio_main(*split_message)
                dfi.export(risk[risk.columns[:3]], 'Runtime/dataframe.png')
                update.message.bot.send_photo(update.message['chat']['id'],photo=open('Runtime/dataframe.png', 'rb'))
            else:
                update.message.reply_text("mind your own book")
        elif split_message[0]=='plex':
            if update.effective_message.chat['first_name'] in whitelist:
                plex = ftx_portoflio_main(*split_message)
                dfi.export(plex, 'Runtime/dataframe.png')
                update.message.bot.send_photo(update.message['chat']['id'],photo=open('Runtime/dataframe.png', 'rb'))
            else:
                update.message.reply_text("mind your own book")
        elif split_message[0] == 'execreport':
            diff = ftx_portoflio_main(*split_message)
            dfi.export(diff, 'Runtime/dataframe.png')
            update.message.bot.send_photo(update.message['chat']['id'], photo=open('Runtime/dataframe.png', 'rb'))

        elif split_message[0]=='sysperp':
            data = strategies_main(*split_message)
            dfi.export(data, 'Runtime/dataframe.png')
            update.message.bot.send_photo(update.message['chat']['id'], photo=open('Runtime/dataframe.png', 'rb'))

        elif split_message[0]=='hist':
            if len(split_message)<2:
                raise Exception('missing underlying')
            underlying = split_message[1].upper()
            days = 7                if len(split_message)<3 else int(split_message[2])
            timeframe ='1h'         if len(split_message)<4 else split_message[3]
            exchange_name = 'ftx'   if len(split_message)<5 else split_message[4]
            update.message.reply_text('ok so history of ' + str(underlying) + ' for ' + str(days) + ' days every ' + str(timeframe) + ' on '+ exchange_name)

            exchange = open_exchange(exchange_name)
            futures = pd.DataFrame(fetch_futures(exchange))
            futures = futures[futures['underlying'] == underlying]
            data = build_history(futures, exchange,
                        end=datetime.now(),start=datetime.now()-timedelta(days=days),
                        timeframe=timeframe,
                        dirname='')

            ### send xls
            if update.effective_message.chat['first_name'] in whitelist:
                filename="Runtime/temporary_parquets/telegram_file.xlsx"
                data.to_excel(filename)
                with open(filename, "rb") as file:
                    update.message.bot.sendDocument(update.message['chat']['id'],document=file)

            ### display graphs
            fig, ax = plt.subplots( nrows=3, ncols=1 )
            plt.close(fig)
            spot_mask=data.columns.get_level_values(5)=='spot'
            data.loc[data.index,spot_mask].plot(ax=ax[0])
            ax[0].legend([data.columns[spot_mask].get_level_values(4)[0]])

            future_mask=(data.columns.get_level_values(5)=='perpetual')|(data.columns.get_level_values(5)=='future')
            data.loc[data.index,future_mask].plot(ax=ax[1])
            ax[1].legend(list(data.columns[future_mask].get_level_values(4)))

            borrow_mask=data.columns.get_level_values(5)=='borrow'
            data.loc[data.index,borrow_mask].plot(ax=ax[2])
            ax[2].legend(list(data.columns[borrow_mask].get_level_values(4)))

            fig.savefig('Runtime/hist.png')
            update.message.bot.send_photo(update.message['chat']['id'], photo=open('Runtime/hist.png', 'rb'))
        elif split_message[0]=='basis':
            type='future' if len(split_message)<2 else str(split_message[1])
            depth=1000 if len(split_message)<3 else int(split_message[2])
            exchange_name = 'ftx' if len(split_message) < 4 else split_message[3]
            update.message.reply_text('ok so basis for ' + type + ' for size ' + str(depth))

            exchange = open_exchange(exchange_name,'')
            futures = pd.DataFrame(fetch_futures(exchange))

            data = enricher(exchange, futures, timedelta(weeks=1), depth,
                     slippage_override=-999, slippage_orderbook_depth=depth,
                     slippage_scaler=1.0,
                     params={'override_slippage': False, 'type_allowed': [type], 'fee_mode': 'retail'})

            outputit(data,'basis',exchange.describe()['id'],{'excelit':True,'pickleit':False})

            ### send xls
            if True:# update.effective_message.chat['first_name'] == 'daviidarr':
                with open("Runtime/temporary_parquets/" + exchange.describe()['id'] + "basis.xlsx", "rb") as file:
                    update.message.bot.sendDocument(update.message['chat']['id'],document=file)

        else:
            help(update,context)

    except Exception as e:
        update.message.reply_text(str(e))

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater('1752990518:AAF0NpZBMgBzRTSfoaDDk69Zr5AdtoKtWGk', use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    main()