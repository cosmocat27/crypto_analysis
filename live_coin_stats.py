# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:22:25 2018

@author: cosmo

Automatically reads in coin stats data from coinmarketcap.com
and updates an excel file
"""

import json, requests, openpyxl


url = "https://api.coinmarketcap.com/v1/ticker/"
response = requests.get(url)
response.raise_for_status()

data_json = json.loads(response.text)

wb = openpyxl.load_workbook("crypto_hist.xlsx")
ws = wb.get_sheet_by_name("current_prices")

for rank, coin in enumerate(data_json):
    ws.cell(row=rank+2, column=1).value = int(coin['rank'])
    ws.cell(row=rank+2, column=2).value = coin['symbol']
    ws.cell(row=rank+2, column=3).value = coin['name']
    ws.cell(row=rank+2, column=4).value = round(float(coin['price_usd']), 2)
    ws.cell(row=rank+2, column=5).value = round(float(coin['market_cap_usd']), 2)
    ws.cell(row=rank+2, column=6).value = round(float(coin['24h_volume_usd']), 2)
    ws.cell(row=rank+2, column=7).value = round(float(coin['percent_change_1h'])/100, 4)
    ws.cell(row=rank+2, column=8).value = round(float(coin['percent_change_24h'])/100, 4)
    ws.cell(row=rank+2, column=9).value = round(float(coin['percent_change_7d'])/100, 4)
    ws.cell(row=rank+2, column=10).value = round(float(coin['available_supply']))
    ws.cell(row=rank+2, column=11).value = round(float(coin['total_supply']))
    ws.cell(row=rank+2, column=12).value = round(float(coin['max_supply'])) \
        if coin['max_supply'] else ""

wb.save("crypto_hist.xlsx")
