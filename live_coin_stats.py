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

wb = openpyxl.load_workbook("crypto_wkst.xlsx")
ws = wb.get_sheet_by_name("CoinMCap_data")

for rank, coin in enumerate(data_json):
    ws.cell(row=rank+2, column=1).value = int(coin['rank'])
    ws.cell(row=rank+2, column=2).value = coin['symbol']
    ws.cell(row=rank+2, column=3).value = coin['name']
    ws.cell(row=rank+2, column=4).value = coin['price_usd']
    ws.cell(row=rank+2, column=5).value = coin['market_cap_usd']
    ws.cell(row=rank+2, column=6).value = coin['24h_volume_usd']
    ws.cell(row=rank+2, column=7).value = coin['percent_change_1h']
    ws.cell(row=rank+2, column=8).value = coin['percent_change_24h']
    ws.cell(row=rank+2, column=9).value = coin['percent_change_7d']
    ws.cell(row=rank+2, column=10).value = coin['available_supply']
    ws.cell(row=rank+2, column=11).value = coin['total_supply']
    ws.cell(row=rank+2, column=12).value = coin['max_supply']

wb.save("crypto_wkst.xlsx")
