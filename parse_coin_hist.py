# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:34:20 2018

@author: cosmo
"""

import requests, openpyxl, sys


def parse_hist(url = None, filename = None, coin_name = "new coin"):
    if url is None and filename is None:
        return "missing source!"
    
    if url is not None and filename is not None:
        return "choose url or filename!"
    
    if url is not None:
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.split('\n')
    
    if filename is not None:
        with open(filename) as source:
            lines = source.readlines()
    
    wb = openpyxl.load_workbook("crypto_hist.xlsx")
    if coin_name not in wb.get_sheet_names():
        wb.create_sheet(index=len(wb.get_sheet_names()), title=coin_name)
    ws = wb.get_sheet_by_name(coin_name)
    
    start_read = 0
    curr_row = 0
    curr_col = 0
    
    for line in lines:
        if "<table" in line:
            start_read = 1
            #print("found table")
        if start_read == 0:
            continue
        if "<tr" in line:
            curr_row += 1
            #print("new row")
        if ("<th" in line and "<thead" not in line):
            curr_col += 1
            pos0 = line.find("<th")
            pos1 = line[pos0:].find(">") + pos0
            pos2 = line[pos1:].find("<") + pos1
            ws.cell(row=curr_row, column=curr_col).value = line[pos1+1:pos2]
            #print(pos0, pos1, pos2)
            #print(line, line[pos1+1:pos2])
        if "<td" in line:
            curr_col += 1
            pos0 = line.find("<td")
            pos1 = line[pos0:].find(">") + pos0
            pos2 = line[pos1:].find("<") + pos1
            ws.cell(row=curr_row, column=curr_col).value = line[pos1+1:pos2]
            #print(pos0, pos1, pos2)
            #print(line, line[pos1+1:pos2])
        if "</tr" in line:
            curr_col = 0
            #print("end row")
    
    wb.save("crypto_hist.xlsx")
    return True


if __name__ == '__main__':
    
    url = None
    filename = None
    coin_name = "new coin"
    url_list = 1
    
    if len(sys.argv) > 1:
        if sys.argv[1].find("http") >= 0:
            url = sys.argv[1]
            url_list = 0
        else:
            filename = sys.argv[1]
    if len(sys.argv) > 2:
        coin_name = sys.argv[2]
        url_list = 0
    
    if url_list == 1:
        with open(filename) as source:
            lines = source.readlines()
            for line in lines:
                coin_name, url = line.split(" ")
                print(coin_name, parse_hist(url, None, coin_name))
    else:
        print(coin_name, parse_hist(url, filename, coin_name))
