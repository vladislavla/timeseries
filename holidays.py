# This Python file uses the following encoding: utf-8
"""
Created on Wed Jun 15 21:20:39 2022

@author: Vlada
"""
from datetime import date, timedelta
from dateutil.easter import easter, EASTER_ORTHODOX


def Holidays(year):
    """

    Za izabranu godinu, funkcija vraca listu datuma praznika u Srbiji te godine

    """
    JAN = 1
    FEB = 2
    MAY = 5
    NOV = 11
    WEEKEND = [5, 6]
    SUN = 7
    praznici_datumi = []
    # New Year's Day
    praznici_datumi.append(date(year, JAN, 1))
    praznici_datumi.append(date(year, JAN, 2))
    if date(year, JAN, 1).weekday() in WEEKEND:
        praznici_datumi.append(date(year, JAN, 3))
    # Orthodox Christmas
    praznici_datumi.append(date(year, JAN, 7))
    # International Workers' Day
    praznici_datumi.append(date(year, MAY, 1))
    praznici_datumi.append(date(year, MAY, 2))
    if date(year, MAY, 1).weekday() in WEEKEND:
        if date(year, MAY, 2) == easter(year, method=EASTER_ORTHODOX):
            praznici_datumi.append(date(year, MAY, 4))
        else:
            praznici_datumi.append(date(year, MAY, 3))
    # Armistice day
    praznici_datumi.append(date(year, NOV, 11))
    if date(year, NOV, 11).weekday() == SUN:
        praznici_datumi.append(date(year, NOV, 12))
    # Easter
    praznici_datumi.append(easter(year, method=EASTER_ORTHODOX)
                           - timedelta(days=2))
    praznici_datumi.append(easter(year, method=EASTER_ORTHODOX)
                           - timedelta(days=1))
    praznici_datumi.append(easter(year, method=EASTER_ORTHODOX))
    praznici_datumi.append(easter(year, method=EASTER_ORTHODOX)
                           + timedelta(days=1))
    # Statehood day
    praznici_datumi.append(date(year, FEB, 15))
    praznici_datumi.append(date(year, FEB, 16))
    if date(year, FEB, 15).weekday() in WEEKEND:
        praznici_datumi.append(date(year, FEB, 17))
    return praznici_datumi
