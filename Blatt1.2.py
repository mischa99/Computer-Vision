
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:11:39 2019
Aufgabenblatt 1.2
@author: Mikhail Raudin, Timm Dill, Max Kubicek, Jan Strich
"""

#Aufgabe 1----------------------------
num = int(input("Gib mir eine Zahl: "))
numlist = list(range(1,num+1,2)) #Liste mit ungeraden Zahlen
#berechne Fakultät
sum=1
for t in numlist:
    sum = sum * t
print(sum)

#Aufgabe 2------------------------------
werte = {"a": 1, "b": 3, "c": 4, "d": 1, "e": 1, "f": 4, 
             "g": 2,"h": 2, "i": 1, "j": 6, "k": 4, "l": 2, 
             "m": 3, "n": 1, "o":2, "p": 4, "q": 10, 
             "r": 1, "s": 1, "t": 1, "u": 1, "v": 6, 
             "w": 3, "x": 8, "y": 10, "z": 3, "ä": 6,
             "ö": 8, "ü": 6}
'''
berechnet scrabble Wert von word mit Nutzung des Dictionaries Werte
'''
def scrabble(word):
    x = 0
    for t in word:
        x+=werte[t]
    
    return x
  
print(scrabble("informatikum"))
    

 #Aufgabe 3-------------------------------
num1 = float(input("Eine Zahl bitte: "))
num2 = float(input("Eine Zahl bitte: "))
num3 = float(input("Eine Zahl bitte: "))
num4 = float(input("Eine Zahl bitte: "))
num5 = float(input("Eine Zahl bitte: ")) 

def summary(num1, num2, num3, num4, num5):
    numlist=[num1, num2, num3, num4, num5]
    print(numlist)
    print(min(numlist) , numlist.index(min(numlist)))
    print(max(numlist) , numlist.index(max(numlist)))
    numlist.sort()
    print(numlist[2]) #Median sortierter Liste mit 5 Elementen am Index 1!
    ug = 0
    g = 0
    for t in numlist:
        if (t%2==0):
            g=g+1
        else:
            ug=ug+1
    print("ungerade " + str(ug))
    print("gerade "+ str(g))
    
    s = set(numlist)
    print("verschieden " + str(len(s)))
    
    ganzeZahl=0
    keineGanzeZahl=0
    for t in numlist:
        if t.is_integer()==1:
            ganzeZahl+=1;
        else:
            keineGanzeZahl+=1
            
    print("ganze Zahlen " + str(ganzeZahl))
    print("reelle Zahlen ohne ganze Zahlen " + str(keineGanzeZahl))
    
    
print(summary(num1, num2, num3, num4, num5))


#Aufgabe 4
price = float(input("Was kostet das Essen: "))  

def bestPaymentOption(price):
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    g=0
    h=0
  
    while(price>0):
        if (price >=200):
            a+=1
            price-=200
            continue
        elif(price>=100):
            b+=1
            price-=100
            continue
        elif(price>=50):
            c+=1
            price-=50
            continue
        elif(price>=20):
            d+=1
            price-=20
            continue
        elif(price>=10):
            e+=1
            price-=10
            continue
        elif(price>=5):
            f+=1
            price-=5
            continue
        elif(price>=2):
            g+=1
            price-=2
            continue
        elif(price>=1):
            h+=1
            price-=1
            continue
        
    coins = {"200" : a, "100":b, "50":c, "20":d,
             "10": e, "5":f, "2":g, "1":h}
            
    for item in coins:
        print("{}: {}".format(item,coins[item])) #'{}' is placeholder for the arguments of format function
        
print(bestPaymentOption(price)) 
    