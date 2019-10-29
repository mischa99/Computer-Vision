# Blatt 1
#author: Mikhail Raudin, Timm Dill, Max Kubicek, Jan Strich

##Aufgabe 1 ------------
#1
list1 = [0]*5;
list2 = [1]*6;
list3 = [2]*4;

#2 - verbinde 2 Listen
list_all=list1 + list2 + list3

#3 Duplikate rausfiltern mithilfe von Sets
s = set(list_all)

#4
numlist=list(range(1,11)) # erzeugt Liste von 1 bis 10

#5
s2 = set(numlist)

final_s = s.union(s2) #Vereinigung von 2 sets

#Zusatzaufgaben
#6
list_all = list(final_s)
ungerade = list_all[1:11:2]

#7
ungerade2 = list(range(1,11,2))

#8
ungerade2 = ungerade2[1:len(ungerade2)-1]

#9
ungerade = ungerade[-2:-5:-1]

#10
print(2 in ungerade)
print(2 in ungerade2)

##Aufgabe 2----------

l = list(range(1,10))

index = input("Gib einen Index an: ")
index = int(index)
l[index]

help_l = l[index:len(l)] ## [a:b] = von Index a bis Index b-1
help2_l = l[0:index-1]

help_l = help_l[::-1]
help2_l= help2_l[::-1]

final_l = help2_l + help_l
final_l.insert(index-1,index)
print(final_l)

##Aufgabe 3------------

country= dict([("England",1.5) ,("Schweiz",1.2), ("USA",2), ("Norwegen",10),
          ("Polen",3), ("Japan",10)]);
country_and_currency = dict([("England", "Pfund") ,("Schweiz","Schweizer Franken"), 
                            ("USA", "$"), ("Norwegen","NOK"),
          ("Polen","Zloty"), ("Japan","Yen")]);

current_country = input ("Ein Land bitte: ");
euro = input("Wie viel Euro hast du? ");
euro=int(euro)

new_currency = euro* country[current_country]

print(str(new_currency) + " " + country_and_currency[current_country])
          

