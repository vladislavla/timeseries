Prognoza potrošnje električne energije. 

Ovaj projekat se bavi prognozom potrošnje električne energije 24 h unapred,
koristeći podatke o pređašnjoj potrošnji i o vremenskim podacima za tu lokaciju.

Prikupljeni podaci nalaze se u tri fajla: "EMS_Load.csv", "EMS_Weather_Daily.csv" i  
"EMS_Weather_Hourly.csv" i postoje za period od 2013. do 2018. godine. Podaci
o opterećenju postoje od 15. aprila 2013., dok su podaci o vremenskim prilikama
dostupni od 1. januara 2013., pa je vremenske podatke od 1. januara do 15. aprila
neophodno zanemariti prilikom formiranja dataseta.

Osim toga, postoje vremenski trenuci za koje pojedine promenljive nemaju definisanu
vrednost, pa je izvršena popuna ovih podataka. Pored toga, u potrošnji se javljaju
i 3 outlier-a, 3 nulte potrošnje krajem marta 2014., 2015. i 2016. godine, sve u 2:00 ujutru.
Verovatno su u pitanju planska isključenja mreže od strane nadležnog distributera
električne energije, ali, iako su u pitanju potencijalno podaci iz realnog pogona,
ova 3 nulta podatka su interpolirana kako bi bolje koristila modelu.

Podaci o opterećenju sadržali su i jedan ceo dan nedostajućih podataka, pa su 
oni popunjeni srednjom vrednošću za svaki čas pojedinačno tokom tog meseca.

Zbog velikog broja nedostajućih podataka (preko 65% ukupnih podataka, ili više od 30000),
kao i upitne korelacije sa potrošnjom električne energije, podaci o oblačnosti
su potpuno izuzeti iz modela. Pored toga, nedostaju promenljive preko kojih bi se
oblačnost mogla predvideti (npr podaci o solarnoj iradijaciji za datu lokaciju).

Podaci o vetrovitosti uzeti su u obzir, s obzirom da, iako imaju veliki broj 
nedostajućih podataka, postoji podatak o srednjoj brzini vetra za svaki dan,
te su ovi podaci ubačeni tamo gde nije postojala vrednost u fajlu "EMS_Weather_Hourly".

Pored ovih promenljivih, obeleženi su i praznici i vikendi, tj neradni dani.
Tu bi od značaja bile i informacije o tipu potrošača, kako bi bilo poznato 
šta bi moglo da se očekuje po pitanju profila potrošnje. 

Vetrovitost, praznici i vikendi nisu nosili mnogo težine, tj njihov značaj
nije bio veliki za sam model. Model je najveći značaj davao podacima o samoj 
potrošnji (ubedljivo), temperaturi i vremenskom trenutku. 