import csv

# Maintain mapping for tournament teams to ensure consistency
mapping = {
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "South Korea": "Korea Republic",
    "D.R. Congo": "Congo DR",
    "D.R.[1] Congo": "Congo DR",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Côte d'Ivoire",
    "Curacao": "Curaçao",
}

raw_data = """Argentina,Iceland,3,0
Iraq,Venezuela,0,2
Saudi Arabia,Senegal,0,0
Liberia,Sierra Leone,3,1
Azerbaijan,San Marino,2,1
Togo,Benin,5,1
Algeria U23,Mauritania U23,1,3
Angola,Central Africa,3,0
Egypt U19,Russia U19,2,5
Hungary,Kazakhstan,3,1
Russia,Trinidad & Tobago,3,0
Belarus,Burkina Faso,2,2
Costa Rica U23,Cuba U23,0,0
D.R.[1] Congo,Chile,1,2
Moldova U21,Georgia U21,3,2
Armenia,Moldova,1,1
Ethiopia,Malawi,1,1
Tajikistan,India,1,1
USA U19,Japan U19,5,2
Ireland U21,Qatar U23,0,0
Kyrgyzstan,Palestine,0,0
Russia U21,Iraq U23,1,0
Azerbaijan U21,Kyrgyzstan U20,1,3
Azerbaijan U20,Pakistan U20,1,0
Indonesia,Mozambique,1,0
Cambodia,Hong Kong,2,0
China,Thailand,0,0
China U23,Tajikistan U23,3,0
Philippines,Myanmar,5,1
South Korea U23,Kyrgyzstan U23,0,1
Finland U18,Norway U18,3,1
Thailand U23,United Arab Emirates U23,2,5
Turkey U18,Ukraine U18,0,1
Equatorial Guinea,Comoros,0,1
Vanuatu,Fiji,2,2
Peru,Spain,1,3
Philadelphia Union II,Ivory Coast,0,2
France,Northern Ireland,3,1
Niger,Mauritania,0,1
Netherlands,Uzbekistan,2,1
Italy U21,Albania U21,1,0
Norway U21,Finland U21,0,2
USA U21,Uzbekistan U23,0,1
Tunisia U20,Algeria U20,1,1
Belarus U21,Kazakhstan U21,2,1
Iraq U20,Jordan U20,0,0
USA U18,Qatar U20,2,0
Latvia U19,Estonia U19,3,2
Saudi Arabia U20,Panama U20,0,0
Japan U21,Ukraine U21,3,0
North Macedonia U21,USA U20,0,2
Sri Lanka,Bhutan,4,1
Sweden U18,Wales U19,3,1
Albania U17,Bulgaria U17,5,0
Colombia,Jordan,2,0
Ecuador,Guatemala,3,0
Brazil U17,USA U17,4,0
Greece,Italy,0,1
Morocco,Norway,1,1
Croatia,Slovenia,2,1
Kosovo,Andorra,3,0
Denmark,Ukraine,2,1
Maldives,Bangladesh U23,1,1
Kenya,Lesotho,4,0
Liechtenstein,Cyprus,0,2
Oman,Mozambique,4,1
Afghanistan,Pakistan,0,2
Bulgaria U19,Albania U19,3,0
Jamaica U20,Haiti U20,9,0
Argentina,Honduras,2,0
Curacao,Aruba,4,0
Brazil,Egypt,2,1
Venezuela,Turkey,1,2
Chile U20,Brazil U20,0,1
Jamaica,South Africa,1,1
Bolivia,Scotland,0,4
Cape Verde,Bermuda,3,0
England,New Zealand,1,0
Qatar,El Salvador,0,0
Morocco U23,Comoros,2,1
Panama,Bosnia & Herzegovina,1,1
Switzerland,Australia,1,1
USA,Germany,1,2
Albania,Luxembourg,0,1
Portugal,Chile,2,1
Romania,Wales,2,1
British Virgin Islands,Bonaire,2,0
Egypt U19,Russia U19,3,1
Gibraltar,Cayman Islands,4,1
Sierra Leone,Liberia,1,0
Croatia U21,Ireland U21,2,2
Finland U18,Turkey U18,3,1
Ethiopia,Malawi,1,0
Jordan U20,Panama U20,1,2
Kosovo U21,Luxembourg U21,0,0
Kyrgyzstan,Palestine,0,0
Armenia,Kazakhstan,1,1
Azerbaijan U21,Bahrain U20,3,0
Estonia U19,Lithuania U19,3,1
Belgium,Tunisia,5,0
Myanmar,Guam,6,1
Kyrgyzstan U23,United Arab Emirates U23,1,1
Thailand U23,South Korea U23,2,3
Ukraine U18,Norway U18,1,1
Czech Republic U18,Romania U18,0,1
China U23,Tajikistan U23,1,0
Vanuatu,Fiji,2,1
Mexico U20,Japan U19,1,3
Haiti,Peru,1,2
Canada,Ireland,1,1
Puerto Rico,Saudi Arabia,0,3
Paraguay,Nicaragua,4,0
Angola,Mauritania,1,1
Slovenia U21,Albania U21,1,1
Azerbaijan,Malta,0,2
Benin,Niger,1,1
Central Africa,Togo,1,1
Hungary,Finland,2,1
Algeria U23,Mauritania U23,2,0
Moldova,Bulgaria,2,2
Russia,Burkina Faso,3,0
San Marino,Bangladesh,1,2
Slovakia,Montenegro,2,2
Belarus,Syria,4,1
Georgia,Bahrain,2,0
USA U21,Ukraine U21,3,1
Tunisia U20,Algeria U20,1,0
Albania U17,Bulgaria U17,1,4
Montenegro U21,Cyprus U21,1,1
Tajikistan,India,3,1
USA U18,Sweden U18,0,2
Russia U21,Iraq U23,5,1
USA U20,Georgia U21,1,0
Belarus U21,Kazakhstan U21,2,0
Uzbekistan U23,Japan U21,0,1
Indonesia,Oman,3,0
Thailand,Kuwait,2,2
Hong Kong,Mongolia,2,0
Singapore,China,1,2
Qatar U20,Wales U19,0,2
Serbia U17,Romania U17,2,1
Mexico,Serbia,5,1
Czech Republic,Guatemala,3,1
Ireland,Grenada,5,0"""

with open("data/friendlies.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["team_home", "team_away", "score_home", "score_away"])
    for line in raw_data.split("\n"):
        parts = line.split(",")
        if len(parts) == 4:
            home, away, s_h, s_a = parts
            home = mapping.get(home, home)
            away = mapping.get(away, away)
            # Include ALL matches without filtering by teams_list
            writer.writerow([home, away, s_h, s_a])
