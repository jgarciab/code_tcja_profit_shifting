import json
from pathlib import Path
from pandas import read_csv
import numpy as np

base_path = Path(__file__).parent


##Useful data
#Convert country name to ISO3
file_path = (base_path / "harmonizing/name2iso3.json").resolve() #relative paths within script
with open(file_path,"r") as f:
    name2iso3 = json.loads(f.read())
#     name2iso3["Curaçao and Sint Maarten"] = "ANT" 
    name2iso3["Africa, other countries"] = "IRS_Other_Africa" 
    name2iso3["Americas, other countries"] = "IRS_Other_America" 
    name2iso3["Asia & Oceania, other countries"] = "IRS_Other_Asia_Oceania" 
    name2iso3["Europe, other countries"] = "IRS_Other_Europe" 
    name2iso3["Bolivia (Plurin. State of)"] = name2iso3["Bolivia"]
    name2iso3["Micronesia (Fed. States of)"] = name2iso3["Micronesia"] 
    name2iso3["Netherlands Antilles [former]"] = "ANT" 
    name2iso3["Sudan [former]"] = name2iso3["Sudan"] 
    name2iso3["Venezuela (Boliv. Rep. of)"] = name2iso3["Venezuela"] 
    name2iso3["Bonaire, St. Eustatius & Saba"] = "BES" 
    name2iso3["Côte d`Ivoire"] = name2iso3["Côte d'Ivoire"] 
    name2iso3["China, Taiwan"] = name2iso3["Taiwan"] 
    name2iso3["China, People`s Republic of"] = "CHN" 
    name2iso3["Columbia"] = name2iso3["Colombia"] #german spelling?
    name2iso3["Macau (China)"] = name2iso3["Macau"]
    name2iso3["China (People’s Republic of)"] = "CHN"
    name2iso3["DPRK"] = name2iso3["North Korea"]
    name2iso3["Palestinian Authority"] = name2iso3["Palestine"]
    name2iso3["Côte d’Ivoire"] = name2iso3["Côte d'Ivoire"] 
    name2iso3["DRC"] = name2iso3["Democratic Republic of the Congo"]
    name2iso3["Bailiwick of Guernsey"] = name2iso3["Guernsey"]
    name2iso3["Hong Kong (China)"] = name2iso3["Hong Kong"]
    # So we don't drop the UK islands
    name2iso3["UK_Caribbean"] = "UK_Caribbean"
    name2iso3["United Kingdom Islands, Caribbean"] = "UK_Caribbean"
    
    # To clean the 
    #Official exchange rate, LCU per USD, period average,,  Source: Global Economic Monitor (GEM).
    name2iso3["Congo Dem. Rep."] = name2iso3["Congo (Democratic Republic)"]
    name2iso3["Congo Rep."] = name2iso3["Congo (Rep.)"]
    name2iso3["Egypt Arab Rep."] = name2iso3["Egypt"]
    name2iso3["Gambia The"] = name2iso3["Gambia"]
    name2iso3["Hong Kong China"] = name2iso3["Hong Kong"]
    name2iso3["Iran Islamic Rep."] = name2iso3["Iran"]
    name2iso3["Korea Rep."] = name2iso3["Korea"]
    name2iso3["Macao China"] = name2iso3["Macao"]
    name2iso3["Macedonia FYR"] = name2iso3["Macedonia, FYR"]
    name2iso3["Taiwan China"] = name2iso3["Taiwan"]
    name2iso3["Yemen Rep."] = name2iso3["Yemen"]
    name2iso3["Bahamas The"] = name2iso3["Bahamas"]
    name2iso3["Venezuela RB"] = name2iso3["Venezuela"]
    name2iso3["Micronesia Fed. Sts."] = name2iso3["Micronesia"]

    nd = dict()
    for i in name2iso3:
        nd[i.replace(" ","")] = name2iso3[i]
    name2iso3.update(nd)
# import json
# with open(file_path, 'w') as fp:
#     json.dump(name2iso3, fp)

    
file_path = (base_path / "harmonizing/iso3_iso2_name.csv").resolve() #relative paths within script
iso32name = read_csv(file_path,sep="\t").set_index("iso3").to_dict()["name"]
iso32iso2 = read_csv(file_path,sep="\t").set_index("iso3").to_dict()["iso2"]
    
    
def iso3_to_name(iso3):
    """
    Returns ISO3 code of a country name
    """
    
    if iso32name.get(iso3) is None:
        print("{} not matched to any file".format(iso3))
        return iso3
    else:
        return iso32name[iso3]
    
def iso3_to_iso2(iso3):
    """
    Returns ISO3 code of a country name
    """
    
    if iso32iso2.get(iso3) is None:
        print("{} not matched to any file".format(iso3))
        return iso3
    else:
        return iso32iso2[iso3]
    

##Useful functions
def get_iso3(country_name,print_failure=True):
    """
    Returns ISO3 code of a country name
    """
    
    if name2iso3.get(country_name) is None:
        if print_failure == True:
            print("{} not matched to any file".format(country_name))
        return np.nan#country_name
    else:
        return name2iso3[country_name]
    
    

def save_to_sheets(datas,sheet_names,filename,startrow=0,comment=None,startcol=0):
    from openpyxl import load_workbook
    writer = pd.ExcelWriter(filename, engine="xlsxwriter")
#     if comment is not None:
#         startrow = 1
#     else:
#         startrow = 0


    if len(datas) != len(sheet_names):
        raise("Provide same number of dataframes as sheet names")
    for data,sheet_name in zip(datas,sheet_names):
        data.to_excel(writer,sheet_name=sheet_name,startrow=startrow,index=False,startcol=startcol)

#     if comment is not None:
#         w = writer.sheets[cf]
#         w.write_string(0,1,comment)

    writer.save()
    writer.close()
  