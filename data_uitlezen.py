import pandas as pd

data = pd.read_csv("data/datasealvl.txt", header=None, sep='\s+')
data.columns = ["altimeter", "cycle", "year", "observations", "weighted_observations", "GMSLNA", "std_GMSLNA", "smoothed_GMSLNA", "GMSLA", "std_GMSLA", "smoothed_GMSLA", "smoothed_GMSLA_(semi)-an_rm"]
data.to_csv("sealevel_data.csv", index=False)

januari = []
februari = []
maart = []
april = []
mei = [] 
juni = [] 
juli = [] 
augustus = [] 
september = [] 
oktober = [] 
november = []
december = []
i = 0
j = 0
temp = 0
for a in data["GMSLNA"]:
    if i%3 == 0:
        if j == 0 and i != 0:
            januari.append(temp/3)
        elif j == 1:
            februari.append(temp/3)
        elif j == 2:
            maart.append(temp/3)
        elif j == 3:
            april.append(temp/3)
        elif j == 4:
            mei.append(temp/3)
        elif j == 5:
            juni.append(temp/3)
        elif j == 6:
            juli.append(temp/3)
        elif j == 7:
            augustus.append(temp/3)
        elif j == 8:
            september.append(temp/3)
        elif j == 9:
            oktober.append(temp/3)
        elif j == 10:
            november.append(temp/3)
        else:
            december.append(temp/3)
            j = -1 
        
        j+=1
        temp = 0
    temp =+a
    i +=1

print(januari)
    
    
