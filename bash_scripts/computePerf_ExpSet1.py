import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import statistics
import os.path
import pandas as pd
import statistics

which=0; # 0 = solValue
if (len(sys.argv) == 1):
    print("usage: plot [vqm] [printLegend:1 or 0] <results filenames>");
    exit();

###parse arguments
fnames=[];
algnames=[];
datanames = []
nalgs=0;

for i in range(2, len(sys.argv)):
    arg=sys.argv[i];
    fnames.append( arg );
    pos = arg.rfind('_');
    arg2 = arg[pos+1:]
    pos = arg2.find('.csv');
    alg = arg2[0:pos];
    if(alg=="PGB"):
        alg="LS+PGB"
    # print("alg:", alg)
    algnames.append( alg );
    nalgs = nalgs + 1;
    pos = arg.find("/");
    arg = arg[pos+1:];
    pos = arg.find("/");
    arg = arg[pos+1:];
    pos = arg.find('_');
    dataname=arg[0:pos];
    # print(dataname)
    if(dataname[0:2] == "ba" or dataname[0:2] == "BA"):
        dataname2 = "MaxCover(BA)"
    if(dataname[0:2] == "er" or dataname[0:2] == "ER"):
        dataname2 = "MaxCover(ER)"
    if(dataname[0:2] == "ws" or dataname[0:2] == "WS"):
        dataname2 = "MaxCover(WS)"
    if(dataname[0:3] == "sbm" or dataname[0:2] == "SBM"):
        dataname2 = "MaxCover(SBM)"
    if(dataname[0:5] == "twitt" or dataname == "RevenueMax" or dataname == "YOUTUBE2000" or dataname[0:5] == "Orkut"):
        dataname2 = "RevMax"
    if(dataname[0:6] == "RevMax"):
        dataname2 = "RevenueMax(WS)"
    if(dataname[0:5] == "image" or dataname == "IMAGESUMM"):
        dataname2 = "ImageSumm"
    if(dataname[0:6] == "friend" or dataname == "INFLUENCEEPINIONS" or dataname[0:7]=="Youtube"):
        dataname2 = "InfluenceMax"
    if(dataname[0:6] == "InfMax"):
        dataname2 = "InfluenceMax(ER)"
    if(dataname[0:5] == "movie" or dataname == "MOVIERECC"):
        dataname2 = "MovieRecc"
    datanames.append(dataname2)
# print(datanames)
# print(algnames)


X = [];
Obj = [];
ObjStd = [];
skip = [ False for i in range(0,nalgs) ];
nodes = 0;
kmin=0;
which=0
count_btr = 0
count_btr2 = 0
total_k = 0

if (sys.argv[1][0] == 'v'):
    print ("\n\nRESULTS FOR OBJECTIVE:- Alg1 = ", algnames[1], "; Alg2 = ", algnames[0]);
if (sys.argv[1][0] == 't'):
    print ("\n\nRESULTS FOR PARALLEL RUNTIME:- Alg1 = ", algnames[1], "; Alg2 = ", algnames[0]);
if (sys.argv[1][0] == 'q'):
    print ("\n\n\nRESULTS FOR NUMBER OF QUERY CALLS:- Alg1 = ", algnames[1], "; Alg2 = ", algnames[0]);
k_prev = [0]*1000
i = 0
apps = []
# for i in range( 0, nalgs, 1 ):
while i < nalgs:
    Obj_tmp = []
    ObjStd_tmp = []
    X_tmp = []
    fname = fnames[ i ];
    if i%2==1 and (not os.path.isfile( fnames[i-1] )):
        print(i, "NO COMPARISON FILE - ",fnames[i-1])
        i += 1
        continue
    elif i%2==0 and not os.path.isfile( fname ):
        i += 2
        continue
    # print(i,fname)
    if (os.path.isfile( fname )):
        skip[i]=False;
        # print ("Reading from file", fname);
        data = pd.read_csv(fname)  
        # print(data)
        k_distinct = data.k.unique()
        if(i%2==0):
            k_prev = k_distinct
        else:
            if(len(k_distinct)>len(k_prev)):
                k_distinct = k_prev

        if(i%2==1):
            total_k += len(k_distinct)
            # print("\nApp\t", "\tk\t","\tFAST\t" ,"\t\tPGB\t", "\t\tSpeedUp(%)")
        k = list(data.k)
        f_of_S = list(data.f_of_S)
        # qry = list(data.Queries)
        time = list(data.Time)
        nodes = list(data.n)[0]
        
        for j in range(len(k_distinct)):
            X_tmp.append(k_distinct[j])
            if(which==0):
                if (sys.argv[1][0] == 'v'):
                    obj_ele = [elem for ii, elem in enumerate(f_of_S) if k_distinct[j] == k[ii]]
                if (sys.argv[1][0] == 't'):
                    obj_ele = [elem for ii, elem in enumerate(time) if k_distinct[j] == k[ii]]
                if (sys.argv[1][0] == 'q'):
                    obj_ele = [elem for ii, elem in enumerate(qry) if k_distinct[j] == k[ii]]
            obj_mean = np.mean(obj_ele)
            # print("1. Algo: ", algnames[i], " App: ", datanames[i], "k: ",k_distinct[j]," " ,  Obj, obj_mean, obj_ele)
            if(i%2==1 and which == 0):
                k_prev = [0]*1000
                
                if (sys.argv[1][0] == 'v'):
                    # print(Obj[i], Obj[i-1])
                    if((obj_mean > (Obj[len(Obj)-1][j]))):
                        count_btr += 1  
                    if(Obj[len(Obj)-1][j] > (obj_mean)):
                        count_btr2 += 1
                else:
                    if((1.01*obj_mean) <= Obj[len(Obj)-1][j]):
                        count_btr += 1
                    if((1.01*Obj[len(Obj)-1][j]) <= obj_mean):
                        count_btr2 += 1

                Obj_tmp.append(obj_mean)
            else:
                Obj_tmp.append(obj_mean)
                k_prev = k_distinct
    apps.append(datanames[i])  
    Obj.append(Obj_tmp)
    ObjStd.append(Obj_tmp)
    i += 1
if(len(Obj)==0):
    exit()

print("\n")
final_Obj_1=[]
final_Obj_2=[]
final_Obj_3=[]
# printing Aligned Header
if (sys.argv[1][0] in ['v']):
    print(f"{'data' : <25}{'Avg Alg1/Alg2' : <25}{'Mean Alg1' : <25}{'Mean Alg2' : <25}")
else:
    print(f"{'data' : <25}{'Avg Alg1/Alg2' : <25}{'Avg Alg2/Alg1' : <25}{'Mean Alg1' : <25}{'Mean Alg2' : <25}")
  
# printing values of variables in Aligned manner
for i in range(1,len(Obj)):
    if(i%2==1):
        minLen = min(len(Obj[i-1]), len(Obj[i]))
        # print(len(Obj[i-1]), len(Obj[i]), min(len(Obj[i-1]), len(Obj[i])))
        final_Obj_1 = np.append(final_Obj_1,Obj[i-1][:minLen])
        final_Obj_2 = np.append(final_Obj_2,Obj[i][:minLen])
        # final_Obj_3 = np.append(final_Obj_3,Obj[i])
        if (sys.argv[1][0] in ['v']):
            print(f"{apps[i] : <25}{np.sum(Obj[i][:minLen])/np.sum(Obj[i-1][:minLen]) : <25}{np.mean(Obj[i][:minLen]) : <25}{np.mean(Obj[i-1][:minLen]) : <25}") #{np.min(Obj[i]) : ^25}{np.max(Obj[i]) : >5}")
        else:
            print(f"{apps[i] : <25}{np.sum(Obj[i][:minLen])/np.sum(Obj[i-1][:minLen]) : <25}{np.sum(Obj[i-1][:minLen])/np.sum(Obj[i][:minLen]) : <25}{np.mean(Obj[i][:minLen]) : <25}{np.mean(Obj[i-1][:minLen]) : <25}") 


if (sys.argv[1][0] == 'v'):
    final_obj1 = [x/y for x, y in zip(map(float, final_Obj_2), map(float, final_Obj_1))]
    print("\nOverall Avg Alg1/Alg2: ",np.mean(final_obj1) )

else:
    final_obj1 = [x/y for x, y in zip(map(float, final_Obj_2), map(float, final_Obj_1))]
    print("\nOverall Avg Alg1/Alg2: ",np.mean(final_obj1) )
    final_obj2 = [x/y for x, y in zip(map(float, final_Obj_1), map(float, final_Obj_2))]
    print("\nOverall Avg Alg2/Alg1: ",np.mean(final_obj2) )