import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import statistics
import os.path
import pandas as pd

which=0; # 0 = solValue
if (len(sys.argv) == 1):
    print("usage: plot [vqm] [printLegend:1 or 0] <results filenames>");
    exit();

###parse arguments
fnames=[];
algnames=[];
datanames=[]
dataname2 = "DEFAULT"
nalgs=0;
postfix='_exp1.pdf';
colorFlag = False
metaFlag = False
rgcFlag = False
repFlag = False
for i in range(3, len(sys.argv)):
    arg=sys.argv[i];
    fnames.append( arg );
    pos = arg.rfind('_');
    arg2 = arg[pos+1:]
    pos = arg2.find('.csv');
    alg = arg2[0:pos];
    if(alg[0:6]=="DASH-8"):
        alg="R-DASH"
    if(alg[0:6]=="DASH-2"):
        alg="R-DASH-2"
        repFlag = True
    elif(alg[0:5]=="GDASH"):
        alg="G-DASH"
        repFlag = True
    elif(alg[0:5]=="TDASH"):
        alg="T-DASH"
        repFlag = True
    elif(alg[0:6]=="DD-8-4"):
        alg="DDist-8"
    elif(alg[0:8]=="Parallel"):
        alg="PAlg"
    elif(alg[0:3]=="PGB"):
        alg="LS+PGB"
        repFlag = True
    elif(alg[0:3]=="BiC"):
        alg= "BCG" # "BiCriteriaGreedy"
        colorFlag = True
    elif(alg=="RandGreedI-8-1"):
        alg= "RG-8"
        rgcFlag = True
    elif(alg=="RandGreedI-32-1"):
        alg= "RG"
        # rgcFlag = True
    elif(alg=="DD-8-1"):
        alg= "DDist-8"
        rgcFlag = True
    elif(alg=="DD-32-1"):
        alg= "DDist"
        # rgcFlag = True
    elif(alg[0:13]=="RandGreedILAG"):
        alg= "RG-LAG"
    elif(alg[0:13]=="RandGreedILAG"):
        alg= "RG-LAG" #"RandGreedI"
        colorFlag = True
    elif(alg[0:2]=="RG" or alg[0:4]=="Rand"):
        alg= "RG" #"RandGreedI"
        colorFlag = True
    elif(alg[0:4]=="Meta"):
        alg= "MED+(RDASH)" #"RandGreedI"
        metaFlag = True
        repFlag = True
    elif(alg[0:7]=="MEDDASH"):
        alg= "MED+(RDASH)" #"RandGreedI"
        metaFlag = True
        repFlag = True
    # elif(alg[0:4]=="Meta"):
    #     alg="MetaGreedy"
    # elif(alg[0:4]=="ParallelAlgo"):
    #     alg="ParallelAlgo"
            
    pos = arg.find("/");
    arg = arg[pos+1:];
    # print("alg:", alg)
    algnames.append( alg );
    nalgs = nalgs + 1;
    pos = arg.find("/");
    arg = arg[pos+1:];
    pos = arg.find('_');
    dataname=arg[0:pos];
    if(i==3):
        if(dataname[0:2] == "ba" or dataname[0:2] == "BA"):
            dataname2 = "MaxCover(BA)"
        if(dataname[0:2] == "er" or dataname[0:2] == "ER"):
            dataname2 = "MaxCover(ER)"
        if(dataname[0:2] == "ws" or dataname[0:2] == "WS"):
            dataname2 = "MaxCover(WS)"
        if(dataname[0:3] == "sbm" or dataname[0:2] == "SBM"):
            dataname2 = "MaxCover(SBM)"
        if(dataname[0:5] == "twitt" or dataname == "RevenueMax" or dataname == "YOUTUBE2000" or dataname[0:5] == "Orkut"):
            dataname2 = "RevenueMax"
        if(dataname[0:6] == "RevMax"):
            dataname2 = "RevenueMax(WS)"
        if (dataname[0:5] == "Orkut"):
            dataname2 = "RevenueMax"
        if(dataname[0:5] == "image" or dataname == "IMAGESUMM"):
            dataname2 = "ImageSumm"
        if(dataname[0:6] == "friend" or dataname == "INFLUENCEEPINIONS" or dataname[0:7] == "Youtube"):
            dataname2 = "InfluenceMax"
        if(dataname[0:6] == "InfMax"):
            dataname2 = "InfluenceMax(ER)"
        if(dataname[0:5] == "movie" or dataname == "MOVIERECC"):
            dataname2 = "MovieRecc"
    

normalizeX=False;

plot_dir = "plots/"
if (sys.argv[2][0] == '1'):
    printlegend=True;
else:
    printlegend=False;
    
scaleDiv=1;
normalize=False;

colorsHash = {}
colorsHash['RG'] = 'goldenrod'
colorsHash['BCG'] = 'c'
colorsHash['PAlg'] = 'y'
colorsHash['DDist'] = 'darkviolet'
colorsHash['R-DASH'] = 'b'
colorsHash['R-DASH-8'] = 'b'
colorsHash['R-DASH-2'] = 'm'
colorsHash['T-DASH'] = 'g'
colorsHash['G-DASH'] = 'r'
colorsHash['LS+PGB'] = 'olivedrab'
colorsHash['(LS+PGB)$^{H}$'] = 'olivedrab'
colorsHash['RG-LAG'] = 'm'
colorsHash['MED+(RDASH)'] = 'k'
colorsHash['RG-8'] = 'lime'
colorsHash['RG-32'] = 'goldenrod'
colorsHash['DDist-8'] = 'plum'
colorsHash['DDist-32'] = 'darkviolet'

markersHash = {}
markersHash['RG'] = 's'
markersHash['BCG'] = '>'
markersHash['PAlg'] = 'd'
markersHash['DDist'] = '*'
markersHash['R-DASH'] = 'X'
markersHash['R-DASH-8'] = 'X'
markersHash['R-DASH-2'] = '^'
markersHash['T-DASH'] = '^'
markersHash['G-DASH'] = 'o'
markersHash['LS+PGB'] = '.'
markersHash['(LS+PGB)$^{H}$'] = '.'
markersHash['RG-LAG'] = '^'
markersHash['MED+(RDASH)'] = '>'
markersHash['RG-8'] = 's'
markersHash['RG-32'] = '^'
markersHash['DDist-8'] = '>'
markersHash['DDist-32'] = '*'

linesHash = {}
linesHash['RG'] = '-'
linesHash['BCG'] = ':'
linesHash['PAlg'] = '--'
linesHash['DDist'] = '-.'
linesHash['R-DASH'] = '-'
linesHash['R-DASH-8'] = '-'
linesHash['R-DASH-2'] = '-'
linesHash['T-DASH'] = ':'
linesHash['G-DASH'] = '--'
linesHash['LS+PGB'] = '-.'
linesHash['(LS+PGB)$^{H}$'] = '-.'
linesHash['RG-LAG'] = '-.'
linesHash['MED+(RDASH)'] = '--'
linesHash['RG-8'] = '-'
linesHash['RG-32'] = '-.'
linesHash['DDist-8'] = ':'
linesHash['DDist-32'] = '-.'

if (sys.argv[1][0] == 'v'):
    outFName_val= 'val-';
    normalize=True;
    which=0;
else:
    if (sys.argv[1][0] == 'q'):
        which = 1; #1 = queries
        outFName_val= 'query-';
    else:
        if (sys.argv[1][0] == 't'):
            which = 3; #1 = time
            outFName_val='time-';
        else:
            if (sys.argv[1][0] == 'd'):
                which = 4; #1 = time
                outFName_val= 'timedist-';
            else:
                if (sys.argv[1][0] == 'p'):
                    which = 5; #1 = time
                    outFName_val= 'timepost-'
                


for i in range(0,nalgs):
    if(algnames[i] == "R-DASH"):
        if (os.path.isfile( fnames[i] )):
            datatmp = pd.read_csv(fnames[i])
            datatmp = datatmp[datatmp['f_of_S'] != 'f_of_S']  
            k_distinct = datatmp.k.unique()
            print(k_distinct)
        else:
            exit()

all_OBj = []
X = [];
Obj = [];
ObjStd = [];
skip = [ False for i in range(0,nalgs) ];
nodes = 0;
nproc = 0
kmin=0;
for i in range( 0, nalgs ):
    Obj_tmp = []
    ObjStd_tmp = []
    X_tmp = []
    fname = fnames[ i ];
    if (os.path.isfile( fname )):
        skip[i]=False;
        print ("Reading from file", fname);
        data = pd.read_csv(fname) 
        
        if repFlag:
            data = data[data['f_of_S'] != 'f_of_S']
        # print(algnames[i] ,data)
        k_distinct = list(data.k.unique())
        k = list(data.k)
        f_of_S = list(data.f_of_S)
        time = list(data.Time)
        
        nodes = int(list(data.n)[0])
        nproc = int(list(data.nNodes)[0])
        
        k_max = int(nodes/pow(nproc,2))
        for j in range(len(k_distinct)):
            if k_distinct[j]=='k':
                continue
            
            X_tmp.append(int(k_distinct[j]))
            if(which==0):
                obj_ele = [float(elem) for ii, elem in enumerate(f_of_S) if k_distinct[j] == k[ii]]
            if(which==3):
                obj_ele = [float(elem) for ii, elem in enumerate(time) if k_distinct[j] == k[ii]]
            
            obj_mean = np.mean(obj_ele)
            obj_std = np.std(obj_ele)
            if(i!=0 and which == 7): #Normalize Other Algos
                obj_mean = obj_mean/Obj[0][j]
                Obj_tmp.append(obj_mean)
                obj_std = obj_std/Obj[0][j]
                ObjStd_tmp.append(obj_std)
            else:
                Obj_tmp.append(obj_mean)
                ObjStd_tmp.append(obj_std)
    # print(algnames[i], X_tmp, Obj_tmp)            
    Obj.append(Obj_tmp)
    ObjStd.append(ObjStd_tmp)
    X.append(X_tmp)
    all_OBj.extend(Obj_tmp)
if(which ==0):
    for j in range( 0, len( Obj[ 0 ] ) ):
            # Obj[0][j] = Obj[0][j]/Obj[0][j] #Normalize Algo 0
            Obj[0][j] = Obj[0][j]
            ObjStd[0][j] = 0   

# if(len(Obj[0])==0):
#     exit()
print(Obj)


   
plt.gcf().clear();
plt.rcParams['pdf.fonttype'] = 42;
plt.rcParams.update({'font.size': 25});
plt.rcParams.update({'font.weight': "bold"});
plt.rcParams["axes.labelweight"] = "bold";
# plt.xscale('log');
title = str(dataname2) + " ($n=" + str(nodes) + "$)"
plt.rc('font', size=19) 
# if(which==0):

plt.figure(figsize=(4, 4))
plt.title(title, fontsize=18, pad=20)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0) );

if (which == 1):
    print (nodes);
    plt.ylabel( 'Queries / $n$', fontsize=22 );
    plt.yscale('log', basey=10);
    for i in range( 0, nalgs ):
            for j in range( 0, len( Obj[ i ] ) ):
                Obj[i][j] = Obj[i][j]/nodes;
                ObjStd[i][j] = ObjStd[i][j]/nodes;
else:
    if (which == 3):
            # plt.ylabel( 'Adaptive Rounds / $n$' );
            plt.ylabel( 'Time Taken (s)', fontsize=22 );
            plt.yscale('log');
            for i in range( 0, nalgs ):
                for j in range( 0, len( Obj[ i ] ) ):
                    Obj[i][j] = Obj[i][j];
                    ObjStd[i][j] = ObjStd[i][j];
    else:
        if (which == 4):
                # plt.ylabel( 'Adaptive Rounds / $n$' );
                plt.ylabel( 'Time Taken Dist. (s)', fontsize=22 );
                plt.yscale('log');
                for i in range( 0, nalgs ):
                    for j in range( 0, len( Obj[ i ] ) ):
                        Obj[i][j] = Obj[i][j];
                        ObjStd[i][j] = ObjStd[i][j];
        else:
            if (which == 5):
                    # plt.ylabel( 'Adaptive Rounds / $n$' );
                    plt.ylabel( 'Time Taken Post-Proc (s)', fontsize=22 );
                    plt.yscale('log');
                    for i in range( 0, nalgs ):
                        for j in range( 0, len( Obj[ i ] ) ):
                            Obj[i][j] = Obj[i][j];
                            ObjStd[i][j] = ObjStd[i][j];
            else:
                # plt.yscale('log');
                # plt.ylabel( "Objective / Greedy" )
                # plt.ylabel( str("Objective/"+ str(algnames[0]) ))
                plt.ylabel( str("Objective"), fontsize=22)
        

if(nodes in [50000, 53889] or nodes > 100000 or metaFlag):
    plt.xlabel( "$k$", fontsize=25 );
    # for i in range( 0, nalgs ):
        # if algnames[i] == "R-DASH":
        #     algnames[i] = "R-DASH-8"
    #     for j in range( 0, len( X[ i ] ) ):
    #         X[i][j] = X[i][j] / k_max;
    # plt.xlabel( "$k$/$(n/\ell^2)$", fontsize=25);
    postfix='_exp2.pdf';
    if(algnames[1]=="LS+PGB"):
        algnames[1] = "(LS+PGB)$^{H}$"

    # plt.xlim( 0.005,0.105  );     
else:
    plt.xlabel( "$k$", fontsize=25 );
    for i in range( 0, nalgs ):
        for j in range( 0, len( X[ i ] ) ):
            # X[i][j] = X[i][j] / k_max;
            X[i][j] = X[i][j];

outFName= plot_dir +outFName_val  + dataname + postfix;


if which==0 and normalize==False:
    #normalize by nodes
    for i in range(0,nalgs):
        for j in range( 0, len( Obj[ i ] ) ):
            Obj[i][j] = Obj[i][j];
            ObjStd[i][j] = ObjStd[i][j];
            # plt.ylabel( "Objective / Greedy" );
            plt.ylabel( "Objective Value" );            

# if rgcFlag:
#     plt.xlabel( "$k / n$" );
#     for i in range( 0, nalgs ):
#         for j in range( 0, len( X[ i ] ) ):
#             X[i][j] = X[i][j] / nodes;

markSize=16;

if normalize:
    algmin = 0;
    algmax = nalgs;
    # plt.ylim( 0.0, 1.25 );
else:
    algmin = 0;
    algmax = nalgs;


print( nodes );

if (which == 1):
    #plt.axhline( nodes, color='r' );
    ax = plt.gca();
    #ax.annotate('$n$', xy=(float(kmin), nodes), xytext=(float(kmin)-75, nodes - 100000), size=15 );

for i in range(algmin,algmax):
    mi = 1;
    plt.plot( X[i], Obj[i], ':', marker=markersHash[algnames[i]],  label=algnames[i],ms = markSize,color = colorsHash[algnames[i]], markevery = mi, linestyle = linesHash[algnames[i]]);
    BObj = np.asarray( Obj[i] );
    BObjStd = np.asarray( ObjStd[i] );
    if i != 3:
        plt.fill_between( X[i], BObj - BObjStd, BObj + BObjStd,
                          alpha=0.5, edgecolor=colorsHash[algnames[i]], facecolor=colorsHash[algnames[i]]);
if dataname2 == "ImageSumm":
    if(which == 0):
        if(nodes in [50000, 53889] or nodes > 100000):
            plt.ylim( 0, 50000 );
        else:
            plt.ylim( 0, 10000 );

if rgcFlag:
    plt.yscale('linear');
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0) );
#plt.errorbar( X, Obj, yerr=BObjStd, fmt='-');
dist_set_max = nodes/nproc
# plt.axvline(x=dist_set_max/nodes, color='r', linestyle='-')
if(which == 0):
    yht = 0.00000001   
else:
    yht = float(np.min(all_OBj))


plt.gca().grid(which='major', axis='both', linestyle='--')


#plt.grid(color='grey', linestyle='--' );
if((dataname2 == "ImageSumm")  and which == 0):
    if printlegend:
        plt.legend(loc='lower right', numpoints=1,prop={'size':18},framealpha=0.6);


plt.savefig( outFName, bbox_inches='tight', dpi=400 );