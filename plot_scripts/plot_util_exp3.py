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
for i in range(3, len(sys.argv)):
    arg=sys.argv[i];
    fnames.append( arg );
    pos = arg.rfind('_');
    arg2 = arg[pos+1:]
    pos = arg2.find('.csv');
    alg = arg2[0:pos];
    if(alg[0:3]=="Dis"):
        alg="DTSB"
    elif(alg[0:3]=="PGB"):
        alg="LS+PGB"
    elif(alg[0:3]=="BiC"):
        alg="BiCriteriaGreedy"
        colorFlag = True
    elif(alg[0:2]=="RG"):
        alg="RandGreedI"
        colorFlag = True
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
        if(dataname[0:5] == "Orkut" or dataname == "RevenueMax" or dataname == "YOUTUBE2000"):
            dataname2 = "RevMax"
        if(dataname[0:6] == "RevMax"):
            dataname2 = "RevenueMax(WS)"
        if(dataname[0:5] == "image" or dataname == "IMAGESUMM"):
            dataname2 = "ImageSumm"
        if(dataname[0:7] == "Youtube" or dataname == "INFLUENCEEPINIONS"):
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


colors = [ 'r',
           'darkviolet',
           # 'olivedrab',
           'goldenrod',
           'c',
           'darkorange',
           'dodgerblue',
           'darkseagreen',
           'yellowgreen',
           'g',
           'm',
           'lime',
           'r',
           'b',
           'k',
           'y',
           ];

markers = [ 'X',
            's',
            # 'o',
            '>',
            'd',
            '1',
            '*',
            'X',
            '^',
            'P',
            'p',
            'v',
            '<'  ];

linestyles = [ '-',
            # '--',
            ':',
            '-.',
            '-',
            '--',
            ':',
            '-.',
            '-',
            '--',
            ':',
            '-.' ];
            
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
                
if (os.path.isfile( fnames[0] )):
    datatmp = pd.read_csv(fnames[0])  
    k_distinct = datatmp.k.unique()

else:
    exit()
all_OBj = []
X = [];
Obj = [];
ObjStd = [];
Obj2 = [];
ObjStd2 = [];
skip = [ False for i in range(0,nalgs) ];
nodes = 0;
nproc = 0
kmin=0;

nproc_vec = []
DLSPGB_vec = []
RGGB_vec = []
BCG_vec = []
DLSPGB_vecDist = []
RGGB_vecDist = []
BCG_vecDist = []
DLSPGB_vecPost = []
RGGB_vecPost = []
BCG_vecPost = []

for i in range( 0, nalgs ):
    if(which==0):
        Obj_tmp = []
        ObjStd_tmp = []
    if(which==3):
        Obj_tmpDist = []
        Obj_tmpPost = []
        ObjStd_tmpDist = []
        ObjStd_tmpPost = []
    X_tmp = []
    fname = fnames[ i ];
    if (os.path.isfile( fname )):
        skip[i]=False;
        print ("Reading from file", fname);
        data = pd.read_csv(fname)  
        # k_distinct = data.k.unique()
        k = list(data.k)
        f_of_S = list(data.f_of_S)
        qry = list(data.Queries)
        time = list(data.Time)
        timeDist = list(data.TimeDist)
        timePost = list(data.TimePost)
        nodes = list(data.n)[0]
        nproc = list(data.nNodes)[0]
        nproc_vec.append(nproc)
        
        if(which==0):
            obj_ele = [elem for ii, elem in enumerate(f_of_S)]
            obj_mean = np.mean(obj_ele)
            obj_std = np.std(obj_ele)
            Obj_tmp.append(obj_mean)
            ObjStd_tmp.append(obj_std)

            if(algnames[i][0:4]=="DASH"):
                DLSPGB_vec.append(obj_mean)     
                  
            elif(algnames[i][0:4]=="BiCr"):
                BCG_vec.append(obj_mean)
                
            elif(algnames[i][0:4]=="Rand"):
                RGGB_vec.append(obj_mean)
        
        if(which==3):
            obj_eleDist = [elem for ii, elem in enumerate(time)]
            obj_elePost = [elem for ii, elem in enumerate(f_of_S)]
        
            obj_meanDist = np.mean(obj_eleDist)
            obj_stdDist = np.std(obj_eleDist)
            Obj_tmpDist.append(obj_meanDist)
            ObjStd_tmpDist.append(obj_stdDist)
            
            obj_meanPost = np.mean(obj_elePost)
            obj_stdPost = np.std(obj_elePost)
            Obj_tmpPost.append(obj_meanPost)
            ObjStd_tmpPost.append(obj_stdDist)    
        # print(algnames[i])
            if(algnames[i][0:4]=="DASH"):
                DLSPGB_vecDist.append(obj_meanDist)
                DLSPGB_vecPost.append(obj_meanPost)     
                  
            elif(algnames[i][0:4]=="BiCr"):
                BCG_vecDist.append(obj_meanDist)
                BCG_vecPost.append(obj_meanPost)

            elif(algnames[i][0:4]=="Rand"):
                RGGB_vecDist.append(obj_meanDist)
                RGGB_vecPost.append(obj_meanPost)
        
    # elif(alg=="LS+PGB"):
# print(DLSPGB_vec)
tmp = [0] * len(nproc_vec)
if(which==0):
    if(len(DLSPGB_vec)):
        Obj.append(DLSPGB_vec)
        ObjStd.append(tmp)
    if(len(RGGB_vec)):
        Obj.append(RGGB_vec)
        ObjStd.append(tmp)
    if(len(BCG_vec)):
        Obj.append(BCG_vec)
        ObjStd.append(tmp)

if(which==3):
    if(len(DLSPGB_vecDist)):
        Obj.append(DLSPGB_vecDist)
        Obj2.append(DLSPGB_vecPost)
        ObjStd.append(tmp)

    if(len(RGGB_vecDist)):    
        Obj.append(RGGB_vecDist)
        Obj2.append(RGGB_vecPost)
        ObjStd.append(tmp)

    if(len(BCG_vecDist)): 
        Obj.append(BCG_vecDist)
        Obj2.append(BCG_vecPost)
        ObjStd.append(tmp)

if(len(Obj[0])==0):
    exit()
# print(algnames)
# print(Obj)

postfix='_exp3.pdf';        

outFName= plot_dir +outFName_val  + dataname + postfix;
   
plt.gcf().clear();
plt.rcParams['pdf.fonttype'] = 42;
plt.rcParams.update({'font.size': 25});
plt.rcParams.update({'font.weight': "bold"});
plt.rcParams["axes.labelweight"] = "bold";

title = str(dataname2) + " ($n=" + str(nodes) + "$, $k=" + str(int(k[0])) + "$)"
# print(title)
plt.rc('font', size=19) 
# if(which==0):
plt.title(title, fontsize=25, pad=20)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0) );
nalgs = len(Obj)
if (which == 1):
    # print (nodes);
    plt.ylabel( 'Queries / $n$' );
    plt.yscale('log');
    for i in range( 0, nalgs ):
            if(len(Obj[i])==0):
                continue
            for j in range( 0, len( Obj[ i ] ) ):
                Obj[i][j] = Obj[i][j]/nodes;
                ObjStd[i][j] = ObjStd[i][j]/nodes;
else:
    if (which == 3):
            # plt.ylabel( 'Adaptive Rounds / $n$' );
            plt.ylabel( 'Time Taken (s)' );
            # plt.yscale('log');
            for i in range( 0, nalgs ):
                if(len(Obj[i])==0):
                    continue
                for j in range( 0, len( Obj[ i ] ) ):
                    Obj[i][j] = Obj[i][j];
                    ObjStd[i][j] = ObjStd[i][j];
    else:
        if (which == 4):
                # plt.ylabel( 'Adaptive Rounds / $n$' );
                plt.ylabel( 'Time Taken Dist. (s)' );
                plt.yscale('log');
                for i in range( 0, nalgs ):
                    if(len(Obj[i])==0):
                        continue
                    for j in range( 0, len( Obj[ i ] ) ):
                        Obj[i][j]   = Obj[i][j];
                        Obj2[i][j]  = Obj2[i][j];
                        ObjStd[i][j]= ObjStd[i][j];
        else:
            if (which == 5):
                    # plt.ylabel( 'Adaptive Rounds / $n$' );
                    plt.ylabel( 'Time Taken Post-Proc (s)' );
                    plt.yscale('log');
                    for i in range( 0, nalgs ):
                        if(len(Obj[i])==0):
                            continue
                        for j in range( 0, len( Obj[ i ] ) ):
                            Obj[i][j] = Obj[i][j];
                            ObjStd[i][j] = ObjStd[i][j];
            else:
                minY = np.min(Obj[0])/2
                maxY = np.max(Obj[0]) * 1.25
                plt.ylim( minY, maxY );
                plt.ylabel( str("Objective"))
        



plt.xlabel( "$No. of \ Machines$" );

    
if which==0 and normalize==False:
    #normalize by nodes
    for i in range(0,nalgs):
        for j in range( 0, len( Obj[ i ] ) ):
            Obj[i][j] = Obj[i][j];
            ObjStd[i][j] = ObjStd[i][j];

            # plt.ylabel( "Objective / Greedy" );
            plt.ylabel( "Objective Value" );            

markSize=16;

if normalize:
    algmin = 0;
    algmax = len(Obj);
    
else:
    algmin = 0;
    algmax = len(Obj);

# plt.xlim( 0.0005,0.15  );
# print( nodes );

if (which == 1):
    ax = plt.gca();
X = nproc_vec
print(X)
algnames = ["DASH", "RandGreedI","BiCriteriaGreedy"]
ax = plt.gca();

X = [str(x) for x in X]
width = 0.25      # the width of the bars: can also be len(x) sequence
for i in range(algmin,algmax):
    print(Obj[i])
    p1 = ax.bar(X, height=Obj[i],   yerr=ObjStd[i], color = colors[i])
    ax.set_ylabel("Time(s)", color="red",fontsize=22)
    ax.set_xlabel( "$No. of \ Machines$" );
    # set y-axis label
ax2=ax.twinx()
# make a plot with different y-axis using second axis object       
mi = 1;
maxObj2 = np.max(Obj2[i])*(1.43)
minObj2 = np.max(Obj2[i])*(0.143)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0) );
ax2.set_ylim(minObj2, maxObj2)
for i in range(algmin,algmax):
    print(Obj2[i])
    p2 = ax2.plot(X, Obj2[i], ':', marker=markers[i],  label=algnames[i],ms = markSize,color = 'b', markevery = mi, linestyle = linestyles[i])
    ax2.set_ylabel("Objective", color="blue",fontsize=22)
    plt.legend((p1[0], p2[0]), ('RunTime', 'Solution value'), loc='upper right')
    # plt.show()
#plt.errorbar( X, Obj, yerr=BObjStd, fmt='-');
dist_set_max = nodes/nproc
plt.gca().grid(which='major', axis='both', linestyle='--')


#plt.grid(color='grey', linestyle='--' );
if(which == 0):
    if printlegend:
        plt.legend(loc='upper right', numpoints=1,prop={'size':18},framealpha=0.6);

plt.savefig( outFName, bbox_inches='tight', dpi=400 );