#LAUNCH THE FORTRAN PROGRAM
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

s = 15
m = 18
l = 20

plt.rc('font', size=s)          #controls default text sizes
plt.rc('axes', labelsize=m)     #fontsize of the x and y labels
plt.rc('xtick', labelsize=s)    #fontsize of the tick labels
plt.rc('ytick', labelsize=s)    #fontsize of the tick labels
plt.rc('legend', fontsize=s)    #legend fontsize
plt.rc('figure', titlesize=l)

def Performance(path, xlab):
    #INPUT PARAMETERS
    #path: path of the directory in which find the files
    #       -> string
    #xlab: label for x axis -> string

    #DETAILS
    #The function permits to plot the performance of the
    #methods for an arbitrary model: cpu time and redidual
    #energy of the the three algorithms are compare in two
    #different graph that are also saved in the directory
    #defined by the input variable "graph"
    #If the in the directory are also found the performance
    #for aqc with linear system resolution, the results
    #for aqc with the two type of resolution are compared
    #in two plots, one for cpu time and one for residual
    #energy and saved in the directory defined by the input
    #variable path.

    # Load the file containing the performance results
    QA = np.loadtxt(path+'QAperf.txt').reshape(-1,3)
    SA = np.loadtxt(path+'SAperf.txt')
    SQA = np.loadtxt(path+'SQAperf.txt')

    # Computational time
    fig = plt.figure(figsize=(6,5))
    plt.plot(QA[:,0],QA[:,1], '*--', label='AQO') 
    plt.plot(SA[:,0],SA[:,1], 'o-', label='SA') 
    plt.plot(SQA[:,0],SQA[:,1], '^-.', label='SQA') 
    plt.xlabel(xlab)
    plt.ylabel('Time [s]')
    plt.yscale('log')
    plt.legend(shadow=True)
    ax = plt.axes()
    ax.xaxis.grid(True, ls='dashed',which="both")
    ax.yaxis.grid(True, ls='dashed',which="major")
    plt.tight_layout()
    fig.savefig(path+'time.pdf')
    plt.close()

    # Residual energy
    fig = plt.figure(figsize=(6,5))
    plt.plot(QA[:,0],QA[:,2], '*--', label='AQO') 
    plt.plot(SA[:,0],SA[:,2], 'o-', label='SA') 
    plt.plot(SQA[:,0],SQA[:,2], '^-.', label='SQA') 
    plt.xlabel(xlab)
    plt.ylabel('Residual energy')
    plt.legend(shadow=True)
    ax = plt.axes()
    ax.xaxis.grid(True, ls='dashed',which="both")
    ax.yaxis.grid(True, ls='dashed',which="major")
    #insert the zoom
    axins = zoomed_inset_axes(ax, 4, loc=10)
    axins.plot(SA[:,0],SA[:,2],'o-',color='C1',lw=1,markersize=4) 
    axins.plot(SQA[:,0],SQA[:,2],'^-.',color='C2',lw=1,markersize=4) 
    axins.plot(QA[:,0],QA[:,2],'*--',color='C0',lw=1,markersize=4) 
    #limits 
    if QA.shape[0] < 3: plus = 0.25
    else: plus = 1
    x1, x2 = np.min(QA[:,0])-plus, np.max(QA[:,0])+plus
    y1, y2 = -np.max(SA[:,2])/100, np.max(QA[:,2])+np.max(SA[:,2])/100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.xaxis.grid(True, ls='dashed',which="both")
    axins.yaxis.grid(True, ls='dashed',which="major")
    axins.tick_params(labelsize=5)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='0.8', alpha=0.8)
    plt.tight_layout()
    fig.savefig(path+'resen.pdf')
    plt.close()

    #Try to plot the results also for aqc with linear 
    #resolution
    try:
        QALIN = np.loadtxt(path+'QAperfLin.txt').reshape(-1,3)
        # Computational time
        fig = plt.figure(figsize=(6,5))
        plt.plot(QA[:,0],QA[:,1], '*--', label='AQO') 
        plt.plot(QALIN[:,0],QALIN[:,1], '*--', 
                    color='C3', label='AQO Lin System') 
        plt.xlabel(xlab)
        plt.ylabel('Time [s]')
        plt.yscale('log')
        plt.legend(shadow=True)
        ax = plt.axes()
        ax.xaxis.grid(True, ls='dashed',which="both")
        ax.yaxis.grid(True, ls='dashed',which="major")
        plt.tight_layout()
        fig.savefig(path+'time-AQO.pdf')
        plt.close()

        # Residual energy
        fig = plt.figure(figsize=(6,5))
        plt.plot(QA[:,0],QA[:,2], '*--', label='AQO') 
        plt.plot(QALIN[:,0],QALIN[:,2], '*--', 
                    color='C3', label='AQO Lin System') 
        plt.xlabel(xlab)
        plt.ylabel('Residual energy')
        #plt.yscale('log')
        plt.legend(shadow=True)
        ax = plt.axes()
        ax.xaxis.grid(True, ls='dashed',which="both")
        ax.yaxis.grid(True, ls='dashed',which="major")
        plt.tight_layout()
        fig.savefig(path+'resen-AQO.pdf')
        plt.close()

    except:
        return



def Degeneration(eigvals):
    #INPUT PARAMETERS
    #eigavals: array containing the eigenvalues of the 
    #           hamiltoninan of the problem -> array

    #DETAILS
    #The function computes the degenarion of the 
    #ground state by comparing the first eigenvalues
    #with the others.

    deg = -1
    gs = eigvals[0]
    for eig in eigvals:
        if abs(eig - gs) < 0.0001: deg +=1
    return deg+1



def AQCResults(path):
    #INPUT PARAMETERS
    #path: path of the directory in which find the files
    #       -> string

    #DETAILS
    #The function plots the results of the adiabatic
    #quantum optimization if they are present in the
    #directory defined by the input variable.
    #If the results are present the following plots
    #are generated: energy at time t of the evolution
    #vs time t; residual energy at the final time vs
    #final time; difference between the ground state
    #energy and first excited state at time t vs
    #time t; eigenvalues at time t vs time t; probability
    #that the system is in the ground state at time t
    #vs time t.

    try:

        QAEnergy = np.loadtxt(path+'QAenergy.txt')
        QAResEnergy = np.loadtxt(path+'QAresidual.txt')
        QAEigvals = np.loadtxt(path+'QAeigenvals.txt')
        QAProb = np.loadtxt(path+'QAprob.txt')

        #Time
        time = np.arange(0,1001000,1000)

        #Energy plot
        fig = plt.figure(figsize=(6,5))
        plt.plot(time, QAEnergy, '--', label='AQO') 
        plt.xlabel('Time [iterations]')
        plt.ylabel('Energy')
        plt.legend(shadow=True)
        plt.grid(True, ls='dashed',which="both")
        plt.tight_layout()
        fig.savefig(path+'Plot/AQCenergy.pdf')
        plt.close()

        #Residual energy plot
        fig = plt.figure(figsize=(6,5))
        plt.plot(QAResEnergy[:,0],QAResEnergy[:,1],'*--',label='AQO') 
        plt.ylabel('Residual Energy')
        plt.xlabel('Final [iterations]')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(shadow=True)
        ax = plt.axes()
        ax.xaxis.grid(True, ls='dashed',which="both")
        ax.yaxis.grid(True, ls='dashed',which="major")
        plt.tight_layout()
        fig.savefig(path+'Plot/AQCresidual.pdf')
        plt.close()

        #Difference gs and first excited   
        deg = Degeneration(QAEigvals[-1,:]) 
        print('The degeneracy is',deg)  
        fig = plt.figure(figsize=(6,5))
        plt.plot(time,abs(QAEigvals[:,0]-QAEigvals[:,deg]),'-')
        plt.xlabel('Time [iterations]')
        plt.ylabel(r'$\Delta E$')
        plt.grid(True, ls='dashed',which="both")
        plt.tight_layout()
        fig.savefig(path+'Plot/AQCdeltaE.pdf')
        plt.close()
        print('Best final time: 1/DeltaE^2 =', 
                (1/np.min(abs(QAEigvals[:,0]-QAEigvals[:,deg])))**2)

        #Eigenvalues
        fig = plt.figure(figsize=(6,5))
        for eig in range(deg+1):
            label =  r'$E_{%i}$' %(eig+1)
            plt.plot(time,QAEigvals[:,eig], '-',label=label)
        plt.xlabel('Time [iterations]')
        plt.ylabel('Eigenvalues')
        plt.legend(shadow=True)
        plt.grid(True, ls='dashed',which="both")
        plt.tight_layout()
        fig.savefig(path+'Plot/AQCeigval.pdf')
        plt.close()

        #Prob
        fig = plt.figure(figsize=(6,5))
        plt.plot(time,QAProb, '-') 
        plt.xlabel('Time [iterations]')
        plt.ylabel('Probability')
        plt.grid(True, ls='dashed',which="both")
        plt.tight_layout()
        fig.savefig(path+'Plot/AQCprob.pdf')
        plt.close()

    except:
        return

def CompareSim(path):
    #INPUT PARAMETERS
    #path: path of the directory in which find the files
    #       -> string

    #DETAILS
    #The function plots the results of the simulations
    #are shown in the following plots: results of the 
    #tuning and results of the whole simulation with 
    #the best parameters.
    #Results of the tuning: estimated ground state
    #energy vs temperature for simulated annealing
    #and estimated ground state energy vs temperature
    #for different number of Trotter replicas for 
    #simulated quantum annealing.
    #Results of the simulation with the best parameters
    #for both simulated annealing and simulated annealing
    #compared in the same plot: energy at iteration i 
    #vs iteration i; residual energy at the final iteration 
    #vs final iteration.

    #Tuning parameters
    SATun = np.loadtxt(path+'SAtuning.txt')
    SQATun = np.loadtxt(path+'SQAtuning.txt')

    #SA
    fig = plt.figure(figsize=(6,5))
    Temperature = [0.1,0.4,0.7,1,4,7,10]
    plt.plot(Temperature,SATun,'o-') 
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.xscale('log')
    ax = plt.axes()
    ax.xaxis.grid(True, ls='dashed',which="both")
    ax.yaxis.grid(True, ls='dashed',which="major")
    plt.tight_layout()
    fig.savefig(path+'Plot/SAtunining.pdf')
    plt.close()

    #SQA
    fig = plt.figure(figsize=(6,5))
    Temperature = [0.01,0.04,0.07,0.1,0.4,0.7,1.0]
    plt.plot(Temperature,SQATun[:,0], '^-.', label='M=20') 
    plt.plot(Temperature,SQATun[:,1], '^-.', label='M=30') 
    plt.plot(Temperature,SQATun[:,2], '^-.', label='M=40') 
    plt.plot(Temperature,SQATun[:,3], '^-.', label='M=50') 
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.xscale('log')
    plt.legend(shadow=True)
    ax = plt.axes()
    ax.xaxis.grid(True, ls='dashed',which="both")
    ax.yaxis.grid(True, ls='dashed',which="major")
    plt.tight_layout()
    fig.savefig(path+'Plot/SQAtunining.pdf')
    plt.close()


    #Load the files containg the energies

    #AQC energy

    SAEnergy = np.loadtxt(path+'SAenergy.txt')
    SQAEnergy = np.loadtxt(path+'SQAenergy.txt')
    SAResEnergy = np.loadtxt(path+'SAresidual.txt')
    SQAResEnergy = np.loadtxt(path+'SQAresidual.txt')

    #Energy plot
    fig = plt.figure(figsize=(6,5))
    plt.plot(SAEnergy, '-', color='C1', label='SA') 
    plt.plot(SQAEnergy, '-.', color='C2', label='SQA')
    plt.xlabel('Time [MCS]')
    plt.ylabel('Energy')
    plt.legend(shadow=True)
    plt.grid(True, ls='dashed',which="both")
    plt.tight_layout()
    fig.savefig(path+'Plot/SIMenergy.pdf')
    plt.close()

    #Residual energy plot
    fig = plt.figure(figsize=(6,5))
    plt.plot(SAResEnergy[:,0],SAResEnergy[:,1],
                'o-',color='C1',label='SA') 
    plt.plot(SQAResEnergy[:,0],SQAResEnergy[:,1],
                '^-.',color='C2',label='SQA')   
    plt.ylabel('Residual Energy')
    plt.xlabel('Final [MCS]')
    plt.xscale('log')
    #plt.yscale('log')
    plt.legend(shadow=True)
    ax = plt.axes()
    ax.xaxis.grid(True, ls='dashed',which="both")
    ax.yaxis.grid(True, ls='dashed',which="major")
    plt.tight_layout()
    fig.savefig(path+'Plot/SIMresidual.pdf')
    plt.close()


def Graph(path, spin, node_colors, show_edges, position=None):
    #INPUT PARAMETERS
    #path: path of the directory in which find the files
    #       -> string
    #spin: array containing the spin configuration -> array
    #node_colors: colors of the nodes of the graph -> array
    #show_edges: bool variable that permits to choose if
    #            shown the weight of the edges in the graph
    #            -> boolean 
    #position: position of the nodes in the graph 
    #          -> None (not defined)
    #          -> string
    #          -> array of tuples

    #DETAILS
    #The function permits to show the graph considering a sample
    #configuration of an arbitary models with an arbitrary 
    #algorithm.

    adj = np.loadtxt(path+'adj.txt')
    G = nx.from_numpy_matrix(np.matrix(adj))

    # Position
    if position is None:
        position = nx.spring_layout(G)
        # Case of TS with three cities
        if adj.shape[1]==3:
            position = {}
            position[0] = (1,1)
            position[1] = (0,0)
            position[2] = (0,2)
    elif position == 'Lattice':
        number_nodes =int(np.sqrt(adj.shape[0]))
        position = {}
        n=0
        for i in range(number_nodes):
            for j in range(number_nodes):  
                position[n] = (i,j)
                n +=1

    # Labels
    labels = {}
    number_nodes=len(spin)
    for i in range(len(spin)):
        labels[i] = int(spin[i])
    
    if show_edges:
        fig = plt.figure(figsize=(6,5))
        edges, weights = zip(*nx.get_edge_attributes(G, 
                                'weight').items())
        cmap = plt.cm.pink 
        nx.draw_networkx(G, pos=position, with_labels=True, 
                            labels=labels, node_color='darkcyan', 
                            node_size=400, edgelist=edges, 
                            edge_color=weights, edge_vmin=min(weights), 
                            edge_vmax=1,edge_cmap=cmap,
                            font_color='black')
        # Colourbar                  
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                    norm=plt.Normalize(
                                        vmin=0, vmax=1))
        sm._A = []
        plt.colorbar(sm)
    else:
        fig = plt.figure(figsize=(6,5))
        cmap = plt.cm.jet
        if type(node_colors[0]) is str:
            font_color = 'black'       
        else:
            font_color = 'white'

        nx.draw_networkx(G,pos=position,with_labels=True,labels=labels,
                            node_color=node_colors,node_size=400,
                            cmap=cmap,font_color=font_color)

    return fig, position


def IsingModel():
    #INPUT PARAMETERS
    #none

    #DETAILS
    #The function permits to show the results obtained
    #for the ising model.
    #At first, the perfomance of the algorithm are shown
    #by calling the function "Performance".
    #Then all the directories created by the Fortran 
    #program containing the results of an arbitrary sample
    #are defined and for each directory (for each sample),
    #a new directory "Plot" in which save the plots is built
    #and the results obtained through the three algorithms
    #are saved by calling the functions "AQCResults" and 
    #"CompareSim".
    #Finally the graph for the considered sample is built
    #and saved for all the methods by setting the parameter
    #"position" to "Lattice".

    #Plot performance
    Performance('IM/', 'Side of Lattice')

    directories = []
    for d in os.listdir('IM/'):
        if '.' not in d:
            directories.append(d)

    for d in directories:
        print('Configuration:',d)
        path = 'IM/'+d+'/'

        #Create the directory to save the plots
        os.makedirs(path+'Plot/', exist_ok=True)
        
        #Results for AQC
        AQCResults(path)
        #Compare Sim: Energy Plot and Graph Ploy
        CompareSim(path)

        #AQC if present
        try:
            spin = np.loadtxt(path+'QAspin.txt')
            node_colors = spin
            fig, _ = Graph(path, spin, node_colors, False, 'Lattice')
            plt.tight_layout()
            fig.savefig(path+'Plot/AQCsample.pdf')
            plt.close()
        except:
            'AQC do not perform for this configuration'

        #SA
        spin = np.loadtxt(path+'SAspin.txt')
        node_colors = spin
        fig, _ = Graph(path, spin, node_colors, False, 'Lattice')
        plt.tight_layout()
        fig.savefig(path+'Plot/SAsample.pdf')
        plt.close()

        #SQA
        spin = np.loadtxt(path+'SQAspin.txt')
        spin = spin[0,:]
        node_colors = spin
        fig, _ = Graph(path, spin, node_colors, False, 'Lattice')
        plt.tight_layout()
        fig.savefig(path+'Plot/SQAsample.pdf')
        plt.close()


def GraphPartitioning():
    #INPUT PARAMETERS
    #none

    #DETAILS
    #The function permits to show the results obtained
    #for the graph partitioning NP-problem.
    #At first, the perfomance of the algorithm are shown
    #by calling the function "Performance".
    #Then all the directories created by the Fortran 
    #program containing the results of an arbitrary sample
    #are defined and for each directory (for each sample),
    #a new directory "Plot" in which save the plots is built
    #and the results obtained through the three algorithms
    #are saved by calling the functions "AQCResults" and 
    #"CompareSim".
    #Finally the graph for the considered sample is built
    #and saved for all the methods by using the same position.
    
    #Plot performance
    Performance('GP/', 'Nodes')

    directories = []
    for d in os.listdir('GP/'):
        if '.' not in d:
            directories.append(d)

    for d in directories:
        print('Configuration:',d)
        path = 'GP/'+d+'/'

        #Create the directory to save the plots
        os.makedirs(path+'Plot/', exist_ok=True)
        
        #Results for AQC
        AQCResults(path)
        #Compare Sim: Energy Plot and Graph Ploy
        CompareSim(path)

        #SA
        spin = np.loadtxt(path+'SAspin.txt')
        node_colors = spin
        fig, pos = Graph(path, spin, node_colors, False)
        plt.tight_layout()
        fig.savefig(path+'Plot/SAsample.pdf')
        plt.close()

        #SQA
        spin = np.loadtxt(path+'SQAspin.txt')
        spin = spin[0,:]
        node_colors = spin
        fig, _ = Graph(path, spin, node_colors, False, pos)
        plt.tight_layout()
        fig.savefig(path+'Plot/SQAsample.pdf')
        plt.close()

        #AQC if present
        try:
            spin = np.loadtxt(path+'QAspin.txt')
            node_colors = spin
            fig, _ = Graph(path, spin, node_colors, False, pos)
            plt.tight_layout()
            fig.savefig(path+'Plot/AQCsample.pdf')
            plt.close()
        except:
            'AQC do not perform for this configuration'


def VertexCover():
    #INPUT PARAMETERS
    #none

    #DETAILS
    #The function permits to show the results obtained
    #for the vertex cover NP-problem.
    #At first, the perfomance of the algorithm are shown
    #by calling the function "Performance".
    #Then all the directories created by the Fortran 
    #program containing the results of an arbitrary sample
    #are defined and for each directory (for each sample),
    #a new directory "Plot" in which save the plots is built
    #and the results obtained through the three algorithms
    #are saved by calling the functions "AQCResults" and 
    #"CompareSim".
    #Finally the graph for the considered sample is built
    #and saved for all the methods by using the same position.
    
    #Plot performance
    Performance('VC/', 'Nodes')

    directories = []
    for d in os.listdir('VC/'):
        if '.' not in d:
            directories.append(d) 

    for d in directories:
        print('Configuration:',d)
        path = 'VC/'+d+'/'

        #Create the directory to save the plots
        os.makedirs(path+'Plot/', exist_ok=True)
        
        #Results for AQC
        AQCResults(path)
        #Compare Sim: Energy Plot and Graph Ploy
        CompareSim(path)

        #SA
        spin = np.loadtxt(path+'SAspin.txt')
        node_colors = []
        for node in spin:
            if node == 1:
                node_colors.append('goldenrod')
            else: 
                node_colors.append('lightgray') 
        fig, pos = Graph(path, spin, node_colors, False)
        plt.tight_layout()
        fig.savefig(path+'Plot/SAsample.pdf')
        plt.close()
        
        #SQA
        spin = np.loadtxt(path+'SQAspin.txt')
        spin = spin[0,:]
        node_colors = []
        for node in spin:
            if node == 1:
                node_colors.append('goldenrod')
            else: 
                node_colors.append('lightgray') 
        fig, _ = Graph(path, spin, node_colors, False, pos)
        plt.tight_layout()
        fig.savefig(path+'Plot/SQAsample.pdf')
        plt.close()

        #AQC if present
        try:
            spin = np.loadtxt(path+'QAspin.txt')
            node_colors = []
            for node in spin:
                if node == 1:
                    node_colors.append('goldenrod')
                else: 
                    node_colors.append('lightgray') 
            fig, _ = Graph(path, spin, node_colors, False, pos)
            plt.tight_layout()
            fig.savefig(path+'Plot/AQCsample.pdf')
            plt.close()
        except:
            'AQC do not perform for this configuration'



def TravelingSalesman():
    #INPUT PARAMETERS
    #none

    #DETAILS
    #The function permits to show the results obtained
    #for the traveling salesman NP-problem.
    #At first, the perfomance of the algorithm are shown
    #by calling the function "Performance".
    #Then all the directories created by the Fortran 
    #program containing the results of an arbitrary sample
    #are defined and for each directory (for each sample),
    #a new directory "Plot" in which save the plots is built
    #and the results obtained through the three algorithms
    #are saved by calling the functions "AQCResults" and 
    #"CompareSim".
    #Finally the graph for the considered sample is built
    #and saved for all the methods by using the same position.

    #Plot performance
    Performance('TS/', 'Cities')

    directories = []
    for d in os.listdir('TS/'):
        if '.' not in d:
            directories.append(d)


    for d in directories:
        print('Configuration:',d)
        path = 'TS/'+d+'/'

        #Create the directory to save the plots
        os.makedirs(path+'Plot/', exist_ok=True)

        #Results for AQC
        AQCResults(path)
        #Compare Sim: Energy Plot and Graph Ploy
        CompareSim(path)

        #SA
        spin = np.loadtxt(path+'SAspin.txt')
        node_colors = spin
        fig, pos = Graph(path, spin, node_colors, True)
        plt.tight_layout()
        fig.savefig(path+'Plot/SAsample.pdf')
        plt.close()

        #SQA
        spin = np.loadtxt(path+'SQAspin.txt')
        spin = spin[0,:]
        node_colors = spin
        fig, _ = Graph(path, spin, node_colors, True, pos)
        plt.tight_layout()
        fig.savefig(path+'Plot/SQAsample.pdf')
        plt.close()

        #AQC if present
        try:
            spin = np.loadtxt(path+'QAspin.txt')
            node_colors = spin
            fig, _ = Graph(path, spin, node_colors, True, pos)
            plt.tight_layout()
            fig.savefig(path+'Plot/AQCsample.pdf')
            plt.close()
        except:
            'AQC do not perform for this configuration'




#--------------------------RUN------------------------#
if __name__ == "__main__":

    #The user can choose if compile and run the fortran
    #program or not.
    #The user can choose if create the plot of the results
    #for the ising model or not.
    #The user can choose if create the plot of the results
    #for the graph partitioning NP-problem or not.
    #The user can choose if create the plot of the results
    #for the vertex cover NP-problem or not.
    #The user can choose if create the plot of the results
    #for the traveling salesman NP-problem or not.


    #if the user want to execute the program
    choice = input('Do you want to re-execute the FORTRAN '+
                    'program? (y/n)  ')
    #the default is false
    if choice == 'y' or choice == 'Y': run=True
    else: run=False
    if run:

        #remove the exe.out
        command='rm exe.out'
        os.system(command)

        print('Compiling...')

        #compile the program
        # -fcheck=bounds <- to check malloc error
        os.system("gfortran checkpoint.f90 utils.f90\
                    initialization.f90 hamiltonians.f90 energy.f90\
                    debug.f90 quantum.f90 sa.f90 sqa.f90\
                    im.f90 gp.f90 ts.f90 vc.f90 main.f90\
                    -o exe.out -llapack -O3 -fcheck=all")

        print('Executing...')
        #run
        command = './exe.out'
        os.system(command)


    #if the user want to plot the results

    #Ising Model
    choice = input('Do you want to show the results for '+
                    'Ising Model? (y/n)  ')
    #the default is false
    if choice == 'y' or choice == 'Y': run=True
    else: run=False
    if run:
        IsingModel()

    #Graph Partitioning
    choice = input('Do you want to show the results for ' +
                    'Graph Partitioning? (y/n)  ')
    #the default is false
    if choice == 'y' or choice == 'Y': run=True
    else: run=False
    if run:
       GraphPartitioning()


    #Vertex Cover
    choice = input('Do you want to show the results for ' +
                    'Vertex Cover? (y/n)  ')
    #the default is false
    if choice == 'y' or choice == 'Y': run=True
    else: run=False
    if run:
       VertexCover()


    #Traveling Salesman
    choice = input('Do you want to show the results for '+
                    'Traveling Salesman? (y/n)  ')
    #the default is false
    if choice == 'y' or choice == 'Y': run=True
    else: run=False
    if run:
       TravelingSalesman()