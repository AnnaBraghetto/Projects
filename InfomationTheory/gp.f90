MODULE GP

    !Module containing the subroutines necessary to
    !perform the adiabatic quantum optimization, the
    !simulated annealing and  the simulated quantum 
    !annealing for the graph partitioning NP-problem.
    !All the necessary subroutines are then collected
    !in the subroutine called "GraphPartitioning" that 
    !performs all the wanted computations.

USE CHECKPOINT
USE INSTRUTILS
USE INITIALIZATION
USE HAMILTONIANS
USE ENERGY
USE QUANTUM
USE SIMULATEDANNEALING
USE SIMULATEDQUANTUMANNEALING


IMPLICIT NONE 

    !DEBUG
    logical :: debgp=.True.
    character(len=10) :: choice

    !MAIN PARAMETERS
    real*8 :: A, B
    integer :: N, E
    !PATH
    character(len=100) ::  Path

    !INITIALIZATION  PARAMETERS
    integer, dimension(:,:), allocatable :: ADJ !-> adjency matrix

    !QUANTUM ANNEALING PARAMETERS
    type(hamiltonian) :: hamilt !-> quantum hamiltonian
    real*8 :: dtime=0.0001, h0param=1d0
    integer, dimension(:), allocatable :: spinQA

    !SIMULATED ANNEALING PARAMETERS
    real*8 :: TemperatureSA

    !SIMULATED QUANTUM ANNEALING PARAMETERS
    real*8 :: TemperatureSQA
    integer :: M !-> number of the replicas

    !COMMON PARAMETERS OF THE PROCESS
    integer, dimension(:), allocatable :: MaxTime !-> max time of simulation
    real*8 :: E0, ComputedE0 !-> 
    !correct energy of the ground state, computed energy of the ground state
    integer :: Repeat !-> times that the simulated (quantum) annealing is repeated
    
    integer :: ii, jj !-> loops

    !Tuning parameters
    real*8, dimension(:), allocatable :: Ttuning
    integer, dimension(:), allocatable :: Mtuning
    real*8, dimension(:), allocatable :: EnergyTunSA
    real*8, dimension(:,:), allocatable :: EnergyTunSQA
    integer, dimension(2) :: loc
    integer :: timetun


CONTAINS 

SUBROUTINE GraphPartitioning()
    !INPUT PARAMETERS
    !none

    !DETAILS
    !The subroutine allows the user to choose if the performance
    !have to be analyzed or not and if a sample has to be analyzed or
    !not.
    !If the user wants to analyze the performance, the subroutine
    !"Performance" is called.
    !If the user wants to analyze a sample all the needed subroutines
    !are called: "InitializationParameters" permits to initialize the
    !parameters of the models; "TuningParamters" permits to tune the
    !specific simulation parameters; "QAResolution" permits to define 
    !true ground state energy and to perform the adiabatic quantum
    !optimization for less than 12 nodes while for more than 12 nodes
    !the true ground state energy is determined through the subroutine
    !"GSSQA"; "SAResolution" permits to perform the simulated annealing 
    !and "SQAResolution" permits to perform the simulated quantum 
    !annealing.

    print *, ''
    print *, 'GRAPH PARTITIONING'
    print *, 'Do you want to study the performance? (y/n)'
    read (*,*) choice

    if (trim(choice) .eq. 'y' .or. trim(choice) .eq. 'Y') then
        print *, 'Analyzing the performance...'
        !analysis on the performance
        call Performance()
    end if

    print *, ''
    print *, 'Do you want to analyze a sample? (y/n)'
    read (*,*) choice

    if (trim(choice) .eq. 'y' .or. trim(choice) .eq. 'Y') then
        !Initialize the parameters
        call InitializationParameters()

        !Tuning the parameters
        call TuningParameters(.True., .True.)

        !COMPARISON AMONG THE 3 METHODS FOR LESS THAN 12 NODES

        !QUANTUM ANNEALING (performed just for less than 12 nodes)
        if (N <= 12) then
            call QAResolution()
        else 
            call GSSQA()
            !print ground state energy
            print *, 'Ground state energy: ', E0
        end if

        !SIMULATED ANNEALING
        call SAResolution()

        !QUANTUM SIMULATED ANNEALING
        call SQAResolution()

        !Finally deallocate all
        deallocate(ADJ, MaxTime)
    end if

END SUBROUTINE



!---------------------------SUBROUTINES---------------------------!


SUBROUTINE InitializationParameters()
    !INPUT PARAMETERS
    !none

    !DETAILS
    !The subroutine permits to initialize the parameters of the 
    !problem from the command line: number of nodes N, number of
    !edges E, coefficients A and B of the hamiltonian.
    !The directory in which save the results is created and then
    !the adjency matrix is built.

    !OBJECT TO BUILD THE PATH
    character(len=100) :: Astr, Bstr, Nstr, Estr

    !READ TE MAIN PARAMETERS FROM THE COMMAND LINE
    print*, ''
    print *, 'Number of nodes:'
    read(*,*) N
    print *, 'Suggested number of edges:', &
                int(Floor(1.0*(N-1)*(N+2)/4))
    print *, 'Number of edges:'
    read(*,*) E
    print *, 'Coefficient A of the Hamiltonian:'
    read(*,*) A
    print *, 'Coefficient B of the Hamiltonian:'
    read(*,*) B

    !BUILD THE PATH
    write(Nstr,*) N
    write(Estr,*) E
    write(Astr,*) int(A)
    write(Bstr,*) int(B)
    path=trim('GP/N'//trim(adjustl(Nstr))//'E'// &
                trim(adjustl(Estr))//'A'//       &
                trim(adjustl(Astr))//'B'//       &
                trim(adjustl(Bstr))//'/')

    !CREATE THE OUTPUT LOCATION
    call PrepareLocation(path)

    !Allocate timing of the computation
    allocate(MaxTime(7))

    !Initialize the problem -> build the adjency matrix of the graph
    call ProblemInitialization(N, E, ADJ, &
                            trim(path)//'adj.txt', .True.)
        
END SUBROUTINE


!-----------------------TuningModelParameters---------------------!

SUBROUTINE TuningParameters(Verbose, Analysis)
    !INPUT PARAMETERS
    !Verbose: bool variable that permits to choose if print the 
    !         results on the screen -> logical
    !Analysis: bool variable that permits to choose if save the
    !          tuning values in external files -> logical

    !DETAILS
    !The subroutine tunes the parameters of the simulation by 
    !performing a grid search on the temperature for the simulated
    !annealing and on the temperature and number of replicas
    !for the simulated quantum annealing.
    !If Verbose is True, the results are shown on the scree.
    !If Analysis is True, the tuning values are saved in external
    !files.

    !input
    logical :: Verbose, Analysis

    if (Verbose) then
        print*, ''
        print*, 'Tuning the simulation parameters...'
    end if
    
    !allocate parameters and energies
    allocate(Ttuning(7), Mtuning(4))
    allocate(EnergyTunSA(size(Ttuning)), &
                EnergyTunSQA(size(Ttuning),size(Mtuning)))

    !averages
    Repeat = 30
    timetun = 1000

    !tuning SIMULATED ANNEALING
    Ttuning = (/0.1,0.4,0.7,1.0,4.0,7.0,10.0/)
    do ii=1,size(Ttuning)
        call Metropolis(N, ADJ, A, B, Ttuning(ii), Repeat, &
                        timetun, EnergyTunSA(ii),          &
                        'None', 'None', .False., .False.)
    end do
    TemperatureSA = Ttuning(minloc(EnergyTunSA, dim=1))

    !tuning SIMULATED QUANTUM ANNEALING
    Ttuning = (/0.01,0.04,0.07,0.1,0.4,0.7,1.0/)
    Mtuning = (/20,30,40,50/)
    do ii=1,size(Ttuning)
        do jj=1,size(Mtuning)
            call QuantumMetropolis(N,Mtuning(jj),ADJ,A,B,Ttuning(ii), &
                                Repeat, timetun, EnergyTunSQA(ii,jj), &
                                'None','None', .False., .False.)
        end do
    end do
    loc = minloc(EnergyTunSQA)
    TemperatureSQA = Ttuning(loc(1))
    M = Mtuning(loc(2))

    if (Analysis) then
        !save energy
        call Save(EnergyTunSA, trim(path)//'SAtuning.txt')
        call Save(EnergyTunSQA, trim(path)//'SQAtuning.txt')
    end if

    !deallocate all
    deallocate(Ttuning, Mtuning, EnergyTunSA, EnergyTunSQA)

    if (Verbose) then
        print *, 'Best initial temperature for Simulated', &
                    ' Annealing: ', TemperatureSA
        print *, 'Best temperature for Simulated Quantum Annealing: ', &
                    TemperatureSQA
        print *, 'Best number of replicas: ', M
    end if

END SUBROUTINE


!-------------------------------GSwithSQA--------------------------!

SUBROUTINE GSSQA()
    !INPUT PARAMETERS
    !none

    !DETAILS
    !The subroutine permits to compute the "true" ground state by 
    !selecting the minimum energy obtained by repeating the simulated
    !quantum annealing for 100 times.

    real*8, dimension(:), allocatable :: Energy

    Repeat = 100
    allocate(Energy(Repeat))

    do ii=1,Repeat
        call QuantumMetropolis(N, M, ADJ, A, B, TemperatureSQA, 1, &
                                10000, Energy(ii),    &
                                'None', 'None',       &
                                .False., .False.)
    end do
    !min energy
    E0 = minval(Energy)

    deallocate(Energy)

END SUBROUTINE


!------------------------------RESOLUTION-------------------------!

!----------------------------AQCResolution------------------------!

SUBROUTINE QAResolution()
    !INPUT PARAMETERS
    !none

    !DETAILS
    !The subroutine builds the quantum hamiltonian, computes the
    !true ground state by diagonalizing it and then performs the
    !adiabatic quantum computation for different final time.

    !Build the quantum hamiltonian
    call QuantumH(N, ADJ, A, B, hamilt%hP, &
                    trim(path)//'QAhamiltonian.txt', .True.)

    !Compute the quantum ground state
    call QuantumGroundState(hamilt%hP, E0)

    !print ground state energy
    print*, ''
    print *, 'Ground state energy: ', E0

    print*, ''
    print *, 'ADIABATIC QUANTUM COMPUTATION'

    !Perform the Quantum Annealing for different timing
    MaxTime = (/1e0,3*1e0,6*1e0,1e1,3*1e1,6*1e1,1e2/)
    do ii=1,size(MaxTime)-1
        call AQCAlgorithm(N, hamilt, MaxTime(ii), dtime, h0param, &
                        spinQA, trim(path)//'QAenergy.txt',       &
                        trim(path)//'QAprob.txt',                 &
                        trim(path)//'QAeigenvals.txt', .False.,   &
                        debgp)
        deallocate(spinQA)
        call ResidualEnergy(E0, hamilt%E_gs, hamilt%T_final, N, &
                        trim(path)//'QAresidual.txt')
    end do

    !Save the other results
    call AQCAlgorithm(N, hamilt, MaxTime(size(MaxTime)),    &
                    dtime, h0param, spinQA,                 &
                    trim(path)//'QAenergy.txt',             &     
                    trim(path)//'QAprob.txt',               &
                    trim(path)//'QAeigenvals.txt', .True.,  &
                    debgp)
    !Save configuration
    call Save(spinQA, trim(path)//'QAspin.txt')
    deallocate(spinQA)

    call ResidualEnergy(E0, hamilt%E_gs, hamilt%T_final, N, &
                    trim(path)//'QAresidual.txt')

    !deallocate the hamiltonian
    deallocate(hamilt%hP)

END SUBROUTINE


!----------------------------SAResolution-------------------------!


SUBROUTINE SAResolution()
    !INPUT PARAMETERS
    !none

    !DETAILS
    !The subroutine performs the simulated annealing for different
    !number of iterations by taking averages over 100 repetitions.

    !Perform the Simulated Annealing for different timing
    print*, ''
    print *, 'SIMULATED ANNEALING'

    Repeat = 100
    MaxTime = (/1e2,3*1e2,6*1e2,1e3,3*1e3,6*1e3,1e4/)
    do ii=1,size(MaxTime)
        call Metropolis(N, ADJ, A, B, TemperatureSA, Repeat,  &
                            MaxTime(ii), ComputedE0,          &
                            trim(path)//'SAenergy.txt',       &
                            trim(path)//'SAspin.txt', .True., &
                            debgp)
        call ResidualEnergy(E0, ComputedE0, MaxTime(ii), N,   &
                            trim(path)//'SAresidual.txt')
    end do

END SUBROUTINE



!----------------------------SQAResolution------------------------!

SUBROUTINE SQAResolution()
    !INPUT PARAMETERS
    !none

    !DETAILS
    !The subroutine performs the simulated quantum annealing for 
    !different number of iterations by taking averages over 100 
    !repetitions.

    !Perform the Simulated Quantum Annealing for different timing
    print*, ''
    print *, 'SIMULATED QUANTUM ANNEALING'

    Repeat = 100
    MaxTime = (/1e2,3*1e2,6*1e2,1e3,3*1e3,6*1e3,1e4/)
    do ii=1,size(MaxTime)
        call QuantumMetropolis(N,M,ADJ,A,B,TemperatureSQA,      &
                                Repeat,MaxTime(ii), ComputedE0, &
                                trim(path)//'SQAenergy.txt',    &
                                trim(path)//'SQAspin.txt',      &
                                .True., debgp)
        call ResidualEnergy(E0, ComputedE0, MaxTime(ii), N,     &
                                trim(path)//'SQAresidual.txt')
    end do  

END SUBROUTINE


!---------------------------Performance--------------------------!

SUBROUTINE Performance()
    !INPUT PARAMETERS
    !none

    !DETAILS
    !The subroutine analyzes the performance of the three methods for
    !the resultion of the graph partitioning. The three methods are 
    !applied for different values of nodes (with suggested number of 
    !edges) and the cpu time needed to perform the computation and 
    !the residual energy obtained are stored in external files. 
    !In particular, in the cpu time for the simulations, the time needed
    !to tune the parameters is included.

    real*8 :: tstart, tstop
    real*8, dimension(:,:), allocatable :: sa, sqa, aqc
    integer ::  nn, edges
    integer, dimension(:), allocatable :: nodes

    !allocate the nodes
    allocate(nodes(8))
    nodes = (/4,6,8,20,40,60,80,100/)

    !allocate time and res energy
    allocate(sa(size(nodes),3),sqa(size(nodes),3),aqc(3,3))
    sa = 0
    sqa = 0
    aqc = 0

    !tuning parameters
    allocate(Ttuning(7), Mtuning(4))
    allocate(EnergyTunSA(size(Ttuning)), &
                EnergyTunSQA(size(Ttuning),size(Mtuning)))

    !averages for the tuning
    Repeat = 30
    timetun = 1000

    !perform the computation
    do nn=1,size(nodes)
        
        call debug(debgp,'Node', nodes(nn))
        call debug(debgp,'-----------------')
        !edges
        edges = int(Floor(1.0*(nodes(nn)-1)*(nodes(nn)+2)/4))
        N=nodes(nn)
        E=edges
        A=1d0
        B=1d0

        !initialize the problem 
        call ProblemInitialization(N,E,ADJ,'None',.False.)
        
        
        !tuning SIMULATED ANNEALING
        
        call cpu_time(tstart)
        Ttuning = (/0.1,0.4,0.7,1.0,4.0,7.0,10.0/)
        do ii=1,size(Ttuning)
            call Metropolis(N, ADJ, A, B, Ttuning(ii), Repeat, &
                            timetun, EnergyTunSA(ii),          &
                            'None', 'None', .False., .False.)
        end do
        TemperatureSA = Ttuning(minloc(EnergyTunSA, dim=1))
        call cpu_time(tstop)
        sa(nn,2) = abs(tstart-tstop)
        call debug(debgp, 'Temperature for SA', TemperatureSA)


        !tuning SIMULATED QUANTUM ANNEALING

        call cpu_time(tstart)
        Ttuning = (/0.01,0.04,0.07,0.1,0.4,0.7,1.0/)
        Mtuning = (/20,30,40,50/)
        do ii=1,size(Ttuning)
            do jj=1,size(Mtuning)
                call QuantumMetropolis(N, Mtuning(jj), ADJ, &
                                    A, B, Ttuning(ii),      &
                                    Repeat, timetun,        &
                                     EnergyTunSQA(ii,jj),   &
                                    'None','None', .False., .False.)
            end do
        end do
        loc = minloc(EnergyTunSQA)
        TemperatureSQA = Ttuning(loc(1))
        M = Mtuning(loc(2))
        call cpu_time(tstop)
        sqa(nn,2) = abs(tstart-tstop)   
        call debug(debgp, 'Temperature for SQA', TemperatureSQA)
        call debug(debgp, 'M for SQA', M)


        !------------Computing Ground State----------|

        !AQC
        if (N<=12) then
            !start the time computation
            call cpu_time(tstart)
            !Build the quantum hamiltonian
            call QuantumH(N, ADJ, A, B, hamilt%hP, &
                            'None', .False.)
            call AQCAlgorithm(N, hamilt, 100,           & 
                                dtime, h0param, spinQA, &
                                'None','None','None',   &
                                .False., .False.)
            !stop time computation
            call cpu_time(tstop)
            !Compute the quantum ground state
            call QuantumGroundState(hamilt%hP, E0)
            !deallocate spin and HP
            deallocate(spinQA, hamilt%hP)
            !save results
            aqc(nn,1) = N
            aqc(nn,2) = abs(tstart-tstop)
            aqc(nn,3) = abs(E0-hamilt%E_gs)/N
            call debug(debgp,'AQC energy',hamilt%E_gs)
        else
            call GSSQA()
        end if

        !SIMULATED ANNEALING
        !start time computation
        call cpu_time(tstart)
        call Metropolis(N, ADJ, A, B, TemperatureSA, &
                        100, 10000, ComputedE0,      &
                        'None', 'None',              &
                        .False., .False.)
        !stop time computation
        call cpu_time(tstop)
        !save results
        sa(nn,1) = N
        sa(nn,2) = sa(nn,2) +abs(tstart-tstop)/100
        sa(nn,3) = abs(E0-ComputedE0)/N
        call debug(debgp,'SA energy',ComputedE0)

        !SIMULATED QUANTUM ANNEALING
        !start time computation
        call cpu_time(tstart)
        call QuantumMetropolis(N, M, ADJ, A, B,      &
                                TemperatureSQA, 100, &
                                10000,  ComputedE0,  &
                                'None', 'None',      &
                                .False., .False.)
        !stop time computation
        call cpu_time(tstop)
        !save results
        sqa(nn,1) = N
        sqa(nn,2) = sqa(nn,2) +abs(tstart-tstop)/100
        sqa(nn,3) = abs(E0-ComputedE0)/N
        call debug(debgp,'SQA energy',ComputedE0)
        
        !deallocate adjency matrix
        deallocate(ADJ)  
    end do

    call Save(sa, 'GP/SAperf.txt')
    call Save(sqa, 'GP/SQAperf.txt')
    call Save(aqc, 'GP/QAperf.txt')

    !deallocate all
    deallocate(nodes, sa, sqa, aqc)
    deallocate(Ttuning, Mtuning, EnergyTunSA, EnergyTunSQA)

END SUBROUTINE

END MODULE