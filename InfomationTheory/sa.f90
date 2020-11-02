MODULE SIMULATEDANNEALING

    !Module containing the subroutines necessary to
    !perform the simulated annealing for all the 
    !considered problems. 
    !The simulated annealing is then performed by
    !means of all the the needed subroutines for 
    !each problem. 

USE CHECKPOINT
USE HAMILTONIANS
USE INSTRUTILS
USE ENERGY

IMPLICIT NONE

logical :: debsa = .False.

!Interface for the simulated annealing

INTERFACE Metropolis
    MODULE PROCEDURE IsingModelMetropolis, &
                        GraphPartitioningMetropolis, &
                        VertexCoverMetropolis, &   
                        TravelingSalesmanMetropolis
END INTERFACE

CONTAINS 


SUBROUTINE SpinInitialization(Spin)
    !INPUT PARAMETERS
    !Spin: array containing the spins 
    !       -> integer, dimension(:)

    !DETAILS
    !The subroutines initializes randomly the
    !spins with +1 and -1.

    !input
    integer, dimension(:) :: Spin 
    !random number
    real*8 :: u
    !loops
    integer :: ii

    Spin = 0
    !initialize randomly the spins with +1 and -1
    do ii=1,size(Spin)
        call random_number(u)
        Spin(ii) = 2*FLOOR(2D0*u)-1
    end do

END SUBROUTINE


SUBROUTINE Tt(Temperature, DeltaT)
    !INPUT PARAMETERS
    !Temperature: temperature at a current time 
    !               -> real*8
    !DeltaT: variation of temperature in a unit of time
    !          -> real*8

    !DETAILS
    !The subroutine updates the temperature by subtracting
    !DeltaT from Temperature.
    !It is called at each time step to reduce the initial 
    !temperature to 0 linearly.

    !input
    real*8 :: Temperature, DeltaT

    !check on temperature
    if (Temperature<=0) then
        Temperature= DeltaT
    end if
    !decrease temperature
    Temperature = Temperature - DeltaT

END SUBROUTINE


SUBROUTINE AcceptOrRefuse(DeltaE, Temperature, Sign)
    !INPUT PARAMETERS
    !DeltaE: variation of the energy among two configurations
    !       -> real*8
    !Temperature: temperature of the system -> real*8
    !Sign: result of the choice -> integer

    !DETAILS
    !The subroutine accepts the configuration if the energy
    !decreases: DeltaE<=0. Otherwise, the move is accepted with
    !probability e^(-DeltaE/Temperature).
    !If the move is accepted the sign is set to -1 (actually flip
    !the spin), otherwise it is set to +1 (not flip the spin).

    !input
    real*8 :: DeltaE, Temperature
    integer :: Sign
    
    !Variables to help the computation: random number, probability
    real*8 :: u, probability

    !if the energy decreases accept the move
    if (DeltaE<=0) then 
        !flip 
        Sign = -1
        continue
    !otherwise accept it with probability p
    else
        !compute the probability
        probability = exp(-DeltaE/(Temperature))
        
        call random_number(u)
        if (probability>u) then
            call debug(debsa, 'Flip with probability:',probability)
            !flip
            Sign = -1
        else
            !otherwise refuse it and do not do anything
            Sign = +1
        end if     

    end if

END SUBROUTINE


SUBROUTINE Results(AverageEnergy, Repeat, ComputedE0, Spin, &
                    FILEEN, FILESP, Analysis)
    !INPUT PARAMETERS
    !AverageEnergy: array containing the sum of the energy of 
    !               the systems of different repetitions  at 
    !               different times -> real*8, dimension(:)
    !Repeat: number of repetitions -> integer
    !ComputedE0: estimation of the ground state -> real*8
    !Spin: configuration of the last repetition 
    !       -> integer, dimension(:)
    !FILEEN: file in which save the evolution of the 
    !        average energy -> character(len=*)
    !FILESP: file in which save the spin configuration 
    !       -> character(len=*)
    !Analysis: bool variables that permits to choose if 
    !          save or not the results in the external files
    !          -> logical

    !DETAILS
    !The subroutine computes the average energy of the system
    !at different times, estimates the ground state energy by
    !means of GSEstimation subroutines contained in the module
    !Energy and then saves the average energy and the spin
    !configuration in the external files if Analysis is set to
    !True.

    !input
    real*8, dimension(:) :: AverageEnergy 
    real*8 :: ComputedE0
    integer :: Repeat 
    integer, dimension(:) :: Spin
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis

    !determine the estimation of the energy
    AverageEnergy = AverageEnergy/(Repeat)
    call GSEstimation(AverageEnergy,ComputedE0)


    if (Analysis) then
        !save energy evolution
        call Save(AverageEnergy, FILEEN)
        !save final spin configuration
        call Save(Spin, FILESP)
    end if
    
END SUBROUTINE


!------------------------------MODELS----------------------------!

!----------------------------IsingModel--------------------------!

SUBROUTINE IsingModelMetropolis(N, Jij, Temperature, Repeat, &
                                MaxTime, ComputedE0, FILEEN, &
                                FILESP, Analysis, Verbose)
    !INPUT PARAMETERS
    !N: number of nodes/spins -> integer
    !Jij: adjency matrix of the lattice describing the strenght
    !     of the interaction between nearest neighbours 
    !     -> real*8, dimension(:,:)
    !Temperature: initial temperature of the systems -> real*8
    !Repeat: number of repetitions to perform averages -> integer
    !MaxTime: number of iterations -> integer
    !ComputedE0: estimation of the ground state -> real*8
    !FILEEN: file in which save the evolution of the 
    !        average energy -> character(len=*)
    !FILESP: file in which save the spin configuration 
    !       -> character(len=*)
    !Analysis: bool variables that permits to choose if 
    !          save or not the results in the external files
    !          -> logical 
    !Verbose: bool variables that permits to choose if print or
    !         not results on screen -> logical 

    !DETAILS
    !The subroutine performs the simulated annealing for the Ising
    !model.
    !The algorithm is perform "Repeat" times for "MaxTime" iterations.
    !At each repetition, the spins are initialized randomly by using
    !the "SpinInitialization" subroutine, then at each iteration, 
    !for each spin, the flip is supposed and the move is accepted
    !or refuse by analyzing the variation of  the energy "DeltaE" in the 
    !subroutine "AcceptOrRefuse". After the check of the flip for all
    !the spins, the temperature is updated by subtracting a term 
    !"DeltaT" in the subroutine "Tt".
    !At each iteration, the energy of the system is also computed and 
    !saved in the array "AverageEnergy".
    !After all the iterations and all the repetitions the averages are
    !computed, the ground state energy is estimated and the results 
    !are saved in the external files in the subroutine "Results".

    !input
    integer :: N, Repeat, MaxTime
    real*8, dimension(:,:) :: Jij
    real*8 :: Temperature, ComputedE0
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis, Verbose

    !Spin
    integer, dimension(:), allocatable :: Spin
    !Initial Temperature, Delta Temperature
    real*8 :: T0, DeltaT
    !Random index of the spin to flip 
    integer :: jj
    !Delta Energy
    real*8 :: DeltaE
    !Sign for the spin flip
    integer :: Sign

    !Store the energy
    real*8, dimension(:), allocatable :: AverageEnergy

    !tmp variable
    real*8 :: tmp
    !loops
    integer :: rr, time, ii

    !Spin
    allocate(Spin(N)) !spin array 

    !Allocate Energy Variable
    allocate(AverageEnergy(MaxTime))
    AverageEnergy = 0.0

    !Delta T for temperature reduction
    DeltaT = Temperature/MaxTime

    do rr=1,Repeat
        !initialize Temperature
        T0 = Temperature
        !initialize Spin
        call SpinInitialization(Spin)

        do time=1,MaxTime
            do jj=1,n
                !Suppose the spin flip at index j and compute the 
                !variation of the energy
                tmp = 0
                do ii=1,N 
                    tmp = tmp + 2d0*Jij(ii,jj)*Spin(ii)*Spin(jj)
                end do
                DeltaE = tmp

                !Accept or refuse the move
                call AcceptOrRefuse(DeltaE, T0, Sign)
                !Flip or not the spin jj
                Spin(jj) = Sign*Spin(jj)
            end do

            !compute the energy of the system
            call ClassicalH(N,Spin,Jij,tmp)
            AverageEnergy(time) = AverageEnergy(time)+tmp

            

            !reduce temperature
            call Tt(T0, DeltaT)

        end do
    end do
    

    !compute ground state and save results
    call Results(AverageEnergy, Repeat, ComputedE0, Spin, &
                    FILEEN, FILESP, Analysis)

    if (Verbose) then
        print *, 'Number of iterations:', MaxTime, &
                    '- Energy estimated:', ComputedE0
    end if

    !deallocate the variables
    deallocate(AverageEnergy, Spin)


END SUBROUTINE

!-------------------------GraphPartitioning-----------------------!

SUBROUTINE GraphPartitioningMetropolis(N, Jij, A,B, Temperature, &
                                        Repeat, MaxTime,         &
                                        ComputedE0, FILEEN,      &
                                        FILESP, Analysis, Verbose)

    !INPUT PARAMETERS
    !N: number of nodes/spins -> integer
    !Jij: adjency matrix of graph -> integer, dimension(:,:)
    !A,B: coefficients of the hamiltonians -> real*8
    !Temperature: initial temperature of the systems -> real*8
    !Repeat: number of repetitions to perform averages -> integer
    !MaxTime: number of iterations -> integer
    !ComputedE0: estimation of the ground state -> real*8
    !FILEEN: file in which save the evolution of the 
    !        average energy -> character(len=*)
    !FILESP: file in which save the spin configuration 
    !       -> character(len=*)
    !Analysis: bool variables that permits to choose if 
    !          save or not the results in the external files
    !          -> logical 
    !Verbose: bool variables that permits to choose if print or
    !         not results on screen -> logical 

    !DETAILS
    !The subroutine performs the simulated annealing for the graph
    !partitioning NP-problem.
    !The algorithm is perform "Repeat" times for "MaxTime" iterations.
    !At each repetition, the spins are initialized randomly by using
    !the "SpinInitialization" subroutine, then at each iteration, 
    !for each spin, the flip is supposed and the move is accepted
    !or refuse by analyzing the variation of  the energy "DeltaE" in the 
    !subroutine "AcceptOrRefuse". After the check of the flip for all
    !the spins, the temperature is updated by subtracting a term 
    !"DeltaT" in the subroutine "Tt".
    !At each iteration, the energy of the system is also computed and 
    !saved in the array "AverageEnergy".
    !After all the iterations and all the repetitions the averages are
    !computed, the ground state energy is estimated and the results 
    !are saved in the external files in the subroutine "Results".

    !input
    integer :: N, Repeat, MaxTime
    integer, dimension(:,:) :: Jij
    real*8 :: A, B, Temperature, ComputedE0
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis, Verbose

    !Spin
    integer, dimension(:), allocatable :: Spin
    !Initial Temperature, Delta Temperature
    real*8 :: T0, DeltaT
    !Random index of the spin to flip 
    integer :: jj
    !Delta Energy
    real*8 :: DeltaE
    !Sign for the spin flip
    integer :: Sign

    !Store the energy
    real*8, dimension(:), allocatable :: AverageEnergy

    !tmp variables
    real*8 :: tmp1, tmp2
    !loops
    integer :: rr, time, ii

    !Spin
    allocate(Spin(N)) !spin array 

    !Allocate Energy Variable
    allocate(AverageEnergy(MaxTime))
    AverageEnergy = 0.0

    !Delta T for temperature reduction
    DeltaT = Temperature/MaxTime

    do rr=1,Repeat
        !initialize Temperature
        T0 = Temperature
        !initialize Spin
        call SpinInitialization(Spin)

        do time=1,MaxTime

            do jj=1,n
                !Suppose the spin flip at index j and 
                !compute the variation of the energy
                tmp1 = 0
                tmp2 = 0
                do ii=1,N 
                    tmp1 = tmp1 + Spin(ii)
                    tmp2 = tmp2 + 2.0*Jij(ii,jj)* &
                                    Spin(ii)*Spin(jj)
                end do
                tmp1 = 4d0*Spin(jj)*(Spin(jj)-tmp1)
                DeltaE = A*tmp1+B*tmp2
            
                !Accept or refuse the move
                call AcceptOrRefuse(DeltaE, T0, Sign)
                !Flip or not the spin jj
                Spin(jj) = Sign*Spin(jj)
        
            end do
        
            !compute the energy of the system
            call ClassicalH(N,Jij,Spin,A,B,tmp1)
            AverageEnergy(time) = AverageEnergy(time)+tmp1
        
            !reduce temperature
            call Tt(T0, DeltaT)
        
        end do
    end do

    !compute ground state and save results
    call Results(AverageEnergy, Repeat, ComputedE0, Spin, &
                    FILEEN, FILESP, Analysis)

    if (Verbose) then
        print *, 'Number of iterations:', MaxTime, &
                    '- Energy estimated:', ComputedE0
    end if

    !deallocate the variables
    deallocate(AverageEnergy, Spin)


END SUBROUTINE



!-------------------------VertexCover------------------------!

SUBROUTINE VertexCoverMetropolis(N, A,B, Juv, Temperature,   &
                                    Repeat, MaxTime,         &
                                    ComputedE0, FILEEN,      &
                                    FILESP, Analysis, Verbose)
    !INPUT PARAMETERS
    !N: number of nodes/spins -> integer
    !A,B: coefficients of the hamiltonians -> real*8
    !Juv: adjency matrix of graph -> integer, dimension(:,:) 
    !Temperature: initial temperature of the system -> real*8
    !Repeat: number of repetitions to perform averages -> integer
    !MaxTime: number of iterations -> integer
    !ComputedE0: estimation of the ground state -> real*8
    !FILEEN: file in which save the evolution of the 
    !        average energy -> character(len=*)
    !FILESP: file in which save the spin configuration 
    !       -> character(len=*)
    !Analysis: bool variables that permits to choose if 
    !          save or not the results in the external files
    !          -> logical 
    !Verbose: bool variables that permits to choose if print or
    !         not results on screen -> logical 

    !DETAILS
    !The subroutine performs the simulated annealing for the vertex
    !cover NP-problem.
    !The algorithm is perform "Repeat" times for "MaxTime" iterations.
    !At each repetition, the spins are initialized randomly by using
    !the "SpinInitialization" subroutine, then at each iteration, 
    !for each spin, the flip is supposed and the move is accepted
    !or refuse by analyzing the variation of  the energy "DeltaE" in the 
    !subroutine "AcceptOrRefuse". After the check of the flip for all
    !the spins, the temperature is updated by subtracting a term 
    !"DeltaT" in the subroutine "Tt".
    !At each iteration, the energy of the system is also computed and 
    !saved in the array "AverageEnergy".
    !After all the iterations and all the repetitions the averages are
    !computed, the ground state energy is estimated and the results 
    !are saved in the external files in the subroutine "Results".
    !In particular the configuration saved is recomputed in 0,1 instead
    !of -1,+1.
                                    
    !input
    integer :: N, Repeat, MaxTime
    real*8 :: A, B, Temperature, ComputedE0
    integer, dimension(:,:) :: Juv
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis, Verbose

    !Spin
    integer, dimension(:), allocatable :: Spin
    !Initial Temperature, Delta Temperature
    real*8 :: T0, DeltaT
    !Random index of the spin to flip 
    integer :: jj
    !Delta Energy
    real*8 :: DeltaE
    !Sign for the spin flip
    integer :: Sign

    !Store the energy
    real*8, dimension(:), allocatable :: AverageEnergy

    !tmp variables
    real*8 :: tmp1
    !loops
    integer :: rr, time, ii

    !Spin
    allocate(Spin(N)) !spin array 

    !Allocate Energy Variable
    allocate(AverageEnergy(MaxTime))
    AverageEnergy = 0.0

    !Delta T for temperature reduction
    DeltaT = Temperature/MaxTime

    do rr=1,Repeat
        !initialize Temperature
        T0 = Temperature
        !initialize Spin
        call SpinInitialization(Spin)

        do time=1,MaxTime

            do jj=1,n
                !Suppose the spin flip at index j and 
                !compute the variation of the energy
                tmp1 = 0
                do ii=1,N 
                    tmp1 = tmp1 + Juv(ii,jj)* &
                            (1-Spin(ii))*Spin(jj)
                end do
                DeltaE = A*tmp1-B*Spin(jj)
            
                !Accept or refuse the move
                call AcceptOrRefuse(DeltaE, T0, Sign)
                !Flip or not the spin jj
                Spin(jj) = Sign*Spin(jj)
        
            end do
        
            !compute the energy of the system
            call ClassicalH(N,A,B,Juv,Spin,tmp1)
            AverageEnergy(time) = AverageEnergy(time)+tmp1
        
            !reduce temperature
            call Tt(T0, DeltaT)
        
        end do
    end do

    !convert spins into color
    Spin = int((Spin+1)/2d0)

    !compute ground state and save results
    call Results(AverageEnergy, Repeat, ComputedE0, Spin, &
                    FILEEN, FILESP, Analysis)

    if (Verbose) then
        print *, 'Number of iterations:', MaxTime, &
                    '- Energy estimated:', ComputedE0
    end if

    !deallocate the variables
    deallocate(AverageEnergy, Spin)


END SUBROUTINE


!-------------------------TravelingSalesman-----------------------!

SUBROUTINE TravelingSalesmanMetropolis(C, Juv, A,B, Temperature, &
                                        Repeat, MaxTime,         &
                                        ComputedE0, FILEEN,      &  
                                        FILESP, Analysis, Verbose)
    !INPUT PARAMETERS
    !C: number of nodes/cities -> integer
    !Juv: adjency matrix of graph -> real*8, dimension(:,:) 
    !A,B: coefficients of the hamiltonians -> real*8
    !Temperature: initial temperature of the system -> real*8
    !Repeat: number of repetitions to perform averages -> integer
    !MaxTime: number of iterations -> integer
    !ComputedE0: estimation of the ground state -> real*8
    !FILEEN: file in which save the evolution of the 
    !        average energy -> character(len=*)
    !FILESP: file in which save the spin configuration 
    !       -> character(len=*)
    !Analysis: bool variables that permits to choose if 
    !          save or not the results in the external files
    !          -> logical 
    !Verbose: bool variables that permits to choose if print or
    !         not results on screen -> logical 

    !DETAILS
    !The subroutine performs the simulated annealing for the vertex
    !cover NP-problem.
    !The algorithm is perform "Repeat" times for "MaxTime" iterations.
    !At each repetition, the spins are initialized randomly by using
    !the "SpinInitialization" subroutine, then at each iteration, 
    !for each spin, the flip is supposed and the move is accepted
    !or refuse by analyzing the variation of  the energy "DeltaE" in the 
    !subroutine "AcceptOrRefuse". After the check of the flip for all
    !the spins, the temperature is updated by subtracting a term 
    !"DeltaT" in the subroutine "Tt".
    !At each iteration, the energy of the system is also computed and 
    !saved in the array "AverageEnergy".
    !After all the iterations and all the repetitions the averages are
    !computed, the ground state energy is estimated and the results 
    !are saved in the external files in the subroutine "Results".
    !In particular the configuration saved corresponds to the order
    !in which the nodes/cities are visited.

    !input
    integer :: C, Repeat, MaxTime
    real*8, dimension(:,:) :: Juv
    real*8 :: A, B, Temperature, ComputedE0
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis, Verbose

    
    !Spin
    integer :: N
    integer, dimension(:), allocatable :: Spin, Order
    !Initial Temperature, Delta Temperature
    real*8 :: T0, DeltaT
    !Random index of the spin to flip 
    integer :: vj, vs, js
    !Delta Energy
    real*8 :: DeltaE
    !Sign for the spin flip
    integer :: Sign

    !Store the energy
    real*8, dimension(:), allocatable :: AverageEnergy

    !tmp variables
    real*8 :: tmp1, tmp2, tmp3, tmp4
    !loops
    integer :: rr, time, vv, jj, uu

    !Spin
    N=C**2
    allocate(Spin(N)) !spin array 
    allocate(Order(C))

    !Allocate Energy Variable
    allocate(AverageEnergy(MaxTime))
    AverageEnergy = 0.0

    !Delta T for temperature reduction
    DeltaT = Temperature/MaxTime

    do rr=1,Repeat
        !initialize Temperature
        T0 = Temperature
        !initialize Spin
        call SpinInitialization(Spin)

        do time=1,MaxTime

            do vj=1,N

                !Suppose the spin flip at index vj and 
                !compute the variation of the energy

                js=-1
                do jj=1,N,C
                    if (jj <= vj) then
                        js = js+1
                    else
                        exit
                    end if
                end do
                vs = vj-C*js

                call debug(debsa, 'Original index', vj)
                call debug(debsa, 'Index of time', js)
                call debug(debsa, 'Index of city', vs)

                !first term HA
                tmp1 = 0
                tmp2 = 0
                tmp3 = 0
                  
                do jj=0,C-1
                    tmp1 = tmp1 + (Spin(vs+jj*C)+1.0)
                end do             
                tmp1 = Spin(vj)*(2-tmp1+Spin(vj))
                
                do vv=1,C
                    tmp2 = tmp2 + (Spin(vv+js*C)+1.0)
                end do
                tmp2 = Spin(vj)*(2-tmp2+Spin(vj))

                do uu=1,C 
                    if (Juv(uu,vs)==0) then
                        if (js-1<0) then
                            tmp3 = tmp3 - Spin(vj)* &
                                (Spin(uu+(js+1)*C)+ &
                                Spin(uu+(C-1)*C) +2)/2d0
                        else if (js+1>C-1) then
                            tmp3 = tmp3 - Spin(vj)* &
                                (Spin(uu+(js-1)*C)+ &
                                Spin(uu)+2)/2d0
                        else
                            tmp3 = tmp3 - Spin(vj)* &
                                (Spin(uu+(js+1)*C)+ &
                                Spin(uu+(js-1)*C)+2)/2d0
                        end if
                    end if
                end do                

                !second term HB
                tmp4 = 0
                do uu=1,C 
                    if (Juv(uu,vs)>0 ) then
                        if (js-1<0) then
                            tmp4 = tmp4 - Juv(uu,vs)*Spin(vj)* &
                                (Spin(uu+(js+1)*C)+ &
                                Spin(uu+(C-1)*C) +2)/2d0
                        else if (js+1>C-1) then
                            tmp4 = tmp4 - Juv(uu,vs)*Spin(vj)* &
                                (Spin(uu+(js-1)*C)+ &
                                Spin(uu)+2)/2d0
                        else
                            tmp4 = tmp4 - Juv(uu,vs)*Spin(vj)* &
                                (Spin(uu+(js+1)*C)+ &
                                Spin(uu+(js-1)*C)+2)/2d0
                        end if
                    end if
                end do

                DeltaE = A*(tmp1+tmp2+tmp3)+B*tmp4  
            
                !Accept or refuse the move
                call AcceptOrRefuse(DeltaE, T0, Sign)
                !Flip or not the spin vj
                Spin(vj) = Sign*Spin(vj)

                !print *, tmp1, tmp2, tmp3, tmp4, deltaE, sign
            end do
        
            
            !compute the energy of the system
            call ClassicalH(C,Juv,Spin,A,B,tmp1)
            AverageEnergy(time) = AverageEnergy(time)+tmp1
        
            !reduce temperature
            call Tt(T0, DeltaT)
        
        end do
    end do

    !convert spins into order
    Order = 0
    do vv=1,C 
        do jj=0,C-1 
            if (Spin(vv+jj*C) > 0) then
                Order(vv) = jj+1
                exit
            end if
        end do
    end do

    !compute ground state and save results
    call Results(AverageEnergy, Repeat, ComputedE0, Order, &
                    FILEEN, FILESP, Analysis)

    if (Verbose) then
        print *, 'Number of iterations:', MaxTime, &
                    '- Energy estimated:', ComputedE0
    end if

    !deallocate the variables
    deallocate(AverageEnergy, Spin, Order)


END SUBROUTINE

END MODULE 