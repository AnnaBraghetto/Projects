MODULE SIMULATEDQUANTUMANNEALING

    !Module containing the subroutines necessary to
    !perform the simulated quantum annealing for 
    !all the considered problems. 
    !The simulated quantum annealing is then 
    !performed by means of all the the needed 
    !subroutines for each problem. 

USE CHECKPOINT
USE HAMILTONIANS
USE INSTRUTILS
USE ENERGY

IMPLICIT NONE

logical :: debsqa = .False.

!Interface for the simulated quantum annealing

INTERFACE QuantumMetropolis
    MODULE PROCEDURE IsingModelQuantumMetropolis, &
                        GraphPartitioningQuantumMetropolis, &
                        VertexCoverQuantumMetropolis, &
                        TravelingSalesmanQuantumMetropolis
END INTERFACE

CONTAINS 


SUBROUTINE SpinInitialization(Spin)
    !INPUT PARAMETERS
    !Spin: matrix containing the spins 
    !       -> integer, dimension(:,:)

    !DETAILS
    !The subroutines initializes randomly the
    !spins with +1 and -1.
    !On the rows of the matrix there is the spin 
    !configuration for each replica.

    !input
    integer, dimension(:,:) :: Spin 
    !random number
    real*8 :: u
    !loops
    integer :: ii, mm

    Spin = 0
    !initialize randomly the spins with +1 and -1
    do ii=1,size(Spin,2)
        do mm=1,size(Spin,1)
            call random_number(u)
            Spin(mm,ii) = 2*FLOOR(2D0*u)-1
        end do
    end do

END SUBROUTINE


SUBROUTINE TrotterInteraction(J, Spin, mm, jj, DeltaE)
    !INPUT PARAMETERS
    !J: external field at the current time -> real*8
    !Spin: matrix containing the spins -> integer, dimension(:,:)
    !mm: index of the replica -> integer
    !jj: index of the spin in each replica -> integer
    !DeltaE: variation of the energy due to the trotter 
    !        interaction -> real*8

    !DETAILS
    !The subroutine computes the variation of energy after each
    !local moves due to the trotter interaction considering 
    !all the cases of index mm.

    !input
    real*8 :: J, DeltaE
    integer, dimension(:,:) :: Spin
    integer :: mm,jj

    !case of last replica
    if (mm == size(Spin,1)) then
        DeltaE = 2*J*(Spin(mm-1,jj)*Spin(mm,jj)+0.0)
    !case of first replica
    else if (mm == 1) then
        DeltaE = 2*J*(0.0+Spin(mm,jj)*Spin(mm+1,jj))
    else 
        DeltaE = 2*J*(Spin(mm-1,jj)*Spin(mm,jj)+Spin(mm,jj)*Spin(mm+1,jj))
    end if

END SUBROUTINE


SUBROUTINE ACoeffs(As, Time, MaxTime)
    !INPUT PARAMETERS
    !As: value of the coefficient A at time s
    !    -> real*8
    !Time: iteration -> integer
    !MaxTime: number of total iterations -> integer

    !DETAILS
    !The subroutine computes the value of the coefficient
    !As at each time considering the scheduling:
    !t=Time/MaxTime
    !A(t) = 8*t^2-9,6*t+2.88 if t<=0.6
    !A(t) = 0 otherwise 

    !input
    real*8 :: As
    integer :: Time, MaxTime

    !current time in [0,1]
    real*8 :: t 
    t = (Time*1.0)/MaxTime

    if (t<=0.6) then
        As = 8.0*t*t-9.6*t+2.88
    else
        As = 0.0
    end if

END SUBROUTINE


SUBROUTINE BCoeffs(Bs, Time, MaxTime)
    !INPUT PARAMETERS
    !Bs: value of the coefficient B at time s
    !    -> real*8
    !Time: iteration -> integer
    !MaxTime: number of total iterations -> integer

    !DETAILS
    !The subroutine computes the value of the coefficient
    !Bs at each time considering the scheduling:
    !t=Time/MaxTime
    !B(t) = 5.2*t^2+0.2*t

    !input
    real*8 :: Bs
    integer :: Time, MaxTime

    !current time in [0,1]
    real*8 :: t 
    t = (Time*1.0)/MaxTime

    Bs = 5.2*t*t+0.2*t

END SUBROUTINE

SUBROUTINE Jt(J, As, M, Temperature)
    !INPUT PARAMETERS
    !J: value of the external field at time s -> real*8
    !As: value of the coefficient A at time s -> real*8
    !M: number of trotter replicas -> integer
    !Temperature: temperature of the systems -> real*8


    !DETAILS
    !The subroutine computes the value the external field
    !due to the Trotter decomposition at time s:
    !J(t) = -(M*Temperature/2)*log(tanh(As/(M*Temperature)))

    !input
    real*8 :: J, As, Temperature
    integer :: M

    !compute Jt
    J = -(0.5*M*Temperature)*log(tanh(As/(M*Temperature)))

END SUBROUTINE


SUBROUTINE AcceptOrRefuse(DeltaE, M, Temperature, Sign)
    !INPUT PARAMETERS
    !DeltaE: variation of the energy among two configurations
    !       -> real*8
    !M: number of Trotter replicas -> integer
    !Temperature: temperature of the system -> real*8
    !Sign: result of the choice -> integer

    !DETAILS
    !The subroutine accepts the configuration if the energy
    !decreases: DeltaE<=0. Otherwise, the move is accepted with
    !probability e^(-DeltaE/(M*Temperature)).
    !If the move is accepted the sign is set to -1 (actually flip
    !the spin), otherwise it is set to +1 (not flip the spin).

    !input
    real*8 :: DeltaE, Temperature
    integer :: M, Sign
    
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
        probability = exp(-DeltaE/(M*Temperature))
        
        call random_number(u)
        if (probability>u) then
            call debug(debsqa, 'Flip with probability:',probability)
            !flip
            Sign = -1
        else
            !otherwise refuse it and do not do anything
            Sign = +1
        end if     

    end if

END SUBROUTINE


SUBROUTINE Results(AverageEnergy, M, Repeat, ComputedE0, Spin, &
                    FILEEN, FILESP, Analysis)
    !INPUT PARAMETERS
    !AverageEnergy: array containing the sum of the energy of 
    !               the systems of different repetitions  at 
    !               different times -> real*8, dimension(:)
    !M: number of Trotter replicas -> integer
    !Repeat: number of repetitions -> integer
    !ComputedE0: estimation of the ground state -> real*8
    !Spin: configuration of the last repetition 
    !       -> integer, dimension(:,:)
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
    integer :: M, Repeat 
    integer, dimension(:,:) :: Spin
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis

    !determine the estimation of the energy
    AverageEnergy = AverageEnergy/(Repeat*M)
    call GSEstimation(AverageEnergy,ComputedE0)

    if (Analysis) then
        !save energy evolution
        call Save(AverageEnergy, FILEEN)
        !save final spin configuration
        call Save(Spin, FILESP)
    end if

END SUBROUTINE


!---------------------------------MODELS--------------------------------!


!-------------------------------IsingModel------------------------------!

SUBROUTINE IsingModelQuantumMetropolis(N, M, Jij, Temperature, Repeat, &
                                        MaxTime, ComputedE0,           &
                                        FILEEN, FILESP,                &
                                        Analysis, Verbose)
    !INPUT PARAMETERS
    !N: number of nodes/spins -> integer
    !M: number of Trotter replicas -> integer
    !Jij: adjency matrix of the lattice describing the strenght
    !     of the interaction between nearest neighbours 
    !     -> real*8, dimension(:,:)
    !Temperature: temperature of the system -> real*8
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
    !The subroutine performs the simulated quantum annealing for the 
    !Ising model.
    !The algorithm is perform "Repeat" times for "MaxTime" iterations.
    !At each repetition, the spins are initialized randomly by using
    !the "SpinInitialization" subroutine; at each iteration the
    !value of the coeffienct As, Bs and J are computed through the 
    !subroutines "ACoeffs", "BCoeffs" and "Jt" respectevely. Then, two
    !moves are considered. LOCAL MOVE: for each spin in each replica, 
    !the flip is supposed and the move is accepted or refuse by 
    !analyzing the variation of the energy "DeltaE" in the subroutine 
    !"AcceptOrRefuse". GLOBAL MOVE: each position in the system is
    !supposed to flip in all the replicas concurrently and the move is 
    !accepted or refuse by analyzing the variation of the energy 
    !"DeltaE" in the subroutine "AcceptOrRefuse".
    !At each iteration, after the check of flips in both local and 
    !global moves, the energy of the system is also computed and 
    !saved in the array "AverageEnergy".
    !After all the iterations and all the repetitions the averages are
    !computed, the ground state energy is estimated and the results 
    !are saved in the external files in the subroutine "Results".

    !input
    integer :: N, M, Repeat, MaxTime
    real*8, dimension(:,:) :: Jij
    real*8 :: Temperature, ComputedE0
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis, Verbose

    !Spin
    integer, dimension(:,:), allocatable :: Spin
    !Trotter replicas interaction strenght
    real*8 :: J
    !Scheduler
    real*8 :: As, Bs !As scheduler of H0, Bs scheduler of HP
    !Random index of the spin to flip in the columns
    integer :: jj
    !Delta Energy
    real*8 :: DeltaE
    !Sign for the spin flip
    integer :: Sign

    !Store the energy
    real*8, dimension(:), allocatable :: EnergyM, AverageEnergy

    !tmp variable
    real*8 :: tmp
    !loops
    integer :: rr, time, ii, mm

    !Spin
    allocate(Spin(M,N)) !spin matrix: trotter dimension on the rows 

    !Allocate Energy Variables
    allocate(EnergyM(M),AverageEnergy(MaxTime))
    AverageEnergy = 0.0

    do rr=1,Repeat
        !initialize Spin
        call SpinInitialization(Spin)

        do time=1,MaxTime

            !SCHEDULERS
            
            !scheduler for H0
            call ACoeffs(As, Time, MaxTime)
            !Scheduler fo HP
            call BCoeffs(Bs,Time, MaxTime)
            !define trotter interaction strenght J(t) at time t
            call Jt(J,As, M, Temperature)
            
            call debug(debsqa, 'Time', time)
            call debug(debsqa, 'A(s)', As)
            call debug(debsqa, 'B(s)', Bs)
            call debug(debsqa, 'J(t)', J)

  
            !LOCAL MOVE

            !loop on the replicas
            do mm=1,M
                do jj=1,n
                    !Suppose the spin flip at index j and compute the 
                    !variation of the energy

                    !Variation of the energy for the interaction 
                    call TrotterInteraction(J, Spin, mm, jj, DeltaE)
                    !Add variation of the energy on the potential
                    tmp = 0
                    do ii=1,N 
                        tmp = tmp + 2d0*Jij(ii,jj)*Spin(mm,ii)*Spin(mm,jj)
                    end do
                    DeltaE = DeltaE +Bs*tmp

                    !Accept or refuse the move
                    call AcceptOrRefuse(DeltaE, M, Temperature, Sign)
                    !Flip or not the spin mm,jj
                    Spin(mm,jj) = Sign*Spin(mm,jj)
                end do
            end do 

            !GLOBAL MOVE

            do jj=1,n

                !Suppose all the spins in columns jj flip and compute
                !the variation of the energy

                !Variation of the energy on the potential
                tmp=0
                do mm=1,M 
                    do ii=1,N 
                        tmp = tmp + 2d0*Jij(ii,jj)*Spin(mm,ii)*Spin(mm,jj)
                    end do
                end do
                DeltaE = Bs*tmp

                !Accept or refuse the move
                call AcceptOrRefuse(DeltaE, M, Temperature, Sign)
                !Flip or not the spin mm,jj for all mm
                do mm=1,M 
                    Spin(mm,jj) = Sign*Spin(mm,jj)
                end do

            end do
            
            !ENERGY

            !Compute the average energy of the system
            do mm=1,M 
                call ClassicalH(N,Spin(mm,:),Jij,EnergyM(mm))
            end do
            AverageEnergy(time)=AverageEnergy(time)+sum(EnergyM)

            call debug(debsqa,'Average Energy', sum(EnergyM)/M)

        end do
    end do

    !compute the ground state and save the results
    call Results(AverageEnergy, M, Repeat, ComputedE0, Spin, &
                    FILEEN, FILESP, Analysis)


    if (Verbose) then
        print *, 'Number of iterations:', MaxTime, &
                    '- Energy estimated:', ComputedE0
    end if

    !deallocate the variables
    deallocate(EnergyM, AverageEnergy, Spin)

END SUBROUTINE


!----------------------------GraphPartitioning--------------------------!

SUBROUTINE GraphPartitioningQuantumMetropolis(N, M, Jij, A, B,          &
                                                Temperature, Repeat,    &
                                                MaxTime, ComputedE0,    &
                                                FILEEN, FILESP,         &
                                                Analysis, Verbose)
    !INPUT PARAMETERS
    !N: number of nodes/spins -> integer
    !M: number of Trotter replicas -> integer
    !Jij: adjency matrix of graph -> integer, dimension(:,:)
    !A,B: coefficients of the hamiltonians -> real*8
    !Temperature: temperature of the system -> real*8
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
    !The subroutine performs the simulated quantum annealing for the 
    !graph partitioning NP-problem.
    !The algorithm is perform "Repeat" times for "MaxTime" iterations.
    !At each repetition, the spins are initialized randomly by using
    !the "SpinInitialization" subroutine; at each iteration the
    !value of the coeffienct As, Bs and J are computed through the 
    !subroutines "ACoeffs", "BCoeffs" and "Jt" respectevely. Then, two
    !moves are considered. LOCAL MOVE: for each spin in each replica, 
    !the flip is supposed and the move is accepted or refuse by 
    !analyzing the variation of the energy "DeltaE" in the subroutine 
    !"AcceptOrRefuse". GLOBAL MOVE: each position in the system is
    !supposed to flip in all the replicas concurrently and the move is 
    !accepted or refuse by analyzing the variation of the energy 
    !"DeltaE" in the subroutine "AcceptOrRefuse".
    !At each iteration, after the check of flips in both local and 
    !global moves, the energy of the system is also computed and 
    !saved in the array "AverageEnergy".
    !After all the iterations and all the repetitions the averages are
    !computed, the ground state energy is estimated and the results 
    !are saved in the external files in the subroutine "Results".

    !input
    integer :: N, M, Repeat, MaxTime
    integer, dimension(:,:) :: Jij
    real*8 :: A, B, Temperature, ComputedE0
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis, Verbose

    !Spin
    integer, dimension(:,:), allocatable :: Spin
    !Trotter replicas interaction strenght
    real*8 :: J
    !Scheduler
    real*8 :: As, Bs !As scheduler of H0, Bs scheduler of HP
    !Random index of the spin to flip in the columns
    integer :: jj
    !Delta Energy
    real*8 :: DeltaE
    !Sign for the spin flip
    integer :: Sign

    !Store the energy
    real*8, dimension(:), allocatable :: EnergyM, AverageEnergy

    !tmp variables
    real*8 :: tmp1, tmp2, tmp3
    !loops
    integer :: rr, time, ii, mm

    !Spin
    allocate(Spin(M,N)) !spin matrix: trotter dimension on the rows 

    !Allocate Energy Variables
    allocate(EnergyM(M),AverageEnergy(MaxTime))
    AverageEnergy = 0.0

    do rr=1,Repeat
        !initialize Spin
        call SpinInitialization(Spin)
        
        do time=1,MaxTime
        
            !SCHEDULERS
            
            !scheduler for H0
            call ACoeffs(As, Time, MaxTime)
            !Scheduler fo HP
            call BCoeffs(Bs,Time, MaxTime)
            !define trotter interaction strenght J(t) at time t
            call Jt(J,As, M, Temperature)
            
            call debug(debsqa, 'Time', time)
            call debug(debsqa, 'A(s)', As)
            call debug(debsqa, 'B(s)', Bs)
            call debug(debsqa, 'J(t)', J)

            !LOCAL MOVE

            !loop on the replicas
            do mm=1,M
                do jj=1,n
                    !Suppose the spin flip at index j and compute the 
                    !variation of the energy

                    !Variation of the energy for the interaction 
                    call TrotterInteraction(J, Spin, mm, jj, DeltaE)
                    !Add variation of the energy on the potential
                    tmp1 = 0
                    tmp2 = 0
                    do ii=1,N 
                        tmp1=tmp1 +Spin(mm,ii) !HA
                        tmp2=tmp2 +2d0*Jij(ii,jj)* & !HB
                                    Spin(mm,ii)*Spin(mm,jj)
                    end do
                    tmp1= 4d0*Spin(mm,jj)*(Spin(mm,jj)-tmp1)
                    DeltaE = DeltaE +Bs*(A*tmp1+B*tmp2)

                    !Accept or refuse the move
                    call AcceptOrRefuse(DeltaE, M, Temperature, Sign)
                    !Flip or not the spin mm,jj
                    Spin(mm,jj) = Sign*Spin(mm,jj)
                end do

            end do 

            !GLOBAL MOVE

            do jj=1,n

                !Suppose all the spins in columns jj flip and 
                !compute the variation of the energy

                !Variation of the energy on the potential
                tmp1 = 0
                tmp2 = 0
                do mm=1,m
                    tmp3=0
                    do ii=1,N 
                        tmp3 = tmp3 +Spin(mm,ii) !HA
                        tmp2 = tmp2 +2d0*Jij(ii,jj)* & !HB
                                    Spin(mm,ii)*Spin(mm,jj)
                    end do
                    tmp3 = 4d0*Spin(mm,jj)*(Spin(mm,jj)-tmp3)
                    tmp1 = tmp1+tmp3
                end do
    
                DeltaE = Bs*(A*tmp1+ B*tmp2)

                !Accept or refuse the move
                call AcceptOrRefuse(DeltaE, M, Temperature, Sign)
                !Flip or not the spin mm,jj for all mm
                do mm=1,M 
                    Spin(mm,jj) = Sign*Spin(mm,jj)
                end do

            end do
            
            !ENERGY

            !Compute the average energy of the system
            do mm=1,M 
                call ClassicalH(N,Jij,Spin(mm,:),A,B,EnergyM(mm))
            end do
            AverageEnergy(time)=AverageEnergy(time)+sum(EnergyM)

            call debug(debsqa,'Average Energy', sum(EnergyM)/M)

        end do
    end do

    !compute the ground state and save the results
    call Results(AverageEnergy, M, Repeat, ComputedE0, Spin, &
                    FILEEN, FILESP, Analysis)

    if (Verbose) then
        print *, 'Number of iterations:', MaxTime, &
                    '- Energy estimated:', ComputedE0
    end if

    !deallocate the variables
    deallocate(EnergyM, AverageEnergy, Spin)

END SUBROUTINE


!------------------------------VertexCover---------------------------!

SUBROUTINE VertexCoverQuantumMetropolis(N, M, A, B, Juv,            &
                                            Temperature, Repeat,    &
                                            MaxTime, ComputedE0,    &
                                            FILEEN, FILESP,         &
                                            Analysis, Verbose)
    !INPUT PARAMETERS
    !N: number of nodes/spins -> integer
    !M: number of Trotter replicas -> integer
    !A,B: coefficients of the hamiltonians -> real*8
    !Juv: adjency matrix of graph -> integer, dimension(:,:)
    !Temperature: temperature of the systems -> real*8
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
    !The subroutine performs the simulated quantum annealing for the 
    !vertex cover NP-problem.
    !The algorithm is perform "Repeat" times for "MaxTime" iterations.
    !At each repetition, the spins are initialized randomly by using
    !the "SpinInitialization" subroutine; at each iteration the
    !value of the coeffienct As, Bs and J are computed through the 
    !subroutines "ACoeffs", "BCoeffs" and "Jt" respectevely. Then, two
    !moves are considered. LOCAL MOVE: for each spin in each replica, 
    !the flip is supposed and the move is accepted or refuse by 
    !analyzing the variation of the energy "DeltaE" in the subroutine 
    !"AcceptOrRefuse". GLOBAL MOVE: each position in the system is
    !supposed to flip in all the replicas concurrently and the move is 
    !accepted or refuse by analyzing the variation of the energy 
    !"DeltaE" in the subroutine "AcceptOrRefuse".
    !At each iteration, after the check of flips in both local and 
    !global moves, the energy of the system is also computed and 
    !saved in the array "AverageEnergy".
    !After all the iterations and all the repetitions the averages are
    !computed, the ground state energy is estimated and the results 
    !are saved in the external files in the subroutine "Results".
    !In particular the configuration saved is recomputed in 0,1 instead
    !of -1,+1.

    !input
    integer :: N, M, Repeat, MaxTime
    real*8 :: A, B, Temperature, ComputedE0
    integer, dimension(:,:) :: Juv
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis, Verbose

    !Spin
    integer, dimension(:,:), allocatable :: Spin
    !Trotter replicas interaction strenght
    real*8 :: J
    !Scheduler
    real*8 :: As, Bs !As scheduler of H0, Bs scheduler of HP
    !Random index of the spin to flip in the columns
    integer :: jj
    !Delta Energy
    real*8 :: DeltaE
    !Sign for the spin flip
    integer :: Sign

    !Store the energy
    real*8, dimension(:), allocatable :: EnergyM, AverageEnergy

    !tmp variables
    real*8 :: tmp1, tmp2, tmp3
    !loops
    integer :: rr, time, ii, mm

    !Spin
    allocate(Spin(M,N)) !spin matrix: trotter dimension on the rows 

    !Allocate Energy Variables
    allocate(EnergyM(M),AverageEnergy(MaxTime))
    AverageEnergy = 0.0

    do rr=1,Repeat
        !initialize Spin
        call SpinInitialization(Spin)
        
        do time=1,MaxTime
        
            !SCHEDULERS
            
            !scheduler for H0
            call ACoeffs(As, Time, MaxTime)
            !Scheduler fo HP
            call BCoeffs(Bs,Time, MaxTime)
            !define trotter interaction strenght J(t) at time t
            call Jt(J,As, M, Temperature)
            
            call debug(debsqa, 'Time', time)
            call debug(debsqa, 'A(s)', As)
            call debug(debsqa, 'B(s)', Bs)
            call debug(debsqa, 'J(t)', J)

             !LOCAL MOVE

            !loop on the replicas
            do mm=1,M
                do jj=1,n
                    !Suppose the spin flip at index j and compute the 
                    !variation of the energy

                    !Variation of the energy for the interaction 
                    call TrotterInteraction(J, Spin, mm, jj, DeltaE)
                    !Add variation of the energy on the potential
                    tmp1 = 0
                    do ii=1,N 
                        tmp1 = tmp1 + Juv(ii,jj)* &
                                (1-Spin(mm,ii))*Spin(mm,jj)
                    end do
                    DeltaE = DeltaE +Bs*(A*tmp1-B*Spin(mm,jj))

                    !Accept or refuse the move
                    call AcceptOrRefuse(DeltaE, M, Temperature, Sign)
                    !Flip or not the spin mm,jj
                    Spin(mm,jj) = Sign*Spin(mm,jj)
                end do

            end do 

            !GLOBAL MOVE

            do jj=1,n

                !Suppose all the spins in columns jj flip and 
                !compute the variation of the energy

                !Variation of the energy on the potential
                tmp1 = 0
                tmp2 = 0
                do mm=1,m
                    tmp3=0
                    do ii=1,N 
                        tmp3 = tmp3 + Juv(ii,jj)* &
                            (1-Spin(mm,ii))*Spin(mm,jj)
                    end do
                    tmp1 = tmp1+tmp3
                    tmp2 = tmp2 +Spin(mm,jj)
                end do
    
                DeltaE = Bs*(A*tmp1 - B*tmp2)

                !Accept or refuse the move
                call AcceptOrRefuse(DeltaE, M, Temperature, Sign)
                !Flip or not the spin mm,jj for all mm
                do mm=1,M 
                    Spin(mm,jj) = Sign*Spin(mm,jj)
                end do

            end do
            
            !ENERGY

            !Compute the average energy of the system
            do mm=1,M 
                call ClassicalH(N,A,B,Juv,Spin(mm,:),EnergyM(mm))
            end do
            AverageEnergy(time)=AverageEnergy(time)+sum(EnergyM)

            call debug(debsqa,'Average Energy', sum(EnergyM)/M)

        end do
    end do

    !convert spins into color
    Spin = int((Spin+1)/2d0)
    
    !compute the ground state and save the results
    call Results(AverageEnergy, M, Repeat, ComputedE0, Spin, &
                    FILEEN, FILESP, Analysis)

    if (Verbose) then
        print *, 'Number of iterations:', MaxTime, &
                    '- Energy estimated:', ComputedE0
    end if

    !deallocate the variables
    deallocate(EnergyM, AverageEnergy, Spin)

END SUBROUTINE


!----------------------------TravelingSalesman--------------------------!

SUBROUTINE TravelingSalesmanQuantumMetropolis(C, M, Juv, A, B,          &
                                                Temperature, Repeat,    &
                                                MaxTime, ComputedE0,    &
                                                FILEEN, FILESP,         &
                                                Analysis, Verbose)
    !INPUT PARAMETERS
    !N: number of nodes/cities -> integer
    !M: number of Trotter replicas -> integer
    !Juv: adjency matrix of graph -> real*8, dimension(:,:)
    !A,B: coefficients of the hamiltonians -> real*8
    !Temperature: temperature of the systems -> real*8
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
    !The subroutine performs the simulated quantum annealing for the 
    !traveling salesman NP-problem.
    !The algorithm is perform "Repeat" times for "MaxTime" iterations.
    !At each repetition, the spins are initialized randomly by using
    !the "SpinInitialization" subroutine; at each iteration the
    !value of the coeffienct As, Bs and J are computed through the 
    !subroutines "ACoeffs", "BCoeffs" and "Jt" respectevely. Then, two
    !moves are considered. LOCAL MOVE: for each spin in each replica, 
    !the flip is supposed and the move is accepted or refuse by 
    !analyzing the variation of the energy "DeltaE" in the subroutine 
    !"AcceptOrRefuse". GLOBAL MOVE: each position in the system is
    !supposed to flip in all the replicas concurrently and the move is 
    !accepted or refuse by analyzing the variation of the energy 
    !"DeltaE" in the subroutine "AcceptOrRefuse".
    !At each iteration, after the check of flips in both local and 
    !global moves, the energy of the system is also computed and 
    !saved in the array "AverageEnergy".
    !After all the iterations and all the repetitions the averages are
    !computed, the ground state energy is estimated and the results 
    !are saved in the external files in the subroutine "Results".
    !In particular the configuration saved corresponds to the order
    !in which the nodes/cities are visited.

    !input
    integer :: C, M, Repeat, MaxTime
    real*8, dimension(:,:) :: Juv
    real*8 :: A, B, Temperature, ComputedE0
    character(len=*) :: FILEEN, FILESP
    logical :: Analysis, Verbose

    !Spin
    integer :: N 
    integer, dimension(:,:), allocatable :: Spin, Order
    !Trotter replicas interaction strenght
    real*8 :: J
    !Scheduler
    real*8 :: As, Bs !As scheduler of H0, Bs scheduler of HP
    !Random index of the spin to flip in the columns
    integer :: vj, vs, js
    !Delta Energy
    real*8 :: DeltaE
    !Sign for the spin flip
    integer :: Sign

    !Store the energy
    real*8, dimension(:), allocatable :: EnergyM, AverageEnergy

    !tmp variables
    real*8 :: tmp1, tmp2, tmp3, tmp4, tmp5
    !loops
    integer :: rr, time, jj, vv, uu, mm

    !Spin
    N=C**2
    allocate(Spin(M,N)) !spin matrix: trotter dimension on the rows 
    allocate(Order(M,C))

    !Allocate Energy Variables
    allocate(EnergyM(M),AverageEnergy(MaxTime))
    AverageEnergy = 0.0

    do rr=1,Repeat
        !initialize Spin
        call SpinInitialization(Spin)
        
        do time=1,MaxTime
        
            !SCHEDULERS
            
            !scheduler for H0
            call ACoeffs(As, Time, MaxTime)
            !Scheduler fo HP
            call BCoeffs(Bs,Time, MaxTime)
            !define trotter interaction strenght J(t) at time t
            call Jt(J,As, M, Temperature)
            
            call debug(debsqa, 'Time', time)
            call debug(debsqa, 'A(s)', As)
            call debug(debsqa, 'B(s)', Bs)
            call debug(debsqa, 'J(t)', J)

             !LOCAL MOVE

            !loop on the replicas
            do mm=1,M
                do vj=1,N    
                    !Suppose the spin flip at index vj and compute the 
                    !variation of the energy

                    !Variation of the energy for the interaction 
                    call TrotterInteraction(J, Spin, mm, vj, DeltaE)
                    !Add variation of the energy on the potential

                    js=-1
                    do jj=1,N,C
                        if (jj <= vj) then
                            js = js+1
                        else
                            exit
                        end if
                    end do
                    vs = vj-C*js

                    !first term HA
                    tmp1 = 0
                    tmp2 = 0
                    tmp3 = 0

                    do jj=0,C-1
                        tmp1 = tmp1 + (Spin(mm,vs+jj*C)+1.0)
                    end do             
                    tmp1 = Spin(mm,vj)*(2-tmp1+Spin(mm,vj))
                    
                    do vv=1,C
                        tmp2 = tmp2 + (Spin(mm,vv+js*C)+1.0)
                    end do
                    tmp2 = Spin(mm,vj)*(2-tmp2+Spin(mm,vj))
    
                    do uu=1,C 
                        if (Juv(uu,vs)==0) then
                            if (js-1<0) then
                                tmp3 = tmp3 - Spin(mm,vj)* &
                                    (Spin(mm,uu+(js+1)*C)+ &
                                    Spin(mm,uu+(C-1)*C)+2)/2d0
                            else if (js+1>C-1) then
                                tmp3 = tmp3 - Spin(mm,vj)* &
                                    (Spin(mm,uu+(js-1)*C)+ &
                                    Spin(mm,uu)+2)/2d0
                            else
                                tmp3 = tmp3 - Spin(mm,vj)* &
                                    (Spin(mm,uu+(js+1)*C)+ &
                                    Spin(mm,uu+(js-1)*C)+2)/2d0
                            end if
                        end if
                    end do

                    !second term HB
                    tmp4 = 0
                    do uu=1,C 
                        if (Juv(uu,vs)>0 ) then
                            if (js-1<0) then
                                tmp4 = tmp4 - Juv(uu,vs)*Spin(mm,vj)* &
                                    (Spin(mm,uu+(js+1)*C)+ &
                                    Spin(mm,uu+(C-1)*C)+2)/2d0
                            else if (js+1>C-1) then
                                tmp4 = tmp4 - Juv(uu,vs)*Spin(mm,vj)* &
                                    (Spin(mm,uu+(js-1)*C)+ &
                                    Spin(mm,uu)+2)/2d0
                            else
                                tmp4 = tmp4 - Juv(uu,vs)*Spin(mm,vj)* &
                                    (Spin(mm,uu+(js+1)*C)+ &
                                    Spin(mm,uu+(js-1)*C)+2)/2d0
                            end if
                        end if
                    end do
                
                    DeltaE = DeltaE +Bs*(A*(tmp1+tmp2+tmp3)+B*tmp4)

                    !Accept or refuse the move
                    call AcceptOrRefuse(DeltaE, M, Temperature, Sign)
                    !Flip or not the spin mm,vj
                    Spin(mm,vj) = Sign*Spin(mm,vj)
                end do

            end do 

            !GLOBAL MOVE

            do vj=1,N

                !Suppose all the spins in columns vj flip and 
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

                !Variation of the energy on the potential
                tmp4 = 0
                tmp5 = 0
                do mm=1,m
                    !first term HA
                    tmp1 = 0
                    tmp2 = 0
                    tmp3 = 0

                    do jj=0,C-1
                        tmp1 = tmp1 + (Spin(mm,vs+jj*C)+1.0)
                    end do             
                    tmp1 = Spin(mm,vj)*(2-tmp1+Spin(mm,vj))
                    
                    do vv=1,C
                        tmp2 = tmp2 + (Spin(mm,vv+js*C)+1.0)
                    end do
                    tmp2 = Spin(mm,vj)*(2-tmp2+Spin(mm,vj))
    
                    do uu=1,C 
                        if (Juv(uu,vs)==0) then
                            if (js-1<0) then
                                tmp3 = tmp3 - Spin(mm,vj)* &
                                    (Spin(mm,uu+(js+1)*C)+ &
                                    Spin(mm,uu+(C-1)*C)+2)/2d0
                            else if (js+1>C-1) then
                                tmp3 = tmp3 - Spin(mm,vj)* &
                                    (Spin(mm,uu+(js-1)*C)+ &
                                    Spin(mm,uu) +2)/2d0
                            else
                                tmp3 = tmp3 - Spin(mm,vj)* &
                                    (Spin(mm,uu+(js+1)*C)+ &
                                    Spin(mm,uu+(js-1)*C)+2)/2d0
                            end if
                        end if
                    end do

                    !second term HB
                    do uu=1,C 
                        if (Juv(uu,vs)>0 ) then
                            if (js-1<0) then
                                tmp4 = tmp4 - Juv(uu,vs)*Spin(mm,vj)* &
                                    (Spin(mm,uu+(js+1)*C)+ &
                                    Spin(mm,uu+(C-1)*C)+2)/2d0
                            else if (js+1>C-1) then
                                tmp4 = tmp4 - Juv(uu,vs)*Spin(mm,vj)* &
                                    (Spin(mm,uu+(js-1)*C)+ &
                                    Spin(mm,uu) +2)/2d0
                            else
                                tmp4 = tmp4 - Juv(uu,vs)*Spin(mm,vj)* &
                                    (Spin(mm,uu+(js+1)*C)+ &
                                    Spin(mm,uu+(js-1)*C)+2)/2d0
                            end if
                    end if
                    end do
                    
                    tmp5 = tmp5 + tmp1+tmp2+tmp3
                end do
    
                DeltaE = Bs*(A*tmp5+ B*tmp4)

                !Accept or refuse the move
                call AcceptOrRefuse(DeltaE, M, Temperature, Sign)
                !Flip or not the spin mm,vj for all mm
                do mm=1,M 
                    Spin(mm,vj) = Sign*Spin(mm,vj)
                end do

            end do
            
            !ENERGY

            !Compute the average energy of the system
            do mm=1,M 
                call ClassicalH(C,Juv,Spin(mm,:),A,B,EnergyM(mm))
            end do
            AverageEnergy(time)=AverageEnergy(time)+sum(EnergyM)

            call debug(debsqa,'Average Energy', sum(EnergyM)/M)

        end do
    end do

    !convert spins into order
    Order = 0
    do mm=1,M
        do vv=1,C 
            do jj=0,C-1 
                if (Spin(mm,vv+jj*C) > 0) then
                    Order(mm,vv) = jj+1
                    exit
                end if
            end do
        end do
    end do


    

    !compute the ground state and save the results
    call Results(AverageEnergy, M, Repeat, ComputedE0, Order, &
                    FILEEN, FILESP, Analysis)

    if (Verbose) then
        print *, 'Number of iterations:', MaxTime, &
                    '- Energy estimated:', ComputedE0
    end if

    !deallocate the variables
    deallocate(EnergyM, AverageEnergy, Spin, Order)

END SUBROUTINE


END MODULE