    MODULE ENERGY
        !Module containing subroutines dedicated to the computation
        !of estimates of ground state energy or residual energy with 
        !different approaches, for the algorithms employed.

        USE CHECKPOINT 

        IMPLICIT NONE

        logical :: deben = .False.


        INTERFACE GSEstimation
            MODULE PROCEDURE GSEstimationSimulation
        END INTERFACE


        CONTAINS 

        SUBROUTINE QuantumGroundState(Hamiltonian, Energy)
            !INPUT PARAMETERS
            !Hamiltonian: the matrix to diagonalize -> real*8, dimension(:,:)
            !Energy: the value of the ground state energy -> real*8

            !DETAILS
            !The subroutine computes the energy of the ground state by 
            !directly diagonalizing the quantum hamiltonian.  
            !This estimate will be used as a theoretical comparison with
            !the value obtained from the AQC algorithm. 
            
            !input
            real*8, dimension(:,:) :: Hamiltonian
            real*8 :: Energy
            !temporary store all the eigenvalues
            real*8, dimension(:), allocatable :: Eigval

            !elements needed to compute the eigenvalues
            integer :: LWORK, INFO
            real*8, dimension(:), allocatable :: WORK

            !elements needed to perform the computation
            LWORK = 3*size(Hamiltonian,1)-1
            allocate(WORK(LWORK), Eigval(size(Hamiltonian,1)))

            !diagonalize
            call dsyev("N", "U", size(Hamiltonian,1), Hamiltonian, &
                        size(Hamiltonian,1), Eigval, WORK, LWORK, INFO)

            !Check post-condition for successful exit from
            !Lapack subroutine
            if (info.ne.0) then
                print*, "An error occured while handling ", &
                "Lapack dsyev subroutine - the program ", &
                "will be interrupted."
                STOP 
            end if

            !Ground state
            Energy = Eigval(1)

            call debug(deben, 'Diagonalization-INFO', INFO)
            call debug(deben, 'Ground State Energy', Energy)

            deallocate(WORK, Eigval)

        END SUBROUTINE 
        

        SUBROUTINE GSEstimationSimulation(Energy, Estimate)
            !INPUT PARAMETERS
            !Energy: array containing the energies -> real*8, dimension(:)
            !Estimate: estimate of the ground state energy -> real*8

            !DETAILS
            !The subroutine computes the estimate of the ground state 
            !energy for the annealing simulation approaches. Such values
            !are obtained as averages of the last 10% iterations of the
            !algorithm.
            
            !input
            real*8, dimension(:) :: Energy 
            real*8 :: Estimate
            !number of samples considered in the average
            integer :: Samples
            !loop
            integer :: ii

            !samples considered: the last 10% of the energies
            Samples = floor(0.1*size(Energy))

            !compute the average
            Estimate = 0
            do ii=0,Samples-1
                Estimate = Estimate+Energy(size(Energy)-ii)
            end do
            Estimate = Estimate/Samples

        END SUBROUTINE


        SUBROUTINE ResidualEnergy(CorrectGroundState, ObtainedGroundState, MaxTime, N, FILERE)
            !INPUT PARAMETERS
            !CorrectGroundState: theoretical energy of the ground state 
            !                               -> real*8
            !ObtainedGroundState: energy of the computed ground state -> real*8
            !MaxTime: time of the simulation/computation -> integer
            !N: number of particles -> integer
            !FILERE: file to save the residual energy wrt maxtime

            !DETAILS
            !The subroutine computed the residual energy and writes it 
            !in a file including the timing of the computation
            
            !input 
            real*8 :: CorrectGroundState, ObtainedGroundState
            integer :: MaxTime, N
            character(len=*) :: FILERE
            !result of the residual energy
            real*8 :: Difference

            !computation of the residual energy
            Difference = (ObtainedGroundState - CorrectGroundState)/N

            !open the file, write and close
            open(unit=50,file=FILERE,status="unknown",  access = "append") 
            write(50,*) MaxTime, Difference
            close(50)

        END SUBROUTINE


    END MODULE ENERGY