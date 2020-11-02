    MODULE QUANTUM
        !This module is dedicated to the initialization and the 
        !full implementation of the Adiabatic Quantum Computation
        !(AQC) algorithm, also known as Adiabatic Quantum Optimization.
        !All the quantum machinery is properly defined and initialized
        !and the full algorithm is then laid out in a specific 
        !subroutine.

        USE MATHUTILS
        USE INSTRUTILS
        USE DEBUGGER
        USE HAMILTONIANS
        USE INITIALIZATION

        IMPLICIT NONE

        !to debug
        logical :: debaqc = .False.
        !to choose the kind of evolution
        logical :: linear = .False.

        TYPE HAMILTONIAN
            !ATTRIBUTES
            !H0: hamiltonian of known ground state -> real*8, dimension(:,:)
            !HP: hamiltonian of NP-problem -> real*8, dimension(:,:)
            !H_tot: total time-dep hamiltonian -> real*8, dimension(:,:)
            !T_final: extreme of time interval -> integer
            !E_gs: ground state of hamiltonian at T_final -> real*8

            !DETAILS
            !A matrix type is defined which will be filled
            !by real*8 values, representing the hamiltonian to be
            !evolved over time for AQC.            
            
            integer :: T_final
            real*8 :: E_gs
            real*8, dimension(:,:), allocatable :: h0
            real*8, dimension(:,:), allocatable :: hP
            real*8, dimension(:,:), allocatable :: h_tot 

        END TYPE HAMILTONIAN

        !Pauli matrices 
        real*8, dimension(2,2) :: sigma_x = &
                        reshape((/ 0,1,1,0  /), shape(sigma_x))
        !real*8, dimension(2,2) :: sigma_z = &
        !                reshape((/ 1,0,0,-1 /), shape(sigma_z))

    
        CONTAINS

        SUBROUTINE H0Initialization(hamilt, spins, h0param)
            !INPUT PARAMETERS
            !hamilt: full hamiltonian -> hamiltonian type
            !spins: number of spins in system -> real*8

            !DETAILS
            !This subroutine computes the H0 term of the hamiltonian
            !to be employed for AQC. This is an hamiltonian whose
            !ground state is well-known (Ising transverse field)

            integer :: spins, ss
            real*8 :: h0param
            real*8, dimension(:,:), allocatable :: tmp
            type(hamiltonian) :: hamilt

            allocate(hamilt%h0(2**spins,2**spins))
            hamilt%h0 = 0.
            
            !Tensor products
            do ss=1,spins
                call Element(ss, 2, spins, sigma_x, tmp)
                hamilt%h0 = hamilt%h0+tmp
                deallocate(tmp)
            end do

            hamilt%h0 = -hamilt%h0*h0param


        END SUBROUTINE H0Initialization


        SUBROUTINE HTotInitialization(hamilt, tt)
            !INPUT PARAMETERS
            !hamilt: full hamiltonian -> hamiltonian type 
            !t: time variable -> integer
            
            !DETAILS
            !The subroutine initializes the full time-dependent
            !hamiltonian to be used for AQC in each NP-problem.
            !It should be called after hamilt%h0 and hamilt%hP
            !have been properly defined and initialized, and a
            !time interval extreme hamilt%T_final has been chosen.
            !We stress that time steps are integers increased 
            !unitarily up to hamilt%T_final, hence:
            !t=0,1,2,...,hamilt%T_final

            INTEGER :: tt
            real*8 :: time_ratio
            type(hamiltonian) :: hamilt
            
            !The integers need to be converted to real*8s before
            !taking their ratio (for proper handling)
            time_ratio = (real(tt)/real(hamilt%T_final))

            !Initialization according to AQC
            hamilt%h_tot = 0.
            hamilt%h_tot = (1.-time_ratio)*hamilt%h0 + &
                            time_ratio*hamilt%hP


        END SUBROUTINE HTotInitialization


        SUBROUTINE Psi0Initialization(psi0, spins)
            !INPUT PARAMETERS
            !psi0: wavefunction at t=0 -> real*8, dimension(:)
            !spins: number of spins in system -> real*8
            
            !DETAILS
            !The subroutine initializes the wavefunction at t=0;
            !all states are equiprobable, so the psi0 is an array
            !of 1s with proper normalization.

            integer :: spins 
            real*8 :: normalization
            real*8, dimension(:), allocatable :: psi0

            allocate(psi0(2**spins))

            normalization = 2**(-0.5*spins)
            psi0 = 1.*normalization


        END SUBROUTINE Psi0Initialization


        SUBROUTINE PsiTimeEvolution(psi_cmplx, hamilt, dtime)
            !INPUT PARAMETERS
            !psi: wavefunction at given t -> complex*16, dimension(:)
            !hamilt: full hamiltonian -> hamiltonian type
            
            !DETAILS
            !The subroutine performs the time evolution of the
            !wavefunction by means of direct numerical integration,
            !applying the Lapack chesv subroutine or by using the 
            !approximated temporal operator by means of the logical
            !variable "linear".
            !Please note that the wavefunction MUST be turned into 
            !a complex*16 for things to be properly handled.
            !The time interval dt is defined as the inverse of 
            !hamilt%T_final, extreme of the chosen time interval.

            integer :: ssize
            real*8 :: dtime
            complex*16 :: factor
            complex*16, dimension(:), allocatable :: psi_cmplx
            real*8, dimension(:,:), allocatable :: idmat
            complex*16, dimension(:,:), allocatable :: AA, BB
            type(hamiltonian) :: hamilt
            real*8 :: norm

            if (linear) then
                !Time evolution (Crank-Nicolson) 
                ssize = size(hamilt%h_tot,1)  !square matrix
                allocate(AA(ssize,ssize), BB(ssize,ssize))
                
                call Identity(idmat, ssize)
                factor = dcmplx(0.,1.)*dtime*dcmplx(0.5,0.)
                AA = idmat + hamilt%h_tot*factor
                BB = idmat - hamilt%h_tot*factor
  
                psi_cmplx = matmul(BB, psi_cmplx)
                !Invocation of Lapack subroutine to solve
                !linear system (dir num int)
                call LinSysSolver(AA, psi_cmplx)  

            else
                !Simpler method
                factor = dcmplx(0.,1.)*dtime
                psi_cmplx = psi_cmplx - &
                        matmul(hamilt%h_tot*factor,psi_cmplx)
                
                norm = sqrt(dot_product(psi_cmplx,psi_cmplx))
                psi_cmplx = psi_cmplx/norm

            end if

        END SUBROUTINE PsiTimeEvolution


        SUBROUTINE PsiToConfig(psi, nbinary)
            !INPUT PARAMETERS
            !psi: wavefunction at the end of time evolution
            !                     -> complex*16, dimension(:)
            
            !DETAILS
            !The subroutine computes the probabilities as the
            !square modulus of the wavefunction, looks for the index
            !at highest probability and converts that value in binary
            !form, so that it is possible to obtain the spin
            !configuration related to that state.
            !Note that no specific check on size is to be done, since
            !the index will always be smaller than size(psi) and 
            !the array representing the binary number is allocated
            !accordingly.

            integer :: tmp, ii, nbit, ndec
            integer, dimension(:), allocatable :: nbinary
            real*8, dimension(:), allocatable :: probabilities
            complex*16, dimension(:) :: psi

            allocate(probabilities(size(psi)))
            !Since size(psi) = 2**nbit, we find nbit
            !using log2 and applying elementary log properties
            nbit = int(log(size(psi)*1.)/log(2.))
            allocate(nbinary(nbit))
            
            nbinary = 0
            !Probabilities are square modulus of wavefunction
            probabilities = dreal(conjg(psi)*psi)
            !We find index of max prob and subtract 1 to account
            !for number 0
            ndec = maxloc(probabilities,dim=1)-1
            
            !Start conversion
            tmp = ndec
            do ii=nbit,1,-1 
                !print *, mod(tmp,2) 
                nbinary(ii) = mod(tmp,2)
                tmp = tmp/2
            end do

            !Transformation {0,1} to {-1,1} to get actual
            !spins values
            nbinary = -1*(2*nbinary-1)

            deallocate(probabilities)

        END SUBROUTINE PsiToConfig


        SUBROUTINE AQCAlgorithm(n_spins, hamilt, tot_time, dtime, h0param, &
                                spin_config, filename1, filename2, filename3, &
                                spectrum_analysis, verbose)
            !INPUT PARAMETERS
            !n_spins: size of the system -> integer
            !tot_time: extreme of time interval [0,T] -> integer

            !DETAILS
            !This subroutine groups together all the steps needed to 
            !implement the Adiabatic Quantum Computation (AQC) algorithm. 
            !The hP hamiltonian in the hamiltonian type is intended to
            !be already initialized according to the specific NP problem
            !to tackle.

            integer :: n_spins, deg, num_eig
            integer :: tt, tot_time, index
            real*8 :: dtime
            real*8 :: h0param
            integer, dimension(:), allocatable :: spin_config
            real*8, dimension(:), allocatable :: psi0, energies, prob, eigs
            complex*16, dimension(:), allocatable :: psi_cmplx, psi_hamilt
            real*8, dimension(:,:), allocatable :: eigvals
            type(hamiltonian) :: hamilt
            !Files for saving
            character(len=*) :: filename1, filename2, filename3
            !If True, further analyses on hamiltonian spectrum
            !and probabilities calculations are performed.
            !This implies longer computation times.
            logical :: spectrum_analysis, verbose
            !Number of eigenvalues for spectrum studies
            num_eig = n_spins*2

            !iterations
            hamilt%T_final = int(tot_time*1.0/dtime)

            allocate(hamilt%h_tot(2**n_spins, 2**n_spins))

            if (spectrum_analysis) then
                allocate(energies(int(hamilt%T_final/1000)+1), &
                            prob(int(hamilt%T_final/1000)+1),  &
                            eigvals(int(hamilt%T_final/1000)+1, num_eig))
            end if

            !AQC ALGORITHM
            !Hamiltonian H0 init
            call H0Initialization(hamilt, n_spins, h0param)
            !Check pre-condition
            call MatCommutationCheck(hamilt%h0, hamilt%hP)
            !Wavefunction init
            call Psi0Initialization(psi0, n_spins)
            !The wavefunction needs to be turned complex*16
            !for time evolution 
            allocate(psi_cmplx(2**n_spins),psi_hamilt(2**n_spins))
            psi_cmplx     = dcmplx(psi0, 0.)
            psi_hamilt    = dcmplx(0.,0.)
            deallocate(psi0)


            !Determine the degenerary of the ground state
            if (spectrum_analysis) then
                call HamiltSpectrumCheck(hamilt%hP, eigs)
                deg = 1
                do while( abs(eigs(deg) - eigs(deg+1)) < 1e-5)
                    deg = deg+1
                end do
                deallocate(eigs)
            end if

            !Start time evolution
            index=1
            do tt = 0, hamilt%T_final
                call HTotInitialization(hamilt, tt)
                call PsiTimeEvolution(psi_cmplx, hamilt, dtime)

                if (spectrum_analysis .and. mod(tt,1000).eq.0) then
                    !diagonalization
                    call EigenvecsGSCheck(hamilt%h_tot, psi_cmplx, &
                                eigvals(index, :), prob(index),deg)
                    psi_hamilt = matmul(psi_cmplx, hamilt%h_tot)
                    !energy
                    energies(index) = dot_product(psi_hamilt, psi_cmplx)    
                    index = index+1
                end if

            
                !Counter for status
                if (debaqc) then
                    if (mod(tt,1000).eq.0) then
                        print*, "Iteration ", tt, "/", hamilt%T_final
                        print*, "Index max is ", maxloc(abs(conjg(psi_cmplx)&
                                    *psi_cmplx))
                        print*, "Maximum value is ", maxval(abs( &
                                                conjg(psi_cmplx)*psi_cmplx))
                    end if
                end if

            end do

            !Check post-condition 
            call MatEqualCheck(hamilt%h_tot, hamilt%hP)

            !Compute GS energy <psi|H(T)|psi> (the wavefunction
            !is already normalized)
            psi_hamilt = matmul(psi_cmplx, hamilt%h_tot)
            hamilt%E_gs = dot_product(psi_hamilt, psi_cmplx)


            !Obtain configuration of spin from GS
            call PsiToConfig(psi_cmplx, spin_config)

            !Final deallocations
            deallocate(psi_cmplx, psi_hamilt)
            deallocate(hamilt%h0, hamilt%h_tot)

            !File savings
            if (spectrum_analysis) then  
                call Save(energies, filename1)
                call Save(prob, filename2)
                call Save(eigvals, filename3)
                deallocate(energies, prob, eigvals)
            end if


            if (Verbose) then
                print *, 'Number of iterations:', hamilt%T_final, &
                        '- Energy estimated:', hamilt%E_gs
            end if


        END SUBROUTINE AQCAlgorithm

    END MODULE QUANTUM