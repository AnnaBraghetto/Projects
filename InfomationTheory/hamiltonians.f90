MODULE HAMILTONIANS
    
!Module dedicated to the initialization of the 
!HP hamiltonian for each NP problem considered in the project.
!This term, in the quantum cases, will constitute the hamilt%hP
!attribute of the hamiltonian type created for AQC.

USE CHECKPOINT
USE MATHUTILS !contains identity and tensor product
USE INSTRUTILS !contains savemaT

IMPLICIT NONE

    !Pauli matrix sigma_z
    real*8, dimension(2,2) :: sigma_z = &
                    reshape((/ 1,0,0,-1 /), shape(sigma_z))

    !debug
    logical :: debham = .False.


!Interface for the problem hamiltonians

!QUANTUM
INTERFACE QuantumH
    MODULE PROCEDURE IsingModelQuantumHamiltonian, &
                        GraphPartitioningQuantumHamiltonian, &
                        VertexCoverQuantumHamiltonian,       &
                        TravelingSalesmanQuantumHamiltonian

END INTERFACE

!CLASSICAL
INTERFACE ClassicalH
    MODULE PROCEDURE IsingModelClassicalHamiltonian, &
                        GraphPartitioningClassicalHamiltonian, &
                        VertexCoverClassicalHamiltonian,       &
                        TravelingSalesmanClassicalHamiltonian
END INTERFACE


CONTAINS 


!---------------------------IsingModel------------------------!


!QUANTUM

SUBROUTINE IsingModelQuantumHamiltonian(N, J, HP, FILEHP, Analysis)
    !INPUT PARAMETERS
    !N: number of nodes -> integer
    !J: adjacency matrix describing the lattice
    !       -> real*8, dimension(:,:)
    !HP: NP problem hamiltonian 
    !                 -> real*8, dimension(:,:), allocatable
    !FILEHP: file in which save the hamiltonian -> character(len=*)
    !Analysis: bool variable that permits to save or not the 
    !          quantum hamiltonina in an external file -> logical
                                                
    !DETAILS
    !The subroutine computes the quantum hamiltonian of the
    !Ising model (the term HP in the full hamiltonian type).

    !input
    integer :: N 
    real*8, dimension(:,:) :: J
    real*8, dimension(:,:), allocatable :: HP
    character(len=*) :: FILEHP
    logical :: Analysis
    !temporary matrices to help computation
    real*8, dimension(:,:), allocatable :: sigma_i, sigma_j

    !loops
    integer :: ii,jj

    allocate(HP(2**N,2**N))

    HP = 0
    do ii=1,N 
        do jj=ii,N 
            call Element(ii,2,N,sigma_z,sigma_i)
            call Element(jj,2,N,sigma_z,sigma_j)
            HP = HP - J(ii,jj)*sigma_i*sigma_j
            deallocate(sigma_i, sigma_j)
        end do
    end do

    if (Analysis) then
        !write HP
        call Save(HP, FILEHP)
    end if
    
END SUBROUTINE

!CLASSICAL
    
SUBROUTINE IsingModelClassicalHamiltonian(N, SPIN, J, HP)
    !INPUT PARAMETERS
    !N: number of nodes -> integer
    !SPIN: value of the spins -> integer, dimension(:)
    !J: adjacency matrix -> real*8, dimension(:,:)
    !HP: problem hamiltonian -> real*8
                                                    
    !DETAILS
    !The subroutine computes the classical hamiltonian of the 
    !Ising model.    

    !input
    integer :: N 
    integer, dimension(:) :: SPIN
    real*8, dimension(:,:) :: J
    real*8 :: HP

    !loops
    integer :: ii,jj

    HP = 0
    do ii=1,N 
        do jj=ii,N 
            HP = HP - J(ii,jj)*SPIN(ii)*SPIN(jj)
        end do
    end do

END SUBROUTINE


!-----------------------GraphPartitioning---------------------!


!QUANTUM

SUBROUTINE GraphPartitioningQuantumHamiltonian(N, ADJ, A, B, &
                                                HP, FILEHP,  &
                                                Analysis)
    !INPUT PARAMETERS
    !N: number of nodes -> integer
    !ADJ: adjacency matrix -> integer, dimension(:,:)
    !A, B: coefficients of the hamiltonian -> real*8
    !HP: NP problem hamiltonian 
    !                 -> real*8, dimension(:,:), allocatable
    !FILEHP: file in which save the hamiltonian -> character(len=*)
    !Analysis: bool variable that permits to save or not the 
    !          quantum hamiltonina in an external file -> logical

    !DETAILS
    !The subroutine computes the quantum hamiltonian of the
    !graph partitioning NP problem (the term HP in the full
    !hamiltonian type).
    
    !input
    integer :: N
    real*8 :: A, B
    integer, dimension(:,:) :: ADJ
    real*8, dimension(:,:), allocatable :: HP
    character(len=*) ::  FILEHP
    logical :: Analysis

    !hamiltonian: first part HA + second part HB 
    real*8, dimension(:,:), allocatable :: HA, HB

    !loops
    integer :: ii, jj
    !temporary matrices to help computation
    real*8, dimension(:,:), allocatable :: sigma_i, sigma_u, &
                                         sigma_v, id

    !debug
    call debug(debham,'Graph Partitioning Quantum Hamiltoninan')

    !allocate hamiltonian matrix: 2^Nx2^N
    allocate(HP(2**N,2**N),HA(2**N,2**N),HB(2**N,2**N))

    !build first part of the hamiltonian HA
    HA = 0
    do ii=1,N
        !define sigma_z_i
        call Element(ii,2,N,sigma_z,sigma_i)
        !sum
        HA = HA + sigma_i
        !deallocate sigma_z_i
        deallocate(sigma_i)
    end do
    !compute the square
    HA = matmul(HA,HA)
    !debug
    call debug(debham,'HA computed!')

    !build second part of the hamiltonian HB
    HB = 0
    do ii=1,size(ADJ,1)
        do jj=1,size(ADJ,2)
            !check the connection
            if (ADJ(ii,jj) == 1) then
                !debug
                call debug(debham, 'Connection found')

                call Element(ii,2,N,sigma_z,sigma_u)
                call Element(jj,2,N,sigma_z,sigma_v)
                !retrieve identity
                call Identity(id,size(sigma_u,1))
                !compute the sum
                HB = HB + (id-matmul(sigma_u, sigma_v))/2d0
                !deallocate
                deallocate(sigma_u, sigma_v, id)
            end if
        end do
    end do
    
    !debug
    call debug(debham,'HB computed!')

    !total hamiltonian
    HP = A*HA + B*HB

    if (Analysis) then
        !write HP
        call Save(HP, FILEHP)
    end if 

    !deallocate HA and HB
    deallocate(HA,HB)
END SUBROUTINE 


!CLASSICAL

SUBROUTINE GraphPartitioningClassicalHamiltonian(N, ADJ, SPIN, &
                                                    A, B, HP)
    !INPUT PARAMETERS
    !N: number of nodes -> integer
    !ADJ: adjacency matrix -> integer, dimension(:,:)
    !SPIN: value of the spins -> integer, dimension(:)
    !A, B: coefficients of the hamiltonian -> real*8
    !HP: problem hamiltonian -> real*8

    !DETAILS
    !The subroutine computes the classical hamiltonian of the 
    !graph partitioning NP problem.      
    
    !input
    integer :: N
    real*8 :: A, B
    integer, dimension(:) :: SPIN
    integer, dimension(:,:) :: ADJ
    real*8 :: HP

    !hamiltonian: first part HA + second part HB 
    real*8 :: HA, HB

    !loops
    integer :: ii, jj

    !debug
    call debug(debham,'Graph Partitioning Classical Hamiltoninan')

    !build first part of the hamiltonian HA
    HA = sum(SPIN)**2
    !debug
    call debug(debham,'HA computed!')

    !build first part of the hamiltonian HB
    HB = 0
    do ii=1,N
        do jj=1,N
            !check the connection
            if (ADJ(ii,jj) == 1) then
                !compute the sum
                HB = HB + (1-SPIN(ii)*SPIN(jj))/2d0
            end if
        end do
    end do
    
    !debug
    call debug(debham,'HB computed!')

    !total hamiltonian
    HP = A*HA + B*HB

END SUBROUTINE 


!--------------------------VertexCover-------------------------!


!QUANTUM

SUBROUTINE VertexCoverQuantumHamiltonian(N, A, B, Juv, HP,  &
                                            FILEHP, Analysis)
    !INPUT PARAMETERS
    !N: number of nodes -> integer
    !A, B: coefficients of the hamiltonian -> real*8
    !Juv: adjacency matrix -> integer, dimension(:,:)
    !HP: NP problem hamiltonian 
    !                 -> real*8, dimension(:,:), allocatable
    !FILEHP: file in which save the hamiltonian -> character(len=*)
    !Analysis: bool variable that permits to save or not the 
    !          quantum hamiltonina in an external file -> logical

    !DETAILS
    !The subroutine computes the quantum hamiltonian of the
    !vertex cover NP problem (the term HP in the full
    !hamiltonian type).

    !input
    integer :: N
    integer, dimension(:,:) :: Juv
    real*8 :: A, B
    real*8, dimension(:,:), allocatable :: HP
    character(len=*) ::  FILEHP
    logical :: Analysis

    !hamiltonian: first part HA + second part HB 
    real*8, dimension(:,:), allocatable :: HA, HB

    !loops
    integer :: ii, jj
    !temporary matrices to help computation
    real*8, dimension(:,:), allocatable :: sigma_i, sigma_u, &
                                         sigma_v, id

    !debug
    call debug(debham,'Vertex Cover Quantum Hamiltoninan')

    !allocate hamiltonian matrix: 2^Nx2^N
    allocate(HP(2**N,2**N),HA(2**N,2**N),HB(2**N,2**N))

    !build first part of the hamiltonian HA
    HA = 0
    do ii=1,size(Juv,1)
        do jj=1,size(Juv,2)
            !check the connection
            if (Juv(ii,jj) == 1) then
                !debug
                call debug(debham, 'Connection found')

                call Element(ii,2,N,sigma_z,sigma_u)
                call Element(jj,2,N,sigma_z,sigma_v)
                !retrieve identity
                call Identity(id,size(sigma_u,1))
                !compute the sum
                HA = HA + matmul(id-sigma_u,id-sigma_v)
                !deallocate
                deallocate(sigma_u, sigma_v, id)
            end if
        end do
    end do
    HA = HA/4d0

    !debug
    call debug(debham,'HA computed!')

    !build second part of the hamiltonian HB
    HB = 0
    do ii=1,N
        !define sigma_i
        call Element(ii,2,N,sigma_z,sigma_i)
        !retrieve identity
        call Identity(id,size(sigma_i,1))
        !sum
        HB = HB + (sigma_i+id)
        !deallocate sigma_i
        deallocate(sigma_i,id)
    end do
    HB = HB/2d0
    !debug
    call debug(debham,'HB computed!')

    !total hamiltonian
    HP = A*HA + B*HB

    if (Analysis) then
        !write HP
        call Save(HP, FILEHP)
    end if

    !deallocate HA and HB
    deallocate(HA,HB)
END SUBROUTINE 


!CLASSICAL

SUBROUTINE VertexCoverClassicalHamiltonian(N, A, B,    &
                                            Juv, SPIN, &
                                            HP)
    !INPUT PARAMETERS
    !N: number of nodes -> integer
    !A, B: coefficients of the hamiltonian -> real*8
    !Juv: adjacency matrix -> integer, dimension(:,:)
    !SPIN: value of the spins -> integer, dimension(:)
    !HP: problem hamiltonian -> real*8

    !DETAILS
    !The subroutine computes the classical hamiltonian of the 
    !vertex cover NP problem.      

    !input
    integer :: N
    real*8 :: A, B
    integer, dimension(:,:) :: Juv
    integer, dimension(:) :: SPIN
    real*8 :: HP

    !hamiltonian: first part HA + second part HB 
    real*8 :: HA, HB

    !loops
    integer :: ii, jj

    !debug
    call debug(debham,'Vertex Cover Classical Hamiltoninan')

    !build first part of the hamiltonian HA
    HA = 0
    do ii=1,N
        do jj=1,N
            !check the connection
            if (Juv(ii,jj) == 1) then
                !compute the sum
                HA = HA + (1-SPIN(ii))*(1-SPIN(jj))
            end if
        end do
    end do
    HA = HA/4d0
    !debug
    call debug(debham,'HA computed!')

    !build first part of the hamiltonian HB
    HB = sum(Spin)
    HB = (HB + N)/2d0
    !debug
    call debug(debham,'HB computed!')

    !total hamiltonian
    HP = A*HA + B*HB

END SUBROUTINE 

!------------------------TravelingSalesman----------------------!


!QUANTUM

SUBROUTINE TravelingSalesmanQuantumHamiltonian(C, ADJ, A, B, &
                                                HP, FILEHP, &
                                                Analysis)
    !INPUT PARAMETERS
    !C: number of nodes/cities -> integer
    !ADJ: adjacency matrix -> real*8, dimension(:,:)
    !A, B: coefficients of the hamiltonian -> real*8
    !HP: NP problem hamiltonian 
    !                 -> real*8, dimension(:,:), allocatable
    !FILEHP: file in which save the hamiltonian -> character(len=*)
    !Analysis: bool variable that permits to save or not the 
    !          quantum hamiltonina in an external file -> logical

    !DETAILS
    !The subroutine computes the quantum hamiltonian of the
    !traveling salesman NP problem (the term HP in the full
    !hamiltonian type).

    !input
    integer :: C
    real*8 :: A, B
    real*8, dimension(:,:) :: ADJ
    real*8, dimension(:,:), allocatable :: HP
    character(len=*) ::  FILEHP
    logical :: Analysis

    !hamiltonian: first part HA + second part HB 
    real*8, dimension(:,:), allocatable :: HA, HB

    !number of particles
    integer :: N

    !loops
    integer :: jj, vv, uu
    !temporary matrices to help computation
    real*8, dimension(:,:), allocatable :: sigma_vj, sigma_uj1, &
                                            tmp, id

    !the number of particles is the squared of the cities
    N=C**2

    !allocate the hamiltonians
    allocate(HP(2**N,2**N),HA(2**N,2**N),HB(2**N,2**N))
    allocate(tmp(2**N,2**N))


    !build the cycle hamiltonians HA
    !first term
    do vv=1,C 
        tmp = 0 
        do jj=0,C-1 
            !retrieve identity
            call Element(vv+jj*C,2,N,sigma_z,sigma_vj)
            call Identity(id,size(sigma_vj,1))
            tmp = tmp + (sigma_vj+id)*0.5
            deallocate(sigma_vj,id)
        end do
        call Identity(id,size(tmp,1))
        HA = HA + matmul(id-tmp,id-tmp)
        deallocate(id)
    end do
    
    !second term
    do jj=0,C-1 
        tmp = 0
        do vv=1,C  
            call Element(vv+jj*C,2,N,sigma_z,sigma_vj)
            call Identity(id,size(sigma_vj,1))
            tmp = tmp + (sigma_vj+id)*0.5
            deallocate(sigma_vj,id)
        end do
        call Identity(id,size(tmp,1))
        HA = HA + matmul(id-tmp,id-tmp)
        deallocate(id)
    end do
    
    !third term
    do uu=1,C
        do vv=1,C
            do jj=0,C-2
                !check the connection
                if (ADJ(uu,vv) == 0 .and. uu /= vv) then
                    call Element(vv+jj*C,2,N,sigma_z,sigma_vj)
                    call Element(uu+(jj+1)*C,2,N,sigma_z,sigma_uj1)
                    call Identity(id,size(sigma_vj,1))
                    HA = HA + ((sigma_vj+id)*0.5)* &
                                ((sigma_uj1+id)*0.5)
                    deallocate(sigma_vj,sigma_uj1,id)
                end if
            end do
            !check the connection
            if (ADJ(uu,vv) == 0 .and. uu /= vv) then
                call Element(vv+(C-1)*C,2,N,sigma_z,sigma_vj)
                call Element(uu,2,N,sigma_z,sigma_uj1)
                call Identity(id,size(sigma_vj,1))
                HA = HA + ((sigma_vj+id)*0.5)* &
                            ((sigma_uj1+id)*0.5)
                deallocate(sigma_vj,sigma_uj1,id)
            end if
        end do
    end do
    
    
    !build the shortest path contraint HB
    HB = 0
    do uu=1,C
        do vv=1,C
            do jj=0,C-2
                !check the connection
                if (ADJ(uu,vv) > 0) then
                    call Element(vv+jj*C,2,N,sigma_z,sigma_vj)
                    call Element(uu+(jj+1)*C,2,N,sigma_z,sigma_uj1)
                    call Identity(id,size(sigma_vj,1))
                    HB = HB + ADJ(uu,vv)* &
                                ((sigma_vj+id)*0.5)* &
                                ((sigma_uj1+id)*0.5)
                    deallocate(sigma_vj,sigma_uj1,id)
                end if
            end do
            !check the connection
            if (ADJ(uu,vv) > 0) then
                call Element(vv+(C-1)*C,2,N,sigma_z,sigma_vj)
                call Element(uu,2,N,sigma_z,sigma_uj1)
                call Identity(id,size(sigma_vj,1))
                HB = HB + ADJ(uu,vv)* &
                            ((sigma_vj+id)*0.5)* &
                            ((sigma_uj1+id)*0.5)
                deallocate(sigma_vj,sigma_uj1,id)
            end if
        end do
    end do
    
    !total hamiltonian
    HP = A*HA + B*HB

    if(Analysis) then
        !write HP
        call Save(HP, FILEHP)
    end if

    !deallocate HA and HB
    deallocate(HA,HB,tmp)
END SUBROUTINE


!CLASSICAL

SUBROUTINE TravelingSalesmanClassicalHamiltonian(C, ADJ, SPIN, &
                                                    A, B, HP)
    !INPUT PARAMETERS
    !C: number of nodes/cities -> integer
    !ADJ: adjacency matrix -> real*8, dimension(:,:)
    !A, B: coefficients of the hamiltonian -> real*8
    !SPIN: value of the spins -> integer, dimension(:)
    !HP: problem hamiltonian -> real*8

    !DETAILS
    !The subroutine computes the classical hamiltonian of the 
    !traveling salesman NP problem.  

    !input
    integer :: C
    real*8 :: A, B
    integer, dimension(:) :: SPIN
    !on the row there is the order
    !on the col there is the city
    real*8, dimension(:,:) :: ADJ
    real*8 :: HP

    !hamiltonian: first part HA + second part HB 
    real*8 :: HA, HB

    !loops
    integer :: jj, vv, uu
    !temporary variable
    real*8 :: tmp

    !build the cycle hamiltonians HA
    HA = 0

    !first term
    do vv=1,C 
        tmp = 0 
        do jj=0,C-1 
            tmp = tmp + (SPIN(vv+jj*C)+1d0)*0.5
        end do
        HA = HA + (1d0-tmp)**2
    end do

    !second term
    do jj=0,C-1 
        tmp = 0
        do vv=1,C
            tmp = tmp + (SPIN(vv+jj*C)+1d0)*0.5
        end do
        HA = HA + (1d0-tmp)**2
    end do

    !third term
    do uu=1,C
        do vv=1,C
            do jj=0,C-2
                !check the connection
                if (ADJ(uu,vv) == 0 .and. uu /= vv) then
                    HA = HA + ((SPIN(uu+jj*C)+1d0)*0.5)* &
                                ((SPIN(vv+(jj+1)*C)+1d0)*0.5)
                end if
            end do
            !cycle
            if (ADJ(uu,vv) == 0 .and. uu /= vv) then
                HA = HA + ((SPIN(uu+(C-1)*C)+1d0)*0.5)* &
                                ((SPIN(vv)+1d0)*0.5)
            end if
        end do
    end do


    !build the shortest path constraint HB
    HB = 0
    do uu=1,C
        do vv=1,C
            do jj=0,C-2
                !check the connection
                if (ADJ(uu,vv) > 0 ) then
                    HB = HB + ADJ(uu,vv)* &
                            ((SPIN(uu+jj*C)+1d0)*0.5)* &
                            ((SPIN(vv+(jj+1)*C)+1d0)*0.5)
                end if
            end do
            !check the connection
            if (ADJ(uu,vv) > 0 ) then
                HB = HB + ADJ(uu,vv)* &
                        ((SPIN(uu+(C-1)*C)+1d0)*0.5)* &
                        ((SPIN(vv)+1d0)*0.5)
            end if
        end do
    end do

    !total hamiltonian
    HP = A*HA + B*HB

END SUBROUTINE

END MODULE HAMILTONIANS