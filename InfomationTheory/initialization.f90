MODULE INITIALIZATION

!Module dedicated to the initializations of each 
!NP problem as far as the mathematical objects and quantities
!needed to either visualize it or implement it.
!Please notice that the hamiltonians HP are initialized in a
!different, dedicated module.
    
USE CHECKPOINT
USE MATHUTILS !contains identity and tensor product
USE INSTRUTILS !contains savemat

IMPLICIT NONE

    logical :: debinit = .False.

!Interface for the problem initialization
INTERFACE ProblemInitialization
    MODULE PROCEDURE IsingModelInitialization, &
                        GraphInitialization,   &
                        TravelingSalesmanInitialization
END INTERFACE


CONTAINS 


!---------------------------IsingModel-------------------------!


SUBROUTINE IsingModelInitialization(L, Jval, J, FILEJ, Analysis) 
    !INPUT PARAMETERS
    !L: side of the lattice L*L -> integer
    !Jval: strenght of the interaction for connected spins i,j 
    !       that is constant for all the couples i,j -> real*8
    !J: adjency matrix for the lattice with 0 and Jval as values
    !   -> real*8, dimension(:,:), allocatable
    !FILEJ: file in which save the adjency matrix J if required
    !       -> character(len=*)
    !Analysis: bool variable that permits to save or not the 
    !           adjency matrix in an external file -> logical

    !DETAILS
    !The subroutine builds the adjency matrix of a lattice with side L, 
    !where the weight of the connenctions is 0 if the spins are not 
    !connected and Jval otherwise.
    !Then, if Analysis is set to True, the adjency matrix is saved 
    !in the file FILEJ.
   
    !input
    integer :: L
    real*8 :: Jval
    real*8, dimension(:,:), allocatable :: J
    character(len=*) :: FILEJ
    logical :: Analysis

    !number of particles
    integer :: N

    !loop
    integer :: ii

    !allocate spin
    N = L**2
    allocate(J(N,N))
    J = 0

    do ii=1,N
        if (mod(ii,L)/=0) then
            J(ii,ii+1) = Jval
        end if
        if (mod(ii-1,L)/=0) then
            J(ii,ii-1) = Jval
        end if
        if (ii+L<=N) then
            J(ii,ii+L) = Jval
        end if
        if (ii-L>0) then
            J(ii,ii-L) = Jval
        end if
    end do

    if (Analysis) then
        !write J
        call Save(J, FILEJ)
    end if

END SUBROUTINE


!-------------------GraphPartitioning&VertexCover----------------!


SUBROUTINE GraphInitialization(N, E, ADJ, FILEADJ, Analysis)
    !INPUT PARAMETERS
    !N: number of nodes -> integer
    !E: number of edges -> integer
    !ADJ: adjacency matric 
    !               -> integer, dimension(:,:), allocatable
    !FILEADJ: file in which save the adjacency matrix 
    !               -> character
    !Analysis: bool variable that permits to save or not the 
    !          adjency matrix in an external file -> logical

    !DETAILS
    !The subroutine randomly builds the adjacency matrix of a 
    !graph, given the number of nodes and of connecting edges. 
    !The weight of each present edges is fixed to 1.

    !input
    integer :: N, E
    integer, dimension(:,:), allocatable :: ADJ
    character(len=*) :: FILEADJ
    logical :: Analysis

    !random numbers
    real*8 :: u
    integer :: xx, yy

    !loop
    integer :: ii

    !debug
    call debug(debinit,'Graph Initialization')
    !allocate adjacency matrix: NxN
    allocate(ADJ(N,N))
    
    !debug
    call debug(debinit,'Size of the adjacency matrix',size(ADJ))
    !initialize randomly the adjacency matrix
    ADJ = 0 !no connections
    do ii=1,E
        !draw x and y
        !u in [0,1] -> floor(2*u) = {0,1}
        call random_number(u)
        xx = floor(N*u+1)
        call random_number(u)
        yy = floor(N*u+1)

        !If edge already present or self-interaction,
        !random number is re-drawn
        do while (xx == yy .or. ADJ(xx,yy) == 1 &
                                .or. ADJ(yy,xx) == 1)
            call random_number(u)
            xx = floor(N*u+1)
            call random_number(u)
            yy = floor(N*u+1)
        end do

        !debug
        call debug(debinit,'Edge number',ii)
        !draw an edge, imposing the adj mat to 
        !be symmetric, meaning undirected edges
        ADJ(xx,yy) = 1
        ADJ(yy,xx) = 1
    end do

    !debug
    call debug(debinit,'Insert all the connections!')

    if (Analysis) then
        !write ADJ
        call Save(ADJ, FILEADJ)
    end if

END SUBROUTINE 


!----------------TravelingSalesmanInitialization----------------!


SUBROUTINE TravelingSalesmanInitialization(C, E, ADJ, FILEADJ, &
                                            Analysis)
    !INPUT PARAMETERS
    !C: number of nodes -> integer
    !E: number of edges -> integer
    !ADJ: adjacency matric 
    !               -> real*8, dimension(:,:), allocatable
    !FILEADJ: file in which save the adjacency matrix 
    !               -> character
    !Analysis: bool variable that permits to save or not the 
    !          adjency matrix in an external file -> logical

    !DETAILS
    !The subroutine randomly builds the adjacency matrix of a 
    !graph, given the number of nodes and of connecting edges. 
    !The weight of each present edges is chosen randomly in the
    !interval [1e-2,1].

    !input
    integer :: C, E
    real*8, dimension(:,:), allocatable :: ADJ 
    character(len=*) :: FILEADJ
    logical :: Analysis
    
    !random numbers
    real*8 :: u
    integer :: xx, yy

    !loop
    integer :: ii

    !allocate adj
    allocate(ADJ(C,C))

    !initialize randomly the adjacency matrix
    ADJ = 0 !no connections
    do ii=1,E
        !draw x and y
        !u in [0,1] -> floor(2*u) = {0,1}
        call random_number(u)
        xx = floor(C*u+1)
        call random_number(u)
        yy = floor(C*u+1)

        !If edge already present or self-interaction,
        !random number is re-drawn
        do while (xx == yy .or. ADJ(xx,yy) >0 &
                                .or. ADJ(yy,xx) >0)
            call random_number(u)
            xx = floor(C*u+1)
            call random_number(u)
            yy = floor(C*u+1)
        end do

        !debug
        call debug(debinit,'Edge number',ii)
        !draw an edge, imposing the adj mat to 
        !be symmetric, meaning undirected edges

        call random_number(u)
        do while(u<1e-2)
            call random_number(u)
        end do
        ADJ(xx,yy) = u
        ADJ(yy,xx) = ADJ(xx,yy)
    end do

    if (Analysis) then
        !write ADJ
        call Save(ADJ, FILEADJ)
    end if

END SUBROUTINE


END MODULE INITIALIZATION   