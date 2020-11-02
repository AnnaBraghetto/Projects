    MODULE MATHUTILS
        !Module containing the mathematical basic tools
        !that will be needed throughout the project, especially
        !to deal with the quantum formalism and to ease the 
        !implementation of basic Ising models.

        USE CHECKPOINT

        IMPLICIT NONE

        CONTAINS


        SUBROUTINE TensorProd(First, Second, Result)
            !INPUT PARAMETER
            !First, Second: factor for tensor product 
            !                     -> real*8, dimension(:,:)
            !Result: result of tensor product 
            !                     -> real*8, dimension(:,:), allocatable
            
            !DETAILS
            !The subroutine computes the tensor product between two 
            !different matrices. The dimension of result is allocated
            !automatically.

            !input arrays
            real*8, dimension(:,:), intent(in) :: First, Second
            !dimension of first, dimension of second
            integer :: dim1,dim2
            !result of the tensor product
            real*8, dimension(:,:), allocatable :: Result
            !loops
            integer :: ii, jj, ri, rj, ci,cj

            !defining the dimensions
            dim1 = size(First,1)
            dim2 = size(Second,1)

            !allocating the result
            allocate(Result(dim1*dim2,dim1*dim2))

            !counter
            ci=0
            cj=0
            do ii = 1, dim1    
                do jj = 1, dim1
                        do ri=1, dim2
                            do rj=1,dim2
                                    !compute the element
                                    Result(ri+ci,rj+cj) = First(ii,jj) &
                                    *Second(ri,rj)
                            end do
                        end do 
                        cj= cj+dim2
                end do 
                ci = ci+dim2
                cj= 0
            end do

        END SUBROUTINE TensorProd


        SUBROUTINE Identity(Mat, ssize)
            !INPUT PARAMETERS
            !Mat: to-be identity matrix 
            !                     -> real*8, dimension(:,:), allocatable
            !ssize: needed size of matrix -> integer
            
            !DETAILS
            !The subroutine initializes an identity matrix of given 
            !dimension.
            
            !input matrix
            real*8, dimension(:,:), allocatable :: Mat
            !size of identity
            integer :: ssize
            !integer for loops
            integer :: ii

            allocate(Mat(ssize,ssize))

            Mat = 0.
            !setting the identity
            do ii=1,size(Mat,1)
                Mat(ii,ii) = 1
            end do

        END SUBROUTINE Identity


        SUBROUTINE Element(i, dim, N, sigma, output)
            !INPUT PARAMETERS
            !i: element of summatory -> integer
            !dim: number of possible spins -> integer
            !N: number of particles in system -> integer
            !sigma: needed Pauli matrix -> real*8, dimension(:,:)
            !output: real*8, dimension(:,:), allocatable

            !DETAILS
            !The subroutine returns the i-th element of the spin in the 
            !dim^N space. It is used to initialize any Ising hamiltonian
            !according to the Pauli matrix assigned in input. 

            !input index, dimension and particles
            integer :: i, dim, N
            !input pauli matrix
            real*8, dimension(:,:) :: sigma
            !input output
            real*8, dimension(:,:), allocatable :: output
            !matrices to help computation
            real*8, dimension(:,:), allocatable :: right, left, tens

            if (i == 1) then
            
                !retrieve the identity
                call Identity(right, dim**(N-1))
                !compute the tensor product
                call TensorProd(sigma,right, output)
                !deallocate
                deallocate(right)
                
            else if (i == N) then
        
                !retrieve the identity
                call Identity(left, dim**(N-1))
                !compute the tensor product
                call TensorProd(left, sigma, output)
                !deallocate
                deallocate(left)

            else

                !allocate the left element as the tensor prod of i-1 
                !identities
                call Identity(left,dim**(i-1))
                !allocate the right element as the tensor prod of N-i 
                !identities
                call Identity(right,dim**(N-i))
                !compute the tensor product of the first i elements
                call TensorProd(left, sigma, tens)
                !compute the tensor product of the last elements
                call TensorProd(tens, right, output)
                !deallocate
                deallocate(right, left, tens)

            end if

        END SUBROUTINE Element


        SUBROUTINE LinSysSolver(AA, bb)
            !INPUT PARAMETERS
            !AA: symmetric matrix of coefficients (AAx=bb) 
            !    -> complex*16, dimension(:,:)
            !bb: vector of coefficients (AAx=bb) 
            !    -> complex*16, dimension(:)

            !DETAILS
            !The subroutine permits to solve a linear
            !system AAx=bb, determining the vector x, by
            !using the Lapack subroutine zysysv for 
            !symmetric matrices AA.

            !needed inputs
            integer :: N, nrhs, lda, ldb
            real*8 :: norm
            integer, parameter :: lwmax = 1000000000
            complex*16, dimension(:) :: bb
            complex*16, dimension(:,:) :: AA 
            integer :: info, lwork
            integer, dimension(:), allocatable :: ipiv
            complex*16 :: work(lwmax)

            N    = size(AA,1)
            nrhs = 0
            lda = N ; ldb = N
            allocate(ipiv(N))

            !a query of the optimal workspace needs to be
            !performed
            lwork = -1
            call zsysv('U', N, nrhs, AA, lda, ipiv, bb, ldb,&
                    work, lwork, info)

            lwork = min( lwmax, int(work(1)) )
            !Now we solve the AX=b linear system
            call zsysv('U', N, nrhs, AA, lda, ipiv, bb, ldb,&
                    work, lwork, info)
            deallocate(ipiv)

            !Normalization of the wavefunction;
            !dot_product automatically takes the conj of the
            !first argument if the vectors are complex
            norm = sqrt(dot_product(bb, bb))
            bb = bb/norm
            
            !check post-condition for successful exit from
            !Lapack subroutine
            if (info.ne.0) then
                print*, "An error occured while handling ", &
                "Lapack chesv subroutine - the program ", &
                "will be interrupted."
                STOP 
            end if

        END SUBROUTINE LinSysSolver
        
        
        SUBROUTINE RandomIndex(N, index)
            !INPUT PARAMETERS
            !N: upper limit -> integer
            !index: random extracted index -> integer
        
            !DETAILS
            !The subroutine extracts randomly an integer
            !between 1 and N.
        
            !input
            integer :: N, index
        
            !random number
            real*8 :: u
        
            call random_number(u)
            index = FLOOR(N*u+1)
            !check the index
            do while (index>N) 
                call random_number(u)
                index = FLOOR(N*u+1)
            end do 
        
        END SUBROUTINE

    END MODULE MATHUTILS


    ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

    
    MODULE INSTRUTILS
        !Module containing the instrumental tools that are
        !needed for the main general purpose applications, such as 
        !saving matrices or array to file or having them print in 
        !an ordered fashion to terminal.

        IMPLICIT NONE

        !Interface for the matrix and array saving

        INTERFACE Save
            MODULE PROCEDURE SaveRealMat, SaveReal8Mat, SaveIntMat
            MODULE PROCEDURE SaveRealArr, SaveReal8Arr, SaveIntArr
        END INTERFACE

        !Interface for the print of the matrix

        INTERFACE PrintMat
            MODULE PROCEDURE PrintMatReal, PrintMatCmplx
        END INTERFACE

        CONTAINS


        SUBROUTINE SaveRealMat(MAT, FILE)
            !INPUT PARAMETERS
            !mat: matrix to save in a file -> real, dimension(:,:)
            !file: file in which save the input matrix 
            !        -> character(len=*)

            !DETAILS
            !The subroutine saves the input real matrix in an 
            !arbitrary file in readable form

            !input
            real, dimension(:,:) :: MAT
            character(len=*) :: FILE
            !loops
            integer :: ii, jj

            open(unit=50,file=TRIM(FILE),status="unknown") 
            do ii=1,size(MAT,1)
                write(50,*) (MAT(ii,jj), jj=1,size(MAT,2))
            end do
            close(50)

        END SUBROUTINE SaveRealMat


        SUBROUTINE SaveReal8Mat(MAT, FILE)
            !INPUT PARAMETERS
            !mat: matrix to save in a file -> real*8, dimension(:,:)
            !file: file in which save the input matrix 
            !        -> character(len=*)

            !DETAILS
            !The subroutine saves the input real*8 matrix in an 
            !arbitrary file in readable form

            !input
            real*8, dimension(:,:) :: MAT
            character(len=*) :: FILE

            !loops
            integer :: ii, jj

            open(unit=50,file=TRIM(FILE),status="unknown") 
            do ii=1,size(MAT,1)
                write(50,*) (MAT(ii,jj), jj=1,size(MAT,2))
            end do
            close(50)

        END SUBROUTINE SaveReal8Mat


        SUBROUTINE SaveIntMat(MAT, FILE)
            !INPUT PARAMETERS
            !mat: matrix to save in a file -> integer, dimension(:,:)
            !file: file in which save the input matrix 
            !        -> character(len=*)

            !DETAILS
            !The subroutine saves the input integer matrix in an 
            !arbitrary file in readable form

            !input
            integer, dimension(:,:) :: MAT
            character(len=*) :: FILE
            !loops
            integer :: ii, jj

            open(unit=50,file=TRIM(FILE),status="unknown") 
            do ii=1,size(MAT,1)
                write(50,*) (MAT(ii,jj), jj=1,size(MAT,2))
            end do
            close(50)

        END SUBROUTINE SaveIntMat


        SUBROUTINE SaveRealArr(ARR, FILE)
            !INPUT PARAMETERS
            !arr: array to save in a file -> real, dimension(:)
            !file: file in which save the input array
            !        -> character(len=*)

            !DETAILS
            !The subroutine saves the input real array in an 
            !arbitrary file in readable form

            !input
            real, dimension(:) :: ARR
            character(len=*) :: FILE
            !loops
            integer :: ii

            open(unit=50,file=TRIM(FILE),status="unknown") 
            do ii=1,size(ARR)
                write(50,*) ARR(ii)
            end do
            close(50)

        END SUBROUTINE SaveRealArr


        SUBROUTINE SaveReal8Arr(ARR, FILE)
            !INPUT PARAMETERS
            !arr: array to save in a file -> real*8, dimension(:)
            !file: file in which save the input array
            !        -> character(len=*)

            !DETAILS
            !The subroutine saves the input real*8 array in an 
            !arbitrary file in readable form

            !input
            real*8, dimension(:) :: ARR
            character(len=*) :: FILE

            !loops
            integer :: ii

            open(unit=50,file=TRIM(FILE),status="unknown") 
            do ii=1,size(ARR)
                write(50,*) ARR(ii)
            end do
            close(50)

        END SUBROUTINE SaveReal8Arr
        
        
        SUBROUTINE SaveIntArr(ARR, FILE)
            !INPUT PARAMETERS
            !arr: array to save in a file -> integer, dimension(:)
            !file: file in which save the input array
            !        -> character(len=*)

            !DETAILS
            !The subroutine saves the input integer array in an 
            !arbitrary file in readable form

            !input
            integer, dimension(:) :: ARR
            character(len=*) :: FILE
            !loops
            integer :: ii

            open(unit=50,file=TRIM(FILE),status="unknown") 
            do ii=1,size(ARR)
                write(50,*) ARR(ii)
            end do
            close(50)

        END SUBROUTINE SaveIntArr


        SUBROUTINE PrintMatReal(matrix)
            !INPUT PARAMETERS
            !matrix: matrix to print on screen -> real*8, dimension(:,:)

            !DETAILS
            !The subroutine prints the input real*8 matrix 
            !in readable form

            !input
            integer :: ii, jj
            real*8, dimension(:,:) :: matrix

            do ii=1,size(matrix,1)
                print*, (int(matrix(ii,jj)), jj=1,size(matrix,2))
            end do

        END SUBROUTINE PrintMatReal


        SUBROUTINE PrintMatCmplx(matrix)
            !INPUT PARAMETERS
            !matrix: matrix to print on screen -> complex*16, dimension(:,:)

            !DETAILS
            !The subroutine prints the input complex*16 matrix 
            !in readable form

            !input
            integer :: ii, jj
            complex*16, dimension(:,:) :: matrix

            do ii=1,size(matrix,1)
                print*, (matrix(ii,jj), jj=1,size(matrix,2))
            end do

        END SUBROUTINE PrintMatCmplx


        SUBROUTINE PrepareLocation(path)
            !INPUT PARAMETERS
            !path: name of the directory that will be created 
            !       -> character(len=*)

            !DETAILS
            !The subroutine removes the directory called "path"
            !and creates it again in order to save the results 
            !of the computation

            !input
            character(len=*) :: path
            !command 
            character(len=100) :: command

            !remove the old direcory
            command='rm -r ' // path
            call system(trim(command))

            !create the directory
            command='mkdir -p ' // path
            call system(trim(command))

        END SUBROUTINE


    END MODULE INSTRUTILS