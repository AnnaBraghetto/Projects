    MODULE DEBUGGER

    !Module containing debugging subroutines for problems and
    !algorithms studied in the projects.
    !Subroutines for pre- and post-conditions checks are also
    !contained here.
        USE CHECKPOINT

        IMPLICIT NONE

        logical :: deb4qa = .False.

        CONTAINS
        

        SUBROUTINE MatCommutationCheck(mat1, mat2)
            !INPUT PARAMETERS
            !mat1, mat2: matrices to check for commutation 
            !                               -> real*8, dimension(:,:)

            !DETAILS
            !The subroutine checks if the two matrices given
            !as input commute. This will be used as a pre-condition
            !to verify AQC applicability: if they do, the program is 
            !stopped.

            real*8, dimension(:,:) :: mat1, mat2
            real*8, dimension(:,:), allocatable :: AA, BB
            character (len=200) :: message

            allocate(AA(size(mat1,1),size(mat1,2)), &
                     BB(size(mat2,1),size(mat2,2)))

            AA = matmul(mat1, mat2)
            BB = matmul(mat2, mat1)

            if (all(AA-BB.le.1e-5)) then
                print*, "The two hamiltonians commute - ", &
                            "AQC cannot be applied!"

                STOP
            else
                message = "The two hamiltonians do not commute - "// &
                            "AQC condition is valid!"
                call debug(deb4qa, trim(message))
            end if

            deallocate(AA,BB)

        END SUBROUTINE MatCommutationCheck


        SUBROUTINE MatEqualCheck(mat1, mat2)
            !INPUT PARAMETERS
            !mat1, mat2: hamiltonians used for AQC -> hamiltonian type

            !DETAILS
            !The subroutine checks if two matrices are equal (obviously
            !within finite architecture machine limits).
            !It will be as a post-condtion check to verify that the 
            !total hamiltonian H(t) in AQC ends up being equal to the NP  
            !problem hamiltonian HP when t=hamilt%T_final, hence at the 
            !end of the adiabatic transformation.

            real*8, dimension(:,:) :: mat1, mat2
            character (len=200) :: message

            if (all(abs(mat1-mat2).le.1e-5)) then
                message = "The two hamiltonians are equal at t=T - OK"
                call debug(deb4qa, trim(message))
            else
                print*, "The two hamiltonians are not equal at t=T ", &
                        "- something wrong occured during AQC"
                !The program is stopped
                STOP
            end if

    
        END SUBROUTINE MatEqualCheck


        SUBROUTINE HamiltSpectrumCheck(matrix, eigenval)
            !INPUT PARAMETERS
            !matrix: hamiltonian of the problem Hp -> real*8, dimension(:,:)
            !eigeval: array containing the eigenvalues of the input matrix
            !         -> real*8, dimension(:), allocatable

            !DETAILS
            !The subroutine diagionalizes the hamiltonian of the problem
            !Hp and computes its eigenvalues in order to determine the degeneration
            !of its ground state.


            real*8, dimension(:,:) :: matrix
            real*8, dimension(:), allocatable :: eigenval
            real*8, dimension(:,:), allocatable :: eeiggvecs
            !Parameters needed for ssyev subroutine
            integer :: iinfo, lwork
            integer, parameter :: lwork_max = 10000
            real*8, dimension(lwork_max) :: work 

            !matrix%eigenvecs = matrix%elem
            allocate(eigenval(size(matrix,1)))
            allocate(eeiggvecs(size(matrix,1),size(matrix,1)))
            eeiggvecs = matrix

            
            !a query of the optimal workspace needs to be
            !performed
            lwork = -1
            call dsyev("N", "U", size(matrix,1), eeiggvecs, &
                        size(matrix,1), eigenval, work, lwork, iinfo)

            lwork = min( lwork_max, int(work(1)) )
            !Now we solve the eigenproblem
            call dsyev("N", "U", size(matrix,1), eeiggvecs, &
                        size(matrix,1), eigenval, work, lwork, iinfo)

            !Check post-condition for successful exit from
            !Lapack subroutine
            if (iinfo.ne.0) then
                print*, "An error occured while handling ", &
                "Lapack ssyev subroutine - the program ", &
                "will be interrupted."
                STOP 
            end if


            deallocate(eeiggvecs)

            !print*, "GS eigenvectors", psi_diag

        END SUBROUTINE HamiltSpectrumCheck


        SUBROUTINE EigenvecsGSCheck(matrix, psi, eigrecord, prob, deg)
            !INPUT PARAMETERS
            !matrix: hamiltonian H(t) where  t=hamilt%T_final 
            !       -> real*8, dimension(:,:)
            !psi: evoluted eigenfunction at time t -> complex*16, dimension(:)
            !eigrecord: array for the eigenvalues of H(t) to save 
            !           -> real*8, dimension(:)
            !prob: probability to be in the ground state of H(t) given the 
            !      time evoluted eigenfunction psi(t) -> real*8
            !deg: degeneration of the ground state of the hamiltonian of 
            !     the problem Hp -> integer

            !DETAILS
            !The subroutine diagonalizes the hamiltonian H(t) at time 
            !t=hamilt%T_final and computes both the eigenvalues and eigenvectors.
            !An arbitrary number of eigenvalues are then stored.
            !The probability that the evoluted system is in the ground state of H(t)
            !is computed through the product of psi(t) with the first  "deg" 
            !eigenfunctions of H(t) that are relative to the ground state of 
            !Hp.

            integer :: deg, jj, num_eig
            real*8 :: prob
            real*8, dimension(:) :: eigrecord
            complex*16, dimension(:) :: psi
            real*8, dimension(:,:) :: matrix
            real*8, dimension(:), allocatable :: eigenvals
            real*8, dimension(:,:), allocatable :: eeiggvecs
            !temporary store all the eigenvalues
            !real*8, dimension(:), allocatable :: Eigval, psi_diag
            !Parameters needed for ssyev subroutine
            integer :: iinfo, lwork
            integer, parameter :: lwork_max = 10000
            real*8, dimension(lwork_max) :: work 

            !matrix%eigenvecs = matrix%elem
            allocate(eigenvals(size(matrix,1)))
            allocate(eeiggvecs(size(matrix,1),size(matrix,1)))
            eeiggvecs = matrix
            !Number of eigenvalues to retain for record
            num_eig = size(eigrecord)
            
            !a query of the optimal workspace needs to be
            !performed
            lwork = -1
            call dsyev("V", "U", size(matrix,1), eeiggvecs, &
                        size(matrix,1), eigenvals, work, lwork, iinfo)

            lwork = min( lwork_max, int(work(1)) )
            !Now we solve the eigenproblem
            call dsyev("V", "U", size(matrix,1), eeiggvecs, &
                        size(matrix,1), eigenvals, work, lwork, iinfo)

            !Check post-condition for successful exit from
            !Lapack subroutine
            if (iinfo.ne.0) then
                print*, "An error occured while handling ", &
                "Lapack ssyev subroutine - the program ", &
                "will be interrupted."
                STOP 
            end if
            prob = 0.
            !Save needed eigenvalues
            do jj=1,num_eig
                eigrecord(jj) = eigenvals(jj)
            end do 
            !Save needed probabilities
            do jj=1,deg
                prob = prob +abs(dot_product(psi, eeiggvecs(:,jj)))**2
            end do
            deallocate(eeiggvecs, eigenvals)

        END SUBROUTINE EigenvecsGSCheck

    END MODULE DEBUGGER