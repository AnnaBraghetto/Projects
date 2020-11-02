      MODULE CHECKPOINT
      !Module for general purposes checkpoints.

      IMPLICIT NONE

      INTERFACE DEBUG

      MODULE PROCEDURE NOVAR
      MODULE PROCEDURE EQ
      MODULE PROCEDURE INT2, INT4, REAL4, REAL8, COMPLEX8, COMPLEX16
      
      END INTERFACE


      CONTAINS 

      !GENERAL DOCUMENTATION
      !We define all the possible checkpoint subroutines:
      !+ no var;
      !+ eq to verify conditions;
      !+ int, int*2, real*4, real*8, complex*8, complex*16.
      !They just serve the purpose of acting as a checkpoint
      !over the modules and main program, while at the same time
      !printing key variables either for testing or debugging
      !purposes.

      SUBROUTINE NoVar(FLAG, STR)

            logical :: FLAG
            character(len=1000) :: STR_
            character(len=*), optional :: STR

            !check the presence of the optional string
            if(present(STR))then
                  STR_=STR
            else
                  STR_=""
            endif

            !if statement to check the logical variable
            if (FLAG) then 
                  print *, TRIM(str_)
            end if

      END SUBROUTINE NoVar


      SUBROUTINE EQ(FLAG, STR, VAR)

            logical :: FLAG
            character(len=1000) :: STR_
            character(len=*), optional :: STR
            logical :: VAR

            !check the presence of the optional string
            if(present(STR))then
                  STR_=STR
            else
                  STR_=""
            endif

            !if statement to check the logical variable
            if (FLAG) then 
                  print *, TRIM(str_), var
            end if

      END SUBROUTINE EQ


      SUBROUTINE INT2(FLAG, STR, VAR)

            logical :: FLAG
            character(len=1000) :: STR_
            character(len=*), optional :: STR
            integer*2 :: VAR
            
            !check the presence of the optional string
            if(present(STR))then
                  STR_=STR
            else
                  STR_=""
            endif

            !if statement to check the logical variable
            if (FLAG) then 
                  print *, TRIM(str_), var 
            end if

      END SUBROUTINE INT2


      SUBROUTINE INT4(FLAG, STR, VAR)

            logical :: FLAG
            character(len=1000) :: STR_
            character(len=*), optional :: STR
            integer*4 :: VAR

            !check the presence of the optional string
            if(present(STR))then
                  STR_=STR
            else
                  STR_=""
            endif

            !if statement to check the logical variable
            if (FLAG) then 
                  print *, TRIM(str_), var
            end if

      END SUBROUTINE INT4


      SUBROUTINE REAL4(FLAG, STR, VAR)

            logical :: FLAG
            character(len=1000) :: STR_
            character(len=*), optional :: STR
            real*4 :: VAR
            
            !check the presence of the optional string
            if(present(STR))then
                  STR_=STR
            else
                  STR_=""
            endif

            !if statement to check the logical variable
            if (FLAG) then 
                  print *, TRIM(str_), var 
            end if

      END SUBROUTINE REAL4


      SUBROUTINE REAL8(FLAG, STR, VAR)
      
            logical :: FLAG
            character(len=1000) :: STR_
            character(len=*), optional :: STR
            real*8 :: VAR
            
            !check the presence of the optional string
            if(present(STR))then
                  STR_=STR
            else
                  STR_=""
            endif
      
            !if statement to check the logical variable
            if (FLAG) then 
                  print *, TRIM(str_), var
            end if
      
      END SUBROUTINE REAL8


      SUBROUTINE COMPLEX8(FLAG, STR, VAR)
      
            logical :: FLAG
            character(len=1000) :: STR_
            character(len=*), optional :: STR
            complex*8 :: VAR
            
            !check the presence of the optional string
            if(present(STR))then
                  STR_=STR
            else
                  STR_=""
            endif
      
            !if statement to check the logical variable
            if (FLAG) then 
                  print *, TRIM(str_), var
            end if
      
      END SUBROUTINE COMPLEX8


      SUBROUTINE COMPLEX16(FLAG, STR, VAR)
      
            logical :: FLAG
            character(len=1000) :: STR_
            character(len=*), optional :: STR
            complex*16 :: VAR
            
            !check the presence of the optional string
            if(present(STR))then
                  STR_=STR
            else
                  STR_=""
            endif
      
            !if statement to check the logical variable
            if (FLAG) then 
                  print *, TRIM(str_), var
            end if
      
      END SUBROUTINE COMPLEX16
      

      END MODULE CHECKPOINT