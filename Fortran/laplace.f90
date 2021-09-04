module Laplacian_Form

  use omp_lib
  use iso_fortran_env
  
  implicit none
  public
  
  integer :: &
    MAX_ITERATIONS = 100
    !MAX_ITERATIONS = 102400
  real ( real64 ) :: &
    MAX_ERROR = 1e-5_real64

contains 

  subroutine Initialize ( nCells, T, T_Init )
    
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), dimension ( :, : ), allocatable, intent ( out ) :: &
      T, &
      T_Init
      
    allocate ( T      ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    allocate ( T_Init ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    
    call random_number ( T )
    T_Init = T
         
  end subroutine Initialize
  
  
  subroutine Compute ( T, nCells, Error, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Error
    integer, intent ( out ) :: &
      nIterations
    
    real ( real64 ) :: &
      TimeStart, &
      TimeTotal
    real ( real64 ), dimension ( :, : ), allocatable :: &
      T_New
    
    integer :: &
      iV, jV
      
    allocate ( T_New, mold = T )
    
    nIterations = 0
    Error       = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Error  > MAX_ERROR )
               
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      
      nIterations = nIterations + 1
      
      Error = maxval ( abs ( T_new - T ) )
      T = T_New
    
    end do
    
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== Serial =========='
    print*, 'Error       :', Error
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
    print*, ''
        
  end subroutine Compute
 
 
  subroutine Compute_CPU_OpenMP ( T, nCells, Error, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Error
    integer, intent ( out ) :: &
      nIterations
    
    real ( real64 ) :: &
      TimeStart, &
      TimeTotal
    real ( real64 ), dimension ( :, : ), allocatable :: &
      T_New
    
    integer :: &
      iV, jV
      
    allocate ( T_New, mold = T )
    
    nIterations = 0
    Error       = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Error  > MAX_ERROR )
      !$OMP parallel do
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      !$OMP end parallel do
      
      nIterations = nIterations + 1
      
      Error = tiny ( 1.0_real64 )
      
      !$OMP parallel do reduction ( max : Error )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Error = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Error )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end parallel do
    
    end do
    
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== CPU_OpenMP =========='
    print*, 'Error       :', Error
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
    print*, ''
        
  end subroutine Compute_CPU_OpenMP
  
  
  subroutine Compute_GPU_OpenMP_1 ( T, nCells, Error, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Error
    integer, intent ( out ) :: &
      nIterations
    
    real ( real64 ) :: &
      TimeStart, &
      TimeTotal
    real ( real64 ), dimension ( :, : ), allocatable :: &
      T_New
    
    integer :: &
      iV, jV
      
    allocate ( T_New, mold = T )
    
    nIterations = 0
    Error       = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Error  > MAX_ERROR )
      !$OMP target teams distribute parallel do collapse ( 2 )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      !$OMP end target teams distribute parallel do
      
      nIterations = nIterations + 1
      
      Error = tiny ( 1.0_real64 )
      
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   reduction ( max : Error )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Error = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Error )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute parallel do
    
    end do
    
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== GPU_OpenMP_1 =========='
    print*, 'Error       :', Error
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
    print*, ''
        
  end subroutine Compute_GPU_OpenMP_1
  
  
  subroutine Compute_GPU_OpenMP_2 ( T, nCells, Error, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Error
    integer, intent ( out ) :: &
      nIterations
    
    real ( real64 ) :: &
      TimeStart, &
      TimeTotal
    real ( real64 ), dimension ( :, : ), allocatable :: &
      T_New
    
    integer :: &
      iV, jV
      
    allocate ( T_New, mold = T )
    
    nIterations = 0
    Error       = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Error  > MAX_ERROR )
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP map ( T, T_New )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      !$OMP end target teams distribute parallel do
      
      nIterations = nIterations + 1
      
      Error = tiny ( 1.0_real64 )
      
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   reduction ( max : Error ) map ( T, T_New )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Error = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Error )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute parallel do
    
    end do
    
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== GPU_OpenMP_1 =========='
    print*, 'Error       :', Error
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
    print*, ''
        
  end subroutine Compute_GPU_OpenMP_2
  
  
  subroutine Compute_GPU_OpenMP_3 ( T, nCells, Error, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Error
    integer, intent ( out ) :: &
      nIterations
    
    real ( real64 ) :: &
      TimeStart, &
      TimeTotal
    real ( real64 ), dimension ( :, : ), allocatable :: &
      T_New
    
    integer :: &
      iV, jV
      
    allocate ( T_New, mold = T )
    
    nIterations = 0
    Error       = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Error  > MAX_ERROR )
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP map ( to : T ) map ( from : T_New )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      !$OMP end target teams distribute parallel do
      
      nIterations = nIterations + 1
      
      Error = tiny ( 1.0_real64 )
      
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   reduction ( max : Error ) map ( tofrom : T ) map ( to : T_New )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Error = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Error )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute parallel do
    
    end do
    
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== GPU_OpenMP_3 =========='
    print*, 'Error       :', Error
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
    print*, ''
        
  end subroutine Compute_GPU_OpenMP_3
  
  
  subroutine Compute_GPU_OpenMP_4 ( T, nCells, Error, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Error
    integer, intent ( out ) :: &
      nIterations
    
    real ( real64 ) :: &
      TimeStart, &
      TimeTotal
    real ( real64 ), dimension ( :, : ), allocatable :: &
      T_New
    
    integer :: &
      iV, jV
      
    allocate ( T_New, mold = T )
    
    nIterations = 0
    Error       = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    !$OMP target enter data map ( to : T )
    !$OMP target enter data map ( alloc : T_New )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Error  > MAX_ERROR )
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   schedule ( static, 1 )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      !$OMP end target teams distribute parallel do
      
      nIterations = nIterations + 1
      
      Error = tiny ( 1.0_real64 )
      
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   schedule ( static, 1 ) &
      !$OMP   reduction ( max : Error ) 
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Error = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Error )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute parallel do
    
    end do
    
    !$OMP target exit data map ( delete : T_New )
    !$OMP target exit data map ( from : T )
    
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== GPU_OpenMP_4 =========='
    print*, 'Error       :', Error
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
    print*, ''
        
  end subroutine Compute_GPU_OpenMP_4
  
  
  subroutine Validate ( T1, T2 ) 
  
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( in ) :: &
      T1, T2
    logical :: &
      V

    !-- Only compare the proper ( inner ) cells. 
    !   The following associate construct creates 'aliases' 
    !   that points to the proper cells in each dimension.
    associate ( &
      T1_P => T1 ( 1 : size ( T1, dim = 1 ) - 2, &
                   1 : size ( T1, dim = 2 ) - 2 ), &
      T2_P => T1 ( 1 : size ( T2, dim = 1 ) - 2, &
                   1 : size ( T2, dim = 2 ) - 2 ) )
                    
    associate ( Error => abs ( ( T1_P - T2_P ) / ( T1_P ) ) )
    
    if ( all ( Error  <= 20 * MAX_ERROR ) ) then
      print*, 'Validate    : ', 'PASSED' 
    else
      print*, 'Validate    : ', 'FAILED' 
      print*, 'Error       : ', maxval ( Error )
    
    end if
    
    end associate  !-- Error
    
    end associate  !-- T1_P, T2_
  
  end subroutine Validate 
  

end module Laplacian_Form


program Laplace

  use iso_fortran_env
  use Laplacian_Form
  
  implicit none
  
  integer :: &
    nIterations
  integer, dimension ( 2 ) :: &
    nCells
  real ( real64 ) :: &
    Error
  real ( real64 ), dimension ( :, : ), allocatable :: &
    T, &          !-- 
    T_Init, &     !-- A copy of initial condition
    T_Results     !-- A copy of results from serial calculation
    
  !nCells = [ 4096, 4096 ]
  nCells = [ 1024, 1024 ]
  print*, nCells
  
  call Initialize ( nCells, T, T_Init )
  
  call Compute ( T, nCells, Error, nIterations )
  T_Results = T
  
  T = T_Init
  call Compute_CPU_OpenMP ( T, nCells, Error, nIterations )
  call Validate ( T, T_Results )
  
  T = T_Init
  call Compute_GPU_OpenMP_1 ( T, nCells, Error, nIterations )
  call Validate ( T, T_Results )
  
  T = T_Init
  call Compute_GPU_OpenMP_2 ( T, nCells, Error, nIterations )
  call Validate ( T, T_Results )
  
  T = T_Init
  call Compute_GPU_OpenMP_3 ( T, nCells, Error, nIterations )
  call Validate ( T, T_Results )
  
  T = T_Init
  call Compute_GPU_OpenMP_4 ( T, nCells, Error, nIterations )
  call Validate ( T, T_Results )
  
end program Laplace
