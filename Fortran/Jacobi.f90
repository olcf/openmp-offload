module Jacobi_Form

  use omp_lib
  use iso_fortran_env
  
  implicit none
  public
  
  integer :: &
    MAX_ITERATIONS = 10000
  real ( real64 ), parameter :: &
    MAX_RESIDUAL = 1e-5_real64

contains 

  subroutine Initialize ( T, T_Init, nCells )
    
    real ( real64 ), dimension ( :, : ), allocatable, intent ( out ) :: &
      T, &
      T_Init
    integer, dimension ( : ), intent ( out ) :: &
      nCells
      
    character ( 31 ) :: &
      ExecName, &
      nCellsString, &
      MaxIterationsString
      
    !-- Parse command line options
    call get_command_argument ( 0, Value = ExecName )
    
    if ( command_argument_count ( ) == 2 ) then
      call get_command_argument ( 1, nCellsString )
      read ( nCellsString, fmt = '( i7 )' ) nCells ( 1 )
      nCells ( 2 ) = nCells ( 1 )

      call get_command_argument ( 2, MaxIterationsString )
      read ( MaxIterationsString, fmt = '( i7 )' ) MAX_ITERATIONS
      
      print*, 'Executing      : ', ExecName
      print*, 'nCells         : ', nCells ( 1 ), nCells ( 2 )
      print*, 'Max iterations : ', MAX_ITERATIONS
      print*, ''
      
    else
      print*, 'Usage: ' // trim ( ExecName ) // ' <nCells> <MaxIterations>'
      print*, '  where <nCells> and <MaxIterations> are integers.'
    end if
    
    !-- End parsing 
      
    allocate ( T      ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    allocate ( T_Init ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    
    call random_number ( T )
    T_Init = T
         
  end subroutine Initialize
  
  
  subroutine Compute ( T, nCells, Residual, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Residual
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
    Residual       = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Residual  > MAX_RESIDUAL )
               
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      
      nIterations = nIterations + 1
      
      Residual = maxval ( abs ( T_new - T ) )
      T = T_New
    
    end do
    
    nIterations = nIterations - 1
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== Serial =========='
    print*, 'Residual    :', Residual
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
        
  end subroutine Compute
 
 
  subroutine Compute_CPU_OpenMP ( T, nCells, Residual, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Residual
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
    Residual    = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Residual  > MAX_RESIDUAL )
      !$OMP parallel do collapse ( 2 )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      !$OMP end parallel do
      
      nIterations = nIterations + 1
      
      Residual = tiny ( 1.0_real64 )
      
      !$OMP parallel do collapse ( 2 ) &
      !$OMP   reduction ( max : Residual )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Residual = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Residual )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end parallel do
    
    end do
    
    nIterations = nIterations - 1
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== CPU_OpenMP =========='
    print*, 'nThreads    :', omp_get_max_threads()
    print*, 'Residual    :', Residual
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
        
  end subroutine Compute_CPU_OpenMP
  
  
  subroutine Compute_GPU_OpenMP_1 ( T, nCells, Residual, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Residual
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
    Residual    = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Residual  > MAX_RESIDUAL )
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
      
      Residual = tiny ( 1.0_real64 )
      
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   reduction ( max : Residual )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Residual = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Residual )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute parallel do
    
    end do
    
    nIterations = nIterations - 1
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== GPU_OpenMP_1 =========='
    print*, 'Residual    :', Residual
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
        
  end subroutine Compute_GPU_OpenMP_1
  
  
  subroutine Compute_GPU_OpenMP_2 ( T, nCells, Residual, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Residual
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
    Residual    = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Residual  > MAX_RESIDUAL )
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
      
      Residual = tiny ( 1.0_real64 )
      
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   reduction ( max : Residual ) map ( T, T_New )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Residual = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Residual )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute parallel do
    
    end do
    
    nIterations = nIterations - 1
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== GPU_OpenMP_1 =========='
    print*, 'Residual    :', Residual
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
        
  end subroutine Compute_GPU_OpenMP_2
  
  
  subroutine Compute_GPU_OpenMP_3 ( T, nCells, Residual, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Residual
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
    Residual    = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Residual  > MAX_RESIDUAL )
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
      
      Residual = tiny ( 1.0_real64 )
      
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   reduction ( max : Residual ) &
      !$OMP   map ( tofrom : T ) map ( to : T_New )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Residual = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Residual )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute parallel do
    
    end do
    
    nIterations = nIterations - 1
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== GPU_OpenMP_3 =========='
    print*, 'Residual    :', Residual
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
        
  end subroutine Compute_GPU_OpenMP_3
  
  
  subroutine Compute_GPU_OpenMP_4 ( T, nCells, Residual, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Residual
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
    Residual    = huge ( 1.0_real64 )
    
    TimeStart = omp_get_wtime ( )
    
    !$OMP target enter data map ( to : T )
    !$OMP target enter data map ( alloc : T_New )
    
    do while ( nIterations <= MAX_ITERATIONS &
               .and. Residual  > MAX_RESIDUAL )
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
      
      Residual = tiny ( 1.0_real64 )
      
      !$OMP target teams distribute parallel do collapse ( 2 ) &
      !$OMP   reduction ( max : Residual ) 
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Residual = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Residual )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute parallel do
    
    end do
    
    !$OMP target exit data map ( delete : T_New )
    !$OMP target exit data map ( from : T )
    
    nIterations = nIterations - 1
    TimeTotal = omp_get_wtime ( ) - TimeStart
    
    print*, '======== GPU_OpenMP_4 =========='
    print*, 'Residual    :', Residual
    print*, 'nIterations :', nIterations
    print*, 'Time (s)    :', TimeTotal
        
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
    
    if ( all ( Error  <= 20 * MAX_RESIDUAL ) ) then
      print*, 'Validation  : ', 'PASSED' 
    else
      print*, 'Validation  : ', 'FAILED' 
      print*, '     Error  : ', maxval ( Error )
    end if
    
    print*, ''
    
    end associate  !-- Error
    
    end associate  !-- T1_P, T2_P
  
  end subroutine Validate 
  

end module Jacobi_Form


program Jacobi

  use iso_fortran_env
  use Jacobi_Form
  
  implicit none
  
  integer :: &
    nIterations
  integer, dimension ( 2 ) :: &
    nCells
  real ( real64 ) :: &
    Residual
  real ( real64 ), dimension ( :, : ), allocatable :: &
    T, &          !-- 
    T_Init, &     !-- A copy of initial condition
    T_Results     !-- A copy of results from serial calculation
    
  nCells = -1
  
  call Initialize ( T, T_Init, nCells )
  if ( nCells ( 1 ) == -1 ) return
  
  call Compute ( T, nCells, Residual, nIterations )
  T_Results = T
  print*, ''
  
  T = T_Init
  call Compute_CPU_OpenMP ( T, nCells, Residual, nIterations )
  call Validate ( T, T_Results )
  
  T = T_Init
  call Compute_GPU_OpenMP_1 ( T, nCells, Residual, nIterations )
  call Validate ( T, T_Results )
  
  T = T_Init
  call Compute_GPU_OpenMP_2 ( T, nCells, Residual, nIterations )
  call Validate ( T, T_Results )
  
  T = T_Init
  call Compute_GPU_OpenMP_3 ( T, nCells, Residual, nIterations )
  call Validate ( T, T_Results )
  
  T = T_Init
  call Compute_GPU_OpenMP_4 ( T, nCells, Residual, nIterations )
  call Validate ( T, T_Results )
  
end program Jacobi
