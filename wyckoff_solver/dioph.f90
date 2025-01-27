!MODULE kinds
!  IMPLICIT NONE
!  INTEGER, PARAMETER :: i2b = selected_int_kind(4)
!  INTEGER, PARAMETER :: i4b = selected_int_kind(9)
!  INTEGER, PARAMETER :: i8b = selected_int_kind(18)
!  INTEGER, PARAMETER :: int16 = selected_int_kind(4)
!  INTEGER, PARAMETER :: int32 = selected_int_kind(9)
!  INTEGER, PARAMETER :: int64 = selected_int_kind(18)
!
!  INTEGER, PARAMETER :: DP = selected_real_kind(14, 200)
!  INTEGER, PARAMETER :: float32 = selected_real_kind(6, 37)
!  INTEGER, PARAMETER :: float64 = selected_real_kind(14, 200)
!
!  INTEGER, PARAMETER :: CHAR_LEN = 8
!END MODULE kinds

MODULE NonNegLinDiophSolver_module
  USE kinds, ONLY : int64
  IMPLICIT NONE
  !INTEGER, PARAMETER :: int64 = selected_int_kind(18)
  PRIVATE
  PUBLIC :: count_solutions, generate_solutions

CONTAINS
  FUNCTION count_solutions(a, b) RESULT(count)
    IMPLICIT NONE
    integer(int64), intent(in) :: a(:)
    integer(int64), intent(in) :: b
    integer(int64) :: count
    integer(int64) :: i, j
    integer(int64), allocatable :: dp(:)
    logical :: all_ones

    all_ones = .true.
    DO i = 1, size(a)
      IF (a(i) /= 1) THEN
        all_ones = .false.
        EXIT
      END IF
    END DO

    IF (all_ones) THEN
      count = binomial_coefficient(size(a)-1_int64+b, size(a)-1_int64)
    ELSE
      allocate(dp(0:b))
      dp = 0
      dp(0) = 1

      DO i = 1, size(a)
        DO j = a(i), b
          dp(j) = dp(j) + dp(j - a(i))
        END DO
      END DO

      count = dp(b)
      deallocate(dp)
    END IF
  END FUNCTION count_solutions

  SUBROUTINE generate_solutions(a, b, solutions, num_solutions)
    IMPLICIT NONE
    integer(int64), intent(in) :: a(:)
    integer(int64), intent(in) :: b
    integer(int64), intent(in) :: num_solutions
    integer(int64), intent(out) :: solutions(num_solutions,size(a))

    integer(int64) :: i_solutions, solution_size
    integer(int64), allocatable :: current(:)
    solution_size = size(a)

    allocate(current(solution_size))
    current = 0_int64  ! initialize current solutions to all zeros
    i_solutions = 0_int64
    CALL backtrack(a, b, 1_int64, current, solutions, i_solutions)

    deallocate(current)
  END SUBROUTINE generate_solutions

  RECURSIVE SUBROUTINE backtrack(a, target, index, current, solutions, i_solutions)
    IMPLICIT NONE
    integer(int64), intent(in) :: a(:)
    integer(int64), intent(in) :: target, index
    integer(int64), intent(inout) :: current(:)
    integer(int64), intent(inout) :: solutions(:,:)
    integer(int64), intent(inout) :: i_solutions

    integer(int64) :: i, max_i

    IF (target == 0) THEN
      i_solutions = i_solutions + 1
      solutions(i_solutions,:) = current
      RETURN
    END IF

    IF (target < 0 .or. index > size(a)) THEN
      RETURN
    END IF

    max_i = target / a(index)

    DO i = 0, max_i
      current(index) = i
      CALL backtrack(a, target - i * a(index), index + 1, current, solutions, i_solutions)
      current(index) = 0
    END DO
  END SUBROUTINE backtrack

  FUNCTION binomial_coefficient(n, k) RESULT(res)
    integer(int64), intent(in) :: n, k

    integer(int64) :: res
    integer(int64) :: i, numerator, denominator
    numerator = 1
    denominator = 1
    DO i = 1, min(k, n-k)
      numerator = numerator * (n-i+1)
      denominator = denominator * i
    END DO
    res = numerator / denominator
  END FUNCTION binomial_coefficient

END MODULE NonNegLinDiophSolver_module
